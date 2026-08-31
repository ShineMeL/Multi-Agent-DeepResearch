from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from dataclasses import dataclass
from decimal import Decimal
from threading import Lock
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from deepresearch.domain import (
    ResearchPlan,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
    SubQuestion,
)
from deepresearch.providers import (
    Deadline,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ProviderError,
)
from deepresearch.reporting import ContentBoundary, identity_content_boundary
from deepresearch.retrieval import normalize_text
from deepresearch.runtime import CancellationToken
from deepresearch.storage import LocalArtifactStore

from .validation import PlanValidationReport, PlanValidator

_CANDIDATE_MEDIA_TYPE = "application/vnd.deepresearch.plan-candidate+json"
_PLAN_SYSTEM_PROMPT = (
    "Return exactly one ResearchPlan JSON object. Use only explicit public fields; "
    "do not include hidden reasoning."
)
_REPAIR_SYSTEM_PROMPT = (
    "Repair one rejected ResearchPlan candidate using only the supplied JSON and stable codes. "
    "Return exactly one corrected ResearchPlan JSON object."
)
_QUERY_SYSTEM_PROMPT = "Return deterministic search queries for the supplied subquestion."
_MAX_REPAIR_CANDIDATE_CHARS = 1_000_000
_MAX_REPAIR_JSON_DEPTH = 64
_MAX_REPAIR_JSON_NODES = 20_000
_LOCK_POLL_SECONDS = 0.01
_MODEL_REQUEST_FIXED_BYTES = 512


class PlanGenerationError(RuntimeError):
    code: Literal["PLAN_INVALID"] = "PLAN_INVALID"
    retryable: Literal[False] = False

    def __init__(self) -> None:
        super().__init__("plan generation failed validation")


class _QueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    queries: tuple[str, ...] = Field(min_length=1)


@dataclass
class _QueryGate:
    lock: asyncio.Lock
    users: int = 0


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _canonical_json(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def _deadline_error(*, provider: str, deadline: Deadline) -> ProviderError | None:
    if not math.isfinite(deadline):
        return ProviderError(
            code="INVALID_REQUEST",
            provider=provider,
            operation="model",
            public_message="deadline must be a finite absolute monotonic deadline",
            retryable=False,
        )
    if time.monotonic() >= deadline:
        return ProviderError(
            code="TIMEOUT",
            provider=provider,
            operation="model",
            public_message="model deadline expired",
            retryable=False,
        )
    return None


def _check_call_boundary(
    *,
    provider: str,
    deadline: Deadline,
    cancellation_token: CancellationToken,
) -> None:
    cancellation_token.raise_if_cancelled()
    error = _deadline_error(provider=provider, deadline=deadline)
    if error is not None:
        raise error


def _bound_json(
    value: JsonValue,
    boundary: ContentBoundary,
    *,
    bound_mapping_keys: bool = False,
) -> JsonValue:
    if isinstance(value, str):
        bounded_text = boundary(value)
        if not isinstance(bounded_text, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            raise TypeError("content boundary must return text")
        return bounded_text
    if isinstance(value, list):
        return [
            _bound_json(item, boundary, bound_mapping_keys=bound_mapping_keys)
            for item in value
        ]
    if isinstance(value, dict):
        bounded_mapping: dict[str, JsonValue] = {}
        for key, item in value.items():
            bounded_key = boundary(key) if bound_mapping_keys else key
            if not isinstance(bounded_key, str):  # pyright: ignore[reportUnnecessaryIsInstance]
                raise TypeError("content boundary must return text")
            if bounded_key in bounded_mapping:
                raise ValueError("content boundary produced duplicate mapping keys")
            bounded_mapping[bounded_key] = _bound_json(
                item,
                boundary,
                bound_mapping_keys=bound_mapping_keys,
            )
        return bounded_mapping
    return value


def _model_id(model: ModelProvider) -> str:
    value = getattr(model, "model_id", model.provider_id)
    return value if isinstance(value, str) and value else model.provider_id


def _request(
    *,
    model: ModelProvider,
    system_prompt: str,
    user_prompt: str,
    prompt_version: str,
    output_schema: type[BaseModel] | None = None,
    max_output_tokens: int,
) -> ModelRequest:
    schema_json = "" if output_schema is None else _canonical_json(output_schema.model_json_schema())
    return ModelRequest(
        model_id=_model_id(model),
        messages=(
            ModelMessage(role="system", content=system_prompt),
            ModelMessage(role="user", content=user_prompt),
        ),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=max_output_tokens,
        prompt_version=prompt_version,
        system_prompt_hash=_sha256_text(system_prompt),
        tool_schema_hash=_sha256_text("[]"),
        output_schema_hash=_sha256_text(schema_json),
    )


def _remaining_token_budget(budget: RunBudget) -> int:
    used = sum(usage.total_tokens for usage in budget.used_by_node.values())
    return max(0, budget.max_total_tokens - used)


def _request_token_upper_bound(request: ModelRequest) -> int:
    prompt_bytes = sum(
        len(message.content.encode("utf-8")) for message in request.messages
    )
    return _MODEL_REQUEST_FIXED_BYTES + prompt_bytes + request.max_output_tokens


def _ensure_request_within_budget(
    request: ModelRequest,
    *,
    budget: RunBudget,
    provider: str,
) -> None:
    try:
        upper_bound = _request_token_upper_bound(request)
    except UnicodeEncodeError:
        raise ProviderError(
            code="INVALID_REQUEST",
            provider=provider,
            operation="model",
            public_message="model request contains invalid text",
            retryable=False,
        ) from None
    if upper_bound > _remaining_token_budget(budget):
        raise ProviderError(
            code="INVALID_REQUEST",
            provider=provider,
            operation="model",
            public_message="model request exceeds remaining token budget",
            retryable=False,
        )


def _repair_candidate(raw: str) -> JsonValue | None:
    if len(raw) > _MAX_REPAIR_CANDIDATE_CHARS:
        return None
    try:
        def reject_duplicate_names(
            pairs: list[tuple[str, object]],
        ) -> dict[str, object]:
            result: dict[str, object] = {}
            for key, item in pairs:
                if key in result:
                    raise ValueError("duplicate JSON object name")
                result[key] = item
            return result

        parsed: object = json.loads(
            raw,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
            object_pairs_hook=reject_duplicate_names,
        )
    except (RecursionError, ValueError):
        return None
    stack: list[tuple[object, int]] = [(parsed, 0)]
    node_count = 0
    while stack:
        item, depth = stack.pop()
        node_count += 1
        if node_count > _MAX_REPAIR_JSON_NODES or depth > _MAX_REPAIR_JSON_DEPTH:
            return None
        if isinstance(item, dict):
            mapping = cast("dict[object, object]", item)
            if any(not isinstance(key, str) for key in mapping):
                return None
            stack.extend((child, depth + 1) for child in mapping.values())
        elif isinstance(item, list):
            values = cast("list[object]", item)
            stack.extend((child, depth + 1) for child in values)
        elif (
            isinstance(item, float) and not math.isfinite(item)
        ) or not (
            item is None
            or isinstance(item, (str, int, float, bool))
        ):
            return None
    return cast("JsonValue", parsed)


class FixedPlanner:
    variant = "P1"

    def __init__(
        self,
        *,
        model: ModelProvider,
        artifact_store: LocalArtifactStore,
        budget: RunBudget,
        search_depth: int = 2,
        prompt_version: str = "fixed-planner-v1",
        content_boundary: ContentBoundary = identity_content_boundary,
    ) -> None:
        if isinstance(search_depth, bool) or search_depth <= 0:
            raise ValueError("search_depth must be a positive integer")
        if not prompt_version.strip():
            raise ValueError("prompt_version must not be empty")
        self.model = model
        self.artifact_store = artifact_store
        self.budget = budget
        self.search_depth = search_depth
        self.prompt_version = prompt_version
        self.content_boundary = content_boundary
        self.validator = PlanValidator(search_depth=search_depth)
        self._query_cache: dict[tuple[str, str, str], tuple[str, ...]] = {}
        self._query_locks: dict[tuple[str, str, str], _QueryGate] = {}
        self._initial_model_tokens = sum(
            usage.total_tokens for usage in budget.used_by_node.values()
        )
        self._model_tokens_used = 0
        self._model_tokens_reserved = 0
        self._token_lock = Lock()

    def _token_budget_error(self) -> ProviderError:
        return ProviderError(
            code="INVALID_REQUEST",
            provider=self.model.provider_id,
            operation="model",
            public_message="model request exceeds remaining token budget",
            retryable=False,
        )

    def _remaining_model_tokens(self) -> int:
        with self._token_lock:
            return max(
                0,
                self.budget.max_total_tokens
                - self._initial_model_tokens
                - self._model_tokens_used
                - self._model_tokens_reserved,
            )

    def _reserve_request(self, request: ModelRequest) -> int | None:
        try:
            reservation = _request_token_upper_bound(request)
        except UnicodeEncodeError:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.model.provider_id,
                operation="model",
                public_message="model request contains invalid text",
                retryable=False,
            ) from None
        with self._token_lock:
            remaining = (
                self.budget.max_total_tokens
                - self._initial_model_tokens
                - self._model_tokens_used
                - self._model_tokens_reserved
            )
            if reservation > remaining:
                return None
            self._model_tokens_reserved += reservation
        return reservation

    def _settle_request(self, reservation: int, actual_tokens: int | None) -> bool:
        with self._token_lock:
            self._model_tokens_reserved -= reservation
            if actual_tokens is not None:
                self._model_tokens_used += actual_tokens
            return (
                self._initial_model_tokens
                + self._model_tokens_used
                + self._model_tokens_reserved
                <= self.budget.max_total_tokens
            )

    def _effective_budget(self) -> RunBudget:
        with self._token_lock:
            local_tokens = self._model_tokens_used
        used_by_node = dict(self.budget.used_by_node)
        planner_usage = used_by_node.get("Planner", ResourceUsage.zero())
        used_by_node["Planner"] = planner_usage.model_copy(
            update={
                "input_tokens": planner_usage.input_tokens + local_tokens,
                "total_tokens": planner_usage.total_tokens + local_tokens,
            }
        )
        return self.budget.model_copy(update={"used_by_node": used_by_node})

    def _plan_request(self, request: ResearchRequest) -> ModelRequest:
        raw = request.model_dump(mode="json")
        payload = {
            key: _bound_json(
                cast("JsonValue", value),
                self.content_boundary,
                bound_mapping_keys=key == "output_requirements",
            )
            for key, value in raw.items()
        }
        model_request = _request(
            model=self.model,
            system_prompt=_PLAN_SYSTEM_PROMPT,
            user_prompt=_canonical_json(payload),
            prompt_version=self.prompt_version,
            max_output_tokens=max(1, min(8_000, self.budget.max_total_tokens)),
        )
        _ensure_request_within_budget(
            model_request,
            budget=self.budget,
            provider=self.model.provider_id,
        )
        return model_request

    def _repair_requests(
        self,
        raw: str,
        report: PlanValidationReport,
    ) -> tuple[ModelRequest, ModelRequest | None]:
        remaining = self._remaining_model_tokens()
        rejected = (
            None
            if len(raw) > remaining or len(raw.encode("utf-8")) > remaining
            else _repair_candidate(raw)
        )
        bounded = (
            None
            if rejected is None
            else _bound_json(
                rejected,
                self.content_boundary,
                bound_mapping_keys=True,
            )
        )
        def build(candidate: JsonValue | None) -> ModelRequest:
            payload = {
                "candidate": candidate,
                "error_codes": list(report.error_codes),
                "instruction": "Return one corrected ResearchPlan JSON object.",
            }
            return _request(
                model=self.model,
                system_prompt=_REPAIR_SYSTEM_PROMPT,
                user_prompt=_canonical_json(payload),
                prompt_version=f"{self.prompt_version}-repair",
                max_output_tokens=max(
                    1,
                    min(8_000, self.budget.max_total_tokens),
                ),
            )

        if bounded is None:
            return build(None), None
        return build(bounded), build(None)

    async def _generate_candidate(
        self,
        request: ModelRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
        local_capacity_fallback: ModelRequest | None = None,
    ) -> tuple[str, str]:
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        reservation = self._reserve_request(request)
        if reservation is None and local_capacity_fallback is not None:
            request = local_capacity_fallback
            reservation = self._reserve_request(request)
        if reservation is None:
            raise self._token_budget_error()
        try:
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
        except BaseException:
            self._settle_request(reservation, None)
            raise
        try:
            result = await self.model.complete(
                request,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
        except ProviderError as error:
            self._settle_request(
                reservation,
                None if error.usage is None else error.usage.total_tokens,
            )
            raise
        except BaseException:
            self._settle_request(reservation, None)
            raise
        publishable = self._settle_request(reservation, result.usage.total_tokens)
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        raw = result.output
        try:
            data = raw.encode("utf-8")
        except UnicodeEncodeError:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.model.provider_id,
                operation="complete",
                public_message="model response contains invalid text",
                retryable=False,
            ) from None
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        ref = self.artifact_store.put_bytes(data, media_type=_CANDIDATE_MEDIA_TYPE)
        if not publishable:
            raise self._token_budget_error()
        return raw, ref.artifact_id

    async def create_plan(
        self,
        request: ResearchRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> ResearchPlan:
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        initial_request = self._plan_request(request)
        raw, artifact_id = await self._generate_candidate(
            initial_request,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        report = self.validator.validate_generated_candidate(
            raw,
            request=request,
            budget=self._effective_budget(),
            candidate_artifact_id=artifact_id,
        )
        if report.valid and report.candidate is not None:
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            return report.candidate

        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        repair_request, null_repair_fallback = self._repair_requests(raw, report)
        repaired_raw, repaired_artifact_id = await self._generate_candidate(
            repair_request,
            deadline=deadline,
            cancellation_token=cancellation_token,
            local_capacity_fallback=null_repair_fallback,
        )
        repaired = self.validator.validate_generated_candidate(
            repaired_raw,
            request=request,
            budget=self._effective_budget(),
            candidate_artifact_id=repaired_artifact_id,
        )
        if not repaired.valid or repaired.candidate is None:
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            raise PlanGenerationError
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return repaired.candidate

    def _query_request(self, subquestion: SubQuestion, plan_id: str) -> ModelRequest:
        raw_subquestion = cast("JsonValue", subquestion.model_dump(mode="json"))
        payload: dict[str, JsonValue] = {
            "plan_id": _bound_json(plan_id, self.content_boundary),
            "search_depth": self.search_depth,
            "subquestion": _bound_json(raw_subquestion, self.content_boundary),
        }
        model_request = _request(
            model=self.model,
            system_prompt=_QUERY_SYSTEM_PROMPT,
            user_prompt=_canonical_json(payload),
            prompt_version=f"{self.prompt_version}-queries",
            output_schema=_QueryOutput,
            max_output_tokens=max(128, min(2_000, self.budget.max_total_tokens)),
        )
        _ensure_request_within_budget(
            model_request,
            budget=self.budget,
            provider=self.model.provider_id,
        )
        return model_request

    async def _acquire_query_lock(
        self,
        lock: asyncio.Lock,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> None:
        while True:
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            remaining = deadline - time.monotonic()
            try:
                await asyncio.wait_for(
                    lock.acquire(),
                    timeout=min(_LOCK_POLL_SECONDS, remaining),
                )
            except TimeoutError:
                continue
            try:
                _check_call_boundary(
                    provider=self.model.provider_id,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                )
            except BaseException:
                lock.release()
                raise
            return

    async def queries_for(
        self,
        subquestion: SubQuestion,
        *,
        plan_id: str,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> tuple[str, ...]:
        key = (plan_id, subquestion.id, self.prompt_version)
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        cached = self._query_cache.get(key)
        if cached is not None:
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            return cached
        gate = self._query_locks.setdefault(key, _QueryGate(lock=asyncio.Lock()))
        gate.users += 1
        acquired = False
        try:
            await self._acquire_query_lock(
                gate.lock,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            acquired = True
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            cached = self._query_cache.get(key)
            if cached is not None:
                return cached
            request = self._query_request(subquestion, plan_id)
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            reservation = self._reserve_request(request)
            if reservation is None:
                raise self._token_budget_error()
            try:
                _check_call_boundary(
                    provider=self.model.provider_id,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                )
            except BaseException:
                self._settle_request(reservation, None)
                raise
            try:
                result = await self.model.structured(
                    request,
                    _QueryOutput,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                )
            except ProviderError as error:
                self._settle_request(
                    reservation,
                    None if error.usage is None else error.usage.total_tokens,
                )
                raise
            except BaseException:
                self._settle_request(reservation, None)
                raise
            publishable = self._settle_request(
                reservation,
                result.usage.total_tokens,
            )
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            if not publishable:
                raise self._token_budget_error()
            output = result.output
            queries: list[str] = []
            seen: set[str] = set()
            for raw_query in output.queries:
                normalized = normalize_text(raw_query)
                if normalized and normalized not in seen:
                    seen.add(normalized)
                    queries.append(normalized)
                if len(queries) == self.search_depth:
                    break
            if not queries:
                raise ProviderError(
                    code="INVALID_RESPONSE",
                    provider=self.model.provider_id,
                    operation="structured",
                    public_message="query generation response is invalid",
                    retryable=False,
                )
            value = tuple(queries)
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            self._query_cache[key] = value
            try:
                _check_call_boundary(
                    provider=self.model.provider_id,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                )
            except BaseException:
                self._query_cache.pop(key, None)
                raise
            return value
        finally:
            if acquired:
                gate.lock.release()
            gate.users -= 1
            if gate.users == 0 and self._query_locks.get(key) is gate:
                self._query_locks.pop(key, None)


__all__ = ["FixedPlanner", "PlanGenerationError"]
