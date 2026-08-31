from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
from decimal import Decimal
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, Field, JsonValue

from deepresearch.domain import ResearchPlan, ResearchRequest, RunBudget, SubQuestion
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


class PlanGenerationError(RuntimeError):
    code: Literal["PLAN_INVALID"] = "PLAN_INVALID"
    retryable: Literal[False] = False

    def __init__(self) -> None:
        super().__init__("plan generation failed validation")


class _QueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    queries: tuple[str, ...] = Field(min_length=1)


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


def _repair_candidate(raw: str) -> JsonValue | None:
    try:
        parsed: object = json.loads(
            raw,
            parse_constant=lambda _value: (_ for _ in ()).throw(ValueError()),
        )
    except (ValueError, json.JSONDecodeError):
        return None
    if isinstance(parsed, (dict, list, str, int, float, bool)) or parsed is None:
        return cast("JsonValue", parsed)
    return None


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
        self.validator = PlanValidator()
        self._query_cache: dict[tuple[str, str, str], tuple[str, ...]] = {}
        self._query_locks: dict[tuple[str, str, str], asyncio.Lock] = {}

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
        return _request(
            model=self.model,
            system_prompt=_PLAN_SYSTEM_PROMPT,
            user_prompt=_canonical_json(payload),
            prompt_version=self.prompt_version,
            max_output_tokens=max(1, min(8_000, self.budget.max_total_tokens)),
        )

    def _repair_request(self, raw: str, report: PlanValidationReport) -> ModelRequest:
        rejected = _repair_candidate(raw)
        bounded = None if rejected is None else _bound_json(rejected, self.content_boundary)
        payload = {
            "candidate": bounded,
            "error_codes": list(report.error_codes),
            "instruction": "Return one corrected ResearchPlan JSON object.",
        }
        return _request(
            model=self.model,
            system_prompt=_REPAIR_SYSTEM_PROMPT,
            user_prompt=_canonical_json(payload),
            prompt_version=f"{self.prompt_version}-repair",
            max_output_tokens=max(1, min(8_000, self.budget.max_total_tokens)),
        )

    async def _generate_candidate(
        self,
        request: ModelRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> tuple[str, str]:
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        result = await self.model.complete(
            request,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        raw = result.output
        data = raw.encode("utf-8")
        _check_call_boundary(
            provider=self.model.provider_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        ref = self.artifact_store.put_bytes(data, media_type=_CANDIDATE_MEDIA_TYPE)
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
        report = self.validator.validate_candidate(
            raw,
            request=request,
            budget=self.budget,
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
        repair_request = self._repair_request(raw, report)
        repaired_raw, repaired_artifact_id = await self._generate_candidate(
            repair_request,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        repaired = self.validator.validate_candidate(
            repaired_raw,
            request=request,
            budget=self.budget,
            candidate_artifact_id=repaired_artifact_id,
        )
        if not repaired.valid or repaired.candidate is None:
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
            "plan_id": plan_id,
            "search_depth": self.search_depth,
            "subquestion": _bound_json(raw_subquestion, self.content_boundary),
        }
        return _request(
            model=self.model,
            system_prompt=_QUERY_SYSTEM_PROMPT,
            user_prompt=_canonical_json(payload),
            prompt_version=f"{self.prompt_version}-queries",
            output_schema=_QueryOutput,
            max_output_tokens=max(128, min(2_000, self.budget.max_total_tokens)),
        )

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
        lock = self._query_locks.setdefault(key, asyncio.Lock())
        async with lock:
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
            result = await self.model.structured(
                request,
                _QueryOutput,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            _check_call_boundary(
                provider=self.model.provider_id,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
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
            return value


__all__ = ["FixedPlanner", "PlanGenerationError"]
