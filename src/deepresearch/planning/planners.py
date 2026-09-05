from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from decimal import Decimal
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from deepresearch.domain import SubQuestion
from deepresearch.providers import (
    Deadline,
    ModelMessage,
    ModelProvider,
    ModelRequest,
)
from deepresearch.runtime import CancellationToken

from .contracts import PlannerDecision, PlannerState, QueryCandidate
from .priority import compute_priority
from .stop import evaluate_stop


class PlannerInvariantError(RuntimeError):
    code: Literal["P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE"] = (
        "P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE"
    )

    def __init__(self, message: str = "planner target exhaustion lacks typed stop evidence") -> None:
        super().__init__(message)


@runtime_checkable
class Planner(Protocol):
    variant: Literal["P0", "P1", "P2"]

    async def next_action(
        self,
        state: PlannerState,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PlannerDecision: ...


class _QueryDeduper(Protocol):
    async def dedupe(
        self,
        candidates: Sequence[QueryCandidate],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[QueryCandidate]: ...


class _ReactAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    action: Literal["REFERENCE", "SEARCH", "STOP"] = "REFERENCE"
    query: str | None = None
    decision_code: str = "REACT_REFERENCE"


class _QueryOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    queries: tuple[str, ...] = Field(default_factory=tuple)


def _model_id(model_provider: ModelProvider) -> str:
    model_id = getattr(model_provider, "model_id", None)
    if isinstance(model_id, str) and model_id:
        return model_id
    return model_provider.provider_id


def _json_hash(value: object) -> str:
    encoded = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _request(
    model_provider: ModelProvider,
    *,
    prompt: str,
    prompt_version: str,
    output_schema: type[BaseModel],
) -> ModelRequest:
    schema_hash = _json_hash(output_schema.model_json_schema())
    system_prompt = "Return only the requested public structured planner action."
    return ModelRequest(
        model_id=_model_id(model_provider),
        messages=(
            ModelMessage(role="system", content=system_prompt),
            ModelMessage(role="user", content=prompt),
        ),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=256,
        prompt_version=prompt_version,
        system_prompt_hash=_json_hash(system_prompt),
        tool_schema_hash=_json_hash([]),
        output_schema_hash=schema_hash,
    )


def _first_target(state: PlannerState) -> SubQuestion | None:
    for subquestion in state.plan.subquestions:
        if state.ledger.get(subquestion.id).attempt_count == 0:
            return subquestion
    return None


def _candidate_tuple(target: SubQuestion, queries: Sequence[str]) -> tuple[QueryCandidate, ...]:
    if not target.information_needs:
        raise PlannerInvariantError("subquestion has no information need for query attribution")
    return tuple(
        QueryCandidate(
            subquestion_id=target.id,
            information_need_id=target.information_needs[index % len(target.information_needs)].need_id,
            query=query,
            priority_hint=target.importance,
            estimated_tokens=256,
            estimated_search_calls=1,
            estimated_seconds=5.0,
        )
        for index, query in enumerate(queries)
    )


class P0ReActPlanner:
    variant: Literal["P0"] = "P0"

    def __init__(self, *, model_provider: ModelProvider) -> None:
        self.model_provider = model_provider

    async def next_action(
        self,
        state: PlannerState,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PlannerDecision:
        if not state.plan.subquestions:
            raise PlannerInvariantError("P0 requires at least one subquestion")
        target = state.plan.subquestions[0]
        if not target.information_needs:
            raise PlannerInvariantError("P0 requires an information need")
        request = _request(
            self.model_provider,
            prompt=json.dumps(
                {"plan_id": state.plan.plan_id, "subquestion": target.model_dump(mode="json")},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            prompt_version="react-planner-v1",
            output_schema=_ReactAction,
        )
        result = await self.model_provider.structured(
            request,
            _ReactAction,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        action = result.output
        query = action.query or target.question
        decision_code = action.decision_code.strip() or "REACT_REFERENCE"
        candidates = _candidate_tuple(target, (query,))
        return PlannerDecision("SEARCH", target.id, candidates, None, decision_code)


class P1FixedPlanner:
    variant: Literal["P1"] = "P1"

    def __init__(self, *, delegate: object) -> None:
        self.delegate = delegate

    async def next_action(
        self,
        state: PlannerState,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PlannerDecision:
        stop = evaluate_stop(state, state.budget_snapshot, blocked_needs=state.blocked_needs)
        if stop is not None:
            return PlannerDecision("STOP", None, (), stop, stop.code.value)
        target = _first_target(state)
        if target is None:
            raise PlannerInvariantError
        queries_for = getattr(self.delegate, "queries_for", None)
        if queries_for is None:
            raise TypeError("delegate must provide queries_for")
        queries = await queries_for(
            target,
            plan_id=state.plan.plan_id,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        candidates = _candidate_tuple(target, queries)
        if not candidates:
            raise PlannerInvariantError("fixed planner produced no query candidates")
        return PlannerDecision("SEARCH", target.id, candidates, None, "P1_FIXED_QUERIES")


class P2AdaptivePlanner:
    variant: Literal["P2"] = "P2"

    def __init__(
        self,
        *,
        query_scheduler: _QueryDeduper,
        model_provider: ModelProvider,
    ) -> None:
        self.query_scheduler = query_scheduler
        self.model_provider = model_provider

    def _target_priority(self, state: PlannerState, subquestion: SubQuestion) -> float:
        entry = state.ledger.get(subquestion.id)
        required_sources = subquestion.evidence_requirements.min_independent_sources
        return compute_priority(
            importance=subquestion.importance,
            coverage_score=entry.coverage_score,
            new_source_need=float(entry.independent_source_count < required_sources),
            conflict_resolution_need=float(bool(entry.unresolved_conflict_ids)),
            token_fraction=0.5,
            search_fraction=0.5,
            time_fraction=0.5,
            recent_gain=(
                state.recent_marginal_gains[-1] if state.recent_marginal_gains else 0.5
            ),
            historical_success=0.5,
        )

    def _target(self, state: PlannerState) -> SubQuestion:
        if not state.plan.subquestions:
            raise PlannerInvariantError("adaptive planner requires at least one subquestion")
        return max(
            enumerate(state.plan.subquestions),
            key=lambda item: (self._target_priority(state, item[1]), -item[0]),
        )[1]

    async def generate_queries(
        self,
        target: SubQuestion,
        state: PlannerState,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> tuple[QueryCandidate, ...]:
        request = _request(
            self.model_provider,
            prompt=json.dumps(
                {"plan_id": state.plan.plan_id, "subquestion": target.model_dump(mode="json")},
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ),
            prompt_version="adaptive-planner-v1",
            output_schema=_QueryOutput,
        )
        result = await self.model_provider.structured(
            request,
            _QueryOutput,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        queries = tuple(query.strip() for query in result.output.queries if query.strip())
        if not queries:
            queries = (target.question,)
        candidates = _candidate_tuple(target, queries)
        dedupe = getattr(self.query_scheduler, "dedupe", None)
        if dedupe is not None:
            candidates = tuple(
                await dedupe(
                    candidates,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                )
            )
        return candidates

    async def next_action(
        self,
        state: PlannerState,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PlannerDecision:
        stop = evaluate_stop(state, state.budget_snapshot, blocked_needs=state.blocked_needs)
        if stop is not None:
            return PlannerDecision("STOP", None, (), stop, stop.code.value)
        target = self._target(state)
        candidates = await self.generate_queries(
            target,
            state,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        if not candidates:
            raise PlannerInvariantError("adaptive planner produced no query candidates")
        return PlannerDecision("SEARCH", target.id, candidates, None, "P2_INCREMENTAL_REPLAN")


__all__ = [
    "P0ReActPlanner",
    "P1FixedPlanner",
    "P2AdaptivePlanner",
    "Planner",
    "PlannerInvariantError",
]
