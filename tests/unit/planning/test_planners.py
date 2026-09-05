from __future__ import annotations

from collections.abc import AsyncIterator, Sequence
from dataclasses import replace
from typing import Any

import pytest

from deepresearch.domain import (
    EvidenceRequirements,
    FreshnessRequirement,
    InformationNeed,
    ResearchPlan,
    ResearchScope,
    ResourceUsage,
    RunBudget,
    SubQuestion,
)
from deepresearch.planning.contracts import PlannerState, QueryCandidate
from deepresearch.planning.ledger import CoverageLedger
from deepresearch.planning.planners import (
    P0ReActPlanner,
    P1FixedPlanner,
    P2AdaptivePlanner,
    PlannerInvariantError,
)
from deepresearch.planning.stop import BlockedNeed, StopCode
from deepresearch.providers import (
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    StructuredModelResult,
)
from deepresearch.runtime import BudgetAccountant, CancellationToken


def make_plan(*, subquestion_ids: tuple[str, ...] = ("sq-1",)) -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-planners",
        scope=ResearchScope(
            included_topics=("planner optimization",),
            excluded_topics=(),
            answer_shape="brief",
        ),
        subquestions=tuple(
            SubQuestion(
                id=subquestion_id,
                question=f"What is documented for {subquestion_id}?",
                rationale_code="coverage",
                importance=0.8 if subquestion_id == "sq-1" else 0.7,
                dependencies=(),
                information_needs=(
                    InformationNeed(
                        need_id=f"need-{subquestion_id}",
                        text="Documented methods",
                        importance=0.8,
                    ),
                ),
                evidence_requirements=EvidenceRequirements(
                    min_independent_sources=1,
                    allowed_source_types=frozenset({"paper"}),
                    must_include_primary=False,
                    freshness=FreshnessRequirement(kind="none"),
                ),
                status="pending",
            )
            for subquestion_id in subquestion_ids
        ),
        created_by_model="test-model",
        prompt_version="planners-v1",
    )


def budget_snapshot() -> Any:
    return BudgetAccountant(
        RunBudget.preset("low").model_copy(update={"max_cost_usd": None}),
        run_scope="planners-test",
    ).snapshot()


def state(
    plan: ResearchPlan,
    *,
    coverages: dict[str, float] | None = None,
    attempts: dict[str, int] | None = None,
    blocked_needs: tuple[BlockedNeed, ...] = (),
    round_index: int = 0,
    gains: tuple[float, ...] = (),
) -> PlannerState:
    coverages = coverages or {}
    attempts = attempts or {}
    ledger = CoverageLedger.empty_for(plan)
    for subquestion in plan.subquestions:
        entry = ledger.get(subquestion.id)
        ledger = ledger.replace(
            entry.model_copy(
                update={
                    "coverage_score": coverages.get(subquestion.id, 0.0),
                    "uncertainty_score": 1.0 - coverages.get(subquestion.id, 0.0),
                    "attempt_count": attempts.get(subquestion.id, 0),
                }
            )
        )
    return PlannerState(
        plan=plan,
        ledger=ledger,
        budget_snapshot=budget_snapshot(),
        blocked_needs=blocked_needs,
        round_index=round_index,
        recent_marginal_gains=gains,
        query_history=(),
    )


def token() -> CancellationToken:
    return CancellationToken()


class FakeFixedPlanner:
    def __init__(self, *, queries: Sequence[str], search_depth: int = 2) -> None:
        self.queries = tuple(queries)
        self.search_depth = search_depth
        self.create_plan_calls = 0
        self.queries_for_calls = 0
        self.plan_ids: list[str] = []

    async def create_plan(self, *args: object, **kwargs: object) -> ResearchPlan:
        del args, kwargs
        self.create_plan_calls += 1
        return make_plan()

    async def queries_for(
        self,
        subquestion: SubQuestion,
        *,
        plan_id: str,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[str, ...]:
        del subquestion, deadline
        cancellation_token.raise_if_cancelled()
        self.queries_for_calls += 1
        self.plan_ids.append(plan_id)
        return self.queries


class FakeModel:
    provider_id = "fake-model"
    model_id = "fake-model-v1"

    def __init__(self) -> None:
        self.structured_calls = 0

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        del request, deadline
        cancellation_token.raise_if_cancelled()
        return ModelResult(
            output="reference",
            usage=ResourceUsage.zero(),
            provider_id=self.provider_id,
            model_id=self.model_id,
            raw_response_artifact_id="sha256:" + "a" * 64,
        )

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[Any],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[Any]:
        del request, deadline
        cancellation_token.raise_if_cancelled()
        self.structured_calls += 1
        if "queries" in output_schema.model_fields:
            payload: dict[str, object] = {"queries": ("adaptive query",)}
        else:
            payload = {"action": "REFERENCE", "decision_code": "REACT_REFERENCE"}
        return StructuredModelResult(
            output=output_schema.model_validate(payload),
            usage=ResourceUsage.zero(),
            provider_id=self.provider_id,
            model_id=self.model_id,
            raw_response_artifact_id="sha256:" + "b" * 64,
            output_schema_hash="c" * 64,
        )

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        del request, deadline, cancellation_token
        raise NotImplementedError
        yield  # pragma: no cover


class FakeScheduler:
    async def dedupe(
        self,
        candidates: Sequence[QueryCandidate],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[QueryCandidate]:
        del deadline
        cancellation_token.raise_if_cancelled()
        return list(candidates)


@pytest.mark.asyncio
async def test_p1_does_not_replan_after_initial_plan() -> None:
    delegate = FakeFixedPlanner(queries=("q-1", "q-2"))
    planner = P1FixedPlanner(delegate=delegate)
    plan = make_plan()
    initial = state(plan)
    later = replace(initial, round_index=2)

    first = await planner.next_action(initial, deadline=100, cancellation_token=token())
    second = await planner.next_action(later, deadline=100, cancellation_token=token())

    assert second.candidates == first.candidates
    assert delegate.create_plan_calls == 0
    assert delegate.queries_for_calls == 2
    assert delegate.plan_ids == [plan.plan_id, plan.plan_id]


@pytest.mark.asyncio
async def test_p1_fixed_exhaustion_uses_typed_blocked_not_plateau() -> None:
    planner = P1FixedPlanner(delegate=FakeFixedPlanner(queries=()))
    exhausted = BlockedNeed(
        need_id="need-1",
        required_source_unavailable=True,
        alternative_strategies_exhausted=True,
        retries_used=2,
        max_retries=2,
    )

    result = await planner.next_action(
        state(make_plan(), blocked_needs=(exhausted,)),
        deadline=100,
        cancellation_token=token(),
    )

    assert result.stop is not None
    assert result.stop.code is StopCode.BLOCKED


@pytest.mark.asyncio
async def test_p1_targetless_state_without_stop_evidence_is_invariant_error() -> None:
    planner = P1FixedPlanner(delegate=FakeFixedPlanner(queries=()))
    plan = make_plan()

    with pytest.raises(PlannerInvariantError) as error:
        await planner.next_action(
            state(plan, attempts={"sq-1": 1}),
            deadline=100,
            cancellation_token=token(),
        )

    assert error.value.code == "P1_NO_TARGET_WITHOUT_TYPED_STOP_EVIDENCE"


@pytest.mark.asyncio
async def test_p2_selects_highest_priority_uncovered_subquestion() -> None:
    plan = make_plan(subquestion_ids=("sq-1", "sq-2"))
    result = await P2AdaptivePlanner(
        query_scheduler=FakeScheduler(),
        model_provider=FakeModel(),
    ).next_action(
        state(plan, coverages={"sq-1": 0.2, "sq-2": 0.8}),
        deadline=100,
        cancellation_token=token(),
    )

    assert result.subquestion_id == "sq-1"


@pytest.mark.asyncio
async def test_p0_never_requires_or_updates_coverage_ledger() -> None:
    plan = make_plan()
    initial = state(plan)
    before = initial.ledger.entries()
    model = FakeModel()

    result = await P0ReActPlanner(model_provider=model).next_action(
        initial,
        deadline=100,
        cancellation_token=token(),
    )

    assert result.decision_code == "REACT_REFERENCE"
    assert initial.ledger.entries() == before
    assert model.structured_calls == 1
