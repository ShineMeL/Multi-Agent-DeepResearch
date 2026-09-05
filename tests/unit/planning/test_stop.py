from __future__ import annotations

import pytest

from deepresearch.domain import (
    CoverageLedgerEntry,
    EvidenceRequirements,
    FreshnessRequirement,
    InformationNeed,
    ResearchPlan,
    ResearchScope,
    RunBudget,
    SubQuestion,
)
from deepresearch.planning.contracts import PlannerState
from deepresearch.planning.ledger import CoverageLedger
from deepresearch.planning.stop import (
    BlockedNeed,
    StopCode,
    evaluate_stop,
)
from deepresearch.runtime import BudgetAccountant, BudgetSnapshot


def make_plan() -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-stop",
        scope=ResearchScope(
            included_topics=("planner optimization",),
            excluded_topics=(),
            answer_shape="brief",
        ),
        subquestions=(
            SubQuestion(
                id="sq-1",
                question="What is documented?",
                rationale_code="coverage",
                importance=0.8,
                dependencies=(),
                information_needs=(
                    InformationNeed(need_id="need-1", text="Documented methods", importance=0.8),
                ),
                evidence_requirements=EvidenceRequirements(
                    min_independent_sources=2,
                    allowed_source_types=frozenset({"paper"}),
                    must_include_primary=False,
                    freshness=FreshnessRequirement(kind="none"),
                ),
                status="pending",
            ),
        ),
        created_by_model="test-model",
        prompt_version="stop-v1",
    )


def budget_ok() -> BudgetSnapshot:
    return BudgetAccountant(RunBudget.preset("low"), run_scope="stop-test").snapshot()


def budget_exhausted() -> BudgetSnapshot:
    return budget_ok().model_copy(update={"exhausted": frozenset({"search_calls"})})


def state_with_coverage(
    coverage: float,
    *,
    sources: int = 2,
    gains: tuple[float, ...] = (),
    conflicts: tuple[str, ...] = (),
) -> PlannerState:
    plan = make_plan()
    ledger = CoverageLedger.empty_for(plan).replace(
        CoverageLedgerEntry(
            subquestion_id="sq-1",
            coverage_score=coverage,
            independent_source_count=sources,
            unresolved_conflict_ids=conflicts,
            uncertainty_score=1.0 - coverage,
            last_marginal_gain=gains[-1] if gains else 0.0,
            evidence_ids=(),
            attempt_count=len(gains),
            last_decision_code="RANKED",
        )
    )
    return PlannerState(
        plan=plan,
        ledger=ledger,
        budget_snapshot=budget_ok(),
        blocked_needs=(),
        round_index=len(gains),
        recent_marginal_gains=gains,
        query_history=(),
    )


def test_sufficient_requires_coverage_sources_and_no_high_priority_conflict() -> None:
    result = evaluate_stop(state_with_coverage(0.90, sources=2), budget_ok())

    assert result is not None
    assert result.code is StopCode.SUFFICIENT
    assert result.is_partial is False


def test_budget_exhausted_precedes_plateau() -> None:
    result = evaluate_stop(
        state_with_coverage(0.20, gains=(0.04, 0.03)),
        budget_exhausted(),
    )

    assert result is not None
    assert result.code is StopCode.BUDGET_EXHAUSTED


def test_first_failed_query_does_not_block() -> None:
    first_failure = BlockedNeed(
        need_id="need-1",
        required_source_unavailable=True,
        alternative_strategies_exhausted=False,
        retries_used=1,
        max_retries=2,
    )

    result = evaluate_stop(
        state_with_coverage(0.20),
        budget_ok(),
        blocked_needs=[first_failure],
    )

    assert result is None


def test_blocked_requires_unavailable_source_and_exhausted_alternatives_and_retries() -> None:
    exhausted = BlockedNeed(
        need_id="need-1",
        required_source_unavailable=True,
        alternative_strategies_exhausted=True,
        retries_used=2,
        max_retries=2,
    )

    result = evaluate_stop(
        state_with_coverage(0.20),
        budget_ok(),
        blocked_needs=[exhausted],
    )

    assert result is not None
    assert result.code is StopCode.BLOCKED
    assert result.is_partial is True
    assert result.uncovered_information_needs == ("need-1",)


def test_two_low_gain_rounds_trigger_plateau() -> None:
    result = evaluate_stop(
        state_with_coverage(0.20, gains=(0.04, 0.03)),
        budget_ok(),
    )

    assert result is not None
    assert result.code is StopCode.PLATEAU


def test_high_priority_conflict_prevents_sufficient_stop() -> None:
    result = evaluate_stop(
        state_with_coverage(0.90, sources=2, conflicts=("claim-1",)),
        budget_ok(),
    )

    assert result is None


def test_blocked_need_rejects_invalid_retry_counters() -> None:
    with pytest.raises(ValueError, match="retry"):
        BlockedNeed(
            need_id="need-1",
            required_source_unavailable=True,
            alternative_strategies_exhausted=True,
            retries_used=3,
            max_retries=2,
        )


def test_stop_decision_data_is_not_mutable() -> None:
    from deepresearch.planning.stop import StopDecision

    decision = StopDecision(StopCode.SUFFICIENT, ("coverage_thresholds_met",), (), False)

    with pytest.raises(AttributeError):
        decision.code = StopCode.PLATEAU  # type: ignore[misc]
