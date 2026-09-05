from __future__ import annotations

from typing import Any

import pytest

from deepresearch.domain import (
    EvidenceRequirements,
    FreshnessRequirement,
    InformationNeed,
    ResearchPlan,
    ResearchScope,
    RunBudget,
    SubQuestion,
)
from deepresearch.planning.contracts import (
    PlannerDecision,
    PlannerState,
    QueryCandidate,
)
from deepresearch.runtime import BudgetAccountant


def make_plan() -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-1",
        scope=ResearchScope(
            included_topics=("planner optimization",),
            excluded_topics=(),
            answer_shape="brief",
        ),
        subquestions=(
            SubQuestion(
                id="sq-1",
                question="Which planner optimization methods are documented?",
                rationale_code="coverage",
                importance=0.8,
                dependencies=(),
                information_needs=(
                    InformationNeed(
                        need_id="need-1",
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
            ),
        ),
        created_by_model="test-model",
        prompt_version="planner-contracts-v1",
    )


def budget_ok() -> Any:
    return BudgetAccountant(RunBudget.preset("low"), run_scope="contracts-test").snapshot()


def candidate(query: str = "planner optimization") -> QueryCandidate:
    return QueryCandidate("sq-1", "need-1", query, 0.5, 100, 1, 1.0)


def test_query_candidate_rejects_negative_costs() -> None:
    with pytest.raises(ValueError, match="estimated_tokens"):
        QueryCandidate("sq-1", "need-1", "q", 0.5, -1, 1, 1.0)


def test_planner_decision_stop_cannot_contain_search_candidates() -> None:
    with pytest.raises(ValueError, match="STOP"):
        PlannerDecision("STOP", None, (candidate(),), stop=None, decision_code="STOP")


def test_planner_state_rejects_none_ledger() -> None:
    with pytest.raises(ValueError, match="ledger"):
        PlannerState(
            plan=make_plan(),
            ledger=None,
            budget_snapshot=budget_ok(),
            blocked_needs=(),
            round_index=0,
            recent_marginal_gains=(),
            query_history=(),
        )


def test_query_candidate_and_search_decision_are_immutable() -> None:
    item = candidate()
    decision = PlannerDecision("SEARCH", "sq-1", (item,), stop=None, decision_code="SEARCH")

    assert item.query == "planner optimization"
    assert decision.candidates == (item,)
    with pytest.raises(AttributeError):
        item.query = "mutated"  # type: ignore[misc]

