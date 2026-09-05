from __future__ import annotations

from typing import cast

from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunBudget
from deepresearch.runtime import BudgetAccountant, checkpoint_serializer
from deepresearch.workflow.state import ResearchState, validate_research_state


def test_checkpoint_serializer_keeps_research_state_primitive_extensions() -> None:
    request = ResearchRequest(
        question="Which methods are documented?",
        output_requirements={"answer_shape": "brief"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="low",
    )
    state = cast(
        "ResearchState",
        {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "request": request,
            "config_sha256": "a" * 64,
            "plan_id": None,
            "plan_artifact_id": None,
            "pending_subquestion_ids": (),
            "active_subquestion_id": None,
            "query_ids": (),
            "source_ids": (),
            "evidence_ids": (),
            "selected_evidence_ids": (),
            "coverage_ledger": (),
            "high_priority_unresolved_conflict_ids": (),
            "blocked_needs": (),
            "recent_marginal_gains": (),
            "baseline_work_artifact_ids": (),
            "budget_snapshot": BudgetAccountant(RunBudget.preset("low")).snapshot(),
            "stop_reason": None,
            "is_partial": False,
            "draft_artifact_id": None,
            "report_artifact_id": None,
            "evidence_graph_artifact_id": None,
            "manifest_artifact_id": None,
            "next_event_seq": 1,
            "failed_node": None,
            "elapsed_wall_seconds": 0.0,
            "error_code": None,
            "planner_round_index": 1,
            "directional_research_rounds": 0,
            "claim_ids": ("c-1",),
            "unsupported_claim_ids": (),
        },
    )
    validated = validate_research_state(state)
    serializer = checkpoint_serializer()
    restored = serializer.loads_typed(serializer.dumps_typed(validated))
    assert restored["planner_round_index"] == 1
    assert restored["directional_research_rounds"] == 0
    assert restored["claim_ids"] == ("c-1",)
