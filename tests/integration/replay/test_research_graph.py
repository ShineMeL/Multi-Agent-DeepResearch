from __future__ import annotations

from collections.abc import Mapping
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.domain import (
    FreshnessRequirement,
    ResearchRequest,
    RunBudget,
    RunConfig,
    RunEvent,
)
from deepresearch.runtime import BudgetAccountant, CancellationToken, checkpoint_serializer
from deepresearch.workflow.research_graph import (
    ClaimResolutionRecord,
    ResearchGraphDependencies,
    build_research_graph,
    route_after_decide,
    route_after_verify,
)
from deepresearch.workflow.runner import LangGraphResearchRunner
from deepresearch.workflow.state import (
    BaselineBlockedNeed,
    ResearchState,
    blocked_need_from_checkpoint,
    blocked_need_to_checkpoint,
    validate_research_state,
)


def _request() -> ResearchRequest:
    return ResearchRequest(
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


def _state() -> ResearchState:
    request = _request()
    return cast(
        "ResearchState",
        {
            "run_id": "run-1",
            "thread_id": "thread-1",
            "request": request,
            "config_sha256": "a" * 64,
            "plan_id": None,
            "plan_artifact_id": None,
            "pending_subquestion_ids": ("sq-1",),
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
        },
    )


class PlanNode:
    def __init__(self, generator: object, calls: list[str]) -> None:
        self.initial_plan_generator = generator
        self._calls = calls

    async def __call__(self, state: ResearchState) -> Mapping[str, object]:
        del state
        self._calls.append("Plan")
        return {"plan_id": "plan-1", "pending_subquestion_ids": ()}


class _EventSink:
    async def __call__(self, event: RunEvent) -> None:
        del event

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        del run_id, seq
        return None


def _dependencies(calls: list[str]) -> ResearchGraphDependencies:
    generator = object()
    plan_node = PlanNode(generator, calls)

    def node(name: str):
        async def invoke(state: ResearchState) -> Mapping[str, object]:
            del state
            calls.append(name)
            if name == "DecideNext":
                return {"decision_route": "STOP", "stop_reason": "SUFFICIENT"}
            if name == "VerifyClaims":
                return {"verification_route": "FINALIZE", "unsupported_claim_ids": ()}
            if name == "PersistResults":
                return {
                    "report_artifact_id": "sha256:" + "1" * 64,
                    "manifest_artifact_id": "sha256:" + "2" * 64,
                }
            return {}

        return invoke

    return ResearchGraphDependencies(
        validate_request=node("ValidateRequest"),
        initial_plan_generator=cast("Any", generator),
        plan=cast("Any", plan_node),
        decide_next=node("DecideNext"),
        search=node("Search"),
        fetch=node("Fetch"),
        parse_and_normalize=node("ParseAndNormalize"),
        store_evidence=node("StoreEvidence"),
        rank_evidence=node("RankEvidence"),
        draft_report=node("DraftReport"),
        extract_claims=node("ExtractClaims"),
        verify_claims=node("VerifyClaims"),
        targeted_research=node("TargetedResearch"),
        resolve_unsupported_claims=node("ResolveUnsupportedClaims"),
        finalize_citations=node("FinalizeCitations"),
        persist_results=node("PersistResults"),
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
    )


@pytest.mark.asyncio
async def test_graph_routes_sufficient_to_terminal_without_replan() -> None:
    calls: list[str] = []
    graph = cast("Any", build_research_graph(_dependencies(calls)))

    output = await graph.ainvoke(
        _state(),
        {"configurable": {"thread_id": "thread-1"}},
    )

    assert output["stop_reason"] == "SUFFICIENT"
    assert calls == [
        "ValidateRequest",
        "Plan",
        "DecideNext",
        "DraftReport",
        "ExtractClaims",
        "VerifyClaims",
        "FinalizeCitations",
        "PersistResults",
    ]


def test_research_state_blocked_need_strictly_roundtrips_with_core_serializer() -> None:
    record: BaselineBlockedNeed = {
        "need_id": "need-1",
        "required_source_unavailable": True,
        "alternative_strategies_exhausted": True,
        "retry_count": 2,
        "max_retries": 2,
    }
    state = _state()
    state = cast("ResearchState", {**state, "blocked_needs": (record,), "recent_marginal_gains": (0.04, 0.03)})
    restored = checkpoint_serializer().loads_typed(checkpoint_serializer().dumps_typed(state))
    assert restored["blocked_needs"] == (record,)
    assert type(restored["blocked_needs"][0]) is dict
    internal = blocked_need_from_checkpoint(restored["blocked_needs"][0])
    assert internal.terminal is True
    assert blocked_need_to_checkpoint(internal) == record


def test_research_state_rejects_unknown_route_values() -> None:
    candidate = {**_state(), "decision_route": "INVALID"}
    with pytest.raises(ValueError, match="checkpoint state is invalid"):
        validate_research_state(candidate)


def test_research_routes_require_validated_public_labels() -> None:
    candidate = {**_state(), "decision_route": "STOP"}
    assert route_after_decide(cast("ResearchState", candidate)) == "STOP"
    candidate = {**_state(), "verification_route": "FINALIZE"}
    assert route_after_verify(cast("ResearchState", candidate)) == "FINALIZE"


def test_claim_resolution_record_requires_public_replacement_rules() -> None:
    assert ClaimResolutionRecord(
        claim_id="c-1",
        action="DELETE",
        reason_code="UNSUPPORTED_FACT",
        replacement_text=None,
    ).replacement_text is None
    with pytest.raises(ValueError, match="replacement"):
        ClaimResolutionRecord(
            claim_id="c-1",
            action="REWRITE",
            reason_code="OVERSTATED_SUPPORT",
            replacement_text=None,
        )


def test_existing_runner_dispatches_research_graph_without_second_runner_class() -> None:
    calls: list[str] = []
    research_graph = build_research_graph(_dependencies(calls))
    baseline_graph = object()
    runner = LangGraphResearchRunner(
        baseline_graph=cast("Any", baseline_graph),
        research_graph=research_graph,
    )
    assert type(runner) is LangGraphResearchRunner
    config = cast(
        "Any",
        type("Config", (), {"workflow_id": "research-v1"})(),
    )
    assert runner._graph_for_config(config) is research_graph  # pyright: ignore[reportPrivateUsage]


@pytest.mark.asyncio
async def test_runner_executes_injected_research_graph() -> None:
    calls: list[str] = []
    research_graph = build_research_graph(_dependencies(calls))
    runner = LangGraphResearchRunner(
        baseline_graph=cast("Any", object()),
        research_graph=research_graph,
    )
    request = _request()
    config = RunConfig(
        request=request,
        workflow_id="research-v1",
        planner_id="P1",
        ranker_id="R1",
        budget=RunBudget.preset("low"),
        prompt_versions={},
    )

    result = await runner.run(
        run_id="run-research-1",
        thread_id="thread-research-1",
        config=config,
        checkpoint=None,
        emit=cast("Any", _EventSink()),
        cancellation_token=CancellationToken(),
    )

    assert result.status == "completed"
    assert result.stop_reason == "SUFFICIENT"
