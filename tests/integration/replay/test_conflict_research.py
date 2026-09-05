from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunBudget
from deepresearch.planning.stop import StopCode
from deepresearch.runtime import BudgetAccountant, checkpoint_serializer
from deepresearch.workflow.research_graph import (
    ResearchGraphDependencies,
    build_research_graph,
    route_after_verify,
)
from deepresearch.workflow.state import ResearchState

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "replay"


def _state(**updates: object) -> ResearchState:
    request = ResearchRequest(
        question="Which planner strategy is supported?",
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
    base: dict[str, object] = {
        "run_id": "run-conflict",
        "thread_id": "thread-conflict",
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
        "directional_research_rounds": 0,
        "unsupported_claim_ids": (),
    }
    base.update(updates)
    return cast("ResearchState", base)


def test_conflict_route_allows_one_directional_round_then_resolves() -> None:
    first = _state(
        directional_research_rounds=0,
        verification_route="TARGETED_RESEARCH",
        unsupported_claim_ids=("claim-1",),
    )
    assert route_after_verify(first) == "TARGETED_RESEARCH"

    second = _state(
        directional_research_rounds=1,
        verification_route="TARGETED_RESEARCH",
        unsupported_claim_ids=("claim-1",),
    )
    assert route_after_verify(second) == "RESOLVE_UNSUPPORTED"

    finalized = _state(
        directional_research_rounds=1,
        verification_route="TARGETED_RESEARCH",
        unsupported_claim_ids=(),
    )
    assert route_after_verify(finalized) == "FINALIZE"

    stale_route = _state(
        directional_research_rounds=1,
        verification_route="RESOLVE_UNSUPPORTED",
        unsupported_claim_ids=(),
    )
    assert route_after_verify(stale_route) == "FINALIZE"


def test_conflict_stop_code_remains_publicly_typed() -> None:
    assert StopCode.PLATEAU.value == "PLATEAU"


def test_conflict_fixture_allows_at_most_one_targeted_round() -> None:
    scenario = json.loads(
        (FIXTURE_ROOT / "conflict_research" / "scenario.json").read_text(
            encoding="utf-8"
        )
    )
    assert scenario["targeted_research_event_count"] == 1
    assert scenario["targeted_research_rounds"] == 1


def test_blocked_fixture_exposes_alternative_before_terminal_block() -> None:
    scenario = json.loads(
        (FIXTURE_ROOT / "stop_blocked" / "scenario.json").read_text(encoding="utf-8")
    )
    assert scenario["public_steps"] == [
        "PROVIDER_FAILURE",
        "ALTERNATIVE_STRATEGY",
        "BLOCKED",
    ]


class _PlanNode:
    def __init__(self, generator: object, calls: list[str]) -> None:
        self.initial_plan_generator = generator
        self._calls = calls

    async def __call__(self, state: ResearchState) -> Mapping[str, object]:
        del state
        self._calls.append("Plan")
        return {}


def _dependencies(calls: list[str]) -> ResearchGraphDependencies:
    generator = object()
    plan_node = _PlanNode(generator, calls)
    verify_calls = 0
    decide_calls = 0

    def node(name: str):
        async def invoke(state: ResearchState) -> Mapping[str, object]:
            nonlocal decide_calls, verify_calls
            del state
            calls.append(name)
            if name == "DecideNext":
                decide_calls += 1
                return {"decision_route": "STOP", "stop_reason": "PLATEAU"}
            if name == "VerifyClaims":
                verify_calls += 1
                if verify_calls == 1:
                    return {
                        "verification_route": "TARGETED_RESEARCH",
                        "unsupported_claim_ids": ("claim-1",),
                        "directional_research_rounds": 0,
                    }
                return {
                    "verification_route": "FINALIZE",
                    "unsupported_claim_ids": (),
                }
            if name == "TargetedResearch":
                return {"directional_research_rounds": 1}
            if name == "PersistResults":
                return {
                    "report_artifact_id": "sha256:conflict-report",
                    "manifest_artifact_id": "sha256:conflict-manifest",
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
async def test_conflict_graph_executes_at_most_one_directional_research_round() -> None:
    calls: list[str] = []
    graph = cast("Any", build_research_graph(_dependencies(calls)))
    output = await graph.ainvoke(
        _state(),
        {"configurable": {"thread_id": "thread-conflict"}},
    )

    assert output["directional_research_rounds"] == 1
    assert calls.count("TargetedResearch") == 1
    first_verify = calls.index("VerifyClaims")
    second_verify = calls.index("VerifyClaims", first_verify + 1)
    assert first_verify < calls.index("TargetedResearch") < second_verify
