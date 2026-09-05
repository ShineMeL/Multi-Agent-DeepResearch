from __future__ import annotations

from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any, Literal, Protocol, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import (  # pyright: ignore[reportMissingTypeStubs]
    CompiledStateGraph,
)

from deepresearch.planning import FixedPlanner

from .state import (
    ResearchState,
    blocked_need_from_checkpoint,
    blocked_need_to_checkpoint,
    validate_research_state,
)


@dataclass(frozen=True)
class ClaimResolutionRecord:
    claim_id: str
    action: Literal["DELETE", "REWRITE", "MOVE_TO_LIMITATIONS"]
    reason_code: Literal[
        "UNSUPPORTED_FACT",
        "OVERSTATED_SUPPORT",
        "CONTRADICTED",
        "UNCERTAIN",
    ]
    replacement_text: str | None

    def __post_init__(self) -> None:
        if type(self.claim_id) is not str or not self.claim_id.strip():
            raise ValueError("claim_id is required")
        if self.action not in {"DELETE", "REWRITE", "MOVE_TO_LIMITATIONS"}:
            raise ValueError("action is invalid")
        if self.reason_code not in {
            "UNSUPPORTED_FACT",
            "OVERSTATED_SUPPORT",
            "CONTRADICTED",
            "UNCERTAIN",
        }:
            raise ValueError("reason_code is invalid")
        if self.action == "DELETE":
            if self.replacement_text is not None:
                raise ValueError("DELETE replacement must be None")
        elif type(self.replacement_text) is not str or not self.replacement_text.strip():
            raise ValueError("rewrite replacement must be non-empty")


type NodeHandler = Callable[
    [ResearchState], Awaitable[Mapping[str, object]]
]


class InitialPlanNode(Protocol):
    initial_plan_generator: FixedPlanner

    async def __call__(self, state: ResearchState) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class ResearchGraphDependencies:
    validate_request: NodeHandler
    initial_plan_generator: FixedPlanner
    plan: InitialPlanNode
    decide_next: NodeHandler
    search: NodeHandler
    fetch: NodeHandler
    parse_and_normalize: NodeHandler
    store_evidence: NodeHandler
    rank_evidence: NodeHandler
    draft_report: NodeHandler
    extract_claims: NodeHandler
    verify_claims: NodeHandler
    targeted_research: NodeHandler
    resolve_unsupported_claims: NodeHandler
    finalize_citations: NodeHandler
    persist_results: NodeHandler
    checkpointer: BaseCheckpointSaver[str]


def route_after_decide(state: ResearchState) -> Literal["SEARCH", "STOP"]:
    restored = validate_research_state(cast("Mapping[str, object]", state))
    route = restored.get("decision_route")
    if type(route) is not str or route not in {"SEARCH", "STOP"}:
        raise ValueError("decision_route must be SEARCH or STOP")
    return cast("Literal['SEARCH', 'STOP']", route)


def route_after_verify(
    state: ResearchState,
) -> Literal["TARGETED_RESEARCH", "RESOLVE_UNSUPPORTED", "FINALIZE"]:
    restored = validate_research_state(cast("Mapping[str, object]", state))
    route = restored.get("verification_route")
    if type(route) is not str or route not in {
        "TARGETED_RESEARCH",
        "RESOLVE_UNSUPPORTED",
        "FINALIZE",
    }:
        raise ValueError(
            "verification_route must be TARGETED_RESEARCH, RESOLVE_UNSUPPORTED, or FINALIZE"
        )
    return cast(
        "Literal['TARGETED_RESEARCH', 'RESOLVE_UNSUPPORTED', 'FINALIZE']",
        route,
    )


def build_research_graph(
    dependencies: ResearchGraphDependencies,
) -> CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]:
    initial_plan_generator = getattr(
        dependencies.plan,
        "initial_plan_generator",
        None,
    )
    if initial_plan_generator is not dependencies.initial_plan_generator:
        raise ValueError("Plan node must own the configured initial_plan_generator")

    graph = cast("Any", StateGraph(ResearchState))
    graph.add_node("ValidateRequest", dependencies.validate_request)
    graph.add_node("Plan", dependencies.plan)
    graph.add_node("DecideNext", dependencies.decide_next)
    graph.add_node("Search", dependencies.search)
    graph.add_node("Fetch", dependencies.fetch)
    graph.add_node("ParseAndNormalize", dependencies.parse_and_normalize)
    graph.add_node("StoreEvidence", dependencies.store_evidence)
    graph.add_node("RankEvidence", dependencies.rank_evidence)
    graph.add_node("DraftReport", dependencies.draft_report)
    graph.add_node("ExtractClaims", dependencies.extract_claims)
    graph.add_node("VerifyClaims", dependencies.verify_claims)
    graph.add_node("TargetedResearch", dependencies.targeted_research)
    graph.add_node("ResolveUnsupportedClaims", dependencies.resolve_unsupported_claims)
    graph.add_node("FinalizeCitations", dependencies.finalize_citations)
    graph.add_node("PersistResults", dependencies.persist_results)

    graph.set_entry_point("ValidateRequest")
    graph.add_edge("ValidateRequest", "Plan")
    graph.add_edge("Plan", "DecideNext")
    graph.add_conditional_edges(
        "DecideNext",
        route_after_decide,
        {"SEARCH": "Search", "STOP": "DraftReport"},
    )
    graph.add_edge("Search", "Fetch")
    graph.add_edge("Fetch", "ParseAndNormalize")
    graph.add_edge("ParseAndNormalize", "StoreEvidence")
    graph.add_edge("StoreEvidence", "RankEvidence")
    graph.add_edge("RankEvidence", "DecideNext")
    graph.add_edge("DraftReport", "ExtractClaims")
    graph.add_edge("ExtractClaims", "VerifyClaims")
    graph.add_conditional_edges(
        "VerifyClaims",
        route_after_verify,
        {
            "TARGETED_RESEARCH": "TargetedResearch",
            "RESOLVE_UNSUPPORTED": "ResolveUnsupportedClaims",
            "FINALIZE": "FinalizeCitations",
        },
    )
    graph.add_edge("TargetedResearch", "Search")
    graph.add_edge("ResolveUnsupportedClaims", "FinalizeCitations")
    graph.add_edge("FinalizeCitations", "PersistResults")
    graph.add_edge("PersistResults", END)
    return cast(
        "CompiledStateGraph[ResearchState, None, ResearchState, ResearchState]",
        graph.compile(checkpointer=dependencies.checkpointer),
    )


__all__ = [
    "ClaimResolutionRecord",
    "InitialPlanNode",
    "NodeHandler",
    "ResearchGraphDependencies",
    "blocked_need_from_checkpoint",
    "blocked_need_to_checkpoint",
    "build_research_graph",
    "route_after_decide",
    "route_after_verify",
]
