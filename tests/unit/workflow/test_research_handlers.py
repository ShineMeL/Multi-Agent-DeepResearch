from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Any, cast

import pytest

from deepresearch.domain import EvidenceSpan, HtmlLocator
from deepresearch.runtime import CancellationToken
from deepresearch.storage import LocalArtifactStore
from deepresearch.workflow.baseline_graph import BaselineNodeHandlers
from deepresearch.workflow.research_handlers import ResearchNodeHandlers
from deepresearch.workflow.state import ResearchState
from tests.integration.replay.test_research_graph import _state


class _EvidenceStore:
    def __init__(self, values: tuple[EvidenceSpan, ...]) -> None:
        self._values = {value.evidence_id: value for value in values}

    def get_evidence(self, evidence_id: str) -> EvidenceSpan:
        return self._values[evidence_id]


class _PlanNode:
    initial_plan_generator = object()

    async def __call__(self, state: ResearchState) -> dict[str, object]:
        del state
        return {}


def _handlers(tmp_path, evidence: EvidenceSpan) -> ResearchNodeHandlers:
    baseline = cast("Any", object.__new__(BaselineNodeHandlers))
    baseline.artifact_store = LocalArtifactStore(tmp_path)
    baseline.evidence_store = _EvidenceStore((evidence,))
    baseline.plan = _PlanNode()
    baseline.initial_plan_generator = _PlanNode.initial_plan_generator
    baseline._audit_composition = None
    baseline._audit_result_artifact = lambda context, artifact_id: None
    return ResearchNodeHandlers(cast("BaselineNodeHandlers", baseline))


def _context() -> SimpleNamespace:
    return SimpleNamespace(
        deadline=100.0,
        cancellation_token=CancellationToken(),
        elapsed_tracker=SimpleNamespace(recovered_offset_seconds=0.0),
    )


def _evidence(excerpt: str, evidence_id: str = "e-1") -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id="source-1",
        locator=HtmlLocator(
            paragraph_id="p-1",
            start_char=0,
            end_char=len(excerpt),
        ),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=("need-1",),
    )


@pytest.mark.asyncio
async def test_extract_and_verify_persist_canonical_claim_graph(tmp_path, monkeypatch) -> None:
    evidence = _evidence("The planner improves retrieval quality.")
    handlers = _handlers(tmp_path, evidence)
    context = _context()
    monkeypatch.setattr(ResearchNodeHandlers, "_runtime", staticmethod(lambda: context))
    state = cast(
        "ResearchState",
        {
            **_state(),
            "selected_evidence_ids": (evidence.evidence_id,),
        },
    )
    draft = handlers.artifact_store.put_bytes(
        b"The planner improves retrieval quality. [e-1]",
        media_type="text/markdown; charset=utf-8",
    )
    state = cast("ResearchState", {**state, "draft_artifact_id": draft.artifact_id})

    extracted = await handlers.extract_claims(state)
    assert extracted["claim_ids"]
    claim_graph_id = cast("str", extracted["rank_artifact_id"])
    claim_graph = json.loads(handlers.artifact_store.get_bytes(claim_graph_id))
    assert list(claim_graph) == ["claims", "evidence", "links"]
    assert claim_graph["claims"][0]["verification_status"] == "uncertain"

    verified = await handlers.verify_claims(cast("ResearchState", {**state, **extracted}))
    assert verified["verification_route"] == "FINALIZE"
    assert verified["unsupported_claim_ids"] == ()
    verified_graph = json.loads(
        handlers.artifact_store.get_bytes(cast("str", verified["rank_artifact_id"]))
    )
    assert verified_graph["claims"][0]["verification_status"] == "supported"
    assert verified_graph["links"][0]["relation"] == "support"


@pytest.mark.asyncio
async def test_unsupported_claims_route_to_conservative_resolution(tmp_path, monkeypatch) -> None:
    evidence = _evidence("The planner improves retrieval quality.")
    handlers = _handlers(tmp_path, evidence)
    context = _context()
    monkeypatch.setattr(ResearchNodeHandlers, "_runtime", staticmethod(lambda: context))
    state = cast(
        "ResearchState",
        {
            **_state(),
            "selected_evidence_ids": (evidence.evidence_id,),
        },
    )
    draft = handlers.artifact_store.put_bytes(
        b"Bananas orbit purple galaxies. [e-1]",
        media_type="text/markdown; charset=utf-8",
    )
    state = cast("ResearchState", {**state, "draft_artifact_id": draft.artifact_id})
    extracted = await handlers.extract_claims(state)
    verified = await handlers.verify_claims(cast("ResearchState", {**state, **extracted}))
    assert verified["verification_route"] == "RESOLVE_UNSUPPORTED"
    assert verified["unsupported_claim_ids"]

    resolved = await handlers.resolve_unsupported_claims(
        cast("ResearchState", {**state, **extracted, **verified})
    )
    resolution = json.loads(
        handlers.artifact_store.get_bytes(cast("str", resolved["claim_resolution_artifact_id"]))
    )
    assert resolution["actions"] == {
        cast("tuple[str, ...]", verified["unsupported_claim_ids"])[0]: "DELETE"
    }
    assert (
        handlers.artifact_store.get_bytes(cast("str", resolved["draft_artifact_id"]))
        == b"## Limitations"
    )
