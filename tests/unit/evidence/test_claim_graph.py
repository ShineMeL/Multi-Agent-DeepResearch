from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from deepresearch.domain import Claim, ClaimEvidenceLink, EvidenceSpan, HtmlLocator, SourceDocument
from deepresearch.evidence.graph import ClaimEvidenceGraph, GraphValidationResult


def claim(claim_id: str = "c-1") -> Claim:
    return Claim(
        claim_id=claim_id,
        text="The planner improves retrieval quality.",
        claim_type="fact",
        entities=("planner",),
        numbers=(),
        qualifiers=(),
        report_section="findings",
        verification_status="supported",
    )


def evidence(evidence_id: str = "e-1") -> EvidenceSpan:
    excerpt = f"Excerpt {evidence_id}"
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id="source-1",
        locator=HtmlLocator(paragraph_id=evidence_id, start_char=0, end_char=len(excerpt)),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=("need-1",),
    )


def source() -> SourceDocument:
    return SourceDocument.model_validate(
        {
            "source_id": "source-1",
            "canonical_url": "https://example.com/source",
            "title": "Planner paper",
            "authors": ("Author",),
            "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
            "content_hash": "a" * 64,
            "parsed_content_hash": "b" * 64,
            "source_type": "paper",
            "source_family_id": "family-1",
            "parser_version": "parser-v1",
        }
    )


def link(claim_id: str, evidence_id: str, relation: str = "support") -> ClaimEvidenceLink:
    return ClaimEvidenceLink(
        claim_id=claim_id,
        evidence_id=evidence_id,
        relation=relation,
        entailment_score=0.8,
        relevance_score=0.9,
        judge_model="judge-v1",
        prompt_version="judge-v1",
        decision_code="SUPPORTED",
    )


def graph_with_claim_and_evidence() -> ClaimEvidenceGraph:
    graph = ClaimEvidenceGraph()
    graph.add_claim(claim())
    graph.add_evidence(evidence("e-1"))
    graph.add_evidence(evidence("e-2"))
    return graph


def test_graph_rejects_link_to_unknown_evidence() -> None:
    graph = ClaimEvidenceGraph()
    graph.add_claim(claim())

    with pytest.raises(ValueError, match="unknown evidence"):
        graph.add_link(link("c-1", "e-missing"))


def test_graph_keeps_support_and_contradiction_edges() -> None:
    graph = graph_with_claim_and_evidence()
    graph.add_link(link("c-1", "e-1", relation="support"))
    graph.add_link(link("c-1", "e-2", relation="contradict"))

    assert {item.relation for item in graph.links_for_claim("c-1")} == {
        "support",
        "contradict",
    }


def test_valid_graph_returns_typed_validation_result_and_json() -> None:
    graph = graph_with_claim_and_evidence()
    graph.add_link(link("c-1", "e-1"))

    assert graph.validate() == GraphValidationResult(valid=True, error_codes=())
    payload = graph.to_json()
    assert payload["claims"][0]["claim_id"] == "c-1"  # type: ignore[index]
    assert payload["links"][0]["relation"] == "support"  # type: ignore[index]


def test_graph_reports_duplicate_links() -> None:
    graph = graph_with_claim_and_evidence()
    graph.add_link(link("c-1", "e-1"))

    with pytest.raises(ValueError, match="duplicate"):
        graph.add_link(link("c-1", "e-1"))

