from __future__ import annotations

import hashlib
from collections.abc import Mapping
from datetime import UTC, datetime

from deepresearch.domain import (
    Claim,
    ClaimEvidenceLink,
    EvidenceSpan,
    HtmlLocator,
    PdfLocator,
    SourceDocument,
)
from deepresearch.evidence.citation_guard import (
    CitationGuard,
    CitationGuardResult,
    CitationMaterialResolver,
)
from deepresearch.evidence.graph import ClaimEvidenceGraph


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


class Materials(CitationMaterialResolver):
    def __init__(self, *, raw: bytes, normalized: str, blocks: Mapping[tuple[str, int, int], str]) -> None:
        self.raw = raw
        self.normalized = normalized
        self.blocks = dict(blocks)

    def raw_bytes_for_source(self, source_id: str) -> bytes:
        del source_id
        return self.raw

    def normalized_document_text(self, source_id: str) -> str:
        del source_id
        return self.normalized

    def html_paragraph_text(self, source_id: str, paragraph_id: str) -> str:
        del source_id, paragraph_id
        return "prefix evidence suffix"

    def pdf_block_text(self, source_id: str, page_index: int, block_index: int) -> str:
        return self.blocks[(source_id, page_index, block_index)]


def _source(source_id: str = "source-1", *, raw: bytes = b"raw") -> SourceDocument:
    normalized = "prefix evidence suffix"
    return SourceDocument.model_validate(
        {
            "source_id": source_id,
            "canonical_url": "https://example.com/source",
            "title": "Source",
            "authors": ("Author",),
            "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
            "content_hash": hashlib.sha256(raw).hexdigest(),
            "parsed_content_hash": _sha256_text(normalized),
            "source_type": "paper",
            "source_family_id": "family-1",
            "parser_version": "parser-v1",
        }
    )


def _evidence(
    evidence_id: str = "E-1",
    *,
    locator: HtmlLocator | PdfLocator | None = None,
    excerpt: str = "evidence",
) -> EvidenceSpan:
    selected_locator = locator or HtmlLocator(
        paragraph_id="p-1", start_char=7, end_char=15
    )
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id="source-1",
        locator=selected_locator,
        excerpt=excerpt,
        excerpt_hash=_sha256_text(excerpt),
        language="en",
        information_need_ids=("need-1",),
    )


def _graph(*spans: EvidenceSpan) -> ClaimEvidenceGraph:
    graph = ClaimEvidenceGraph()
    graph.add_claim(
        Claim(
            claim_id="c-1",
            text="A claim.",
            claim_type="fact",
            entities=(),
            numbers=(),
            qualifiers=(),
            report_section="findings",
            verification_status="supported",
        )
    )
    for span in spans:
        graph.add_evidence(span)
        graph.add_link(
            ClaimEvidenceLink(
                claim_id="c-1",
                evidence_id=span.evidence_id,
                relation="support",
                entailment_score=1.0,
                relevance_score=1.0,
                judge_model="judge-v1",
                prompt_version="judge-v1",
                decision_code="SUPPORTED",
            )
        )
    return graph


def _graph_without_links(span: EvidenceSpan) -> ClaimEvidenceGraph:
    graph = ClaimEvidenceGraph()
    graph.add_claim(
        Claim(
            claim_id="c-1",
            text="A claim.",
            claim_type="fact",
            entities=(),
            numbers=(),
            qualifiers=(),
            report_section="findings",
            verification_status="supported",
        )
    )
    graph.add_evidence(span)
    return graph


def test_guard_rejects_unknown_citation_id() -> None:
    result = CitationGuard().verify(
        "[E-missing] claim",
        _graph(_evidence()),
        {"E-1": _evidence()},
        {"source-1": _source()},
        materials=Materials(
            raw=b"raw", normalized="prefix evidence suffix", blocks={}
        ),
    )

    assert result.valid is False
    assert "unknown citation" in " ".join(result.errors)
    assert result.checked_citation_ids == ("E-missing",)


def test_guard_rejects_citation_without_graph_edge() -> None:
    span = _evidence()
    result = CitationGuard().verify(
        "claim [E-1]",
        _graph_without_links(span),
        {"E-1": span},
        {"source-1": _source()},
        materials=Materials(
            raw=b"raw", normalized="prefix evidence suffix", blocks={}
        ),
    )

    assert result.valid is False
    assert "graph link" in " ".join(result.errors)


def test_guard_rejects_excerpt_hash_mismatch() -> None:
    tampered = _evidence(excerpt="tampered")
    result = CitationGuard().verify(
        "claim [E-1]",
        _graph(tampered),
        {"E-1": tampered},
        {"source-1": _source()},
        materials=Materials(
            raw=b"raw", normalized="prefix evidence suffix", blocks={}
        ),
    )

    assert result.valid is False
    assert "excerpt hash" in " ".join(result.errors)


def test_guard_resolves_pdf_page_and_block_before_offsets() -> None:
    span = _evidence(
        "E-pdf",
        locator=PdfLocator(page_index=1, block_index=3, start_char=7, end_char=15),
    )
    result = CitationGuard().verify(
        "claim [E-pdf]",
        _graph(span),
        {"E-pdf": span},
        {"source-1": _source()},
        materials=Materials(
            raw=b"raw",
            normalized="prefix evidence suffix",
            blocks={("source-1", 1, 3): "prefix evidence suffix"},
        ),
    )

    assert result == CitationGuardResult(
        valid=True, errors=(), checked_citation_ids=("E-pdf",)
    )


def test_guard_rejects_raw_content_hash_mismatch() -> None:
    result = CitationGuard().verify(
        "claim [E-1]",
        _graph(_evidence()),
        {"E-1": _evidence()},
        {"source-1": _source()},
        materials=Materials(
            raw=b"tampered", normalized="prefix evidence suffix", blocks={}
        ),
    )

    assert result.valid is False
    assert "content hash" in " ".join(result.errors)
