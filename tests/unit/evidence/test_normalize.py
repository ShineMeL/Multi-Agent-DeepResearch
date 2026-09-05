from __future__ import annotations

import hashlib
from datetime import UTC, datetime

from deepresearch.domain import EvidenceSpan, HtmlLocator, SourceDocument
from deepresearch.evidence.normalize import EvidenceCandidate, EvidenceNormalizer


def source(source_id: str, *, parsed_hash: str, family_id: str) -> SourceDocument:
    digest = hashlib.sha256(source_id.encode()).hexdigest()
    return SourceDocument.model_validate(
        {
            "source_id": source_id,
            "canonical_url": f"https://example.com/{source_id}",
            "title": "Shared title",
            "authors": ("Author",),
            "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
            "content_hash": digest,
            "parsed_content_hash": parsed_hash,
            "source_type": "paper",
            "source_family_id": family_id,
            "parser_version": "parser-v1",
        }
    )


def evidence(evidence_id: str, source_id: str, excerpt: str) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=HtmlLocator(paragraph_id=evidence_id, start_char=0, end_char=len(excerpt)),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=("need-1",),
    )


def candidate(
    evidence_id: str,
    source_id: str,
    *,
    parsed_hash: str,
    excerpt: str,
    rank: int = 1,
) -> EvidenceCandidate:
    document = source(source_id, parsed_hash=parsed_hash, family_id=f"family-{source_id}")
    return EvidenceCandidate(
        evidence=evidence(evidence_id, source_id, excerpt),
        source=document,
        search_rank=rank,
        source_family_id="placeholder",
    )


def test_same_parsed_hash_assigns_same_source_family() -> None:
    families = EvidenceNormalizer().assign_source_families(
        [
            source("s-1", parsed_hash="a" * 64, family_id="f-1"),
            source("s-2", parsed_hash="a" * 64, family_id="f-2"),
        ]
    )

    assert families["s-1"] == families["s-2"]


def test_dedupe_keeps_distinct_spans_from_same_source_family() -> None:
    first = candidate("e-1", "s-1", parsed_hash="a" * 64, excerpt="route A")
    second = candidate("e-2", "s-2", parsed_hash="a" * 64, excerpt="route B")

    kept = EvidenceNormalizer().dedupe([first, second])

    assert {item.evidence.evidence_id for item in kept} == {"e-1", "e-2"}
    assert len({item.source_family_id for item in kept}) == 1


def test_dedupe_removes_exact_repeated_excerpt() -> None:
    first = candidate("e-1", "s-1", parsed_hash="a" * 64, excerpt="same route", rank=1)
    second = candidate("e-2", "s-2", parsed_hash="a" * 64, excerpt="same route", rank=2)

    kept = EvidenceNormalizer().dedupe([second, first])

    assert [item.evidence.evidence_id for item in kept] == ["e-1"]
