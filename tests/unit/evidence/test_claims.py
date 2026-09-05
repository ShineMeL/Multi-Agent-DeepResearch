from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from deepresearch.domain import Claim, EvidenceSpan, HtmlLocator, SourceDocument
from deepresearch.evidence.claims import ClaimExtractor, EvidenceJudge
from deepresearch.runtime import CancellationToken


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


def evidence() -> EvidenceSpan:
    excerpt = "The planner improves retrieval quality."
    return EvidenceSpan(
        evidence_id="e-1",
        source_id="source-1",
        locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=len(excerpt)),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=("need-1",),
    )


@pytest.mark.asyncio
async def test_claim_extractor_returns_public_claims_from_cited_markdown() -> None:
    claims = await ClaimExtractor().extract(
        "The planner improves retrieval quality. [E-1]",
        evidence_ids={"E-1"},
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert len(claims) == 1
    assert claims[0].verification_status == "uncertain"
    assert claims[0].claim_type == "fact"


@pytest.mark.asyncio
async def test_evidence_judge_emits_support_link_with_public_provenance() -> None:
    claim = Claim(
        claim_id="c-1",
        text="The planner improves retrieval quality.",
        claim_type="fact",
        entities=("planner",),
        numbers=(),
        qualifiers=(),
        report_section="findings",
        verification_status="uncertain",
    )

    links = await EvidenceJudge().judge(
        claim,
        [evidence()],
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert len(links) == 1
    assert links[0].claim_id == "c-1"
    assert links[0].evidence_id == "e-1"
    assert links[0].relation == "support"
    assert links[0].decision_code

