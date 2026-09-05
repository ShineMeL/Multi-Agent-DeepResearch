from __future__ import annotations

import hashlib
from datetime import UTC, datetime

import pytest

from deepresearch.domain import (
    EvidenceRequirements,
    EvidenceSpan,
    FreshnessRequirement,
    HtmlLocator,
    InformationNeed,
    SourceDocument,
    SubQuestion,
)
from deepresearch.evidence.normalize import EvidenceCandidate
from deepresearch.evidence.pass_a import PassASelector
from deepresearch.runtime import CancellationToken


def subquestion() -> SubQuestion:
    return SubQuestion(
        id="sq-1",
        question="Which planner methods are documented?",
        rationale_code="coverage",
        importance=0.8,
        dependencies=(),
        information_needs=(
            InformationNeed(need_id="need-1", text="Documented methods", importance=0.8),
        ),
        evidence_requirements=EvidenceRequirements(
            min_independent_sources=1,
            allowed_source_types=frozenset({"paper"}),
            must_include_primary=False,
            freshness=FreshnessRequirement(kind="none"),
        ),
        status="pending",
    )


def candidate(evidence_id: str, excerpt: str, rank: int) -> EvidenceCandidate:
    source_id = f"source-{evidence_id}"
    digest = hashlib.sha256(source_id.encode()).hexdigest()
    source = SourceDocument.model_validate(
        {
            "source_id": source_id,
            "canonical_url": f"https://example.com/{source_id}",
            "title": f"Source {evidence_id}",
            "authors": ("Author",),
            "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
            "content_hash": digest,
            "parsed_content_hash": hashlib.sha256(excerpt.encode()).hexdigest(),
            "source_type": "paper",
            "source_family_id": f"family-{evidence_id}",
            "parser_version": "parser-v1",
        }
    )
    evidence = EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=HtmlLocator(paragraph_id=evidence_id, start_char=0, end_char=len(excerpt)),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=("need-1",),
    )
    return EvidenceCandidate(evidence, source, rank, source.source_family_id)


@pytest.mark.asyncio
async def test_pass_a_retains_independent_conflicting_evidence() -> None:
    selector = PassASelector()
    result = await selector.select(
        subquestion(),
        [
            candidate("e-support", "The planner improves retrieval quality.", 1),
            candidate("e-conflict", "The planner does not improve retrieval quality.", 2),
        ],
        context_budget=1000,
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert {item.evidence.evidence_id for item in result.selected} == {
        "e-support",
        "e-conflict",
    }
    assert result.rejected_evidence_ids == ()
    assert result.used_context_tokens > 0


@pytest.mark.asyncio
async def test_pass_a_rejects_candidates_over_context_budget_deterministically() -> None:
    selector = PassASelector()
    result = await selector.select(
        subquestion(),
        [candidate("e-1", "planner method evidence", 1), candidate("e-2", "other evidence", 2)],
        context_budget=3,
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert len(result.selected) == 1
    assert result.rejected_evidence_ids == ("e-2",)
    assert result.used_context_tokens == 3
