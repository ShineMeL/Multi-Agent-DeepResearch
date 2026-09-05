from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from math import isclose

import pytest

from deepresearch.domain import (
    ClaimEvidenceLink,
    CoverageLedgerEntry,
    EvidenceRequirements,
    EvidenceSpan,
    FreshnessRequirement,
    HtmlLocator,
    InformationNeed,
    ResearchPlan,
    ResearchScope,
    SourceDocument,
    SubQuestion,
)
from deepresearch.planning.ledger import CoverageLedger, update_coverage


def make_plan(*, subquestion_ids: tuple[str, ...] = ("sq-1",)) -> ResearchPlan:
    return ResearchPlan(
        plan_id="plan-ledger",
        scope=ResearchScope(
            included_topics=("planner optimization",),
            excluded_topics=(),
            answer_shape="brief",
        ),
        subquestions=tuple(
            SubQuestion(
                id=subquestion_id,
                question=f"What is documented for {subquestion_id}?",
                rationale_code="coverage",
                importance=0.8 if subquestion_id == "sq-1" else 0.2,
                dependencies=(),
                information_needs=(
                    InformationNeed(
                        need_id=f"need-{subquestion_id}",
                        text="Documented methods",
                        importance=0.8,
                    ),
                ),
                evidence_requirements=EvidenceRequirements(
                    min_independent_sources=1,
                    allowed_source_types=frozenset({"paper"}),
                    must_include_primary=False,
                    freshness=FreshnessRequirement(kind="none"),
                ),
                status="pending",
            )
            for subquestion_id in subquestion_ids
        ),
        created_by_model="test-model",
        prompt_version="ledger-v1",
    )


def source(source_id: str, family_id: str) -> SourceDocument:
    digest = hashlib.sha256(source_id.encode()).hexdigest()
    return SourceDocument.model_validate(
        {
            "source_id": source_id,
            "canonical_url": f"https://example.com/{source_id}",
            "title": f"Source {source_id}",
            "authors": ("Author",),
            "retrieved_at": datetime(2026, 1, 1, tzinfo=UTC),
            "content_hash": digest,
            "parsed_content_hash": digest,
            "source_type": "paper",
            "source_family_id": family_id,
            "parser_version": "test-parser-v1",
        }
    )


def span(evidence_id: str, source_id: str, need_id: str = "need-sq-1") -> EvidenceSpan:
    excerpt = f"Excerpt {evidence_id}"
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=len(excerpt)),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=(need_id,),
    )


def test_coverage_counts_unique_source_families() -> None:
    ledger = CoverageLedger.empty_for(make_plan())

    updated = update_coverage(
        ledger,
        "sq-1",
        selected_evidence=[span("e-1", "s-1"), span("e-2", "s-2")],
        links=[],
        source_documents={"s-1": source("s-1", "f-1"), "s-2": source("s-2", "f-2")},
        marginal_gain=0.3,
        decision_code="RANKED",
    )

    assert updated.get("sq-1").independent_source_count == 2
    assert isclose(updated.get("sq-1").coverage_score, 0.3)
    assert updated.get("sq-1").evidence_ids == ("e-1", "e-2")
    assert ledger.get("sq-1").coverage_score == 0


def test_coverage_deduplicates_source_family_and_conflict_ids() -> None:
    ledger = CoverageLedger.empty_for(make_plan())
    link = ClaimEvidenceLink(
        claim_id="claim-1",
        evidence_id="e-1",
        relation="contradict",
        entailment_score=0.2,
        relevance_score=0.8,
        judge_model="judge",
        prompt_version="v1",
        decision_code="CONTRADICTS",
    )

    updated = update_coverage(
        ledger,
        "sq-1",
        selected_evidence=[span("e-1", "s-1"), span("e-2", "s-2")],
        links=[link, link],
        source_documents={"s-1": source("s-1", "f-1"), "s-2": source("s-2", "f-1")},
        marginal_gain=2.0,
        decision_code="CONFLICT",
    )

    entry = updated.get("sq-1")
    assert entry.independent_source_count == 1
    assert entry.unresolved_conflict_ids == ("claim-1",)
    assert entry.coverage_score == 1.0
    assert entry.uncertainty_score == 0.0
    assert entry.attempt_count == 1


def test_empty_ledger_initializes_every_planned_subquestion() -> None:
    plan = make_plan(subquestion_ids=("sq-1", "sq-2"))
    ledger = CoverageLedger.empty_for(plan)

    assert tuple(entry.subquestion_id for entry in ledger.entries()) == ("sq-1", "sq-2")
    assert all(entry.coverage_score == 0 for entry in ledger.entries())
    assert ledger.weighted_coverage() == 0.0


def test_ledger_requires_one_entry_per_planned_subquestion() -> None:
    plan = make_plan(subquestion_ids=("sq-1", "sq-2"))
    entry = CoverageLedger.empty_for(make_plan()).get("sq-1")

    with pytest.raises(ValueError, match="exactly one"):
        CoverageLedger(plan, {"sq-1": entry})


def test_ledger_replace_returns_new_snapshot() -> None:
    ledger = CoverageLedger.empty_for(make_plan())
    replacement = CoverageLedgerEntry(
        subquestion_id="sq-1",
        coverage_score=0.5,
        independent_source_count=1,
        unresolved_conflict_ids=(),
        uncertainty_score=0.5,
        last_marginal_gain=0.5,
        evidence_ids=("e-1",),
        attempt_count=1,
        last_decision_code="RANKED",
    )

    updated = ledger.replace(replacement)

    assert ledger.get("sq-1").coverage_score == 0
    assert updated.get("sq-1") == replacement
