from __future__ import annotations

import hashlib
from collections.abc import Collection, Sequence
from datetime import UTC, date, datetime
from math import isclose
from typing import Literal

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
from deepresearch.evidence.features import (
    DefaultEvidenceFeatureCalculator,
    EvidenceQualityObservation,
    StoreBackedFeatureMaterials,
    SupportObservation,
    coverage_gain,
    freshness_score,
    independence_score,
    normalized_cosine_score,
    risk_score,
    source_quality,
    support_score,
)
from deepresearch.evidence.normalize import EvidenceCandidate
from deepresearch.runtime import CancellationToken

RUN_STARTED_AT = datetime(2026, 1, 10, tzinfo=UTC)


def source(*, source_type: str = "paper", published_at: datetime | None = None) -> SourceDocument:
    return SourceDocument.model_validate(
        {
            "source_id": "source-1",
            "canonical_url": "https://example.com/source",
            "title": "A documented planner method",
            "authors": ("Author",),
            "published_at": published_at,
            "retrieved_at": RUN_STARTED_AT,
            "content_hash": "a" * 64,
            "parsed_content_hash": "b" * 64,
            "source_type": source_type,
            "source_family_id": "family-1",
            "parser_version": "parser-v1",
        }
    )


def span(*, need_ids: tuple[str, ...] = ("need-1",), excerpt: str = "planner method") -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id="e-1",
        source_id="source-1",
        locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=len(excerpt)),
        excerpt=excerpt,
        excerpt_hash=hashlib.sha256(excerpt.encode()).hexdigest(),
        language="en",
        information_need_ids=need_ids,
    )


def candidate() -> EvidenceCandidate:
    return EvidenceCandidate(span(), source(), 1, "family-1")


def subquestion() -> SubQuestion:
    return SubQuestion(
        id="sq-1",
        question="Which planner method is documented?",
        rationale_code="coverage",
        importance=0.8,
        dependencies=(),
        information_needs=(
            InformationNeed(need_id="need-1", text="Documented method", importance=0.75),
            InformationNeed(need_id="need-2", text="Documented limitation", importance=0.25),
        ),
        evidence_requirements=EvidenceRequirements(
            min_independent_sources=1,
            allowed_source_types=frozenset({"paper"}),
            must_include_primary=False,
            freshness=FreshnessRequirement(kind="none"),
        ),
        status="pending",
    )


def support_observation(
    level: Literal["none", "weak", "moderate", "direct"] = "direct",
) -> SupportObservation:
    return SupportObservation(
        level=level,
        judge_model="judge-v1",
        prompt_version="support-v1",
        decision_code="DIRECTNESS_DIRECT",
    )


class FakeEmbedder:
    provider_id = "fake-embedder"
    model_id = "embed-v1"
    model_revision = "1"
    snapshot_sha256 = "c" * 64

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        del deadline
        cancellation_token.raise_if_cancelled()
        return tuple((1.0, 0.0) for _ in texts)


class FakeMaterials:
    def evidence_for_ids(self, evidence_ids: Collection[str]) -> tuple[EvidenceSpan, ...]:
        del evidence_ids
        return ()

    def quality_for(self, evidence_id: str) -> EvidenceQualityObservation:
        assert evidence_id == "e-1"
        return EvidenceQualityObservation()


class FakeSupportJudge:
    async def observe(
        self,
        information_need: InformationNeed,
        evidence: EvidenceSpan,
        *,
        source: SourceDocument,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> SupportObservation:
        del information_need, evidence, source, deadline
        cancellation_token.raise_if_cancelled()
        return support_observation()


def test_support_strength_uses_exact_four_level_rubric() -> None:
    assert support_score(support_observation("none")) == 0.0
    assert isclose(support_score(support_observation("weak")), 1 / 3)
    assert isclose(support_score(support_observation("moderate")), 2 / 3)
    assert support_score(support_observation("direct")) == 1.0


def test_source_quality_uses_exact_weights() -> None:
    quality = EvidenceQualityObservation(
        provenance_completeness=0.5,
        directness=0.75,
        data_verifiability=0.25,
    )
    assert isclose(source_quality(source(source_type="standard"), quality), 0.70)
    assert isclose(source_quality(source(source_type="paper"), quality), 0.66)
    assert isclose(source_quality(source(source_type="unknown"), quality), 0.30)


def test_coverage_gain_uses_need_importance_and_prior_coverage() -> None:
    question = subquestion()
    need = question.information_needs[0]
    assert isclose(coverage_gain(question, need, span(), 0.20), 0.60)
    assert coverage_gain(question, need, span(need_ids=("other",)), 0.20) == 0.0


def test_freshness_and_risk_use_public_buckets() -> None:
    assert freshness_score(None, source(published_at=None), RUN_STARTED_AT) == 1.0
    requirement = FreshnessRequirement(kind="published_after", published_after=date(2025, 1, 1))
    assert freshness_score(requirement, source(published_at=None), RUN_STARTED_AT) == 0.5
    assert freshness_score(
        requirement, source(published_at=datetime(2024, 12, 1, tzinfo=UTC)), RUN_STARTED_AT
    ) == 0.0
    assert risk_score(EvidenceQualityObservation(parse_confidence=1.0)) == 0.0
    assert risk_score(EvidenceQualityObservation(parse_confidence=0.79)) == 0.5
    assert risk_score(EvidenceQualityObservation(is_snippet_only=True)) == 1.0


def test_independence_and_normalized_cosine_are_bounded() -> None:
    assert independence_score("family-new", frozenset({"family-old"})) == 1.0
    assert independence_score("family-old", frozenset({"family-old"})) == 0.0
    assert normalized_cosine_score([1.0, 0.0], [0.0, 1.0]) == 0.5


@pytest.mark.asyncio
async def test_default_calculator_consumes_materials_and_writes_all_features() -> None:
    calculator = DefaultEvidenceFeatureCalculator(
        embedder=FakeEmbedder(),
        embedding_model_id="embed-v1",
        materials=FakeMaterials(),
        support_judge=FakeSupportJudge(),
    )
    question = subquestion()
    need = question.information_needs[0]

    observation = await calculator.compute(
        candidate=candidate(),
        subquestion=question,
        information_need=need,
        subquestion_importance=question.importance,
        need_importance=need.importance,
        freshness_requirement=None,
        coverage_score=0.0,
        selected_evidence_ids=frozenset(),
        selected_source_family_ids=frozenset(),
        context_budget=100,
        evaluation_time=RUN_STARTED_AT,
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert set(observation.values.__dataclass_fields__) == {
        "relevance",
        "support_strength",
        "source_quality",
        "coverage_gain",
        "independence",
        "freshness",
        "redundancy",
        "risk",
    }
    assert set(observation.provenance) == set(observation.values.__dataclass_fields__)


def test_store_backed_materials_resolve_ids_in_stable_order() -> None:
    values = {
        "e-1": span(),
        "e-2": span().model_copy(update={"evidence_id": "e-2"}),
    }
    materials = StoreBackedFeatureMaterials(
        get_evidence=values.__getitem__,
        get_quality=lambda evidence_id: EvidenceQualityObservation(),
    )

    assert [item.evidence_id for item in materials.evidence_for_ids({"e-2", "e-1"})] == [
        "e-1",
        "e-2",
    ]
