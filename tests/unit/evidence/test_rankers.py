from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import pytest

from deepresearch.domain import InformationNeed, RerankScore, SubQuestion
from deepresearch.evidence.features import EvidenceFeatures, FeatureObservation
from deepresearch.evidence.normalize import EvidenceCandidate
from deepresearch.evidence.rankers import (
    R0SearchOrder,
    R1SimilarityOnly,
    R2EvidenceUtility,
)
from deepresearch.runtime import CancellationToken


def subquestion() -> SubQuestion:
    from tests.unit.evidence.test_features import subquestion as make_subquestion

    return make_subquestion()


def need() -> InformationNeed:
    return subquestion().information_needs[0]


def candidates() -> list[EvidenceCandidate]:
    from tests.unit.evidence.test_features import candidate as make_candidate

    first = make_candidate()
    second = EvidenceCandidate(
        first.evidence.model_copy(update={"evidence_id": "e-2"}),
        first.source,
        1,
        first.source_family_id,
    )
    return [second, first]


@pytest.mark.asyncio
async def test_r0_preserves_search_order_and_deterministic_tie_break() -> None:
    ranking = await R0SearchOrder().score(
        subquestion(),
        need(),
        candidates(),
        coverage_score=0.0,
        selected_evidence_ids=frozenset(),
        selected_source_family_ids=frozenset(),
        context_budget=100,
        evaluation_time=datetime(2026, 1, 1, tzinfo=UTC),
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert [score.evidence_id for score in ranking.scores] == ["e-1", "e-2"]
    assert ranking.feature_provenance_by_evidence["e-1"] == {"search_rank": "search_rank:1"}


class FakeSimilarity:
    ranker_id = "R1"

    async def score(
        self,
        information_need: str,
        evidence_spans: Sequence[object],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[RerankScore]:
        del information_need, deadline
        cancellation_token.raise_if_cancelled()
        return [
            RerankScore(
                evidence_id="e-1",
                total=0.8,
                feature_scores={"relevance": 0.8},
                model_id="embed-v1",
            )
            for _ in evidence_spans[:1]
        ]


@pytest.mark.asyncio
async def test_r1_score_contains_only_relevance() -> None:
    ranking = await R1SimilarityOnly(delegate=FakeSimilarity()).score(
        subquestion(),
        need(),
        candidates(),
        coverage_score=0.0,
        selected_evidence_ids=frozenset(),
        selected_source_family_ids=frozenset(),
        context_budget=100,
        evaluation_time=datetime(2026, 1, 1, tzinfo=UTC),
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert set(ranking.scores[0].feature_scores) == {"relevance"}
    assert set(ranking.feature_provenance_by_evidence["e-1"]) == {"relevance"}


class FakeFeatureCalculator:
    def __init__(self) -> None:
        self.last_call: object | None = None

    async def compute(self, **kwargs: object) -> FeatureObservation:
        self.last_call = kwargs
        return FeatureObservation(
            values=EvidenceFeatures(
                relevance=0.9,
                support_strength=0.8,
                source_quality=0.7,
                coverage_gain=0.6,
                independence=0.5,
                freshness=0.4,
                redundancy=0.3,
                risk=0.2,
            ),
            provenance={
                "relevance": "embedding:embed-v1",
                "support_strength": "judge:judge-v1",
                "source_quality": "source:paper",
                "coverage_gain": "coverage:0.25",
                "independence": "family:family-1",
                "freshness": "requirement:none",
                "redundancy": "embedding:embed-v1",
                "risk": "parse_confidence:1.0",
            },
        )


@pytest.mark.asyncio
async def test_r2_persists_all_features_and_clips_total() -> None:
    calculator = FakeFeatureCalculator()
    ranking = await R2EvidenceUtility(feature_calculator=calculator).score(
        subquestion(),
        need(),
        candidates(),
        coverage_score=0.25,
        selected_evidence_ids=frozenset({"e-existing"}),
        selected_source_family_ids=frozenset({"family-existing"}),
        context_budget=100,
        evaluation_time=datetime(2026, 1, 1, tzinfo=UTC),
        deadline=100,
        cancellation_token=CancellationToken(),
    )

    assert 0 <= ranking.scores[0].total <= 1
    assert set(ranking.scores[0].feature_scores) == {
        "relevance",
        "support_strength",
        "source_quality",
        "coverage_gain",
        "independence",
        "freshness",
        "redundancy",
        "risk",
    }
    assert set(ranking.feature_provenance_by_evidence[ranking.scores[0].evidence_id]) == set(
        ranking.scores[0].feature_scores
    )
