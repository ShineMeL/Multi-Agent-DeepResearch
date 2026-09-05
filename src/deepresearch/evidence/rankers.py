from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from datetime import datetime
from math import isfinite
from types import MappingProxyType
from typing import Literal, Protocol

from deepresearch.domain import EvidenceSpan, InformationNeed, RerankScore, SubQuestion
from deepresearch.providers import Deadline
from deepresearch.runtime import CancellationToken

from .features import EvidenceFeatureCalculator, EvidenceFeatures
from .normalize import EvidenceCandidate


def _clip(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class EvidenceRankingResult:
    scores: tuple[RerankScore, ...]
    feature_provenance_by_evidence: Mapping[str, Mapping[str, str]]

    def __post_init__(self) -> None:
        if type(self.scores) is not tuple:
            raise TypeError("scores must be a tuple")
        score_ids = {score.evidence_id for score in self.scores}
        if set(self.feature_provenance_by_evidence) != score_ids:
            raise ValueError("feature provenance must cover every score")
        for score in self.scores:
            provenance = self.feature_provenance_by_evidence[score.evidence_id]
            if set(score.feature_scores) != set(provenance):
                raise ValueError("feature scores and provenance keys must match")


class EvidenceRanker(Protocol):
    ranker_id: Literal["R0", "R1", "R2"]

    async def score(
        self,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        candidates: Sequence[EvidenceCandidate],
        *,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> EvidenceRankingResult: ...


class _SimilarityDelegate(Protocol):
    ranker_id: str

    async def score(
        self,
        information_need: str,
        evidence_spans: Sequence[EvidenceSpan],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[RerankScore]: ...


class R0SearchOrder:
    ranker_id: Literal["R0"] = "R0"

    async def score(
        self,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        candidates: Sequence[EvidenceCandidate],
        *,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> EvidenceRankingResult:
        del (
            subquestion,
            information_need,
            coverage_score,
            selected_evidence_ids,
            selected_source_family_ids,
            context_budget,
            evaluation_time,
            deadline,
        )
        cancellation_token.raise_if_cancelled()
        ordered = sorted(candidates, key=lambda item: (item.search_rank, item.evidence.evidence_id))
        scores: list[RerankScore] = []
        provenance: dict[str, Mapping[str, str]] = {}
        for candidate in ordered:
            relevance = _clip(1.0 / candidate.search_rank, field="search_rank")
            evidence_id = candidate.evidence.evidence_id
            scores.append(
                RerankScore(
                    evidence_id=evidence_id,
                    total=relevance,
                    feature_scores={"search_rank": relevance},
                    model_id="search-order-v1",
                    prompt_version=None,
                )
            )
            provenance[evidence_id] = {"search_rank": f"search_rank:{candidate.search_rank}"}
        return EvidenceRankingResult(tuple(scores), MappingProxyType(provenance))


class R1SimilarityOnly:
    ranker_id: Literal["R1"] = "R1"

    def __init__(self, *, delegate: _SimilarityDelegate) -> None:
        self.delegate = delegate

    async def score(
        self,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        candidates: Sequence[EvidenceCandidate],
        *,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> EvidenceRankingResult:
        del subquestion, coverage_score, selected_evidence_ids
        del selected_source_family_ids, context_budget, evaluation_time
        cancellation_token.raise_if_cancelled()
        scores = tuple(
            await self.delegate.score(
                information_need.text,
                [item.evidence for item in candidates],
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
        )
        ordered = tuple(sorted(scores, key=lambda item: (-item.total, item.evidence_id)))
        provenance = {
            score.evidence_id: {"relevance": f"SimilarityRanker:{self.delegate.ranker_id}"}
            for score in ordered
        }
        return EvidenceRankingResult(ordered, MappingProxyType(provenance))


class R2EvidenceUtility:
    ranker_id: Literal["R2"] = "R2"

    def __init__(
        self,
        *,
        feature_calculator: EvidenceFeatureCalculator,
        model_id: str = "evidence-features-v1",
        prompt_version: str = "r2-utility-v1",
    ) -> None:
        self.feature_calculator = feature_calculator
        self.model_id = model_id
        self.prompt_version = prompt_version

    async def score(
        self,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        candidates: Sequence[EvidenceCandidate],
        *,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> EvidenceRankingResult:
        scores: list[RerankScore] = []
        provenance: dict[str, Mapping[str, str]] = {}
        for candidate in candidates:
            observation = await self.feature_calculator.compute(
                candidate=candidate,
                subquestion=subquestion,
                information_need=information_need,
                subquestion_importance=subquestion.importance,
                need_importance=information_need.importance,
                freshness_requirement=subquestion.evidence_requirements.freshness,
                coverage_score=coverage_score,
                selected_evidence_ids=selected_evidence_ids,
                selected_source_family_ids=selected_source_family_ids,
                context_budget=context_budget,
                evaluation_time=evaluation_time,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            features: EvidenceFeatures = observation.values
            feature_scores = asdict(features)
            total = _clip(
                0.25 * features.relevance
                + 0.20 * features.support_strength
                + 0.15 * features.source_quality
                + 0.20 * features.coverage_gain
                + 0.10 * features.independence
                + 0.05 * features.freshness
                - 0.15 * features.redundancy
                - 0.05 * features.risk,
                field="total",
            )
            score = RerankScore(
                evidence_id=candidate.evidence.evidence_id,
                total=total,
                feature_scores=feature_scores,
                model_id=self.model_id,
                prompt_version=self.prompt_version,
            )
            scores.append(score)
            provenance[score.evidence_id] = dict(observation.provenance)
        ordered = tuple(sorted(scores, key=lambda item: (-item.total, item.evidence_id)))
        return EvidenceRankingResult(ordered, MappingProxyType(provenance))


__all__ = [
    "EvidenceRanker",
    "EvidenceRankingResult",
    "R0SearchOrder",
    "R1SimilarityOnly",
    "R2EvidenceUtility",
]
