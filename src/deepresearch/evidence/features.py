from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Callable, Collection, Mapping, Sequence
from collections.abc import Set as AbstractSet
from dataclasses import dataclass, fields
from datetime import datetime
from decimal import Decimal
from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from deepresearch.domain import (
    EvidenceSpan,
    FreshnessRequirement,
    InformationNeed,
    SourceDocument,
    SubQuestion,
)
from deepresearch.providers import (
    Deadline,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    TextEmbedder,
)
from deepresearch.runtime import (
    BudgetAccountant,
    CancellationToken,
    ResourceEstimate,
)

from .normalize import EvidenceCandidate

SOURCE_TYPE_SCORES: Mapping[str, float] = {
    "paper": 0.9,
    "official_documentation": 1.0,
    "standard": 1.0,
    "primary_data": 1.0,
    "first_party_statement": 0.8,
    "secondary_analysis": 0.6,
    "news": 0.4,
    "unknown": 0.0,
}
SUPPORT_SCORES: Mapping[str, float] = {
    "none": 0.0,
    "weak": 1 / 3,
    "moderate": 2 / 3,
    "direct": 1.0,
}
_FEATURE_NAMES = (
    "relevance",
    "support_strength",
    "source_quality",
    "coverage_gain",
    "independence",
    "freshness",
    "redundancy",
    "risk",
)


def _clip(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(min(1.0, max(0.0, value)))


@dataclass(frozen=True)
class EvidenceFeatures:
    relevance: float
    support_strength: float
    source_quality: float
    coverage_gain: float
    independence: float
    freshness: float
    redundancy: float
    risk: float

    def __post_init__(self) -> None:
        for field in fields(self):
            value = getattr(self, field.name)
            clipped = _clip(value, field=field.name)
            if type(value) not in (int, float) or clipped != value:
                raise ValueError(f"{field.name} must be within [0, 1]")


class EvidenceQualityObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    provenance_completeness: float = Field(default=0.5, ge=0.0, le=1.0)
    directness: float = Field(default=0.5, ge=0.0, le=1.0)
    data_verifiability: float = Field(default=0.5, ge=0.0, le=1.0)
    parse_confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    is_truncated: bool = False
    is_snippet_only: bool = False
    has_stable_locator: bool = True


class SupportObservation(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["none", "weak", "moderate", "direct"]
    judge_model: str
    prompt_version: str
    decision_code: str


@dataclass(frozen=True)
class FeatureObservation:
    values: EvidenceFeatures
    provenance: Mapping[str, str]

    def __post_init__(self) -> None:
        if set(self.provenance) != set(_FEATURE_NAMES):
            raise ValueError("feature provenance must cover all eight features")
        if any(type(value) is not str or not value for value in self.provenance.values()):
            raise ValueError("feature provenance values must be non-empty strings")


@runtime_checkable
class EvidenceFeatureCalculator(Protocol):
    async def compute(
        self,
        *,
        candidate: EvidenceCandidate,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        subquestion_importance: float,
        need_importance: float,
        freshness_requirement: FreshnessRequirement | None,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> FeatureObservation: ...


@runtime_checkable
class EvidenceFeatureMaterialResolver(Protocol):
    def evidence_for_ids(self, evidence_ids: Collection[str]) -> tuple[EvidenceSpan, ...]: ...

    def quality_for(self, evidence_id: str) -> EvidenceQualityObservation: ...


class StoreBackedFeatureMaterials(EvidenceFeatureMaterialResolver):
    def __init__(
        self,
        *,
        get_evidence: Callable[[str], EvidenceSpan],
        get_quality: Callable[[str], EvidenceQualityObservation],
    ) -> None:
        self.get_evidence = get_evidence
        self.get_quality = get_quality

    def evidence_for_ids(self, evidence_ids: Collection[str]) -> tuple[EvidenceSpan, ...]:
        return tuple(self.get_evidence(evidence_id) for evidence_id in sorted(evidence_ids))

    def quality_for(self, evidence_id: str) -> EvidenceQualityObservation:
        return self.get_quality(evidence_id)


@runtime_checkable
class SupportStrengthJudge(Protocol):
    async def observe(
        self,
        information_need: InformationNeed,
        evidence: EvidenceSpan,
        *,
        source: SourceDocument,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> SupportObservation: ...


class _SupportOutput(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    level: Literal["none", "weak", "moderate", "direct"]


def _hash_json(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


class StructuredSupportStrengthJudge(SupportStrengthJudge):
    def __init__(
        self,
        *,
        model_provider: ModelProvider,
        budget: BudgetAccountant,
        model_id: str,
        prompt_version: str,
    ) -> None:
        if not model_id.strip() or not prompt_version.strip():
            raise ValueError("model_id and prompt_version must not be empty")
        self.model_provider = model_provider
        self.budget = budget
        self.model_id = model_id
        self.prompt_version = prompt_version

    async def observe(
        self,
        information_need: InformationNeed,
        evidence: EvidenceSpan,
        *,
        source: SourceDocument,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> SupportObservation:
        request = ModelRequest(
            model_id=getattr(self.model_provider, "model_id", self.model_provider.provider_id),
            messages=(
                ModelMessage(
                    role="system",
                    content="Classify evidence support using exactly none, weak, moderate, or direct.",
                ),
                ModelMessage(
                    role="user",
                    content=json.dumps(
                        {
                            "information_need": information_need.text,
                            "evidence_excerpt": evidence.excerpt,
                            "source_type": source.source_type,
                        },
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                ),
            ),
            temperature=Decimal(0),
            seed=0,
            max_output_tokens=128,
            prompt_version=self.prompt_version,
            system_prompt_hash=_hash_json("support-system-v1"),
            tool_schema_hash=_hash_json([]),
            output_schema_hash=_hash_json(_SupportOutput.model_json_schema()),
        )
        reservation = self.budget.reserve(
            ResourceEstimate(tokens=128, wall_seconds=5.0, cost_usd=Decimal(0)),
            node="Ranker",
            idempotency_key="judge:" + hashlib.sha256(
                f"{information_need.need_id}:{evidence.evidence_id}".encode()
            ).hexdigest(),
        )
        try:
            result = await self.model_provider.structured(
                request,
                _SupportOutput,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            actual = result.usage
            if self.budget.snapshot().used_cost_usd is not None and actual.cost_usd is None:
                actual = actual.model_copy(update={"cost_usd": Decimal(0)})
            self.budget.settle(reservation, actual=actual)
        except BaseException:
            try:
                self.budget.release(reservation)
            except (RuntimeError, ValueError):
                pass
            raise
        level = result.output.level
        return SupportObservation(
            level=level,
            judge_model=self.model_id,
            prompt_version=self.prompt_version,
            decision_code=f"DIRECTNESS_{level.upper()}",
        )


def support_score(observation: SupportObservation) -> float:
    return SUPPORT_SCORES[observation.level]


def source_quality(
    source: SourceDocument,
    quality: EvidenceQualityObservation,
) -> float:
    return _clip(
        0.40 * SOURCE_TYPE_SCORES[source.source_type]
        + 0.20 * quality.provenance_completeness
        + 0.20 * quality.directness
        + 0.20 * quality.data_verifiability,
        field="source_quality",
    )


def coverage_gain(
    subquestion: SubQuestion,
    information_need: InformationNeed,
    evidence: EvidenceSpan,
    coverage_score: float,
) -> float:
    if information_need.need_id not in evidence.information_need_ids:
        return 0.0
    total_importance = sum(need.importance for need in subquestion.information_needs)
    if total_importance <= 0.0:
        return 0.0
    return _clip(
        (1.0 - _clip(coverage_score, field="coverage_score"))
        * information_need.importance
        / total_importance,
        field="coverage_gain",
    )


def freshness_score(
    requirement: FreshnessRequirement | None,
    source: SourceDocument,
    evaluation_time: datetime,
) -> float:
    if requirement is None or requirement.kind == "none":
        return 1.0
    if requirement.kind == "published_after":
        if source.published_at is None:
            return 0.5
        assert requirement.published_after is not None
        return float(source.published_at.date() >= requirement.published_after)
    assert requirement.retrieved_within_days is not None
    age_days = (evaluation_time - source.retrieved_at).total_seconds() / 86_400
    return float(age_days <= requirement.retrieved_within_days)


def risk_score(quality: EvidenceQualityObservation) -> float:
    if quality.is_snippet_only or not quality.has_stable_locator:
        return 1.0
    if quality.is_truncated or quality.parse_confidence < 0.80:
        return 0.5
    return 0.0


def independence_score(
    source_family_id: str,
    selected_source_family_ids: AbstractSet[str],
) -> float:
    return float(source_family_id not in selected_source_family_ids)


def normalized_cosine_score(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions must match and be non-empty")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not math.isfinite(left_norm) or not math.isfinite(right_norm):
        raise ValueError("embedding values must be finite")
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.5
    cosine = sum(
        (left_value / left_norm) * (right_value / right_norm)
        for left_value, right_value in zip(left, right, strict=True)
    )
    return _clip((cosine + 1.0) / 2.0, field="relevance")


class DefaultEvidenceFeatureCalculator(EvidenceFeatureCalculator):
    def __init__(
        self,
        *,
        embedder: TextEmbedder,
        embedding_model_id: str,
        materials: EvidenceFeatureMaterialResolver,
        support_judge: SupportStrengthJudge,
    ) -> None:
        self.embedder = embedder
        self.embedding_model_id = embedding_model_id
        self.materials = materials
        self.support_judge = support_judge

    async def compute(
        self,
        *,
        candidate: EvidenceCandidate,
        subquestion: SubQuestion,
        information_need: InformationNeed,
        subquestion_importance: float,
        need_importance: float,
        freshness_requirement: FreshnessRequirement | None,
        coverage_score: float,
        selected_evidence_ids: frozenset[str],
        selected_source_family_ids: frozenset[str],
        context_budget: int,
        evaluation_time: datetime,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> FeatureObservation:
        if subquestion_importance != subquestion.importance:
            raise ValueError("subquestion importance mismatch")
        if need_importance != information_need.importance:
            raise ValueError("information-need importance mismatch")
        if type(context_budget) is not int or context_budget <= 0:
            raise ValueError("context_budget must be positive")
        cancellation_token.raise_if_cancelled()
        selected = self.materials.evidence_for_ids(sorted(selected_evidence_ids))
        texts = [information_need.text, candidate.evidence.excerpt]
        texts.extend(item.excerpt for item in selected)
        vectors = await self.embedder.embed(
            texts,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        if len(vectors) != len(texts):
            raise ValueError("embedder returned an unexpected number of vectors")
        support = await self.support_judge.observe(
            information_need,
            candidate.evidence,
            source=candidate.source,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        quality = self.materials.quality_for(candidate.evidence.evidence_id)
        values = EvidenceFeatures(
            relevance=normalized_cosine_score(vectors[0], vectors[1]),
            support_strength=support_score(support),
            source_quality=source_quality(candidate.source, quality),
            coverage_gain=coverage_gain(
                subquestion,
                information_need,
                candidate.evidence,
                coverage_score,
            ),
            independence=independence_score(
                candidate.source_family_id,
                selected_source_family_ids,
            ),
            freshness=freshness_score(
                freshness_requirement,
                candidate.source,
                evaluation_time,
            ),
            redundancy=max(
                (
                    normalized_cosine_score(vectors[1], vector)
                    for vector in vectors[2:]
                ),
                default=0.0,
            ),
            risk=risk_score(quality),
        )
        provenance = {
            "relevance": f"embedding:{self.embedding_model_id}:need_excerpt",
            "support_strength": (
                f"judge:{support.judge_model}:{support.prompt_version}:{support.decision_code}"
            ),
            "source_quality": (
                f"source:{candidate.source.source_type}:parser:{candidate.source.parser_version}"
            ),
            "coverage_gain": (
                f"need:{information_need.need_id}:importance:{need_importance}:coverage:{coverage_score}"
            ),
            "independence": f"source_family:{candidate.source_family_id}",
            "freshness": (
                f"requirement:{freshness_requirement.kind if freshness_requirement else 'none'}:"
                f"published_after:{freshness_requirement.published_after if freshness_requirement else None}:"
                f"retrieved_within_days:{freshness_requirement.retrieved_within_days if freshness_requirement else None}:"
                f"evaluation_time:{evaluation_time.isoformat()}"
            ),
            "redundancy": f"embedding:{self.embedding_model_id}:max_selected",
            "risk": (
                f"parse_confidence:{quality.parse_confidence}:truncated:{quality.is_truncated}:"
                f"snippet:{quality.is_snippet_only}:stable_locator:{quality.has_stable_locator}"
            ),
        }
        return FeatureObservation(values=values, provenance=provenance)


__all__ = [
    "SOURCE_TYPE_SCORES",
    "SUPPORT_SCORES",
    "DefaultEvidenceFeatureCalculator",
    "EvidenceFeatureCalculator",
    "EvidenceFeatureMaterialResolver",
    "EvidenceFeatures",
    "EvidenceQualityObservation",
    "FeatureObservation",
    "StoreBackedFeatureMaterials",
    "StructuredSupportStrengthJudge",
    "SupportObservation",
    "SupportStrengthJudge",
    "coverage_gain",
    "freshness_score",
    "independence_score",
    "normalized_cosine_score",
    "risk_score",
    "source_quality",
    "support_score",
]
