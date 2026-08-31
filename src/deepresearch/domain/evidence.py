import re
from datetime import datetime
from math import isfinite
from typing import Annotated, Literal

from pydantic import AnyHttpUrl, ConfigDict, Field, field_serializer, field_validator

from .enums import ClaimType, SourceType, VerificationStatus
from .locators import Locator, _DomainModel  # pyright: ignore[reportPrivateUsage]
from .research import _freeze_mapping  # pyright: ignore[reportPrivateUsage]


def _require_sha256(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError("value must be a lowercase 64-character SHA-256")
    return value


def _require_timezone(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("datetime must be timezone-aware")
    return value


def _require_finite(value: float) -> float:
    if not isfinite(value):
        raise ValueError("score must be finite")
    return value


class CoverageLedgerEntry(_DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    subquestion_id: str
    coverage_score: Annotated[float, Field(ge=0.0, le=1.0)]
    independent_source_count: Annotated[int, Field(ge=0)]
    unresolved_conflict_ids: tuple[str, ...]
    uncertainty_score: Annotated[float, Field(ge=0.0, le=1.0)]
    last_marginal_gain: Annotated[float, Field(ge=0.0, le=1.0)]
    evidence_ids: tuple[str, ...]
    attempt_count: Annotated[int, Field(ge=0)]
    last_decision_code: str

    _finite_scores = field_validator(
        "coverage_score", "uncertainty_score", "last_marginal_gain"
    )(_require_finite)


class SourceDocument(_DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    canonical_url: AnyHttpUrl
    title: str
    authors: tuple[str, ...]
    published_at: datetime | None = None
    retrieved_at: datetime
    content_hash: str
    parsed_content_hash: str
    source_type: SourceType
    source_family_id: str
    parser_version: str

    _timezone_datetimes = field_validator("published_at", "retrieved_at")(_require_timezone)
    _sha256_hashes = field_validator("content_hash", "parsed_content_hash")(_require_sha256)


class EvidenceSpan(_DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    source_id: str
    locator: Locator
    excerpt: str
    excerpt_hash: str
    language: str
    information_need_ids: tuple[str, ...]

    _sha256_hash = field_validator("excerpt_hash")(_require_sha256)


class Claim(_DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    text: str
    claim_type: ClaimType
    entities: tuple[str, ...]
    numbers: tuple[str, ...]
    qualifiers: tuple[str, ...]
    report_section: str
    verification_status: VerificationStatus


class ClaimEvidenceLink(_DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    claim_id: str
    evidence_id: str
    relation: Literal["support", "contradict", "context", "insufficient"]
    entailment_score: Annotated[float, Field(ge=0.0, le=1.0)]
    relevance_score: Annotated[float, Field(ge=0.0, le=1.0)]
    judge_model: str
    prompt_version: str
    decision_code: str

    _finite_scores = field_validator("entailment_score", "relevance_score")(_require_finite)


class RerankScore(_DomainModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str
    total: Annotated[float, Field(ge=0.0, le=1.0)]
    feature_scores: dict[str, float]
    model_id: str | None = None
    prompt_version: str | None = None

    _finite_total = field_validator("total")(_require_finite)

    @field_serializer("feature_scores", when_used="json")
    def serialize_feature_scores(self, value: dict[str, float]) -> dict[str, float]:
        return {key: value[key] for key in sorted(value)}

    @field_validator("feature_scores")
    @classmethod
    def require_finite_feature_scores(cls, value: dict[str, float]) -> dict[str, float]:
        for score in value.values():
            _require_finite(score)
        return value

    @field_validator("feature_scores")
    @classmethod
    def freeze_feature_scores(cls, value: dict[str, float]) -> dict[str, float]:
        return _freeze_mapping(value)


__all__ = [
    "Claim",
    "ClaimEvidenceLink",
    "CoverageLedgerEntry",
    "EvidenceSpan",
    "RerankScore",
    "SourceDocument",
]
