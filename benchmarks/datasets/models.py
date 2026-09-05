from __future__ import annotations

import base64
import binascii
import hashlib
import math
import re
from collections.abc import Mapping
from datetime import date, datetime
from enum import StrEnum
from typing import Annotated, Literal, Self, cast

from pydantic import AnyHttpUrl, BaseModel, ConfigDict, Field, field_validator, model_validator

from deepresearch.domain import Claim, Locator, ResearchRequest, SourceType, StopReason

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _BenchmarkModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _sha256(value: str, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _nonblank(value: str, *, field: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{field} must be non-empty")
    return value


def _unique(values: list[str] | tuple[str, ...], *, label: str) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"duplicate {label}")


def _timezone_aware(value: datetime | None, *, field: str) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError(f"{field} must be timezone-aware")
    return value


class TaskCategory(StrEnum):
    TECHNICAL_SURVEY = "technical_survey"
    METHOD_COMPARISON = "method_comparison"
    MULTI_HOP_HISTORY = "multi_hop_history"
    FRESHNESS = "freshness"
    BILINGUAL = "bilingual"
    SOURCE_CONFLICT = "source_conflict"


class GoldInformationNeed(_BenchmarkModel):
    need_id: str
    text: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]
    acceptable_claim_ids: list[str]

    @field_validator("need_id", "text")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "value")
        return _nonblank(value, field=str(field))

    @field_validator("acceptable_claim_ids")
    @classmethod
    def validate_claim_ids(cls, value: list[str]) -> list[str]:
        if not value or any(type(item) is not str or not item.strip() for item in value):
            raise ValueError("acceptable_claim_ids must be non-empty")
        _unique(value, label="acceptable claim id")
        return value

    @field_validator("importance")
    @classmethod
    def validate_importance(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("importance must be finite")
        return value


class GoldEvidenceSpan(_BenchmarkModel):
    evidence_id: str
    source_id: str
    locator: Locator
    relevance_grade: Literal[0, 1, 2, 3]
    excerpt_hash: str

    @field_validator("evidence_id", "source_id")
    @classmethod
    def validate_ids(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "id")
        return _nonblank(value, field=str(field))

    @field_validator("excerpt_hash")
    @classmethod
    def validate_excerpt_hash(cls, value: str) -> str:
        return _sha256(value, field="excerpt_hash")


class GoldClaimEvidenceLink(_BenchmarkModel):
    evidence_id: str
    relation: Literal["support", "contradict", "context"]

    @field_validator("evidence_id")
    @classmethod
    def validate_evidence_id(cls, value: str) -> str:
        return _nonblank(value, field="evidence_id")


class GoldClaimLink(_BenchmarkModel):
    claim_id: str
    evidence_links: list[GoldClaimEvidenceLink]

    @field_validator("claim_id")
    @classmethod
    def validate_claim_id(cls, value: str) -> str:
        return _nonblank(value, field="claim_id")

    @model_validator(mode="after")
    def validate_evidence_links(self) -> Self:
        if not self.evidence_links:
            raise ValueError("evidence_links must not be empty")
        pairs = [(item.evidence_id, item.relation) for item in self.evidence_links]
        if len(pairs) != len(set(pairs)):
            raise ValueError("duplicate evidence link")
        return self


class RubricDimension(_BenchmarkModel):
    rubric_id: str
    description: str
    weight: Annotated[float, Field(gt=0.0, le=1.0)]
    levels: dict[Literal[0, 1, 2, 3], str]

    @field_validator("rubric_id", "description")
    @classmethod
    def validate_text(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "value")
        return _nonblank(value, field=str(field))

    @field_validator("weight")
    @classmethod
    def validate_weight(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("weight must be finite")
        return value

    @field_validator("levels", mode="before")
    @classmethod
    def normalize_level_keys(cls, value: object) -> object:
        if isinstance(value, Mapping):
            normalized: dict[object, object] = {}
            for key, text in cast("Mapping[object, object]", value).items():
                if isinstance(key, str) and key in {"0", "1", "2", "3"}:
                    normalized[int(key)] = text
                else:
                    normalized[key] = text
            return normalized
        return value

    @field_validator("levels")
    @classmethod
    def validate_levels(cls, value: dict[Literal[0, 1, 2, 3], str]) -> dict[Literal[0, 1, 2, 3], str]:
        if set(value) != {0, 1, 2, 3}:
            raise ValueError("rubric levels must contain 0, 1, 2 and 3")
        if any(type(text) is not str or not text.strip() for text in value.values()):
            raise ValueError("rubric levels must be non-empty")
        return value


class FrozenEvidenceRecord(_BenchmarkModel):
    task_id: str
    evidence_id: str
    source_id: str
    source_family_id: str
    canonical_url: AnyHttpUrl
    title: str
    authors: tuple[str, ...]
    media_type: str
    raw_body_b64: str
    content_hash: str
    normalized_text: str
    parsed_content_hash: str
    locator_text: str
    locator: Locator
    excerpt: str
    excerpt_hash: str
    published_at: datetime | None = None
    unknown_published_at_reason: str | None = None
    retrieved_at: datetime
    language: str
    source_type: SourceType

    @field_validator(
        "task_id",
        "evidence_id",
        "source_id",
        "source_family_id",
        "title",
        "media_type",
        "raw_body_b64",
        "normalized_text",
        "locator_text",
        "excerpt",
        "language",
    )
    @classmethod
    def validate_nonblank(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "value")
        return _nonblank(value, field=str(field))

    @field_validator("authors")
    @classmethod
    def validate_authors(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(type(author) is not str or not author.strip() for author in value):
            raise ValueError("authors must contain non-empty strings")
        return value

    @field_validator("content_hash", "parsed_content_hash", "excerpt_hash")
    @classmethod
    def validate_hashes(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "hash")
        return _sha256(value, field=str(field))

    @field_validator("published_at", "retrieved_at")
    @classmethod
    def validate_datetime(cls, value: datetime | None, info: object) -> datetime | None:
        field = getattr(info, "field_name", "datetime")
        return _timezone_aware(value, field=str(field))

    @field_validator("unknown_published_at_reason")
    @classmethod
    def validate_unknown_reason(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("unknown_published_at_reason must be non-empty")
        return value

    @model_validator(mode="after")
    def validate_material_and_publication(self) -> Self:
        try:
            raw_bytes = base64.b64decode(self.raw_body_b64, validate=True)
        except (binascii.Error, ValueError):
            raise ValueError("raw_body_b64 is not valid base64") from None
        if hashlib.sha256(raw_bytes).hexdigest() != self.content_hash:
            raise ValueError("content_hash does not match raw_body_b64")
        if hashlib.sha256(self.normalized_text.encode("utf-8")).hexdigest() != self.parsed_content_hash:
            raise ValueError("parsed_content_hash does not match normalized_text")

        if self.published_at is None:
            if self.unknown_published_at_reason is None:
                raise ValueError("exactly one publication date or unknown reason is required")
        elif self.unknown_published_at_reason is not None:
            raise ValueError("exactly one publication date or unknown reason is required")

        locator = self.locator
        if not 0 <= locator.start_char < locator.end_char <= len(self.locator_text):
            raise ValueError("locator range is invalid")
        if self.locator_text[locator.start_char : locator.end_char] != self.excerpt:
            raise ValueError("locator slice does not match excerpt")
        if hashlib.sha256(self.excerpt.encode("utf-8")).hexdigest() != self.excerpt_hash:
            raise ValueError("excerpt_hash does not match excerpt")
        return self


class AnnotatedQuestion(_BenchmarkModel):
    task_id: str
    split: Literal["dev", "test"]
    category: TaskCategory
    request: ResearchRequest
    evaluation_cutoff: date
    information_needs: list[GoldInformationNeed]
    acceptable_claims: list[Claim]
    candidate_source_ids: list[str]
    gold_source_family_ids: list[str]
    snapshot_id: str
    corpus_version: str
    index_version: str
    gold_evidence_spans: list[GoldEvidenceSpan]
    gold_claim_links: list[GoldClaimLink]
    rubric: dict[str, RubricDimension]
    expected_stop_reason: StopReason
    expected_is_partial: bool
    created_at: date
    annotation_version: str

    @field_validator(
        "task_id",
        "snapshot_id",
        "corpus_version",
        "index_version",
        "annotation_version",
    )
    @classmethod
    def validate_identity(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "identity")
        return _nonblank(value, field=str(field))

    @field_validator("candidate_source_ids", "gold_source_family_ids")
    @classmethod
    def validate_source_ids(cls, value: list[str], info: object) -> list[str]:
        field = getattr(info, "field_name", "source ids")
        if not value or any(type(item) is not str or not item.strip() for item in value):
            raise ValueError(f"{field} must be non-empty")
        _unique(value, label=str(field))
        return value

    @model_validator(mode="after")
    def validate_annotation_graph(self) -> Self:
        if not self.information_needs:
            raise ValueError("information_needs must not be empty")
        if not self.acceptable_claims:
            raise ValueError("acceptable_claims must not be empty")
        if not self.gold_evidence_spans:
            raise ValueError("gold_evidence_spans must not be empty")
        if not self.gold_claim_links:
            raise ValueError("gold_claim_links must not be empty")
        if not self.rubric:
            raise ValueError("rubric must not be empty")

        need_ids = [item.need_id for item in self.information_needs]
        claim_ids = [item.claim_id for item in self.acceptable_claims]
        evidence_ids = [item.evidence_id for item in self.gold_evidence_spans]
        _unique(need_ids, label="information need")
        _unique(claim_ids, label="claim")
        _unique(evidence_ids, label="evidence")
        _unique([item.claim_id for item in self.gold_claim_links], label="claim link")

        known_claims = set(claim_ids)
        known_evidence = set(evidence_ids)
        known_sources = set(self.candidate_source_ids)
        for need in self.information_needs:
            unknown = set(need.acceptable_claim_ids) - known_claims
            if unknown:
                raise ValueError(f"unknown claim_id: {min(unknown)}")
        for span in self.gold_evidence_spans:
            if span.source_id not in known_sources:
                raise ValueError(f"source ID absent from candidate_source_ids: {span.source_id}")
        for claim_link in self.gold_claim_links:
            if claim_link.claim_id not in known_claims:
                raise ValueError(f"unknown claim_id: {claim_link.claim_id}")
            for evidence_link in claim_link.evidence_links:
                if evidence_link.evidence_id not in known_evidence:
                    raise ValueError(f"unknown evidence_id: {evidence_link.evidence_id}")

        for key, dimension in self.rubric.items():
            if key != dimension.rubric_id:
                raise ValueError(f"rubric key does not match rubric_id: {key}")
        weight_sum = math.fsum(item.weight for item in self.rubric.values())
        if not math.isclose(weight_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
            raise ValueError("rubric weights must sum to 1.0")
        if self.expected_stop_reason == "SUFFICIENT" and self.expected_is_partial:
            raise ValueError("SUFFICIENT cannot be partial")
        if self.expected_stop_reason == "BUDGET_EXHAUSTED" and not self.expected_is_partial:
            raise ValueError("BUDGET_EXHAUSTED must be partial")
        if self.expected_stop_reason in {"PLATEAU", "BLOCKED"} and not self.expected_is_partial:
            raise ValueError(f"{self.expected_stop_reason} must be partial")
        return self


class RuntimeTask(_BenchmarkModel):
    task_id: str
    category: TaskCategory
    request: ResearchRequest
    evaluation_cutoff: date
    snapshot_id: str
    corpus_version: str
    index_version: str


class DatasetManifest(_BenchmarkModel):
    dataset_id: str
    version: str
    record_count: Annotated[int, Field(ge=0)]
    split_counts: dict[Literal["dev", "test"], Annotated[int, Field(ge=0)]]
    category_counts: dict[TaskCategory, Annotated[int, Field(ge=0)]]
    public_runtime_files: list[str]
    private_manifest_sha256: str
    snapshot_collection_sha256: str
    cost_subset_sha256: str
    created_at: datetime

    @field_validator("dataset_id", "version")
    @classmethod
    def validate_identity(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "identity")
        return _nonblank(value, field=str(field))

    @field_validator("private_manifest_sha256", "snapshot_collection_sha256", "cost_subset_sha256")
    @classmethod
    def validate_hash(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "hash")
        return _sha256(value, field=str(field))

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, field="created_at")  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_counts(self) -> Self:
        if sum(self.split_counts.values()) != self.record_count:
            raise ValueError("split_counts must sum to record_count")
        if sum(self.category_counts.values()) != self.record_count:
            raise ValueError("category_counts must sum to record_count")
        if any(type(path) is not str or not path.strip() for path in self.public_runtime_files):
            raise ValueError("public_runtime_files must contain non-empty paths")
        return self


class PrivateDatasetManifest(_BenchmarkModel):
    dataset_id: str
    version: str
    record_count: Annotated[int, Field(ge=0)]
    split_counts: dict[Literal["dev", "test"], Annotated[int, Field(ge=0)]]
    category_counts: dict[TaskCategory, Annotated[int, Field(ge=0)]]
    batch_sha256: dict[TaskCategory, str]
    snapshot_manifest_sha256: dict[str, str]
    public_runtime_files: list[str]
    private_test_runtime_files: list[str]
    main_test_task_ids: tuple[str, ...]
    stability_task_ids: tuple[str, ...]
    cost_subset_task_ids: tuple[str, ...]
    p0_task_ids: tuple[str, ...]
    oracle_task_ids: tuple[str, ...]
    subset_seed: int
    created_at: datetime

    @field_validator("dataset_id", "version")
    @classmethod
    def validate_identity(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "identity")
        return _nonblank(value, field=str(field))

    @field_validator("batch_sha256")
    @classmethod
    def validate_batch_hashes(cls, value: dict[TaskCategory, str]) -> dict[TaskCategory, str]:
        if not value:
            raise ValueError("batch_sha256 must not be empty")
        for category, digest in value.items():
            _sha256(digest, field=f"batch_sha256[{category}]")
        return value

    @field_validator("snapshot_manifest_sha256")
    @classmethod
    def validate_snapshot_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        for task_id, digest in value.items():
            _nonblank(task_id, field="snapshot task id")
            _sha256(digest, field=f"snapshot_manifest_sha256[{task_id}]")
        return value

    @field_validator("created_at")
    @classmethod
    def validate_created_at(cls, value: datetime) -> datetime:
        return _timezone_aware(value, field="created_at")  # type: ignore[return-value]

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if sum(self.split_counts.values()) != self.record_count:
            raise ValueError("split_counts must sum to record_count")
        if sum(self.category_counts.values()) != self.record_count:
            raise ValueError("category_counts must sum to record_count")
        for label, ids in (
            ("main_test_task_ids", self.main_test_task_ids),
            ("stability_task_ids", self.stability_task_ids),
            ("cost_subset_task_ids", self.cost_subset_task_ids),
            ("p0_task_ids", self.p0_task_ids),
            ("oracle_task_ids", self.oracle_task_ids),
        ):
            if any(type(task_id) is not str or not task_id.strip() for task_id in ids):
                raise ValueError(f"{label} must contain non-empty task IDs")
            _unique(list(ids), label=label)
        return self


__all__ = [
    "AnnotatedQuestion",
    "DatasetManifest",
    "FrozenEvidenceRecord",
    "GoldClaimEvidenceLink",
    "GoldClaimLink",
    "GoldEvidenceSpan",
    "GoldInformationNeed",
    "PrivateDatasetManifest",
    "RubricDimension",
    "RuntimeTask",
    "TaskCategory",
]
