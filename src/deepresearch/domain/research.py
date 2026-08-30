from datetime import date
from math import isfinite
from typing import Annotated, Literal, Self

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from .enums import AccessProfile, ExecutionMode, RunPurpose, SourceType


class FreshnessRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: Literal["none", "published_after", "retrieved_within_days"]
    published_after: date | None = None
    retrieved_within_days: Annotated[int | None, Field(ge=1)] = None

    @model_validator(mode="after")
    def validate_payload(self) -> Self:
        expected = {
            "none": (False, False),
            "published_after": (True, False),
            "retrieved_within_days": (False, True),
        }[self.kind]
        actual = (
            self.published_after is not None,
            self.retrieved_within_days is not None,
        )
        if actual != expected:
            raise ValueError("freshness payload does not match kind")
        return self


class ResearchRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    question: str
    output_requirements: dict[str, JsonValue]
    report_language: str
    source_languages: tuple[str, ...]
    freshness_requirement: FreshnessRequirement
    execution_mode: ExecutionMode
    access_profile: AccessProfile
    provider_profile_id: str
    run_purpose: RunPurpose
    budget_preset: Literal["low", "medium", "high"]

    @field_serializer("output_requirements", when_used="json")
    def serialize_output_requirements(
        self, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return {key: _canonical_json(value[key]) for key in sorted(value)}


class DateRange(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    start: date | None = None
    end: date | None = None

    @model_validator(mode="after")
    def validate_range(self) -> Self:
        if self.start is not None and self.end is not None and self.start > self.end:
            raise ValueError("date range start must not exceed end")
        return self


class ResearchScope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    included_topics: tuple[str, ...]
    excluded_topics: tuple[str, ...]
    date_range: DateRange | None = None
    answer_shape: str


class InformationNeed(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    need_id: str
    text: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]

    @field_validator("importance")
    @classmethod
    def require_finite_importance(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("importance must be finite")
        return value


class EvidenceRequirements(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    min_independent_sources: Annotated[int, Field(ge=1)]
    allowed_source_types: frozenset[SourceType]
    must_include_primary: bool
    freshness: FreshnessRequirement | None = None

    @field_serializer("allowed_source_types", when_used="json")
    def serialize_allowed_source_types(self, value: frozenset[SourceType]) -> list[SourceType]:
        return sorted(value)


class SubQuestion(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str
    question: str
    rationale_code: str
    importance: Annotated[float, Field(ge=0.0, le=1.0)]
    dependencies: tuple[str, ...]
    information_needs: tuple[InformationNeed, ...]
    evidence_requirements: EvidenceRequirements
    status: Literal["pending", "active", "covered", "blocked"]

    @field_validator("importance")
    @classmethod
    def require_finite_importance(cls, value: float) -> float:
        if not isfinite(value):
            raise ValueError("importance must be finite")
        return value


class ResearchPlan(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    plan_id: str
    scope: ResearchScope
    subquestions: tuple[SubQuestion, ...]
    created_by_model: str
    prompt_version: str

    @model_validator(mode="after")
    def validate_graph(self) -> Self:
        subquestion_ids = [item.id for item in self.subquestions]
        need_ids = [
            need.need_id
            for subquestion in self.subquestions
            for need in subquestion.information_needs
        ]
        _require_unique(subquestion_ids, label="subquestion")
        _require_unique(need_ids, label="information need")

        known_ids = set(subquestion_ids)
        for subquestion in self.subquestions:
            unknown = set(subquestion.dependencies) - known_ids
            if unknown:
                names = ", ".join(sorted(unknown))
                raise ValueError(f"unknown dependencies for {subquestion.id}: {names}")

        dependencies = {item.id: item.dependencies for item in self.subquestions}
        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(subquestion_id: str) -> None:
            if subquestion_id in visiting:
                raise ValueError("cycle detected in subquestion dependencies")
            if subquestion_id in visited:
                return
            visiting.add(subquestion_id)
            for dependency_id in dependencies[subquestion_id]:
                visit(dependency_id)
            visiting.remove(subquestion_id)
            visited.add(subquestion_id)

        for subquestion_id in subquestion_ids:
            visit(subquestion_id)
        return self


def _require_unique(values: list[str], *, label: str) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for value in values:
        if value in seen:
            duplicates.add(value)
        seen.add(value)
    if duplicates:
        names = ", ".join(sorted(duplicates))
        raise ValueError(f"duplicate {label} IDs: {names}")


def _canonical_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return {key: _canonical_json(value[key]) for key in sorted(value)}
    if isinstance(value, list):
        return [_canonical_json(item) for item in value]
    return value


__all__ = [
    "DateRange",
    "EvidenceRequirements",
    "FreshnessRequirement",
    "InformationNeed",
    "ResearchPlan",
    "ResearchRequest",
    "ResearchScope",
    "SubQuestion",
]
