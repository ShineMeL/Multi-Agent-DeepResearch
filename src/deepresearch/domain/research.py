from __future__ import annotations

from collections.abc import Iterable, Iterator
from datetime import date
from math import isfinite
from typing import Annotated, Literal, Never, Self, SupportsIndex, TypeVar

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

_Key = TypeVar("_Key")
_Value = TypeVar("_Value")
_Item = TypeVar("_Item")


class _FrozenDict(dict[_Key, _Value]):
    @staticmethod
    def _raise_immutable() -> Never:
        raise TypeError("domain mappings are immutable")

    def __setitem__(self, key: _Key, value: _Value) -> Never:
        self._raise_immutable()

    def __delitem__(self, key: _Key) -> Never:
        self._raise_immutable()

    def __ior__(self, value: object) -> Never:
        self._raise_immutable()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        memo[id(self)] = self
        return self

    def clear(self) -> Never:
        self._raise_immutable()

    def pop(self, key: _Key, default: object = None) -> Never:
        self._raise_immutable()

    def popitem(self) -> Never:
        self._raise_immutable()

    def setdefault(self, key: _Key, default: _Value | None = None) -> Never:
        self._raise_immutable()

    def update(self, *args: object, **kwargs: object) -> Never:
        self._raise_immutable()


class _FrozenList(list[_Item]):
    @staticmethod
    def _raise_immutable() -> Never:
        raise TypeError("domain lists are immutable")

    def __setitem__(
        self,
        index: SupportsIndex | slice[SupportsIndex | None, SupportsIndex | None, SupportsIndex | None],
        value: _Item | Iterable[_Item],
    ) -> Never:
        self._raise_immutable()

    def __delitem__(
        self,
        index: SupportsIndex | slice[SupportsIndex | None, SupportsIndex | None, SupportsIndex | None],
    ) -> Never:
        self._raise_immutable()

    def __iadd__(self, value: Iterable[_Item]) -> Never:
        self._raise_immutable()

    def __imul__(self, value: SupportsIndex) -> Never:
        self._raise_immutable()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        memo[id(self)] = self
        return self

    def append(self, value: _Item) -> Never:
        self._raise_immutable()

    def clear(self) -> Never:
        self._raise_immutable()

    def extend(self, values: Iterable[_Item]) -> Never:
        self._raise_immutable()

    def insert(self, index: SupportsIndex, value: _Item) -> Never:
        self._raise_immutable()

    def pop(self, index: SupportsIndex = -1) -> Never:
        self._raise_immutable()

    def remove(self, value: _Item) -> Never:
        self._raise_immutable()

    def reverse(self) -> Never:
        self._raise_immutable()

    def sort(self, *args: object, **kwargs: object) -> Never:
        self._raise_immutable()


class _CanonicalSourceTypes(frozenset[SourceType]):
    def __iter__(self) -> Iterator[SourceType]:
        return iter(sorted(super().__iter__()))


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

    @field_validator("output_requirements")
    @classmethod
    def freeze_output_requirements(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)

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

    @field_validator("allowed_source_types")
    @classmethod
    def canonicalize_source_types(
        cls, value: frozenset[SourceType]
    ) -> frozenset[SourceType]:
        return _CanonicalSourceTypes(value)


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


def _freeze_mapping[Key, Value](value: dict[Key, Value]) -> dict[Key, Value]:
    return _FrozenDict(value)


def _freeze_json_mapping(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _freeze_mapping({key: _freeze_json(item) for key, item in value.items()})


def _freeze_json(value: JsonValue) -> JsonValue:
    if isinstance(value, dict):
        return _freeze_json_mapping(value)
    if isinstance(value, list):
        return _FrozenList(_freeze_json(item) for item in value)
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
