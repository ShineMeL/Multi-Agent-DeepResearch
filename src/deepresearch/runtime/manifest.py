from __future__ import annotations

import hashlib
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping
from datetime import datetime
from decimal import ROUND_HALF_EVEN, Decimal
from itertools import pairwise
from typing import Annotated, Any, Literal, Never, Self, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from deepresearch.domain import (
    ExecutionMode,
    ResourceUsage,
    RunBudget,
    RunStatus,
    StopReason,
)
from deepresearch.retrieval import canonicalize_url, normalize_text
from deepresearch.storage import (
    EmbedCacheKey,
    FetchCacheKey,
    ModelCacheKey,
    ParseCacheKey,
    SearchCacheKey,
)

_SHA256 = re.compile(r"[0-9a-f]{64}")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:/-]{0,255}")
_ARTIFACT_ID = re.compile(r"sha256:[0-9a-f]{64}")
_MIME_TYPE = re.compile(r"[a-z0-9!#$&^_.+-]+/[a-z0-9!#$&^_.+-]+")


class _FrozenDict(dict[str, Any]):
    @staticmethod
    def _immutable() -> Never:
        raise TypeError("manifest mappings are immutable")

    def __setitem__(self, key: str, value: Any) -> Never:
        self._immutable()

    def __delitem__(self, key: str) -> Never:
        self._immutable()

    def clear(self) -> Never:
        self._immutable()

    def pop(self, key: str, default: object = None) -> Never:
        self._immutable()

    def popitem(self) -> Never:
        self._immutable()

    def setdefault(self, key: str, default: Any = None) -> Never:
        self._immutable()

    def update(self, *args: object, **kwargs: object) -> Never:
        self._immutable()

    def __ior__(self, other: object) -> Never:
        self._immutable()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        memo[id(self)] = self
        return self


def _thaw(value: object) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return {key: _thaw(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw(item) for item in cast("list[object] | tuple[object, ...]", value)]
    return value


def _freeze(value: object) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return _FrozenDict({key: _freeze(item) for key, item in mapping.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in cast("list[object]", value))
    return value


def _require_sha256(value: str) -> str:
    if _SHA256.fullmatch(value) is None:
        raise ValueError("value must be a lowercase 64-character SHA-256")
    return value


def _require_optional_sha256(value: str | None) -> str | None:
    return None if value is None else _require_sha256(value)


def _require_identifier(value: str) -> str:
    if _IDENTIFIER.fullmatch(value) is None:
        raise ValueError("value must be a non-empty stable identifier")
    return value


def _require_aware(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("datetime must be timezone-aware")
    return value


def _revalidate_usage(value: object) -> ResourceUsage:
    payload = value.model_dump(round_trip=True) if isinstance(value, ResourceUsage) else value
    usage = ResourceUsage.model_validate(payload)
    if not math.isfinite(usage.wall_seconds):
        raise ValueError("usage wall_seconds must be finite")
    if usage.cost_usd is not None and not usage.cost_usd.is_finite():
        raise ValueError("usage cost_usd must be finite")
    return usage


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        _thaw(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


class _ManifestModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @override
    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(round_trip=True)
        values.update(update)
        return type(self).model_validate(values)


class PricingSnapshot(_ManifestModel):
    snapshot_id: str
    provider_id: str
    endpoint_type: str
    model_id: str
    effective_at: datetime
    currency: Literal["USD"]
    input_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]
    output_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]
    cached_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]
    reasoning_tokens_per_million_usd: Annotated[Decimal, Field(ge=0)]

    _ids = field_validator("snapshot_id", "provider_id", "endpoint_type", "model_id")(
        _require_identifier
    )
    _aware_effective_at = field_validator("effective_at")(_require_aware)


class CostBreakdown(_ManifestModel):
    pricing_snapshot_id: str
    input_usd: Annotated[Decimal, Field(ge=0)]
    cached_input_usd: Annotated[Decimal, Field(ge=0)]
    output_usd: Annotated[Decimal, Field(ge=0)]
    reasoning_usd: Annotated[Decimal, Field(ge=0)]
    total_usd: Annotated[Decimal, Field(ge=0)]


class CostCalculator:
    QUANTUM = Decimal("0.000000001")

    @staticmethod
    def estimate(usage: ResourceUsage, pricing: PricingSnapshot) -> CostBreakdown:
        validated_usage = _revalidate_usage(usage)
        unit = Decimal(1_000_000)
        billable_input = max(validated_usage.input_tokens - validated_usage.cached_tokens, 0)
        input_usd = (
            Decimal(billable_input) * pricing.input_tokens_per_million_usd / unit
        )
        cached_usd = (
            Decimal(validated_usage.cached_tokens)
            * pricing.cached_tokens_per_million_usd
            / unit
        )
        output_usd = (
            Decimal(validated_usage.output_tokens)
            * pricing.output_tokens_per_million_usd
            / unit
        )
        reasoning_usd = (
            Decimal(validated_usage.reasoning_tokens)
            * pricing.reasoning_tokens_per_million_usd
            / unit
        )

        def quantize(value: Decimal) -> Decimal:
            return value.quantize(CostCalculator.QUANTUM, rounding=ROUND_HALF_EVEN)

        return CostBreakdown(
            pricing_snapshot_id=pricing.snapshot_id,
            input_usd=quantize(input_usd),
            cached_input_usd=quantize(cached_usd),
            output_usd=quantize(output_usd),
            reasoning_usd=quantize(reasoning_usd),
            total_usd=quantize(input_usd + cached_usd + output_usd + reasoning_usd),
        )


class ProviderProfileRecord(_ManifestModel):
    profile_id: str
    execution_mode: ExecutionMode
    provider_ids: tuple[str, ...]
    configuration_sha256: str

    _profile_id = field_validator("profile_id")(_require_identifier)
    _configuration_hash = field_validator("configuration_sha256")(_require_sha256)

    @field_validator("provider_ids")
    @classmethod
    def validate_provider_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or len(set(value)) != len(value):
            raise ValueError("provider_ids must be non-empty and unique")
        return tuple(_require_identifier(item) for item in value)


class ProviderCallRecord(_ManifestModel):
    operation: Literal["model", "search", "fetch", "parse", "embed"]
    node: str
    provider_id: str
    endpoint_type: str
    model_id: str | None = None
    model_revision: str | None = None
    request_sha256: str
    snapshot_id: str | None = None
    normalized_query: str | None = None
    locale: str | None = None
    complete_parameters: dict[str, JsonValue] = Field(default_factory=dict)
    time_policy: str | None = None
    prompt_version: str | None = None
    system_prompt_hash: str | None = None
    tool_schema_hash: str | None = None
    output_schema_hash: str | None = None
    temperature: Decimal | None = None
    seed: int | None = None
    started_at: datetime
    finished_at: datetime
    latency_ms: Annotated[int, Field(ge=0)]
    attempt: Annotated[int, Field(ge=1)]
    cache_hit: bool
    outcome_code: str
    usage: ResourceUsage
    pricing_snapshot_id: str | None = None
    estimated_cost_usd: Annotated[Decimal | None, Field(ge=0)] = None

    _ids = field_validator("node", "provider_id", "endpoint_type")(_require_identifier)
    _request_hash = field_validator("request_sha256")(_require_sha256)
    _optional_hashes = field_validator(
        "system_prompt_hash", "tool_schema_hash", "output_schema_hash"
    )(_require_optional_sha256)
    _aware_times = field_validator("started_at", "finished_at")(_require_aware)
    _usage = field_validator("usage", mode="before")(_revalidate_usage)

    @field_validator("complete_parameters", mode="before")
    @classmethod
    def thaw_parameters(cls, value: object) -> object:
        return _thaw(value)

    @field_validator("complete_parameters")
    @classmethod
    def freeze_parameters(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", _freeze(value))

    @field_serializer("complete_parameters")
    def serialize_parameters(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", _thaw(value))

    @model_validator(mode="after")
    def validate_operation_contract(self) -> ProviderCallRecord:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        measured_ms = round((self.finished_at - self.started_at).total_seconds() * 1000)
        if self.latency_ms != measured_ms:
            raise ValueError("latency_ms must match the recorded elapsed interval")
        if not self.outcome_code:
            raise ValueError("outcome_code must not be empty")

        def exact(names: set[str]) -> None:
            if set(self.complete_parameters) != names:
                raise ValueError(
                    f"{self.operation} complete_parameters must contain exactly {sorted(names)}"
                )

        def forbid(*values: object) -> None:
            if any(value is not None for value in values):
                raise ValueError(f"{self.operation} call contains operation-irrelevant fields")

        if self.operation == "model":
            values = (
                self.model_id,
                self.model_revision,
                self.prompt_version,
                self.system_prompt_hash,
                self.tool_schema_hash,
                self.output_schema_hash,
            )
            if any(value is None or value == "" for value in values):
                raise ValueError("model calls require model, revision, prompt and schema fields")
            if self.temperature is None:
                raise ValueError("model calls require temperature")
            exact({"seed_supported"})
            if not isinstance(self.complete_parameters.get("seed_supported"), bool):
                raise ValueError("model calls require explicit seed_supported")
            if self.seed is not None and self.complete_parameters["seed_supported"] is not True:
                raise ValueError("model seed requires seed support")
            forbid(self.snapshot_id, self.normalized_query, self.locale, self.time_policy)
            ModelCacheKey(
                provider_id=self.provider_id,
                endpoint_type=self.endpoint_type,
                model_id=cast("str", self.model_id),
                prompt_version=cast("str", self.prompt_version),
                system_prompt_hash=cast("str", self.system_prompt_hash),
                tool_schema_hash=cast("str", self.tool_schema_hash),
                output_schema_hash=cast("str", self.output_schema_hash),
                temperature=self.temperature,
                seed=self.seed,
                canonical_request_hash=self.request_sha256,
            )
            return self
        if self.operation == "search":
            if (
                not self.snapshot_id
                or not self.normalized_query
                or not self.locale
                or not self.time_policy
                or not self.complete_parameters
            ):
                raise ValueError("search calls require snapshot/query/locale/parameters/time policy")
            exact({"filters", "limit"})
            limit = self.complete_parameters["limit"]
            filters = self.complete_parameters["filters"]
            if type(limit) is not int or limit <= 0:
                raise ValueError("search limit must be a positive integer")
            if filters is not None and not isinstance(filters, dict):
                raise ValueError("search filters must be a mapping or null")
            forbid(
                self.model_id, self.model_revision, self.prompt_version,
                self.system_prompt_hash, self.tool_schema_hash, self.output_schema_hash,
                self.temperature, self.seed,
            )
            if normalize_text(self.normalized_query) != self.normalized_query:
                raise ValueError("normalized_query is not canonical")
            SearchCacheKey(
                snapshot_id=self.snapshot_id,
                normalized_query=self.normalized_query,
                provider_id=self.provider_id,
                endpoint_type=self.endpoint_type,
                locale=self.locale,
                complete_parameters=self.complete_parameters,
                time_policy=self.time_policy,
            )
        elif self.operation == "fetch":
            if not self.snapshot_id:
                raise ValueError("fetch calls require snapshot_id")
            exact({"canonical_url", "fetch_policy", "accepted_content_types"})
            forbid(
                self.model_id, self.model_revision, self.normalized_query, self.locale,
                self.time_policy, self.prompt_version, self.system_prompt_hash,
                self.tool_schema_hash, self.output_schema_hash, self.temperature, self.seed,
            )
            url = self.complete_parameters["canonical_url"]
            accepted = self.complete_parameters["accepted_content_types"]
            if not isinstance(url, str) or canonicalize_url(url) != url:
                raise ValueError("fetch canonical_url is not canonical")
            if not isinstance(accepted, (list, tuple)) or any(
                not isinstance(item, str) or _MIME_TYPE.fullmatch(item) is None
                for item in accepted
            ):
                raise ValueError("accepted_content_types must be MIME type strings")
            FetchCacheKey.model_validate(
                {
                    "snapshot_id": self.snapshot_id,
                    "canonical_url": url,
                    "fetch_policy": self.complete_parameters["fetch_policy"],
                    "accepted_content_types": accepted,
                }
            )
        elif self.operation == "parse":
            if not self.snapshot_id:
                raise ValueError("parse calls require snapshot_id")
            names = {
                "raw_content_hash",
                "parser_id",
                "parser_version",
                "normalization_version",
            }
            exact(names)
            forbid(
                self.model_id, self.model_revision, self.normalized_query, self.locale,
                self.time_policy, self.prompt_version, self.system_prompt_hash,
                self.tool_schema_hash, self.output_schema_hash, self.temperature, self.seed,
            )
            for name in ("parser_id", "parser_version", "normalization_version"):
                value = self.complete_parameters[name]
                if not isinstance(value, str):
                    raise TypeError(f"{name} must be a stable identifier")
                _require_identifier(value)
            ParseCacheKey.model_validate(
                {"snapshot_id": self.snapshot_id, **dict(self.complete_parameters)}
            )
        else:
            if self.model_id is None or self.model_revision is None:
                raise ValueError("embed calls require model_id and model_revision")
            names = {
                "snapshot_sha256",
                "normalize_embeddings",
                "canonical_texts_hash",
            }
            exact(names)
            forbid(
                self.snapshot_id, self.normalized_query, self.locale, self.time_policy,
                self.prompt_version, self.system_prompt_hash, self.tool_schema_hash,
                self.output_schema_hash, self.temperature, self.seed,
            )
            if type(self.complete_parameters["normalize_embeddings"]) is not bool:
                raise ValueError("normalize_embeddings must be a boolean")
            EmbedCacheKey.model_validate(
                {
                    "model_id": self.model_id,
                    "model_revision": self.model_revision,
                    **dict(self.complete_parameters),
                }
            )
        for name in ("raw_content_hash", "snapshot_sha256", "canonical_texts_hash"):
            value = self.complete_parameters.get(name)
            if value is not None:
                if not isinstance(value, str):
                    raise ValueError(f"{name} must be a SHA-256 string")
                _require_sha256(value)
        if self.estimated_cost_usd is not None or self.pricing_snapshot_id is not None:
            raise ValueError("non-model calls must not have pricing metadata")
        return self


class NodeExecutionRecord(_ManifestModel):
    node: str
    attempt: Annotated[int, Field(ge=1)]
    started_at: datetime
    finished_at: datetime
    latency_ms: Annotated[int, Field(ge=0)]
    status: RunStatus
    input_artifact_ids: tuple[str, ...]
    output_artifact_ids: tuple[str, ...]
    usage: ResourceUsage
    error_code: str | None = None

    _node_id = field_validator("node")(_require_identifier)
    _aware_times = field_validator("started_at", "finished_at")(_require_aware)
    _usage = field_validator("usage", mode="before")(_revalidate_usage)

    @field_validator("input_artifact_ids", "output_artifact_ids")
    @classmethod
    def validate_artifact_references(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("node artifact references must be unique")
        if any(_ARTIFACT_ID.fullmatch(item) is None for item in value):
            raise ValueError("node artifact references must be content-addressed SHA-256 IDs")
        return value

    @model_validator(mode="after")
    def validate_timing_and_status(self) -> NodeExecutionRecord:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        measured_ms = round((self.finished_at - self.started_at).total_seconds() * 1000)
        if self.latency_ms != measured_ms:
            raise ValueError("latency_ms must match the recorded elapsed interval")
        if self.status == "failed" and not self.error_code:
            raise ValueError("failed node executions require error_code")
        return self


class ParsedArtifactRecord(_ManifestModel):
    source_id: str
    raw_content_hash: str
    parsed_content_hash: str
    parser_id: str
    parser_version: str
    normalization_version: str
    artifact_id: str

    _ids = field_validator(
        "source_id", "parser_id", "parser_version", "normalization_version"
    )(_require_identifier)
    _hashes = field_validator("raw_content_hash", "parsed_content_hash")(_require_sha256)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if _ARTIFACT_ID.fullmatch(value) is None:
            raise ValueError("artifact_id must be a content-addressed SHA-256 ID")
        return value


class EvidenceHashRecord(_ManifestModel):
    evidence_id: str
    source_id: str
    locator_sha256: str
    excerpt_hash: str
    artifact_id: str

    _ids = field_validator("evidence_id", "source_id")(_require_identifier)
    _hashes = field_validator("locator_sha256", "excerpt_hash")(_require_sha256)

    @field_validator("artifact_id")
    @classmethod
    def validate_artifact_id(cls, value: str) -> str:
        if _ARTIFACT_ID.fullmatch(value) is None:
            raise ValueError("artifact_id must be a content-addressed SHA-256 ID")
        return value


class RunManifest(_ManifestModel):
    schema_version: Literal["run-manifest-v1"]
    run_id: str
    thread_id: str
    code_commit: str
    dependency_lock_sha256: str
    request_sha256: str
    config_sha256: str
    workflow_id: Literal["baseline-v1", "research-v1"]
    graph_version: str
    planner_id: Literal["P0", "P1", "P2"]
    provider_profiles: tuple[ProviderProfileRecord, ...]
    model_ids: tuple[str, ...]
    prompt_versions: dict[str, str]
    parser_versions: dict[str, str]
    ranker_id: Literal["R0", "R1", "R2"]
    ranker_weights_version: str | None
    budget: RunBudget
    usage: ResourceUsage
    usage_by_node: dict[str, ResourceUsage]
    pricing_status: Literal["estimated", "unknown"]
    pricing_snapshots: tuple[PricingSnapshot, ...]
    provider_calls: tuple[ProviderCallRecord, ...]
    node_executions: tuple[NodeExecutionRecord, ...]
    parsed_artifacts: tuple[ParsedArtifactRecord, ...]
    evidence_hashes: tuple[EvidenceHashRecord, ...]
    source_snapshot_ids: tuple[str, ...]
    artifact_ids: tuple[str, ...]
    run_event_count: Annotated[int, Field(ge=0)]
    run_events_sha256: str
    seed: int | None = None
    seed_supported: bool
    cache_hit_count: Annotated[int, Field(ge=0)]
    stop_reason: StopReason | None = None
    is_partial: bool
    failure_codes: tuple[str, ...]
    replay_parent: str | None = None
    started_at: datetime
    finished_at: datetime
    manifest_sha256: str = ""

    _ids = field_validator("run_id", "thread_id", "graph_version")(_require_identifier)
    _hashes = field_validator(
        "dependency_lock_sha256", "request_sha256", "config_sha256", "run_events_sha256"
    )(_require_sha256)
    _aware_times = field_validator("started_at", "finished_at")(_require_aware)
    _usage = field_validator("usage", mode="before")(_revalidate_usage)

    @field_validator("code_commit")
    @classmethod
    def validate_code_commit(cls, value: str) -> str:
        if _GIT_COMMIT.fullmatch(value) is None:
            raise ValueError("code_commit must be a lowercase Git object hash")
        return value

    @field_validator("manifest_sha256")
    @classmethod
    def validate_manifest_hash_shape(cls, value: str) -> str:
        if value:
            _require_sha256(value)
        return value

    @field_validator("prompt_versions", "parser_versions", mode="before")
    @classmethod
    def thaw_string_mapping(cls, value: object) -> object:
        return _thaw(value)

    @field_validator("prompt_versions", "parser_versions")
    @classmethod
    def freeze_string_mapping(cls, value: dict[str, str]) -> dict[str, str]:
        if any(not key or not item for key, item in value.items()):
            raise ValueError("version mappings must not contain empty names or versions")
        return cast("dict[str, str]", _FrozenDict(dict(value)))

    @field_serializer("prompt_versions", "parser_versions")
    def serialize_string_mapping(self, value: dict[str, str]) -> dict[str, str]:
        return {key: value[key] for key in sorted(value)}

    @field_validator("usage_by_node", mode="before")
    @classmethod
    def revalidate_usage_mapping(cls, value: object) -> object:
        thawed = _thaw(value)
        if not isinstance(thawed, dict):
            return thawed
        mapping = cast("dict[str, object]", thawed)
        return {key: _revalidate_usage(item) for key, item in mapping.items()}

    @field_validator("usage_by_node")
    @classmethod
    def freeze_usage_mapping(
        cls, value: dict[str, ResourceUsage]
    ) -> dict[str, ResourceUsage]:
        for key in value:
            _require_identifier(key)
        return cast("dict[str, ResourceUsage]", _FrozenDict(dict(value)))

    @field_serializer("usage_by_node")
    def serialize_usage_mapping(
        self, value: dict[str, ResourceUsage]
    ) -> dict[str, ResourceUsage]:
        return {key: value[key] for key in sorted(value)}

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("artifact_ids must be unique")
        if any(_ARTIFACT_ID.fullmatch(item) is None for item in value):
            raise ValueError("artifact_ids must be content-addressed SHA-256 IDs")
        return value

    @field_validator("source_snapshot_ids", "model_ids")
    @classmethod
    def validate_unique_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("identifier lists must be unique")
        return tuple(_require_identifier(item) for item in value)

    @classmethod
    def create(cls, payload: Mapping[str, object]) -> RunManifest:
        """Construct a new manifest and intentionally compute its content hash."""
        values = dict(payload)
        values["manifest_sha256"] = ""
        return cls.model_validate(values, context={"rehash_manifest": True})

    @override
    def model_copy(
        self, *, update: Mapping[str, Any] | None = None, deep: bool = False
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(round_trip=True)
        values.update(update)
        values["manifest_sha256"] = ""
        return type(self).model_validate(values, context={"rehash_manifest": True})

    @model_validator(mode="after")
    def validate_manifest(self, info: ValidationInfo) -> RunManifest:
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must not precede started_at")
        if self.seed is not None and not self.seed_supported:
            raise ValueError("seed requires seed_supported")
        if self.cache_hit_count != sum(call.cache_hit for call in self.provider_calls):
            raise ValueError("cache_hit_count does not match provider calls")

        expected_models = tuple(
            dict.fromkeys(
                call.model_id
                for call in self.provider_calls
                if call.operation == "model" and call.model_id is not None
            )
        )
        if self.model_ids != expected_models:
            raise ValueError("model_ids must equal model call model IDs in first-use order")
        containing_executions = self._validate_execution_history()
        self._validate_contiguous_attempts(containing_executions)
        self._validate_usage_reconciliation(containing_executions)
        self._validate_pricing()
        self._validate_budget_limits()
        parsed_ids = {record.artifact_id for record in self.parsed_artifacts}
        evidence_ids = {record.artifact_id for record in self.evidence_hashes}
        node_artifact_ids = {
            artifact_id
            for execution in self.node_executions
            for artifact_id in (
                *execution.input_artifact_ids,
                *execution.output_artifact_ids,
            )
        }
        if not (parsed_ids | evidence_ids | node_artifact_ids).issubset(
            set(self.artifact_ids)
        ):
            raise ValueError("record artifact IDs must be present in artifact_ids")
        expected_hash = self.canonical_sha256()
        if self.manifest_sha256:
            if self.manifest_sha256 != expected_hash:
                raise ValueError("manifest_sha256 does not match canonical manifest content")
        elif info.context and info.context.get("rehash_manifest") is True:
            object.__setattr__(self, "manifest_sha256", expected_hash)
        else:
            raise ValueError("blank manifest_sha256 requires RunManifest.create")
        return self

    def _validate_execution_history(self) -> tuple[int, ...]:
        """Validate ordered executions and resolve calls using half-open starts.

        Execution histories are ordered, non-overlapping intervals. A call may end at
        an execution's closed finish, but its start must be strictly before that
        finish. Consequently, a zero-duration call on a touching boundary belongs to
        the later ``[started_at, finished_at)`` execution only.
        """
        executions_by_node: defaultdict[str, list[tuple[int, NodeExecutionRecord]]] = (
            defaultdict(list)
        )
        for index, execution in enumerate(self.node_executions):
            if not (
                self.started_at <= execution.started_at
                and execution.finished_at <= self.finished_at
            ):
                raise ValueError("node execution must be inside the run envelope")
            executions_by_node[execution.node].append((index, execution))

        for executions in executions_by_node.values():
            attempts = [execution.attempt for _, execution in executions]
            if attempts != list(range(1, len(attempts) + 1)):
                raise ValueError("node execution attempts must be ordered and contiguous")
            for (_, previous), (_, current) in pairwise(executions):
                if previous.finished_at > current.started_at:
                    raise ValueError(
                        "node execution attempts must be chronological and non-overlapping"
                    )

        containing_executions: list[int] = []
        for call in self.provider_calls:
            containing = [
                index
                for index, execution in enumerate(self.node_executions)
                if execution.node == call.node
                and execution.started_at <= call.started_at
                and call.started_at < execution.finished_at
                and call.finished_at <= execution.finished_at
            ]
            if len(containing) != 1:
                raise ValueError(
                    "provider call must have exactly one containing node execution"
                )
            containing_executions.append(containing[0])
        return tuple(containing_executions)

    @staticmethod
    def _provider_call_identity(call: ProviderCallRecord) -> bytes:
        """Return the canonical, result-affecting invocation/cache identity."""
        fields = {
            "operation",
            "provider_id",
            "endpoint_type",
            "model_id",
            "model_revision",
            "request_sha256",
            "snapshot_id",
            "normalized_query",
            "locale",
            "complete_parameters",
            "time_policy",
            "prompt_version",
            "system_prompt_hash",
            "tool_schema_hash",
            "output_schema_hash",
            "temperature",
            "seed",
        }
        return _canonical_bytes(call.model_dump(mode="json", include=fields))

    def _validate_contiguous_attempts(
        self, containing_executions: tuple[int, ...]
    ) -> None:
        call_attempts: defaultdict[tuple[int, bytes], list[int]] = defaultdict(list)
        for execution_index, call in zip(
            containing_executions, self.provider_calls, strict=True
        ):
            call_attempts[
                (execution_index, self._provider_call_identity(call))
            ].append(call.attempt)
        for attempts in call_attempts.values():
            if attempts != list(range(1, len(attempts) + 1)):
                raise ValueError("provider call attempts must be ordered and contiguous")

    @staticmethod
    def _aggregate_usage(
        usages: tuple[ResourceUsage, ...], *, wall_seconds: float, cost_usd: Decimal | None
    ) -> ResourceUsage:
        return ResourceUsage(
            input_tokens=sum(item.input_tokens for item in usages),
            output_tokens=sum(item.output_tokens for item in usages),
            reasoning_tokens=sum(item.reasoning_tokens for item in usages),
            cached_tokens=sum(item.cached_tokens for item in usages),
            total_tokens=sum(item.total_tokens for item in usages),
            search_calls=sum(item.search_calls for item in usages),
            pages=sum(item.pages for item in usages),
            retries=sum(item.retries for item in usages),
            wall_seconds=wall_seconds,
            cost_usd=cost_usd,
        )

    def _charged_cost(self, calls: tuple[ProviderCallRecord, ...]) -> Decimal | None:
        if self.pricing_status == "unknown":
            return None
        charged = Decimal(0)
        for call in calls:
            if call.operation != "model" or call.cache_hit:
                continue
            if call.estimated_cost_usd is None:
                raise ValueError("estimated model calls require an estimated cost")
            charged += call.estimated_cost_usd
        return charged.quantize(CostCalculator.QUANTUM, rounding=ROUND_HALF_EVEN)

    def _validate_usage_reconciliation(
        self, containing_executions: tuple[int, ...]
    ) -> None:
        """Reconcile calls into executions, nodes, then the run without summing wall time."""
        additive = (
            "input_tokens", "output_tokens", "reasoning_tokens", "cached_tokens",
            "total_tokens", "search_calls", "pages", "retries",
        )
        calls_by_execution: defaultdict[int, list[ProviderCallRecord]] = defaultdict(list)
        for execution_index, call in zip(
            containing_executions, self.provider_calls, strict=True
        ):
            calls_by_execution[execution_index].append(call)
        for index, calls in calls_by_execution.items():
            execution = self.node_executions[index]
            for field in additive:
                if sum(getattr(call.usage, field) for call in calls) > getattr(execution.usage, field):
                    raise ValueError("provider call usage exceeds its node execution usage")

        executions_by_node: defaultdict[str, list[NodeExecutionRecord]] = defaultdict(list)
        for execution in self.node_executions:
            executions_by_node[execution.node].append(execution)
        if set(self.usage_by_node) != set(executions_by_node):
            raise ValueError("usage_by_node keys must equal node execution nodes")
        for node, executions in executions_by_node.items():
            node_calls = tuple(
                call for call in self.provider_calls if call.node == node
            )
            expected = self._aggregate_usage(
                tuple(item.usage for item in executions),
                wall_seconds=max((item.usage.wall_seconds for item in executions), default=0),
                cost_usd=self._charged_cost(node_calls),
            )
            if self.usage_by_node[node] != expected:
                raise ValueError("usage_by_node does not match aggregated node executions")
        expected_run = self._aggregate_usage(
            tuple(self.usage_by_node.values()),
            wall_seconds=(self.finished_at - self.started_at).total_seconds(),
            cost_usd=self._charged_cost(self.provider_calls),
        )
        if self.usage != expected_run:
            raise ValueError("manifest usage does not match reconciled node usage")

    def _validate_pricing(self) -> None:
        model_calls = tuple(call for call in self.provider_calls if call.operation == "model")
        if self.pricing_status == "unknown":
            if self.pricing_snapshots:
                raise ValueError("unknown pricing requires no pricing_snapshots")
            if any(
                call.estimated_cost_usd is not None or call.pricing_snapshot_id is not None
                for call in model_calls
            ):
                raise ValueError("unknown pricing requires null call estimates")
            if self.usage.cost_usd is not None:
                raise ValueError("unknown pricing requires null usage cost")
            return

        snapshot_by_key: dict[tuple[str, str, str], PricingSnapshot] = {}
        snapshot_ids: set[str] = set()
        for snapshot in self.pricing_snapshots:
            key = (snapshot.provider_id, snapshot.endpoint_type, snapshot.model_id)
            if key in snapshot_by_key or snapshot.snapshot_id in snapshot_ids:
                raise ValueError("pricing snapshots must be unique by model and snapshot_id")
            snapshot_by_key[key] = snapshot
            snapshot_ids.add(snapshot.snapshot_id)
        required_keys = {
            (call.provider_id, call.endpoint_type, cast("str", call.model_id))
            for call in model_calls
        }
        if not required_keys.issubset(snapshot_by_key):
            raise ValueError("estimated pricing requires one snapshot per model endpoint")

        charged = Decimal(0)
        for call in model_calls:
            key = (call.provider_id, call.endpoint_type, cast("str", call.model_id))
            pricing = snapshot_by_key[key]
            if call.pricing_snapshot_id != pricing.snapshot_id:
                raise ValueError("provider call pricing_snapshot_id does not match")
            if call.cache_hit:
                if call.estimated_cost_usd != Decimal(0):
                    raise ValueError("model cache hit estimated cost must be zero")
                continue
            calculated = CostCalculator.estimate(call.usage, pricing).total_usd
            if call.estimated_cost_usd != calculated:
                raise ValueError("provider call estimated cost does not match pricing snapshot")
            charged += calculated
        charged = charged.quantize(CostCalculator.QUANTUM, rounding=ROUND_HALF_EVEN)
        if self.usage.cost_usd != charged:
            raise ValueError("manifest usage cost must equal charged non-cache model calls")

    def _validate_budget_limits(self) -> None:
        if self.usage.total_tokens > self.budget.max_total_tokens:
            raise ValueError("manifest token usage exceeds budget")
        if self.usage.search_calls > self.budget.max_search_calls:
            raise ValueError("manifest search usage exceeds budget")
        if self.usage.pages > self.budget.max_pages:
            raise ValueError("manifest page usage exceeds budget")
        if self.usage.retries > self.budget.max_retries:
            raise ValueError("manifest retry usage exceeds budget")
        if self.usage.wall_seconds > self.budget.max_wall_time_seconds:
            raise ValueError("manifest wall time exceeds budget")
        if (
            self.budget.max_cost_usd is not None
            and self.usage.cost_usd is not None
            and self.usage.cost_usd > self.budget.max_cost_usd
        ):
            raise ValueError("manifest cost exceeds budget")

    def canonical_sha256(self) -> str:
        payload = self.model_dump(mode="json", exclude={"manifest_sha256"})
        return hashlib.sha256(_canonical_bytes(payload)).hexdigest()


__all__ = [
    "CostBreakdown",
    "CostCalculator",
    "EvidenceHashRecord",
    "NodeExecutionRecord",
    "ParsedArtifactRecord",
    "PricingSnapshot",
    "ProviderCallRecord",
    "ProviderProfileRecord",
    "RunManifest",
]
