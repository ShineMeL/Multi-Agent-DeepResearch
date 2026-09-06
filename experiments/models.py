"""Canonical immutable experiment records; Core owns usage and pricing."""

from __future__ import annotations

import re
from collections.abc import Mapping
from datetime import datetime
from enum import StrEnum
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast, override

from pydantic import AfterValidator, BaseModel, ConfigDict, Field, field_serializer, model_validator

from benchmarks.datasets.models import RuntimeTask, TaskCategory
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.evaluators.metrics import MetricValue
from deepresearch.domain import ResourceUsage, RunBudget, RunStatus
from deepresearch.runtime.manifest import (
    RunManifest,
    _canonical_bytes,  # pyright: ignore[reportPrivateUsage]
)

if TYPE_CHECKING:
    from experiments.config import FormalExperimentConfig


def canonical_sha256(value: object) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def require_hash(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
        raise ValueError("must be a non-zero lowercase SHA-256")
    return value


def require_revision(value: str) -> str:
    if re.fullmatch(r"[0-9a-f]{40}", value) is None or value == "0" * 40:
        raise ValueError("immutable 40-character repository revision required")
    return value


def relative_path(value: str) -> str:
    path = PurePosixPath(value)
    if (
        not value
        or path.is_absolute()
        or ".." in path.parts
        or ":" in value
        or "\\" in value
        or path.as_posix() != value
        or value == "."
    ):
        raise ValueError("canonical relative path required")
    return value


Sha256 = Annotated[str, AfterValidator(require_hash)]
Revision = Annotated[str, AfterValidator(require_revision)]
RelativePath = Annotated[str, AfterValidator(relative_path)]
BudgetPreset = Literal["low", "medium", "high"]
ProtocolName = Literal["ranker_component", "planner_policy", "end_to_end", "reference"]


class SealedModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    @override
    def model_copy(self, *, update: Mapping[str, Any] | None = None, deep: bool = False) -> Self:
        del deep
        values = self.model_dump(mode="python")
        values.update(update or {})
        return type(self).model_validate(values)


class ExperimentVariant(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"
    P0 = "P0"
    ORACLE = "ORACLE"


class RankerComponentVariant(StrEnum):
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"


COMPONENT_IDS: Mapping[str, tuple[Literal["P0", "P1", "P2"], Literal["R0", "R1", "R2"]]] = (
    MappingProxyType(
        {
            "A": ("P1", "R1"),
            "B": ("P1", "R2"),
            "C": ("P2", "R1"),
            "D": ("P2", "R2"),
            "P0": ("P0", "R0"),
        }
    )
)


class DecodingConfig(SealedModel):
    temperature: Annotated[float, Field(ge=0.0, le=0.0)]
    top_p: Annotated[float, Field(ge=1.0, le=1.0)]
    top_k: Literal[-1]
    max_tokens: Annotated[int, Field(gt=0)]
    repetition_penalty: Annotated[float, Field(ge=1.0, le=1.0)]
    thinking_mode: Literal["enabled", "disabled"]
    tensor_parallel_size: Annotated[int, Field(gt=0)]
    dtype: Literal["bfloat16", "float16"]
    max_model_len: Annotated[int, Field(gt=0)]


class ReplicationConfig(SealedModel):
    seed_values: Annotated[tuple[int, ...], Field(min_length=1)]
    candidate_pool_seed: int
    unseeded_repeat_count: Annotated[int, Field(ge=1)]

    @model_validator(mode="after")
    def unique_seeds(self) -> Self:
        if len(set(self.seed_values)) != len(self.seed_values):
            raise ValueError("seed_values must be unique")
        return self


class ModelFileLock(SealedModel):
    path: RelativePath
    size: Annotated[int, Field(ge=0)]
    git_blob_or_lfs_oid: str

    @model_validator(mode="after")
    def oid(self) -> Self:
        if len(self.git_blob_or_lfs_oid) == 40:
            require_revision(self.git_blob_or_lfs_oid)
        else:
            require_hash(self.git_blob_or_lfs_oid)
        return self


class ModelSnapshotLock(SealedModel):
    repository_id: str
    requested_revision: Revision
    resolved_revision: Revision
    files: Annotated[tuple[ModelFileLock, ...], Field(min_length=1)]
    snapshot_sha256: Sha256

    @model_validator(mode="after")
    def verify_snapshot(self) -> Self:
        if not self.repository_id.strip() or self.requested_revision != self.resolved_revision:
            raise ValueError("repository identity or response revision mismatch")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(set(paths))):
            raise ValueError("model files must be sorted and unique")
        if self.snapshot_sha256 != canonical_sha256(
            [f.model_dump(mode="json") for f in self.files]
        ):
            raise ValueError("model snapshot hash mismatch")
        return self


class LockedDistribution(SealedModel):
    name: str
    version: str
    artifact_sha256: Sha256

    @model_validator(mode="after")
    def identity(self) -> Self:
        if not self.name.strip() or not self.version.strip():
            raise ValueError("distribution name and version required")
        return self


class InferenceEnvironmentLock(SealedModel):
    python_version: str
    platform: str
    cuda_version: str
    driver_version: str
    gpu_model: str
    distributions: Annotated[tuple[LockedDistribution, ...], Field(min_length=1)]
    launch_arguments_sha256: Sha256
    model_snapshot_sha256: Sha256
    environment_sha256: Sha256

    @model_validator(mode="after")
    def verify_environment(self) -> Self:
        names = tuple(item.name for item in self.distributions)
        if names != tuple(sorted(set(names))):
            raise ValueError("distributions must be sorted and unique")
        for value in (
            self.python_version,
            self.platform,
            self.cuda_version,
            self.driver_version,
            self.gpu_model,
        ):
            if not value.strip():
                raise ValueError("environment facts must be non-empty")
        payload = self.model_dump(mode="json", exclude={"environment_sha256"})
        if canonical_sha256(payload) != self.environment_sha256:
            raise ValueError("environment hash mismatch")
        return self


class ExperimentTaskRun(SealedModel):
    task_id: str
    protocol: ProtocolName
    variant: ExperimentVariant | RankerComponentVariant
    planner_id: str
    ranker_id: str
    budget_preset: BudgetPreset
    seed: int | None = None
    repeat_id: Annotated[int | None, Field(ge=1)] = None
    status: RunStatus
    validity: Literal["valid", "invalid"] = "valid"
    error_code: str | None = None
    candidate_pool_hash: Sha256 | None = None
    manifest_path: str
    artifact_ids: tuple[str, ...]
    usage: ResourceUsage
    pricing_snapshot_ids: Annotated[tuple[str, ...], Field(min_length=1, max_length=1)]
    pricing_status: Literal["estimated"]
    cost_label: Literal["estimated_from_normalized_schedule"]
    category: TaskCategory | None = None
    metrics: dict[str, MetricValue] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_metrics(self) -> Self:
        if any(name != metric.name for name, metric in self.metrics.items()):
            raise ValueError("metric name must match its dictionary key")
        return self

    @field_serializer("metrics")
    def serialize_metrics(self, value: dict[str, MetricValue]) -> dict[str, object]:
        return {key: value[key].model_dump(mode="json") for key in sorted(value)}

    @model_validator(mode="after")
    def validate_protocol(self) -> Self:
        if (self.seed is None) == (self.repeat_id is None):
            raise ValueError("exactly one seed/repeat_id is required")
        if self.variant == ExperimentVariant.ORACLE:
            raise ValueError("ORACLE is evaluator-only")
        if self.protocol == "ranker_component":
            if (
                not isinstance(self.variant, RankerComponentVariant)
                or self.ranker_id != self.variant.value
                or self.planner_id != "P1"
            ):
                raise ValueError("ranker component requires P1, matching R variant and pool hash")
            if self.status == "completed" and self.candidate_pool_hash is None:
                raise ValueError("successful ranker component requires a verified pool hash")
        elif (
            not isinstance(self.variant, ExperimentVariant)
            or COMPONENT_IDS[self.variant] != (self.planner_id, self.ranker_id)
            or (self.variant == ExperimentVariant.P0) != (self.protocol == "reference")
        ):
            raise ValueError("variant/protocol/component mismatch")
        if self.error_code == "REPLAY_MISS" and (
            self.status,
            self.validity,
            self.error_code,
        ) != ("failed", "invalid", "REPLAY_MISS"):
            raise ValueError("REPLAY_MISS must be failed and invalid")
        if self.status == "completed" and (
            self.validity != "valid" or self.error_code or self.usage.cost_usd is None
        ):
            raise ValueError("successful formal records require verified estimated usage")
        return self


class ExperimentRunResult(SealedModel):
    group_id: str
    protocol: ProtocolName
    variant_components: dict[str, tuple[str, str]]
    runs: tuple[ExperimentTaskRun, ...]

    @model_validator(mode="after")
    def validate_runs(self) -> Self:
        for run in self.runs:
            if run.protocol != self.protocol or self.variant_components.get(run.variant.value) != (
                run.planner_id,
                run.ranker_id,
            ):
                raise ValueError("run disagrees with result protocol/components")
        object.__setattr__(
            self, "variant_components", MappingProxyType(dict(self.variant_components))
        )
        return self

    @field_serializer("variant_components")
    def serialize_components(self, value: dict[str, tuple[str, str]]) -> dict[str, tuple[str, str]]:
        return dict(value)


def task_run_from_manifest(
    manifest: RunManifest,
    *,
    config: FormalExperimentConfig,
    task: RuntimeTask,
    protocol: ProtocolName,
    variant: ExperimentVariant | RankerComponentVariant,
    manifest_path: str,
    status: RunStatus,
    seed: int | None = None,
    repeat_id: int | None = None,
    candidate_pool_hash: str | None = None,
    metrics: Mapping[str, MetricValue] | None = None,
) -> ExperimentTaskRun:
    """Bind a verified Core manifest to its authorized staged request and budget."""
    from experiments.config import authorized_staged_task

    task = RuntimeTask.model_validate_json(task.model_dump_json(), strict=True)
    authorized_staged_task(
        config,
        task,
        staged_sha256=canonical_sha256(task.model_dump(mode="json")),
        budget_preset=task.request.budget_preset,
    )
    manifest = RunManifest.model_validate_json(manifest.model_dump_json())
    if manifest.pricing_status != "estimated" or manifest.pricing_snapshots != (
        config.pricing_snapshot,
    ):
        raise ValueError("manifest pricing must exactly equal the sealed pricing snapshot")
    # Core hashes ResearchRequest without the benchmark JSONL trailing newline.
    request_hash = sha256_bytes(_canonical_bytes(task.request.model_dump(mode="json")))
    if manifest.request_sha256 != request_hash:
        raise ValueError("manifest request hash does not match the authorized staged task")
    expected_limits = RunBudget.preset(task.request.budget_preset).model_dump(
        mode="json", exclude={"used_by_node"}
    )
    if manifest.budget.model_dump(mode="json", exclude={"used_by_node"}) != expected_limits:
        raise ValueError("manifest budget limits do not match the authorized request preset")
    if manifest.workflow_id != "research-v1" or manifest.seed != seed:
        raise ValueError("formal manifest workflow/seed mismatch")
    replay_miss = "REPLAY_MISS" in manifest.failure_codes
    if replay_miss and status != "failed":
        raise ValueError("REPLAY_MISS must have failed status")
    return ExperimentTaskRun(
        task_id=task.task_id,
        protocol=protocol,
        variant=variant,
        planner_id=manifest.planner_id,
        ranker_id=manifest.ranker_id,
        budget_preset=task.request.budget_preset,
        seed=seed,
        repeat_id=repeat_id,
        status=status,
        validity="invalid" if replay_miss else "valid",
        error_code="REPLAY_MISS" if replay_miss else next(iter(manifest.failure_codes), None),
        candidate_pool_hash=candidate_pool_hash,
        manifest_path=manifest_path,
        artifact_ids=manifest.artifact_ids,
        usage=manifest.usage,
        pricing_snapshot_ids=tuple(item.snapshot_id for item in manifest.pricing_snapshots),
        pricing_status="estimated",
        cost_label="estimated_from_normalized_schedule",
        category=task.category,
        metrics=dict(metrics or {}),
    )


class OracleReferenceResult(SealedModel):
    task_id: str
    dataset_version: str
    private_manifest_sha256: Sha256
    frozen_snapshot_id: str
    approved_id_set_sha256: Sha256
    metric_values: dict[str, MetricValue]
    evaluator_version: str
    created_at: datetime

    @model_validator(mode="after")
    def freeze_metrics(self) -> Self:
        if self.created_at.tzinfo is None:
            raise ValueError("created_at must be timezone aware")
        # Reference outputs carry scalars only; arbitrary MetricNote.fields can leak gold.
        if any(metric.notes or name != metric.name for name, metric in self.metric_values.items()):
            raise ValueError("oracle metrics must be named scalars without private notes")
        object.__setattr__(self, "metric_values", MappingProxyType(dict(self.metric_values)))
        return self

    @field_serializer("metric_values")
    def serialize_metrics(self, value: dict[str, MetricValue]) -> dict[str, object]:
        return {key: item.model_dump(mode="json") for key, item in value.items()}


class EvaluatorReferenceManifest(SealedModel):
    group_id: str
    private_manifest_sha256: Sha256
    evaluator_version: str
    task_ids_sha256: Sha256
    oracle_results_sha256: Sha256
    created_at: datetime


def immutable_hash_map(value: dict[str, str]) -> dict[str, str]:
    if tuple(value) != tuple(sorted(value)):
        raise ValueError("task hash map must be sorted")
    for task_id, digest in value.items():
        if not task_id.strip():
            raise ValueError("task ID must not be empty")
        require_hash(digest)
    return cast("dict[str, str]", MappingProxyType(dict(value)))
