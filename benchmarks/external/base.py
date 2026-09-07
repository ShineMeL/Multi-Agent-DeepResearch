"""Offline, hash-addressed adapters for optional external benchmarks.

The external benchmarks deliberately live behind a small boundary.  The
objects in this module are either evaluator-side metadata or the canonical
``RuntimeTask``/``FrozenEvidenceRecord`` objects already owned by Core.  No
external answer, rubric, or relevance label is ever projected into the agent
process.

This module does not download anything.  ``benchmarks.scripts.fetch_external``
is the only place that may materialise a raw payload, and it refuses to do so
unless a caller supplies an immutable upstream revision and a verified local
payload.  That makes the normal repository checkout useful without pretending
that the optional third-party datasets are present.
"""

from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Literal, Protocol, Self, cast
from urllib.parse import urlsplit

import yaml
from pydantic import (
    AliasChoices,
    AnyHttpUrl,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from benchmarks.datasets.models import FrozenEvidenceRecord, RuntimeTask
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.scripts.build_snapshot import build_one
from deepresearch.domain import FreshnessRequirement, ResearchRequest, SourceType
from deepresearch.domain.locators import HtmlLocator
from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from experiments.models import BudgetPreset, SealedModel, Sha256

BenchmarkName = Literal["livedrbench", "frames", "deepresearchbench"]
BENCHMARK_NAMES: tuple[BenchmarkName, ...] = (
    "livedrbench",
    "frames",
    "deepresearchbench",
)
BENCHMARK_COUNTS: Mapping[BenchmarkName, int] = {
    "livedrbench": 10,
    "frames": 20,
    "deepresearchbench": 10,
}

EXTERNAL_RAW_ROOT = "benchmarks/private/external/raw"
EXTERNAL_STAGING_ROOT = "benchmarks/private/external/staging"
EXTERNAL_SNAPSHOT_ROOT = "benchmarks/snapshots/external"
EXTERNAL_CONFIG_SCHEMA = "external-config-v1"
EXTERNAL_LOCK_SCHEMA = "external-lock-v1"

_REVISION_RE = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _nonblank(value: str, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _immutable_revision(value: str) -> str:
    if type(value) is not str or _REVISION_RE.fullmatch(value) is None or set(value) <= {"0"}:
        raise ValueError("upstream revision must be an immutable 40- or 64-character ID")
    return value


def _immutable_version(value: str, *, label: str) -> str:
    value = _nonblank(value, label=label)
    if value.casefold() in {"latest", "main", "master", "head", "trunk", "tip"}:
        raise ValueError(f"{label} must not be a moving version")
    if any(character.isspace() or character in "/\\" for character in value):
        raise ValueError(f"{label} must not contain whitespace or path separators")
    return value


def _aware(value: datetime, *, label: str) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{label} must be timezone-aware")
    return value.astimezone(UTC)


def _canonical_name(value: str, *, label: str = "external ID") -> str:
    if type(value) is not str or re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", value) is None:
        raise ValueError(f"{label} must be a stable single path component")
    return value


def _relative(value: str, *, label: str) -> str:
    """Validate a repository-relative POSIX path without resolving it."""

    if type(value) is not str or not value or value.startswith("/"):
        raise ValueError(f"{label} must be a canonical relative path")
    if "\\" in value or ":" in value:
        raise ValueError(f"{label} must use POSIX separators")
    parts = tuple(value.split("/"))
    if any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"{label} must not contain traversal")
    if value != "/".join(parts):
        raise ValueError(f"{label} is not canonical")
    lowered = value.casefold()
    if any(token in lowered for token in ("benchmarks/private", "gold", "rubric", "acceptable_claim")):
        raise ValueError(f"{label} must not disclose private evaluator paths")
    return value


class ExternalSnapshotLock(SealedModel):
    """One verified Task 3 snapshot belonging to an external task."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkName
    external_id: str
    task_id: str
    snapshot_id: str
    corpus_version: str
    index_version: str
    snapshot_relative_path: str
    records_sha256: Sha256
    manifest_sha256: Sha256

    @field_validator("external_id", "task_id", "snapshot_id", "corpus_version", "index_version")
    @classmethod
    def nonempty_identity(cls, value: str) -> str:
        return _nonblank(value, label="snapshot identity")

    @field_validator("external_id")
    @classmethod
    def stable_external_id(cls, value: str) -> str:
        return _canonical_name(value)

    @field_validator("task_id")
    @classmethod
    def stable_task_id(cls, value: str) -> str:
        return _canonical_name(value, label="task ID")

    @field_validator("snapshot_id")
    @classmethod
    def stable_snapshot_id(cls, value: str) -> str:
        return _canonical_name(value, label="snapshot ID")

    @field_validator("corpus_version", "index_version")
    @classmethod
    def immutable_snapshot_versions(cls, value: str, info: object) -> str:
        label = getattr(info, "field_name", "snapshot version")
        return _immutable_version(value, label=str(label))

    @field_validator("snapshot_relative_path")
    @classmethod
    def safe_snapshot_path(cls, value: str) -> str:
        value = _relative(value, label="snapshot_relative_path")
        if value.startswith(("benchmarks/", "experiments/")):
            raise ValueError("snapshot_relative_path must be relative to external snapshot root")
        return value

    @model_validator(mode="after")
    def check_namespace(self) -> Self:
        expected = f"ext-{self.benchmark}-{self.external_id}"
        if self.task_id != expected:
            raise ValueError("external snapshot task_id must match its benchmark namespace")
        expected_path = f"{self.benchmark}/{self.task_id}"
        if self.snapshot_relative_path != expected_path:
            raise ValueError("external snapshot path must match its benchmark namespace")
        return self


class ExternalSourceLock(SealedModel):
    """Hash and provenance for one immutable upstream raw payload."""

    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        populate_by_name=True,
    )

    benchmark: BenchmarkName
    upstream_revision: str = Field(
        validation_alias=AliasChoices("upstream_revision", "commit_hash", "revision", "commit")
    )
    source_url: str = Field(validation_alias=AliasChoices("source_url", "url"))
    raw_relative_path: str = Field(
        validation_alias=AliasChoices("raw_relative_path", "raw_path", "path")
    )
    sha256: Sha256 = Field(
        validation_alias=AliasChoices("sha256", "raw_sha256", "dataset_sha256")
    )
    license_id: str = Field(validation_alias=AliasChoices("license_id", "license"))
    adapter_version: str
    fetched_at: datetime
    corpus_version: str
    index_version: str
    snapshot_locks: tuple[ExternalSnapshotLock, ...] = Field(
        default=(), validation_alias=AliasChoices("snapshot_locks", "snapshots")
    )

    @field_validator("upstream_revision")
    @classmethod
    def immutable_revision(cls, value: str) -> str:
        return _immutable_revision(value)

    @field_validator("source_url")
    @classmethod
    def direct_source_url(cls, value: str) -> str:
        value = _nonblank(value, label="source_url")
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("source_url must be a direct credential-free URL")
        return value

    @field_validator("raw_relative_path")
    @classmethod
    def safe_raw_path(cls, value: str) -> str:
        return _relative(value, label="raw_relative_path")

    @field_validator("license_id", "adapter_version", "corpus_version", "index_version")
    @classmethod
    def nonempty_metadata(cls, value: str) -> str:
        return _immutable_version(value, label="external lock metadata")

    @field_validator("fetched_at")
    @classmethod
    def timezone(cls, value: datetime) -> datetime:
        return _aware(value, label="fetched_at")

    @field_validator("snapshot_locks")
    @classmethod
    def sorted_snapshots(cls, value: tuple[ExternalSnapshotLock, ...]) -> tuple[ExternalSnapshotLock, ...]:
        keys = tuple((item.task_id, item.external_id) for item in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("snapshot locks must be sorted and unique")
        return value

    @model_validator(mode="after")
    def benchmark_matches(self) -> Self:
        if any(item.benchmark != self.benchmark for item in self.snapshot_locks):
            raise ValueError("snapshot lock benchmark disagrees with source lock")
        if any(
            item.corpus_version != self.corpus_version
            or item.index_version != self.index_version
            for item in self.snapshot_locks
        ):
            raise ValueError("snapshot lock versions disagree with source lock")
        return self


class ExternalBenchmarkLock(SealedModel):
    """The public aggregate lock stored at ``benchmarks/external``.

    Keeping one field per benchmark makes the lock human-auditable and also
    preserves the small ``lock_payload["frames"]`` mutation contract used by
    the validation tests.  A few historical names are accepted on input, but
    output is always canonical and deterministic.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external-lock-v1"] = EXTERNAL_LOCK_SCHEMA
    livedrbench: ExternalSourceLock
    frames: ExternalSourceLock
    deepresearchbench: ExternalSourceLock

    @model_validator(mode="before")
    @classmethod
    def normalize_legacy_shape(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = dict(cast(Mapping[str, object], value))
        if "benchmarks" in raw and not any(name in raw for name in BENCHMARK_NAMES):
            entries = raw.pop("benchmarks")
            if isinstance(entries, Mapping):
                raw.update(cast(Mapping[str, object], entries))
            elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
                for item in cast(Sequence[object], entries):
                    if isinstance(item, Mapping):
                        row = cast(Mapping[str, object], item)
                        if isinstance(row.get("benchmark"), str):
                            raw[str(row["benchmark"])] = row
        if "sources" in raw and not any(name in raw for name in BENCHMARK_NAMES):
            entries = raw.pop("sources")
            if isinstance(entries, Mapping):
                raw.update(cast(Mapping[str, object], entries))
        raw.setdefault("schema_version", EXTERNAL_LOCK_SCHEMA)
        # A lock writer may choose to put all child snapshot rows at the top
        # level.  Move them into the matching source entry before validation.
        top_snapshots = raw.pop("snapshots", None)
        if isinstance(top_snapshots, Sequence) and not isinstance(top_snapshots, (str, bytes, bytearray)):
            by_benchmark: dict[str, list[object]] = {name: [] for name in BENCHMARK_NAMES}
            for item in cast(Sequence[object], top_snapshots):
                if isinstance(item, Mapping):
                    row = cast(Mapping[str, object], item)
                    benchmark = row.get("benchmark")
                    if isinstance(benchmark, str) and benchmark in by_benchmark:
                        by_benchmark[benchmark].append(row)
            for name in BENCHMARK_NAMES:
                entry = raw.get(name)
                if isinstance(entry, Mapping) and by_benchmark[name]:
                    updated = dict(cast(Mapping[str, object], entry))
                    updated.setdefault("snapshot_locks", by_benchmark[name])
                    raw[name] = updated
        return raw

    @model_validator(mode="after")
    def benchmark_keys_match(self) -> Self:
        for name in BENCHMARK_NAMES:
            if getattr(self, name).benchmark != name:
                raise ValueError(f"{name} lock has the wrong benchmark identity")
        return self

    def entry(self, benchmark: BenchmarkName) -> ExternalSourceLock:
        return cast(ExternalSourceLock, getattr(self, benchmark))

    def snapshot_locks(self, benchmark: BenchmarkName | None = None) -> tuple[ExternalSnapshotLock, ...]:
        entries = BENCHMARK_NAMES if benchmark is None else (benchmark,)
        return tuple(
            sorted(
                (lock for name in entries for lock in self.entry(name).snapshot_locks),
                key=lambda item: (item.benchmark, item.task_id, item.external_id),
            )
        )


class ExternalSnapshotBuildResult(SealedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshots: tuple[ExternalSnapshotLock, ...]
    frozen_records_by_task: dict[str, tuple[FrozenEvidenceRecord, ...]]

    @model_validator(mode="after")
    def sorted_and_private_free(self) -> Self:
        if tuple(item.task_id for item in self.snapshots) != tuple(
            sorted(item.task_id for item in self.snapshots)
        ):
            raise ValueError("external snapshots must be sorted by task ID")
        if len({item.task_id for item in self.snapshots}) != len(self.snapshots):
            raise ValueError("external snapshot task IDs must be unique")
        expected = {item.task_id for item in self.snapshots}
        if set(self.frozen_records_by_task) != expected:
            raise ValueError("external records must cover every snapshot")
        for task_id, records in self.frozen_records_by_task.items():
            if any(record.task_id != task_id for record in records):
                raise ValueError("external frozen records disagree with snapshot task ID")
        return self


class ExternalEvaluationPlan(SealedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkName
    external_id: str
    supported_metric_names: tuple[str, ...]
    private_scoring_reference: str
    upstream_record_sha256: Sha256

    @field_validator("external_id", "private_scoring_reference")
    @classmethod
    def nonempty(cls, value: str) -> str:
        return _nonblank(value, label="external evaluation identity")

    @field_validator("external_id")
    @classmethod
    def valid_id(cls, value: str) -> str:
        return _canonical_name(value)

    @field_validator("supported_metric_names")
    @classmethod
    def metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or any(type(item) is not str or not item.strip() for item in value):
            raise ValueError("supported_metric_names must be non-empty")
        if tuple(value) != tuple(sorted(set(value))):
            raise ValueError("supported_metric_names must be sorted and unique")
        return value


class ExternalTaskSelection(SealedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    runtime_task: RuntimeTask
    evaluation_plan: ExternalEvaluationPlan

    @model_validator(mode="after")
    def bind_ids(self) -> Self:
        task_id = self.runtime_task.task_id
        expected = f"ext-{self.evaluation_plan.benchmark}-{self.evaluation_plan.external_id}"
        if task_id != expected:
            raise ValueError("RuntimeTask is not namespaced for its evaluation plan")
        return self


class ExternalBenchmarkSpec(SealedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    corpus_version: str
    expected_count: int = Field(gt=0)
    adapter_version: str
    supported_metric_names: tuple[str, ...]
    source_url: str | None = None
    license_id: str | None = None

    @field_validator("corpus_version", "adapter_version")
    @classmethod
    def spec_text(cls, value: str) -> str:
        return _nonblank(value, label="external benchmark spec")

    @field_validator("corpus_version")
    @classmethod
    def spec_version(cls, value: str) -> str:
        return _immutable_version(value, label="corpus_version")

    @field_validator("supported_metric_names")
    @classmethod
    def spec_metrics(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value or tuple(value) != tuple(sorted(set(value))):
            raise ValueError("benchmark metric names must be sorted and unique")
        return value

    @field_validator("source_url")
    @classmethod
    def spec_url(cls, value: str | None) -> str | None:
        if value is None:
            return None
        parsed = urlsplit(_nonblank(value, label="source_url"))
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError("source_url must be a direct credential-free URL")
        return value

    @field_validator("license_id")
    @classmethod
    def spec_license(cls, value: str | None) -> str | None:
        return None if value is None else _nonblank(value, label="license_id")


class ExternalConfig(SealedModel):
    """Reviewable external configuration; it is not a data lock."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["external-config-v1"] = EXTERNAL_CONFIG_SCHEMA
    raw_root: Literal["benchmarks/private/external/raw"] = EXTERNAL_RAW_ROOT
    documents_staging_root: Literal["benchmarks/private/external/staging"] = EXTERNAL_STAGING_ROOT
    snapshot_root: Literal["benchmarks/snapshots/external"] = EXTERNAL_SNAPSHOT_ROOT
    index_version: str
    evaluation_cutoff: date = date(2026, 8, 29)
    livedrbench: ExternalBenchmarkSpec
    frames: ExternalBenchmarkSpec
    deepresearchbench: ExternalBenchmarkSpec

    @model_validator(mode="before")
    @classmethod
    def normalize_benchmark_map(cls, value: object) -> object:
        if not isinstance(value, Mapping):
            return value
        raw = dict(cast(Mapping[str, object], value))
        if "benchmarks" in raw and not any(name in raw for name in BENCHMARK_NAMES):
            entries = raw.pop("benchmarks")
            if isinstance(entries, Mapping):
                raw.update(cast(Mapping[str, object], entries))
            elif isinstance(entries, Sequence) and not isinstance(entries, (str, bytes, bytearray)):
                for item in cast(Sequence[object], entries):
                    if isinstance(item, Mapping):
                        row = cast(Mapping[str, object], item)
                        if isinstance(row.get("benchmark"), str):
                            raw[str(row["benchmark"])] = row
        return raw

    @field_validator("index_version")
    @classmethod
    def index_text(cls, value: str) -> str:
        return _immutable_version(value, label="index_version")

    @model_validator(mode="after")
    def fixed_counts(self) -> Self:
        for name in BENCHMARK_NAMES:
            spec = cast(ExternalBenchmarkSpec, getattr(self, name))
            if spec.expected_count != BENCHMARK_COUNTS[name]:
                raise ValueError(f"{name} expected_count must be {BENCHMARK_COUNTS[name]}")
        return self

    def spec(self, benchmark: BenchmarkName) -> ExternalBenchmarkSpec:
        return cast(ExternalBenchmarkSpec, getattr(self, benchmark))


# Names used by early portfolio notes; keep them as explicit aliases rather
# than creating duplicate models.
ExternalBenchmarkConfig = ExternalConfig
ExternalLock = ExternalBenchmarkLock


def _config_payload(path: Path) -> object:
    source = Path(path)
    _reject_symlink_prefix(source)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    return yaml.safe_load(source.read_bytes())


def load_external_config(path: Path) -> ExternalConfig:
    try:
        return ExternalConfig.model_validate(_config_payload(path))
    except (OSError, TypeError, ValueError, yaml.YAMLError, ValidationError) as error:
        raise ValueError("external configuration is invalid") from error


def load_external_lock(path: Path) -> ExternalBenchmarkLock:
    source = Path(path)
    _reject_symlink_prefix(source)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    try:
        # JSON has no tuple/datetime types; strict validation is appropriate
        # for agent RuntimeTask records but would reject the canonical JSON
        # representation of this metadata lock.  The model validators still
        # enforce every identity/hash/path invariant.
        return ExternalBenchmarkLock.model_validate_json(source.read_bytes())
    except (OSError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as error:
        raise ValueError("external lock is invalid") from error


def canonical_external_config_hash(path: Path) -> str:
    source = Path(path)
    _reject_symlink_prefix(source)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    return sha256_bytes(source.read_bytes())


def canonical_external_lock_hash(path: Path) -> str:
    source = Path(path)
    _reject_symlink_prefix(source)
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    return sha256_bytes(source.read_bytes())


def _reject_symlink_prefix(path: Path) -> None:
    absolute = Path(path).absolute()
    current = absolute
    while current != Path(current.anchor):
        if _is_link_or_reparse(current):
            raise ValueError("external path contains a symlink")
        current = current.parent


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0) & reparse_flag
    )


def _infer_repo_root(
    path: Path,
    *,
    kind: Literal["raw", "staging", "snapshot"],
) -> Path:
    suffix_length = {"raw": 4, "staging": 4, "snapshot": 3}[kind]
    absolute = Path(path).absolute()
    if len(absolute.parts) < suffix_length:
        raise ValueError("external root cannot determine repository root")
    return Path(*absolute.parts[:-suffix_length])


def fixed_external_root(
    path: Path,
    *,
    kind: Literal["raw", "staging", "snapshot"],
    repo_root: Path | None = None,
    create: bool = False,
) -> Path:
    """Resolve an external root while preserving the fixed final components.

    When ``repo_root`` is supplied (as all production boundaries do), the
    complete path is bound to that checkout.  The fallback is retained only
    for low-level isolated fixtures; callers that accept user paths must pass
    the explicit repository root.
    """

    expected = {
        "raw": ("benchmarks", "private", "external", "raw"),
        "staging": ("benchmarks", "private", "external", "staging"),
        "snapshot": ("benchmarks", "snapshots", "external"),
    }[kind]
    candidate = Path(path)
    if ".." in candidate.parts:
        raise ValueError("external root contains traversal")
    if repo_root is not None:
        repository = Path(repo_root)
        if ".." in repository.parts:
            raise ValueError("repository root contains traversal")
        repository = repository.absolute()
        _reject_symlink_prefix(repository)
        expected_path = repository.joinpath(*expected)
        absolute = candidate if candidate.is_absolute() else repository / candidate
        absolute = absolute.absolute()
        if os.path.normcase(str(absolute)) != os.path.normcase(str(expected_path)):
            raise ValueError(f"external {kind} root is not fixed to this repository")
    else:
        absolute = candidate.absolute()
        if tuple(part.casefold() for part in absolute.parts[-len(expected) :]) != expected:
            raise ValueError(f"external {kind} root is not fixed")
    if tuple(part.casefold() for part in absolute.parts[-len(expected) :]) != expected:
        raise ValueError(f"external {kind} root is not fixed")
    _reject_symlink_prefix(absolute)
    if create:
        absolute.mkdir(parents=True, exist_ok=True)
        _reject_symlink_prefix(absolute)
    return absolute


def _safe_child(root: Path, relative: str, *, label: str, require_file: bool = False) -> Path:
    relative = _relative(relative, label=label)
    root = Path(root).absolute()
    _reject_symlink_prefix(root)
    candidate = root.joinpath(*relative.split("/"))
    if not candidate.resolve(strict=False).is_relative_to(root.resolve(strict=False)):
        raise ValueError(f"{label} escapes its fixed root")
    current = root
    for part in candidate.relative_to(root).parts:
        current = current / part
        if _is_link_or_reparse(current):
            raise ValueError(f"{label} contains a symlink")
    if require_file and (not candidate.is_file() or candidate.is_symlink()):
        raise FileNotFoundError(candidate)
    return candidate


def _invalid_snapshot(message: str) -> ProviderError:
    return ProviderError(
        code="INVALID_SNAPSHOT",
        provider="external-corpus",
        operation="snapshot",
        public_message=message,
        retryable=False,
    )


def verify_external_snapshot(lock: ExternalSnapshotLock, *, snapshot_root: Path) -> FrozenCorpusSnapshot:
    """Load one child through Task 3 and verify its outer manifest hash."""

    try:
        root = _safe_child(snapshot_root, lock.snapshot_relative_path, label="snapshot", require_file=False)
        if not root.is_dir() or root.is_symlink():
            raise FileNotFoundError(root)
        outer = root / "manifest.sha256"
        if not outer.is_file() or outer.is_symlink():
            raise FileNotFoundError(outer)
        if sha256_bytes(outer.read_bytes()) != lock.manifest_sha256:
            raise ValueError("external snapshot manifest hash mismatch")
        snapshot = FrozenCorpusSnapshot.load(root, task_id=lock.task_id)
        manifest = snapshot.manifest
        if (
            manifest.snapshot_id,
            manifest.corpus_version,
            manifest.index_version,
        ) != (lock.snapshot_id, lock.corpus_version, lock.index_version):
            raise ValueError("external snapshot identity mismatch")
        if manifest.documents_sha256 != lock.records_sha256:
            # ``records_sha256`` names the canonical documents JSONL hash.  A
            # legacy lock may use the same bytes under an explicit sha256 alias;
            # accepting only this one equivalent is safe and deterministic.
            raise ValueError("external snapshot records hash mismatch")
        return snapshot
    except ProviderError:
        raise
    except (OSError, TypeError, ValueError, ValidationError, json.JSONDecodeError) as error:
        if isinstance(error, FileNotFoundError):
            raise _invalid_snapshot("external frozen snapshot is unavailable") from error
        raise _invalid_snapshot("external frozen snapshot is invalid") from error


def _json_objects(payload: bytes, *, source: Path) -> tuple[dict[str, object], ...]:
    try:
        decoded = json.loads(payload)
    except json.JSONDecodeError:
        rows: list[dict[str, object]] = []
        for number, line in enumerate(payload.splitlines(), start=1):
            if not line.strip():
                raise ValueError(f"blank external raw line at {source}:{number}")
            value = json.loads(line)
            if not isinstance(value, Mapping):
                raise TypeError(f"external raw row is not an object at {source}:{number}")
            rows.append(dict(cast(Mapping[str, object], value)))
        return tuple(rows)
    values: object
    if isinstance(decoded, list):
        values = cast(list[object], decoded)
    elif isinstance(decoded, Mapping):
        mapping = cast(Mapping[str, object], decoded)
        values = mapping.get(
            "items", mapping.get("records", mapping.get("tasks", cast(object, decoded)))
        )
        if values is decoded:
            values = [cast(object, decoded)]
    else:
        raise TypeError("external raw payload must be an object or array")
    if not isinstance(values, Sequence) or isinstance(values, (str, bytes, bytearray)):
        raise TypeError("external raw collection must be an array")
    rows: list[dict[str, object]] = []
    for value in cast(Sequence[object], values):
        if not isinstance(value, Mapping):
            raise TypeError("external raw item must be an object")
        rows.append(dict(cast(Mapping[str, object], value)))
    return tuple(rows)


def _parse_datetime(value: object, *, fallback: datetime) -> datetime:
    if isinstance(value, datetime):
        return _aware(value, label="retrieved_at")
    if isinstance(value, str) and value.strip():
        try:
            return _aware(datetime.fromisoformat(value), label="retrieved_at")
        except ValueError:
            pass
    return fallback


def _source_type(value: object) -> SourceType:
    allowed = {"paper", "official_documentation", "standard", "primary_data", "first_party_statement", "secondary_analysis", "news", "unknown"}
    return cast(SourceType, value if isinstance(value, str) and value in allowed else "unknown")


def _text(value: object, *, fallback: str) -> str:
    if isinstance(value, str) and value.strip():
        return value.strip()
    return fallback


def _document_rows(item: Mapping[str, object]) -> tuple[Mapping[str, object], ...]:
    for key in ("documents", "evidence", "context", "sources", "passages"):
        value: object = item.get(key)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            rows: list[Mapping[str, object]] = []
            for entry in cast(Sequence[object], value):
                if isinstance(entry, Mapping):
                    rows.append(cast(Mapping[str, object], entry))
                elif isinstance(entry, str) and entry.strip():
                    rows.append({"text": entry})
            if rows:
                return tuple(rows)
        if isinstance(value, str) and value.strip():
            return ({"text": value},)
    return (item,)


def _runtime_question(item: Mapping[str, object], *, external_id: str) -> str:
    for key in ("question", "query", "prompt", "instruction", "title"):
        value = item.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    request = item.get("request")
    if isinstance(request, Mapping):
        request_mapping = cast(Mapping[str, object], request)
        request_question = request_mapping.get("question")
        if isinstance(request_question, str) and request_question.strip():
            return request_question.strip()
    raise ValueError(f"external task {external_id} has no question")


def _as_frozen_records(
    item: Mapping[str, object],
    *,
    benchmark: BenchmarkName,
    external_id: str,
    task_id: str,
    retrieved_at: datetime,
) -> tuple[FrozenEvidenceRecord, ...]:
    rows = _document_rows(item)
    records: list[FrozenEvidenceRecord] = []
    question = _runtime_question(item, external_id=external_id)
    for index, row in enumerate(rows, start=1):
        # Raw fixtures may already contain canonical Core records.  Rebind only
        # task identity and otherwise require the exact schema/hash checks.
        if {"evidence_id", "source_id", "raw_body_b64", "content_hash", "parsed_content_hash"} <= set(row):
            try:
                existing = FrozenEvidenceRecord.model_validate(row, strict=True)
            except (TypeError, ValueError, ValidationError) as error:
                raise ValueError("external canonical evidence row is invalid") from error
            if existing.task_id != task_id:
                existing = existing.model_copy(update={"task_id": task_id})
            records.append(existing)
            continue
        evidence_id = _canonical_name(
            str(row.get("evidence_id", row.get("id", f"ev-{index}"))), label="evidence ID"
        )
        source_id = _canonical_name(
            str(row.get("source_id", f"src-{external_id}-{index}")), label="source ID"
        )
        text_value = row.get(
            "normalized_text", row.get("text", row.get("content", row.get("body")))
        )
        if not isinstance(text_value, str) or not text_value.strip():
            raise ValueError(
                f"external task {external_id} evidence {evidence_id} has no text"
            )
        text = text_value.strip()
        raw_value = row.get("raw_body_b64")
        if isinstance(raw_value, str):
            try:
                raw_bytes = base64.b64decode(raw_value, validate=True)
            except (ValueError, TypeError):
                raw_bytes = raw_value.encode("utf-8")
        else:
            raw_bytes = text.encode("utf-8")
        content_hash = hashlib.sha256(raw_bytes).hexdigest()
        parsed_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()
        excerpt = _text(row.get("excerpt"), fallback=text[: min(len(text), 500)])
        locator_text = _text(row.get("locator_text"), fallback=text)
        if excerpt not in locator_text:
            locator_text = excerpt
        start = locator_text.find(excerpt)
        if start < 0:
            start = 0
            locator_text = excerpt
        locator = HtmlLocator(
            paragraph_id=f"external-{index}",
            start_char=start,
            end_char=start + len(excerpt),
        )
        published = row.get("published_at")
        published_at: datetime | None = None
        unknown_reason: str | None = None
        if isinstance(published, str) and published.strip():
            try:
                published_at = _aware(
                    datetime.fromisoformat(published),
                    label="published_at",
                )
            except ValueError:
                unknown_reason = "upstream publication timestamp is not unambiguous"
        else:
            unknown_reason = "upstream publication timestamp is unavailable"
        canonical_url_value = row.get("canonical_url", row.get("url", row.get("link")))
        if not isinstance(canonical_url_value, str) or not canonical_url_value.strip():
            raise ValueError(
                f"external task {external_id} evidence {evidence_id} has no canonical URL"
            )
        canonical_url = canonical_url_value.strip()
        authors_raw = row.get("authors", ())
        authors: tuple[str, ...]
        if isinstance(authors_raw, Sequence) and not isinstance(authors_raw, (str, bytes, bytearray)):
            authors = tuple(
                str(author).strip()
                for author in cast(Sequence[object], authors_raw)
                if str(author).strip()
            )
        else:
            authors = ()
        record = FrozenEvidenceRecord(
            task_id=task_id,
            evidence_id=evidence_id,
            source_id=source_id,
            source_family_id=_canonical_name(
                _text(row.get("source_family_id"), fallback=f"family-{benchmark}-{index}"),
                label="source family ID",
            ),
            canonical_url=cast(AnyHttpUrl, canonical_url),
            title=_text(row.get("title"), fallback=question),
            authors=authors,
            media_type=_text(row.get("media_type", row.get("content_type")), fallback="text/html"),
            raw_body_b64=base64.b64encode(raw_bytes).decode("ascii"),
            content_hash=content_hash,
            normalized_text=text,
            parsed_content_hash=parsed_hash,
            locator_text=locator_text,
            locator=locator,
            excerpt=excerpt,
            excerpt_hash=hashlib.sha256(excerpt.encode("utf-8")).hexdigest(),
            published_at=published_at,
            unknown_published_at_reason=unknown_reason,
            retrieved_at=_parse_datetime(row.get("retrieved_at"), fallback=retrieved_at),
            language=_text(row.get("language"), fallback="en"),
            source_type=_source_type(row.get("source_type")),
        )
        records.append(record)
    if not records:
        raise ValueError("external task has no evidence records")
    return tuple(sorted(records, key=lambda record: (record.evidence_id, record.source_id)))


class ExternalTaskAdapter(Protocol):
    def frozen_records_for(self, external_id: str) -> tuple[FrozenEvidenceRecord, ...]: ...

    def snapshot_lock_for(self, task_id: str) -> ExternalSnapshotLock: ...


class BaseExternalAdapter:
    benchmark: BenchmarkName
    expected_count: int
    supported_metric_names: tuple[str, ...]

    def __init__(
        self,
        *,
        lock_file: Path,
        raw_root: Path,
        snapshot_root: Path,
        external_config: ExternalConfig | None = None,
        repo_root: Path | None = None,
    ) -> None:
        self.lock_file = Path(lock_file)
        self.repo_root = (
            Path(repo_root).absolute()
            if repo_root is not None
            else _infer_repo_root(raw_root, kind="raw")
        )
        expected_lock = self.repo_root / "benchmarks" / "external" / "external.lock.json"
        candidate_lock = self.lock_file if self.lock_file.is_absolute() else self.repo_root / self.lock_file
        candidate_lock = candidate_lock.absolute()
        if os.path.normcase(str(candidate_lock)) != os.path.normcase(str(expected_lock)):
            raise ValueError("external lock must use the fixed repository path")
        _reject_symlink_prefix(candidate_lock)
        self.lock_file = candidate_lock
        self.raw_root = fixed_external_root(raw_root, kind="raw", repo_root=self.repo_root)
        self.snapshot_root = fixed_external_root(
            snapshot_root, kind="snapshot", repo_root=self.repo_root
        )
        self.external_config = external_config
        self._lock = load_external_lock(self.lock_file)
        self._entry = self._lock.entry(self.benchmark)
        if self._entry.benchmark != self.benchmark:
            raise ValueError("external lock benchmark identity mismatch")
        if len(self._entry.snapshot_locks) != self.expected_count:
            raise ValueError("external lock snapshot count is not canonical")
        if external_config is not None:
            spec = external_config.spec(self.benchmark)
            if (
                spec.expected_count != BENCHMARK_COUNTS[self.benchmark]
                or spec.expected_count != self.expected_count
                or spec.adapter_version != self._entry.adapter_version
            ):
                raise ValueError("external config and lock adapter identity mismatch")
            if self._entry.corpus_version != spec.corpus_version or self._entry.index_version != external_config.index_version:
                raise ValueError("external config and lock corpus/index identity mismatch")
        self._items: dict[str, Mapping[str, object]] | None = None
        self._raw_payload: bytes | None = None
        self._records: dict[str, tuple[FrozenEvidenceRecord, ...]] = {}
        self._locks_by_task = {lock.task_id: lock for lock in self._entry.snapshot_locks}
        if len(self._locks_by_task) != len(self._entry.snapshot_locks):
            raise ValueError("external snapshot lock task IDs must be unique")

    def _load_items(self) -> Mapping[str, Mapping[str, object]]:
        raw_path = _safe_child(self.raw_root, self._entry.raw_relative_path, label="external raw", require_file=True)
        # Read once and use exactly those immutable bytes for both hash and
        # parsing.  A cached selection is revalidated on every access so a
        # raw-file mutation cannot bypass the lock after the first call.
        payload = raw_path.read_bytes()
        if sha256_bytes(payload) != self._entry.sha256:
            raise ProviderError(
                code="INVALID_SNAPSHOT",
                provider="external-corpus",
                operation="raw",
                public_message="external raw payload hash mismatch",
                retryable=False,
            )
        if self._raw_payload is not None and payload != self._raw_payload:
            raise ProviderError(
                code="INVALID_SNAPSHOT",
                provider="external-corpus",
                operation="raw",
                public_message="external raw payload changed after selection",
                retryable=False,
            )
        if self._items is not None:
            return self._items
        rows = _json_objects(payload, source=raw_path)
        items: dict[str, Mapping[str, object]] = {}
        for row in rows:
            raw_id = row.get("external_id", row.get("id", row.get("task_id", row.get("uid"))))
            if not isinstance(raw_id, str) or not raw_id.strip():
                raise ValueError("external raw item has no stable external_id")
            external_id = _canonical_name(raw_id.strip())
            if external_id in items:
                raise ValueError("duplicate external_id in raw payload")
            items[external_id] = row
        self._items = items
        self._raw_payload = payload
        return items

    def _eligible(self, item: Mapping[str, object]) -> bool:
        return True

    def _item_digest(self, item: Mapping[str, object]) -> str:
        return sha256_bytes(canonical_json_bytes(item))

    def _snapshot_lock(self, external_id: str) -> ExternalSnapshotLock:
        expected_prefix = f"ext-{self.benchmark}-{external_id}"
        candidates = [lock for lock in self._entry.snapshot_locks if lock.external_id == external_id]
        if len(candidates) != 1:
            candidates = [lock for lock in self._entry.snapshot_locks if lock.task_id == expected_prefix]
        if len(candidates) != 1:
            raise _invalid_snapshot(f"external snapshot lock is missing for {external_id}")
        return candidates[0]

    def snapshot_lock_for(self, task_id: str) -> ExternalSnapshotLock:
        lock = self._locks_by_task.get(task_id)
        if lock is None:
            raise _invalid_snapshot(f"external snapshot lock is missing for {task_id}")
        return lock

    def frozen_records_for(self, external_id: str) -> tuple[FrozenEvidenceRecord, ...]:
        # Revalidate the immutable raw bytes even when normalized records are
        # cached.  Otherwise a mutation after the first materialization could
        # bypass the source-lock hash through this cache path.
        self._load_items()
        if external_id in self._records:
            return self._records[external_id]
        item = self._load_items().get(external_id)
        if item is None:
            raise KeyError(external_id)
        lock = self._snapshot_lock(external_id)
        records = _as_frozen_records(
            item,
            benchmark=self.benchmark,
            external_id=external_id,
            task_id=lock.task_id,
            retrieved_at=self._entry.fetched_at,
        )
        self._records[external_id] = records
        return records

    def _selection_ids(self) -> tuple[str, ...]:
        items = self._load_items()
        candidates = [external_id for external_id, item in items.items() if self._eligible(item)]
        candidates.sort(key=lambda value: (hashlib.sha256(value.encode("utf-8")).hexdigest(), value))
        selected = tuple(candidates[: self.expected_count])
        if len(selected) != self.expected_count:
            raise ProviderError(
                code="REPLAY_MISS",
                provider="external-corpus",
                operation="select",
                public_message=f"{self.benchmark} has fewer than {self.expected_count} eligible records",
                retryable=False,
            )
        return selected

    def select(
        self,
        *,
        provider_profile_id: str,
        budget_preset: BudgetPreset,
    ) -> tuple[ExternalTaskSelection, ...]:
        provider_profile_id = _nonblank(provider_profile_id, label="provider_profile_id")
        if budget_preset not in {"low", "medium", "high"}:
            raise ValueError("budget_preset is invalid")
        selected: list[ExternalTaskSelection] = []
        cutoff = self.external_config.evaluation_cutoff if self.external_config else date(2026, 8, 29)
        for external_id in self._selection_ids():
            item = self._load_items()[external_id]
            lock = self._snapshot_lock(external_id)
            verify_external_snapshot(lock, snapshot_root=self.snapshot_root)
            task_id = lock.task_id
            question = _runtime_question(item, external_id=external_id)
            request = ResearchRequest(
                question=question,
                output_requirements={"answer_shape": "markdown"},
                report_language="en",
                source_languages=("en",),
                freshness_requirement=FreshnessRequirement(kind="none"),
                execution_mode="hybrid",
                access_profile="local",
                provider_profile_id=provider_profile_id,
                run_purpose="benchmark",
                budget_preset=budget_preset,
            )
            runtime = RuntimeTask(
                task_id=task_id,
                category=self.runtime_category,
                request=request,
                evaluation_cutoff=cutoff,
                snapshot_id=lock.snapshot_id,
                corpus_version=lock.corpus_version,
                index_version=lock.index_version,
            )
            plan = ExternalEvaluationPlan(
                benchmark=self.benchmark,
                external_id=external_id,
                supported_metric_names=self.supported_metric_names,
                private_scoring_reference=f"external-evaluator://{self.benchmark}/{external_id}",
                upstream_record_sha256=self._item_digest(item),
            )
            selected.append(ExternalTaskSelection(runtime_task=runtime, evaluation_plan=plan))
        return tuple(selected)

    @property
    def runtime_category(self) -> Any:
        raise NotImplementedError


class ExternalSnapshotMaterializer:
    """Build external snapshots by delegating to the canonical Task 3 builder."""

    def build(
        self,
        *,
        selection_manifest_path: Path,
        documents_staging_root: Path,
        snapshot_root: Path,
        repo_root: Path | None = None,
        external_config: ExternalConfig | None = None,
        source_locks: Mapping[BenchmarkName, ExternalSourceLock] | None = None,
    ) -> ExternalSnapshotBuildResult:
        materializer_repo_root = (
            Path(repo_root).absolute()
            if repo_root is not None
            else _infer_repo_root(documents_staging_root, kind="staging")
        )
        staging = fixed_external_root(
            documents_staging_root, kind="staging", repo_root=materializer_repo_root
        )
        snapshots_root = fixed_external_root(
            snapshot_root,
            kind="snapshot",
            repo_root=materializer_repo_root,
            create=True,
        )
        raw_selection_path = Path(selection_manifest_path)
        if ".." in raw_selection_path.parts:
            raise ValueError("selection manifest contains traversal")
        selection_path = raw_selection_path.absolute()
        try:
            selection_relative = selection_path.relative_to(staging)
        except ValueError as error:
            raise ValueError("selection manifest must be inside external staging root") from error
        if not selection_relative.parts:
            raise ValueError("selection manifest must be a file")
        manifest_path = _safe_child(
            staging,
            selection_relative.as_posix(),
            label="selection manifest",
            require_file=True,
        )
        try:
            payload = json.loads(manifest_path.read_bytes())
        except (OSError, TypeError, json.JSONDecodeError) as error:
            raise ValueError("external selection manifest is invalid") from error
        if not isinstance(payload, Mapping):
            raise TypeError("external selection manifest is invalid")
        payload_mapping = cast(Mapping[str, object], payload)
        rows_value: object = payload_mapping.get(
            "selections", payload_mapping.get("records")
        )
        if not isinstance(rows_value, Sequence) or isinstance(rows_value, (str, bytes, bytearray)):
            raise TypeError("external selection manifest has no selections")
        rows = [
            cast(Mapping[str, object], row)
            for row in cast(Sequence[object], rows_value)
            if isinstance(row, Mapping)
        ]
        if len(rows) != len(cast(Sequence[object], rows_value)):
            raise ValueError("external selection manifest contains an invalid row")
        rows.sort(key=lambda row: (str(row.get("benchmark", "")), str(row.get("task_id", ""))))
        locks: list[ExternalSnapshotLock] = []
        records_by_task: dict[str, tuple[FrozenEvidenceRecord, ...]] = {}
        for row in rows:
            benchmark = cast(BenchmarkName, row.get("benchmark"))
            if benchmark not in BENCHMARK_NAMES:
                raise ValueError("external selection benchmark is invalid")
            task_id = str(row.get("task_id", ""))
            external_id = str(row.get("external_id", ""))
            if not task_id or not external_id:
                raise ValueError("external selection identity is incomplete")
            documents_relative = row.get("documents_relative_path", row.get("documents_path", f"{task_id}.jsonl"))
            if not isinstance(documents_relative, str):
                raise TypeError("external documents path is invalid")
            documents = _safe_child(staging, documents_relative, label="external documents", require_file=True)
            output_relative = f"{benchmark}/{task_id}"
            output = _safe_child(snapshots_root, output_relative, label="external snapshot output")
            if output.exists() or output.is_symlink():
                raise FileExistsError(output)
            corpus_value = row.get("corpus_version")
            index_value = row.get("index_version")
            if not isinstance(corpus_value, str) or not isinstance(index_value, str):
                raise TypeError("external selection versions are required")
            corpus_version = corpus_value
            index_version = index_value
            expected_spec = external_config.spec(benchmark) if external_config else None
            expected_index_version = external_config.index_version if external_config else None
            expected_source = source_locks.get(benchmark) if source_locks is not None else None
            if source_locks is not None and expected_source is None:
                raise ValueError("external source lock is missing for benchmark")
            if expected_spec is not None and (
                corpus_version != expected_spec.corpus_version
                or index_version != expected_index_version
            ):
                raise ValueError("external selection versions disagree with config")
            if expected_source is not None and (
                corpus_version != expected_source.corpus_version
                or index_version != expected_source.index_version
            ):
                raise ValueError("external selection versions disagree with source lock")
            built = build_one(
                task_id=task_id,
                documents=documents,
                output=output,
                corpus_version=corpus_version,
                index_version=index_version,
            )
            loaded = FrozenCorpusSnapshot.load(output, task_id=task_id)
            outer_hash = sha256_bytes((output / "manifest.sha256").read_bytes())
            snapshot_lock = ExternalSnapshotLock(
                benchmark=benchmark,
                external_id=external_id,
                task_id=task_id,
                snapshot_id=loaded.manifest.snapshot_id,
                corpus_version=loaded.manifest.corpus_version,
                index_version=loaded.manifest.index_version,
                snapshot_relative_path=output_relative,
                records_sha256=loaded.manifest.documents_sha256,
                manifest_sha256=outer_hash,
            )
            if built != loaded.manifest:
                raise ValueError("external snapshot self-verification mismatch")
            locks.append(snapshot_lock)
            records_by_task[task_id] = loaded.records
        locks.sort(key=lambda item: item.task_id)
        return ExternalSnapshotBuildResult(
            snapshots=tuple(locks),
            frozen_records_by_task=records_by_task,
        )


def write_external_json(path: Path, payload: object | bytes) -> Path:
    """Publish one ignored external artifact with atomic no-replace semantics."""

    target = Path(path)
    if ".." in target.parts:
        raise ValueError("external artifact path contains traversal")
    _reject_symlink_prefix(target.parent)
    target.parent.mkdir(parents=True, exist_ok=True)
    _reject_symlink_prefix(target.parent)
    data = payload if isinstance(payload, bytes) else canonical_json_bytes(payload)
    if target.exists() or target.is_symlink():
        raise FileExistsError(target)
    # Keep the sibling name short enough for Windows' default MAX_PATH while
    # retaining a per-publication nonce.
    staging = target.with_name(f".{target.name}.{uuid.uuid4().hex[:8]}.staging")
    try:
        with staging.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        # A hard-link publication is atomic and, unlike a pre-check followed by
        # rename/replace, cannot overwrite a concurrent existing target.
        os.link(staging, target)
        try:
            directory_fd = os.open(str(target.parent), os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return target


__all__ = [
    "BENCHMARK_COUNTS",
    "BENCHMARK_NAMES",
    "EXTERNAL_CONFIG_SCHEMA",
    "EXTERNAL_LOCK_SCHEMA",
    "EXTERNAL_RAW_ROOT",
    "EXTERNAL_SNAPSHOT_ROOT",
    "EXTERNAL_STAGING_ROOT",
    "BaseExternalAdapter",
    "BenchmarkName",
    "ExternalBenchmarkConfig",
    "ExternalBenchmarkLock",
    "ExternalBenchmarkSpec",
    "ExternalConfig",
    "ExternalEvaluationPlan",
    "ExternalLock",
    "ExternalSnapshotBuildResult",
    "ExternalSnapshotLock",
    "ExternalSnapshotMaterializer",
    "ExternalSourceLock",
    "ExternalTaskAdapter",
    "ExternalTaskSelection",
    "canonical_external_config_hash",
    "canonical_external_lock_hash",
    "fixed_external_root",
    "load_external_config",
    "load_external_lock",
    "verify_external_snapshot",
    "write_external_json",
]
