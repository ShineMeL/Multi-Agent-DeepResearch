"""Fail-closed fetch/materialise/restore commands for external benchmarks.

The repository intentionally ships no external data.  The commands below are
therefore useful in two situations only: a caller supplies a real immutable
upstream payload, or a local evaluator has already produced a hash-verified
download lock.  There is no fallback HTTP request and no synthetic record
generation in the command-line path.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, cast

from pydantic import BaseModel, ConfigDict, ValidationError, field_validator

from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.external.base import (
    BENCHMARK_COUNTS,
    BENCHMARK_NAMES,
    BenchmarkName,
    ExternalBenchmarkLock,
    ExternalConfig,
    ExternalSnapshotLock,
    ExternalSnapshotMaterializer,
    ExternalSourceLock,
    _as_frozen_records,  # pyright: ignore[reportPrivateUsage]
    _canonical_name,  # pyright: ignore[reportPrivateUsage]
    _json_objects,  # pyright: ignore[reportPrivateUsage]
    _safe_child,  # pyright: ignore[reportPrivateUsage]
    fixed_external_root,
    load_external_config,
    load_external_lock,
    verify_external_snapshot,
    write_external_json,
)
from deepresearch.providers.errors import ProviderError


class ExternalSelectionRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    benchmark: BenchmarkName
    external_id: str
    task_id: str
    documents_relative_path: str
    corpus_version: str
    index_version: str
    upstream_record_sha256: str

    @field_validator("external_id", "task_id", "documents_relative_path", "corpus_version", "index_version")
    @classmethod
    def nonempty(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("selection metadata must be non-empty")
        return value

    @field_validator("upstream_record_sha256")
    @classmethod
    def digest(cls, value: str) -> str:
        if len(value) != 64 or value != value.lower() or value == "0" * 64:
            raise ValueError("selection hash must be a non-zero SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("selection hash must be a non-zero SHA-256") from error
        return value


class ExternalSelectionManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = "external-selection-v1"
    selections: tuple[ExternalSelectionRow, ...]

    @field_validator("schema_version")
    @classmethod
    def schema_version_value(cls, value: str) -> str:
        if value != "external-selection-v1":
            raise ValueError("external selection schema version is invalid")
        return value

    @field_validator("selections")
    @classmethod
    def sorted_unique(cls, value: tuple[ExternalSelectionRow, ...]) -> tuple[ExternalSelectionRow, ...]:
        keys = tuple((row.benchmark, row.task_id, row.external_id) for row in value)
        if keys != tuple(sorted(keys)) or len(keys) != len(set(keys)):
            raise ValueError("external selections must be sorted and unique")
        return value


def _load_download_lock(path: Path) -> ExternalBenchmarkLock:
    """Load provisional locks as the same canonical aggregate shape.

    A provisional lock has empty child snapshot tuples; the final lock is the
    only artifact accepted by adapters and ``verify-snapshots``.
    """

    return load_external_lock(path)


def _validate_external_config(config: ExternalConfig) -> None:
    """Use every fixed/configured value at each ingestion boundary."""

    if (
        config.raw_root != "benchmarks/private/external/raw"
        or config.documents_staging_root != "benchmarks/private/external/staging"
        or config.snapshot_root != "benchmarks/snapshots/external"
    ):
        raise ValueError("external config roots are not fixed")
    for benchmark in BENCHMARK_NAMES:
        spec = config.spec(benchmark)
        if spec.expected_count != BENCHMARK_COUNTS[benchmark] or not spec.adapter_version:
            raise ValueError("external config benchmark metadata is not canonical")


def _read_json(path: Path) -> object:
    source = Path(path).absolute()
    # The caller must first bind this path to a fixed evaluator root.  Keep a
    # defensive check here as this helper is also used by the CLI boundary.
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(source)
    return json.loads(source.read_bytes())


def _fixed_file(
    root: Path,
    *,
    kind: Literal["raw", "staging", "snapshot"],
    path: Path,
    label: str,
    require_file: bool = False,
    repo_root: Path | None = None,
) -> Path:
    """Bind a file path to one of the three fixed external roots."""

    raw_path = Path(path)
    if ".." in raw_path.parts:
        raise ValueError(f"{label} contains traversal")
    repository = (
        Path(repo_root).absolute()
        if repo_root is not None
        else _infer_repo_root(root, kind=kind)
    )
    fixed = fixed_external_root(root, kind=kind, repo_root=repository)
    candidate = raw_path if raw_path.is_absolute() else repository / raw_path
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(fixed)
    except ValueError as error:
        raise ValueError(f"{label} must be inside the fixed external {kind} root") from error
    if not relative.parts:
        raise ValueError(f"{label} must name a file")
    return _safe_child(
        fixed,
        relative.as_posix(),
        label=label,
        require_file=require_file,
    )


def _infer_repo_root(
    root: Path,
    *,
    kind: Literal["raw", "staging", "snapshot"],
) -> Path:
    suffix_length = {"raw": 4, "staging": 4, "snapshot": 3}[kind]
    absolute = Path(root).absolute()
    if len(absolute.parts) < suffix_length:
        raise ValueError("external root cannot determine repository root")
    return Path(*absolute.parts[:-suffix_length])


def _public_lock_path(
    path: Path,
    *,
    snapshot_root: Path,
    repo_root: Path | None = None,
) -> Path:
    """Require the final aggregate lock at ``benchmarks/external``."""

    repository = (
        Path(repo_root).absolute()
        if repo_root is not None
        else _infer_repo_root(snapshot_root, kind="snapshot")
    )
    fixed_external_root(snapshot_root, kind="snapshot", repo_root=repository)
    raw_path = Path(path)
    if ".." in raw_path.parts:
        raise ValueError("external lock contains traversal")
    candidate = raw_path if raw_path.is_absolute() else repository / raw_path
    candidate = candidate.absolute()
    expected = repository / "benchmarks" / "external" / "external.lock.json"
    # ``root.parents[2]`` is the repository root for the fixed
    # ``<repo>/benchmarks/snapshots/external`` layout.  Comparing the full
    # path catches a caller accidentally writing a second lock beside raw
    # data, while still allowing temporary repositories in tests.
    if os.path.normcase(str(candidate)) != os.path.normcase(str(expected)):
        raise ValueError("external lock must be benchmarks/external/external.lock.json")
    if candidate.is_symlink():
        raise ValueError("external lock contains a symlink")
    return candidate


def _fixed_repo_file(path: Path, *, repo_root: Path, relative: str) -> Path:
    """Bind a CLI metadata file to an exact repository-relative location."""

    repository = Path(repo_root).absolute()
    candidate = Path(path) if Path(path).is_absolute() else repository / path
    candidate = candidate.absolute()
    expected = repository / Path(relative)
    if ".." in candidate.parts or os.path.normcase(str(candidate)) != os.path.normcase(str(expected)):
        raise ValueError(f"metadata file must be {relative} inside the repository")
    current = repository
    while current != Path(current.anchor):
        if current.is_symlink():
            raise ValueError("repository root contains a symlink")
        current = current.parent
    current = repository
    for part in Path(relative).parts:
        current = current / part
        if current.is_symlink():
            raise ValueError("metadata file contains a symlink")
    if not candidate.is_file():
        raise FileNotFoundError(candidate)
    return candidate


def _raw_rows(entry: ExternalSourceLock, raw_root: Path) -> tuple[dict[str, object], ...]:
    source = _safe_child(raw_root, entry.raw_relative_path, label="external raw", require_file=True)
    payload = source.read_bytes()
    if sha256_bytes(payload) != entry.sha256:
        raise ValueError("external raw payload hash mismatch")
    return _json_objects(payload, source=source)


def _row_id(row: Mapping[str, object]) -> str:
    value = row.get("external_id", row.get("id", row.get("task_id", row.get("uid"))))
    if not isinstance(value, str) or not value.strip():
        raise ValueError("external raw item has no stable external_id")
    return _canonical_name(value.strip())


def _eligible(benchmark: BenchmarkName, row: Mapping[str, object]) -> bool:
    if benchmark == "frames":
        documents_count = 0
        for key in ("documents", "context", "evidence", "sources", "passages"):
            value = row.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                documents_count = len(cast(Sequence[object], value))
                break
        mapping = row.get("evidence_mapping")
        return documents_count >= 2 and isinstance(mapping, Mapping) and bool(cast(Mapping[object, object], mapping))
    if benchmark == "livedrbench":
        value = row.get("task_type", row.get("category", row.get("domain")))
        if value is None:
            return False
        text = str(value).casefold().replace("_", "-")
        padded = f" {text} "
        return (
            (
                any(token in text for token in ("computer-science", "computer science"))
                or " cs " in padded
            )
            and any(
                token in text
                for token in ("prior-art", "prior art", "dataset-discovery", "dataset discovery")
            )
        )
    value = row.get("task_type", row.get("category", row.get("kind")))
    if value is None or "research-report" not in str(value).casefold().replace("_", "-"):
        return False
    citation = next(
        (row.get(key) for key in ("citations", "citation_metadata", "references")),
        None,
    )
    rubric = row.get("rubric")
    if isinstance(citation, Mapping):
        has_citation = bool(cast(Mapping[object, object], citation))
    elif isinstance(citation, (list, tuple)):
        has_citation = bool(cast(list[object] | tuple[object, ...], citation))
    else:
        has_citation = False
    has_rubric = isinstance(rubric, Mapping) and bool(cast(Mapping[object, object], rubric))
    return has_citation and has_rubric


def _selected_rows(
    benchmark: BenchmarkName,
    entry: ExternalSourceLock,
    raw_root: Path,
) -> tuple[tuple[str, Mapping[str, object]], ...]:
    rows = _raw_rows(entry, raw_root)
    values: dict[str, Mapping[str, object]] = {}
    for row in rows:
        external_id = _row_id(row)
        if external_id in values:
            raise ValueError("duplicate external_id in raw payload")
        values[external_id] = row
    candidates = [external_id for external_id, row in values.items() if _eligible(benchmark, row)]
    candidates.sort(key=lambda value: (sha256_bytes(value.encode("utf-8")), value))
    if len(candidates) < BENCHMARK_COUNTS[benchmark]:
        raise RuntimeError(f"{benchmark} has fewer than its fixed offline selection count")
    return tuple((external_id, values[external_id]) for external_id in candidates[: BENCHMARK_COUNTS[benchmark]])


def materialize_records(
    *,
    config: ExternalConfig,
    download_lock_path: Path,
    raw_root: Path,
    documents_staging_root: Path,
    selection_manifest_path: Path,
    repo_root: Path | None = None,
) -> ExternalSelectionManifest:
    _validate_external_config(config)
    repository = Path(repo_root).absolute() if repo_root is not None else _infer_repo_root(raw_root, kind="raw")
    raw = fixed_external_root(raw_root, kind="raw", repo_root=repository)
    staging = fixed_external_root(
        documents_staging_root, kind="staging", repo_root=repository, create=True
    )
    download_lock = _fixed_file(
        staging,
        kind="staging",
        path=download_lock_path,
        label="download lock",
        require_file=True,
        repo_root=repository,
    )
    selection_path = _fixed_file(
        staging,
        kind="staging",
        path=selection_manifest_path,
        label="selection manifest",
        repo_root=repository,
    )
    lock = _load_download_lock(download_lock)
    rows: list[ExternalSelectionRow] = []
    for benchmark in BENCHMARK_NAMES:
        entry = lock.entry(benchmark)
        spec = config.spec(benchmark)
        if (entry.corpus_version, entry.index_version, entry.adapter_version) != (
            spec.corpus_version,
            config.index_version,
            spec.adapter_version,
        ):
            raise ValueError("external config and download lock disagree")
        for external_id, item in _selected_rows(benchmark, entry, raw):
            task_id = f"ext-{benchmark}-{external_id}"
            records = _as_frozen_records(
                item,
                benchmark=benchmark,
                external_id=external_id,
                task_id=task_id,
                retrieved_at=entry.fetched_at,
            )
            document_path = _safe_child(staging, f"{task_id}.jsonl", label="external staging document")
            payload = b"".join(canonical_json_bytes(record.model_dump(mode="json")) for record in records)
            write_external_json(document_path, payload)
            rows.append(
                ExternalSelectionRow(
                    benchmark=benchmark,
                    external_id=external_id,
                    task_id=task_id,
                    documents_relative_path=f"{task_id}.jsonl",
                    corpus_version=entry.corpus_version,
                    index_version=entry.index_version,
                    upstream_record_sha256=sha256_bytes(canonical_json_bytes(item)),
                )
            )
    manifest = ExternalSelectionManifest(selections=tuple(sorted(rows, key=lambda row: (row.benchmark, row.task_id))))
    write_external_json(selection_path, manifest.model_dump(mode="json"))
    return manifest


def build_snapshots(
    *,
    config: ExternalConfig,
    download_lock_path: Path,
    selection_manifest_path: Path,
    documents_staging_root: Path,
    snapshot_root: Path,
    lock_path: Path,
    repo_root: Path | None = None,
) -> ExternalBenchmarkLock:
    repository = Path(repo_root).absolute() if repo_root is not None else _infer_repo_root(snapshot_root, kind="snapshot")
    staging = fixed_external_root(
        documents_staging_root, kind="staging", repo_root=repository
    )
    snapshots = fixed_external_root(
        snapshot_root, kind="snapshot", repo_root=repository, create=True
    )
    download_lock_file = _fixed_file(
        staging,
        kind="staging",
        path=download_lock_path,
        label="download lock",
        require_file=True,
        repo_root=repository,
    )
    selection_file = _fixed_file(
        staging,
        kind="staging",
        path=selection_manifest_path,
        label="selection manifest",
        require_file=True,
        repo_root=repository,
    )
    final_lock_path = _public_lock_path(
        lock_path, snapshot_root=snapshots, repo_root=repository
    )
    download_lock = _load_download_lock(download_lock_file)
    selection = ExternalSelectionManifest.model_validate(_read_json(selection_file))
    if len(selection.selections) != sum(BENCHMARK_COUNTS.values()):
        raise ValueError("external selection counts are incomplete")
    source_locks: dict[BenchmarkName, ExternalSourceLock] = {
        name: download_lock.entry(name) for name in BENCHMARK_NAMES
    }
    for row in selection.selections:
        source = source_locks[row.benchmark]
        spec = config.spec(row.benchmark)
        if (
            row.corpus_version,
            row.index_version,
        ) != (source.corpus_version, source.index_version) or (
            row.corpus_version,
            row.index_version,
        ) != (spec.corpus_version, config.index_version):
            raise ValueError("external selection/config/source versions disagree")
    built = ExternalSnapshotMaterializer().build(
        selection_manifest_path=selection_file,
        documents_staging_root=staging,
        snapshot_root=snapshots,
        repo_root=repository,
        external_config=config,
        source_locks=source_locks,
    )
    by_benchmark: dict[BenchmarkName, list[object]] = {name: [] for name in BENCHMARK_NAMES}
    for snap in built.snapshots:
        by_benchmark[snap.benchmark].append(snap)
    entries: dict[BenchmarkName, ExternalSourceLock] = {}
    for benchmark in BENCHMARK_NAMES:
        source = download_lock.entry(benchmark)
        expected = BENCHMARK_COUNTS[benchmark]
        if len(by_benchmark[benchmark]) != expected:
            raise ValueError(f"{benchmark} snapshot count is incomplete")
        snapshots = tuple(
            sorted(
                (cast(ExternalSnapshotLock, row) for row in by_benchmark[benchmark]),
                key=lambda row: (row.task_id, row.external_id),
            )
        )
        source = download_lock.entry(benchmark)
        spec = config.spec(benchmark)
        if any(
            snap.corpus_version != source.corpus_version
            or snap.index_version != source.index_version
            or snap.corpus_version != spec.corpus_version
            or snap.index_version != config.index_version
            for snap in snapshots
        ):
            raise ValueError("external snapshot/config/source versions disagree")
        entries[benchmark] = source.model_copy(update={"snapshot_locks": snapshots})
    final = ExternalBenchmarkLock(
        livedrbench=entries["livedrbench"],
        frames=entries["frames"],
        deepresearchbench=entries["deepresearchbench"],
    )
    write_external_json(final_lock_path, final.model_dump(mode="json"))
    return final


def verify_snapshots(
    *,
    lock_path: Path,
    snapshot_root: Path,
    expected_counts: Mapping[BenchmarkName, int] | None = None,
    repo_root: Path | None = None,
) -> dict[str, int]:
    repository = Path(repo_root).absolute() if repo_root is not None else _infer_repo_root(snapshot_root, kind="snapshot")
    root = fixed_external_root(snapshot_root, kind="snapshot", repo_root=repository)
    lock = load_external_lock(
        _public_lock_path(lock_path, snapshot_root=root, repo_root=repository)
    )
    counts = dict(BENCHMARK_COUNTS)
    if expected_counts is not None and dict(expected_counts) != counts:
        raise ValueError("expected external snapshot counts must be canonical 10/20/10")
    actual: dict[str, int] = {}
    for benchmark in BENCHMARK_NAMES:
        snapshots = lock.entry(benchmark).snapshot_locks
        if len(snapshots) != counts[benchmark]:
            raise ValueError(f"{benchmark} snapshot count mismatch")
        for snapshot_lock in snapshots:
            verify_external_snapshot(snapshot_lock, snapshot_root=root)
        actual[benchmark] = len(snapshots)
    return actual


def restore(
    *,
    config: ExternalConfig,
    raw_root: Path,
    documents_staging_root: Path,
    snapshot_root: Path,
    lock_path: Path,
    repo_root: Path | None = None,
) -> dict[str, int]:
    """Rebuild absent ignored snapshots from the committed lock and raw bytes."""

    repository = Path(repo_root).absolute() if repo_root is not None else _infer_repo_root(snapshot_root, kind="snapshot")
    raw = fixed_external_root(raw_root, kind="raw", repo_root=repository)
    staging = fixed_external_root(
        documents_staging_root, kind="staging", repo_root=repository, create=True
    )
    snapshots = fixed_external_root(
        snapshot_root, kind="snapshot", repo_root=repository, create=True
    )
    public_lock = _public_lock_path(
        lock_path, snapshot_root=snapshots, repo_root=repository
    )
    lock = load_external_lock(public_lock)
    rows: list[ExternalSelectionRow] = []
    for benchmark in BENCHMARK_NAMES:
        entry = lock.entry(benchmark)
        spec = config.spec(benchmark)
        if (
            len(entry.snapshot_locks) != BENCHMARK_COUNTS[benchmark]
            or (entry.corpus_version, entry.index_version, entry.adapter_version)
            != (spec.corpus_version, config.index_version, spec.adapter_version)
        ):
            raise ValueError("external lock/config versions or counts are not canonical")
        for snapshot_lock in entry.snapshot_locks:
            if (
                snapshot_lock.corpus_version != entry.corpus_version
                or snapshot_lock.index_version != entry.index_version
            ):
                raise ValueError("external snapshot lock versions disagree with source lock")
            # Rehash raw bytes before deriving ignored documents.  A lock can
            # never cause a different source to be selected silently.
            item = next(
                (
                    row
                    for row in _raw_rows(entry, raw)
                    if _row_id(row) == snapshot_lock.external_id
                ),
                None,
            )
            if item is None:
                raise ValueError(
                    f"external raw payload has no locked task {snapshot_lock.external_id}"
                )
            task_id = snapshot_lock.task_id
            records = _as_frozen_records(
                item,
                benchmark=benchmark,
                external_id=snapshot_lock.external_id,
                task_id=task_id,
                retrieved_at=entry.fetched_at,
            )
            documents_path = _safe_child(staging, f"{task_id}.jsonl", label="external staging document")
            payload = b"".join(
                canonical_json_bytes(record.model_dump(mode="json")) for record in records
            )
            if documents_path.exists():
                if documents_path.read_bytes() != payload:
                    raise ValueError("external staging document hash mismatch")
            else:
                write_external_json(documents_path, payload)
            rows.append(
                ExternalSelectionRow(
                    benchmark=benchmark,
                    external_id=snapshot_lock.external_id,
                    task_id=task_id,
                    documents_relative_path=f"{task_id}.jsonl",
                    corpus_version=entry.corpus_version,
                    index_version=entry.index_version,
                    upstream_record_sha256=sha256_bytes(canonical_json_bytes(item)),
                )
            )
    selection_path = staging / "restore-selection.json"
    if selection_path.is_symlink():
        raise ValueError("restore selection manifest contains a symlink")
    if selection_path.exists():
        selection_path.unlink()
    write_external_json(selection_path, ExternalSelectionManifest(selections=tuple(sorted(rows, key=lambda row: (row.benchmark, row.task_id)))).model_dump(mode="json"))
    # Materializer rejects completed children. Verify every existing child,
    # reject an incomplete one, and invoke Task 3 only for absent children.
    missing: list[str] = []
    for snap in lock.snapshot_locks():
        try:
            verify_external_snapshot(snap, snapshot_root=snapshots)
        except ProviderError:
            child = _safe_child(snapshots, snap.snapshot_relative_path, label="external snapshot")
            if child.exists() or child.is_symlink():
                raise RuntimeError("existing external snapshot is incomplete or invalid")
            missing.append(snap.task_id)
    if missing:
        missing_rows = tuple(row for row in rows if row.task_id in set(missing))
        missing_path = staging / "restore-missing-selection.json"
        if missing_path.is_symlink():
            raise ValueError("restore missing-selection manifest contains a symlink")
        if missing_path.exists():
            missing_path.unlink()
        write_external_json(
            missing_path,
            ExternalSelectionManifest(selections=tuple(sorted(missing_rows, key=lambda row: (row.benchmark, row.task_id)))).model_dump(mode="json"),
        )
        ExternalSnapshotMaterializer().build(
            selection_manifest_path=missing_path,
            documents_staging_root=staging,
            snapshot_root=snapshots,
            repo_root=repository,
            external_config=config,
            source_locks={name: lock.entry(name) for name in BENCHMARK_NAMES},
        )
    return verify_snapshots(
        lock_path=public_lock, snapshot_root=snapshots, repo_root=repository
    )


def download(
    *,
    config: ExternalConfig,
    raw_root: Path,
    output_path: Path,
    repo_root: Path | None = None,
) -> ExternalBenchmarkLock:
    """Create a provisional lock only when immutable local payload metadata exists.

    No URL is fetched here.  A real ingestion tool may prepare the raw files
    and call this function with an explicit ``file://``-free source lock; the
    default reviewable YAML intentionally has no such lock and fails closed.
    """

    # Validate the caller-selected location without creating an ignored data
    # tree as a side effect of a deliberately unavailable download.
    _validate_external_config(config)
    repository = Path(repo_root).absolute() if repo_root is not None else _infer_repo_root(raw_root, kind="raw")
    raw = fixed_external_root(raw_root, kind="raw", repo_root=repository)
    output = _fixed_file(
        raw.parent / "staging",
        kind="staging",
        path=output_path,
        label="download lock",
        repo_root=repository,
    )
    if output.exists() or output.is_symlink():
        raise FileExistsError(output)
    raise RuntimeError(
        "EXTERNAL_DATA_UNAVAILABLE: no immutable local upstream payload is bundled; refusing download"
    )


def _parse_counts(value: str) -> dict[BenchmarkName, int]:
    counts: dict[BenchmarkName, int] = {}
    for item in value.split(","):
        name, _, raw = item.partition("=")
        if name not in BENCHMARK_NAMES or not raw.isdigit():
            raise ValueError("expected counts must use benchmark=count")
        if name in counts:
            raise ValueError("expected counts must not repeat a benchmark")
        counts[name] = int(raw)
    if counts != dict(BENCHMARK_COUNTS):
        raise ValueError("expected counts must be canonical 10/20/10")
    return counts


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.scripts.fetch_external")
    sub = parser.add_subparsers(dest="command", required=True)
    download_parser = sub.add_parser("download")
    download_parser.add_argument("--repo-root", type=Path, required=True)
    download_parser.add_argument("--config", type=Path, required=True)
    download_parser.add_argument("--raw-root", type=Path, required=True)
    download_parser.add_argument("--write-download-lock", type=Path, required=True)
    materialize = sub.add_parser("materialize-records")
    materialize.add_argument("--repo-root", type=Path, required=True)
    materialize.add_argument("--config", type=Path, required=True)
    materialize.add_argument("--download-lock", type=Path, required=True)
    materialize.add_argument("--raw-root", type=Path, required=True)
    materialize.add_argument("--documents-staging-root", type=Path, required=True)
    materialize.add_argument("--write-selection-manifest", type=Path, required=True)
    build = sub.add_parser("build-snapshots")
    build.add_argument("--repo-root", type=Path, required=True)
    build.add_argument("--config", type=Path, required=True)
    build.add_argument("--download-lock", type=Path, required=True)
    build.add_argument("--selection-manifest", type=Path, required=True)
    build.add_argument("--documents-staging-root", type=Path, required=True)
    build.add_argument("--snapshot-root", type=Path, required=True)
    build.add_argument("--write-lock", type=Path, required=True)
    verify = sub.add_parser("verify-snapshots")
    verify.add_argument("--repo-root", type=Path, required=True)
    verify.add_argument("--lock", type=Path, required=True)
    verify.add_argument("--snapshot-root", type=Path, required=True)
    verify.add_argument("--expected-counts", type=str, required=True)
    restore_parser = sub.add_parser("restore")
    restore_parser.add_argument("--repo-root", type=Path, required=True)
    restore_parser.add_argument("--config", type=Path, required=True)
    restore_parser.add_argument("--raw-root", type=Path, required=True)
    restore_parser.add_argument("--documents-staging-root", type=Path, required=True)
    restore_parser.add_argument("--snapshot-root", type=Path, required=True)
    restore_parser.add_argument("--lock", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        repository = Path(args.repo_root).absolute()
        config_path = (
            _fixed_repo_file(
                args.config,
                repo_root=repository,
                relative="benchmarks/configs/external.yaml",
            )
            if hasattr(args, "config")
            else None
        )
        if args.command == "download":
            download(
                config=load_external_config(cast(Path, config_path)),
                raw_root=args.raw_root,
                output_path=args.write_download_lock,
                repo_root=args.repo_root,
            )
        elif args.command == "materialize-records":
            materialize_records(
                config=load_external_config(cast(Path, config_path)),
                download_lock_path=args.download_lock,
                raw_root=args.raw_root,
                documents_staging_root=args.documents_staging_root,
                selection_manifest_path=args.write_selection_manifest,
                repo_root=args.repo_root,
            )
        elif args.command == "build-snapshots":
            build_snapshots(
                config=load_external_config(cast(Path, config_path)),
                download_lock_path=args.download_lock,
                selection_manifest_path=args.selection_manifest,
                documents_staging_root=args.documents_staging_root,
                snapshot_root=args.snapshot_root,
                lock_path=args.write_lock,
                repo_root=args.repo_root,
            )
        elif args.command == "verify-snapshots":
            print(json.dumps(verify_snapshots(lock_path=args.lock, snapshot_root=args.snapshot_root, expected_counts=_parse_counts(args.expected_counts), repo_root=args.repo_root), sort_keys=True))
        elif args.command == "restore":
            print(json.dumps(restore(config=load_external_config(cast(Path, config_path)), raw_root=args.raw_root, documents_staging_root=args.documents_staging_root, snapshot_root=args.snapshot_root, lock_path=args.lock, repo_root=args.repo_root), sort_keys=True))
        return 0
    except (OSError, RuntimeError, TypeError, ValueError, ValidationError) as error:
        print(f"EXTERNAL_FETCH_FAILED: {error}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "ExternalSelectionManifest",
    "ExternalSelectionRow",
    "build_snapshots",
    "download",
    "main",
    "materialize_records",
    "restore",
    "verify_snapshots",
]
