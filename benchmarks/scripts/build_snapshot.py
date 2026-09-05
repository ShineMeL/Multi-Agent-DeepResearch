from __future__ import annotations

import argparse
import ctypes
import errno
import hashlib
import json
import os
import shutil
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol, cast

from pydantic import ValidationError

from benchmarks.datasets.models import AnnotatedQuestion, FrozenEvidenceRecord
from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_index import (
    FrozenBm25Index,
    FrozenCorpusManifest,
    FrozenCorpusSnapshot,
)


class SnapshotBuildError(RuntimeError):
    """Raised when an immutable frozen-corpus snapshot cannot be built."""


class _RenameAt2(Protocol):
    argtypes: object
    restype: object

    def __call__(
        self,
        old_directory: int,
        old_path: bytes,
        new_directory: int,
        new_path: bytes,
        flags: int,
    ) -> int: ...


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        + b"\n"
    )


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _safe_task_id(value: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError("task_id must be non-empty")
    if value in {".", ".."} or Path(value).name != value:
        raise ValueError("task_id must be a single path component")
    return value


def _safe_version(value: str, *, label: str) -> str:
    if type(value) is not str or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    return value


def _read_records(path: Path, *, task_id: str) -> tuple[FrozenEvidenceRecord, ...]:
    """Read and canonicalize one private FrozenEvidenceRecord JSONL file."""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"documents file does not exist: {source}")
    raw = source.read_bytes()
    records: list[FrozenEvidenceRecord] = []
    for line_number, line in enumerate(raw.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank line in documents file at line {line_number}")
        try:
            record = FrozenEvidenceRecord.model_validate_json(line, strict=True)
        except (TypeError, ValueError, ValidationError) as error:
            raise ValueError(
                f"invalid FrozenEvidenceRecord at {source}:{line_number}"
            ) from error
        if record.task_id != task_id:
            raise ValueError(
                f"documents task_id mismatch at {source}:{line_number}: {record.task_id}"
            )
        records.append(record)
    return tuple(sorted(records, key=lambda item: (item.evidence_id, item.source_id)))


def _documents_bytes(records: Sequence[FrozenEvidenceRecord]) -> bytes:
    return b"".join(
        _canonical_json(record.model_dump(mode="json")) for record in records
    )


def _snapshot_id(
    *,
    task_id: str,
    corpus_version: str,
    index_version: str,
    documents_sha256: str,
    index_sha256: str,
) -> str:
    identity = {
        "corpus_version": corpus_version,
        "documents_sha256": documents_sha256,
        "index_version": index_version,
        "index_sha256": index_sha256,
        "task_id": task_id,
    }
    return f"snapshot-{_sha256(_canonical_json(identity))}"


def _write_exclusive(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _fsync_directory(path: Path) -> None:
    """Best-effort directory durability for platforms that expose fsync."""

    flags = getattr(os, "O_DIRECTORY", 0)
    try:
        descriptor = os.open(path, os.O_RDONLY | flags)
    except (OSError, TypeError):
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _publish_no_replace(staging: Path, output: Path) -> None:
    """Atomically publish a directory without allowing a replacement race."""

    if os.name == "nt":
        # Windows MoveFile semantics used by os.rename reject an existing
        # target, including a target created after our initial preflight.
        os.rename(staging, output)
        return
    if sys.platform.startswith("linux"):
        library = ctypes.CDLL(None, use_errno=True)
        try:
            rename_at_2 = cast("_RenameAt2", library.renameat2)
        except AttributeError as error:
            raise OSError(errno.ENOSYS, "renameat2 is required for no-replace publish") from error
        rename_at_2.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        rename_at_2.restype = ctypes.c_int
        result = rename_at_2(
            -100,  # AT_FDCWD
            os.fsencode(staging),
            -100,  # AT_FDCWD
            os.fsencode(output),
            1,  # RENAME_NOREPLACE
        )
        if result == 0:
            return
        error_number = ctypes.get_errno()
        if error_number in {errno.EEXIST, errno.ENOTEMPTY}:
            raise FileExistsError(error_number, "snapshot output already exists", output)
        raise OSError(error_number, os.strerror(error_number), output)
    raise OSError(errno.ENOTSUP, "atomic no-replace directory publish is unavailable")


def _build_into_staging(
    *,
    task_id: str,
    documents: Path,
    staging: Path,
    corpus_version: str,
    index_version: str,
) -> FrozenCorpusManifest:
    records = _read_records(documents, task_id=task_id)
    index = FrozenBm25Index.build(records, index_version=index_version)
    documents_payload = _documents_bytes(records)
    index_payload = _canonical_json(index.to_payload())
    documents_sha256 = _sha256(documents_payload)
    index_sha256 = _sha256(index_payload)
    manifest = FrozenCorpusManifest(
        snapshot_id=_snapshot_id(
            task_id=task_id,
            corpus_version=corpus_version,
            index_version=index_version,
            documents_sha256=documents_sha256,
            index_sha256=index_sha256,
        ),
        task_id=task_id,
        corpus_version=corpus_version,
        index_version=index_version,
        document_count=len(records),
        documents_sha256=documents_sha256,
        index_sha256=index_sha256,
    )
    snapshot_payload = _canonical_json(manifest.model_dump(mode="json"))
    manifest_payload = _canonical_json(
        {
            "file_sha256": {
                "documents.jsonl": documents_sha256,
                "index.json": index_sha256,
                "snapshot.json": _sha256(snapshot_payload),
            }
        }
    )
    _write_exclusive(staging / "documents.jsonl", documents_payload)
    _write_exclusive(staging / "index.json", index_payload)
    _write_exclusive(staging / "snapshot.json", snapshot_payload)
    _write_exclusive(staging / "manifest.sha256", manifest_payload)
    _fsync_directory(staging)

    # Reload the exact bytes before publication.  This catches serialization,
    # cross-record consistency and locator/hash mistakes while still isolated.
    loaded = FrozenCorpusSnapshot.load(staging, task_id=task_id)
    if loaded.manifest != manifest:
        raise SnapshotBuildError("self-verification returned a different manifest")
    return manifest


def build_one(
    *,
    task_id: str,
    documents: Path,
    output: Path,
    corpus_version: str,
    index_version: str,
) -> FrozenCorpusManifest:
    """Build one immutable task snapshot and return its verified manifest."""

    task_id = _safe_task_id(task_id)
    corpus_version = _safe_version(corpus_version, label="corpus_version")
    index_version = _safe_version(index_version, label="index_version")
    output_path = Path(output).absolute()
    if not output_path.name or output_path.name in {".", ".."}:
        raise ValueError("output must name a snapshot directory")
    if _path_exists(output_path):
        raise FileExistsError(f"snapshot output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    staging = output_path.parent / (
        f".{output_path.name}.{os.getpid()}.{uuid.uuid4().hex}.staging"
    )
    if _path_exists(staging):
        raise FileExistsError(f"snapshot staging path already exists: {staging}")
    staging.mkdir()
    try:
        manifest = _build_into_staging(
            task_id=task_id,
            documents=Path(documents),
            staging=staging,
            corpus_version=corpus_version,
            index_version=index_version,
        )
        _publish_no_replace(staging, output_path)
        _fsync_directory(output_path.parent)
        return manifest
    except Exception:
        if _path_exists(staging):
            shutil.rmtree(staging)
        raise


def _read_batch(path: Path) -> tuple[AnnotatedQuestion, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"batch file does not exist: {source}")
    questions: list[AnnotatedQuestion] = []
    for line_number, line in enumerate(source.read_bytes().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank line in batch file at line {line_number}")
        try:
            question = AnnotatedQuestion.model_validate_json(line, strict=True)
        except (TypeError, ValueError, ValidationError) as error:
            raise ValueError(f"invalid AnnotatedQuestion at {source}:{line_number}") from error
        questions.append(question)
    task_ids = [question.task_id for question in questions]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("batch contains duplicate task_id")
    return tuple(sorted(questions, key=lambda item: item.task_id))


def build_batch(
    *,
    batch: Path,
    documents_root: Path,
    output_root: Path,
    corpus_version: str,
    index_version: str,
) -> tuple[FrozenCorpusManifest, ...]:
    """Build every task in lexical order, preserving completed children on failure."""

    corpus_version = _safe_version(corpus_version, label="corpus_version")
    index_version = _safe_version(index_version, label="index_version")
    questions = _read_batch(Path(batch))
    document_root = Path(documents_root).absolute()
    if not document_root.is_dir():
        raise FileNotFoundError(f"documents root does not exist: {document_root}")
    destination_root = Path(output_root).absolute()
    destination_root.mkdir(parents=True, exist_ok=True)
    manifests: list[FrozenCorpusManifest] = []
    for question in questions:
        task_id = _safe_task_id(question.task_id)
        if question.corpus_version != corpus_version:
            raise ValueError(f"task {task_id}: corpus_version does not match batch option")
        if question.index_version != index_version:
            raise ValueError(f"task {task_id}: index_version does not match batch option")
        documents = document_root / f"{task_id}.jsonl"
        output = destination_root / task_id
        try:
            manifests.append(
                build_one(
                    task_id=task_id,
                    documents=documents,
                    output=output,
                    corpus_version=corpus_version,
                    index_version=index_version,
                )
            )
        except Exception as error:
            raise SnapshotBuildError(f"task {task_id} failed: {error}") from error
    return tuple(manifests)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.scripts.build_snapshot")
    subparsers = parser.add_subparsers(dest="command", required=True)
    one = subparsers.add_parser("one", help="build one task snapshot")
    one.add_argument("--task-id", required=True)
    one.add_argument("--documents", type=Path, required=True)
    one.add_argument("--output", type=Path, required=True)
    one.add_argument("--corpus-version", required=True)
    one.add_argument("--index-version", required=True)
    batch = subparsers.add_parser("batch", help="build snapshots for an annotated batch")
    batch.add_argument("--batch", type=Path, required=True)
    batch.add_argument("--documents-root", type=Path, required=True)
    batch.add_argument("--output-root", type=Path, required=True)
    batch.add_argument("--corpus-version", required=True)
    batch.add_argument("--index-version", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        if args.command == "one":
            build_one(
                task_id=args.task_id,
                documents=args.documents,
                output=args.output,
                corpus_version=args.corpus_version,
                index_version=args.index_version,
            )
        elif args.command == "batch":
            build_batch(
                batch=args.batch,
                documents_root=args.documents_root,
                output_root=args.output_root,
                corpus_version=args.corpus_version,
                index_version=args.index_version,
            )
        else:  # pragma: no cover - argparse enforces the subcommand
            return 2
        return 0
    except (OSError, ProviderError, SnapshotBuildError, TypeError, ValueError, ValidationError) as error:
        print(f"SNAPSHOT_BUILD_FAILED: {error}", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["SnapshotBuildError", "build_batch", "build_one", "main"]
