from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import math
import os
import stat
import tempfile
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Annotated, Any, Literal, Self, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from deepresearch.providers import ProviderError, validate_embeddings
from deepresearch.retrieval import normalize_text
from deepresearch.runtime import CancellationToken

from .httpx_transport import await_with_controls, checkpoint

_OFFICIAL_MODEL_ID = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
_OFFICIAL_REVISION = "e62509716f15c5fd03a6fd3156a4bc5e43f83f26"
_OFFICIAL_DIMENSION = 384
_RUNTIME_MODEL_FILES = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "unigram.json",
)
_HEX = frozenset("0123456789abcdef")
_HASH_CHUNK_BYTES = 1024 * 1024


def _canonical_json(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _require_sha256(value: str) -> str:
    if len(value) != 64 or value != value.casefold() or not set(value) <= _HEX:
        raise ValueError("value must be a lowercase SHA-256")
    return value


def _require_revision(value: str) -> str:
    if len(value) != 40 or value != value.casefold() or not set(value) <= _HEX:
        raise ValueError("revision must be a lowercase 40-character commit")
    return value


def _file_hash_and_size(path: Path) -> tuple[str, int]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(_HASH_CHUNK_BYTES):
            digest.update(chunk)
            size += len(chunk)
    return digest.hexdigest(), size


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


class ModelSnapshotUnavailable(ProviderError):
    def __init__(self, public_message: str = "embedding model snapshot is unavailable") -> None:
        super().__init__(
            code="INVALID_SNAPSHOT",
            provider="sentence-transformer",
            operation="embed",
            public_message=public_message,
            retryable=False,
        )


class _LockModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(round_trip=True)
        values.update(update)
        return type(self).model_validate(values)


class EmbeddingModelFile(_LockModel):
    path: str
    sha256: str
    size_bytes: Annotated[int, Field(ge=0)]

    _hash = field_validator("sha256")(_require_sha256)

    @field_validator("path")
    @classmethod
    def validate_relative_path(cls, value: str) -> str:
        path = PurePosixPath(value)
        if (
            not value
            or "\\" in value
            or path.is_absolute()
            or ".." in path.parts
            or "." in path.parts
            or path.as_posix() != value
        ):
            raise ValueError("model file path must be canonical and relative")
        return value


class EmbeddingModelLock(_LockModel):
    schema_version: Literal["embedding-model-lock-v1"]
    model_id: str
    revision: str
    vector_dimension: Annotated[int, Field(gt=0)]
    normalize_embeddings: bool
    files: Annotated[tuple[EmbeddingModelFile, ...], Field(min_length=1)]
    snapshot_sha256: str

    _revision = field_validator("revision")(_require_revision)
    _snapshot = field_validator("snapshot_sha256")(_require_sha256)

    def manifest_payload(self) -> dict[str, object]:
        return {
            "files": [item.model_dump(mode="json") for item in self.files],
            "model_id": self.model_id,
            "normalize_embeddings": self.normalize_embeddings,
            "revision": self.revision,
            "schema_version": self.schema_version,
            "vector_dimension": self.vector_dimension,
        }

    @model_validator(mode="after")
    def validate_manifest(self) -> Self:
        if not self.model_id.strip():
            raise ValueError("model_id must not be empty")
        paths = tuple(item.path for item in self.files)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError("model files must be sorted and unique")
        expected = hashlib.sha256(_canonical_json(self.manifest_payload())).hexdigest()
        if self.snapshot_sha256 != expected:
            raise ValueError("snapshot_sha256 does not match canonical root manifest")
        return self

    @classmethod
    def create(
        cls,
        *,
        model_id: str,
        revision: str,
        vector_dimension: int,
        normalize_embeddings: bool,
        files: Sequence[EmbeddingModelFile],
    ) -> EmbeddingModelLock:
        ordered = tuple(sorted(files, key=lambda item: item.path))
        payload = {
            "files": [item.model_dump(mode="json") for item in ordered],
            "model_id": model_id,
            "normalize_embeddings": normalize_embeddings,
            "revision": revision,
            "schema_version": "embedding-model-lock-v1",
            "vector_dimension": vector_dimension,
        }
        return cls.model_validate(
            {
                **payload,
                "snapshot_sha256": hashlib.sha256(
                    _canonical_json(payload)
                ).hexdigest(),
            }
        )

    @classmethod
    def load(cls, path: Path) -> EmbeddingModelLock:
        try:
            requested = Path(path).absolute()
            if _is_link_or_reparse(requested):
                raise ValueError("embedding lock path is unsafe")
            return cls.model_validate_json(requested.read_bytes())
        except (OSError, TypeError, ValueError, ValidationError):
            pass
        raise ModelSnapshotUnavailable("embedding lock is invalid or unavailable")

    def verify(self, model_root: Path) -> None:
        try:
            requested = Path(model_root).absolute()
            if _is_link_or_reparse(requested):
                raise ValueError("model root is unsafe")
            root = requested.resolve(strict=True)
            if not root.is_dir() or _is_link_or_reparse(root):
                raise ValueError("model root is not a safe directory")
            actual_files: dict[str, Path] = {}
            for path in root.rglob("*"):
                if _is_link_or_reparse(path):
                    raise ValueError("model snapshot contains a link or reparse point")
                if path.is_file():
                    relative = path.relative_to(root).as_posix()
                    actual_files[relative] = path
            expected_paths = {item.path for item in self.files}
            if set(actual_files) != expected_paths:
                raise ValueError("model snapshot files do not match the lock")
            for item in self.files:
                path = actual_files[item.path]
                actual_hash, actual_size = _file_hash_and_size(path)
                if actual_size != item.size_bytes:
                    raise ValueError("model snapshot file size does not match the lock")
                if actual_hash != item.sha256:
                    raise ValueError("model snapshot file hash does not match the lock")
            expected_manifest = hashlib.sha256(
                _canonical_json(self.manifest_payload())
            ).hexdigest()
            if expected_manifest != self.snapshot_sha256:
                raise ValueError("model root manifest does not match the lock")
            return
        except (OSError, RuntimeError, ValueError):
            pass
        raise ModelSnapshotUnavailable()


class DeterministicHashTextEmbedder:
    provider_id = "deterministic-hash"
    model_id = "deterministic-hash-v1"
    model_revision = "1"
    network_calls = 0

    def __init__(self, *, dimension: int = 384) -> None:
        if dimension <= 0:
            raise ValueError("embedding dimension must be positive")
        self.dimension = dimension
        self.snapshot_sha256 = hashlib.sha256(
            f"{self.model_id}:{self.model_revision}:{dimension}".encode()
        ).hexdigest()

    def _vector(self, text: str) -> tuple[float, ...]:
        normalized = normalize_text(text).encode("utf-8")
        values: list[float] = []
        counter = 0
        while len(values) < self.dimension:
            digest = hashlib.sha256(
                normalized + counter.to_bytes(8, "big")
            ).digest()
            for offset in range(0, len(digest), 4):
                integer = int.from_bytes(digest[offset : offset + 4], "big")
                values.append((integer / 0xFFFFFFFF) * 2 - 1)
                if len(values) == self.dimension:
                    break
            counter += 1
        norm = math.sqrt(sum(value * value for value in values)) or 1
        return tuple(value / norm for value in values)

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation="embed",
        )
        if not texts:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="embed",
                public_message="embedding input must not be empty",
                retryable=False,
            )
        vectors: list[tuple[float, ...]] = []
        for text in texts:
            checkpoint(
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation="embed",
            )
            vectors.append(self._vector(text))
            await asyncio.sleep(0)
        return validate_embeddings(texts, vectors)


class SentenceTransformerTextEmbedder:
    provider_id = "sentence-transformer"

    def __init__(self, *, lock: EmbeddingModelLock, model_root: Path) -> None:
        self._lock = lock
        self._model_root = model_root
        self._model: Any | None = None
        self._model_lock = threading.Lock()
        self.model_id = lock.model_id
        self.model_revision = lock.revision
        self.snapshot_sha256 = lock.snapshot_sha256

    @classmethod
    def from_lock(
        cls, lock: EmbeddingModelLock, *, model_root: Path
    ) -> SentenceTransformerTextEmbedder:
        lock.verify(model_root)
        return cls(lock=lock, model_root=Path(model_root).resolve(strict=True))

    def _load_model(self) -> Any:
        with self._model_lock:
            if self._model is None:
                self._lock.verify(self._model_root)
                from sentence_transformers import SentenceTransformer

                self._model = SentenceTransformer(
                    str(self._model_root),
                    local_files_only=True,
                    trust_remote_code=False,
                )
            return self._model

    def _embed_sync(self, texts: list[str]) -> tuple[tuple[float, ...], ...]:
        model = self._load_model()
        raw = model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=self._lock.normalize_embeddings,
            show_progress_bar=False,
        )
        vectors = validate_embeddings(texts, raw)
        if any(len(vector) != self._lock.vector_dimension for vector in vectors):
            raise ValueError("embedding dimension does not match the lock")
        return vectors

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation="embed",
        )
        normalized = [normalize_text(text) for text in texts]
        if not normalized:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="embed",
                public_message="embedding input must not be empty",
                retryable=False,
            )
        vectors: tuple[tuple[float, ...], ...] | None = None
        failed = False
        try:
            vectors = await await_with_controls(
                asyncio.to_thread(self._embed_sync, normalized),
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation="embed",
            )
        except (OSError, RuntimeError, TypeError, ValueError, ValidationError):
            failed = True
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation="embed",
        )
        if failed or vectors is None:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="embed",
                public_message="local embedding model returned an invalid result",
                retryable=False,
            )
        return vectors


RevisionResolver = Callable[[str, str], str]
SnapshotDownloader = Callable[[str, str, Path], None]


def _resolve_revision(model_id: str, revision: str) -> str:
    from huggingface_hub import HfApi

    return cast("str", HfApi().model_info(model_id, revision=revision).sha)


def _download_snapshot(model_id: str, revision: str, destination: Path) -> None:
    from huggingface_hub import snapshot_download  # pyright: ignore[reportUnknownVariableType]

    snapshot_download(
        repo_id=model_id,
        revision=revision,
        local_dir=destination,
        allow_patterns=list(_RUNTIME_MODEL_FILES),
    )
    cache = destination / ".cache"
    if cache.exists():
        _remove_owned_tree(cache, parent=destination)


def _snapshot_files(root: Path) -> tuple[EmbeddingModelFile, ...]:
    files: list[EmbeddingModelFile] = []
    for path in sorted(root.rglob("*")):
        if _is_link_or_reparse(path):
            raise ModelSnapshotUnavailable("downloaded snapshot contains an unsafe link")
        if not path.is_file():
            continue
        file_hash, file_size = _file_hash_and_size(path)
        files.append(
            EmbeddingModelFile(
                path=path.relative_to(root).as_posix(),
                sha256=file_hash,
                size_bytes=file_size,
            )
        )
    if not files:
        raise ModelSnapshotUnavailable("downloaded snapshot is empty")
    return tuple(files)


def _remove_owned_tree(root: Path, *, parent: Path) -> None:
    requested = root.absolute()
    resolved_parent = parent.resolve(strict=True)
    if _is_link_or_reparse(requested) or requested.resolve(strict=True).parent != resolved_parent:
        raise ValueError("refusing to remove an unverified tree")
    children = sorted(requested.rglob("*"), key=lambda path: len(path.parts), reverse=True)
    if any(_is_link_or_reparse(path) for path in children):
        raise ValueError("refusing to remove a tree containing links")
    for path in children:
        if path.is_dir():
            path.rmdir()
        else:
            path.unlink()
    requested.rmdir()


def _path_identity(path: Path) -> tuple[int, int]:
    metadata = path.stat(follow_symlinks=False)
    return metadata.st_dev, metadata.st_ino


def _require_path_identity(path: Path, expected: tuple[int, int]) -> None:
    if _is_link_or_reparse(path) or _path_identity(path) != expected:
        raise ValueError("embedding transaction path was substituted")


def _write_lock(path: Path, lock: EmbeddingModelLock) -> None:
    payload = json.dumps(
        lock.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        indent=2,
    ).encode("utf-8") + b"\n"
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        written = 0
        while written < len(payload):
            count = os.write(descriptor, payload[written:])
            if count <= 0:
                raise OSError("embedding lock write made no progress")
            written += count
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def fetch_and_lock(
    *,
    model_id: str,
    revision: str,
    model_root: Path,
    lock_path: Path,
    replace: bool = False,
    revision_resolver: RevisionResolver = _resolve_revision,
    snapshot_downloader: SnapshotDownloader = _download_snapshot,
) -> EmbeddingModelLock:
    _require_revision(revision)
    requested_root = Path(model_root).absolute()
    requested_lock = Path(lock_path).absolute()
    if (
        requested_root == requested_root.parent
        or requested_lock == requested_lock.parent
        or requested_root == requested_lock
    ):
        raise ValueError("model root and lock must be distinct non-root paths")
    requested_root.parent.mkdir(parents=True, exist_ok=True)
    parent = requested_root.parent.resolve(strict=True)
    if requested_lock.parent.resolve(strict=True) != parent:
        raise ValueError("model root and lock must be siblings")
    if any(_is_link_or_reparse(path) for path in (requested_root, requested_lock)):
        raise ValueError("model root and lock paths must not be links")
    if requested_root.exists() and not replace:
        raise FileExistsError(requested_root)
    existing_lock: EmbeddingModelLock | None = None
    if requested_lock.exists():
        try:
            existing_lock = EmbeddingModelLock.load(requested_lock)
        except ModelSnapshotUnavailable:
            if not replace:
                raise
    resolved = revision_resolver(model_id, revision)
    if resolved != revision:
        raise ModelSnapshotUnavailable("resolved model revision does not match request")
    staging = Path(
        tempfile.mkdtemp(prefix=f".{requested_root.name}.staging.", dir=parent)
    )
    staged_lock = parent / f".{requested_lock.name}.staging.{staging.name.rsplit('.', 1)[-1]}"
    published_root = False
    published_lock = False
    root_backup: Path | None = None
    lock_backup: Path | None = None
    root_backup_identity: tuple[int, int] | None = None
    lock_backup_identity: tuple[int, int] | None = None
    published_root_identity: tuple[int, int] | None = None
    published_lock_identity: tuple[int, int] | None = None
    try:
        snapshot_downloader(model_id, revision, staging)
        lock = EmbeddingModelLock.create(
            model_id=model_id,
            revision=revision,
            vector_dimension=_OFFICIAL_DIMENSION,
            normalize_embeddings=True,
            files=_snapshot_files(staging),
        )
        lock.verify(staging)
        if existing_lock is not None and existing_lock != lock and not replace:
            raise FileExistsError("a different embedding lock already exists")
        write_new_lock = existing_lock != lock or replace
        if write_new_lock:
            _write_lock(staged_lock, lock)
        try:
            suffix = staging.name.rsplit(".", 1)[-1]
            if replace and requested_root.exists():
                root_backup = parent / f".{requested_root.name}.backup.{suffix}"
                root_backup_identity = _path_identity(requested_root)
                os.replace(requested_root, root_backup)
            if replace and requested_lock.exists() and write_new_lock:
                lock_backup = parent / f".{requested_lock.name}.backup.{suffix}"
                lock_backup_identity = _path_identity(requested_lock)
                os.replace(requested_lock, lock_backup)
            staging_identity = _path_identity(staging)
            os.replace(staging, requested_root)
            published_root = True
            published_root_identity = staging_identity
            if write_new_lock:
                staged_lock_identity = _path_identity(staged_lock)
                os.replace(staged_lock, requested_lock)
                published_lock = True
                published_lock_identity = staged_lock_identity
            published = EmbeddingModelLock.load(requested_lock)
            if published != lock:
                raise ModelSnapshotUnavailable(
                    "published embedding lock does not match the staged snapshot"
                )
            published.verify(requested_root)
        except Exception:
            if published_root and requested_root.exists():
                assert published_root_identity is not None
                _require_path_identity(requested_root, published_root_identity)
                _remove_owned_tree(requested_root, parent=parent)
            if published_lock and requested_lock.exists():
                assert published_lock_identity is not None
                _require_path_identity(requested_lock, published_lock_identity)
                requested_lock.unlink()
            if root_backup is not None and root_backup.exists():
                assert root_backup_identity is not None
                _require_path_identity(root_backup, root_backup_identity)
                if requested_root.exists():
                    raise ValueError("embedding root destination changed during rollback")
                os.replace(root_backup, requested_root)
            if lock_backup is not None and lock_backup.exists():
                assert lock_backup_identity is not None
                _require_path_identity(lock_backup, lock_backup_identity)
                if requested_lock.exists():
                    raise ValueError("embedding lock destination changed during rollback")
                os.replace(lock_backup, requested_lock)
            raise

        cleanup_errors: list[Exception] = []
        if root_backup is not None:
            try:
                assert root_backup_identity is not None
                _require_path_identity(root_backup, root_backup_identity)
                _remove_owned_tree(root_backup, parent=parent)
            except (OSError, RuntimeError, ValueError) as error:
                cleanup_errors.append(error)
        if lock_backup is not None:
            try:
                assert lock_backup_identity is not None
                _require_path_identity(lock_backup, lock_backup_identity)
                lock_backup.unlink()
            except (OSError, RuntimeError, ValueError) as error:
                cleanup_errors.append(error)
        if cleanup_errors:
            raise OSError(
                "embedding model committed but backup cleanup failed"
            ) from None
        return lock
    finally:
        if staging.exists():
            _remove_owned_tree(staging, parent=parent)
        if staged_lock.exists():
            staged_lock.unlink()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m deepresearch.providers.embeddings")
    subparsers = parser.add_subparsers(dest="command", required=True)
    fetch = subparsers.add_parser("fetch-and-lock")
    fetch.add_argument("--model-id", required=True)
    fetch.add_argument("--revision", required=True)
    fetch.add_argument("--model-root", type=Path, required=True)
    fetch.add_argument("--lock", type=Path, required=True)
    fetch.add_argument("--replace", action="store_true")
    args = parser.parse_args(argv)
    if args.command != "fetch-and-lock":
        parser.error("unknown command")
    fetch_and_lock(
        model_id=cast("str", args.model_id),
        revision=cast("str", args.revision),
        model_root=cast("Path", args.model_root),
        lock_path=cast("Path", args.lock),
        replace=cast("bool", args.replace),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DeterministicHashTextEmbedder",
    "EmbeddingModelFile",
    "EmbeddingModelLock",
    "ModelSnapshotUnavailable",
    "SentenceTransformerTextEmbedder",
    "fetch_and_lock",
    "main",
]
