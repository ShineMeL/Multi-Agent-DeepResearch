from __future__ import annotations

import base64
import hashlib
import json
import os
import re
import stat
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import Annotated, Self, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_ID_PATTERN = re.compile(r"sha256:([0-9a-f]{64})")


class ArtifactIntegrityError(ValueError):
    pass


class ArtifactConflictError(ArtifactIntegrityError):
    pass


class _UnsafeStoragePathError(ValueError):
    pass


def _is_link_or_reparse(path: Path) -> bool:
    try:
        metadata = os.lstat(path)
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    file_attributes = getattr(metadata, "st_file_attributes", 0)
    is_junction = getattr(path, "is_junction", lambda: False)
    return stat.S_ISLNK(metadata.st_mode) or bool(file_attributes & reparse_flag) or is_junction()


def _ensure_safe_directory(root: Path, directory: Path) -> None:
    try:
        relative = directory.absolute().relative_to(root)
    except ValueError as error:
        raise _UnsafeStoragePathError("storage path is not contained by its configured root") from error
    current = root
    for component in relative.parts:
        current /= component
        if _is_link_or_reparse(current):
            raise _UnsafeStoragePathError("storage path contains a symlink or reparse point")
        try:
            current.mkdir()
        except FileExistsError:
            pass
        if _is_link_or_reparse(current):
            raise _UnsafeStoragePathError("storage path contains a symlink or reparse point")
        if not current.is_dir():
            raise _UnsafeStoragePathError("storage path component is not a directory")
        try:
            current.resolve(strict=True).relative_to(root)
        except (OSError, ValueError) as error:
            raise _UnsafeStoragePathError(
                "resolved storage path is not contained by its configured root"
            ) from error


def _ensure_safe_file_path(root: Path, path: Path) -> None:
    _ensure_safe_directory(root, path.parent)
    if _is_link_or_reparse(path):
        raise _UnsafeStoragePathError("storage final or lock path is a symlink or reparse point")


def _try_advisory_lock(descriptor: int) -> bool:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        try:
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl  # pyright: ignore[reportMissingImports]

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        return False
    return True


def _release_advisory_lock(descriptor: int) -> None:
    os.lseek(descriptor, 0, os.SEEK_SET)
    if os.name == "nt":
        import msvcrt

        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl  # pyright: ignore[reportMissingImports]

    fcntl.flock(descriptor, fcntl.LOCK_UN)


class ArtifactRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    artifact_id: str
    sha256: str
    media_type: str
    size_bytes: Annotated[int, Field(ge=0)]

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if _SHA256_PATTERN.fullmatch(self.sha256) is None:
            raise ValueError("sha256 must be a lowercase 64-character SHA-256")
        if self.artifact_id != f"sha256:{self.sha256}":
            raise ValueError("artifact_id must equal sha256:<sha256>")
        if not self.media_type or self.media_type != self.media_type.strip():
            raise ValueError("media_type must be non-empty and trimmed")
        return self


def _digest_from_artifact_id(artifact_id: str) -> str:
    match = _ARTIFACT_ID_PATTERN.fullmatch(artifact_id)
    if match is None:
        raise ArtifactIntegrityError("artifact_id must be a content-addressed SHA-256 ID")
    return match.group(1)


@contextmanager
def _key_lock(lock_path: Path, *, timeout_seconds: float = 10.0) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    if _is_link_or_reparse(lock_path):
        raise _UnsafeStoragePathError("storage lock path is a symlink or reparse point")
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        while not acquired:
            acquired = _try_advisory_lock(descriptor)
            if not acquired:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for storage lock {lock_path.name}")
                time.sleep(0.005)
        yield
    finally:
        if acquired:
            try:
                _release_advisory_lock(descriptor)
            except OSError:
                pass
        os.close(descriptor)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


class LocalArtifactStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._artifact_root = self._root / "artifacts"
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            _ensure_safe_directory(self._root, self._artifact_root)
        except _UnsafeStoragePathError as error:
            raise ArtifactIntegrityError(str(error)) from error

    def _path_for_digest(self, digest: str) -> Path:
        if _SHA256_PATTERN.fullmatch(digest) is None:
            raise ArtifactIntegrityError("artifact digest is invalid")
        return self._artifact_root / digest[:2] / f"{digest}.json"

    def _read(self, artifact_id: str) -> tuple[ArtifactRef, bytes]:
        digest = _digest_from_artifact_id(artifact_id)
        path = self._path_for_digest(digest)
        try:
            _ensure_safe_file_path(self._root, path)
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("artifact envelope is not an object")
            envelope = cast("dict[str, object]", loaded)
            if set(envelope) != {"payload", "payload_sha256"}:
                raise TypeError("artifact envelope fields are invalid")
            raw_payload = envelope["payload"]
            if not isinstance(raw_payload, dict):
                raise TypeError("artifact payload envelope is not an object")
            payload_record = cast("dict[str, object]", raw_payload)
            checksum = envelope["payload_sha256"]
            canonical_payload = json.dumps(
                payload_record,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if not isinstance(checksum, str) or hashlib.sha256(canonical_payload).hexdigest() != checksum:
                raise ArtifactIntegrityError("artifact record payload hash is corrupt")
            ref = ArtifactRef.model_validate(payload_record.get("ref"))
            encoded = payload_record.get("data_base64")
            if not isinstance(encoded, str):
                raise TypeError("artifact payload is not base64 text")
            data = base64.b64decode(encoded, validate=True)
        except FileNotFoundError:
            raise
        except ArtifactIntegrityError:
            raise
        except Exception as error:
            raise ArtifactIntegrityError("artifact record is corrupt") from error
        actual_digest = hashlib.sha256(data).hexdigest()
        if ref.artifact_id != artifact_id or ref.sha256 != digest:
            raise ArtifactIntegrityError("artifact identity metadata does not match its key")
        if actual_digest != ref.sha256:
            raise ArtifactIntegrityError("artifact payload hash does not match metadata")
        if len(data) != ref.size_bytes:
            raise ArtifactIntegrityError("artifact payload size does not match metadata")
        return ref, data

    def put_bytes(self, data: bytes, *, media_type: str) -> ArtifactRef:
        digest = hashlib.sha256(data).hexdigest()
        ref = ArtifactRef(
            artifact_id=f"sha256:{digest}",
            sha256=digest,
            media_type=media_type,
            size_bytes=len(data),
        )
        path = self._path_for_digest(digest)
        lock_path = path.with_suffix(".lock")
        artifact_payload = {
            "data_base64": base64.b64encode(data).decode("ascii"),
            "ref": ref.model_dump(mode="json"),
        }
        canonical_payload = json.dumps(
            artifact_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        record = {
            "payload": artifact_payload,
            "payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        }
        payload = json.dumps(
            record,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        try:
            _ensure_safe_file_path(self._root, path)
            _ensure_safe_file_path(self._root, lock_path)
            with _key_lock(lock_path):
                _ensure_safe_file_path(self._root, path)
                if path.exists():
                    existing_ref, existing_data = self._read(ref.artifact_id)
                    if existing_data != data or existing_ref.media_type != media_type:
                        raise ArtifactConflictError(
                            "artifact ID already has different content or metadata"
                        )
                    return existing_ref
                _ensure_safe_file_path(self._root, path)
                _atomic_write_bytes(path, payload)
        except _UnsafeStoragePathError as error:
            raise ArtifactIntegrityError(str(error)) from error
        return ref

    def get_bytes(self, artifact_id: str) -> bytes:
        return self._read(artifact_id)[1]

    def exists(self, artifact_id: str) -> bool:
        try:
            digest = _digest_from_artifact_id(artifact_id)
        except ArtifactIntegrityError:
            return False
        path = self._path_for_digest(digest)
        try:
            _ensure_safe_file_path(self._root, path)
        except _UnsafeStoragePathError as error:
            raise ArtifactIntegrityError(str(error)) from error
        return path.is_file()


__all__ = [
    "ArtifactConflictError",
    "ArtifactIntegrityError",
    "ArtifactRef",
    "LocalArtifactStore",
]
