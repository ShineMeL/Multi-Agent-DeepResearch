from __future__ import annotations

import asyncio
import base64
import hashlib
import json
import os
import re
import stat
import sys
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from math import isfinite
from pathlib import Path
from typing import Any, TypeVar, cast
from urllib.parse import unquote, urlsplit

from pydantic import JsonValue

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    Fetcher,
    ModelProvider,
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
    RawDocument,
    SearchHit,
    SearchProvider,
    StructuredModelResult,
    TextEmbedder,
    validate_model_stream,
)
from deepresearch.retrieval import canonicalize_url
from deepresearch.runtime import CancellationToken
from deepresearch.storage import FetchCacheKey

from .replay_schema import (
    REPLAY_BUNDLE_SCHEMA_VERSION,
    REPLAY_FILES,
    REPLAY_MANIFEST_SCHEMA_VERSION,
    REPLAY_OPERATIONS,
    REPLAY_REQUEST_SCHEMA_VERSION,
    ReplayBundle,
    ReplayFailure,
    ReplayOperation,
    ReplayProviderSnapshot,
    ReplayRecord,
    ReplayRequestKey,
    ReplaySuccess,
    canonical_json_bytes,
    canonical_request_sha256,
    embed_request_payload,
    fetch_request_payload,
    model_request_payload,
    search_request_payload,
)

T = TypeVar("T")
_RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")
_SAFE_FETCH_HEADERS = frozenset(
    {
        "cache-control",
        "content-language",
        "content-length",
        "content-type",
        "etag",
        "last-modified",
    }
)
_INVALID_PERCENT_ESCAPE = re.compile(r"%(?![0-9A-Fa-f]{2})")
_CREDENTIAL_WORD = re.compile(
    r"(?i)(?:^|[^a-z0-9])"
    r"(?:api[_ .-]?key|access[_ .-]?key|authorization|auth|bearer|cookie|"
    r"credential|password|passwd|secret|private[_ .-]?key|"
    r"session(?:[_ .-]?(?:id|token))?|access[_ .-]?token|token|key)"
    r"(?:$|[^a-z0-9])"
)
_OPAQUE_CREDENTIAL = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:"
    r"sk[-_](?:(?:proj|live|test)[-_])?[a-z0-9_-]{6,}|"
    r"(?:gh[opsu]|github_pat)_[a-z0-9_]{6,}|"
    r"xox[baprs]-[a-z0-9-]{6,}|"
    r"akia[0-9a-z]{12,}|"
    r"eyj[a-z0-9_-]{6,}\.[a-z0-9_-]{6,}\.[a-z0-9_-]{6,}"
    r")(?:$|[^a-z0-9])"
)
_SECRET_PATH_NAMES = frozenset(
    {
        "accesskey",
        "accesstoken",
        "apikey",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "password",
        "passwd",
        "privatekey",
        "secret",
        "sessionid",
        "sessiontoken",
        "token",
    }
)
_RECORD_FILENAME: dict[ReplayOperation, str] = {
    "model.complete": "model_responses.jsonl",
    "model.structured": "model_responses.jsonl",
    "model.stream": "model_responses.jsonl",
    "search": "search.jsonl",
    "fetch": "documents.jsonl",
    "embed": "embeddings.jsonl",
}


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _path_identity(path: Path) -> tuple[int, int]:
    details = path.stat(follow_symlinks=False)
    if not stat.S_ISDIR(details.st_mode):
        raise ValueError("replay staging object must be a directory")
    return (details.st_dev, details.st_ino)


def _remove_exact_created_staging(
    staging: Path,
    *,
    parent: Path,
    expected_identity: tuple[int, int],
    marker_name: str,
) -> None:
    """Remove only the exact directory object created by this create() call."""
    if _is_link_or_reparse(staging):
        raise ValueError("created replay staging was replaced by a link")
    if staging.resolve(strict=True).parent != parent:
        raise ValueError("created replay staging escaped its parent")
    if _path_identity(staging) != expected_identity:
        raise ValueError("created replay staging object was substituted")
    children = tuple(staging.iterdir())
    if any(
        child.name != marker_name or _is_link_or_reparse(child) or child.is_dir()
        for child in children
    ):
        raise ValueError("created replay staging contains an unexpected object")
    for child in children:
        child.unlink()
    staging.rmdir()


def _try_lock(descriptor: int) -> bool:
    if os.name == "nt":
        import msvcrt

        try:
            os.lseek(descriptor, 0, os.SEEK_SET)
            msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
        except OSError:
            return False
        return True
    import fcntl

    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        return False
    return True


def _unlock(descriptor: int) -> None:
    if os.name == "nt":
        import msvcrt

        os.lseek(descriptor, 0, os.SEEK_SET)
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
        return
    import fcntl

    fcntl.flock(descriptor, fcntl.LOCK_UN)


def _write_fsynced(path: Path, payload: bytes) -> None:
    flags = (
        os.O_CREAT
        | os.O_EXCL
        | os.O_WRONLY
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_BINARY", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    try:
        view = memoryview(payload)
        while view:
            written = os.write(descriptor, view)
            view = view[written:]
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        _flush_windows_directory(path)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _flush_windows_directory(path: Path) -> None:
    """Flush directory metadata using a native backup-semantics handle."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path),
        0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    flush_succeeded = False
    try:
        if not kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
        flush_succeeded = True
    finally:
        if not kernel32.CloseHandle(handle) and flush_succeeded:
            raise ctypes.WinError(ctypes.get_last_error())


def _atomic_rename_noreplace_impl(source: Path, destination: Path) -> None:
    """Atomically publish a directory and fail if destination already exists."""
    if os.name == "nt":
        os.rename(source, destination)
        return
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise NotImplementedError("atomic no-replace rename is unavailable")
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise NotImplementedError("atomic no-replace rename is unavailable")
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        raise NotImplementedError("atomic no-replace rename is unsupported")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Publication boundary kept separate for portable fault injection."""
    _atomic_rename_noreplace_impl(source, destination)


def _atomic_move_owned_staging(source: Path, destination: Path) -> None:
    """Cleanup boundary kept separate from final publication."""
    _atomic_rename_noreplace_impl(source, destination)


def _redact_public_message(message: str) -> str:
    redacted = re.sub(r"(?i)\bbearer\s+[^\s,;]+", "Bearer [REDACTED]", message)
    redacted = re.sub(
        r"(?i)\b(api[_-]?key|authorization|cookie|password|secret|access[_-]?token)"
        r"\s*[:=]\s*[^\s,;]+",
        lambda match: f"{match.group(1)}=[REDACTED]",
        redacted,
    )
    return redacted


def _bounded_unquote(value: str) -> str:
    current = value
    for _ in range(6):
        if _INVALID_PERCENT_ESCAPE.search(current):
            raise ValueError("metadata encoding is invalid")
        decoded = unquote(current, encoding="utf-8", errors="strict")
        if decoded == current:
            return current
        current = decoded
    raise ValueError("metadata encoding exceeds the safety bound")


def _contains_credential_shape(value: str) -> bool:
    decoded = _bounded_unquote(value).casefold()
    return (
        _CREDENTIAL_WORD.search(decoded) is not None
        or _OPAQUE_CREDENTIAL.search(decoded) is not None
    )


def _url_metadata_is_safe(url: str) -> bool:
    decoded_url = _bounded_unquote(url)
    split = urlsplit(decoded_url)
    if split.username is not None or split.password is not None:
        return False
    if _contains_credential_shape(split.query) or _contains_credential_shape(
        split.fragment
    ):
        return False
    for segment in split.path.split("/"):
        compact = re.sub(r"[^a-z0-9]", "", segment.casefold())
        if compact in _SECRET_PATH_NAMES or _OPAQUE_CREDENTIAL.search(segment):
            return False
    canonical_url = canonicalize_url(url)
    FetchCacheKey.model_validate(
        {
            "snapshot_id": "replay-recorded-url",
            "canonical_url": canonical_url,
            "fetch_policy": "recorded-redirect",
            "accepted_content_types": (),
        }
    )
    return True


def _fetch_metadata_is_safe(
    requested_url: str, final_url: str, headers: Mapping[str, str]
) -> bool:
    try:
        return (
            _url_metadata_is_safe(requested_url)
            and _url_metadata_is_safe(final_url)
            and not any(
                _contains_credential_shape(value) for value in headers.values()
            )
        )
    except (TypeError, UnicodeError, ValueError):
        return False


def _safe_fetch_response(document: RawDocument) -> dict[str, JsonValue]:
    headers = {
        key.casefold(): value
        for key, value in document.headers.items()
        if key.casefold() in _SAFE_FETCH_HEADERS
    }
    requested_url = str(document.requested_url)
    final_url = str(document.final_url)
    if not _fetch_metadata_is_safe(requested_url, final_url, headers):
        raise ValueError("recorded fetch metadata violates credential-safe policy")
    return {
        "body_base64": base64.b64encode(document.body_bytes).decode("ascii"),
        "content_type": document.content_type,
        "final_url": final_url,
        "headers": cast("dict[str, JsonValue]", headers),
        "requested_url": requested_url,
        "retrieved_at": document.retrieved_at.isoformat().replace("+00:00", "Z"),
        "status": document.status,
    }


def _operation_usage(operation: ReplayOperation, latency_ms: int) -> ResourceUsage:
    return ResourceUsage.zero().model_copy(
        update={
            "search_calls": 1 if operation == "search" else 0,
            "pages": 1 if operation == "fetch" else 0,
            "wall_seconds": latency_ms / 1000,
        }
    )


class ReplayBundleWriter:
    _MARKER_NAME = ".replay-writer-owner.json"

    def __init__(
        self,
        *,
        final_root: Path,
        staging_root: Path,
        run_id: str,
        staging_identity: tuple[int, int],
        lock_path: Path,
        lock_descriptor: int,
    ) -> None:
        self._final_root = final_root
        self._staging_root: Path | None = staging_root
        self._run_id = run_id
        self._staging_identity = staging_identity
        self._lock_path = lock_path
        self._lock_descriptor: int | None = lock_descriptor
        self._records: dict[tuple[str, str, str, str | None, str], ReplayRecord] = {}
        self._providers: dict[str, ReplayProviderSnapshot] = {}
        self._append_lock = asyncio.Lock()
        self._closed = False
        self._published = False
        self._cleanup_tombstone: Path | None = None

    @classmethod
    def create(cls, final_root: Path, *, run_id: str) -> ReplayBundleWriter:
        if _RUN_ID.fullmatch(run_id) is None:
            raise ValueError("run_id must be a safe non-empty identifier")
        requested_final = Path(final_root).absolute()
        parent = requested_final.parent.resolve(strict=True)
        if _is_link_or_reparse(parent) or _is_link_or_reparse(requested_final):
            raise ValueError("replay output path must not use symlinks or reparse points")
        if requested_final.parent.resolve(strict=True) != parent:
            raise ValueError("replay output path escapes its parent")
        if requested_final.exists():
            raise FileExistsError(requested_final)

        lock_path = parent / f".{requested_final.name}.lock"
        if _is_link_or_reparse(lock_path):
            raise ValueError("replay lock path must not be a symlink or reparse point")
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        staging: Path | None = None
        staging_identity: tuple[int, int] | None = None
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            if not _try_lock(descriptor):
                raise BlockingIOError("another replay writer owns the destination lock")
            staging = parent / f".{requested_final.name}.staging.{run_id}"
            if _is_link_or_reparse(staging):
                raise ValueError("replay staging path must not be a symlink or reparse point")
            staging.mkdir(mode=0o700)
            staging_identity = _path_identity(staging)
            marker = {
                "final_name": requested_final.name,
                "run_id": run_id,
                "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
            }
            _write_fsynced(staging / cls._MARKER_NAME, canonical_json_bytes(marker))
            _fsync_directory(staging)
        except Exception:
            cleanup_error: Exception | None = None
            if staging is not None and staging_identity is not None:
                try:
                    _remove_exact_created_staging(
                        staging,
                        parent=parent,
                        expected_identity=staging_identity,
                        marker_name=cls._MARKER_NAME,
                    )
                    _fsync_directory(parent)
                except (OSError, RuntimeError, ValueError) as error:
                    cleanup_error = error
            try:
                _unlock(descriptor)
            except OSError:
                pass
            os.close(descriptor)
            if cleanup_error is not None:
                raise OSError(
                    "replay writer creation failed and staging cleanup failed"
                ) from cleanup_error
            raise
        return cls(
            final_root=requested_final,
            staging_root=staging,
            run_id=run_id,
            staging_identity=staging_identity,
            lock_path=lock_path,
            lock_descriptor=descriptor,
        )

    def register_provider(
        self,
        kind: str,
        *,
        provider_id: str,
        model_id: str | None = None,
        model_revision: str | None = None,
        snapshot_sha256: str | None = None,
    ) -> None:
        if self._closed:
            raise RuntimeError("replay writer is closed")
        snapshot = ReplayProviderSnapshot(
            provider_id=provider_id,
            model_id=model_id,
            model_revision=model_revision,
            snapshot_sha256=snapshot_sha256,
        )
        existing = self._providers.get(kind)
        if existing is not None and existing != snapshot:
            raise ValueError(f"conflicting provider metadata for {kind}")
        self._providers[kind] = snapshot

    def configure_model_provider(
        self, *, provider_id: str, model_revision: str
    ) -> None:
        """Register immutable model identity before constructing its recorder."""
        self.register_provider(
            "model", provider_id=provider_id, model_revision=model_revision
        )

    def model_revision_for(self, provider_id: str) -> str:
        configured = self._providers.get("model")
        if (
            configured is None
            or configured.provider_id != provider_id
            or configured.model_revision is None
        ):
            raise ValueError(
                "recording model provider revision configuration is missing or conflicting"
            )
        return configured.model_revision

    async def append(
        self,
        *,
        key: ReplayRequestKey,
        outcome: ReplaySuccess | ReplayFailure,
        usage: ResourceUsage,
        latency_ms: int,
    ) -> None:
        async with self._append_lock:
            if self._closed or self._staging_root is None:
                raise RuntimeError("replay writer is closed")
            record = ReplayRecord(
                key=key,
                outcome=outcome,
                usage=usage,
                latency_ms=latency_ms,
                outcome_sha256=canonical_request_sha256(outcome.model_dump(mode="json")),
            )
            identity = key.identity()
            if identity in self._records:
                raise ValueError("duplicate replay request key")
            self._records[identity] = record

    async def finalize(self) -> Path:
        async with self._append_lock:
            if self._closed or self._staging_root is None:
                raise RuntimeError("replay writer is closed")
            try:
                staging = self._validate_owned_staging()
                if self._final_root.exists():
                    raise FileExistsError(self._final_root)

                grouped: dict[str, list[ReplayRecord]] = {
                    filename: [] for filename in _RECORD_FILENAME.values()
                }
                for record in self._records.values():
                    grouped[_RECORD_FILENAME[record.key.operation]].append(record)
                for records in grouped.values():
                    records.sort(
                        key=lambda item: canonical_json_bytes(
                            item.key.model_dump(mode="json")
                        )
                    )

                for filename in sorted(grouped):
                    lines = [
                        canonical_json_bytes(item.model_dump(mode="json"))
                        for item in grouped[filename]
                    ]
                    payload = b"" if not lines else b"\n".join(lines) + b"\n"
                    _write_fsynced(staging / filename, payload)
                snapshot = {
                    "providers": {
                        key: value.model_dump(mode="json")
                        for key, value in sorted(self._providers.items())
                    },
                    "run_id": self._run_id,
                    "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
                }
                _write_fsynced(
                    staging / "snapshot.json", canonical_json_bytes(snapshot) + b"\n"
                )
                hashes = {
                    filename: hashlib.sha256(
                        (staging / filename).read_bytes()
                    ).hexdigest()
                    for filename in REPLAY_FILES
                }
                counts = {operation: 0 for operation in REPLAY_OPERATIONS}
                for record in self._records.values():
                    counts[record.key.operation] += 1
                manifest = {
                    "file_sha256": hashes,
                    "record_count_by_operation": counts,
                    "schema_version": REPLAY_MANIFEST_SCHEMA_VERSION,
                }
                _write_fsynced(
                    staging / "manifest.sha256",
                    canonical_json_bytes(manifest) + b"\n",
                )
                _fsync_directory(staging)
                ReplayBundle.load(staging)
                self._validate_owned_staging()
                if self._final_root.exists():
                    raise FileExistsError(self._final_root)
                _atomic_rename_noreplace(staging, self._final_root)
                self._staging_root = None
                self._published = True
                try:
                    published_marker = self._final_root / self._MARKER_NAME
                    if _is_link_or_reparse(published_marker):
                        raise ValueError("published replay marker is unsafe")
                    published_marker.unlink()
                    _fsync_directory(self._final_root)
                    _fsync_directory(self._final_root.parent)
                except (OSError, RuntimeError, ValueError) as error:
                    raise OSError(
                        "replay bundle was published but durability finalization failed"
                    ) from error
                finally:
                    self._closed = True
                    self._release_lock()
                return self._final_root
            except Exception:
                if self._published:
                    raise
                cleanup_error: Exception | None = None
                try:
                    self._remove_owned_staging()
                except (OSError, RuntimeError, ValueError) as error:
                    cleanup_error = error
                finally:
                    self._closed = True
                    self._release_lock()
                if cleanup_error is not None:
                    raise OSError(
                        "replay finalization failed and staging cleanup failed"
                    ) from cleanup_error
                raise

    async def abort(self) -> None:
        async with self._append_lock:
            try:
                if not self._published and self._staging_root is not None:
                    self._remove_owned_staging()
            finally:
                self._closed = True
                self._release_lock()

    def _remove_owned_staging(self) -> None:
        staging = self._validate_owned_staging()
        if self._cleanup_tombstone is None:
            device, inode = self._staging_identity
            tombstone = self._final_root.parent / (
                f".{self._final_root.name}.cleanup.{self._run_id}."
                f"{device:x}.{inode:x}"
            )
            if _is_link_or_reparse(tombstone):
                raise ValueError("replay cleanup tombstone is unsafe")
            _atomic_move_owned_staging(staging, tombstone)
            self._staging_root = tombstone
            self._cleanup_tombstone = tombstone
            _fsync_directory(self._final_root.parent)
            staging = self._validate_owned_staging()
        marker_path = staging / self._MARKER_NAME
        marker_payload = canonical_json_bytes(self._expected_marker())
        children = sorted(
            staging.rglob("*"), key=lambda item: len(item.parts), reverse=True
        )
        for child in children:
            if _is_link_or_reparse(child):
                raise ValueError("owned staging contains a symlink or reparse point")
        for child in children:
            if child == marker_path:
                continue
            if child.is_dir():
                child.rmdir()
            else:
                child.unlink()
        marker_path.unlink()
        try:
            staging.rmdir()
        except OSError:
            _write_fsynced(marker_path, marker_payload)
            _fsync_directory(staging)
            _fsync_directory(staging.parent)
            raise
        self._staging_root = None
        _fsync_directory(staging.parent)

    def _validate_owned_staging(self) -> Path:
        staging = self._staging_root
        if staging is None:
            raise RuntimeError("replay staging directory is unavailable")
        parent = self._final_root.parent.resolve(strict=True)
        if _is_link_or_reparse(staging) or staging.resolve(strict=True).parent != parent:
            raise ValueError("replay staging directory is not an owned sibling")
        if _path_identity(staging) != self._staging_identity:
            raise ValueError("replay staging directory object was substituted")
        marker_path = staging / self._MARKER_NAME
        if _is_link_or_reparse(marker_path):
            raise ValueError("replay staging marker is unsafe")
        marker: object = json.loads(marker_path.read_text(encoding="utf-8"))
        expected = self._expected_marker()
        if marker != expected:
            raise ValueError("replay staging ownership marker does not match")
        return staging

    def _expected_marker(self) -> dict[str, str]:
        return {
            "final_name": self._final_root.name,
            "run_id": self._run_id,
            "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        }

    def _release_lock(self) -> None:
        descriptor = self._lock_descriptor
        if descriptor is None:
            return
        self._lock_descriptor = None
        try:
            _unlock(descriptor)
        finally:
            os.close(descriptor)


class _RecordingBase:
    def __init__(self, writer: ReplayBundleWriter) -> None:
        self._writer = writer
        defaults = ProviderCallPolicy.defaults()
        self._executor = ProviderCallExecutor(
            policy=ProviderCallPolicy(
                default_timeout_seconds=defaults.default_timeout_seconds,
                max_retries=0,
                base_delay_seconds=defaults.base_delay_seconds,
                max_delay_seconds=defaults.max_delay_seconds,
                jitter_ratio=defaults.jitter_ratio,
            ),
        )

    @staticmethod
    def _checkpoint(
        *,
        deadline: float,
        cancellation_token: CancellationToken,
        provider_id: str,
        operation: str,
    ) -> None:
        if not isfinite(deadline):
            raise ValueError("deadline must be finite")
        cancellation_token.raise_if_cancelled()
        if time.monotonic() >= deadline:
            raise ProviderError(
                code="TIMEOUT",
                provider=provider_id,
                operation=operation,
                public_message="provider call deadline exceeded",
                retryable=True,
            )

    async def _record_call(
        self,
        *,
        operation: ReplayOperation,
        executor_operation: str,
        key: ReplayRequestKey,
        deadline: float,
        cancellation_token: CancellationToken,
        invoke: Callable[[float], Awaitable[T]],
        encode: Callable[[T], JsonValue],
        usage_of: Callable[[T, int], ResourceUsage],
    ) -> T:
        self._checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=key.provider_id,
            operation=operation,
        )
        started = time.monotonic()
        try:
            result = await self._executor.call(
                cast("Any", executor_operation), invoke, remaining_deadline=deadline
            )
        except ProviderError as error:
            latency_ms = max(0, round((time.monotonic() - started) * 1000))
            usage = error.usage or _operation_usage(operation, latency_ms)
            await self._writer.append(
                key=key,
                outcome=ReplayFailure(
                    code=error.code,
                    public_message=_redact_public_message(error.public_message),
                    retryable=error.retryable,
                    retry_after=error.retry_after,
                ),
                usage=usage,
                latency_ms=latency_ms,
            )
            self._checkpoint(
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=key.provider_id,
                operation=operation,
            )
            raise
        self._checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=key.provider_id,
            operation=operation,
        )
        latency_ms = max(0, round((time.monotonic() - started) * 1000))
        await self._writer.append(
            key=key,
            outcome=ReplaySuccess(response=encode(result)),
            usage=usage_of(result, latency_ms),
            latency_ms=latency_ms,
        )
        self._checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=key.provider_id,
            operation=operation,
        )
        return result


def _key(
    operation: ReplayOperation,
    provider_id: str,
    payload: object,
    *,
    prompt_version: str | None = None,
) -> ReplayRequestKey:
    return ReplayRequestKey(
        operation=operation,
        provider_id=provider_id,
        request_sha256=canonical_request_sha256(payload),
        prompt_version=prompt_version,
        schema_version=REPLAY_REQUEST_SCHEMA_VERSION,
    )


class RecordingModelProvider(_RecordingBase):
    def __init__(self, delegate: ModelProvider, writer: ReplayBundleWriter) -> None:
        super().__init__(writer)
        self._delegate = delegate
        self.provider_id = delegate.provider_id
        self.model_revision = writer.model_revision_for(self.provider_id)

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        key = _key(
            "model.complete",
            self.provider_id,
            model_request_payload(request, model_revision=self.model_revision),
            prompt_version=request.prompt_version,
        )
        return await self._record_call(
            operation="model.complete",
            executor_operation="model",
            key=key,
            deadline=deadline,
            cancellation_token=cancellation_token,
            invoke=lambda call_deadline: self._delegate.complete(
                request, deadline=call_deadline, cancellation_token=cancellation_token
            ),
            encode=lambda result: cast("JsonValue", result.model_dump(mode="json")),
            usage_of=lambda result, _: result.usage,
        )

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[T]:
        key = _key(
            "model.structured",
            self.provider_id,
            model_request_payload(request, model_revision=self.model_revision),
            prompt_version=request.prompt_version,
        )
        return await self._record_call(
            operation="model.structured",
            executor_operation="model",
            key=key,
            deadline=deadline,
            cancellation_token=cancellation_token,
            invoke=lambda call_deadline: self._delegate.structured(
                request,
                output_schema,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            ),
            encode=lambda result: cast("JsonValue", result.model_dump(mode="json")),
            usage_of=lambda result, _: result.usage,
        )

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        async def collect(call_deadline: float) -> tuple[ModelStreamChunk, ...]:
            self._checkpoint(
                deadline=call_deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation="model.stream",
            )
            stream = self._delegate.stream(
                request,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            )
            chunks: list[ModelStreamChunk] = []
            async for chunk in stream:
                self._checkpoint(
                    deadline=call_deadline,
                    cancellation_token=cancellation_token,
                    provider_id=self.provider_id,
                    operation="model.stream",
                )
                chunks.append(chunk)
            return validate_model_stream(tuple(chunks))

        async def recorded_stream() -> AsyncIterator[ModelStreamChunk]:
            key = _key(
                "model.stream",
                self.provider_id,
                model_request_payload(request, model_revision=self.model_revision),
                prompt_version=request.prompt_version,
            )
            chunks = await self._record_call(
                operation="model.stream",
                executor_operation="model",
                key=key,
                deadline=deadline,
                cancellation_token=cancellation_token,
                invoke=collect,
                encode=lambda result: cast(
                    "JsonValue", [chunk.model_dump(mode="json") for chunk in result]
                ),
                usage_of=lambda result, _: cast("ResourceUsage", result[-1].final_usage),
            )
            for chunk in chunks:
                self._checkpoint(
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                    provider_id=self.provider_id,
                    operation="model.stream",
                )
                yield chunk

        return recorded_stream()


class RecordingSearchProvider(_RecordingBase):
    def __init__(self, delegate: SearchProvider, writer: ReplayBundleWriter) -> None:
        super().__init__(writer)
        self._delegate = delegate
        self.provider_id = delegate.provider_id
        writer.register_provider("search", provider_id=self.provider_id)

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        payload = search_request_payload(query, limit, filters)
        key = _key("search", self.provider_id, payload)
        return await self._record_call(
            operation="search",
            executor_operation="search",
            key=key,
            deadline=deadline,
            cancellation_token=cancellation_token,
            invoke=lambda call_deadline: self._delegate.search(
                query,
                limit,
                filters,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            ),
            encode=lambda result: cast(
                "JsonValue", [hit.model_dump(mode="json") for hit in result]
            ),
            usage_of=lambda _, latency: _operation_usage("search", latency),
        )


class RecordingFetcher(_RecordingBase):
    def __init__(self, delegate: Fetcher, writer: ReplayBundleWriter) -> None:
        super().__init__(writer)
        self._delegate = delegate
        self.provider_id = delegate.provider_id
        writer.register_provider("fetch", provider_id=self.provider_id)

    async def fetch(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> RawDocument:
        key = _key("fetch", self.provider_id, fetch_request_payload(url))
        return await self._record_call(
            operation="fetch",
            executor_operation="fetch",
            key=key,
            deadline=deadline,
            cancellation_token=cancellation_token,
            invoke=lambda call_deadline: self._delegate.fetch(
                url, deadline=call_deadline, cancellation_token=cancellation_token
            ),
            encode=lambda result: cast("JsonValue", _safe_fetch_response(result)),
            usage_of=lambda _, latency: _operation_usage("fetch", latency),
        )


class RecordingTextEmbedder(_RecordingBase):
    def __init__(self, delegate: TextEmbedder, writer: ReplayBundleWriter) -> None:
        super().__init__(writer)
        self._delegate = delegate
        self.provider_id = delegate.provider_id
        self.model_id = delegate.model_id
        self.model_revision = delegate.model_revision
        self.snapshot_sha256 = delegate.snapshot_sha256
        writer.register_provider(
            "embed",
            provider_id=self.provider_id,
            model_id=self.model_id,
            model_revision=self.model_revision,
            snapshot_sha256=self.snapshot_sha256,
        )

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        frozen_texts = tuple(texts)
        key = _key("embed", self.provider_id, embed_request_payload(frozen_texts))
        return await self._record_call(
            operation="embed",
            executor_operation="embed",
            key=key,
            deadline=deadline,
            cancellation_token=cancellation_token,
            invoke=lambda call_deadline: self._delegate.embed(
                frozen_texts,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            ),
            encode=lambda result: cast("JsonValue", [list(vector) for vector in result]),
            usage_of=lambda _, latency: _operation_usage("embed", latency),
        )


__all__ = [
    "RecordingFetcher",
    "RecordingModelProvider",
    "RecordingSearchProvider",
    "RecordingTextEmbedder",
    "ReplayBundleWriter",
]
