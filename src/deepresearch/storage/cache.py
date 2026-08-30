from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
from collections.abc import Generator, Mapping
from contextlib import contextmanager
from datetime import datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Never, Self, TypeAlias, cast, override

from pydantic import (
    AnyHttpUrl,
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
)

from deepresearch.domain import ResourceUsage

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_ARTIFACT_ID_PATTERN = re.compile(r"sha256:[0-9a-f]{64}")
_SECRET_KEY_NAMES = frozenset(
    {
        "accesstoken",
        "apikey",
        "apitoken",
        "authorization",
        "bearer",
        "cookie",
        "credential",
        "credentials",
        "password",
        "passwd",
        "refreshtoken",
        "secret",
        "token",
    }
)
type _InternalJson = JsonValue | tuple[_InternalJson, ...]
_Key = Any
_Value = Any


class CacheIntegrityError(ValueError):
    pass


class CacheConflictError(CacheIntegrityError):
    pass


class _FrozenDict(dict[_Key, _Value]):
    @staticmethod
    def _raise_immutable() -> Never:
        raise TypeError("cache mappings are immutable")

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


def _thaw_json(value: object) -> object:
    if isinstance(value, tuple):
        items = cast("tuple[object, ...]", value)
        return [_thaw_json(item) for item in items]
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        return {key: _thaw_json(item) for key, item in mapping.items()}
    return value


def _freeze_json(value: JsonValue) -> _InternalJson:
    if isinstance(value, dict):
        mapping = cast("dict[str, JsonValue]", value)
        return _FrozenDict({key: _freeze_json(item) for key, item in mapping.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in value)
    return value


def _freeze_json_mapping(value: dict[str, JsonValue]) -> dict[str, JsonValue]:
    frozen = _FrozenDict({key: _freeze_json(item) for key, item in value.items()})
    return cast("dict[str, JsonValue]", frozen)


def _canonical_json(value: object) -> JsonValue:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return {key: _canonical_json(mapping[key]) for key in sorted(mapping)}
    if isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        return [_canonical_json(item) for item in items]
    return cast("JsonValue", value)


def _normalized_secret_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]", "", key.casefold())


def _reject_secrets(value: object) -> None:
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        for key, item in mapping.items():
            if isinstance(key, str) and _normalized_secret_key(key) in _SECRET_KEY_NAMES:
                raise ValueError(f"secret-bearing metadata key is forbidden: {key}")
            _reject_secrets(item)
    elif isinstance(value, (list, tuple)):
        items = cast("list[object] | tuple[object, ...]", value)
        for item in items:
            _reject_secrets(item)


def _require_sha256(value: str) -> str:
    if _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError("value must be a lowercase 64-character SHA-256")
    return value


def _require_optional_sha256(value: str) -> str:
    if value:
        _require_sha256(value)
    return value


class _CacheModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

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


class CacheEntry(_CacheModel):
    key_sha256: str
    value_artifact_id: str
    producer_version: str
    usage: ResourceUsage
    created_at: datetime
    metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _valid_key_hash = field_validator("key_sha256")(_require_sha256)

    @field_validator("value_artifact_id")
    @classmethod
    def validate_value_artifact_id(cls, value: str) -> str:
        if _ARTIFACT_ID_PATTERN.fullmatch(value) is None:
            raise ValueError("value_artifact_id must be a content-addressed SHA-256 ID")
        return value

    @field_validator("created_at")
    @classmethod
    def require_aware_created_at(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")
        return value

    @field_validator("metadata", mode="before")
    @classmethod
    def thaw_metadata(cls, value: object) -> object:
        return _thaw_json(value)

    @field_validator("metadata")
    @classmethod
    def validate_and_freeze_metadata(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _reject_secrets(value)
        return _freeze_json_mapping(value)

    @field_serializer("metadata")
    def serialize_metadata(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", _canonical_json(value))


class SearchCacheKey(_CacheModel):
    operation: Literal["search"] = "search"
    snapshot_id: str
    normalized_query: str
    provider_id: str
    endpoint_type: str
    locale: str | None = None
    complete_parameters: dict[str, JsonValue]
    time_policy: str

    @field_validator("complete_parameters", mode="before")
    @classmethod
    def thaw_parameters(cls, value: object) -> object:
        return _thaw_json(value)

    @field_validator("complete_parameters")
    @classmethod
    def validate_and_freeze_parameters(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        _reject_secrets(value)
        return _freeze_json_mapping(value)

    @field_serializer("complete_parameters")
    def serialize_parameters(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return cast("dict[str, JsonValue]", _canonical_json(value))


class FetchCacheKey(_CacheModel):
    operation: Literal["fetch"] = "fetch"
    snapshot_id: str
    canonical_url: AnyHttpUrl
    fetch_policy: str
    accepted_content_types: tuple[str, ...]


class ParseCacheKey(_CacheModel):
    operation: Literal["parse"] = "parse"
    snapshot_id: str
    raw_content_hash: str
    parser_id: str
    parser_version: str
    normalization_version: str

    _valid_raw_content_hash = field_validator("raw_content_hash")(_require_sha256)


class ModelCacheKey(_CacheModel):
    operation: Literal["model"] = "model"
    provider_id: str
    endpoint_type: str
    model_id: str
    prompt_version: str
    system_prompt_hash: str
    tool_schema_hash: str
    output_schema_hash: str
    temperature: Decimal
    seed: int | None = None
    canonical_request_hash: str

    _required_hashes = field_validator("system_prompt_hash", "canonical_request_hash")(
        _require_sha256
    )
    _optional_hashes = field_validator("tool_schema_hash", "output_schema_hash")(
        _require_optional_sha256
    )


class EmbedCacheKey(_CacheModel):
    operation: Literal["embed"] = "embed"
    model_id: str
    model_revision: str
    snapshot_sha256: str
    normalize_embeddings: bool
    canonical_texts_hash: str

    _valid_hashes = field_validator("snapshot_sha256", "canonical_texts_hash")(
        _require_sha256
    )


CacheKey: TypeAlias = Annotated[  # noqa: UP040 - exact frozen public contract
    SearchCacheKey | FetchCacheKey | ParseCacheKey | ModelCacheKey | EmbedCacheKey,
    Field(discriminator="operation"),
]


def cache_key_json(key: CacheKey) -> str:
    return json.dumps(
        key.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )


def cache_key_sha256(key: CacheKey) -> str:
    return hashlib.sha256(cache_key_json(key).encode("utf-8")).hexdigest()


@contextmanager
def _key_lock(lock_path: Path, *, timeout_seconds: float = 10.0) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for cache lock {lock_path.name}")
            time.sleep(0.005)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


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


class FileCache:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve() / "cache"
        self._root.mkdir(parents=True, exist_ok=True)

    def _path(self, digest: str) -> Path:
        _require_sha256(digest)
        return self._root / digest[:2] / f"{digest}.json"

    def _read(self, digest: str) -> CacheEntry:
        path = self._path(digest)
        try:
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise TypeError("cache envelope is not an object")
            envelope = cast("dict[str, object]", loaded)
            if set(envelope) != {"payload", "payload_sha256"}:
                raise TypeError("cache envelope fields are invalid")
            raw_payload = envelope["payload"]
            if not isinstance(raw_payload, dict):
                raise TypeError("cache payload is not an object")
            entry_payload = cast("dict[str, object]", raw_payload)
            checksum = envelope["payload_sha256"]
            canonical_payload = json.dumps(
                entry_payload,
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            if not isinstance(checksum, str) or hashlib.sha256(canonical_payload).hexdigest() != checksum:
                raise TypeError("cache payload checksum does not match")
            entry = CacheEntry.model_validate(entry_payload)
        except FileNotFoundError:
            raise
        except Exception as error:
            raise CacheIntegrityError("cache entry is corrupt") from error
        if entry.key_sha256 != digest:
            raise CacheIntegrityError("cache entry key_sha256 does not match its file key")
        return entry

    def get(self, key: CacheKey) -> CacheEntry | None:
        digest = cache_key_sha256(key)
        try:
            return self._read(digest)
        except FileNotFoundError:
            return None

    def put_if_absent(self, key: CacheKey, value: CacheEntry) -> CacheEntry:
        digest = cache_key_sha256(key)
        if value.key_sha256 != digest:
            raise CacheIntegrityError("CacheEntry key_sha256 does not match the cache key")
        path = self._path(digest)
        lock_path = path.with_suffix(".lock")
        entry_payload = value.model_dump(mode="json")
        canonical_payload = json.dumps(
            entry_payload,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        envelope = {
            "payload": entry_payload,
            "payload_sha256": hashlib.sha256(canonical_payload).hexdigest(),
        }
        payload = json.dumps(
            envelope,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        with _key_lock(lock_path):
            if path.exists():
                existing = self._read(digest)
                if existing != value:
                    raise CacheConflictError("cache key already contains a different entry")
                return existing
            _atomic_write_bytes(path, payload)
        return value


__all__ = [
    "CacheConflictError",
    "CacheEntry",
    "CacheIntegrityError",
    "CacheKey",
    "EmbedCacheKey",
    "FetchCacheKey",
    "FileCache",
    "ModelCacheKey",
    "ParseCacheKey",
    "SearchCacheKey",
    "cache_key_json",
    "cache_key_sha256",
]
