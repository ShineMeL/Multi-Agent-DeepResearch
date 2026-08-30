from __future__ import annotations

import hashlib
import json
import math
import stat
from collections import Counter
from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any, Literal, Never, Self, cast, override

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    ValidationError,
    field_serializer,
    field_validator,
    model_validator,
)

from deepresearch.domain import ResourceUsage
from deepresearch.providers import ProviderError
from deepresearch.retrieval import canonicalize_url, normalize_text
from deepresearch.storage import FetchCacheKey, SearchCacheKey

ReplayOperation = Literal[
    "model.complete", "model.structured", "model.stream", "search", "fetch", "embed"
]

REPLAY_REQUEST_SCHEMA_VERSION = "replay-request-v1"
REPLAY_BUNDLE_SCHEMA_VERSION = "replay-bundle-v1"
REPLAY_MANIFEST_SCHEMA_VERSION = "replay-manifest-v1"
REPLAY_OPERATIONS: tuple[ReplayOperation, ...] = (
    "model.complete",
    "model.structured",
    "model.stream",
    "search",
    "fetch",
    "embed",
)
REPLAY_FILES: tuple[str, ...] = (
    "documents.jsonl",
    "embeddings.jsonl",
    "model_responses.jsonl",
    "search.jsonl",
    "snapshot.json",
)
_RECORD_FILES: dict[str, tuple[ReplayOperation, ...]] = {
    "documents.jsonl": ("fetch",),
    "embeddings.jsonl": ("embed",),
    "model_responses.jsonl": ("model.complete", "model.structured", "model.stream"),
    "search.jsonl": ("search",),
}
_PROVIDER_ERROR_CODES = frozenset(
    {
        "TIMEOUT",
        "RATE_LIMITED",
        "INVALID_REQUEST",
        "INVALID_RESPONSE",
        "AUTHENTICATION",
        "NETWORK",
        "REPLAY_MISS",
        "INVALID_SNAPSHOT",
        "PARSE_UNSUPPORTED",
        "UPSTREAM_5XX",
    }
)


class _FrozenDict(dict[str, Any]):
    @staticmethod
    def _immutable() -> Never:
        raise TypeError("replay mappings are immutable")

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


def _freeze_json(value: object) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return _FrozenDict({key: _freeze_json(item) for key, item in mapping.items()})
    if isinstance(value, list):
        return tuple(_freeze_json(item) for item in cast("list[object]", value))
    return value


def _thaw_json(value: object) -> object:
    if isinstance(value, dict):
        mapping = cast("dict[str, object]", value)
        return {key: _thaw_json(item) for key, item in mapping.items()}
    if isinstance(value, (list, tuple)):
        return [_thaw_json(item) for item in cast("list[object] | tuple[object, ...]", value)]
    return value


def canonical_json_bytes(value: object) -> bytes:
    """Encode JSON deterministically with the repository's UTF-8 rules."""
    return json.dumps(
        _thaw_json(value),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def canonical_request_sha256(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def model_request_payload(request: object) -> dict[str, JsonValue]:
    dump = getattr(request, "model_dump", None)
    if not callable(dump):
        raise TypeError("model request must support model_dump")
    return cast("dict[str, JsonValue]", dump(mode="json"))


def search_request_payload(
    query: str, limit: int, filters: Mapping[str, JsonValue] | None
) -> dict[str, JsonValue]:
    normalized_query = normalize_text(query)
    SearchCacheKey(
        snapshot_id="replay-request",
        normalized_query=normalized_query,
        provider_id="replay-request",
        endpoint_type="search",
        locale=None,
        complete_parameters={
            "filters": None if filters is None else dict(filters),
            "limit": limit,
        },
        time_policy="recorded",
    )
    return {
        "filters": None if filters is None else dict(filters),
        "limit": limit,
        "query": normalized_query,
    }


def fetch_request_payload(url: str) -> dict[str, JsonValue]:
    canonical_url = canonicalize_url(url)
    FetchCacheKey.model_validate(
        {
            "snapshot_id": "replay-request",
            "canonical_url": canonical_url,
            "fetch_policy": "recorded",
            "accepted_content_types": (),
        }
    )
    return {"url": canonical_url}


def embed_request_payload(texts: tuple[str, ...]) -> dict[str, JsonValue]:
    return {"texts": [normalize_text(text) for text in texts]}


def _require_sha256(value: str) -> str:
    if len(value) != 64 or value != value.lower():
        raise ValueError("value must be a lowercase 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("value must be a lowercase 64-character SHA-256") from error
    return value


class _ReplayModel(BaseModel):
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


class ReplayRequestKey(_ReplayModel):
    operation: ReplayOperation
    provider_id: str
    request_sha256: str
    prompt_version: str | None = None
    schema_version: Literal["replay-request-v1"]

    _request_hash = field_validator("request_sha256")(_require_sha256)

    @model_validator(mode="after")
    def require_prompt_version_only_for_models(self) -> ReplayRequestKey:
        is_model = self.operation.startswith("model.")
        if is_model and not self.prompt_version:
            raise ValueError("model replay keys require prompt_version")
        if not is_model and self.prompt_version is not None:
            raise ValueError("non-model replay keys forbid prompt_version")
        if not self.provider_id:
            raise ValueError("provider_id must not be empty")
        return self

    def identity(self) -> tuple[str, str, str, str | None, str]:
        return (
            self.operation,
            self.provider_id,
            self.request_sha256,
            self.prompt_version,
            self.schema_version,
        )


class ReplaySuccess(_ReplayModel):
    kind: Literal["success"] = "success"
    response: JsonValue

    @field_validator("response", mode="before")
    @classmethod
    def thaw_response(cls, value: object) -> object:
        return _thaw_json(value)

    @field_validator("response")
    @classmethod
    def freeze_response(cls, value: JsonValue) -> JsonValue:
        return cast("JsonValue", _freeze_json(value))

    @field_serializer("response")
    def serialize_response(self, value: JsonValue) -> JsonValue:
        return cast("JsonValue", _thaw_json(value))


class ReplayFailure(_ReplayModel):
    kind: Literal["failure"] = "failure"
    code: str
    public_message: str
    retryable: bool
    retry_after: Annotated[float | None, Field(ge=0)] = None

    @field_validator("code")
    @classmethod
    def validate_code(cls, value: str) -> str:
        if value not in _PROVIDER_ERROR_CODES:
            raise ValueError("recorded failure has an unknown provider error code")
        return value


ReplayOutcome = Annotated[ReplaySuccess | ReplayFailure, Field(discriminator="kind")]


class ReplayRecord(_ReplayModel):
    key: ReplayRequestKey
    outcome: ReplayOutcome
    usage: ResourceUsage
    latency_ms: Annotated[int, Field(ge=0)]
    outcome_sha256: str

    _outcome_hash = field_validator("outcome_sha256")(_require_sha256)

    @field_validator("usage", mode="before")
    @classmethod
    def revalidate_usage(cls, value: object) -> ResourceUsage:
        payload = (
            value.model_dump(round_trip=True) if isinstance(value, ResourceUsage) else value
        )
        usage = ResourceUsage.model_validate(payload)
        if not math.isfinite(usage.wall_seconds):
            raise ValueError("usage wall_seconds must be finite")
        if usage.cost_usd is not None and not usage.cost_usd.is_finite():
            raise ValueError("usage cost_usd must be finite")
        return usage

    @model_validator(mode="after")
    def validate_outcome_hash(self) -> ReplayRecord:
        expected = canonical_request_sha256(self.outcome.model_dump(mode="json"))
        if self.outcome_sha256 != expected:
            raise ValueError("outcome_sha256 does not match the canonical outcome")
        return self


class ReplayVerification(_ReplayModel):
    valid: bool
    record_count_by_operation: dict[str, int]
    file_sha256: dict[str, str]
    errors: tuple[str, ...] = ()

    @field_validator("record_count_by_operation", "file_sha256")
    @classmethod
    def freeze_mappings(cls, value: dict[str, Any]) -> dict[str, Any]:
        return cast("dict[str, Any]", _FrozenDict(dict(value)))

    @model_validator(mode="after")
    def validate_valid_flag(self) -> ReplayVerification:
        if self.valid != (not self.errors):
            raise ValueError("valid must equal not errors")
        return self


class ReplayProviderSnapshot(_ReplayModel):
    provider_id: str
    model_id: str | None = None
    model_revision: str | None = None
    snapshot_sha256: str | None = None

    @field_validator("snapshot_sha256")
    @classmethod
    def validate_optional_hash(cls, value: str | None) -> str | None:
        return None if value is None else _require_sha256(value)


class ReplaySnapshot(_ReplayModel):
    schema_version: Literal["replay-bundle-v1"]
    run_id: str
    providers: dict[str, ReplayProviderSnapshot]

    @field_validator("providers")
    @classmethod
    def freeze_providers(
        cls, value: dict[str, ReplayProviderSnapshot]
    ) -> dict[str, ReplayProviderSnapshot]:
        return cast("dict[str, ReplayProviderSnapshot]", _FrozenDict(dict(value)))


class _ReplayManifest(_ReplayModel):
    schema_version: Literal["replay-manifest-v1"]
    record_count_by_operation: dict[str, int]
    file_sha256: dict[str, str]

    @field_validator("record_count_by_operation")
    @classmethod
    def validate_counts(cls, value: dict[str, int]) -> dict[str, int]:
        if set(value) != set(REPLAY_OPERATIONS):
            raise ValueError("record_count_by_operation is incomplete")
        if any(count < 0 for count in value.values()):
            raise ValueError("record counts must be non-negative")
        return value

    @field_validator("file_sha256")
    @classmethod
    def validate_hashes(cls, value: dict[str, str]) -> dict[str, str]:
        if set(value) != set(REPLAY_FILES):
            raise ValueError("manifest.sha256 is incomplete")
        for digest in value.values():
            _require_sha256(digest)
        return value


def _is_link_or_reparse(path: Path) -> bool:
    if path.is_symlink():
        return True
    try:
        attributes = path.lstat().st_file_attributes
    except (AttributeError, FileNotFoundError, OSError):
        return False
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def _safe_child(root: Path, filename: str) -> Path:
    if filename != Path(filename).name or filename in {"", ".", ".."}:
        raise ValueError("replay bundle filename contains path traversal")
    child = root / filename
    if _is_link_or_reparse(child):
        raise ValueError("replay bundle files must not be symlinks or reparse points")
    if child.resolve(strict=False).parent != root.resolve(strict=True):
        raise ValueError("replay bundle file escapes its root")
    return child


def _invalid_snapshot(message: str, error: Exception | None = None) -> ProviderError:
    result = ProviderError(
        code="INVALID_SNAPSHOT",
        provider="replay-bundle",
        operation="replay.load",
        public_message=message,
        retryable=False,
    )
    if error is not None:
        result.__cause__ = error
    return result


class ReplayBundle:
    def __init__(
        self,
        *,
        root: Path,
        snapshot: ReplaySnapshot,
        records: dict[tuple[str, str, str, str | None, str], ReplayRecord],
        verification: ReplayVerification,
    ) -> None:
        self._root = root
        self._snapshot = snapshot
        self._records = records
        self._verification = verification

    @property
    def root(self) -> Path:
        return self._root

    @property
    def snapshot(self) -> ReplaySnapshot:
        return self._snapshot

    @classmethod
    def load(cls, root: Path) -> ReplayBundle:
        try:
            requested_root = Path(root).absolute()
            if _is_link_or_reparse(requested_root):
                raise ValueError("replay bundle root must not be a symlink or reparse point")
            resolved_root = requested_root.resolve(strict=True)
            if not resolved_root.is_dir():
                raise ValueError("replay bundle root must be a directory")
            manifest_path = _safe_child(resolved_root, "manifest.sha256")
            if not manifest_path.is_file():
                raise ValueError("manifest.sha256 is missing")
            raw_manifest = manifest_path.read_bytes()
            manifest = _ReplayManifest.model_validate_json(raw_manifest)

            actual_hashes: dict[str, str] = {}
            for filename in REPLAY_FILES:
                path = _safe_child(resolved_root, filename)
                if not path.is_file():
                    raise ValueError(f"required replay file is missing: {filename}")
                actual_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
            if actual_hashes != manifest.file_sha256:
                raise ValueError("replay file hashes do not match manifest.sha256")

            snapshot_path = _safe_child(resolved_root, "snapshot.json")
            snapshot = ReplaySnapshot.model_validate_json(snapshot_path.read_bytes())
            records: dict[tuple[str, str, str, str | None, str], ReplayRecord] = {}
            counts: Counter[str] = Counter()
            for filename, allowed_operations in _RECORD_FILES.items():
                path = _safe_child(resolved_root, filename)
                previous_sort_key: bytes | None = None
                for line_number, raw_line in enumerate(path.read_bytes().splitlines(), start=1):
                    if not raw_line.strip():
                        raise ValueError(f"blank JSONL record in {filename}:{line_number}")
                    loaded: object = json.loads(raw_line)
                    record = ReplayRecord.model_validate(loaded)
                    if record.key.operation not in allowed_operations:
                        raise ValueError(f"operation/file mismatch in {filename}:{line_number}")
                    provider_kind = (
                        "model"
                        if record.key.operation.startswith("model.")
                        else record.key.operation
                    )
                    provider = snapshot.providers.get(provider_kind)
                    if provider is None or provider.provider_id != record.key.provider_id:
                        raise ValueError(
                            f"record provider does not match snapshot in {filename}:{line_number}"
                        )
                    canonical_line = canonical_json_bytes(record.model_dump(mode="json"))
                    if raw_line != canonical_line:
                        raise ValueError(f"non-canonical JSONL record in {filename}:{line_number}")
                    sort_key = canonical_json_bytes(record.key.model_dump(mode="json"))
                    if previous_sort_key is not None and sort_key < previous_sort_key:
                        raise ValueError(f"unsorted JSONL records in {filename}")
                    previous_sort_key = sort_key
                    identity = record.key.identity()
                    if identity in records:
                        raise ValueError("duplicate replay request key")
                    records[identity] = record
                    counts[record.key.operation] += 1
            complete_counts = {operation: counts[operation] for operation in REPLAY_OPERATIONS}
            if complete_counts != manifest.record_count_by_operation:
                raise ValueError("replay record counts do not match manifest.sha256")
        except ProviderError:
            raise
        except (OSError, ValueError, TypeError, json.JSONDecodeError, ValidationError) as error:
            raise _invalid_snapshot("replay bundle is invalid", error) from error

        verification = ReplayVerification(
            valid=True,
            record_count_by_operation=complete_counts,
            file_sha256=actual_hashes,
        )
        return cls(
            root=resolved_root,
            snapshot=snapshot,
            records=records,
            verification=verification,
        )

    def lookup(self, key: ReplayRequestKey) -> ReplayRecord:
        try:
            return self._records[key.identity()]
        except KeyError as error:
            raise ProviderError(
                code="REPLAY_MISS",
                provider=key.provider_id,
                operation=key.operation,
                public_message="strict replay has no exact recorded request",
                retryable=False,
            ) from error

    def verify(self) -> ReplayVerification:
        try:
            return type(self).load(self._root)._verification
        except ProviderError as error:
            actual_hashes: dict[str, str] = {}
            for filename in REPLAY_FILES:
                try:
                    path = _safe_child(self._root, filename)
                    if path.is_file():
                        actual_hashes[filename] = hashlib.sha256(path.read_bytes()).hexdigest()
                except (OSError, ValueError):
                    continue
            return ReplayVerification(
                valid=False,
                record_count_by_operation=dict(
                    self._verification.record_count_by_operation
                ),
                file_sha256=actual_hashes,
                errors=(error.public_message,),
            )

    def provider_snapshot(self, operation: str) -> ReplayProviderSnapshot:
        try:
            return self._snapshot.providers[operation]
        except KeyError as error:
            raise _invalid_snapshot(f"snapshot has no provider metadata for {operation}") from error


__all__ = [
    "REPLAY_BUNDLE_SCHEMA_VERSION",
    "REPLAY_FILES",
    "REPLAY_MANIFEST_SCHEMA_VERSION",
    "REPLAY_OPERATIONS",
    "REPLAY_REQUEST_SCHEMA_VERSION",
    "ReplayBundle",
    "ReplayFailure",
    "ReplayOperation",
    "ReplayProviderSnapshot",
    "ReplayRecord",
    "ReplayRequestKey",
    "ReplaySnapshot",
    "ReplaySuccess",
    "ReplayVerification",
    "canonical_json_bytes",
    "canonical_request_sha256",
    "embed_request_payload",
    "fetch_request_payload",
    "model_request_payload",
    "search_request_payload",
]
