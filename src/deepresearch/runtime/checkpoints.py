from __future__ import annotations

import math
import stat
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Literal, cast, override

import aiosqlite
from langgraph.checkpoint.base import BaseCheckpointSaver, CheckpointTuple
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from pydantic import BaseModel

from deepresearch.domain import (
    CoverageLedgerEntry,
    InformationNeed,
    ResearchPlan,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
    SubQuestion,
)

from .budget import BudgetSnapshot
from .ports import CheckpointRef


class CheckpointSerializationError(TypeError):
    def __init__(self) -> None:
        super().__init__("checkpoint value is not serializable")


class CheckpointIdentityError(ValueError):
    code: Literal["CHECKPOINT_MISMATCH"] = "CHECKPOINT_MISMATCH"

    def __init__(self) -> None:
        super().__init__("checkpoint identity does not match")


@dataclass(frozen=True)
class _TupleEnvelope:
    items: list[object]


_ALLOWED_CHECKPOINT_TYPES: tuple[type[BaseModel], ...] = (
    ResearchRequest,
    ResearchPlan,
    SubQuestion,
    InformationNeed,
    CoverageLedgerEntry,
    RunBudget,
    ResourceUsage,
    BudgetSnapshot,
)
_ALLOWED_VALUE_TYPES = frozenset(_ALLOWED_CHECKPOINT_TYPES)


def _validated_for_encoding(value: object, active: set[int]) -> object:
    if value is None or type(value) in {bool, int, str, bytes}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointSerializationError()
        return value
    if type(value) in _ALLOWED_VALUE_TYPES:
        return value
    if isinstance(value, tuple):
        values = cast("tuple[object, ...]", value)
        identity = id(values)
        if identity in active:
            raise CheckpointSerializationError()
        active.add(identity)
        try:
            return _TupleEnvelope(
                [_validated_for_encoding(item, active) for item in values]
            )
        finally:
            active.remove(identity)
    if type(value) is list:
        values = cast("list[object]", value)
        identity = id(values)
        if identity in active:
            raise CheckpointSerializationError()
        active.add(identity)
        try:
            return [_validated_for_encoding(item, active) for item in values]
        finally:
            active.remove(identity)
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
        identity = id(mapping)
        if identity in active:
            raise CheckpointSerializationError()
        active.add(identity)
        try:
            if any(type(key) is not str for key in mapping):
                raise CheckpointSerializationError()
            return {
                cast("str", key): _validated_for_encoding(item, active)
                for key, item in mapping.items()
            }
        finally:
            active.remove(identity)
    raise CheckpointSerializationError()


def _validated_after_decoding(value: object, active: set[int]) -> object:
    if value is None or type(value) in {bool, int, str, bytes}:
        return value
    if type(value) is float:
        if not math.isfinite(value):
            raise CheckpointSerializationError()
        return value
    if type(value) in _ALLOWED_VALUE_TYPES:
        return value
    if type(value) is _TupleEnvelope:
        envelope = value
        if type(envelope.items) is not list:
            raise CheckpointSerializationError()
        identity = id(envelope)
        if identity in active:
            raise CheckpointSerializationError()
        active.add(identity)
        try:
            return tuple(
                _validated_after_decoding(item, active) for item in envelope.items
            )
        finally:
            active.remove(identity)
    if type(value) is list:
        values = cast("list[object]", value)
        identity = id(values)
        if identity in active:
            raise CheckpointSerializationError()
        active.add(identity)
        try:
            return [_validated_after_decoding(item, active) for item in values]
        finally:
            active.remove(identity)
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
        identity = id(mapping)
        if identity in active or any(type(key) is not str for key in mapping):
            raise CheckpointSerializationError()
        active.add(identity)
        try:
            return {
                cast("str", key): _validated_after_decoding(item, active)
                for key, item in mapping.items()
            }
        finally:
            active.remove(identity)
    raise CheckpointSerializationError()


class _StrictCheckpointSerializer(JsonPlusSerializer):
    @override
    def dumps_typed(self, obj: Any) -> tuple[str, bytes]:
        try:
            prepared = _validated_for_encoding(obj, set())
            return super().dumps_typed(prepared)
        except CheckpointSerializationError:
            raise
        except Exception:  # noqa: BLE001 - public serializer boundary fails closed
            raise CheckpointSerializationError() from None

    @override
    def loads_typed(self, data: tuple[str, bytes]) -> Any:
        try:
            label, payload = data
            if type(label) is not str or type(payload) is not bytes:
                raise CheckpointSerializationError()
            if label not in {"null", "bytes", "json", "msgpack"}:
                raise CheckpointSerializationError()
            decoded = super().loads_typed((label, payload))
            if label == "msgpack":
                encoded_label, encoded_payload = JsonPlusSerializer.dumps_typed(
                    self, decoded
                )
                if encoded_label != label or encoded_payload != payload:
                    raise CheckpointSerializationError()
            return _validated_after_decoding(decoded, set())
        except CheckpointSerializationError:
            raise
        except Exception:  # noqa: BLE001 - public serializer boundary fails closed
            raise CheckpointSerializationError() from None


def checkpoint_serializer() -> JsonPlusSerializer:
    allowed_types: tuple[type[object], ...] = (
        *_ALLOWED_CHECKPOINT_TYPES,
        _TupleEnvelope,
    )
    return _StrictCheckpointSerializer(
        allowed_json_modules=tuple(
            (item.__module__, item.__name__) for item in allowed_types
        ),
        allowed_msgpack_modules=allowed_types,
        pickle_fallback=False,
    )


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    attributes = getattr(details, "st_file_attributes", 0)
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(attributes & reparse_flag)


@asynccontextmanager
async def open_sqlite_checkpointer(
    path: Path,
) -> AsyncGenerator[BaseCheckpointSaver[Any]]:
    if not path.is_absolute() or path == Path(path.anchor) or not path.name:
        raise ValueError("checkpoint path must be an absolute file path")
    if _is_link_or_reparse(path):
        raise ValueError("checkpoint file must not be a symlink or reparse point")
    if path.exists() and not path.is_file():
        raise ValueError("checkpoint path must be a file")
    path.parent.mkdir(parents=True, exist_ok=True)
    async with aiosqlite.connect(str(path)) as connection:
        saver = AsyncSqliteSaver(connection, serde=checkpoint_serializer())
        await saver.setup()
        yield saver


def checkpoint_config(ref: CheckpointRef) -> dict[str, dict[str, str]]:
    return {
        "configurable": {
            "thread_id": ref.thread_id,
            "checkpoint_ns": "",
            "checkpoint_id": ref.checkpoint_id,
        }
    }


def checkpoint_ref_from_tuple(value: CheckpointTuple) -> CheckpointRef:
    try:
        config = cast("dict[str, object]", value.config)
        configurable_value = config["configurable"]
        if type(configurable_value) is not dict:
            raise CheckpointIdentityError()
        configurable = cast("dict[str, object]", configurable_value)
        thread_id = configurable["thread_id"]
        checkpoint_id = configurable["checkpoint_id"]
        namespace = configurable.get("checkpoint_ns", "")
        checkpoint = cast("dict[str, object]", value.checkpoint)
        stored_id = checkpoint["id"]
        timestamp = checkpoint["ts"]
        if (
            type(thread_id) is not str
            or not thread_id
            or type(checkpoint_id) is not str
            or not checkpoint_id
            or type(namespace) is not str
            or namespace != ""
            or type(stored_id) is not str
            or stored_id != checkpoint_id
            or type(timestamp) is not str
        ):
            raise CheckpointIdentityError()
        created_at = datetime.fromisoformat(timestamp)
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise CheckpointIdentityError()
        return CheckpointRef(
            checkpoint_id=checkpoint_id,
            thread_id=thread_id,
            created_at=created_at,
        )
    except CheckpointIdentityError:
        raise
    except (KeyError, TypeError, ValueError):
        raise CheckpointIdentityError() from None


__all__ = [
    "CheckpointIdentityError",
    "CheckpointSerializationError",
    "checkpoint_config",
    "checkpoint_ref_from_tuple",
    "checkpoint_serializer",
    "open_sqlite_checkpointer",
]
