from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.engine import make_url

from deepresearch.runtime.checkpoints import (
    CheckpointIdentityError,
    checkpoint_ref_from_tuple,
    checkpoint_serializer,
    open_sqlite_checkpointer,
)
from deepresearch.runtime.ports import CheckpointRef

__all__ = ["BaseCheckpointSaver", "latest_checkpoint_ref", "open_service_checkpointer"]


@asynccontextmanager
async def open_service_checkpointer(
    *, database_url: str, sqlite_path: Path
) -> AsyncGenerator[BaseCheckpointSaver[Any]]:
    url = make_url(database_url)
    if url.get_backend_name() == "postgresql":
        dsn = url.set(drivername="postgresql").render_as_string(hide_password=False)
        async with AsyncPostgresSaver.from_conn_string(dsn, serde=checkpoint_serializer()) as saver:
            await saver.setup()
            yield saver
    elif url.get_backend_name() == "sqlite":
        # Preserve Core's symlink checks on the supplied path before normalization.
        if sqlite_path.is_symlink():
            raise ValueError("checkpoint file must not be a symlink")
        path = sqlite_path.resolve()
        if not path.is_absolute():
            raise ValueError("checkpoint path must resolve to an absolute path")
        async with open_sqlite_checkpointer(path) as saver:
            yield saver
    else:
        raise ValueError("service checkpointer requires SQLite or PostgreSQL")


async def latest_checkpoint_ref(
    checkpointer: BaseCheckpointSaver[Any], *, thread_id: str
) -> CheckpointRef | None:
    value = await checkpointer.aget_tuple({"configurable": {"thread_id": thread_id}})
    if value is None:
        return None
    ref = checkpoint_ref_from_tuple(value)
    if ref.thread_id != thread_id:
        raise CheckpointIdentityError()
    return ref
