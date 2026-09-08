from pathlib import Path

import pytest


async def test_postgres_setup_precedes_yield_and_dsn_preserves_escaped_password(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from deepresearch.runtime import checkpointers

    state = {"setup": False, "closed": False}

    async def setup():
        state["setup"] = True

    @asynccontextmanager
    async def connect(dsn, *, serde):
        assert dsn == "postgresql://user:p%40ss@db/research"
        try:
            yield SimpleNamespace(setup=setup, serde=serde)
        finally:
            state["closed"] = True

    monkeypatch.setattr(checkpointers.AsyncPostgresSaver, "from_conn_string", connect)
    async with checkpointers.open_service_checkpointer(
        database_url="postgresql+asyncpg://user:p%40ss@db/research", sqlite_path=Path("unused")
    ) as saver:
        assert state["setup"]
        assert saver.serde.pickle_fallback is False
        assert saver.serde.loads_typed(saver.serde.dumps_typed({"ids": ("one",)})) == {
            "ids": ("one",)
        }
    assert state["closed"]


async def test_postgres_setup_failure_never_yields(monkeypatch):
    from contextlib import asynccontextmanager
    from types import SimpleNamespace

    from deepresearch.runtime import checkpointers

    async def setup():
        raise RuntimeError("setup failed")

    @asynccontextmanager
    async def connect(dsn, *, serde):
        yield SimpleNamespace(setup=setup, serde=serde)

    monkeypatch.setattr(checkpointers.AsyncPostgresSaver, "from_conn_string", connect)
    with pytest.raises(RuntimeError, match="setup failed"):
        async with checkpointers.open_service_checkpointer(
            database_url="postgresql+asyncpg://user:pass@db/research", sqlite_path=Path("unused")
        ):
            pytest.fail("saver must not be yielded before setup completes")


async def test_latest_ref_reads_persisted_concrete_checkpoint(tmp_path):
    from langgraph.checkpoint.base import empty_checkpoint

    from deepresearch.runtime.checkpointers import latest_checkpoint_ref, open_service_checkpointer

    async with open_service_checkpointer(
        database_url="sqlite+aiosqlite:///unused.db", sqlite_path=tmp_path / "saver.db"
    ) as saver:
        checkpoint = empty_checkpoint()
        await saver.aput(
            {"configurable": {"thread_id": "thread", "checkpoint_ns": ""}},
            checkpoint,
            {"source": "input", "step": 0, "parents": {}},
            {},
        )
        ref = await latest_checkpoint_ref(saver, thread_id="thread")
        assert ref.thread_id == "thread"
        assert ref.checkpoint_id == checkpoint["id"]
        assert ref.created_at.isoformat() == checkpoint["ts"]


async def test_sqlite_service_saver_uses_strict_serializer(tmp_path: Path):
    from deepresearch.domain import ResourceUsage
    from deepresearch.runtime.checkpointers import latest_checkpoint_ref, open_service_checkpointer

    async with open_service_checkpointer(
        database_url="sqlite+aiosqlite:///unused.db", sqlite_path=tmp_path / "checkpoints.db"
    ) as saver:
        assert saver.serde.pickle_fallback is False
        state = {"usage": ResourceUsage.zero(), "ids": ("one",)}
        assert saver.serde.loads_typed(saver.serde.dumps_typed(state)) == state
        assert await latest_checkpoint_ref(saver, thread_id="missing") is None
