from pathlib import Path

from tests.contracts.api.test_sse import OWNER, finalize, make_event, make_manager
from tests.fakes.service_store import make_record


async def test_reconnect_with_fresh_manager_and_reopened_sqlite_replays_only_tail(tmp_path: Path):
    from apps.api.sse import event_stream
    from deepresearch.storage.migrations.runner import upgrade_service_schema
    from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

    url = f"sqlite+aiosqlite:///{tmp_path / 'sse.db'}"
    first_store = SqlAlchemyRunStore(url, tmp_path)
    await upgrade_service_schema(first_store.engine)
    first = make_manager(first_store)
    try:
        await first_store.create_run(make_record("r1", status="running"))
        await first.emit(make_event(1))
        initial = event_stream(
            "r1", last_event_id=0, owner_scope_sha256=OWNER, store=first_store, manager=first
        )
        assert (await anext(initial))["id"] == "1"
        await initial.aclose()
        assert not first._subscribers
        await first.emit(make_event(2, status="failed"))
        await finalize(first_store)
    finally:
        await first.shutdown(0)
        await first_store.engine.dispose()

    reopened = SqlAlchemyRunStore(url, tmp_path)
    second = make_manager(reopened)
    try:
        frames = [
            frame
            async for frame in event_stream(
                "r1", last_event_id=1, owner_scope_sha256=OWNER, store=reopened, manager=second
            )
        ]
        assert [frame["id"] for frame in frames] == ["2", "3"]
        assert [frame["event"] for frame in frames] == ["node_finished", "run_completed"]
        assert not second._subscribers
        for cursor in (3, 2**80):
            assert [
                frame
                async for frame in event_stream(
                    "r1",
                    last_event_id=cursor,
                    owner_scope_sha256=OWNER,
                    store=reopened,
                    manager=second,
                )
            ] == []
    finally:
        await second.shutdown(0)
        await reopened.engine.dispose()


async def test_large_cursor_on_live_sqlite_heartbeats_then_closes(tmp_path: Path, monkeypatch):
    import asyncio

    import pytest

    from apps.api import sse
    from deepresearch.storage.migrations.runner import upgrade_service_schema
    from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

    store = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{tmp_path / 'large.db'}", tmp_path)
    await upgrade_service_schema(store.engine)
    manager = make_manager(store)
    monkeypatch.setattr(sse, "HEARTBEAT_SECONDS", 0.01)
    original_list = store.list_events_after

    async def integer_binding_read(run_id: str, seq: int):
        # The shared schema uses PostgreSQL INT4. Keep the actual SQLite read
        # while enforcing the narrower driver's bind range at this boundary.
        assert -(2**31) <= seq <= 2**31 - 1
        return await original_list(run_id, seq)

    monkeypatch.setattr(store, "list_events_after", integer_binding_read)
    stream = sse.event_stream(
        "r1", last_event_id=2**80, owner_scope_sha256=OWNER, store=store, manager=manager
    )
    try:
        await store.create_run(make_record("r1", status="running"))
        await manager.emit(make_event(1))
        assert await asyncio.wait_for(anext(stream), 1) == {"comment": "heartbeat"}
        await finalize(store)
        with pytest.raises(StopAsyncIteration):
            await asyncio.wait_for(anext(stream), 1)
        assert not manager._subscribers
    finally:
        await stream.aclose()
        await manager.shutdown(0)
        await store.engine.dispose()
