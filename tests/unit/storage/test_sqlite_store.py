from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from deepresearch.domain import ResourceUsage
from deepresearch.storage.protocols import RunFinalization, TerminalEventDraft
from tests.fakes.service_store import make_record


async def test_duplicate_event_sequence_is_rejected_and_gaps_forbidden(store):
    from sqlalchemy.exc import IntegrityError

    from deepresearch.domain import RunEvent

    event = RunEvent(
        run_id="r",
        seq=1,
        timestamp=datetime.now(UTC),
        node="test",
        kind="progress",
        status="running",
        public_payload={},
        usage_delta=ResourceUsage.zero(),
        artifact_ids=(),
        error_code=None,
    )
    await store.append_event(event)
    with pytest.raises(IntegrityError):
        await store.append_event(event)
    with pytest.raises(ValueError):
        await store.append_event(event.model_copy(update={"seq": 3}))
    assert await store.list_events_after("r", 0) == [event]


async def test_concurrent_finalizers_produce_one_terminal_event(store):
    await store.create_run(make_record("r", status="running"))
    finalization = RunFinalization(
        "completed", None, False, None, None, None, ResourceUsage.zero(), None
    )
    draft = TerminalEventDraft(datetime.now(UTC), "test", "done", {})
    outcomes = await asyncio.gather(
        *(store.finalize_run("r", "running", finalization, draft) for _ in range(3)),
        return_exceptions=True,
    )
    assert sum(isinstance(result, ValueError) for result in outcomes) == 2
    assert (await store.get_run("r")).version == 2
    assert len(await store.list_events_after("r", 0)) == 1


async def test_recovery_failure_rolls_back_all_stale_runs(store):
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    await store.create_run(make_record("a", status="running"))
    await store.create_run(make_record("b", status="queued"))
    async with store.engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TRIGGER reject_b BEFORE INSERT ON run_events "
                "WHEN NEW.run_id = 'b' BEGIN SELECT RAISE(ABORT, 'failed'); END"
            )
        )
    with pytest.raises(IntegrityError):
        await store.reconcile_startup(datetime.now(UTC))
    assert (await store.get_run("a")).status == "running"
    assert (await store.get_run("b")).status == "queued"
    assert await store.list_events_after("a", 0) == []


async def test_ledger_ids_attempts_and_settlement_survive_reopening(store, tmp_path):
    from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

    day = date(2026, 8, 29)
    first = await store.reserve_daily_cost(day, "r", Decimal("3.00000000001"), Decimal(10))
    await store.release_daily_cost(first.reservation_id)
    second = await store.reserve_daily_cost(day, "r", Decimal(3), Decimal(10))
    assert (first.attempt_no, second.attempt_no) == (1, 2)
    assert first.reservation_id != second.reservation_id
    other = SqlAlchemyRunStore(str(store.engine.url), tmp_path)
    try:
        await other.settle_daily_cost(second.reservation_id, Decimal("2.00000000001"))
        await other.settle_daily_cost(second.reservation_id, Decimal(100))
        await other.release_daily_cost(second.reservation_id)
        assert await other.ledger_state(second.reservation_id) == "settled"
        with pytest.raises(ValueError):
            await other.reserve_daily_cost(day, "third", Decimal(8), Decimal(10))
        await other.reserve_daily_cost(day, "third", Decimal("7.99999999999"), Decimal(10))
    finally:
        await other.engine.dispose()


async def test_terminal_recovery_settles_known_cost_and_retains_unknown_cost(store):
    day = date(2026, 8, 29)
    settled = await store.reserve_daily_cost(day, "known", Decimal(3), Decimal(10))
    released = await store.reserve_daily_cost(day, "unknown", Decimal(3), Decimal(10))
    await store.create_run(
        replace(
            make_record("known", status="completed"),
            final_usage=ResourceUsage.zero(cost_known=True),
        )
    )
    await store.create_run(make_record("unknown", status="failed"))
    assert (await store.reconcile_startup(datetime.now(UTC))).released_orphan_reservation_ids == ()
    assert await store.ledger_state(settled.reservation_id) == "settled"
    assert await store.ledger_state(released.reservation_id) == "reserved"


async def test_concurrent_migration_initialization_is_idempotent(tmp_path):
    from deepresearch.storage.migrations.runner import upgrade_service_schema
    from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

    url = f"sqlite+aiosqlite:///{tmp_path / 'fresh.db'}"
    stores = [SqlAlchemyRunStore(url, tmp_path) for _ in range(3)]
    try:
        await asyncio.gather(*(upgrade_service_schema(item.engine) for item in stores))
        await stores[0].create_run(make_record("r", status="queued"))
        assert (await stores[1].get_run("r")).status == "queued"
    finally:
        for item in stores:
            await item.engine.dispose()


@pytest.fixture
async def store(tmp_path: Path):
    from deepresearch.storage.migrations.runner import upgrade_service_schema
    from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

    instance = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}", tmp_path)
    await upgrade_service_schema(instance.engine)
    yield instance
    await instance.engine.dispose()


async def test_restart_is_durable_and_retains_linked_reservations(store):
    at = datetime(2026, 8, 29, tzinfo=UTC)
    linked = await store.reserve_daily_cost(at.date(), "a", Decimal(1), Decimal(10))
    orphan = await store.reserve_daily_cost(at.date(), "missing", Decimal(1), Decimal(10))
    await store.create_run(
        replace(
            make_record("a", status="queued"),
            admission_reservation_id=linked.reservation_id,
            admission_attempt_no=linked.attempt_no,
        )
    )
    await store.create_run(make_record("b", status="running"))
    recovery = await store.reconcile_startup(at)
    assert recovery.interrupted_run_ids == ("a", "b")
    assert recovery.released_orphan_reservation_ids == (orphan.reservation_id,)
    for run_id, partial in (("a", False), ("b", True)):
        record = await store.get_run(run_id)
        assert (record.status, record.is_partial, record.error_code) == (
            "interrupted",
            partial,
            "PROCESS_RESTART",
        )
        assert (record.updated_at, record.version) == (at, 2)
        events = await store.list_events_after(run_id, 0)
        assert len(events) == 1
        assert events[0].status == "interrupted"
    assert await store.ledger_state(linked.reservation_id) == "settled"
    assert await store.ledger_state(orphan.reservation_id) == "released"
    assert (await store.reconcile_startup(at)).interrupted_run_ids == ()


async def test_finalization_failure_rolls_back_status_and_event(store):
    from sqlalchemy import text
    from sqlalchemy.exc import IntegrityError

    await store.create_run(make_record("r", status="running"))
    async with store.engine.begin() as conn:
        await conn.execute(
            text(
                "CREATE TRIGGER reject_event BEFORE INSERT ON run_events "
                "BEGIN SELECT RAISE(ABORT, 'test failure'); END"
            )
        )
    result = RunFinalization(
        "completed", None, False, "report", "evidence", "manifest", ResourceUsage.zero(), None
    )
    with pytest.raises(IntegrityError):
        await store.finalize_run(
            "r", "running", result, TerminalEventDraft(datetime.now(UTC), "test", "done", {})
        )
    assert (await store.get_run("r")).status == "running"
    assert await store.list_events_after("r", 0) == []


async def test_independent_store_instances_do_not_overreserve(store, tmp_path):
    from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

    other = SqlAlchemyRunStore(str(store.engine.url), tmp_path)
    try:
        results = await asyncio.gather(
            store.reserve_daily_cost(date(2026, 8, 29), "a", Decimal(7), Decimal(10)),
            other.reserve_daily_cost(date(2026, 8, 29), "b", Decimal(7), Decimal(10)),
            return_exceptions=True,
        )
        assert sum(isinstance(item, ValueError) for item in results) == 1
        assert sum(not isinstance(item, BaseException) for item in results) == 1
    finally:
        await other.engine.dispose()


async def test_migrations_are_atomic_and_idempotent(store, monkeypatch):
    from sqlalchemy import inspect, select, text

    from deepresearch.storage.migrations import runner
    from deepresearch.storage.models import ServiceSchemaVersion

    await runner.upgrade_service_schema(store.engine)
    async with store.engine.connect() as conn:
        assert set(await conn.run_sync(lambda sync: inspect(sync).get_table_names())) == {
            "runs",
            "run_events",
            "artifacts",
            "usage_ledger",
            "service_schema_versions",
        }
        assert (await conn.execute(select(ServiceSchemaVersion.version))).scalars().all() == [1]

    async def fail(conn):
        await conn.execute(text("CREATE TABLE should_rollback (id INTEGER)"))
        raise RuntimeError("boom")

    monkeypatch.setattr(
        runner, "MIGRATIONS", (*runner.MIGRATIONS, runner.ServiceMigration(2, fail))
    )
    with pytest.raises(runner.ServiceMigrationError) as error:
        await runner.upgrade_service_schema(store.engine)
    assert error.value.version == 2
    async with store.engine.connect() as conn:
        assert "should_rollback" not in await conn.run_sync(
            lambda sync: inspect(sync).get_table_names()
        )
        assert (
            await conn.scalar(
                select(ServiceSchemaVersion.version).where(ServiceSchemaVersion.version == 2)
            )
            is None
        )
