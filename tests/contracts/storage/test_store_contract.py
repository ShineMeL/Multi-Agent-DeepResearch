import os
from dataclasses import replace
from datetime import UTC, datetime
from uuid import uuid4

import pytest

from deepresearch.domain import ResourceUsage
from deepresearch.storage.protocols import IdempotencyCollision, RunFinalization, TerminalEventDraft
from tests.fakes.service_store import FakeRunStore, make_record


async def test_startup_recovers_queued_and_running_once(store):
    for run_id, status in (("q", "queued"), ("r", "running")):
        await store.create_run(make_record(run_id, status=status))
    at = datetime(2026, 8, 29, tzinfo=UTC)
    recovery = await store.reconcile_startup(at)
    assert recovery.interrupted_run_ids == ("q", "r")
    assert (await store.get_run("q")).is_partial is False
    assert (await store.get_run("r")).is_partial is True
    assert (await store.get_run("r")).error_code == "PROCESS_RESTART"
    assert (await store.reconcile_startup(at)).interrupted_run_ids == ()
    assert len(await store.list_events_after("q", 0)) == 1


@pytest.fixture(params=["fake", "sqlite", "postgres"])
async def store(request, tmp_path):
    if request.param == "fake":
        yield FakeRunStore()
    else:
        from deepresearch.storage.migrations.runner import upgrade_service_schema
        from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore

        url = f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}"
        schema = None
        if request.param == "postgres":
            configured_url = os.environ.get("DEEPRESEARCH_TEST_POSTGRES_URL")
            if configured_url is None:
                pytest.skip("DEEPRESEARCH_TEST_POSTGRES_URL is not configured")
            url = configured_url
            schema = f"test_service_{uuid4().hex}"
        instance = SqlAlchemyRunStore(url, tmp_path)
        try:
            if schema is not None:
                from sqlalchemy.ext.asyncio import async_sessionmaker
                from sqlalchemy.schema import CreateSchema

                async with instance.engine.begin() as connection:
                    await connection.execute(CreateSchema(schema))
                instance.engine = instance.engine.execution_options(
                    schema_translate_map={None: schema}
                )
                instance.session_factory = async_sessionmaker(
                    instance.engine, expire_on_commit=False
                )
            await upgrade_service_schema(instance.engine)
            yield instance
        finally:
            if schema is not None:
                from sqlalchemy.schema import DropSchema

                async with instance.engine.begin() as connection:
                    await connection.execute(DropSchema(schema, cascade=True))
            await instance.engine.dispose()


async def test_exact_ownership_and_scoped_idempotency(store):
    record = replace(make_record("a", status="queued"), idempotency_key="same")
    await store.create_run(record)
    assert await store.create_run(record) == record
    with pytest.raises(IdempotencyCollision):
        await store.create_run(replace(record, run_id="b"))
    await store.create_run(replace(record, run_id="c", idempotency_scope_sha256="e" * 64))
    assert await store.get_owned_run("a", "wrong") is None
    assert await store.get_owned_run("a", record.owner_scope_sha256) == record
    assert await store.get_by_idempotency(record.idempotency_scope_sha256, "same") == record


async def test_terminal_fields_and_cas_are_preserved(store):
    await store.create_run(make_record("r", status="queued"))
    await store.transition("r", "queued", "running")
    with pytest.raises(ValueError):
        await store.transition("r", "queued", "running")
    with pytest.raises(ValueError):
        await store.transition("r", "running", "completed")
    finalization = RunFinalization(
        "completed", None, False, "report", "evidence", "manifest", ResourceUsage.zero(), None
    )
    saved, event = await store.finalize_run(
        "r",
        "running",
        finalization,
        TerminalEventDraft(datetime.now(UTC), "test", "run_completed", {"done": True}),
    )
    assert await store.get_run("r") == saved
    assert saved.final_usage == finalization.final_usage
    assert (
        saved.report_artifact_id,
        saved.evidence_graph_artifact_id,
        saved.manifest_artifact_id,
    ) == ("report", "evidence", "manifest")
    assert (await store.list_events_after("r", 0)) == [event]
    assert event.artifact_ids == ("report", "evidence", "manifest")
