from __future__ import annotations

from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver
from sqlalchemy import func, select

from apps.api import create_app
from deepresearch.domain import RunBudget
from deepresearch.runtime.checkpoints import checkpoint_serializer
from deepresearch.runtime.limits import LimitManager
from deepresearch.runtime.manager import RunManager, owner_scope_sha256
from deepresearch.runtime.runner_factory import FilePricingCatalog
from deepresearch.storage import LocalArtifactStore
from deepresearch.storage.models import RunRow, UsageLedgerRow
from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore
from tests.fakes.service_store import make_record
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_limits import FakeClock, admit
from tests.unit.runtime.test_manager import Factory, policy
from tests.unit.runtime.test_runner_factory_execution import pricing
from tests.unit.storage.test_sqlite_store import store as store  # noqa: PLC0414 - pytest fixture


def public_config(execution_mode="live"):
    conf = config()
    return conf.model_copy(
        update={
            "request": conf.request.model_copy(
                update={
                    "execution_mode": execution_mode,
                    "access_profile": "public_live",
                    "budget_preset": "medium",
                }
            ),
            "budget": RunBudget.preset("medium"),
        }
    )


def setup_service(store, tmp_path, *, daily_limit="10", clock=None, execution_mode="live"):
    conf = public_config(execution_mode)
    factory = Factory(conf)
    factory.required = {("p", "complete", "m")}
    factory.runner.cost = Decimal("0.10")
    limits = LimitManager(store, daily_limit=Decimal(daily_limit), clock=clock or FakeClock())
    manager = RunManager(
        runner_factory=factory,
        store=store,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        pricing_catalog=FilePricingCatalog({"offline": (pricing("p", "complete", "m"),)}),
        deployment_policy=policy(conf),
        admission=limits,
    )
    app = create_app(
        manager=manager,
        deployment_policy=manager.deployment_policy,
        artifact_store=LocalArtifactStore(tmp_path),
        session_secret=b"s" * 32,
    )
    body = {
        "request": conf.request.model_dump(mode="json"),
        "workflow_id": "baseline-v1",
        "planner_id": "P1",
        "ranker_id": "R1",
    }
    return app, manager, factory, limits, body


async def post(app, body, ip="127.0.0.1"):
    async with httpx.AsyncClient(
        transport=httpx.ASGITransport(app=app, client=(ip, 1234), raise_app_exceptions=False),
        base_url="https://testserver",
    ) as client:
        return await client.post("/runs", json=body)


@pytest.mark.parametrize("ledger_unavailable", [False, True])
async def test_replay_forced_public_live_needs_no_daily_reservation(
    store,
    tmp_path,
    monkeypatch,
    ledger_unavailable,
):
    app, manager, factory, _, body = setup_service(
        store,
        tmp_path,
        daily_limit="0",
        execution_mode="replay",
    )
    body["request"]["access_profile"] = "local"
    factory.runner.finish.set()

    async def unavailable(*args):
        raise OSError("daily ledger unavailable")

    if ledger_unavailable:
        monkeypatch.setattr(store, "reserve_daily_cost", unavailable)
    try:
        response = await post(app, body)
        assert response.status_code == 202
        run_id = response.json()["run_id"]
        accepted = await store.get_run(run_id)
        assert accepted.config_json["request"]["access_profile"] == "public_live"
        assert accepted.config_json["request"]["execution_mode"] == "replay"
        assert accepted.admission_reservation_id is None
        assert accepted.admission_attempt_no is None
        assert (await manager.wait(run_id)).status == "completed"
        async with store.session_factory() as session:
            assert (
                await session.scalar(
                    select(func.count())
                    .select_from(UsageLedgerRow)
                    .where(UsageLedgerRow.run_id == run_id)
                )
                == 0
            )
    finally:
        await manager.shutdown(0)


async def test_third_run_returns_public_429_without_creating_a_run(store, tmp_path):
    app, manager, _, _, body = setup_service(store, tmp_path)
    try:
        assert (await post(app, body, "1.1.1.1")).status_code == 202
        assert (await post(app, body, "1.1.1.2")).status_code == 202
        denied = await post(app, body, "1.1.1.3")
        assert denied.status_code == 429
        assert denied.json() == {
            "code": "RATE_LIMITED",
            "message": "Service capacity or usage limit reached.",
            "run_id": None,
            "retry_after": 1,
        }
        assert denied.headers["Retry-After"] == "1"
        async with store.session_factory() as session:
            assert await session.scalar(select(func.count()).select_from(RunRow)) == 2
    finally:
        await manager.shutdown(0)


async def test_daily_limit_survives_reopened_store_and_manager(store, tmp_path):
    await admit(LimitManager(store, daily_limit=Decimal("0.50")), cost="0.50")
    reopened = SqlAlchemyRunStore(str(store.engine.url), tmp_path)
    app, manager, _, _, body = setup_service(reopened, tmp_path, daily_limit="0.50")
    try:
        response = await post(app, body)
        assert response.status_code == 429
        assert 1 <= response.json()["retry_after"] <= 86400
        assert response.headers["Retry-After"] == str(response.json()["retry_after"])
    finally:
        await manager.shutdown(0)
        await reopened.engine.dispose()


async def test_signed_session_bucket_returns_exact_retry_after(store, tmp_path):
    clock = FakeClock()
    app, manager, factory, _, body = setup_service(store, tmp_path, clock=clock)
    factory.runner.finish.set()
    try:
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app, raise_app_exceptions=False),
            base_url="https://testserver",
        ) as client:
            for _ in range(2):
                response = await client.post("/runs", json=body)
                assert response.status_code == 202
                await manager.wait(response.json()["run_id"])
            denied = await client.post("/runs", json=body)
            assert denied.status_code == 429
            assert denied.headers["Retry-After"] == "120"
            clock.advance(120)
            assert (await client.post("/runs", json=body)).status_code == 202
    finally:
        await manager.shutdown(0)


async def test_restart_then_cancel_releases_persisted_reservation(store, tmp_path):
    first = await admit(LimitManager(store, daily_limit=Decimal(1)), "r1", cost="0.50")
    owner = owner_scope_sha256(client_ip="1", session_id="signed-a")
    await store.create_run(
        replace(
            make_record("r1", status="running"),
            admission_reservation_id=first.reservation_id,
            admission_attempt_no=first.attempt_no,
            owner_scope_sha256=owner,
        )
    )
    await store.reconcile_startup(datetime.now(UTC))
    _, manager, _, limits, _ = setup_service(store, tmp_path, daily_limit="1")
    try:
        await manager.cancel("r1", owner_scope_sha256=owner)
        assert await store.ledger_state(first.reservation_id) == "released"
        assert (await store.get_run("r1")).admission_reservation_id is None
        await admit(limits, "new-a", cost="0.50")
        await admit(limits, "new-b", cost="0.50")
    finally:
        await manager.shutdown(0)


async def test_completed_runs_settle_actual_once_and_keep_frozen_inputs(store, tmp_path):
    app, manager, factory, _, body = setup_service(store, tmp_path, daily_limit="0.80")
    factory.runner.finish.set()
    try:
        first = await post(app, body, "1.1.1.1")
        await manager.wait(first.json()["run_id"])
        saved = await store.get_run(first.json()["run_id"])
        assert saved.final_usage.cost_usd == Decimal("0.10")
        assert saved.admission_reservation_id is None
        assert saved.config_json["budget"]["max_cost_usd"] == "0.50"
        assert saved.pricing_snapshots == factory.creates[0]["pricing_snapshots"]
        # A newer catalog cannot change an already persisted pricing snapshot.
        manager.pricing_catalog = FilePricingCatalog(
            {"offline": (pricing("p", "complete", "m", "2"),)}
        )
        for ip in ("1.1.1.2", "1.1.1.3", "1.1.1.4"):
            response = await post(app, body, ip)
            assert response.status_code == 202
            await manager.wait(response.json()["run_id"])
        assert (await store.get_run(saved.run_id)).pricing_snapshots == saved.pricing_snapshots
        assert (await post(app, body, "1.1.1.5")).status_code == 429
    finally:
        await manager.shutdown(0)


async def test_settlement_failure_keeps_link_until_startup_reconciliation(
    store, tmp_path, monkeypatch
):
    app, manager, factory, _, body = setup_service(store, tmp_path, daily_limit="0.50")
    response = await post(app, body)
    run_id = response.json()["run_id"]
    await factory.runner.started.wait()
    before = await store.get_run(run_id)

    async def fail(*args):
        raise OSError("database unavailable")

    try:
        with monkeypatch.context() as patch:
            patch.setattr(store, "settle_daily_cost", fail)
            factory.runner.finish.set()
            with pytest.raises(OSError):
                await manager.wait(run_id)
        saved = await store.get_run(run_id)
        assert saved.status == "completed"
        assert saved.admission_reservation_id == before.admission_reservation_id
        assert await store.ledger_state(saved.admission_reservation_id) == "reserved"
        await store.reconcile_startup(datetime.now(UTC))
        assert await store.ledger_state(saved.admission_reservation_id) == "settled"
        await store.reserve_daily_cost(
            datetime.now(UTC).date(), "remaining", Decimal("0.40"), Decimal("0.50")
        )
    finally:
        await manager.shutdown(0)
