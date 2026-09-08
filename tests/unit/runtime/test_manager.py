from __future__ import annotations

import asyncio
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.domain import ResourceUsage, RunEvent, RunResult
from deepresearch.runtime.admission import Admission, NoOpAdmissionController
from deepresearch.runtime.checkpoints import checkpoint_serializer
from deepresearch.runtime.deployment_policy import DeploymentPolicy, PolicyViolation
from deepresearch.runtime.manager import (
    CheckpointResumeUnavailable,
    IdempotencyConflict,
    MissingPricingSnapshot,
    RunManager,
    RunNotFound,
    owner_scope_sha256,
    requested_admission_cost,
)
from deepresearch.runtime.runner_factory import FilePricingCatalog, ProviderProfileDrift
from deepresearch.runtime.state_machine import InvalidTransition
from tests.fakes.service_store import FakeRunStore, make_record
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_runner_factory_execution import freeze, pricing

OWNER = owner_scope_sha256(client_ip="127.0.0.1", session_id="local")


class ControlledRunner:
    def __init__(self):
        self.started = asyncio.Event()
        self.finish = asyncio.Event()
        self.calls = []
        self.cost = None
        self.error = None

    async def run(self, **kwargs):
        self.calls.append(kwargs)
        self.started.set()
        while not self.finish.is_set() and not kwargs["cancellation_token"].is_cancelled():
            await asyncio.sleep(0)
        if self.error:
            raise self.error
        return RunResult(
            run_id=kwargs["run_id"],
            thread_id=kwargs["thread_id"],
            status="completed",
            is_partial=False,
            report_artifact_id="report",
            manifest_artifact_id="manifest",
            final_usage=ResourceUsage.zero().model_copy(update={"cost_usd": self.cost}),
        )


class Factory:
    def __init__(self, conf):
        self.routes = freeze(conf, [])
        self.runner = ControlledRunner()
        self.creates = []
        self.required = set()

    def resolve_provider_routes(self, profile):
        return self.routes

    def required_pricing_keys(self, conf, routes):
        return self.required

    def create(self, **kwargs):
        self.creates.append(kwargs)
        return self.runner


class AdmissionSpy(NoOpAdmissionController):
    def __init__(self):
        self.costs = []
        self.operations = []

    async def admit(self, **kwargs):
        self.costs.append(kwargs["requested_cost_usd"])
        return Admission(f"reservation-{len(self.costs)}", len(self.costs))

    async def settle(self, reservation_id, actual_cost_usd):
        self.operations.append(("settle", reservation_id, actual_cost_usd))

    async def release(self, reservation_id):
        self.operations.append(("release", reservation_id))


def policy(conf):
    return DeploymentPolicy(
        forced_access_profile=conf.request.access_profile,
        allowed_execution_modes=frozenset({conf.request.execution_mode}),
        allowed_provider_profile_ids=frozenset({conf.request.provider_profile_id}),
        allowed_run_purposes=frozenset({conf.request.run_purpose}),
        allowed_budget_presets=frozenset({conf.request.budget_preset}),
        budget_presets={conf.request.budget_preset: conf.budget},
    )


@pytest.fixture
async def rig():
    conf = config()
    factory = Factory(conf)
    store = FakeRunStore()
    admission = AdmissionSpy()
    manager = RunManager(
        runner_factory=factory,
        store=store,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        pricing_catalog=FilePricingCatalog({}),
        deployment_policy=policy(conf),
        admission=admission,
    )
    yield manager, conf, factory, store, admission
    factory.runner.finish.set()
    await manager.shutdown(0)


async def create(manager, conf, **kwargs):
    return await manager.create(conf, client_ip="127.0.0.1", session_id="local", **kwargs)


async def seed(rig, status, **changes):
    _, conf, factory, store, _ = rig
    record = replace(
        make_record("seed", status=status),
        config_json=conf.model_dump(mode="json"),
        provider_profile_json=factory.routes.model_dump(mode="json"),
        provider_profile_sha256=factory.routes.configuration_sha256,
        owner_scope_sha256=OWNER,
    )
    return await store.create_run(replace(record, **changes))


async def test_concurrent_idempotency_and_changed_config_conflict(rig):
    manager, conf, factory, _, admission = rig
    first, second = await asyncio.gather(
        create(manager, conf, idempotency_key="k"),
        create(manager, conf, idempotency_key="k"),
    )
    assert first.run_id == second.run_id
    assert len(factory.creates) == len(admission.costs) == 1
    with pytest.raises(IdempotencyConflict):
        await create(manager, conf.model_copy(update={"seed": 99}), idempotency_key="k")


async def test_idempotency_owner_scope_and_nonexistent_are_indistinguishable(rig):
    manager, conf, _, _, _ = rig
    first = await create(manager, conf, idempotency_key="k")
    second = await manager.create(
        conf, client_ip="127.0.0.1", session_id="other", idempotency_key="k"
    )
    assert first.run_id != second.run_id
    for run_id in (first.run_id, "absent"):
        for operation in (manager.get, manager.cancel, manager.subscribe):
            with pytest.raises(RunNotFound):
                await operation(run_id, owner_scope_sha256="wrong")
        with pytest.raises(RunNotFound):
            await manager.resume(run_id, client_ip="other", session_id="other")


@pytest.mark.parametrize("status", ["queued", "interrupted", "cancelled"])
async def test_cancel_inactive_is_idempotent_and_releases_persisted_reservation(rig, status):
    manager, _, _, store, admission = rig
    await seed(rig, status, admission_reservation_id="durable", admission_attempt_no=4)
    first = await manager.cancel("seed", owner_scope_sha256=OWNER)
    second = await manager.cancel("seed", owner_scope_sha256=OWNER)
    assert first.status == second.status == "cancelled"
    assert second.stop_reason is None
    assert (await store.get_run("seed")).admission_reservation_id is None
    assert ("release", "durable") in admission.operations
    assert len(await store.list_events_after("seed", 0)) == (0 if status == "cancelled" else 1)


@pytest.mark.parametrize("status", ["completed", "failed"])
async def test_cancel_terminal_conflicts(rig, status):
    await seed(rig, status)
    with pytest.raises(InvalidTransition):
        await rig[0].cancel("seed", owner_scope_sha256=OWNER)


async def test_cancel_queued_prevents_runner_start(rig):
    manager, conf, factory, store, _ = rig
    view = await create(manager, conf)
    assert view.status == "queued"
    cancelled = await manager.cancel(view.run_id, owner_scope_sha256=OWNER)
    await manager.wait(view.run_id)
    assert cancelled.status == "cancelled"
    assert not factory.runner.calls
    assert len(await store.list_events_after(view.run_id, 0)) == 1


async def test_running_cancel_preserves_core_artifacts_and_usage(rig):
    manager, conf, factory, _, _ = rig
    view = await create(manager, conf)
    await factory.runner.started.wait()
    assert (
        await manager.resume(view.run_id, client_ip="127.0.0.1", session_id="local")
    ).status == "running"
    await manager.cancel(view.run_id, owner_scope_sha256=OWNER)
    final = await manager.wait(view.run_id)
    assert (final.status, final.stop_reason, final.error_code) == (
        "cancelled",
        None,
        "CANCELLED_BY_USER",
    )
    assert final.report_artifact_id == "report"
    assert len(factory.creates) == 1


async def test_finalization_precedes_settlement_clear_and_subscriber_notification(rig):
    manager, conf, factory, store, admission = rig
    factory.required = {("p", "complete", "m")}
    manager.pricing_catalog = FilePricingCatalog({"offline": (pricing("p", "complete", "m"),)})
    factory.runner.cost = Decimal("0.3")
    view = await create(manager, conf)
    subscription = await manager.subscribe(view.run_id, owner_scope_sha256=OWNER)
    original = admission.settle

    async def settle(reservation_id, actual_cost_usd):
        saved = await store.get_run(view.run_id)
        assert saved.status == "completed" and saved.report_artifact_id == "report"
        assert (await store.list_events_after(view.run_id, 0))[-1].status == "completed"
        await original(reservation_id, actual_cost_usd)

    admission.settle = settle
    factory.runner.finish.set()
    await asyncio.wait_for(subscription.wait(), 2)
    saved = await store.get_run(view.run_id)
    assert saved.admission_reservation_id is None
    assert admission.operations == [("settle", "reservation-1", Decimal("0.3"))]
    assert factory.creates[0]["pricing_snapshots"] == saved.pricing_snapshots
    assert factory.creates[0]["checkpointer"] is manager.checkpointer
    await subscription.close()


async def test_emit_persists_and_broadcast_does_not_duplicate(rig):
    manager, conf, _, store, _ = rig
    view = await create(manager, conf)
    sub = await manager.subscribe(view.run_id, owner_scope_sha256=OWNER)
    event = RunEvent(
        seq=1,
        run_id=view.run_id,
        timestamp=datetime.now(UTC),
        node="test",
        kind="progress",
        status="running",
        public_payload={},
        usage_delta=ResourceUsage.zero(),
        artifact_ids=(),
    )
    await manager.emit(event)
    await asyncio.wait_for(sub.wait(), 1)
    await manager.broadcast_persisted(event)
    await asyncio.wait_for(sub.wait(), 1)
    assert await store.list_events_after(view.run_id, 0) == [event]
    await sub.close()


async def test_pricing_and_route_binding_and_policy_refuse_before_admission(rig):
    manager, conf, factory, _, admission = rig
    factory.routes = freeze(
        conf.model_copy(
            update={"request": conf.request.model_copy(update={"execution_mode": "live"})}
        ),
        [],
    )
    with pytest.raises(ProviderProfileDrift):
        await create(manager, conf)
    factory.routes = freeze(conf, [])
    formal = conf.model_copy(
        update={"request": conf.request.model_copy(update={"run_purpose": "benchmark"})}
    )
    manager.deployment_policy = policy(formal)
    factory.required = {("p", "complete", "m")}
    with pytest.raises(MissingPricingSnapshot):
        await create(manager, formal)
    with pytest.raises(PolicyViolation):
        await create(manager, conf)
    assert not admission.costs and not factory.creates


async def test_unknown_and_replay_admission_cost_and_noop_default(rig):
    manager, conf, _, _, admission = rig
    await create(manager, conf)
    assert admission.costs == [Decimal(0)]
    assert requested_admission_cost(conf, "estimated") == Decimal(0)
    other = RunManager(
        runner_factory=rig[2],
        store=FakeRunStore(),
        checkpointer=manager.checkpointer,
        pricing_catalog=manager.pricing_catalog,
        deployment_policy=policy(conf),
    )
    assert isinstance(other.admission, NoOpAdmissionController)
    await other.shutdown(0)


async def test_resume_validates_persisted_routes_before_admission(rig):
    manager, conf, factory, _, admission = rig
    wrong = freeze(
        conf.model_copy(
            update={"request": conf.request.model_copy(update={"execution_mode": "live"})}
        ),
        [],
    )
    await seed(
        rig,
        "interrupted",
        provider_profile_json=wrong.model_dump(mode="json"),
        provider_profile_sha256=wrong.configuration_sha256,
    )
    with pytest.raises(ProviderProfileDrift):
        await manager.resume("seed", client_ip="127.0.0.1", session_id="local")
    assert not admission.costs and not factory.creates


async def test_shutdown_interrupts_active_and_refuses_new_work(rig):
    manager, conf, factory, store, _ = rig
    view = await create(manager, conf)
    await factory.runner.started.wait()
    await manager.shutdown(0)
    saved = await store.get_run(view.run_id)
    assert (saved.status, saved.stop_reason, saved.is_partial, saved.error_code) == (
        "interrupted",
        None,
        True,
        "SERVICE_SHUTDOWN",
    )
    assert (await store.list_events_after(view.run_id, 0))[-1].status == "interrupted"
    with pytest.raises(RuntimeError):
        await create(manager, conf)


async def test_estimated_missing_actual_cost_fails_closed(rig):
    manager, conf, factory, _, admission = rig
    factory.required = {("p", "complete", "m")}
    manager.pricing_catalog = FilePricingCatalog({"offline": (pricing("p", "complete", "m"),)})
    view = await create(manager, conf)
    factory.runner.finish.set()
    result = await manager.wait(view.run_id)
    assert (result.status, result.error_code) == ("failed", "PRICING_INCOMPLETE")
    assert not any(item[0] == "settle" for item in admission.operations)


async def test_resume_capability_refusal_keeps_interrupted_row_and_ignores_catalog(rig):
    manager, conf, factory, store, admission = rig
    original = await seed(rig, "interrupted")
    factory.routes = freeze(
        conf.model_copy(
            update={"request": conf.request.model_copy(update={"execution_mode": "live"})}
        ),
        [],
    )
    with pytest.raises(CheckpointResumeUnavailable):
        await manager.resume("seed", client_ip="127.0.0.1", session_id="local")
    assert await store.get_run("seed") == original
    assert not admission.costs and not factory.creates


async def test_settlement_failure_retains_recovery_link_and_does_not_broadcast(rig):
    manager, conf, factory, store, admission = rig
    factory.required = {("p", "complete", "m")}
    manager.pricing_catalog = FilePricingCatalog({"offline": (pricing("p", "complete", "m"),)})
    factory.runner.cost = Decimal("0.3")
    view = await create(manager, conf)
    subscription = await manager.subscribe(view.run_id, owner_scope_sha256=OWNER)

    async def broken_settle(reservation_id, actual_cost_usd):
        raise OSError("database unavailable")

    admission.settle = broken_settle
    factory.runner.finish.set()
    with pytest.raises(OSError):
        await manager.wait(view.run_id)
    saved = await store.get_run(view.run_id)
    assert saved.status == "completed" and saved.admission_reservation_id == "reservation-1"
    assert not admission.operations
    with pytest.raises(TimeoutError):
        await asyncio.wait_for(subscription.wait(), 0.01)
    await subscription.close()


async def test_database_idempotency_race_releases_losing_reservation(rig):
    manager, conf, factory, store, first_admission = rig
    other_admission = AdmissionSpy()
    second = RunManager(
        runner_factory=factory,
        store=store,
        checkpointer=manager.checkpointer,
        pricing_catalog=manager.pricing_catalog,
        deployment_policy=manager.deployment_policy,
        admission=other_admission,
    )
    reached = asyncio.Event()
    release = asyncio.Event()
    original_create = store.create_run
    arrivals = 0

    async def racing_create(record):
        nonlocal arrivals
        arrivals += 1
        if arrivals == 2:
            reached.set()
        await release.wait()
        return await original_create(record)

    store.create_run = racing_create
    left = asyncio.create_task(create(manager, conf, idempotency_key="race"))
    right = asyncio.create_task(create(second, conf, idempotency_key="race"))
    await asyncio.wait_for(reached.wait(), 1)
    release.set()
    first, duplicate = await asyncio.gather(left, right)
    assert first.run_id == duplicate.run_id
    assert len(first_admission.costs) == len(other_admission.costs) == 1
    assert len(first_admission.operations) + len(other_admission.operations) == 1
    factory.runner.finish.set()
    await manager.shutdown(0)
    await second.shutdown(0)
    assert len(factory.runner.calls) == 1


async def test_exception_is_stable_public_failure_and_unknown_cost_stays_unknown(rig):
    manager, conf, factory, store, _ = rig
    factory.runner.error = RuntimeError("sk-secret-value")
    view = await create(manager, conf)
    factory.runner.finish.set()
    result = await manager.wait(view.run_id)
    assert result.status == "failed" and result.error_code == "INTERNAL_ERROR"
    assert result.final_usage.cost_usd is None
    assert "sk-secret-value" not in repr(await store.list_events_after(view.run_id, 0))


async def test_task_creation_failure_releases_durable_admission(rig, monkeypatch):
    manager, conf, _, store, admission = rig

    def unavailable(coroutine):
        raise RuntimeError("task scheduling unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(asyncio, "create_task", unavailable)
        with pytest.raises(RuntimeError, match="task scheduling unavailable"):
            await create(manager, conf, idempotency_key="scheduling-failure")
    saved = await store.get_by_idempotency(OWNER, "scheduling-failure")
    assert saved.status == "interrupted"
    assert saved.admission_reservation_id is None
    assert admission.operations == [("release", "reservation-1")]
    assert (await store.list_events_after(saved.run_id, 0))[-1].kind == "run_interrupted"
