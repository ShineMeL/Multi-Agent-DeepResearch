"""Service/Core audit integration using deterministic offline providers."""

import asyncio
import time

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.runtime.checkpoints import checkpoint_serializer
from deepresearch.runtime.manager import (
    CheckpointResumeUnavailable,
    MissingPricingSnapshot,
    RunManager,
    owner_scope_sha256,
)
from deepresearch.runtime.manifest import RunManifest
from deepresearch.runtime.runner_factory import (
    FilePricingCatalog,
    FileProviderRouteCatalog,
    LangGraphServiceRunnerFactory,
    ResearchGraphUnavailable,
)
from deepresearch.workflow.runner import BaselineRuntimeHooks
from tests.fakes.service_store import FakeRunStore
from tests.integration.replay.test_baseline_graph import ControlledSegmentClock
from tests.unit.runtime.test_manager import AdmissionSpy, policy
from tests.unit.runtime.test_runner_factory_execution import composition


def setup(tmp_path):
    builder, conf, routes, snapshots, calls, artifacts = composition(tmp_path)
    factory = LangGraphServiceRunnerFactory(builder, FileProviderRouteCatalog({"offline": routes}))
    original_create = factory.create

    def create(**kwargs):
        runner = original_create(**kwargs)
        clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
        runner._runtime_hooks = BaselineRuntimeHooks(
            monotonic=clock.monotonic, utc_now=clock.utc_now
        )
        return runner

    factory.create = create
    saver = InMemorySaver(serde=checkpoint_serializer())
    store = FakeRunStore()
    admission = AdmissionSpy()
    manager = RunManager(
        runner_factory=factory,
        store=store,
        checkpointer=saver,
        pricing_catalog=FilePricingCatalog({"offline": snapshots}),
        deployment_policy=policy(conf),
        admission=admission,
    )
    return manager, conf, calls, artifacts, admission


async def test_real_core_create_uses_durable_sink_and_freezes_pricing_manifest(tmp_path):
    manager, conf, calls, artifacts, admission = setup(tmp_path)
    try:
        view = await manager.create(conf, client_ip="local", session_id="local")
        result = await manager.wait(view.run_id)
        assert result.status == "completed", result.error_code
        manifest = RunManifest.model_validate_json(artifacts.get_bytes(result.manifest_artifact_id))
        record = await manager.store.get_run(view.run_id)
        assert manifest.pricing_snapshots == record.pricing_snapshots
        assert manifest.provider_profiles[0].configuration_sha256 == record.provider_profile_sha256
        assert result.final_usage.cost_usd is not None
        assert calls
        events = await manager.store.list_events_after(view.run_id, 0)
        assert events[-1].kind == "run_completed"
        assert [event.seq for event in events] == list(range(1, len(events) + 1))
        assert ("settle", "reservation-1", result.final_usage.cost_usd) in admission.operations
    finally:
        await manager.shutdown(0)


async def test_real_shutdown_checkpoint_cannot_be_resumed_without_core_changes(tmp_path):
    manager, conf, calls, _, admission = setup(tmp_path)
    reached = asyncio.Event()
    proceed = asyncio.Event()
    original_emit = manager.emit

    async def emit(event):
        await original_emit(event)
        if event.node == "RankEvidence":
            reached.set()
            await proceed.wait()

    manager.emit = emit
    view = await manager.create(conf, client_ip="local", session_id="local")
    await asyncio.wait_for(reached.wait(), 5)
    shutdown = asyncio.create_task(manager.shutdown(0))
    await asyncio.sleep(0)
    await asyncio.sleep(0)
    proceed.set()
    await asyncio.wait_for(shutdown, 5)
    interrupted = await manager.store.get_run(view.run_id)
    assert interrupted.status == "interrupted"
    checkpoint = await manager.checkpointer.aget_tuple(
        {"configurable": {"thread_id": view.thread_id}}
    )
    assert checkpoint is not None
    # Real Core's cooperative cancellation becomes terminal graph error state.
    assert checkpoint.checkpoint["channel_values"]["error_code"] == "CANCELLED"
    events_before = await manager.store.list_events_after(view.run_id, 0)
    # Core's next logical sequence is now occupied by the service terminal.
    assert events_before[-1].seq == checkpoint.checkpoint["channel_values"]["next_event_seq"]
    assert events_before[-1].kind == "run_interrupted"
    before_calls = calls.copy()
    before_admissions = len(admission.costs)
    second = RunManager(
        runner_factory=manager.runner_factory,
        store=manager.store,
        checkpointer=manager.checkpointer,
        pricing_catalog=manager.pricing_catalog,
        deployment_policy=manager.deployment_policy,
        admission=admission,
    )
    with pytest.raises(CheckpointResumeUnavailable) as error:
        await second.resume(view.run_id, client_ip="local", session_id="local")
    assert error.value.code == "CHECKPOINT_RESUME_UNAVAILABLE"
    assert calls == before_calls and len(admission.costs) == before_admissions
    assert await manager.store.list_events_after(view.run_id, 0) == events_before
    assert (
        await second.get(
            view.run_id,
            owner_scope_sha256=owner_scope_sha256(client_ip="local", session_id="local"),
        )
    ).status == "interrupted"
    await second.shutdown(0)


async def test_public_missing_audited_pricing_refuses_before_core_or_admission(tmp_path):
    manager, conf, calls, _, admission = setup(tmp_path)
    public = conf.model_copy(
        update={"request": conf.request.model_copy(update={"access_profile": "public_live"})}
    )
    manager.deployment_policy = policy(public)
    manager.pricing_catalog = FilePricingCatalog({})
    with pytest.raises(MissingPricingSnapshot):
        await manager.create(public, client_ip="public", session_id="public")
    assert not calls and not admission.costs
    await manager.shutdown(0)


async def test_unsupported_research_composition_refuses_before_admission(tmp_path):
    manager, conf, calls, _, admission = setup(tmp_path)
    research = conf.model_copy(
        update={"workflow_id": "research-v1", "planner_id": "P2", "ranker_id": "R2"}
    )
    with pytest.raises(ResearchGraphUnavailable) as error:
        await manager.create(research, client_ip="local", session_id="local")
    assert error.value.code == "RESEARCH_GRAPH_UNAVAILABLE"
    assert not calls and not admission.costs
    await manager.shutdown(0)
