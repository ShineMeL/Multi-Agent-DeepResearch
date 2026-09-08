"""A missing final bill must not erase durable work or free the daily budget."""

from collections.abc import AsyncGenerator, Awaitable, Callable
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.domain import ResourceUsage, RunConfig, RunEvent, RunResult, RunStatus
from deepresearch.runtime.cancellation import CancellationToken
from deepresearch.runtime.deployment_policy import DeploymentPolicy
from deepresearch.runtime.limits import LimitManager, RateLimitExceeded
from deepresearch.runtime.manager import RunManager
from deepresearch.runtime.manifest import PricingSnapshot
from deepresearch.runtime.ports import CheckpointRef, ResearchRunner
from deepresearch.runtime.runner_factory import FilePricingCatalog, FrozenProviderRoutes
from deepresearch.storage.migrations.runner import upgrade_service_schema
from deepresearch.storage.sqlalchemy_store import SqlAlchemyRunStore
from tests.fakes.service_store import make_record
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_runner_factory_execution import freeze, pricing


def usage_event(run_id: str, seq: int, tokens: int) -> RunEvent:
    return RunEvent(
        run_id=run_id,
        seq=seq,
        timestamp=datetime.now(UTC),
        node="Planner",
        kind="progress",
        status="running",
        public_payload={},
        artifact_ids=(),
        usage_delta=ResourceUsage.zero().model_copy(
            update={"input_tokens": tokens, "total_tokens": tokens}
        ),
    )


@pytest.fixture
async def store(tmp_path: Path) -> AsyncGenerator[SqlAlchemyRunStore]:
    instance = SqlAlchemyRunStore(f"sqlite+aiosqlite:///{tmp_path / 'runs.db'}", tmp_path)
    await upgrade_service_schema(instance.engine)
    try:
        yield instance
    finally:
        await instance.engine.dispose()


class UsageThenFailure:
    def __init__(self, returns_result: bool) -> None:
        self.returns_result = returns_result

    async def run(
        self,
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig,
        checkpoint: CheckpointRef | None,
        emit: Callable[[RunEvent], Awaitable[None]],
        cancellation_token: CancellationToken,
    ) -> RunResult:
        await emit(usage_event(run_id, 1, 30))
        await emit(usage_event(run_id, 2, 20))
        if self.returns_result:
            return RunResult(
                run_id=run_id,
                thread_id=thread_id,
                status="completed",
                is_partial=False,
                report_artifact_id=None,
                manifest_artifact_id=None,
                final_usage=ResourceUsage.zero(),
            )
        raise RuntimeError("provider failed after durable usage")


class FailureFactory:
    def __init__(self, conf: RunConfig, returns_result: bool = False) -> None:
        self.routes = freeze(conf, [])
        self.returns_result = returns_result

    def resolve_provider_routes(self, provider_profile_id: str) -> FrozenProviderRoutes:
        return self.routes

    def required_pricing_keys(
        self,
        config: RunConfig,
        provider_routes: FrozenProviderRoutes,
    ) -> set[tuple[str, str, str]]:
        return {("p", "complete", "m")}

    def create(
        self,
        *,
        config: RunConfig,
        provider_routes: FrozenProviderRoutes,
        pricing_snapshots: tuple[PricingSnapshot, ...],
        checkpointer: BaseCheckpointSaver[Any],
    ) -> ResearchRunner:
        return UsageThenFailure(self.returns_result)


@pytest.mark.parametrize("returns_result", [False, True])
async def test_exception_preserves_usage_and_reservation_without_exhausting_capacity(
    store: SqlAlchemyRunStore,
    returns_result: bool,
) -> None:
    conf = config()
    conf = conf.model_copy(
        update={
            "request": conf.request.model_copy(
                update={"access_profile": "public_live", "execution_mode": "live"}
            )
        }
    )
    limits = LimitManager(store, daily_limit=Decimal("0.25"))
    manager = RunManager(
        runner_factory=FailureFactory(conf, returns_result),
        store=store,
        checkpointer=InMemorySaver(),
        pricing_catalog=FilePricingCatalog({"offline": (pricing("p", "complete", "m"),)}),
        deployment_policy=DeploymentPolicy(
            forced_access_profile="public_live",
            allowed_execution_modes=frozenset({"live"}),
            allowed_provider_profile_ids=frozenset({"offline"}),
            allowed_run_purposes=frozenset({"test"}),
            allowed_budget_presets=frozenset({"low"}),
            budget_presets={"low": conf.budget},
        ),
        admission=limits,
    )
    try:
        view = await manager.create(conf, client_ip="peer", session_id="owner")
        final = await manager.wait(view.run_id)
        assert final.status == "failed"
        assert final.final_usage is not None
        assert final.final_usage.total_tokens == 50
        assert final.final_usage.cost_usd is None
        saved = await store.get_run(view.run_id)
        assert saved is not None and saved.admission_reservation_id is not None
        assert await store.ledger_state(saved.admission_reservation_id) == "reserved"
        await store.reconcile_startup(datetime.now(UTC))
        assert await store.ledger_state(saved.admission_reservation_id) == "reserved"
        with pytest.raises(RateLimitExceeded):
            await limits.admit(
                run_id="next",
                client_ip="other",
                session_id="other",
                access_profile="public_live",
                requested_cost_usd=Decimal("0.25"),
            )
        # Retaining unresolved money must not consume a process execution slot.
        await limits.runs.try_acquire()
        await limits.runs.try_acquire()
        await limits.runs.release()
        await limits.runs.release()
    finally:
        await manager.shutdown(0)


@pytest.mark.parametrize("status", ["running", "failed", "cancelled", "interrupted"])
async def test_restart_preserves_incurred_usage_and_unknown_bill(
    store: SqlAlchemyRunStore,
    status: RunStatus,
) -> None:
    now = datetime.now(UTC)
    admission = await store.reserve_daily_cost(now.date(), "r", Decimal("0.25"), Decimal(1))
    assert admission.reservation_id is not None
    await store.create_run(
        replace(
            make_record("r", status=status),
            admission_reservation_id=admission.reservation_id,
            admission_attempt_no=admission.attempt_no,
        )
    )
    await store.append_event(usage_event("r", 1, 30))
    await store.append_event(usage_event("r", 2, 20))
    await store.reconcile_startup(now)
    await store.reconcile_startup(now)
    saved = await store.get_run("r")
    assert saved is not None and saved.final_usage is not None
    assert saved.final_usage.total_tokens == 50
    assert saved.final_usage.cost_usd is None
    assert saved.admission_reservation_id == admission.reservation_id
    assert await store.ledger_state(admission.reservation_id) == "reserved"
