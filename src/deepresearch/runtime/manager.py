"""In-process service lifecycle, durable events, and admission ownership."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Collection
from dataclasses import replace
from datetime import UTC, datetime
from decimal import Decimal
from typing import Any, Literal, Protocol
from uuid import uuid4

from langgraph.checkpoint.base import BaseCheckpointSaver

from deepresearch.domain import ResourceUsage, RunConfig, RunEvent
from deepresearch.runtime.admission import AdmissionController, NoOpAdmissionController
from deepresearch.runtime.cancellation import CancellationToken
from deepresearch.runtime.deployment_policy import DeploymentPolicy
from deepresearch.runtime.manifest import PricingSnapshot
from deepresearch.runtime.ports import ResearchRunner
from deepresearch.runtime.runner_factory import (
    FrozenProviderRoutes,
    PricingCatalog,
    ProviderProfileDrift,
    ServiceRunnerFactory,
    validate_provider_route_binding,
)
from deepresearch.runtime.state_machine import validate_cancel, validate_resume
from deepresearch.storage.protocols import (
    IdempotencyCollision,
    RunFinalization,
    RunRecord,
    RunStore,
    RunView,
    TerminalEventDraft,
)


class RunNotFound(LookupError):
    """Missing runs and owner mismatches deliberately have the same outcome."""


class MissingPricingSnapshot(ValueError):
    code = "MISSING_PRICING_SNAPSHOT"


class IdempotencyConflict(RuntimeError):
    code = "IDEMPOTENCY_CONFLICT"


class CheckpointResumeUnavailable(RuntimeError):
    """Core needs a resumable interruption boundary and shared event sequencing."""

    code = "CHECKPOINT_RESUME_UNAVAILABLE"

    def __init__(self) -> None:
        super().__init__("checkpoint continuation is unavailable in this service version")


class ServiceShuttingDown(RuntimeError):
    code = "SERVICE_SHUTDOWN"


def run_config_sha256(config: RunConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def owner_scope_sha256(*, client_ip: str, session_id: str) -> str:
    return hashlib.sha256(f"{client_ip}\0{session_id}".encode()).hexdigest()


def requested_admission_cost(
    config: RunConfig,
    pricing_status: Literal["estimated", "unknown"],
) -> Decimal:
    if config.request.execution_mode == "replay" or pricing_status == "unknown":
        return Decimal(0)
    cost = config.budget.max_cost_usd
    if cost is None:
        raise MissingPricingSnapshot("priced live runs require a cost budget")
    return cost


class EventSubscription(Protocol):
    async def wait(self) -> None: ...

    async def close(self) -> None: ...


class _Subscription:
    def __init__(self, lock: asyncio.Lock, subscribers: set[_Subscription]) -> None:
        self.event = asyncio.Event()
        self.lock = lock
        self.subscribers = subscribers
        self.closed = False

    async def wait(self) -> None:
        await self.event.wait()
        # No await between waking and clearing; broadcasts coalesce safely.
        if not self.closed:
            self.event.clear()

    async def close(self) -> None:
        async with self.lock:
            self.closed = True
            self.subscribers.discard(self)
            self.event.set()


class _DurableSink:
    """Core's audit boundary needs both publication and durable readback."""

    def __init__(self, manager: RunManager, run_id: str) -> None:
        self.manager = manager
        self.run_id = run_id

    async def __call__(self, event: RunEvent) -> None:
        if event.run_id != self.run_id:
            raise ValueError("event run identity mismatch")
        await self.manager.emit(event)

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        if run_id != self.run_id:
            raise ValueError("event run identity mismatch")
        events = await self.manager.store.list_events_after(run_id, seq - 1)
        return next((event for event in events if event.seq == seq), None)


def _view(record: RunRecord) -> RunView:
    return RunView(
        run_id=record.run_id,
        thread_id=record.thread_id,
        status=record.status,
        stop_reason=record.stop_reason,
        is_partial=record.is_partial,
        report_artifact_id=record.report_artifact_id,
        evidence_graph_artifact_id=record.evidence_graph_artifact_id,
        manifest_artifact_id=record.manifest_artifact_id,
        final_usage=record.final_usage,
        error_code=record.error_code,
    )


def _draft(status: str) -> TerminalEventDraft:
    return TerminalEventDraft(
        timestamp=datetime.now(UTC),
        node="service",
        kind=f"run_{status}",
        public_payload={},
    )


class RunManager:
    def __init__(
        self,
        *,
        runner_factory: ServiceRunnerFactory,
        store: RunStore,
        checkpointer: BaseCheckpointSaver[Any],
        pricing_catalog: PricingCatalog,
        deployment_policy: DeploymentPolicy,
        admission: AdmissionController | None = None,
        secrets: Collection[str] = (),
    ) -> None:
        self.runner_factory = runner_factory
        self.store = store
        self.checkpointer = checkpointer
        self.pricing_catalog = pricing_catalog
        self.deployment_policy = deployment_policy
        self.admission = admission if admission is not None else NoOpAdmissionController()
        # Exception text and configuration never enter public error/event payloads.
        self._secrets = tuple(secrets)
        self._tasks: dict[str, asyncio.Task[None]] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._broadcast_locks: dict[str, asyncio.Lock] = {}
        self._subscribers: dict[str, set[_Subscription]] = {}
        self._user_cancel_requested: set[str] = set()
        self._shutdown_requested: set[str] = set()
        self._closing = False
        self._creation_lock = asyncio.Lock()
        self._shutdown_lock = asyncio.Lock()

    def _lock_for(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    def _ensure_open(self) -> None:
        if self._closing:
            raise ServiceShuttingDown("service is shutting down")

    def _idempotent_view(self, record: RunRecord, digest: str) -> RunView:
        if record.config_sha256 != digest:
            raise IdempotencyConflict("idempotency key was used with a different configuration")
        return _view(record)

    def _pricing(
        self,
        config: RunConfig,
        routes: FrozenProviderRoutes,
    ) -> tuple[Literal["estimated", "unknown"], tuple[PricingSnapshot, ...]]:
        snapshots = self.pricing_catalog.resolve(routes.profile_id)
        keys = {(item.provider_id, item.endpoint_type, item.model_id) for item in snapshots}
        required = self.runner_factory.required_pricing_keys(config, routes)
        if required - keys or not snapshots:
            if (
                config.request.access_profile == "public_live"
                or config.request.run_purpose == "benchmark"
            ):
                raise MissingPricingSnapshot("required pricing snapshot is unavailable")
            return "unknown", ()
        if len(keys) != len(snapshots):
            raise MissingPricingSnapshot("pricing snapshot identities are ambiguous")
        return "estimated", snapshots

    async def create(
        self,
        config: RunConfig,
        *,
        client_ip: str,
        session_id: str,
        idempotency_key: str | None = None,
    ) -> RunView:
        self.deployment_policy.validate_config(config)
        scope = owner_scope_sha256(client_ip=client_ip, session_id=session_id)
        digest = run_config_sha256(config)
        # Serializes local creates and shutdown's admission barrier. Database
        # uniqueness remains authoritative when more than one manager is involved.
        async with self._creation_lock:
            self._ensure_open()
            if idempotency_key is not None:
                existing = await self.store.get_by_idempotency(scope, idempotency_key)
                if existing is not None:
                    return self._idempotent_view(existing, digest)
            run_id = str(uuid4())
            routes = self.runner_factory.resolve_provider_routes(config.request.provider_profile_id)
            validate_provider_route_binding(config, routes)
            pricing_status, snapshots = self._pricing(config, routes)
            requested_cost = requested_admission_cost(config, pricing_status)
            # Construct before admission so unsupported Core graphs and invalid
            # frozen pricing cannot reserve a slot or money.
            runner = self.runner_factory.create(
                config=config,
                provider_routes=routes,
                pricing_snapshots=snapshots,
                checkpointer=self.checkpointer,
            )
            admitted = await self.admission.admit(
                run_id=run_id,
                client_ip=client_ip,
                session_id=session_id,
                access_profile=config.request.access_profile,
                requested_cost_usd=requested_cost,
            )
            record = RunRecord(
                run_id=run_id,
                thread_id=str(uuid4()),
                status="queued",
                config_json=config.model_dump(mode="json"),
                pricing_status=pricing_status,
                pricing_snapshots=snapshots,
                provider_profile_json=routes.model_dump(mode="json"),
                provider_profile_sha256=routes.configuration_sha256,
                config_sha256=digest,
                owner_scope_sha256=scope,
                idempotency_scope_sha256=scope,
                idempotency_key=idempotency_key,
                admission_reservation_id=admitted.reservation_id,
                admission_attempt_no=admitted.attempt_no,
                stop_reason=None,
                is_partial=False,
                report_artifact_id=None,
                evidence_graph_artifact_id=None,
                manifest_artifact_id=None,
                final_usage=None,
                error_code=None,
                updated_at=datetime.now(UTC),
                version=1,
            )
            try:
                saved = await self.store.create_run(record)
            except BaseException as error:
                await self.admission.release(admitted.reservation_id)
                if isinstance(error, IdempotencyCollision) and idempotency_key is not None:
                    existing = await self.store.get_by_idempotency(scope, idempotency_key)
                    if existing is not None:
                        return self._idempotent_view(existing, digest)
                raise
            token = CancellationToken()
            self._tokens[run_id] = token
            execution = self._start(saved, config, runner, token)
            try:
                task = asyncio.create_task(execution)
            except BaseException:
                execution.close()
                self._tokens.pop(run_id, None)
                final = self._inactive_finalization(saved, "interrupted", "TASK_START_FAILED")
                await self._finalize(saved, final, settle=False)
                raise
            self._tasks[run_id] = task
            task.add_done_callback(lambda done: self._forget(run_id, done))
            return _view(saved)

    def _forget(self, run_id: str, task: asyncio.Task[None]) -> None:
        if self._tasks.get(run_id) is task:
            self._tasks.pop(run_id, None)
            self._tokens.pop(run_id, None)
        self._user_cancel_requested.discard(run_id)
        self._shutdown_requested.discard(run_id)
        # Consume task failures; wait() still propagates them to its caller.
        if not task.cancelled():
            task.exception()

    async def _owned(self, run_id: str, scope: str) -> RunRecord:
        record = await self.store.get_owned_run(run_id, scope)
        if record is None:
            raise RunNotFound(run_id)
        return record

    async def get(self, run_id: str, *, owner_scope_sha256: str) -> RunView:
        return _view(await self._owned(run_id, owner_scope_sha256))

    async def resume(self, run_id: str, *, client_ip: str, session_id: str) -> RunView:
        async with self._lock_for(run_id):
            record = await self._owned(
                run_id, owner_scope_sha256(client_ip=client_ip, session_id=session_id)
            )
            self._ensure_open()
            if record.status == "running":
                return _view(record)
            validate_resume(record.status)
            config = RunConfig.model_validate(record.config_json)
            self.deployment_policy.validate_config(config)
            try:
                routes = FrozenProviderRoutes.model_validate(record.provider_profile_json)
            except ValueError:
                raise ProviderProfileDrift() from None
            if routes.configuration_sha256 != record.provider_profile_sha256:
                raise ProviderProfileDrift()
            validate_provider_route_binding(config, routes)
            # Core cancellation checkpoints contain terminal error state. In
            # addition, service finalization occupies Core's next event sequence.
            # Never replay completed nodes or rewrite audited checkpoint/event IDs.
            raise CheckpointResumeUnavailable()

    async def cancel(self, run_id: str, *, owner_scope_sha256: str) -> RunView:
        async with self._lock_for(run_id):
            record = await self._owned(run_id, owner_scope_sha256)
            if record.status == "cancelled":
                await self.admission.release(record.admission_reservation_id)
                return _view(
                    await self.store.clear_admission(run_id, record.admission_reservation_id)
                )
            validate_cancel(record.status)
            if record.status == "running":
                token = self._tokens.get(run_id)
                if token is None:
                    raise CheckpointResumeUnavailable()
                self._user_cancel_requested.add(run_id)
                token.cancel()
                return _view(record)
            if record.status == "queued":
                if token := self._tokens.get(run_id):
                    token.cancel()
                if task := self._tasks.get(run_id):
                    task.cancel()
            final = self._inactive_finalization(record, "cancelled", "CANCELLED_BY_USER")
            return _view(await self._finalize(record, final, settle=False))

    @staticmethod
    def _inactive_finalization(
        record: RunRecord,
        status: Literal["cancelled", "interrupted", "failed"],
        code: str,
    ) -> RunFinalization:
        return RunFinalization(
            status=status,
            stop_reason=None,
            is_partial=record.status != "queued",
            report_artifact_id=record.report_artifact_id,
            evidence_graph_artifact_id=record.evidence_graph_artifact_id,
            manifest_artifact_id=record.manifest_artifact_id,
            final_usage=record.final_usage or ResourceUsage.zero(),
            error_code=code,
        )

    async def _start(
        self,
        record: RunRecord,
        config: RunConfig,
        runner: ResearchRunner,
        token: CancellationToken,
    ) -> None:
        async with self._lock_for(record.run_id):
            current = await self.store.get_run(record.run_id)
            if current is None or current.status != "queued":
                return
            record = await self.store.transition(record.run_id, "queued", "running")
        await self._execute(record, config, runner, token)

    async def _execute(
        self,
        record: RunRecord,
        config: RunConfig,
        runner: ResearchRunner,
        token: CancellationToken,
    ) -> None:
        try:
            result = await runner.run(
                run_id=record.run_id,
                thread_id=record.thread_id,
                config=config,
                checkpoint=None,
                emit=_DurableSink(self, record.run_id),
                cancellation_token=token,
            )
            if result.run_id != record.run_id or result.thread_id != record.thread_id:
                raise ValueError("runner result identity mismatch")
            final = RunFinalization.from_result(result)
        except Exception:  # noqa: BLE001 - public errors never contain exception text
            final = self._inactive_finalization(record, "failed", "INTERNAL_ERROR")
        async with self._lock_for(record.run_id):
            if record.pricing_status == "estimated" and final.final_usage.cost_usd is None:
                final = replace(
                    final, status="failed", stop_reason=None, error_code="PRICING_INCOMPLETE"
                )
            elif record.pricing_status == "unknown":
                final = replace(
                    final, final_usage=final.final_usage.model_copy(update={"cost_usd": None})
                )
            if record.run_id in self._user_cancel_requested:
                final = replace(
                    final, status="cancelled", stop_reason=None, error_code="CANCELLED_BY_USER"
                )
            elif record.run_id in self._shutdown_requested:
                final = replace(
                    final,
                    status="interrupted",
                    stop_reason=None,
                    is_partial=True,
                    error_code="SERVICE_SHUTDOWN",
                )
            await self._finalize(record, final, settle=final.final_usage.cost_usd is not None)

    async def _finalize(
        self,
        record: RunRecord,
        final: RunFinalization,
        *,
        settle: bool,
    ) -> RunRecord:
        saved, terminal = await self.store.finalize_run(
            record.run_id, record.status, final, _draft(final.status)
        )
        reservation = saved.admission_reservation_id
        # Durable terminal comes first. If settlement fails, retain the link for
        # startup reconciliation; never announce a partially settled lifecycle.
        if settle and final.final_usage.cost_usd is not None:
            # settle owns both monetary settlement and the local capacity lease.
            # A failure retains both the durable reservation and recovery link.
            await self.admission.settle(reservation, final.final_usage.cost_usd)
        else:
            await self.admission.release(reservation)
        saved = await self.store.clear_admission(saved.run_id, reservation)
        await self.broadcast_persisted(terminal)
        return saved

    async def wait(self, run_id: str) -> RunView:
        task = self._tasks.get(run_id)
        if task is not None:
            try:
                await asyncio.shield(task)
            except asyncio.CancelledError:
                if not task.cancelled():
                    raise
        record = await self.store.get_run(run_id)
        if record is None:
            raise RunNotFound(run_id)
        return _view(record)

    async def subscribe(self, run_id: str, *, owner_scope_sha256: str) -> EventSubscription:
        await self._owned(run_id, owner_scope_sha256)
        lock = self._broadcast_locks.setdefault(run_id, asyncio.Lock())
        async with lock:
            subscribers = self._subscribers.setdefault(run_id, set())
            subscription = _Subscription(lock, subscribers)
            subscribers.add(subscription)
            return subscription

    async def emit(self, event: RunEvent) -> None:
        persisted = await self.store.append_event(event)
        await self.broadcast_persisted(persisted)

    async def broadcast_persisted(self, event: RunEvent) -> None:
        lock = self._broadcast_locks.setdefault(event.run_id, asyncio.Lock())
        async with lock:
            for subscription in self._subscribers.get(event.run_id, ()):
                subscription.event.set()

    async def shutdown(self, grace_seconds: float = 20.0) -> None:
        if not math.isfinite(grace_seconds) or grace_seconds < 0:
            raise ValueError("grace period must be finite and non-negative")
        async with self._shutdown_lock:
            self._closing = True
            async with self._creation_lock:
                tasks = tuple(self._tasks.values())
            if not tasks:
                return
            _, pending = await asyncio.wait(tasks, timeout=grace_seconds)
            for run_id, task in tuple(self._tasks.items()):
                if task not in pending:
                    continue
                async with self._lock_for(run_id):
                    if task.done():
                        continue
                    record = await self.store.get_run(run_id)
                    if record is None or record.status not in {"queued", "running"}:
                        continue
                    if run_id not in self._user_cancel_requested:
                        self._shutdown_requested.add(run_id)
                    self._tokens[run_id].cancel()
                    if record.status == "queued":
                        task.cancel()
                        final = self._inactive_finalization(
                            record, "interrupted", "SERVICE_SHUTDOWN"
                        )
                        await self._finalize(record, final, settle=False)
            # Cooperative Core completion includes saver writes. Do not cancel a
            # task while a checkpoint or durable finalization is in flight.
            await asyncio.gather(*tasks, return_exceptions=True)
