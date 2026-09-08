from __future__ import annotations

import asyncio
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from decimal import Decimal

from deepresearch.domain import ResourceUsage, RunEvent, RunStatus
from deepresearch.runtime.admission import Admission
from deepresearch.storage.protocols import (
    IdempotencyCollision,
    RunFinalization,
    RunRecord,
    StartupRecovery,
    TerminalEventDraft,
)

_LEGAL_NONTERMINAL_TRANSITIONS = frozenset({("queued", "running"), ("interrupted", "running")})
_LEGAL_TERMINAL_TRANSITIONS = {
    "queued": frozenset({"interrupted", "cancelled"}),
    "running": frozenset({"interrupted", "completed", "failed", "cancelled"}),
    "interrupted": frozenset({"cancelled"}),
}


@dataclass
class _DailyReservation:
    day: date
    run_id: str
    amount: Decimal
    state: str = "reserved"


class FakeRunStore:
    """In-memory RunStore implementation shared by service contract tests."""

    def __init__(self) -> None:
        self._runs: dict[str, RunRecord] = {}
        self._events: dict[str, list[RunEvent]] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        self._idempotency_locks: dict[tuple[str, str], asyncio.Lock] = {}
        self._idempotency_index: dict[tuple[str, str], str] = {}
        self._daily_reservations: dict[str, _DailyReservation] = {}
        self._daily_counter = 0
        self._daily_lock = asyncio.Lock()

    def _lock_for(self, run_id: str) -> asyncio.Lock:
        return self._locks.setdefault(run_id, asyncio.Lock())

    def _idempotency_lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        return self._idempotency_locks.setdefault(key, asyncio.Lock())

    async def get_run(self, run_id: str) -> RunRecord | None:
        return self._runs.get(run_id)

    async def get_owned_run(self, run_id: str, owner_scope_sha256: str) -> RunRecord | None:
        record = self._runs.get(run_id)
        if record is None or record.owner_scope_sha256 != owner_scope_sha256:
            return None
        return record

    async def get_by_idempotency(self, scope_sha256: str, idempotency_key: str) -> RunRecord | None:
        run_id = self._idempotency_index.get((scope_sha256, idempotency_key))
        return self._runs.get(run_id) if run_id is not None else None

    async def create_run(self, record: RunRecord) -> RunRecord:
        if record.idempotency_key is None:
            return await self._create_run(record)
        key = (record.idempotency_scope_sha256, record.idempotency_key)
        async with self._idempotency_lock_for(key):
            indexed_run_id = self._idempotency_index.get(key)
            if indexed_run_id is not None and indexed_run_id != record.run_id:
                raise IdempotencyCollision(record.idempotency_key)
            created = await self._create_run(record)
            self._idempotency_index[key] = created.run_id
            return created

    async def transition(self, run_id: str, expected: RunStatus, target: RunStatus) -> RunRecord:
        if (expected, target) not in _LEGAL_NONTERMINAL_TRANSITIONS:
            raise ValueError("transition is not a legal nonterminal CAS pair")
        async with self._lock_for(run_id):
            record = self._require_expected(run_id, expected)
            updated = replace(record, status=target, version=record.version + 1)
            self._runs[run_id] = updated
            return updated

    async def finalize_run(
        self,
        run_id: str,
        expected: RunStatus,
        finalization: RunFinalization,
        terminal_event: TerminalEventDraft,
    ) -> tuple[RunRecord, RunEvent]:
        if finalization.status not in _LEGAL_TERMINAL_TRANSITIONS.get(expected, frozenset()):
            raise ValueError("finalization is not a legal terminal transition")
        async with self._lock_for(run_id):
            record = self._require_expected(run_id, expected)
            updated = replace(
                record,
                status=finalization.status,
                stop_reason=finalization.stop_reason,
                is_partial=finalization.is_partial,
                report_artifact_id=finalization.report_artifact_id,
                evidence_graph_artifact_id=finalization.evidence_graph_artifact_id,
                manifest_artifact_id=finalization.manifest_artifact_id,
                final_usage=finalization.final_usage,
                error_code=finalization.error_code,
                updated_at=terminal_event.timestamp,
                version=record.version + 1,
            )
            event = RunEvent(
                seq=len(self._events[run_id]) + 1,
                run_id=run_id,
                timestamp=terminal_event.timestamp,
                node=terminal_event.node,
                kind=terminal_event.kind,
                status=finalization.status,
                public_payload=terminal_event.public_payload,
                usage_delta=ResourceUsage.zero(
                    cost_known=finalization.final_usage.cost_usd is not None
                ),
                artifact_ids=tuple(
                    item
                    for item in (
                        finalization.report_artifact_id,
                        finalization.evidence_graph_artifact_id,
                        finalization.manifest_artifact_id,
                    )
                    if item is not None
                ),
                error_code=finalization.error_code,
            )
            self._runs[run_id] = updated
            self._events[run_id].append(event)
            return updated, event

    async def bind_admission(
        self, run_id: str, expected: RunStatus, admission: Admission
    ) -> RunRecord:
        async with self._lock_for(run_id):
            record = self._require_expected(run_id, expected)
            updated = replace(
                record,
                admission_reservation_id=admission.reservation_id,
                admission_attempt_no=admission.attempt_no,
                version=record.version + 1,
            )
            self._runs[run_id] = updated
            return updated

    async def clear_admission(self, run_id: str, reservation_id: str | None) -> RunRecord:
        async with self._lock_for(run_id):
            record = self._runs[run_id]
            if reservation_id is None or record.admission_reservation_id != reservation_id:
                return record
            updated = replace(
                record,
                admission_reservation_id=None,
                admission_attempt_no=None,
                version=record.version + 1,
            )
            self._runs[run_id] = updated
            return updated

    async def append_event(self, event: RunEvent) -> RunEvent:
        async with self._lock_for(event.run_id):
            events = self._events.setdefault(event.run_id, [])
            if event.seq != len(events) + 1:
                raise ValueError("event sequence is not next")
            events.append(event)
            return event

    async def list_events_after(self, run_id: str, seq: int) -> list[RunEvent]:
        return [event for event in self._events.get(run_id, []) if event.seq > seq]

    async def reconcile_startup(self, occurred_at: datetime) -> StartupRecovery:
        interrupted: list[str] = []
        released: list[str] = []
        for run_id, record in sorted(self._runs.items()):
            if record.status not in {"queued", "running"}:
                continue
            await self.finalize_run(
                run_id,
                record.status,
                RunFinalization(
                    status="interrupted",
                    stop_reason=None,
                    is_partial=record.status == "running",
                    report_artifact_id=record.report_artifact_id,
                    evidence_graph_artifact_id=record.evidence_graph_artifact_id,
                    manifest_artifact_id=record.manifest_artifact_id,
                    final_usage=record.final_usage or ResourceUsage.zero(),
                    error_code="PROCESS_RESTART",
                ),
                TerminalEventDraft(
                    timestamp=occurred_at,
                    node="startup",
                    kind="run_interrupted",
                    public_payload={},
                ),
            )
            interrupted.append(run_id)
        for reservation_id, reservation in self._daily_reservations.items():
            if reservation.state != "reserved":
                continue
            record = self._runs.get(reservation.run_id)
            if record is None:
                reservation.state = "released"
                released.append(reservation_id)
            elif record.status in {"completed", "failed", "cancelled"}:
                if record.final_usage is not None and record.final_usage.cost_usd is not None:
                    reservation.amount = record.final_usage.cost_usd
                    reservation.state = "settled"
                else:
                    reservation.state = "released"
        return StartupRecovery(tuple(interrupted), tuple(sorted(released)))

    async def reserve_daily_cost(
        self, day: date, run_id: str, amount: Decimal, limit: Decimal
    ) -> Admission:
        async with self._daily_lock:
            if amount < 0 or limit < 0:
                raise ValueError("daily cost values must be non-negative")
            total = sum(
                reservation.amount
                for reservation in self._daily_reservations.values()
                if reservation.day == day and reservation.state in {"reserved", "settled"}
            )
            if total + amount > limit:
                raise ValueError("daily cost limit exceeded")
            self._daily_counter += 1
            reservation_id = f"daily-{day.isoformat()}-{self._daily_counter}"
            self._daily_reservations[reservation_id] = _DailyReservation(day, run_id, amount)
            return Admission(reservation_id=reservation_id, attempt_no=self._daily_counter)

    async def settle_daily_cost(self, reservation_id: str, actual: Decimal) -> None:
        async with self._daily_lock:
            reservation = self._daily_reservations[reservation_id]
            if reservation.state == "reserved":
                reservation.amount = actual
                reservation.state = "settled"

    async def release_daily_cost(self, reservation_id: str) -> None:
        async with self._daily_lock:
            reservation = self._daily_reservations[reservation_id]
            if reservation.state == "reserved":
                reservation.state = "released"

    def _require_expected(self, run_id: str, expected: RunStatus) -> RunRecord:
        record = self._runs[run_id]
        if record.status != expected:
            raise ValueError(f"expected {expected}, found {record.status}")
        return record

    async def _create_run(self, record: RunRecord) -> RunRecord:
        async with self._lock_for(record.run_id):
            existing = self._runs.get(record.run_id)
            if existing is not None:
                if existing.idempotency_key == record.idempotency_key:
                    return existing
                raise IdempotencyCollision(record.run_id)
            self._runs[record.run_id] = record
            self._events[record.run_id] = []
            return record


def make_record(run_id: str, *, status: RunStatus) -> RunRecord:
    return RunRecord(
        run_id=run_id,
        thread_id="thread-1",
        status=status,
        config_json={},
        pricing_status="unknown",
        pricing_snapshots=(),
        provider_profile_json={},
        provider_profile_sha256="a" * 64,
        config_sha256="b" * 64,
        owner_scope_sha256="c" * 64,
        idempotency_scope_sha256="d" * 64,
        idempotency_key=None,
        admission_reservation_id=None,
        admission_attempt_no=None,
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
