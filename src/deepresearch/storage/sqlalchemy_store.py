from __future__ import annotations

from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from dataclasses import fields
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Any, cast
from uuid import uuid4

from pydantic import TypeAdapter
from sqlalchemy import func, select, update
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from deepresearch.domain import ResourceUsage, RunEvent, RunStatus
from deepresearch.runtime.admission import Admission
from deepresearch.storage.migrations.runner import lock_service_transaction
from deepresearch.storage.models import RunEventRow, RunRow, UsageLedgerRow
from deepresearch.storage.protocols import (
    DailyCostLimitExceeded,
    IdempotencyCollision,
    RunFinalization,
    RunRecord,
    StartupRecovery,
    TerminalEventDraft,
)

_RECORD = TypeAdapter(RunRecord)
_NONTERMINAL = frozenset({("queued", "running"), ("interrupted", "running")})
_TERMINAL = {
    "queued": frozenset({"interrupted", "cancelled"}),
    "running": frozenset({"interrupted", "completed", "failed", "cancelled"}),
    "interrupted": frozenset({"cancelled"}),
}


def _record(row: RunRow) -> RunRecord:
    values = {
        field.name: getattr(
            row,
            {"pricing_snapshots": "pricing_snapshots_json", "final_usage": "final_usage_json"}.get(
                field.name, field.name
            ),
        )
        for field in fields(RunRecord)
    }
    if row.updated_at.tzinfo is None:
        values["updated_at"] = row.updated_at.replace(tzinfo=UTC)
    return _RECORD.validate_python(values)


def _row(record: RunRecord) -> RunRow:
    values = _RECORD.dump_python(record, mode="json")
    values["updated_at"] = record.updated_at
    values["pricing_snapshots_json"] = values.pop("pricing_snapshots")
    values["final_usage_json"] = values.pop("final_usage")
    return RunRow(**values)


def _amount(value: Decimal) -> None:
    if not value.is_finite() or value < 0:
        raise ValueError("daily cost values must be finite and non-negative")


class SqlAlchemyRunStore:
    def __init__(self, database_url: str, artifact_root: Path):
        self.engine = create_async_engine(database_url)
        self.session_factory = async_sessionmaker(self.engine, expire_on_commit=False)
        self.artifact_root = artifact_root

    @asynccontextmanager
    async def _write(self) -> AsyncGenerator[AsyncSession]:
        async with self.session_factory() as session, session.begin():
            await lock_service_transaction(await session.connection())
            yield session

    async def get_run(self, run_id: str) -> RunRecord | None:
        async with self.session_factory() as session:
            row = await session.get(RunRow, run_id)
            return None if row is None else _record(row)

    async def get_owned_run(self, run_id: str, owner_scope_sha256: str) -> RunRecord | None:
        async with self.session_factory() as session:
            row = await session.scalar(
                select(RunRow).where(
                    RunRow.run_id == run_id, RunRow.owner_scope_sha256 == owner_scope_sha256
                )
            )
            return None if row is None else _record(row)

    async def get_by_idempotency(self, scope_sha256: str, idempotency_key: str) -> RunRecord | None:
        async with self.session_factory() as session:
            row = await session.scalar(
                select(RunRow).where(
                    RunRow.idempotency_scope_sha256 == scope_sha256,
                    RunRow.idempotency_key == idempotency_key,
                )
            )
            return None if row is None else _record(row)

    async def create_run(self, record: RunRecord) -> RunRecord:
        async with self._write() as session:
            if record.idempotency_key is not None:
                collision = await session.scalar(
                    select(RunRow).where(
                        RunRow.idempotency_scope_sha256 == record.idempotency_scope_sha256,
                        RunRow.idempotency_key == record.idempotency_key,
                    )
                )
                if collision is not None and collision.run_id != record.run_id:
                    raise IdempotencyCollision(record.idempotency_key)
            existing = await session.get(RunRow, record.run_id)
            if existing is not None:
                if existing.idempotency_key == record.idempotency_key:
                    return _record(existing)
                raise IdempotencyCollision(record.run_id)
            session.add(_row(record))
            return record

    async def _require(
        self, session: AsyncSession, run_id: str, expected: RunStatus | None = None
    ) -> RunRow:
        row = await session.scalar(select(RunRow).where(RunRow.run_id == run_id).with_for_update())
        if row is None:
            raise KeyError(run_id)
        if expected is not None and row.status != expected:
            raise ValueError(f"expected {expected}, found {row.status}")
        return row

    async def _cas(
        self, session: AsyncSession, row: RunRow, expected: RunStatus, values: dict[str, Any]
    ) -> RunRecord:
        result = await session.execute(
            update(RunRow)
            .where(
                RunRow.run_id == row.run_id,
                RunRow.status == expected,
                RunRow.version == row.version,
            )
            .values(**values, version=row.version + 1)
            .returning(RunRow)
            .execution_options(populate_existing=True)
        )
        saved = result.scalar_one_or_none()
        if saved is None:
            raise ValueError("run changed during transition")
        return _record(saved)

    async def transition(self, run_id: str, expected: RunStatus, target: RunStatus) -> RunRecord:
        if (expected, target) not in _NONTERMINAL:
            raise ValueError("transition is not a legal nonterminal CAS pair")
        async with self._write() as session:
            row = await self._require(session, run_id, expected)
            return await self._cas(session, row, expected, {"status": target})

    async def _finalize(
        self,
        session: AsyncSession,
        row: RunRow,
        expected: RunStatus,
        finalization: RunFinalization,
        draft: TerminalEventDraft,
    ) -> tuple[RunRecord, RunEvent]:
        values = {
            field.name: getattr(finalization, field.name) for field in fields(RunFinalization)
        }
        values.pop("final_usage")
        values["final_usage_json"] = finalization.final_usage.model_dump(mode="json")
        values["updated_at"] = draft.timestamp
        saved = await self._cas(session, row, expected, values)
        seq = await session.scalar(
            select(func.max(RunEventRow.seq)).where(RunEventRow.run_id == row.run_id)
        )
        event = RunEvent(
            seq=(seq or 0) + 1,
            run_id=row.run_id,
            timestamp=draft.timestamp,
            node=draft.node,
            kind=draft.kind,
            status=finalization.status,
            public_payload=draft.public_payload,
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
        session.add(
            RunEventRow(
                run_id=event.run_id, seq=event.seq, payload_json=event.model_dump(mode="json")
            )
        )
        await session.flush()
        return saved, event

    async def finalize_run(
        self,
        run_id: str,
        expected: RunStatus,
        finalization: RunFinalization,
        terminal_event: TerminalEventDraft,
    ) -> tuple[RunRecord, RunEvent]:
        if finalization.status not in _TERMINAL.get(expected, frozenset()):
            raise ValueError("finalization is not a legal terminal transition")
        async with self._write() as session:
            row = await self._require(session, run_id, expected)
            return await self._finalize(session, row, expected, finalization, terminal_event)

    async def bind_admission(
        self, run_id: str, expected: RunStatus, admission: Admission
    ) -> RunRecord:
        async with self._write() as session:
            row = await self._require(session, run_id, expected)
            return await self._cas(
                session,
                row,
                expected,
                {
                    "admission_reservation_id": admission.reservation_id,
                    "admission_attempt_no": admission.attempt_no,
                },
            )

    async def clear_admission(self, run_id: str, reservation_id: str | None) -> RunRecord:
        async with self._write() as session:
            row = await self._require(session, run_id)
            if reservation_id is None or row.admission_reservation_id != reservation_id:
                return _record(row)
            return await self._cas(
                session,
                row,
                cast(RunStatus, row.status),
                {"admission_reservation_id": None, "admission_attempt_no": None},
            )

    async def append_event(self, event: RunEvent) -> RunEvent:
        async with self._write() as session:
            seq = await session.scalar(
                select(func.max(RunEventRow.seq)).where(RunEventRow.run_id == event.run_id)
            )
            # Duplicate values are rejected by the database's composite PK.
            if event.seq > (seq or 0) + 1:
                raise ValueError("event sequence is not next")
            session.add(
                RunEventRow(
                    run_id=event.run_id, seq=event.seq, payload_json=event.model_dump(mode="json")
                )
            )
            await session.flush()
            return event

    async def list_events_after(self, run_id: str, seq: int) -> list[RunEvent]:
        async with self.session_factory() as session:
            rows = await session.scalars(
                select(RunEventRow)
                .where(RunEventRow.run_id == run_id, RunEventRow.seq > seq)
                .order_by(RunEventRow.seq)
            )
            return [RunEvent.model_validate(row.payload_json) for row in rows]

    async def reserve_daily_cost(
        self, day: date, run_id: str, amount: Decimal, limit: Decimal
    ) -> Admission:
        _amount(amount)
        _amount(limit)
        async with self._write() as session:
            attempts = list(
                await session.scalars(
                    select(UsageLedgerRow).where(UsageLedgerRow.run_id == run_id).with_for_update()
                )
            )
            for existing in attempts:
                if existing.state == "reserved":
                    return Admission(existing.reservation_id, existing.attempt_no)
            rows = list(
                await session.scalars(
                    select(UsageLedgerRow).where(UsageLedgerRow.day == day).with_for_update()
                )
            )
            total = sum(
                (Decimal(row.amount) for row in rows if row.state in {"reserved", "settled"}),
                Decimal(0),
            )
            if total + amount > limit:
                raise DailyCostLimitExceeded("daily cost limit exceeded")
            attempt = max((row.attempt_no for row in attempts), default=0) + 1
            reservation_id = str(uuid4())
            session.add(
                UsageLedgerRow(
                    reservation_id=reservation_id,
                    day=day,
                    run_id=run_id,
                    attempt_no=attempt,
                    amount=str(amount),
                    state="reserved",
                )
            )
            return Admission(reservation_id=reservation_id, attempt_no=attempt)

    async def settle_daily_cost(self, reservation_id: str, actual: Decimal) -> None:
        _amount(actual)
        async with self._write() as session:
            await self._ledger(session, reservation_id)
            await session.execute(
                update(UsageLedgerRow)
                .where(
                    UsageLedgerRow.reservation_id == reservation_id,
                    UsageLedgerRow.state == "reserved",
                )
                .values(amount=str(actual), state="settled")
            )

    async def release_daily_cost(self, reservation_id: str) -> None:
        async with self._write() as session:
            await self._ledger(session, reservation_id)
            await session.execute(
                update(UsageLedgerRow)
                .where(
                    UsageLedgerRow.reservation_id == reservation_id,
                    UsageLedgerRow.state == "reserved",
                )
                .values(state="released")
            )

    async def _ledger(self, session: AsyncSession, reservation_id: str) -> UsageLedgerRow:
        row = await session.get(UsageLedgerRow, reservation_id)
        if row is None:
            raise KeyError(reservation_id)
        return row

    async def ledger_state(self, reservation_id: str) -> str:
        async with self.session_factory() as session:
            return (await self._ledger(session, reservation_id)).state

    async def reconcile_startup(self, occurred_at: datetime) -> StartupRecovery:
        interrupted: list[str] = []
        released: list[str] = []
        async with self._write() as session:
            stale = list(
                await session.scalars(
                    select(RunRow)
                    .where(RunRow.status.in_(("queued", "running")))
                    .order_by(RunRow.run_id)
                    .with_for_update()
                )
            )
            for row in stale:
                record = _record(row)
                await self._finalize(
                    session,
                    row,
                    record.status,
                    RunFinalization(
                        "interrupted",
                        None,
                        record.status == "running",
                        record.report_artifact_id,
                        record.evidence_graph_artifact_id,
                        record.manifest_artifact_id,
                        record.final_usage or ResourceUsage.zero(),
                        "PROCESS_RESTART",
                    ),
                    TerminalEventDraft(occurred_at, "startup", "run_interrupted", {}),
                )
                interrupted.append(record.run_id)
            reservations = await session.scalars(
                select(UsageLedgerRow).where(UsageLedgerRow.state == "reserved").with_for_update()
            )
            for reservation in reservations:
                row = await session.get(RunRow, reservation.run_id)
                if row is None:
                    reservation.state = "released"
                    released.append(reservation.reservation_id)
                elif row.status in {"completed", "failed", "cancelled"}:
                    usage = _record(row).final_usage
                    if usage is not None and usage.cost_usd is not None:
                        reservation.amount = str(usage.cost_usd)
                        reservation.state = "settled"
                    else:
                        reservation.state = "released"
            return StartupRecovery(tuple(interrupted), tuple(sorted(released)))
