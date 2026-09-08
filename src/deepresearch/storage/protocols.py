from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from decimal import Decimal
from typing import Literal, Protocol, cast

from pydantic import JsonValue

from deepresearch.domain import ResourceUsage, RunEvent, RunResult, RunStatus, StopReason
from deepresearch.runtime.admission import Admission
from deepresearch.runtime.manifest import PricingSnapshot


@dataclass(frozen=True)
class RunFinalization:
    status: Literal["interrupted", "completed", "failed", "cancelled"]
    stop_reason: StopReason | None
    is_partial: bool
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    final_usage: ResourceUsage
    error_code: str | None

    @classmethod
    def from_result(cls, result: RunResult) -> RunFinalization:
        if result.status not in {"interrupted", "completed", "failed", "cancelled"}:
            raise ValueError("run result is not terminal")
        status = cast(
            Literal["interrupted", "completed", "failed", "cancelled"], result.status
        )
        return cls(
            status=status,
            stop_reason=result.stop_reason,
            is_partial=result.is_partial,
            report_artifact_id=result.report_artifact_id,
            evidence_graph_artifact_id=result.evidence_graph_artifact_id,
            manifest_artifact_id=result.manifest_artifact_id,
            final_usage=result.final_usage,
            error_code=result.error_code,
        )


@dataclass(frozen=True)
class TerminalEventDraft:
    timestamp: datetime
    node: str
    kind: str
    public_payload: dict[str, JsonValue]


class IdempotencyCollision(RuntimeError):
    """Raised when an idempotency key names a different run."""


class DailyCostLimitExceeded(ValueError):
    """The durable total cannot accommodate another reservation."""


@dataclass(frozen=True)
class StartupRecovery:
    interrupted_run_ids: tuple[str, ...]
    released_orphan_reservation_ids: tuple[str, ...]


@dataclass(frozen=True)
class RunRecord:
    run_id: str
    thread_id: str
    status: RunStatus
    config_json: dict[str, object]
    pricing_status: Literal["estimated", "unknown"]
    pricing_snapshots: tuple[PricingSnapshot, ...]
    provider_profile_json: dict[str, object]
    provider_profile_sha256: str
    config_sha256: str
    owner_scope_sha256: str
    idempotency_scope_sha256: str
    idempotency_key: str | None
    admission_reservation_id: str | None
    admission_attempt_no: int | None
    stop_reason: StopReason | None
    is_partial: bool
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    final_usage: ResourceUsage | None
    error_code: str | None
    updated_at: datetime
    version: int


@dataclass(frozen=True)
class RunView:
    run_id: str
    thread_id: str
    status: RunStatus
    stop_reason: StopReason | None
    is_partial: bool
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    final_usage: ResourceUsage | None
    error_code: str | None


class RunStore(Protocol):
    async def get_run(self, run_id: str) -> RunRecord | None: ...

    async def get_owned_run(self, run_id: str, owner_scope_sha256: str) -> RunRecord | None: ...

    async def get_by_idempotency(
        self, scope_sha256: str, idempotency_key: str
    ) -> RunRecord | None: ...

    async def create_run(self, record: RunRecord) -> RunRecord: ...

    async def transition(
        self, run_id: str, expected: RunStatus, target: RunStatus
    ) -> RunRecord: ...

    async def finalize_run(
        self,
        run_id: str,
        expected: RunStatus,
        finalization: RunFinalization,
        terminal_event: TerminalEventDraft,
    ) -> tuple[RunRecord, RunEvent]: ...

    async def bind_admission(
        self, run_id: str, expected: RunStatus, admission: Admission
    ) -> RunRecord: ...

    async def clear_admission(
        self, run_id: str, reservation_id: str | None
    ) -> RunRecord: ...

    async def append_event(self, event: RunEvent) -> RunEvent: ...

    async def list_events_after(self, run_id: str, seq: int) -> list[RunEvent]: ...

    async def reconcile_startup(self, occurred_at: datetime) -> StartupRecovery: ...

    async def reserve_daily_cost(
        self, day: date, run_id: str, amount: Decimal, limit: Decimal
    ) -> Admission: ...

    async def settle_daily_cost(self, reservation_id: str, actual: Decimal) -> None: ...

    async def release_daily_cost(self, reservation_id: str) -> None: ...


__all__ = [
    "DailyCostLimitExceeded",
    "IdempotencyCollision",
    "RunFinalization",
    "RunRecord",
    "RunStore",
    "RunView",
    "StartupRecovery",
    "TerminalEventDraft",
]
