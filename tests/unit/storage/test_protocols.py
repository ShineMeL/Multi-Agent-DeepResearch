import asyncio
import inspect
from dataclasses import replace
from datetime import UTC, datetime
from types import TracebackType
from typing import Literal, cast

import pytest

from deepresearch.domain import ResourceUsage, RunResult
from deepresearch.runtime.admission import Admission
from deepresearch.storage.protocols import (
    IdempotencyCollision,
    RunFinalization,
    RunRecord,
    RunStore,
    TerminalEventDraft,
)
from tests.fakes.service_store import FakeRunStore, make_record


def _finalization(
    status: Literal["interrupted", "completed", "failed", "cancelled"] = "completed",
) -> RunFinalization:
    return RunFinalization.from_result(
        RunResult(
            run_id="r1",
            thread_id="thread-1",
            status=status,
            is_partial=False,
            final_usage=ResourceUsage.zero(cost_known=True),
        )
    )


def _draft() -> TerminalEventDraft:
    return TerminalEventDraft(
        timestamp=datetime.now(UTC), node="manager", kind="run_completed", public_payload={}
    )


def test_store_protocol_exposes_atomic_finalization() -> None:
    annotations = inspect.get_annotations(RunStore.finalize_run, eval_str=True)
    assert "RunFinalization" in str(annotations["finalization"])


@pytest.mark.asyncio
async def test_fake_store_finalizes_all_result_fields() -> None:
    store = FakeRunStore()
    await store.create_run(make_record("r1", status="running"))
    usage = ResourceUsage.zero(cost_known=True)
    result = RunResult(
        run_id="r1",
        thread_id="thread-1",
        status="completed",
        is_partial=False,
        report_artifact_id="report-1",
        evidence_graph_artifact_id="evidence-1",
        manifest_artifact_id="manifest-1",
        final_usage=usage,
        error_code=None,
    )
    record, terminal = await store.finalize_run(
        "r1",
        "running",
        RunFinalization.from_result(result),
        TerminalEventDraft(
            timestamp=datetime.now(UTC),
            node="manager",
            kind="run_completed",
            public_payload={},
        ),
    )

    assert (
        record.status,
        record.report_artifact_id,
        record.evidence_graph_artifact_id,
        record.manifest_artifact_id,
        record.final_usage,
        record.error_code,
    ) == ("completed", "report-1", "evidence-1", "manifest-1", usage, None)
    assert (terminal.seq, terminal.status) == (1, "completed")


@pytest.mark.asyncio
async def test_fake_store_rejects_invalid_nonterminal_and_terminal_transitions() -> None:
    store = FakeRunStore()
    await store.create_run(make_record("r1", status="running"))

    with pytest.raises(ValueError):
        await store.transition("r1", "running", "running")
    with pytest.raises(ValueError):
        await store.transition("r1", "running", "completed")
    await store.create_run(make_record("r-terminal", status="completed"))
    with pytest.raises(ValueError):
        await store.transition("r-terminal", "completed", "running")
    await store.create_run(make_record("r2", status="queued"))
    with pytest.raises(ValueError):
        await store.finalize_run("r2", "queued", _finalization(), _draft())


@pytest.mark.asyncio
async def test_fake_store_rejects_terminal_refinalization() -> None:
    store = FakeRunStore()
    await store.create_run(make_record("r1", status="running"))
    await store.finalize_run("r1", "running", _finalization(), _draft())

    with pytest.raises(ValueError):
        await store.finalize_run("r1", "completed", _finalization(), _draft())
    assert len(await store.list_events_after("r1", 0)) == 1


@pytest.mark.asyncio
async def test_clear_admission_ignores_none_and_nonmatching_ids_then_is_idempotent() -> None:
    store = FakeRunStore()
    created = await store.create_run(make_record("r1", status="queued"))
    assert await store.clear_admission("r1", None) == created
    bound = await store.bind_admission("r1", "queued", Admission("reservation-1", 1))

    assert await store.clear_admission("r1", None) == bound
    assert await store.clear_admission("r1", "other") == bound
    cleared = await store.clear_admission("r1", "reservation-1")
    assert cleared.version == bound.version + 1
    assert cleared.admission_reservation_id is None
    assert await store.clear_admission("r1", "reservation-1") == cleared
    assert created.version + 2 == cleared.version


class _NoOpAsyncLock:
    async def __aenter__(self) -> None:
        return None

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return False


class _UnlockedRacingIdempotencyStore(FakeRunStore):
    def __init__(self) -> None:
        super().__init__()
        self._creates_waiting = 0
        self._gate = asyncio.Event()

    def _idempotency_lock_for(self, key: tuple[str, str]) -> asyncio.Lock:
        return cast(asyncio.Lock, _NoOpAsyncLock())

    async def _create_run(self, record: RunRecord) -> RunRecord:
        self._creates_waiting += 1
        if self._creates_waiting == 2:
            self._gate.set()
        await self._gate.wait()
        return await super()._create_run(record)


@pytest.mark.asyncio
async def test_create_run_rejects_concurrent_duplicate_scoped_idempotency_key() -> None:
    first = make_record("r1", status="queued")
    second = make_record("r2", status="queued")
    first = replace(first, idempotency_key="request-1")
    second = replace(second, idempotency_key="request-1")

    store = FakeRunStore()
    outcomes = await asyncio.gather(
        store.create_run(first), store.create_run(second), return_exceptions=True
    )

    assert sum(isinstance(outcome, IdempotencyCollision) for outcome in outcomes) == 1
    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1

    unlocked_store = _UnlockedRacingIdempotencyStore()
    unlocked_outcomes = await asyncio.gather(
        unlocked_store.create_run(first),
        unlocked_store.create_run(second),
        return_exceptions=True,
    )
    assert sum(not isinstance(outcome, BaseException) for outcome in unlocked_outcomes) == 2
