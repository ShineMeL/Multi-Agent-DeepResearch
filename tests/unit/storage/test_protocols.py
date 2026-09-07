import asyncio
import inspect
from dataclasses import replace
from datetime import UTC, datetime
from typing import Literal, override

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


class _RacingIdempotencyStore(FakeRunStore):
    def __init__(self) -> None:
        super().__init__()
        self._lookups = 0
        self._gate = asyncio.Event()

    @override
    async def get_by_idempotency(
        self, scope_sha256: str, idempotency_key: str
    ) -> RunRecord | None:
        found = await super().get_by_idempotency(scope_sha256, idempotency_key)
        self._lookups += 1
        if self._lookups == 2:
            self._gate.set()
        await self._gate.wait()
        return found


@pytest.mark.asyncio
async def test_create_run_rejects_concurrent_duplicate_scoped_idempotency_key() -> None:
    store = _RacingIdempotencyStore()
    first = make_record("r1", status="queued")
    second = make_record("r2", status="queued")
    first = replace(first, idempotency_key="request-1")
    second = replace(second, idempotency_key="request-1")

    outcomes = await asyncio.gather(
        store.create_run(first), store.create_run(second), return_exceptions=True
    )

    assert sum(isinstance(outcome, IdempotencyCollision) for outcome in outcomes) == 1
    assert sum(not isinstance(outcome, BaseException) for outcome in outcomes) == 1
