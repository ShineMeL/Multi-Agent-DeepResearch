import inspect
from datetime import UTC, datetime

import pytest

from deepresearch.domain import ResourceUsage, RunResult
from deepresearch.storage.protocols import RunFinalization, RunStore, TerminalEventDraft
from tests.fakes.service_store import FakeRunStore, make_record


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
