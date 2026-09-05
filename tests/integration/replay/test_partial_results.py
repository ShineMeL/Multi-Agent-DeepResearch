from __future__ import annotations

import json
from pathlib import Path

from deepresearch.planning.stop import StopCode
from deepresearch.workflow.research_graph import result_status_for

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "replay"


def test_partial_stop_with_minimum_report_remains_completed() -> None:
    status, is_partial = result_status_for(StopCode.BUDGET_EXHAUSTED, "sha256:report")
    assert status == "completed"
    assert is_partial is True


def test_blocked_without_report_is_failed_and_has_no_report() -> None:
    status, is_partial = result_status_for(StopCode.BLOCKED, None)
    assert status == "failed"
    assert is_partial is True


def test_budget_fixture_preserves_partial_report_semantics() -> None:
    scenario = json.loads(
        (FIXTURE_ROOT / "stop_budget_exhausted" / "scenario.json").read_text(
            encoding="utf-8"
        )
    )
    status, partial = result_status_for(
        StopCode(scenario["stop_code"]), scenario["report_artifact_id"]
    )
    assert (status, partial) == ("completed", True)
