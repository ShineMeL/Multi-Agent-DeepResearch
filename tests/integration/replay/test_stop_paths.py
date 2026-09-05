from __future__ import annotations

import json
from pathlib import Path

import pytest

from deepresearch.planning.stop import StopCode
from deepresearch.workflow.research_graph import result_status_for

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "replay"


@pytest.mark.parametrize(
    ("stop_code", "report_artifact_id", "expected"),
    [
        (StopCode.SUFFICIENT, None, ("failed", False)),
        (StopCode.SUFFICIENT, "sha256:report", ("completed", False)),
        (StopCode.PLATEAU, None, ("failed", True)),
        (StopCode.PLATEAU, "sha256:report", ("completed", True)),
        (StopCode.BUDGET_EXHAUSTED, None, ("failed", True)),
        (StopCode.BUDGET_EXHAUSTED, "sha256:report", ("completed", True)),
        (StopCode.BLOCKED, None, ("failed", True)),
        (StopCode.BLOCKED, "sha256:report", ("completed", True)),
    ],
)
def test_result_status_for_maps_terminal_stop_to_canonical_result(
    stop_code: StopCode,
    report_artifact_id: str | None,
    expected: tuple[str, bool],
) -> None:
    assert result_status_for(stop_code, report_artifact_id) == expected


@pytest.mark.parametrize("fixture_name", ["stop_plateau", "stop_budget_exhausted", "stop_blocked"])
def test_stop_fixture_contract_is_public_result_only(fixture_name: str) -> None:
    scenario = json.loads(
        (FIXTURE_ROOT / fixture_name / "scenario.json").read_text(encoding="utf-8")
    )
    status, partial = result_status_for(
        StopCode(scenario["stop_code"]), scenario["report_artifact_id"]
    )
    assert status == scenario["expected_status"]
    assert partial is scenario["expected_partial"]
