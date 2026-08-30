import heapq
import json
from datetime import UTC, datetime
from hashlib import sha256
from typing import get_args

import pytest
from pydantic import ValidationError

from deepresearch import domain
from deepresearch.domain import ResourceUsage, RunEvent, RunResult, RunStatus, StopReason


def run_event(**updates: object) -> RunEvent:
    payload: dict[str, object] = {
        "seq": 1,
        "run_id": "run-1",
        "timestamp": datetime(2026, 8, 29, tzinfo=UTC),
        "node": "Plan",
        "kind": "node_started",
        "status": "running",
        "public_payload": {"attempt": 1},
        "usage_delta": ResourceUsage.zero(),
        "artifact_ids": (),
        "error_code": None,
    }
    payload.update(updates)
    return RunEvent.model_validate(payload)


def test_event_timestamp_requires_timezone() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        run_event(timestamp=datetime(2026, 8, 29))  # noqa: DTZ001 - deliberately naive


def test_run_result_separates_terminal_status_stop_reason_and_partiality() -> None:
    result = RunResult(
        run_id="run-1",
        thread_id="thread-1",
        status="interrupted",
        stop_reason="BUDGET_EXHAUSTED",
        is_partial=True,
        final_usage=ResourceUsage.zero(),
    )

    assert result.status == "interrupted"
    assert result.stop_reason == "BUDGET_EXHAUSTED"
    assert result.is_partial is True
    assert result.report_artifact_id is None


def test_run_event_forbids_extra_fields_and_is_frozen() -> None:
    event = run_event()

    with pytest.raises(ValidationError):
        run_event(private_payload={})
    with pytest.raises(ValidationError):
        event.status = "completed"


def test_run_literals_match_contract() -> None:
    assert set(get_args(RunStatus)) == {
        "queued",
        "running",
        "interrupted",
        "completed",
        "failed",
        "cancelled",
    }
    assert set(get_args(StopReason)) == {
        "SUFFICIENT",
        "PLATEAU",
        "BUDGET_EXHAUSTED",
        "BLOCKED",
    }


def test_domain_explicitly_exports_every_public_contract_symbol() -> None:
    expected = {
        "RunStatus",
        "StopReason",
        "ExecutionMode",
        "AccessProfile",
        "RunPurpose",
        "SourceType",
        "ClaimType",
        "VerificationStatus",
        "HtmlLocator",
        "PdfLocator",
        "Locator",
        "FreshnessRequirement",
        "ResearchRequest",
        "DateRange",
        "ResearchScope",
        "InformationNeed",
        "EvidenceRequirements",
        "SubQuestion",
        "ResearchPlan",
        "CoverageLedgerEntry",
        "SourceDocument",
        "EvidenceSpan",
        "Claim",
        "ClaimEvidenceLink",
        "RerankScore",
        "ResourceUsage",
        "RunBudget",
        "RunConfig",
        "RunEvent",
        "RunResult",
    }

    assert set(domain.__all__) == expected


def test_public_payload_serialization_is_canonical() -> None:
    first = run_event(public_payload={"z": {"b": 2, "a": 1}, "a": True})
    second = run_event(public_payload={"a": True, "z": {"a": 1, "b": 2}})

    assert first.model_dump_json() == second.model_dump_json()


def test_public_payload_is_detached_and_recursively_immutable() -> None:
    caller_owned = {"nested": {"items": [{"value": 1}]}}
    event = run_event(public_payload=caller_owned)
    digest = sha256(event.model_dump_json().encode()).digest()
    caller_owned["nested"]["items"][0]["value"] = 2

    assert event.public_payload["nested"]["items"][0]["value"] == 1
    with pytest.raises(TypeError, match="immutable"):
        event.public_payload["new"] = True
    with pytest.raises(AttributeError):
        event.public_payload["nested"]["items"].append("new")
    with pytest.raises(TypeError, match="immutable"):
        event.public_payload["nested"]["items"][0]["value"] = 3
    assert sha256(event.model_dump_json().encode()).digest() == digest
    assert event.model_dump(mode="json")["public_payload"] == {
        "nested": {"items": [{"value": 1}]}
    }


@pytest.mark.parametrize("deep", [False, True])
def test_event_update_copy_detaches_payload_and_revalidates_timestamp(deep: bool) -> None:
    event = run_event()
    caller_owned = {"nested": {"items": [{"value": 1}]}}

    copied = event.model_copy(update={"public_payload": caller_owned}, deep=deep)
    caller_owned["nested"]["items"][0]["value"] = 2

    assert copied.public_payload["nested"]["items"][0]["value"] == 1
    with pytest.raises(AttributeError):
        copied.public_payload["nested"]["items"].append("new")
    with pytest.raises(ValidationError):
        event.model_copy(update={"public_payload": {"invalid": object()}}, deep=deep)
    with pytest.raises(ValidationError, match="timezone"):
        event.model_copy(
            update={"timestamp": datetime(2026, 8, 29)},  # noqa: DTZ001 - invalid fixture
            deep=deep,
        )


@pytest.mark.parametrize("deep", [False, True])
def test_result_update_copy_revalidates_status(deep: bool) -> None:
    result = RunResult(
        run_id="run-1",
        thread_id="thread-1",
        status="completed",
        is_partial=False,
        final_usage=ResourceUsage.zero(),
    )

    with pytest.raises(ValidationError):
        result.model_copy(update={"status": "not-a-status"}, deep=deep)


def test_nested_event_arrays_reject_heapq_and_keep_json_wire_shape() -> None:
    event = run_event(public_payload={"values": [3, 1, 2]})
    values = event.public_payload["values"]
    digest = sha256(event.model_dump_json().encode()).digest()

    assert isinstance(values, tuple)
    with pytest.raises(TypeError):
        heapq.heappush(values, 0)
    assert sha256(event.model_dump_json().encode()).digest() == digest
    assert json.loads(event.model_dump_json())["public_payload"]["values"] == [3, 1, 2]
    assert event.model_dump()["public_payload"]["values"] == [3, 1, 2]


@pytest.mark.parametrize("deep", [False, True])
def test_event_update_copy_accepts_its_own_internal_json_mapping(deep: bool) -> None:
    event = run_event(public_payload={"values": [3, 1, 2]})

    copied = event.model_copy(update={"public_payload": event.public_payload}, deep=deep)

    assert copied == event
    assert isinstance(copied.public_payload["values"], tuple)
