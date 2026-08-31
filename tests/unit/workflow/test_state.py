from __future__ import annotations

import inspect
from datetime import UTC, datetime
from typing import get_args, get_origin, get_type_hints

import pytest

from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunConfig, RunEvent
from deepresearch.providers import ParsedDocument, RawDocument
from deepresearch.runtime import (
    BudgetAccountant,
    CancellationToken,
    CheckpointRef,
    ResearchRunner,
)
from deepresearch.workflow.state import (
    BaselineState,
    StateValidationError,
    validate_baseline_state,
)

_STATE_FIELDS = {
    "active_subquestion_id",
    "baseline_work_artifact_ids",
    "blocked_needs",
    "budget_snapshot",
    "config_sha256",
    "coverage_ledger",
    "draft_artifact_id",
    "elapsed_wall_seconds",
    "error_code",
    "evidence_graph_artifact_id",
    "evidence_ids",
    "failed_node",
    "high_priority_unresolved_conflict_ids",
    "is_partial",
    "manifest_artifact_id",
    "next_event_seq",
    "pending_subquestion_ids",
    "plan_artifact_id",
    "plan_id",
    "query_ids",
    "recent_marginal_gains",
    "report_artifact_id",
    "request",
    "run_id",
    "selected_evidence_ids",
    "source_ids",
    "stop_reason",
    "thread_id",
}
_TUPLE_FIELDS = {
    "baseline_work_artifact_ids",
    "blocked_needs",
    "coverage_ledger",
    "evidence_ids",
    "high_priority_unresolved_conflict_ids",
    "pending_subquestion_ids",
    "query_ids",
    "recent_marginal_gains",
    "selected_evidence_ids",
    "source_ids",
}


def request() -> ResearchRequest:
    return ResearchRequest(
        question="Which methods are documented?",
        output_requirements={"answer_shape": "brief"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="low",
    )


def config() -> RunConfig:
    from deepresearch.domain import RunBudget

    return RunConfig(
        request=request(),
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
        budget=RunBudget.preset("low"),
        prompt_versions={"planner": "p1", "writer": "w1"},
        seed=0,
    )


def state() -> BaselineState:
    current = config()
    return BaselineState(
        run_id="run-1",
        thread_id="thread-1",
        request=current.request,
        config_sha256="a" * 64,
        plan_id=None,
        plan_artifact_id=None,
        pending_subquestion_ids=(),
        active_subquestion_id=None,
        query_ids=(),
        source_ids=(),
        evidence_ids=(),
        selected_evidence_ids=(),
        coverage_ledger=(),
        high_priority_unresolved_conflict_ids=(),
        blocked_needs=(),
        recent_marginal_gains=(),
        baseline_work_artifact_ids=(),
        budget_snapshot=BudgetAccountant(current.budget).snapshot(),
        stop_reason=None,
        is_partial=False,
        draft_artifact_id=None,
        report_artifact_id=None,
        evidence_graph_artifact_id=None,
        manifest_artifact_id=None,
        next_event_seq=1,
        failed_node=None,
        elapsed_wall_seconds=0.0,
        error_code=None,
    )


def test_runtime_ports_have_exact_public_contract_and_additive_exports() -> None:
    assert inspect.signature(ResearchRunner.run) == inspect.Signature(
        parameters=(
            inspect.Parameter("self", inspect.Parameter.POSITIONAL_OR_KEYWORD),
            inspect.Parameter("run_id", inspect.Parameter.KEYWORD_ONLY, annotation="str"),
            inspect.Parameter("thread_id", inspect.Parameter.KEYWORD_ONLY, annotation="str"),
            inspect.Parameter("config", inspect.Parameter.KEYWORD_ONLY, annotation="RunConfig"),
            inspect.Parameter(
                "checkpoint",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="CheckpointRef | None",
            ),
            inspect.Parameter(
                "emit",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="Callable[[RunEvent], Awaitable[None]]",
            ),
            inspect.Parameter(
                "cancellation_token",
                inspect.Parameter.KEYWORD_ONLY,
                annotation="CancellationToken",
            ),
        ),
        return_annotation="RunResult",
    )
    assert CancellationToken is not None
    assert RunEvent is not None


def test_checkpoint_ref_is_frozen_and_requires_aware_timestamp() -> None:
    ref = CheckpointRef(
        checkpoint_id="cp-1",
        thread_id="thread-1",
        created_at=datetime(2026, 9, 1, tzinfo=UTC),
    )

    with pytest.raises(AttributeError):
        ref.thread_id = "changed"  # type: ignore[misc]
    with pytest.raises(ValueError, match="timezone-aware"):
        CheckpointRef(
            checkpoint_id="cp-1",
            thread_id="thread-1",
            created_at=datetime(2026, 9, 1, tzinfo=UTC).replace(tzinfo=None),
        )


def test_baseline_state_has_exact_checkpoint_safe_shape() -> None:
    annotations = get_type_hints(BaselineState)

    assert set(annotations) == _STATE_FIELDS
    assert {RawDocument, ParsedDocument, bytes}.isdisjoint(set(annotations.values()))
    for name in _TUPLE_FIELDS:
        assert get_origin(annotations[name]) is tuple
        assert get_args(annotations[name])


def test_state_validation_returns_fresh_tuple_preserving_copies() -> None:
    original = state()
    original["blocked_needs"] = (
        {
            "need_id": "need-1",
            "required_source_unavailable": True,
            "alternative_strategies_exhausted": True,
            "retry_count": 2,
            "max_retries": 2,
        },
    )

    restored = validate_baseline_state(original)

    assert restored == original
    assert restored is not original
    assert restored["blocked_needs"] is not original["blocked_needs"]
    assert restored["blocked_needs"][0] is not original["blocked_needs"][0]
    assert all(isinstance(restored[name], tuple) for name in _TUPLE_FIELDS)


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("next_event_seq", 0),
        ("elapsed_wall_seconds", float("nan")),
        ("recent_marginal_gains", (float("inf"),)),
        ("pending_subquestion_ids", ["sq-1"]),
    ),
)
def test_state_validation_rejects_invalid_restored_values(
    field: str,
    value: object,
) -> None:
    candidate = dict(state())
    candidate[field] = value

    with pytest.raises(StateValidationError) as error:
        validate_baseline_state(candidate)

    assert error.value.code == "DATA_CORRUPTION"


def test_state_validation_rejects_extra_checkpoint_fields() -> None:
    candidate = dict(state())
    candidate["provider"] = object()

    with pytest.raises(StateValidationError):
        validate_baseline_state(candidate)
