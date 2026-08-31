from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Literal, TypedDict, cast

from pydantic import BaseModel

from deepresearch.domain import CoverageLedgerEntry, ResearchRequest, StopReason
from deepresearch.runtime import BudgetSnapshot


class BaselineBlockedNeed(TypedDict):
    need_id: str
    required_source_unavailable: bool
    alternative_strategies_exhausted: bool
    retry_count: int
    max_retries: int


class BaselineState(TypedDict):
    run_id: str
    thread_id: str
    request: ResearchRequest
    config_sha256: str
    plan_id: str | None
    plan_artifact_id: str | None
    pending_subquestion_ids: tuple[str, ...]
    active_subquestion_id: str | None
    query_ids: tuple[str, ...]
    source_ids: tuple[str, ...]
    evidence_ids: tuple[str, ...]
    selected_evidence_ids: tuple[str, ...]
    coverage_ledger: tuple[CoverageLedgerEntry, ...]
    high_priority_unresolved_conflict_ids: tuple[str, ...]
    blocked_needs: tuple[BaselineBlockedNeed, ...]
    recent_marginal_gains: tuple[float, ...]
    baseline_work_artifact_ids: tuple[str, ...]
    budget_snapshot: BudgetSnapshot
    stop_reason: StopReason | None
    is_partial: bool
    draft_artifact_id: str | None
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    next_event_seq: int
    failed_node: str | None
    elapsed_wall_seconds: float
    error_code: str | None


class StateValidationError(ValueError):
    code: Literal["DATA_CORRUPTION"] = "DATA_CORRUPTION"

    def __init__(self) -> None:
        super().__init__("checkpoint state is invalid")


_FIELDS = frozenset(BaselineState.__annotations__)
_STRING_TUPLES = (
    "pending_subquestion_ids",
    "query_ids",
    "source_ids",
    "evidence_ids",
    "selected_evidence_ids",
    "high_priority_unresolved_conflict_ids",
    "baseline_work_artifact_ids",
)
_OPTIONAL_STRINGS = (
    "plan_id",
    "plan_artifact_id",
    "active_subquestion_id",
    "draft_artifact_id",
    "report_artifact_id",
    "evidence_graph_artifact_id",
    "manifest_artifact_id",
    "failed_node",
    "error_code",
)
_STOP_REASONS = frozenset({"SUFFICIENT", "PLATEAU", "BUDGET_EXHAUSTED", "BLOCKED"})
_BLOCKED_FIELDS = frozenset(BaselineBlockedNeed.__annotations__)


def _invalid() -> StateValidationError:
    return StateValidationError()


def _revalidate_model[Model: BaseModel](value: object, model_type: type[Model]) -> Model:
    if type(value) is not model_type:
        raise _invalid()
    model = cast("BaseModel", value)
    try:
        return model_type.model_validate(model.model_dump(round_trip=True))
    except (AttributeError, TypeError, ValueError):
        raise _invalid() from None


def _validate_blocked_need(value: object) -> BaselineBlockedNeed:
    if not isinstance(value, Mapping):
        raise _invalid()
    raw = cast("Mapping[object, object]", value)
    if set(raw) != set(_BLOCKED_FIELDS):
        raise _invalid()
    need_id = raw["need_id"]
    unavailable = raw["required_source_unavailable"]
    exhausted = raw["alternative_strategies_exhausted"]
    retry_count = raw["retry_count"]
    max_retries = raw["max_retries"]
    if not isinstance(need_id, str) or not need_id:
        raise _invalid()
    if type(unavailable) is not bool or type(exhausted) is not bool:
        raise _invalid()
    if (
        type(retry_count) is not int
        or type(max_retries) is not int
        or retry_count < 0
        or max_retries < 0
    ):
        raise _invalid()
    return BaselineBlockedNeed(
        need_id=need_id,
        required_source_unavailable=unavailable,
        alternative_strategies_exhausted=exhausted,
        retry_count=retry_count,
        max_retries=max_retries,
    )


def validate_baseline_state(value: Mapping[str, object]) -> BaselineState:
    if set(value) != set(_FIELDS):
        raise _invalid()
    run_id = value["run_id"]
    thread_id = value["thread_id"]
    request = value["request"]
    config_sha256 = value["config_sha256"]
    if not isinstance(run_id, str) or not run_id:
        raise _invalid()
    if not isinstance(thread_id, str) or not thread_id:
        raise _invalid()
    restored_request = _revalidate_model(request, ResearchRequest)
    if (
        not isinstance(config_sha256, str)
        or len(config_sha256) != 64
        or config_sha256 != config_sha256.lower()
        or any(character not in "0123456789abcdef" for character in config_sha256)
    ):
        raise _invalid()

    copied = dict(value)
    copied["request"] = restored_request
    for name in _STRING_TUPLES:
        item = value[name]
        if not isinstance(item, tuple):
            raise _invalid()
        members = cast("tuple[object, ...]", item)
        if any(not isinstance(member, str) or not member for member in members):
            raise _invalid()
        copied[name] = tuple(members)
    for name in _OPTIONAL_STRINGS:
        item = value[name]
        if item is not None and (not isinstance(item, str) or not item):
            raise _invalid()

    ledger = value["coverage_ledger"]
    if not isinstance(ledger, tuple):
        raise _invalid()
    ledger_items = cast("tuple[object, ...]", ledger)
    copied["coverage_ledger"] = tuple(
        _revalidate_model(item, CoverageLedgerEntry) for item in ledger_items
    )

    blocked = value["blocked_needs"]
    if not isinstance(blocked, tuple):
        raise _invalid()
    blocked_items = cast("tuple[object, ...]", blocked)
    copied["blocked_needs"] = tuple(
        _validate_blocked_need(item) for item in blocked_items
    )

    gains = value["recent_marginal_gains"]
    if not isinstance(gains, tuple):
        raise _invalid()
    gain_items = cast("tuple[object, ...]", gains)
    if any(
        isinstance(item, bool)
        or not isinstance(item, (int, float))
        or not math.isfinite(float(item))
        or float(item) < 0.0
        for item in gain_items
    ):
        raise _invalid()
    copied["recent_marginal_gains"] = tuple(
        float(cast("int | float", item)) for item in gain_items
    )

    copied["budget_snapshot"] = _revalidate_model(
        value["budget_snapshot"], BudgetSnapshot
    )
    stop_reason = value["stop_reason"]
    if stop_reason is not None and (
        type(stop_reason) is not str or stop_reason not in _STOP_REASONS
    ):
        raise _invalid()
    if type(value["is_partial"]) is not bool:
        raise _invalid()
    next_event_seq = value["next_event_seq"]
    if type(next_event_seq) is not int or next_event_seq < 1:
        raise _invalid()
    elapsed = value["elapsed_wall_seconds"]
    if (
        isinstance(elapsed, bool)
        or not isinstance(elapsed, (int, float))
        or not math.isfinite(float(elapsed))
        or float(elapsed) < 0.0
    ):
        raise _invalid()
    copied["elapsed_wall_seconds"] = float(elapsed)
    return cast("BaselineState", copied)


__all__ = [
    "BaselineBlockedNeed",
    "BaselineState",
    "StateValidationError",
    "validate_baseline_state",
]
