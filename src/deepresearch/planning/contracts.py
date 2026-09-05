from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Any, Literal

from deepresearch.domain import ResearchPlan
from deepresearch.runtime import BudgetSnapshot

# These aliases intentionally remain loose until Tasks 2 and 3 provide the
# concrete ledger and stop contracts.  The dataclass annotations are postponed
# so the later modules can tighten them without introducing import cycles here.
CoverageLedger = Any
BlockedNeed = Any
StopDecision = Any


def _require_non_empty(value: object, *, field: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")


def _require_non_negative(value: object, *, field: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not isfinite(value) or value < 0:
        raise ValueError(f"{field} must be non-negative and finite")


@dataclass(frozen=True)
class QueryCandidate:
    subquestion_id: str
    information_need_id: str
    query: str
    priority_hint: float
    estimated_tokens: int
    estimated_search_calls: int
    estimated_seconds: float

    def __post_init__(self) -> None:
        _require_non_empty(self.subquestion_id, field="subquestion_id")
        _require_non_empty(self.information_need_id, field="information_need_id")
        _require_non_empty(self.query, field="query")
        priority_hint: object = self.priority_hint
        if (
            isinstance(priority_hint, bool)
            or type(priority_hint) not in (int, float)
            or not isfinite(priority_hint)
            or not 0.0 <= priority_hint <= 1.0
        ):
            raise ValueError("priority_hint must be finite and within [0, 1]")
        if type(self.estimated_tokens) is not int or self.estimated_tokens < 0:
            raise ValueError("estimated_tokens must be a non-negative integer")
        if (
            type(self.estimated_search_calls) is not int
            or self.estimated_search_calls < 0
        ):
            raise ValueError("estimated_search_calls must be a non-negative integer")
        _require_non_negative(self.estimated_seconds, field="estimated_seconds")


@dataclass(frozen=True)
class PlannerState:
    plan: ResearchPlan
    ledger: CoverageLedger
    budget_snapshot: BudgetSnapshot
    blocked_needs: tuple[BlockedNeed, ...]
    round_index: int
    recent_marginal_gains: tuple[float, ...]
    query_history: tuple[str, ...]

    def __post_init__(self) -> None:
        ledger: object = self.ledger
        if ledger is None:
            raise ValueError("ledger must be a CoverageLedger")
        blocked_needs: object = self.blocked_needs
        if type(blocked_needs) is not tuple:
            raise TypeError("blocked_needs must be a tuple")
        if type(self.round_index) is not int or self.round_index < 0:
            raise ValueError("round_index must be a non-negative integer")
        recent_marginal_gains: object = self.recent_marginal_gains
        if type(recent_marginal_gains) is not tuple:
            raise TypeError("recent_marginal_gains must be a tuple")
        for gain in recent_marginal_gains:
            _require_non_negative(gain, field="recent_marginal_gains")
        query_history: object = self.query_history
        if type(query_history) is not tuple:
            raise TypeError("query_history must be a tuple")
        for query in query_history:
            _require_non_empty(query, field="query_history entry")


@dataclass(frozen=True)
class PlannerDecision:
    kind: Literal["SEARCH", "STOP"]
    subquestion_id: str | None
    candidates: tuple[QueryCandidate, ...]
    stop: StopDecision | None
    decision_code: str

    def __post_init__(self) -> None:
        if self.kind not in ("SEARCH", "STOP"):
            raise ValueError("kind must be SEARCH or STOP")
        _require_non_empty(self.decision_code, field="decision_code")
        candidates: object = self.candidates
        if type(candidates) is not tuple:
            raise TypeError("candidates must be a tuple")
        stop: object | None = self.stop
        if self.kind == "STOP" and (
            self.subquestion_id is not None or candidates or stop is None
        ):
            raise ValueError("STOP decision cannot contain search fields")
        if self.kind == "SEARCH" and (
            self.subquestion_id is None or not candidates or stop is not None
        ):
            raise ValueError("SEARCH decision requires search fields")
        if self.subquestion_id is not None:
            _require_non_empty(self.subquestion_id, field="subquestion_id")


__all__ = ["PlannerDecision", "PlannerState", "QueryCandidate"]
