from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass
from enum import StrEnum

from deepresearch.runtime import BudgetSnapshot

from .contracts import PlannerState
from .ledger import CoverageLedger

KEY_COVERAGE_THRESHOLD = 0.85
OVERALL_COVERAGE_THRESHOLD = 0.80
PLATEAU_GAIN_THRESHOLD = 0.05
HIGH_PRIORITY_CONFLICT_THRESHOLD = 0.70


class StopCode(StrEnum):
    SUFFICIENT = "SUFFICIENT"
    PLATEAU = "PLATEAU"
    BUDGET_EXHAUSTED = "BUDGET_EXHAUSTED"
    BLOCKED = "BLOCKED"


@dataclass(frozen=True)
class BlockedNeed:
    need_id: str
    required_source_unavailable: bool
    alternative_strategies_exhausted: bool
    retries_used: int
    max_retries: int

    def __post_init__(self) -> None:
        if type(self.need_id) is not str or not self.need_id.strip():
            raise ValueError("need_id is required")
        if type(self.required_source_unavailable) is not bool:
            raise TypeError("required_source_unavailable must be a bool")
        if type(self.alternative_strategies_exhausted) is not bool:
            raise TypeError("alternative_strategies_exhausted must be a bool")
        if type(self.max_retries) is not int or self.max_retries < 0:
            raise ValueError("retry counters are invalid")
        if type(self.retries_used) is not int or not 0 <= self.retries_used <= self.max_retries:
            raise ValueError("retry counters are invalid")

    @property
    def terminal(self) -> bool:
        return (
            self.required_source_unavailable
            and self.alternative_strategies_exhausted
            and self.retries_used >= self.max_retries
        )


@dataclass(frozen=True)
class StopDecision:
    code: StopCode
    reasons: tuple[str, ...]
    uncovered_information_needs: tuple[str, ...]
    is_partial: bool

    def __post_init__(self) -> None:
        if type(self.code) is not StopCode:
            raise TypeError("code must be a StopCode")
        if type(self.reasons) is not tuple or not self.reasons:
            raise ValueError("reasons must be a non-empty tuple")
        if any(type(reason) is not str or not reason.strip() for reason in self.reasons):
            raise ValueError("reasons must contain non-empty strings")
        if type(self.uncovered_information_needs) is not tuple:
            raise TypeError("uncovered_information_needs must be a tuple")
        if any(
            type(need_id) is not str or not need_id.strip()
            for need_id in self.uncovered_information_needs
        ):
            raise ValueError("uncovered_information_needs must contain non-empty strings")
        if type(self.is_partial) is not bool:
            raise TypeError("is_partial must be a bool")


def _uncovered_information_needs(ledger: CoverageLedger) -> tuple[str, ...]:
    uncovered: list[str] = []
    for subquestion in ledger.plan.subquestions:
        entry = ledger.get(subquestion.id)
        sources_ready = (
            entry.independent_source_count
            >= subquestion.evidence_requirements.min_independent_sources
        )
        if entry.coverage_score < KEY_COVERAGE_THRESHOLD or not sources_ready:
            uncovered.extend(need.need_id for need in subquestion.information_needs)
    return tuple(dict.fromkeys(uncovered))


def ledger_meets_sufficient(ledger: CoverageLedger) -> bool:
    if ledger.weighted_coverage() < OVERALL_COVERAGE_THRESHOLD:
        return False
    return all(
        entry.coverage_score >= KEY_COVERAGE_THRESHOLD
        and entry.independent_source_count
        >= subquestion.evidence_requirements.min_independent_sources
        for subquestion in ledger.plan.subquestions
        for entry in (ledger.get(subquestion.id),)
    )


def high_priority_conflicts(ledger: CoverageLedger) -> bool:
    return any(
        entry.unresolved_conflict_ids
        and subquestion.importance >= HIGH_PRIORITY_CONFLICT_THRESHOLD
        for subquestion in ledger.plan.subquestions
        for entry in (ledger.get(subquestion.id),)
    )


def evaluate_stop(
    state: PlannerState,
    budget_snapshot: BudgetSnapshot,
    *,
    blocked_needs: Collection[BlockedNeed] = (),
) -> StopDecision | None:
    uncovered = _uncovered_information_needs(state.ledger)
    if budget_snapshot.exhausted:
        return StopDecision(
            StopCode.BUDGET_EXHAUSTED,
            ("hard_budget_limit",),
            uncovered,
            True,
        )

    terminal_blocked = tuple(
        dict.fromkeys(sorted(item.need_id for item in blocked_needs if item.terminal))
    )
    if terminal_blocked:
        return StopDecision(
            StopCode.BLOCKED,
            ("required_source_and_alternatives_exhausted",),
            terminal_blocked,
            True,
        )

    if ledger_meets_sufficient(state.ledger) and not high_priority_conflicts(state.ledger):
        return StopDecision(StopCode.SUFFICIENT, ("coverage_thresholds_met",), (), False)

    if len(state.recent_marginal_gains) >= 2 and all(
        gain < PLATEAU_GAIN_THRESHOLD for gain in state.recent_marginal_gains[-2:]
    ):
        return StopDecision(StopCode.PLATEAU, ("two_low_gain_rounds",), uncovered, True)
    return None


__all__ = [
    "HIGH_PRIORITY_CONFLICT_THRESHOLD",
    "KEY_COVERAGE_THRESHOLD",
    "OVERALL_COVERAGE_THRESHOLD",
    "PLATEAU_GAIN_THRESHOLD",
    "BlockedNeed",
    "StopCode",
    "StopDecision",
    "evaluate_stop",
    "high_priority_conflicts",
    "ledger_meets_sufficient",
]
