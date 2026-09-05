from __future__ import annotations

from collections.abc import Mapping, Sequence
from math import isfinite
from types import MappingProxyType

from deepresearch.domain import (
    ClaimEvidenceLink,
    CoverageLedgerEntry,
    EvidenceSpan,
    ResearchPlan,
    SourceDocument,
)


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            result.append(value)
    return tuple(result)


def _clip(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(min(1.0, max(0.0, value)))


class CoverageLedger:
    """Immutable coverage snapshot keyed by the planned subquestion IDs."""

    _entries: Mapping[str, CoverageLedgerEntry]
    _plan: ResearchPlan

    __slots__ = ("_entries", "_plan")

    def __init__(
        self,
        plan: ResearchPlan,
        entries: Mapping[str, CoverageLedgerEntry] | None = None,
    ) -> None:
        if type(plan) is not ResearchPlan:
            raise TypeError("plan must be a ResearchPlan")
        if entries is None:
            raise ValueError("entries must contain exactly one item per planned subquestion")
        planned_ids = tuple(subquestion.id for subquestion in plan.subquestions)
        if set(entries) != set(planned_ids) or len(entries) != len(planned_ids):
            raise ValueError("entries must contain exactly one item per planned subquestion")
        normalized: dict[str, CoverageLedgerEntry] = {}
        for subquestion_id in planned_ids:
            entry = entries[subquestion_id]
            if type(entry) is not CoverageLedgerEntry:
                raise TypeError("entries must contain CoverageLedgerEntry values")
            if entry.subquestion_id != subquestion_id:
                raise ValueError("ledger entry key must match subquestion_id")
            normalized[subquestion_id] = entry

        object.__setattr__(self, "_plan", plan)
        object.__setattr__(self, "_entries", MappingProxyType(normalized))

    @classmethod
    def empty_for(cls, plan: ResearchPlan) -> CoverageLedger:
        return cls(
            plan,
            {
                subquestion.id: CoverageLedgerEntry(
                    subquestion_id=subquestion.id,
                    coverage_score=0.0,
                    independent_source_count=0,
                    unresolved_conflict_ids=(),
                    uncertainty_score=1.0,
                    last_marginal_gain=0.0,
                    evidence_ids=(),
                    attempt_count=0,
                    last_decision_code="NOT_ATTEMPTED",
                )
                for subquestion in plan.subquestions
            },
        )

    @property
    def plan(self) -> ResearchPlan:
        return self._plan

    def get(self, subquestion_id: str) -> CoverageLedgerEntry:
        return self._entries[subquestion_id]

    def entries(self) -> tuple[CoverageLedgerEntry, ...]:
        return tuple(self._entries.values())

    def weighted_coverage(self) -> float:
        weighted_total = 0.0
        importance_total = 0.0
        for subquestion in self._plan.subquestions:
            importance_total += subquestion.importance
            weighted_total += subquestion.importance * self._entries[subquestion.id].coverage_score
        if importance_total == 0.0:
            return 0.0
        return _clip(weighted_total / importance_total, field="weighted_coverage")

    def replace(self, entry: CoverageLedgerEntry) -> CoverageLedger:
        if type(entry) is not CoverageLedgerEntry:
            raise TypeError("entry must be a CoverageLedgerEntry")
        if entry.subquestion_id not in self._entries:
            raise KeyError(entry.subquestion_id)
        updated = dict(self._entries)
        updated[entry.subquestion_id] = entry
        return type(self)(self._plan, updated)


def update_coverage(
    ledger: CoverageLedger,
    subquestion_id: str,
    *,
    selected_evidence: Sequence[EvidenceSpan],
    links: Sequence[ClaimEvidenceLink],
    source_documents: Mapping[str, SourceDocument],
    marginal_gain: float,
    decision_code: str,
) -> CoverageLedger:
    if type(ledger) is not CoverageLedger:
        raise TypeError("ledger must be a CoverageLedger")
    entry = ledger.get(subquestion_id)
    decision_code_obj: object = decision_code
    if type(decision_code_obj) is not str or not decision_code_obj.strip():
        raise ValueError("decision_code must be a non-empty string")

    source_families: list[str] = []
    evidence_ids: list[str] = list(entry.evidence_ids)
    for evidence in selected_evidence:
        if type(evidence) is not EvidenceSpan:
            raise TypeError("selected_evidence must contain EvidenceSpan values")
        document = source_documents.get(evidence.source_id)
        if document is None:
            raise ValueError(f"missing source document for {evidence.source_id}")
        if document.source_id != evidence.source_id:
            raise ValueError("source document key must match source_id")
        evidence_ids.append(evidence.evidence_id)
        source_families.append(document.source_family_id)

    conflict_ids = list(entry.unresolved_conflict_ids)
    for link in links:
        if type(link) is not ClaimEvidenceLink:
            raise TypeError("links must contain ClaimEvidenceLink values")
        if link.relation == "contradict":
            conflict_ids.append(link.claim_id)

    coverage_score = _clip(
        entry.coverage_score + _clip(marginal_gain, field="marginal_gain"),
        field="coverage_score",
    )
    selected_family_count = len(_unique(source_families))
    updated_entry = entry.model_copy(
        update={
            "coverage_score": coverage_score,
            "independent_source_count": max(
                entry.independent_source_count, selected_family_count
            ),
            "unresolved_conflict_ids": _unique(conflict_ids),
            "uncertainty_score": _clip(1.0 - coverage_score, field="uncertainty_score"),
            "last_marginal_gain": _clip(marginal_gain, field="marginal_gain"),
            "evidence_ids": _unique(evidence_ids),
            "attempt_count": entry.attempt_count + 1,
            "last_decision_code": decision_code,
        }
    )
    return ledger.replace(updated_entry)


__all__ = ["CoverageLedger", "update_coverage"]
