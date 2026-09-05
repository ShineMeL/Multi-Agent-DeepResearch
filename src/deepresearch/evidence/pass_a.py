from __future__ import annotations

import re
from collections.abc import Sequence
from dataclasses import dataclass

from deepresearch.domain import SubQuestion
from deepresearch.providers import Deadline
from deepresearch.runtime import CancellationToken

from .normalize import EvidenceCandidate, EvidenceNormalizer


@dataclass(frozen=True)
class PassAResult:
    selected: tuple[EvidenceCandidate, ...]
    rejected_evidence_ids: tuple[str, ...]
    used_context_tokens: int

    def __post_init__(self) -> None:
        if type(self.selected) is not tuple:
            raise TypeError("selected must be a tuple")
        if type(self.rejected_evidence_ids) is not tuple:
            raise TypeError("rejected_evidence_ids must be a tuple")
        if type(self.used_context_tokens) is not int or self.used_context_tokens < 0:
            raise ValueError("used_context_tokens must be non-negative")


def _terms(value: str) -> set[str]:
    return set(re.findall(r"\w+", value.casefold(), flags=re.UNICODE))


def _context_tokens(candidate: EvidenceCandidate) -> int:
    return max(1, len(candidate.evidence.excerpt.split()))


class PassASelector:
    def __init__(self, *, normalizer: EvidenceNormalizer | None = None) -> None:
        self._normalizer = normalizer or EvidenceNormalizer()

    async def select(
        self,
        subquestion: SubQuestion,
        candidates: Sequence[EvidenceCandidate],
        *,
        context_budget: int,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> PassAResult:
        del deadline
        if type(context_budget) is not int or context_budget < 0:
            raise ValueError("context_budget must be non-negative")
        cancellation_token.raise_if_cancelled()
        normalized = self._normalizer.dedupe(candidates)
        need_ids = {need.need_id for need in subquestion.information_needs}
        query_terms = _terms(
            " ".join(
                [subquestion.question]
                + [need.text for need in subquestion.information_needs]
            )
        )
        ranked: list[tuple[float, EvidenceCandidate]] = []
        rejected: set[str] = {
            candidate.evidence.evidence_id for candidate in candidates
        }
        for candidate in normalized:
            cancellation_token.raise_if_cancelled()
            evidence = candidate.evidence
            if not evidence.excerpt.strip() or not set(evidence.information_need_ids) & need_ids:
                continue
            terms = _terms(evidence.excerpt)
            overlap = len(query_terms & terms)
            score = overlap + 1.0 / max(1, candidate.search_rank)
            ranked.append((score, candidate))

        ranked.sort(
            key=lambda item: (
                -item[0],
                item[1].search_rank,
                item[1].evidence.evidence_id,
            )
        )
        selected: list[EvidenceCandidate] = []
        used_tokens = 0
        for _, candidate in ranked:
            cancellation_token.raise_if_cancelled()
            required_tokens = _context_tokens(candidate)
            if used_tokens + required_tokens > context_budget:
                continue
            selected.append(candidate)
            rejected.discard(candidate.evidence.evidence_id)
            used_tokens += required_tokens

        selected.sort(key=lambda item: (item.search_rank, item.evidence.evidence_id))
        return PassAResult(
            selected=tuple(selected),
            rejected_evidence_ids=tuple(sorted(rejected)),
            used_context_tokens=used_tokens,
        )


__all__ = ["PassAResult", "PassASelector"]
