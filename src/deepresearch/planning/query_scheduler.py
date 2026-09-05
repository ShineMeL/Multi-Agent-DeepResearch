from __future__ import annotations

import asyncio
import hashlib
import re
import unicodedata
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from math import isfinite, sqrt
from typing import Literal

from pydantic import JsonValue

from deepresearch.domain import ResourceUsage
from deepresearch.providers import Deadline, SearchHit, SearchProvider, TextEmbedder
from deepresearch.runtime import (
    BudgetAccountant,
    BudgetExceeded,
    CancellationToken,
    OperationCancelled,
    ResourceEstimate,
)

from .contracts import QueryCandidate

SEMANTIC_DEDUPE_THRESHOLD = 0.92


@dataclass(frozen=True)
class QueryBatchResult:
    results: tuple[tuple[str, tuple[SearchHit, ...]], ...]
    executed_queries: int
    skipped_queries: tuple[str, ...]
    skipped_reason: Literal["BUDGET_EXHAUSTED", "CANCELLED"] | None

    def __post_init__(self) -> None:
        if type(self.results) is not tuple:
            raise TypeError("results must be a tuple")
        if type(self.executed_queries) is not int or self.executed_queries < 0:
            raise ValueError("executed_queries must be a non-negative integer")
        if self.executed_queries != len(self.results):
            raise ValueError("executed_queries must match results length")
        if type(self.skipped_queries) is not tuple:
            raise TypeError("skipped_queries must be a tuple")
        if any(type(query) is not str or not query for query in self.skipped_queries):
            raise ValueError("skipped_queries must contain non-empty strings")
        if self.skipped_reason not in (None, "BUDGET_EXHAUSTED", "CANCELLED"):
            raise ValueError("skipped_reason is invalid")
        if self.skipped_queries and self.skipped_reason is None:
            raise ValueError("skipped_reason is required when queries are skipped")


@dataclass(frozen=True)
class _DispatchOutcome:
    normalized_query: str
    hits: tuple[SearchHit, ...] | None = None
    skipped_reason: Literal["BUDGET_EXHAUSTED", "CANCELLED"] | None = None


def _collapse_whitespace(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    if len(left) != len(right) or not left:
        raise ValueError("embedding dimensions must match and be non-empty")
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = sqrt(sum(value * value for value in left))
    right_norm = sqrt(sum(value * value for value in right))
    if not isfinite(dot) or not isfinite(left_norm) or not isfinite(right_norm):
        raise ValueError("embedding values must be finite")
    if left_norm == 0.0 or right_norm == 0.0:
        return 0.0
    return max(-1.0, min(1.0, dot / (left_norm * right_norm)))


class QueryScheduler:
    def __init__(self, *, embedder: TextEmbedder, budget: BudgetAccountant) -> None:
        self._embedder = embedder
        self._budget = budget

    def normalize(self, query: str) -> str:
        if type(query) is not str:
            raise TypeError("query must be a string")
        text = unicodedata.normalize("NFC", query).casefold()
        normalized_chars = [
            " " if unicodedata.category(char).startswith("P") else char for char in text
        ]
        return _collapse_whitespace("".join(normalized_chars))

    async def dedupe(
        self,
        candidates: Sequence[QueryCandidate],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[QueryCandidate]:
        exact: list[tuple[QueryCandidate, str]] = []
        positions: dict[str, int] = {}
        for candidate in candidates:
            normalized = self.normalize(candidate.query)
            if not normalized:
                raise ValueError("query normalizes to an empty string")
            position = positions.get(normalized)
            if position is None:
                positions[normalized] = len(exact)
                exact.append((candidate, normalized))
            elif candidate.priority_hint > exact[position][0].priority_hint:
                exact[position] = (candidate, normalized)

        if not exact:
            return []
        cancellation_token.raise_if_cancelled()
        vectors = await self._embedder.embed(
            [candidate.query for candidate, _ in exact],
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        cancellation_token.raise_if_cancelled()
        if len(vectors) != len(exact):
            raise ValueError("embedder returned an unexpected number of vectors")

        kept: list[tuple[QueryCandidate, str, tuple[float, ...]]] = []
        for (candidate, normalized), vector in zip(exact, vectors, strict=True):
            typed_vector = tuple(vector)
            matches = [
                index
                for index, (_, _, kept_vector) in enumerate(kept)
                if _cosine(typed_vector, kept_vector) >= SEMANTIC_DEDUPE_THRESHOLD
            ]
            if not matches:
                kept.append((candidate, normalized, typed_vector))
                continue
            best_match = max(matches, key=lambda index: kept[index][0].priority_hint)
            if candidate.priority_hint > kept[best_match][0].priority_hint:
                first_match = matches[0]
                kept = [item for index, item in enumerate(kept) if index not in matches]
                kept.insert(first_match, (candidate, normalized, typed_vector))
            else:
                kept = [item for index, item in enumerate(kept) if index not in matches[1:]]

        return [candidate for candidate, _, _ in kept]

    async def dispatch(
        self,
        candidates: Sequence[QueryCandidate],
        provider: SearchProvider,
        *,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        max_concurrency: int,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> QueryBatchResult:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        if type(max_concurrency) is not int or max_concurrency <= 0:
            raise ValueError("max_concurrency must be a positive integer")

        semaphore = asyncio.Semaphore(max_concurrency)

        async def execute(candidate: QueryCandidate) -> _DispatchOutcome:
            normalized = self.normalize(candidate.query)
            if not normalized:
                raise ValueError("query normalizes to an empty string")
            async with semaphore:
                try:
                    cancellation_token.raise_if_cancelled()
                    estimate = ResourceEstimate(
                        search_calls=max(1, candidate.estimated_search_calls),
                        tokens=candidate.estimated_tokens,
                        wall_seconds=max(0.001, candidate.estimated_seconds),
                        cost_usd=Decimal(0),
                    )
                    reservation = self._budget.reserve(
                        estimate,
                        node="Tool",
                        idempotency_key="search:"
                        + hashlib.sha256(normalized.encode("utf-8")).hexdigest(),
                    )
                except OperationCancelled:
                    return _DispatchOutcome(normalized, skipped_reason="CANCELLED")
                except BudgetExceeded:
                    return _DispatchOutcome(normalized, skipped_reason="BUDGET_EXHAUSTED")

                try:
                    hits = await provider.search(
                        candidate.query,
                        limit,
                        filters,
                        deadline=deadline,
                        cancellation_token=cancellation_token,
                    )
                    cancellation_token.raise_if_cancelled()
                    actual = ResourceUsage.zero(cost_known=True).model_copy(
                        update={"search_calls": 1}
                    )
                    self._budget.settle(reservation, actual=actual)
                    return _DispatchOutcome(normalized, tuple(hits))
                except OperationCancelled:
                    self._budget.release(reservation)
                    return _DispatchOutcome(normalized, skipped_reason="CANCELLED")
                except Exception:
                    self._budget.release(reservation)
                    raise

        outcomes = await asyncio.gather(*(execute(candidate) for candidate in candidates))
        result_pairs: list[tuple[str, tuple[SearchHit, ...]]] = []
        for outcome in outcomes:
            if outcome.hits is not None:
                result_pairs.append((outcome.normalized_query, outcome.hits))
        result_pairs.sort(key=lambda item: item[0])
        skipped = tuple(
            sorted(
                {
                    outcome.normalized_query
                    for outcome in outcomes
                    if outcome.skipped_reason is not None
                }
            )
        )
        reasons = {outcome.skipped_reason for outcome in outcomes}
        skipped_reason: Literal["BUDGET_EXHAUSTED", "CANCELLED"] | None
        if "CANCELLED" in reasons:
            skipped_reason = "CANCELLED"
        elif "BUDGET_EXHAUSTED" in reasons:
            skipped_reason = "BUDGET_EXHAUSTED"
        else:
            skipped_reason = None
        return QueryBatchResult(
            results=tuple(result_pairs),
            executed_queries=len(result_pairs),
            skipped_queries=skipped,
            skipped_reason=skipped_reason,
        )


__all__ = ["SEMANTIC_DEDUPE_THRESHOLD", "QueryBatchResult", "QueryScheduler"]
