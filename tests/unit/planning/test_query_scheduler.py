from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence

import pytest
from pydantic import JsonValue

from deepresearch.domain import RunBudget
from deepresearch.planning.contracts import QueryCandidate
from deepresearch.planning.query_scheduler import QueryScheduler
from deepresearch.providers import SearchHit
from deepresearch.runtime import BudgetAccountant, CancellationToken


class FakeEmbedder:
    provider_id = "fake-embedder"
    model_id = "fake-model"
    model_revision = "1"
    snapshot_sha256 = "a" * 64

    def __init__(self, vectors: list[list[float]]) -> None:
        self.vectors = [tuple(vector) for vector in vectors]
        self.calls: list[tuple[str, ...]] = []

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.calls.append(tuple(texts))
        return tuple(self.vectors[: len(texts)])


class FakeSearchProvider:
    provider_id = "fake-search"

    def __init__(self) -> None:
        self.calls: list[str] = []

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del limit, filters, deadline
        cancellation_token.raise_if_cancelled()
        self.calls.append(query)
        await asyncio.sleep(0)
        return []


def budget_with_one_search_call() -> BudgetAccountant:
    budget = RunBudget.preset("low").model_copy(
        update={"max_search_calls": 1, "max_cost_usd": None}
    )
    return BudgetAccountant(budget, run_scope="scheduler-test")


def candidate(query: str, priority: float = 0.5) -> QueryCandidate:
    return QueryCandidate("sq-1", "need-1", query, priority, 10, 1, 1.0)


def token() -> CancellationToken:
    return CancellationToken()


@pytest.mark.asyncio
async def test_scheduler_semantic_dedupe_keeps_highest_priority() -> None:
    scheduler = QueryScheduler(
        embedder=FakeEmbedder([[1, 0], [0.999, 0.02]]),
        budget=budget_with_one_search_call(),
    )

    result = await scheduler.dedupe(
        [candidate("q-low", 0.2), candidate("q-high", 0.9)],
        deadline=100,
        cancellation_token=token(),
    )

    assert [item.query for item in result] == ["q-high"]


def test_scheduler_normalizes_unicode_case_whitespace_and_punctuation() -> None:
    scheduler = QueryScheduler(
        embedder=FakeEmbedder([]),
        budget=budget_with_one_search_call(),
    )

    assert scheduler.normalize("  Multimodal—Agents！\n") == "multimodal agents"


@pytest.mark.asyncio
async def test_scheduler_exact_dedupe_keeps_highest_priority() -> None:
    scheduler = QueryScheduler(
        embedder=FakeEmbedder([[1, 0]]),
        budget=budget_with_one_search_call(),
    )

    result = await scheduler.dedupe(
        [candidate("Q: planner", 0.2), candidate(" q planner! ", 0.9)],
        deadline=100,
        cancellation_token=token(),
    )

    assert [item.query for item in result] == [" q planner! "]


@pytest.mark.asyncio
async def test_dispatch_deducts_budget_before_search() -> None:
    scheduler = QueryScheduler(
        embedder=FakeEmbedder([]),
        budget=budget_with_one_search_call(),
    )
    provider = FakeSearchProvider()

    result = await scheduler.dispatch(
        [candidate("q-1"), candidate("q-2")],
        provider,
        limit=5,
        filters=None,
        max_concurrency=2,
        deadline=100,
        cancellation_token=token(),
    )

    assert result.executed_queries == 1
    assert result.skipped_reason == "BUDGET_EXHAUSTED"
    assert len(result.skipped_queries) == 1
    assert len(provider.calls) == 1


@pytest.mark.asyncio
async def test_dispatch_reports_cancellation_without_searching() -> None:
    scheduler = QueryScheduler(
        embedder=FakeEmbedder([]),
        budget=budget_with_one_search_call(),
    )
    provider = FakeSearchProvider()
    cancellation = token()
    cancellation.cancel()

    result = await scheduler.dispatch(
        [candidate("q-1")],
        provider,
        limit=5,
        filters=None,
        max_concurrency=1,
        deadline=100,
        cancellation_token=cancellation,
    )

    assert result.executed_queries == 0
    assert result.skipped_reason == "CANCELLED"
    assert result.skipped_queries == ("q 1",)
    assert provider.calls == []
