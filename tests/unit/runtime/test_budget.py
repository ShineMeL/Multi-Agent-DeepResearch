from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal

import pytest
from pydantic import ValidationError

from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.runtime import (
    BudgetAccountant,
    BudgetExceeded,
    ResourceEstimate,
)

NODES = {"Planner", "Ranker", "Writer", "Judge", "Tool"}


def usage(
    *,
    tokens: int = 0,
    search_calls: int = 0,
    pages: int = 0,
    retries: int = 0,
    wall_seconds: float = 0.0,
    cost: str | None = "0",
) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=tokens,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=tokens,
        search_calls=search_calls,
        pages=pages,
        retries=retries,
        wall_seconds=wall_seconds,
        cost_usd=None if cost is None else Decimal(cost),
    )


def estimate(
    *,
    tokens: int = 100,
    search_calls: int = 1,
    pages: int = 1,
    retries: int = 0,
    wall_seconds: float = 1.0,
    cost: str | None = "0.01",
) -> ResourceEstimate:
    return ResourceEstimate(
        tokens=tokens,
        search_calls=search_calls,
        pages=pages,
        retries=retries,
        wall_seconds=wall_seconds,
        cost_usd=None if cost is None else Decimal(cost),
    )


def test_budget_reserve_rejects_all_hard_limit_overruns_without_mutation() -> None:
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    before = accountant.snapshot()

    with pytest.raises(BudgetExceeded) as error:
        accountant.reserve(
            ResourceEstimate(
                search_calls=9,
                pages=13,
                tokens=40_001,
                wall_seconds=301,
                cost_usd=Decimal("0.51"),
                retries=3,
            ),
            node="Tool",
            idempotency_key="overrun",
        )

    assert error.value.dimensions == {
        "search_calls",
        "pages",
        "tokens",
        "wall_seconds",
        "cost_usd",
        "retries",
    }
    assert error.value.snapshot == before
    assert accountant.snapshot() == before


def test_reservations_count_immediately_and_repeats_are_idempotent() -> None:
    accountant = BudgetAccountant(RunBudget.preset("medium"), run_scope="run-7")
    first = accountant.reserve(estimate(), node="Planner", idempotency_key="request-1")
    second = accountant.reserve(estimate(), node="Planner", idempotency_key="request-1")
    snapshot = accountant.snapshot()

    assert first == second
    assert len(first.reservation_id) == 64
    assert snapshot.reserved_tokens == 100
    with pytest.raises(ValueError, match="idempotency_key"):
        accountant.reserve(
            estimate(tokens=101), node="Planner", idempotency_key="request-1"
        )


def test_failed_call_with_usage_is_charged_once_and_snapshot_is_immutable() -> None:
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    reservation = accountant.reserve(estimate(), node="Planner", idempotency_key="m1")
    first = accountant.settle(reservation, actual=usage(tokens=100, cost="0.01"))
    second = accountant.settle(reservation, actual=usage(tokens=100, cost="0.01"))

    assert first == second
    assert second.used_tokens == 100
    assert second.reserved_tokens == 0
    assert set(second.used_by_node) == NODES
    with pytest.raises(TypeError, match="immutable"):
        second.used_by_node["Planner"] = ResourceUsage.zero()


def test_cached_usage_is_observable_but_not_charged_again() -> None:
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    reservation = accountant.reserve(estimate(), node="Tool", idempotency_key="cached")
    snapshot = accountant.settle(
        reservation,
        actual=usage(tokens=100, cost="0.01"),
        charge=False,
    )

    assert snapshot.used_tokens == 0
    assert snapshot.used_by_node["Tool"].total_tokens == 0
    assert snapshot.last_observed_usage.total_tokens == 100


def test_release_is_idempotent_but_cross_accountant_reservations_are_rejected() -> None:
    left = BudgetAccountant(RunBudget.preset("medium"), run_scope="left")
    right = BudgetAccountant(RunBudget.preset("medium"), run_scope="right")
    reservation = left.reserve(estimate(), node="Tool", idempotency_key="fetch")

    first = left.release(reservation)
    second = left.release(reservation)

    assert first == second
    assert first.reserved_pages == 0
    with pytest.raises(ValueError, match="accountant|unknown"):
        right.release(reservation)
    with pytest.raises(ValueError, match="accountant|unknown"):
        right.settle(reservation, actual=usage())


def test_unknown_cost_budget_keeps_cost_totals_none_but_enforces_tokens() -> None:
    token_only = RunBudget.preset("medium").model_copy(update={"max_cost_usd": None})
    accountant = BudgetAccountant(token_only)
    reservation = accountant.reserve(
        estimate(cost="99.00"), node="Writer", idempotency_key="draft"
    )
    reserved = accountant.snapshot()
    settled = accountant.settle(
        reservation, actual=usage(tokens=100, cost="99.00")
    )

    assert reserved.reserved_cost_usd is None
    assert settled.used_cost_usd is None
    with pytest.raises(BudgetExceeded) as error:
        accountant.reserve(
            estimate(tokens=40_000, cost=None),
            node="Writer",
            idempotency_key="too-many-tokens",
        )
    assert error.value.dimensions == {"tokens"}


def test_exhausted_uses_charged_plus_reserved_and_all_canonical_dimensions() -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    accountant.reserve(
        ResourceEstimate(
            search_calls=4,
            pages=8,
            tokens=20_000,
            wall_seconds=180,
            cost_usd=Decimal("0.25"),
            retries=2,
        ),
        node="Tool",
        idempotency_key="fill",
    )

    assert accountant.snapshot().exhausted == {
        "search_calls",
        "pages",
        "tokens",
        "wall_seconds",
        "cost_usd",
        "retries",
    }


def test_reserve_rejects_noncanonical_nodes_and_invalid_estimates() -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))

    with pytest.raises(ValueError, match="canonical"):
        accountant.reserve(estimate(), node="Search", idempotency_key="bad-node")
    with pytest.raises(ValueError, match="idempotency_key"):
        accountant.reserve(estimate(), node="Tool", idempotency_key="")
    with pytest.raises(ValidationError):
        ResourceEstimate(tokens=-1)


def test_concurrent_duplicate_reservations_charge_capacity_once() -> None:
    accountant = BudgetAccountant(RunBudget.preset("medium"))

    def reserve_once(_: int) -> str:
        return accountant.reserve(
            estimate(), node="Ranker", idempotency_key="same"
        ).reservation_id

    with ThreadPoolExecutor(max_workers=8) as pool:
        reservation_ids = tuple(pool.map(reserve_once, range(100)))

    assert len(set(reservation_ids)) == 1
    assert accountant.snapshot().reserved_tokens == 100


def test_concurrent_settlement_charges_actual_usage_once() -> None:
    accountant = BudgetAccountant(RunBudget.preset("medium"))
    reservation = accountant.reserve(
        estimate(), node="Judge", idempotency_key="same-settlement"
    )

    def settle_once(_: int) -> int:
        return accountant.settle(
            reservation, actual=usage(tokens=100, cost="0.01")
        ).used_tokens

    with ThreadPoolExecutor(max_workers=8) as pool:
        totals = tuple(pool.map(settle_once, range(100)))

    assert totals == (100,) * 100
    assert accountant.snapshot().used_tokens == 100
