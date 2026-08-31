from concurrent.futures import ThreadPoolExecutor
from decimal import Decimal
from typing import cast

import pytest
from pydantic import BaseModel, ValidationError

from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.runtime import (
    BudgetAccountant,
    BudgetExceeded,
    BudgetSnapshot,
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


def unsafe_usage(**updates: object) -> ResourceUsage:
    payload = usage(tokens=1, cost="0.01").model_dump()
    payload.update(updates)
    return ResourceUsage.model_construct(**payload)


def unsafe_estimate(**updates: object) -> ResourceEstimate:
    payload = estimate(cost="0.01").model_dump()
    payload.update(updates)
    return ResourceEstimate.model_construct(**payload)


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


def test_enabled_cost_budget_rejects_unknown_reservation_without_mutation() -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    before = accountant.snapshot()

    with pytest.raises(ValueError, match="known cost"):
        accountant.reserve(
            estimate(tokens=1, cost=None),
            node="Tool",
            idempotency_key="unknown-cost",
        )

    assert accountant.snapshot() == before
    reservation = accountant.reserve(
        estimate(tokens=1, cost="0.25"),
        node="Tool",
        idempotency_key="known-cost",
    )
    assert reservation.estimate.cost_usd == Decimal("0.25")
    assert accountant.snapshot().reserved_cost_usd == Decimal("0.25")


@pytest.mark.parametrize(
    "bad_estimate",
    [
        unsafe_estimate(cost_usd=Decimal("-0.01")),
        unsafe_estimate(cost_usd=Decimal("NaN")),
        unsafe_estimate(cost_usd=Decimal("Infinity")),
        unsafe_estimate(wall_seconds=-1.0),
        unsafe_estimate(wall_seconds=float("inf")),
    ],
)
def test_reserve_revalidates_adversarial_estimate_instances(
    bad_estimate: ResourceEstimate,
) -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    before = accountant.snapshot()

    with pytest.raises(ValueError):
        accountant.reserve(
            bad_estimate, node="Tool", idempotency_key="invalid-estimate"
        )

    assert accountant.snapshot() == before


def test_enabled_cost_budget_rejects_unknown_charged_actual_atomically() -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    reservation = accountant.reserve(
        estimate(cost="0.01"), node="Tool", idempotency_key="unknown-actual"
    )
    before = accountant.snapshot()

    with pytest.raises(ValueError, match="known cost"):
        accountant.settle(reservation, actual=usage(tokens=1, cost=None))

    assert accountant.snapshot() == before
    settled = accountant.settle(
        reservation, actual=usage(tokens=1, cost="0.01")
    )
    assert settled.used_cost_usd == Decimal("0.01")


@pytest.mark.parametrize(
    "bad_actual",
    [
        unsafe_usage(cost_usd=Decimal("-0.01")),
        unsafe_usage(cost_usd=Decimal("NaN")),
        unsafe_usage(cost_usd=Decimal("Infinity")),
        unsafe_usage(search_calls=-1),
        unsafe_usage(pages=-1),
        unsafe_usage(retries=-1),
        unsafe_usage(input_tokens=-1, total_tokens=-1),
        unsafe_usage(wall_seconds=-1.0),
        unsafe_usage(wall_seconds=float("inf")),
    ],
)
def test_rejected_settlement_is_atomic_and_reservation_remains_repeatable(
    bad_actual: ResourceUsage,
) -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    reservation = accountant.reserve(
        estimate(cost="0.01"), node="Judge", idempotency_key="bad-actual"
    )
    before = accountant.snapshot()

    with pytest.raises(ValueError, match="finite|non-negative"):
        accountant.settle(reservation, actual=bad_actual)

    assert accountant.snapshot() == before
    settled = accountant.settle(
        reservation, actual=usage(tokens=1, cost="0.01")
    )
    assert settled.used_tokens == 1
    assert settled.used_cost_usd == Decimal("0.01")


@pytest.mark.parametrize(
    "bad_actual",
    [
        unsafe_usage(cost_usd=Decimal("-0.01")),
        unsafe_usage(cost_usd=Decimal("NaN")),
        unsafe_usage(cost_usd=Decimal("Infinity")),
        unsafe_usage(wall_seconds=float("inf"), cost_usd=None),
    ],
)
def test_uncharged_observation_still_rejects_invalid_numeric_usage(
    bad_actual: ResourceUsage,
) -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    reservation = accountant.reserve(
        estimate(cost="0.01"), node="Tool", idempotency_key="bad-observation"
    )
    before = accountant.snapshot()

    with pytest.raises(ValueError, match="finite|non-negative"):
        accountant.settle(reservation, actual=bad_actual, charge=False)

    assert accountant.snapshot() == before


def test_uncharged_unknown_cost_is_observable_without_nulling_enabled_totals() -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    reservation = accountant.reserve(
        estimate(cost="0.01"), node="Tool", idempotency_key="unknown-observation"
    )

    snapshot = accountant.settle(
        reservation, actual=usage(tokens=1, cost=None), charge=False
    )

    assert snapshot.used_cost_usd == Decimal(0)
    assert snapshot.reserved_cost_usd == Decimal(0)
    assert snapshot.last_observed_usage.cost_usd is None


@pytest.mark.parametrize(
    "bad_seed",
    [
        unsafe_usage(cost_usd=Decimal("-0.01")),
        unsafe_usage(cost_usd=Decimal("NaN")),
        unsafe_usage(cost_usd=Decimal("Infinity")),
        unsafe_usage(wall_seconds=float("inf")),
        unsafe_usage(cost_usd=None),
    ],
)
def test_accountant_rejects_invalid_or_unknown_chargeable_seed_usage(
    bad_seed: ResourceUsage,
) -> None:
    budget = RunBudget.preset("low").model_copy(
        update={
            "used_by_node": {
                **RunBudget.preset("low").used_by_node,
                "Tool": bad_seed,
            }
        }
    )

    with pytest.raises(ValueError, match="finite|non-negative|known cost"):
        BudgetAccountant(budget)


def test_accountant_rejects_finite_seed_wall_times_that_overflow_in_aggregate() -> None:
    huge = usage(tokens=1, wall_seconds=1e308, cost="0")
    budget = RunBudget.preset("low").model_copy(
        update={"used_by_node": {node: huge for node in NODES}}
    )

    with pytest.raises(ValueError, match="aggregate|finite|wall"):
        BudgetAccountant(budget)


def test_accountant_accepts_finite_seed_usage_that_already_exhausts_a_limit() -> None:
    historical = usage(tokens=1, wall_seconds=100.0, cost="0")
    budget = RunBudget.preset("low").model_copy(
        update={"used_by_node": {node: historical for node in NODES}}
    )

    snapshot = BudgetAccountant(budget).snapshot()

    assert snapshot.used_wall_seconds == 500.0
    assert "wall_seconds" in snapshot.exhausted


def test_from_snapshot_restores_exact_usage_and_last_observation_without_reservations() -> None:
    budget = RunBudget.preset("low")
    original = BudgetAccountant(budget, run_scope="original")
    charged = original.reserve(
        estimate(tokens=10, cost="0.01"),
        node="Planner",
        idempotency_key="charged",
    )
    original.settle(charged, actual=usage(tokens=7, cost="0.01"))
    observed = original.reserve(
        estimate(tokens=5, cost="0.01"),
        node="Tool",
        idempotency_key="observed",
    )
    expected = original.settle(
        observed,
        actual=usage(tokens=5, retries=1, cost=None),
        charge=False,
    )

    restored = BudgetAccountant.from_snapshot(
        budget,
        expected,
        run_scope="restored",
    )

    assert restored.snapshot() == expected
    replacement = restored.reserve(
        estimate(tokens=1, cost="0.01"),
        node="Planner",
        idempotency_key="charged",
    )
    assert replacement.reservation_id != charged.reservation_id


def test_from_snapshot_rejects_any_reserved_capacity() -> None:
    accountant = BudgetAccountant(RunBudget.preset("low"))
    accountant.reserve(
        estimate(tokens=1, cost="0.01"),
        node="Tool",
        idempotency_key="active",
    )

    with pytest.raises(ValueError, match="reserved|active"):
        BudgetAccountant.from_snapshot(
            RunBudget.preset("low"),
            accountant.snapshot(),
        )


def test_from_snapshot_revalidates_constructed_snapshot_and_full_reconciliation() -> None:
    valid = BudgetAccountant(RunBudget.preset("low")).snapshot()
    corrupt = cast(
        "BudgetSnapshot", BaseModel.model_copy(valid, update={"used_tokens": 1})
    )

    with pytest.raises(ValueError, match="snapshot|reconcile|usage"):
        BudgetAccountant.from_snapshot(RunBudget.preset("low"), corrupt)


@pytest.mark.parametrize("bad_used_tokens", [True, "0"])
def test_from_snapshot_strictly_rejects_coercible_constructed_values(
    bad_used_tokens: object,
) -> None:
    valid = BudgetAccountant(RunBudget.preset("low")).snapshot()
    corrupt = cast(
        "BudgetSnapshot",
        BaseModel.model_copy(valid, update={"used_tokens": bad_used_tokens}),
    )

    with pytest.raises(ValueError, match="snapshot|invalid"):
        BudgetAccountant.from_snapshot(RunBudget.preset("low"), corrupt)


def test_from_snapshot_rejects_history_below_original_configured_seed() -> None:
    seed = usage(tokens=5, pages=1, cost="0.01")
    budget = RunBudget.preset("low").model_copy(
        update={
            "used_by_node": {
                **RunBudget.preset("low").used_by_node,
                "Planner": seed,
            }
        }
    )
    rewound = BudgetAccountant(RunBudget.preset("low")).snapshot()

    with pytest.raises(ValueError, match="seed|history|rewind"):
        BudgetAccountant.from_snapshot(budget, rewound)


def test_from_snapshot_preserves_unknown_cost_mode_and_observation_exactly() -> None:
    budget = RunBudget.preset("low").model_copy(
        update={
            "max_cost_usd": None,
            "used_by_node": {
                **RunBudget.preset("low").used_by_node,
                "Tool": usage(tokens=3, cost="0.25"),
            },
        }
    )
    original = BudgetAccountant(budget, run_scope="same")
    observed = original.reserve(
        estimate(tokens=2, cost=None),
        node="Tool",
        idempotency_key="observed",
    )
    expected = original.settle(
        observed,
        actual=usage(tokens=2, cost=None),
        charge=False,
    )

    restored = BudgetAccountant.from_snapshot(budget, expected, run_scope="same")

    assert restored.snapshot() == expected
    assert restored.snapshot().used_cost_usd is None
    assert restored.snapshot().used_by_node["Tool"].cost_usd == Decimal("0.25")
    assert restored.snapshot().last_observed_usage.cost_usd is None


@pytest.mark.parametrize("rewound_cost", ["0.10", None])
def test_from_snapshot_rejects_cost_disabled_seed_cost_rewind(
    rewound_cost: str | None,
) -> None:
    preset = RunBudget.preset("low")
    budget = preset.model_copy(
        update={
            "max_cost_usd": None,
            "used_by_node": {
                **preset.used_by_node,
                "Tool": usage(tokens=3, cost="0.25"),
            },
        }
    )
    rewound_budget = budget.model_copy(
        update={
            "used_by_node": {
                **budget.used_by_node,
                "Tool": usage(tokens=3, cost=rewound_cost),
            }
        }
    )
    rewound = BudgetAccountant(rewound_budget).snapshot()

    with pytest.raises(ValueError, match="seed|history|rewind"):
        BudgetAccountant.from_snapshot(budget, rewound)


def test_from_snapshot_does_not_restore_reservation_objects_or_indexes() -> None:
    budget = RunBudget.preset("low")
    original = BudgetAccountant(budget, run_scope="same")
    old = original.reserve(
        estimate(tokens=1, cost="0.01"),
        node="Tool",
        idempotency_key="same-key",
    )
    snapshot = original.release(old)
    restored = BudgetAccountant.from_snapshot(budget, snapshot, run_scope="same")

    with pytest.raises(ValueError, match="accountant|unknown"):
        restored.settle(old, actual=usage(tokens=1, cost="0.01"))
    fresh = restored.reserve(
        estimate(tokens=1, cost="0.01"),
        node="Tool",
        idempotency_key="same-key",
    )
    assert fresh is not old
    assert fresh.reservation_id == old.reservation_id
