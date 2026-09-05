from __future__ import annotations

from decimal import Decimal

from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.runtime import BudgetAccountant, ResourceEstimate


def test_query_planner_consumes_core_budget_accountant_once() -> None:
    # Cost is deliberately disabled so the zero-cost usage object is accepted
    # by the Core accountant; the assertion focuses on idempotent settlement.
    accountant = BudgetAccountant(
        RunBudget.preset("medium").model_copy(update={"max_cost_usd": None})
    )
    reservation = accountant.reserve(
        ResourceEstimate(
            search_calls=1,
            pages=0,
            tokens=100,
            wall_seconds=1,
            cost_usd=Decimal(0),
        ),
        node="Tool",
        idempotency_key="search:q-1",
    )

    first = accountant.settle(reservation, actual=ResourceUsage.zero())
    second = accountant.settle(reservation, actual=ResourceUsage.zero())

    assert first == second
