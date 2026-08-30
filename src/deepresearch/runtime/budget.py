from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from threading import RLock
from typing import (
    Annotated,
    Any,
    Literal,
    Never,
    Self,
    TypeAlias,
    TypeVar,
    override,
)

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator

from deepresearch.domain import ResourceUsage, RunBudget

BudgetNode: TypeAlias = Literal[  # noqa: UP040 - exact frozen public contract
    "Planner", "Ranker", "Writer", "Judge", "Tool"
]
BudgetDimension: TypeAlias = Literal[  # noqa: UP040 - exact frozen public contract
    "search_calls", "pages", "tokens", "wall_seconds", "cost_usd", "retries"
]

_NODES: tuple[BudgetNode, ...] = ("Planner", "Ranker", "Writer", "Judge", "Tool")
_Key = TypeVar("_Key")
_Value = TypeVar("_Value")


class _FrozenDict(dict[_Key, _Value]):
    @staticmethod
    def _raise_immutable() -> Never:
        raise TypeError("runtime mappings are immutable")

    def __setitem__(self, key: _Key, value: _Value) -> Never:
        self._raise_immutable()

    def __delitem__(self, key: _Key) -> Never:
        self._raise_immutable()

    def __ior__(self, value: object) -> Never:
        self._raise_immutable()

    def __copy__(self) -> Self:
        return self

    def __deepcopy__(self, memo: dict[int, object]) -> Self:
        memo[id(self)] = self
        return self

    def clear(self) -> Never:
        self._raise_immutable()

    def pop(self, key: _Key, default: object = None) -> Never:
        self._raise_immutable()

    def popitem(self) -> Never:
        self._raise_immutable()

    def setdefault(self, key: _Key, default: _Value | None = None) -> Never:
        self._raise_immutable()

    def update(self, *args: object, **kwargs: object) -> Never:
        self._raise_immutable()


def _freeze_mapping[Key, Value](value: dict[Key, Value]) -> dict[Key, Value]:
    return _FrozenDict(value)


class _RuntimeModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @override
    def model_copy(
        self,
        *,
        update: Mapping[str, Any] | None = None,
        deep: bool = False,
    ) -> Self:
        if update is None:
            return super().model_copy(deep=deep)
        values = self.model_dump(round_trip=True)
        values.update(update)
        return type(self).model_validate(values)


class ResourceEstimate(_RuntimeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    search_calls: Annotated[int, Field(ge=0)] = 0
    pages: Annotated[int, Field(ge=0)] = 0
    tokens: Annotated[int, Field(ge=0)] = 0
    wall_seconds: Annotated[float, Field(ge=0.0)] = 0.0
    cost_usd: Annotated[Decimal | None, Field(ge=0)] = None
    retries: Annotated[int, Field(ge=0)] = 0


class BudgetReservation(_RuntimeModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reservation_id: str
    idempotency_key: str
    node: BudgetNode
    estimate: ResourceEstimate


class BudgetSnapshot(_RuntimeModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    used_search_calls: Annotated[int, Field(ge=0)]
    used_pages: Annotated[int, Field(ge=0)]
    used_tokens: Annotated[int, Field(ge=0)]
    used_wall_seconds: Annotated[float, Field(ge=0.0)]
    used_cost_usd: Annotated[Decimal | None, Field(ge=0)]
    used_retries: Annotated[int, Field(ge=0)]
    reserved_search_calls: Annotated[int, Field(ge=0)]
    reserved_pages: Annotated[int, Field(ge=0)]
    reserved_tokens: Annotated[int, Field(ge=0)]
    reserved_wall_seconds: Annotated[float, Field(ge=0.0)]
    reserved_cost_usd: Annotated[Decimal | None, Field(ge=0)]
    reserved_retries: Annotated[int, Field(ge=0)]
    exhausted: frozenset[BudgetDimension]
    last_observed_usage: ResourceUsage
    used_by_node: dict[BudgetNode, ResourceUsage]

    @field_validator("used_by_node")
    @classmethod
    def freeze_used_by_node(
        cls, value: dict[BudgetNode, ResourceUsage]
    ) -> dict[BudgetNode, ResourceUsage]:
        if set(value) != set(_NODES):
            raise ValueError("used_by_node must contain the five canonical budget nodes")
        return _freeze_mapping(value)

    @field_serializer("used_by_node", when_used="json")
    def serialize_used_by_node(
        self, value: dict[BudgetNode, ResourceUsage]
    ) -> dict[BudgetNode, ResourceUsage]:
        return {node: value[node] for node in _NODES}


class BudgetExceeded(RuntimeError):
    code: Literal["BUDGET_EXCEEDED"] = "BUDGET_EXCEEDED"

    def __init__(
        self,
        dimensions: frozenset[BudgetDimension],
        snapshot: BudgetSnapshot,
    ) -> None:
        super().__init__(",".join(sorted(dimensions)))
        self.dimensions = dimensions
        self.snapshot = snapshot


@dataclass
class _ReservationState:
    reservation: BudgetReservation
    status: Literal["active", "settled", "released"] = "active"
    terminal_snapshot: BudgetSnapshot | None = None


class BudgetAccountant:
    def __init__(self, budget: RunBudget, *, run_scope: str = "local") -> None:
        if not run_scope:
            raise ValueError("run_scope must not be empty")
        self._budget = budget
        self._run_scope = run_scope
        self._lock = RLock()
        self._states: dict[str, _ReservationState] = {}
        self._idempotency_index: dict[str, str] = {}
        self._cost_enabled = budget.max_cost_usd is not None
        zero = ResourceUsage.zero(cost_known=self._cost_enabled)
        used_by_node: dict[BudgetNode, ResourceUsage] = {}
        for node in _NODES:
            seeded = self._validate_usage(
                budget.used_by_node.get(node, zero),
                require_known_cost=False,
                label=f"seeded {node} usage",
            )
            if self._cost_enabled and seeded.cost_usd is None:
                if not _usage_is_zero(seeded):
                    raise ValueError(f"seeded {node} usage must have known cost")
                seeded = seeded.model_copy(update={"cost_usd": Decimal(0)})
            used_by_node[node] = seeded
        self._used_by_node = used_by_node
        self._last_observed_usage = zero

    def snapshot(self) -> BudgetSnapshot:
        with self._lock:
            return self._snapshot_locked()

    def reserve(
        self,
        estimate: ResourceEstimate,
        *,
        node: str,
        idempotency_key: str,
    ) -> BudgetReservation:
        if node not in _NODES:
            raise ValueError("node must be one of the five canonical budget buckets")
        if not idempotency_key:
            raise ValueError("idempotency_key must not be empty")
        estimate = ResourceEstimate.model_validate(estimate.model_dump())
        if self._cost_enabled and estimate.cost_usd is None:
            raise ValueError("chargeable reservation must have known cost")
        budget_node = node
        with self._lock:
            existing_id = self._idempotency_index.get(idempotency_key)
            if existing_id is not None:
                existing = self._states[existing_id].reservation
                if existing.node != budget_node or existing.estimate != estimate:
                    raise ValueError("idempotency_key was reused with different inputs")
                return existing

            before = self._snapshot_locked()
            offending = self._offending_dimensions(before, estimate)
            if offending:
                raise BudgetExceeded(frozenset(offending), before)

            reservation_id = sha256(
                f"{self._run_scope}:{idempotency_key}".encode()
            ).hexdigest()
            if reservation_id in self._states:
                raise ValueError("reservation ID collision")
            reservation = BudgetReservation(
                reservation_id=reservation_id,
                idempotency_key=idempotency_key,
                node=budget_node,
                estimate=estimate,
            )
            self._states[reservation_id] = _ReservationState(reservation=reservation)
            self._idempotency_index[idempotency_key] = reservation_id
            return reservation

    def settle(
        self,
        reservation: BudgetReservation,
        *,
        actual: ResourceUsage,
        charge: bool = True,
    ) -> BudgetSnapshot:
        with self._lock:
            state = self._owned_state(reservation)
            if state.status == "settled":
                if state.terminal_snapshot is None:
                    raise RuntimeError("settled reservation has no terminal snapshot")
                return state.terminal_snapshot
            if state.status == "released":
                raise ValueError("cannot settle a released reservation")

            actual = self._validate_usage(
                actual,
                require_known_cost=self._cost_enabled and charge,
                label="actual usage",
            )
            tentative_used_by_node = dict(self._used_by_node)
            if charge:
                current = tentative_used_by_node[reservation.node]
                tentative_used_by_node[reservation.node] = self._add_usage(current, actual)
            terminal_snapshot = self._snapshot_locked(
                used_by_node=tentative_used_by_node,
                last_observed_usage=actual,
                exclude_reservation_id=reservation.reservation_id,
            )

            self._used_by_node = tentative_used_by_node
            self._last_observed_usage = actual
            state.status = "settled"
            state.terminal_snapshot = terminal_snapshot
            return terminal_snapshot

    def release(self, reservation: BudgetReservation) -> BudgetSnapshot:
        with self._lock:
            state = self._owned_state(reservation)
            if state.status == "released":
                if state.terminal_snapshot is None:
                    raise RuntimeError("released reservation has no terminal snapshot")
                return state.terminal_snapshot
            if state.status == "settled":
                raise ValueError("cannot release a settled reservation")

            terminal_snapshot = self._snapshot_locked(
                exclude_reservation_id=reservation.reservation_id
            )
            state.status = "released"
            state.terminal_snapshot = terminal_snapshot
            return terminal_snapshot

    def _owned_state(self, reservation: BudgetReservation) -> _ReservationState:
        state = self._states.get(reservation.reservation_id)
        if state is None or state.reservation is not reservation:
            raise ValueError("reservation belongs to another accountant or is unknown")
        return state

    def _snapshot_locked(
        self,
        *,
        used_by_node: dict[BudgetNode, ResourceUsage] | None = None,
        last_observed_usage: ResourceUsage | None = None,
        exclude_reservation_id: str | None = None,
    ) -> BudgetSnapshot:
        usage_by_node = self._used_by_node if used_by_node is None else used_by_node
        observed = (
            self._last_observed_usage
            if last_observed_usage is None
            else last_observed_usage
        )
        used = self._used_totals(usage_by_node)
        reserved = self._reserved_totals(exclude_reservation_id=exclude_reservation_id)
        exhausted = self._exhausted_dimensions(used, reserved)
        return BudgetSnapshot(
            used_search_calls=used.search_calls,
            used_pages=used.pages,
            used_tokens=used.total_tokens,
            used_wall_seconds=used.wall_seconds,
            used_cost_usd=used.cost_usd if self._cost_enabled else None,
            used_retries=used.retries,
            reserved_search_calls=reserved.search_calls,
            reserved_pages=reserved.pages,
            reserved_tokens=reserved.tokens,
            reserved_wall_seconds=reserved.wall_seconds,
            reserved_cost_usd=reserved.cost_usd if self._cost_enabled else None,
            reserved_retries=reserved.retries,
            exhausted=frozenset(exhausted),
            last_observed_usage=observed,
            used_by_node=dict(usage_by_node),
        )

    def _used_totals(
        self, used_by_node: dict[BudgetNode, ResourceUsage]
    ) -> ResourceUsage:
        total = ResourceUsage.zero(cost_known=self._cost_enabled)
        for node in _NODES:
            total = self._add_usage(total, used_by_node[node])
        return total

    def _reserved_totals(
        self, *, exclude_reservation_id: str | None = None
    ) -> ResourceEstimate:
        active = [
            state.reservation.estimate
            for reservation_id, state in self._states.items()
            if state.status == "active" and reservation_id != exclude_reservation_id
        ]
        cost: Decimal | None
        if not self._cost_enabled:
            cost = None
        else:
            cost = sum((item.cost_usd or Decimal(0) for item in active), Decimal(0))
        return ResourceEstimate(
            search_calls=sum(item.search_calls for item in active),
            pages=sum(item.pages for item in active),
            tokens=sum(item.tokens for item in active),
            wall_seconds=sum(item.wall_seconds for item in active),
            cost_usd=cost,
            retries=sum(item.retries for item in active),
        )

    def _add_usage(self, left: ResourceUsage, right: ResourceUsage) -> ResourceUsage:
        cost: Decimal | None
        if not self._cost_enabled:
            cost = None
        else:
            cost = (left.cost_usd or Decimal(0)) + (right.cost_usd or Decimal(0))
        return ResourceUsage(
            input_tokens=left.input_tokens + right.input_tokens,
            output_tokens=left.output_tokens + right.output_tokens,
            reasoning_tokens=left.reasoning_tokens + right.reasoning_tokens,
            cached_tokens=left.cached_tokens + right.cached_tokens,
            total_tokens=left.total_tokens + right.total_tokens,
            search_calls=left.search_calls + right.search_calls,
            pages=left.pages + right.pages,
            retries=left.retries + right.retries,
            wall_seconds=left.wall_seconds + right.wall_seconds,
            cost_usd=cost,
        )

    @staticmethod
    def _validate_usage(
        usage: ResourceUsage,
        *,
        require_known_cost: bool,
        label: str,
    ) -> ResourceUsage:
        if not isfinite(usage.wall_seconds) or usage.wall_seconds < 0:
            raise ValueError(f"{label} wall_seconds must be finite and non-negative")
        if usage.cost_usd is None:
            if require_known_cost:
                raise ValueError(f"{label} must have known cost")
        elif not usage.cost_usd.is_finite() or usage.cost_usd < 0:
            raise ValueError(f"{label} cost_usd must be finite and non-negative")
        try:
            return ResourceUsage.model_validate(usage.model_dump())
        except ValueError as error:
            raise ValueError(f"{label} dimensions must be finite and non-negative") from error

    def _offending_dimensions(
        self, snapshot: BudgetSnapshot, estimate: ResourceEstimate
    ) -> set[BudgetDimension]:
        offending: set[BudgetDimension] = set()
        checks: tuple[tuple[BudgetDimension, int | float, int | float], ...] = (
            (
                "search_calls",
                snapshot.used_search_calls + snapshot.reserved_search_calls + estimate.search_calls,
                self._budget.max_search_calls,
            ),
            (
                "pages",
                snapshot.used_pages + snapshot.reserved_pages + estimate.pages,
                self._budget.max_pages,
            ),
            (
                "tokens",
                snapshot.used_tokens + snapshot.reserved_tokens + estimate.tokens,
                self._budget.max_total_tokens,
            ),
            (
                "wall_seconds",
                snapshot.used_wall_seconds
                + snapshot.reserved_wall_seconds
                + estimate.wall_seconds,
                self._budget.max_wall_time_seconds,
            ),
            (
                "retries",
                snapshot.used_retries + snapshot.reserved_retries + estimate.retries,
                self._budget.max_retries,
            ),
        )
        for dimension, value, limit in checks:
            if value > limit:
                offending.add(dimension)
        if (
            self._budget.max_cost_usd is not None
            and snapshot.used_cost_usd is not None
            and snapshot.reserved_cost_usd is not None
            and estimate.cost_usd is not None
            and snapshot.used_cost_usd
            + snapshot.reserved_cost_usd
            + estimate.cost_usd
            > self._budget.max_cost_usd
        ):
            offending.add("cost_usd")
        return offending

    def _exhausted_dimensions(
        self, used: ResourceUsage, reserved: ResourceEstimate
    ) -> set[BudgetDimension]:
        exhausted: set[BudgetDimension] = set()
        checks: tuple[tuple[BudgetDimension, int | float, int | float], ...] = (
            (
                "search_calls",
                used.search_calls + reserved.search_calls,
                self._budget.max_search_calls,
            ),
            ("pages", used.pages + reserved.pages, self._budget.max_pages),
            ("tokens", used.total_tokens + reserved.tokens, self._budget.max_total_tokens),
            (
                "wall_seconds",
                used.wall_seconds + reserved.wall_seconds,
                self._budget.max_wall_time_seconds,
            ),
            ("retries", used.retries + reserved.retries, self._budget.max_retries),
        )
        for dimension, value, limit in checks:
            if value >= limit:
                exhausted.add(dimension)
        if (
            self._budget.max_cost_usd is not None
            and used.cost_usd is not None
            and reserved.cost_usd is not None
            and used.cost_usd + reserved.cost_usd >= self._budget.max_cost_usd
        ):
            exhausted.add("cost_usd")
        return exhausted


def _usage_is_zero(usage: ResourceUsage) -> bool:
    return (
        usage.total_tokens == 0
        and usage.search_calls == 0
        and usage.pages == 0
        and usage.retries == 0
        and usage.wall_seconds == 0
    )


__all__ = [
    "BudgetAccountant",
    "BudgetDimension",
    "BudgetExceeded",
    "BudgetNode",
    "BudgetReservation",
    "BudgetSnapshot",
    "ResourceEstimate",
]
