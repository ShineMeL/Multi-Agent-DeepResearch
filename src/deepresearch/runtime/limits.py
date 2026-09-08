"""Public admission limits and process-local provider concurrency gates."""

from __future__ import annotations

import asyncio
import math
import time
from collections import defaultdict
from collections.abc import AsyncGenerator, Callable
from contextlib import asynccontextmanager
from datetime import UTC, datetime, timedelta
from decimal import Decimal

from deepresearch.domain import RunBudget
from deepresearch.runtime.admission import Admission
from deepresearch.storage.protocols import DailyCostLimitExceeded, RunStore


class RateLimitExceeded(RuntimeError):
    code = "RATE_LIMITED"

    def __init__(self, retry_after: int = 1) -> None:
        super().__init__("service capacity or usage limit reached")
        self.retry_after = max(1, retry_after)


class CapacityGate:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self._active = 0
        self._lock = asyncio.Lock()

    async def try_acquire(self) -> None:
        async with self._lock:
            if self._active >= self.capacity:
                raise RateLimitExceeded()
            self._active += 1

    async def release(self) -> None:
        async with self._lock:
            if self._active == 0:
                raise ValueError("capacity lease underflow")
            self._active -= 1


class TokenBucketMap:
    def __init__(
        self,
        *,
        capacity: int,
        refill_seconds: int,
        clock: Callable[[], float],
        lock: asyncio.Lock | None = None,
    ) -> None:
        if capacity <= 0 or refill_seconds <= 0:
            raise ValueError("bucket capacity and refill interval must be positive")
        self.capacity = capacity
        self.refill_seconds = refill_seconds
        self.clock = clock
        self._buckets: dict[str, tuple[float, float]] = {}
        self._lock = lock if lock is not None else asyncio.Lock()

    def _refill(self, key: str) -> tuple[float, float]:
        now = self.clock()
        tokens, updated = self._buckets.get(key, (float(self.capacity), now))
        return min(self.capacity, tokens + max(0, now - updated) / self.refill_seconds), now

    async def consume(self, key: str) -> None:
        async with self._lock:
            tokens, now = self._refill(key)
            self._buckets[key] = tokens, now
            if tokens < 1:
                raise RateLimitExceeded(math.ceil((1 - tokens) * self.refill_seconds))
            self._buckets[key] = tokens - 1, now

    async def refund(self, key: str) -> None:
        async with self._lock:
            tokens, now = self._refill(key)
            self._buckets[key] = min(self.capacity, tokens + 1), now


class LimitManager:
    def __init__(
        self,
        store: RunStore,
        daily_limit: Decimal,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not daily_limit.is_finite() or daily_limit < 0:
            raise ValueError("daily limit must be finite and non-negative")
        self.store = store
        self.daily_limit = daily_limit
        self.clock = clock
        self.runs = CapacityGate(2)
        self.search_global = asyncio.Semaphore(4)
        self.host_gates: defaultdict[str, asyncio.Semaphore] = defaultdict(
            lambda: asyncio.Semaphore(2)
        )
        # Both maps share one lock, including refunds of a partially admitted request.
        bucket_lock = asyncio.Lock()
        self.ip_buckets = TokenBucketMap(
            capacity=4, refill_seconds=60, clock=clock, lock=bucket_lock
        )
        self.session_buckets = TokenBucketMap(
            capacity=2, refill_seconds=120, clock=clock, lock=bucket_lock
        )
        self._admission_lock = asyncio.Lock()
        self._local_run_leases: set[str] = set()
        self._local_admissions: dict[str, Admission] = {}

    async def admit(
        self,
        *,
        run_id: str,
        client_ip: str,
        session_id: str,
        access_profile: str,
        requested_cost_usd: Decimal,
    ) -> Admission:
        if access_profile != "public_live":
            return Admission(None, None)
        medium_cost = RunBudget.preset("medium").max_cost_usd
        if (
            not requested_cost_usd.is_finite()
            or requested_cost_usd < 0
            or medium_cost is None
            or requested_cost_usd > medium_cost
        ):
            raise ValueError("public cost must be finite and within the medium budget")
        async with self._admission_lock:
            if run_id in self._local_admissions:
                return self._local_admissions[run_id]
            ip_consumed = session_consumed = acquired = False
            try:
                await self.ip_buckets.consume(client_ip)
                ip_consumed = True
                await self.session_buckets.consume(session_id)
                session_consumed = True
                await self.runs.try_acquire()
                acquired = True
                now = datetime.now(UTC)
                try:
                    admitted = await self.store.reserve_daily_cost(
                        now.date(),
                        run_id,
                        requested_cost_usd,
                        self.daily_limit,
                    )
                except DailyCostLimitExceeded:
                    midnight = datetime.combine(
                        now.date() + timedelta(days=1), datetime.min.time(), UTC
                    )
                    raise RateLimitExceeded(math.ceil((midnight - now).total_seconds())) from None
                if admitted.reservation_id is None:
                    raise ValueError("public admission requires a durable reservation")
                self._local_run_leases.add(admitted.reservation_id)
                self._local_admissions[run_id] = admitted
                return admitted
            except BaseException:
                if acquired:
                    await self.runs.release()
                if session_consumed:
                    await self.session_buckets.refund(session_id)
                if ip_consumed:
                    await self.ip_buckets.refund(client_ip)
                raise

    async def _release_local(self, reservation_id: str) -> None:
        if reservation_id in self._local_run_leases:
            await self.runs.release()
            self._local_run_leases.remove(reservation_id)
            for run_id, admitted in tuple(self._local_admissions.items()):
                if admitted.reservation_id == reservation_id:
                    del self._local_admissions[run_id]

    async def settle(self, reservation_id: str | None, actual_cost_usd: Decimal) -> None:
        if reservation_id is None:
            return
        async with self._admission_lock:
            # On persistence failure keep the capacity lease and durable recovery link.
            await self.store.settle_daily_cost(reservation_id, actual_cost_usd)
            await self._release_local(reservation_id)

    async def release(self, reservation_id: str | None) -> None:
        if reservation_id is None:
            return
        async with self._admission_lock:
            await self.store.release_daily_cost(reservation_id)
            await self._release_local(reservation_id)

    @asynccontextmanager
    async def search_slot(self) -> AsyncGenerator[None]:
        async with self.search_global:
            yield

    @asynccontextmanager
    async def fetch_slot(self, hostname: str) -> AsyncGenerator[None]:
        async with self.host_gates[hostname.rstrip(".").casefold()]:
            yield
