from __future__ import annotations

import asyncio
import ipaddress
import time
from collections import Counter
from contextlib import asynccontextmanager
from datetime import date
from decimal import Decimal
from urllib.parse import urlsplit

import httpx
import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.domain import ResourceUsage
from deepresearch.providers import ProviderUsageResult
from deepresearch.providers.httpx_fetcher import HttpxFetcher
from deepresearch.runtime import CancellationToken
from deepresearch.runtime.limits import (
    CapacityGate,
    LimitManager,
    RateLimitExceeded,
    TokenBucketMap,
)
from deepresearch.runtime.runner_factory import (
    FileProviderRouteCatalog,
    LangGraphServiceRunnerFactory,
)
from tests.unit.runtime.test_runner_factory_execution import composition, freeze, pricing
from tests.unit.storage.test_sqlite_store import store as store  # noqa: PLC0414 - pytest fixture


class FakeClock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


async def admit(limits, run="r", ip=None, session=None, cost="0.01", profile="public_live"):
    return await limits.admit(
        run_id=run,
        client_ip=ip or run,
        session_id=session or run,
        access_profile=profile,
        requested_cost_usd=Decimal(cost),
    )


async def test_capacity_is_nonblocking_atomic_and_rejects_underflow():
    gate = CapacityGate(2)
    results = await asyncio.wait_for(
        asyncio.gather(
            *(gate.try_acquire() for _ in range(8)),
            return_exceptions=True,
        ),
        1,
    )
    assert sum(isinstance(value, RateLimitExceeded) for value in results) == 6
    await gate.release()
    await gate.release()
    with pytest.raises(ValueError):
        await gate.release()
    await gate.try_acquire()


@pytest.mark.parametrize("capacity,interval", [(4, 60), (2, 120)])
async def test_bucket_refills_one_token_at_exact_boundary(capacity, interval):
    clock = FakeClock()
    bucket = TokenBucketMap(capacity=capacity, refill_seconds=interval, clock=clock)
    await asyncio.gather(*(bucket.consume("owner") for _ in range(capacity)))
    with pytest.raises(RateLimitExceeded) as error:
        await bucket.consume("owner")
    assert error.value.retry_after == interval
    clock.advance(interval - 0.1)
    with pytest.raises(RateLimitExceeded) as error:
        await bucket.consume("owner")
    assert error.value.retry_after == 1
    clock.advance(0.1)
    await bucket.consume("owner")
    with pytest.raises(RateLimitExceeded):
        await bucket.consume("owner")
    await bucket.consume("other")


async def test_third_public_live_run_is_rejected(store):
    limits = LimitManager(store, daily_limit=Decimal(10))
    await admit(limits, "r1")
    await admit(limits, "r2")
    with pytest.raises(RateLimitExceeded):
        await admit(limits, "r3")


async def test_daily_cost_reservation_survives_new_manager(store):
    await admit(LimitManager(store, daily_limit=Decimal("0.50")), "r1", cost="0.50")
    with pytest.raises(RateLimitExceeded):
        await admit(LimitManager(store, daily_limit=Decimal("0.50")), "r2")


async def test_failed_session_consumption_refunds_ip_tokens(store):
    clock = FakeClock()
    limits = LimitManager(store, daily_limit=Decimal(10), clock=clock)
    for run in ("a", "b"):
        result = await admit(limits, run, ip="ip", session="s")
        await limits.release(result.reservation_id)
    for _ in range(5):
        with pytest.raises(RateLimitExceeded) as error:
            await admit(limits, "c", ip="ip", session="s")
        assert error.value.retry_after == 120
    for session in ("other-a", "other-b"):
        result = await admit(limits, session, ip="ip", session=session)
        await limits.release(result.reservation_id)
    with pytest.raises(RateLimitExceeded) as error:
        await admit(limits, "ip-full", ip="ip")
    assert error.value.retry_after == 60
    clock.advance(120)
    await admit(limits, "after", ip="ip", session="s")


async def test_failed_capacity_and_daily_admission_refund_all_local_resources(store):
    limits = LimitManager(store, daily_limit=Decimal("0.02"), clock=FakeClock())
    first, second = await admit(limits, "a"), await admit(limits, "b")
    for _ in range(6):
        with pytest.raises(RateLimitExceeded):
            await admit(limits, "c", ip="new", session="new")
    await limits.settle(first.reservation_id, Decimal("0.01"))
    for _ in range(6):
        with pytest.raises(RateLimitExceeded):
            await admit(limits, "c", ip="new", session="new")
    await limits.release(second.reservation_id)
    await admit(limits, "c", ip="new", session="new")


async def test_duplicate_admission_uses_one_lease_and_one_token(store):
    limits = LimitManager(store, daily_limit=Decimal(1), clock=FakeClock())
    results = await asyncio.gather(*(admit(limits, "same") for _ in range(6)))
    assert all(result == results[0] for result in results)
    await admit(limits, "other", ip="same", session="same")
    await limits.release(results[0].reservation_id)
    await limits.release(results[0].reservation_id)
    await admit(limits, "third")
    with pytest.raises(RateLimitExceeded):
        await admit(limits, "fourth")


async def test_store_concurrent_same_run_reserves_once_and_attempts_span_days(store):
    results = await asyncio.gather(
        *(
            store.reserve_daily_cost(date(2026, 8, 29), "same", Decimal("0.5"), Decimal("0.5"))
            for _ in range(4)
        )
    )
    assert len(set(results)) == 1
    first = results[0]
    await store.settle_daily_cost(first.reservation_id, Decimal("0.3"))
    second = await store.reserve_daily_cost(
        date(2026, 8, 30),
        "same",
        Decimal("0.5"),
        Decimal("0.5"),
    )
    assert second.attempt_no == 2
    assert second.reservation_id != first.reservation_id


async def test_same_day_resume_keeps_actual_cost_and_settle_releases_once(store):
    limits = LimitManager(store, daily_limit=Decimal("0.50"))
    first = await admit(limits, cost="0.50")
    await limits.settle(first.reservation_id, Decimal("0.30"))
    await limits.settle(first.reservation_id, Decimal("0.49"))
    await limits.release(first.reservation_id)
    second = await admit(limits, cost="0.20")
    assert second.attempt_no == 2
    with pytest.raises(RateLimitExceeded):
        await admit(limits, "over")
    await limits.release(second.reservation_id)
    await admit(limits, "new", cost="0.20")
    assert await store.ledger_state(first.reservation_id) == "settled"


async def test_settlement_failure_retains_local_capacity_and_reservation(store, monkeypatch):
    limits = LimitManager(store, daily_limit=Decimal(1))
    first = await admit(limits, "a")
    await admit(limits, "b")

    async def fail(*args):
        raise OSError("database unavailable")

    with monkeypatch.context() as patch:
        patch.setattr(store, "settle_daily_cost", fail)
        with pytest.raises(OSError):
            await limits.settle(first.reservation_id, Decimal("0.01"))
    assert await store.ledger_state(first.reservation_id) == "reserved"
    with pytest.raises(RateLimitExceeded):
        await admit(limits, "c")
    await limits.settle(first.reservation_id, Decimal("0.01"))
    await admit(limits, "c")


@pytest.mark.parametrize("profile", ["local", "showcase"])
async def test_nonpublic_admission_has_no_cost_reservation(store, profile):
    limits = LimitManager(store, daily_limit=Decimal(0))
    for index in range(6):
        result = await admit(limits, str(index), profile=profile, cost="0.50")
        assert result.reservation_id is result.attempt_no is None
    await limits.release(None)
    await limits.settle(None, Decimal(0))


@pytest.mark.parametrize("cost", ["NaN", "Infinity", "-0.01", "0.51"])
async def test_invalid_or_above_medium_cost_is_rejected_without_leaking_tokens(store, cost):
    limits = LimitManager(store, daily_limit=Decimal(10), clock=FakeClock())
    for _ in range(5):
        with pytest.raises(ValueError):
            await admit(limits, cost=cost)
    await admit(limits)


async def test_cancelled_reservation_attempt_refunds_local_resources(store, monkeypatch):
    limits = LimitManager(store, daily_limit=Decimal(1), clock=FakeClock())

    async def cancelled(*args):
        raise asyncio.CancelledError

    with monkeypatch.context() as patch:
        patch.setattr(store, "reserve_daily_cost", cancelled)
        for _ in range(5):
            with pytest.raises(asyncio.CancelledError):
                await admit(limits)
    await admit(limits)
    await admit(limits, "second", ip="r", session="r")


class SearchSpy:
    provider_id = "offline-search"

    def __init__(self):
        self.active = 0
        self.max_active = 0

    async def search(self, query, limit, filters, **kwargs):
        return (await self.search_with_usage(query, limit, filters, **kwargs)).value

    async def search_with_usage(self, query, limit, filters, **kwargs):
        self.active += 1
        self.max_active = max(self.active, self.max_active)
        try:
            await asyncio.sleep(0.01)
            return ProviderUsageResult(
                value=[],
                usage=ResourceUsage.zero().model_copy(update={"search_calls": 1}),
            )
        finally:
            self.active -= 1


class FetchTransportSpy:
    def __init__(self):
        self.active = Counter()
        self.max_active = Counter()

    @asynccontextmanager
    async def stream(self, url, *, pinned_ip, **kwargs):
        host = urlsplit(url).hostname
        self.active[host] += 1
        self.max_active[host] = max(self.active[host], self.max_active[host])
        try:
            await asyncio.sleep(0.01)
            redirect = url.endswith("/start")
            yield httpx.Response(
                302 if redirect else 200,
                headers={"location": "https://b.example/final"}
                if redirect
                else {"content-type": "text/html"},
                content=b"<html>public page</html>",
                request=httpx.Request("GET", url),
                extensions={"peer_ip": pinned_ip},
            )
        finally:
            self.active[host] -= 1


async def public_resolver(hostname, port):
    return (ipaddress.ip_address("8.8.8.8"),)


async def test_runner_factory_wires_global_search_and_per_host_fetch_gates(
    store, tmp_path, monkeypatch
):
    from deepresearch.runtime import runner_factory

    limits = LimitManager(store, daily_limit=Decimal(10))
    builder, conf, routes, snapshots, _, _ = composition(tmp_path)
    builder.search_slot = limits.search_slot
    builder.host_slot = limits.fetch_slot
    search = SearchSpy()
    transport = FetchTransportSpy()
    rows = [item.model_dump(mode="json") for item in routes.routes]
    for row in rows:
        if row["operation"] == "fetch":
            row["provider_id"] = "httpx-fetcher"
    routes = freeze(conf, rows)
    snapshots = tuple(item for item in snapshots if item.endpoint_type != "fetch") + (
        pricing("httpx-fetcher", "fetch", "fetch"),
    )
    builder.provider_constructors["offline-search"] = lambda route, secret, slot: search
    # Replace only network boundaries; the production constructor still creates Core's fetcher.
    from deepresearch.runtime.runner_factory import default_provider_constructors

    monkeypatch.setattr(
        "deepresearch.providers.httpx_transport.PinnedPeerTransport", lambda: transport
    )
    original_fetcher = HttpxFetcher

    def fetcher_with_resolver(**kwargs):
        return original_fetcher(**kwargs, resolver=public_resolver)

    monkeypatch.setattr("deepresearch.providers.httpx_fetcher.HttpxFetcher", fetcher_with_resolver)
    builder.provider_constructors["httpx-fetcher"] = default_provider_constructors()[
        "httpx-fetcher"
    ]
    # Capture the actual Core handlers while preserving graph compilation and factory preflight.
    handlers = []
    original_handlers = runner_factory.BaselineNodeHandlers

    def capture(**kwargs):
        value = original_handlers(**kwargs)
        handlers.append(value)
        return value

    monkeypatch.setattr(runner_factory, "BaselineNodeHandlers", capture)
    factory = LangGraphServiceRunnerFactory(
        builder, FileProviderRouteCatalog({routes.profile_id: routes})
    )
    for _ in range(2):
        factory.create(
            config=conf,
            provider_routes=routes,
            pricing_snapshots=snapshots,
            checkpointer=InMemorySaver(),
        )
    token = CancellationToken()
    results = await asyncio.gather(
        *(
            handlers[index % 2].search_provider.search_with_usage(
                "q",
                5,
                None,
                deadline=time.monotonic() + 10,
                cancellation_token=token,
            )
            for index in range(12)
        )
    )
    assert search.max_active == 4
    assert sum(result.usage.search_calls for result in results) == 12
    await asyncio.gather(
        *(
            handlers[index % 2].fetcher.fetch(
                f"https://{host}/page-{index}",
                deadline=time.monotonic() + 10,
                cancellation_token=token,
            )
            for host in ("example.com", "other.example")
            for index in range(8)
        )
    )
    assert transport.max_active == {"example.com": 2, "other.example": 2}


async def test_redirect_reacquires_gate_for_new_hostname(store):
    limits = LimitManager(store, daily_limit=Decimal(1))
    acquired_hosts = []

    @asynccontextmanager
    async def recorded_slot(hostname):
        async with limits.fetch_slot(hostname):
            acquired_hosts.append(hostname)
            yield

    transport = FetchTransportSpy()
    fetcher = HttpxFetcher(transport=transport, resolver=public_resolver, host_slot=recorded_slot)
    result = await fetcher.fetch(
        "https://a.example/start",
        deadline=time.monotonic() + 10,
        cancellation_token=CancellationToken(),
    )
    assert str(result.final_url) == "https://b.example/final"
    assert acquired_hosts == ["a.example", "b.example"]
    assert transport.active == {"a.example": 0, "b.example": 0}


@pytest.mark.parametrize("kind,capacity", [("search", 4), ("fetch", 2)])
async def test_provider_slots_release_after_exception_and_cancellation(store, kind, capacity):
    limits = LimitManager(store, daily_limit=Decimal(1))
    slot = limits.search_slot if kind == "search" else lambda: limits.fetch_slot("EXAMPLE.com.")

    async def failing():
        async with slot():
            raise RuntimeError("provider failed")

    with pytest.raises(RuntimeError):
        await failing()
    entered = asyncio.Event()

    async def blocked():
        async with slot():
            entered.set()
            await asyncio.Event().wait()

    task = asyncio.create_task(blocked())
    await entered.wait()
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    from contextlib import AsyncExitStack

    async with AsyncExitStack() as stack:
        for _ in range(capacity):
            await asyncio.wait_for(stack.enter_async_context(slot()), 1)
