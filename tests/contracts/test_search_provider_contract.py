import asyncio
from collections.abc import Callable
from decimal import Decimal

import pytest

from deepresearch.domain import ResourceUsage
from deepresearch.providers import ProviderError, SearchProvider
from deepresearch.providers.types import SearchHit
from deepresearch.runtime import CancellationToken, OperationCancelled


class AwaitGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started.set()
        await self.release.wait()


class FakeSearchProvider:
    provider_id = "fake-search"

    def __init__(self, *, gate: AwaitGate | None = None) -> None:
        self._gate = gate
        self.calls = 0
        self.response_closed = False

    async def search(
        self,
        query: str,
        limit: int,
        filters: dict[str, object] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del filters, deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        if query == "timeout":
            raise ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation="search",
                public_message="search timed out",
                retryable=True,
            )
        if not query or limit <= 0:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="search",
                public_message="invalid search request",
                retryable=False,
            )
        try:
            if self._gate is not None:
                await self._gate.wait()
            else:
                await _yield_once()
            cancellation_token.raise_if_cancelled()
            return [
                SearchHit(
                    url="https://example.com",
                    title="Example",
                    snippet="Result",
                    rank=1,
                    provider_metadata={"query": query},
                )
            ][:limit]
        finally:
            self.response_closed = True


async def _yield_once() -> None:
    return None


def search_usage() -> ResourceUsage:
    return ResourceUsage(
        input_tokens=0,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=0,
        search_calls=1,
        pages=0,
        retries=0,
        wall_seconds=0.01,
        cost_usd=Decimal("0.001"),
    )


class SearchProviderContract:
    provider: SearchProvider
    provider_factory: Callable[[AwaitGate], SearchProvider]
    response_closed: Callable[[SearchProvider], bool]
    call_count: Callable[[SearchProvider], int]

    @pytest.mark.asyncio
    async def test_success_is_typed_stable_and_has_operation_level_usage(self) -> None:
        hits = await self.provider.search(
            "agent planning",
            5,
            None,
            deadline=10.0,
            cancellation_token=CancellationToken(),
        )

        assert len(hits) == 1
        assert SearchHit.model_validate_json(hits[0].model_dump_json()) == hits[0]
        assert "usage" not in SearchHit.model_fields
        assert search_usage().search_calls == 1

    @pytest.mark.asyncio
    async def test_timeout_and_invalid_request_map_to_stable_codes(self) -> None:
        with pytest.raises(ProviderError) as timeout:
            await self.provider.search(
                "timeout", 1, None, deadline=10.0, cancellation_token=CancellationToken()
            )
        assert timeout.value.code == "TIMEOUT"
        with pytest.raises(ProviderError) as invalid:
            await self.provider.search(
                "", 1, None, deadline=10.0, cancellation_token=CancellationToken()
            )
        assert invalid.value.code == "INVALID_REQUEST"

    @pytest.mark.asyncio
    async def test_cancellation_before_and_after_await_is_enforced(self) -> None:
        before_provider = self.provider_factory(AwaitGate())
        before_token = CancellationToken()
        before_token.cancel()
        with pytest.raises(OperationCancelled):
            await before_provider.search(
                "agent planning",
                5,
                None,
                deadline=10.0,
                cancellation_token=before_token,
            )
        assert self.call_count(before_provider) == 0

        gate = AwaitGate()
        provider = self.provider_factory(gate)
        token = CancellationToken()
        task = asyncio.create_task(
            provider.search(
                "agent planning", 5, None, deadline=10.0, cancellation_token=token
            )
        )
        await gate.started.wait()
        token.cancel()
        gate.release.set()
        with pytest.raises(OperationCancelled):
            await task
        assert self.response_closed(provider) is True


class TestFakeSearchProvider(SearchProviderContract):
    provider = FakeSearchProvider()
    provider_factory = staticmethod(lambda gate: FakeSearchProvider(gate=gate))
    response_closed = staticmethod(lambda provider: provider.response_closed)
    call_count = staticmethod(lambda provider: provider.calls)
