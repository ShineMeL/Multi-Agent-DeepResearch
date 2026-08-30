from decimal import Decimal

import pytest

from deepresearch.domain import ResourceUsage
from deepresearch.providers import ProviderError, SearchProvider
from deepresearch.providers.types import SearchHit
from deepresearch.runtime import CancellationToken, OperationCancelled


class FakeSearchProvider:
    provider_id = "fake-search"

    def __init__(self) -> None:
        self.calls = 0

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


class TestFakeSearchProvider(SearchProviderContract):
    provider = FakeSearchProvider()


@pytest.mark.asyncio
async def test_cancelled_search_fails_before_provider_call() -> None:
    provider = FakeSearchProvider()
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        await provider.search(
            "agent planning", 5, None, deadline=100.0, cancellation_token=token
        )
    assert provider.calls == 0
