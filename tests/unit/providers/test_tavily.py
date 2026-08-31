import time

import pytest
import respx
from pydantic import SecretStr

from deepresearch.providers import (
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
)
from deepresearch.providers.tavily import TavilySearchProvider
from deepresearch.runtime import CancellationToken, OperationCancelled


def _deadline() -> float:
    return time.monotonic() + 10.0


def _provider() -> TavilySearchProvider:
    defaults = ProviderCallPolicy.defaults()
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy(
            default_timeout_seconds=defaults.default_timeout_seconds,
            max_retries=0,
            base_delay_seconds=0,
            max_delay_seconds=0,
            jitter_ratio=0,
        )
    )
    return TavilySearchProvider(
        api_key=SecretStr("TOP-SECRET-SEARCH"), executor=executor
    )


@pytest.mark.asyncio
@respx.mock
async def test_tavily_maps_valid_results_to_stable_rank_and_metadata() -> None:
    route = respx.post("https://api.tavily.com/search").respond(
        200,
        json={
            "query": "agents",
            "results": [
                {
                    "url": "https://example.com/one",
                    "title": "First",
                    "content": "first snippet",
                    "score": 0.9,
                },
                {
                    "url": "https://example.com/two",
                    "title": "Second",
                    "content": "second snippet",
                    "score": 0.7,
                },
            ],
        },
    )
    provider = _provider()

    outcome = await provider.search_with_usage(
        "agents",
        2,
        {"topic": "general"},
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert [hit.rank for hit in outcome.value] == [1, 2]
    assert [hit.title for hit in outcome.value] == ["First", "Second"]
    assert outcome.value[0].provider_metadata == {"score": 0.9}
    assert outcome.usage.search_calls == 1
    assert route.calls.last.request.headers["authorization"] == "Bearer TOP-SECRET-SEARCH"
    assert route.call_count == 1
    assert "TOP-SECRET" not in repr(provider)


@pytest.mark.asyncio
@respx.mock
async def test_tavily_rejects_malformed_result_shape() -> None:
    respx.post("https://api.tavily.com/search").respond(
        200,
        json={
            "results": [
                {
                    "url": "https://example.com/one",
                    "title": "First",
                    "content": 123,
                }
            ]
        },
    )

    with pytest.raises(ProviderError) as error:
        await _provider().search(
            "agents",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.usage is not None
    assert error.value.usage.search_calls == 1


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "unsafe_url",
    (
        "https://user:PASS@example.com/result",
        "https://example.com/result?api_key=TOP-SECRET-SEARCH",
    ),
)
async def test_tavily_rejects_credential_bearing_result_url(
    unsafe_url: str,
) -> None:
    respx.post("https://api.tavily.com/search").respond(
        200,
        json={
            "results": [
                {
                    "url": unsafe_url,
                    "title": "Unsafe",
                    "content": "unsafe snippet",
                }
            ]
        },
    )

    with pytest.raises(ProviderError) as error:
        await _provider().search(
            "agents",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert "TOP-SECRET" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
async def test_tavily_maps_rate_limit_without_serializing_secret() -> None:
    respx.post("https://api.tavily.com/search").respond(
        429,
        headers={"retry-after": "2"},
        text="TOP-SECRET-SEARCH diagnostic",
    )

    with pytest.raises(ProviderError) as error:
        await _provider().search(
            "agents",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "RATE_LIMITED"
    assert error.value.retry_after == 2
    assert "TOP-SECRET" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
async def test_tavily_cancelled_before_call_never_reaches_network() -> None:
    route = respx.post("https://api.tavily.com/search").respond(
        200, json={"results": []}
    )
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        await _provider().search(
            "agents", 1, None, deadline=_deadline(), cancellation_token=token
        )

    assert route.call_count == 0
