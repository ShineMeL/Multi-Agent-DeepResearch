from __future__ import annotations

import json
import time
from collections.abc import Mapping
from math import isfinite
from typing import cast
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue, SecretStr, ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
    ProviderErrorCode,
    SearchHit,
)
from deepresearch.runtime import CancellationToken

from .httpx_transport import await_with_controls, checkpoint


def _search_usage(*, retries: int = 0, wall_seconds: float = 0) -> ResourceUsage:
    return ResourceUsage.zero().model_copy(
        update={
            "search_calls": 1,
            "retries": retries,
            "wall_seconds": wall_seconds,
        }
    )


class TavilySearchProvider:
    provider_id = "tavily"

    def __init__(
        self,
        *,
        api_key: SecretStr,
        endpoint: str = "https://api.tavily.com/search",
        executor: ProviderCallExecutor | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlsplit(endpoint)
        if parsed.scheme != "https" or parsed.hostname is None:
            raise ValueError("Tavily endpoint must be an absolute HTTPS URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("Tavily endpoint must not contain credentials")
        self._endpoint = endpoint
        self._api_key = api_key
        self._executor = executor or ProviderCallExecutor(
            policy=ProviderCallPolicy.defaults()
        )
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None
        self.last_usage: ResourceUsage | None = None

    def __repr__(self) -> str:
        return f"{type(self).__name__}(provider_id={self.provider_id!r})"

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    @staticmethod
    def _status_error(status: int) -> ProviderError | None:
        if 200 <= status < 300:
            return None
        if status in {401, 403}:
            code = "AUTHENTICATION"
            retryable = False
        elif status == 429:
            code = "RATE_LIMITED"
            retryable = True
        elif status >= 500:
            code = "UPSTREAM_5XX"
            retryable = True
        else:
            code = "INVALID_REQUEST"
            retryable = False
        return ProviderError(
            code=cast("ProviderErrorCode", code),
            provider="tavily",
            operation="search",
            public_message=f"search provider returned HTTP {status}",
            retryable=retryable,
            usage=_search_usage(),
        )

    async def _search_once(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        if not query.strip() or limit <= 0:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="search",
                public_message="search query and limit are invalid",
                retryable=False,
                usage=_search_usage(),
            )
        payload: dict[str, JsonValue] = {
            "include_answer": False,
            "max_results": limit,
            "query": query,
        }
        if filters is not None:
            forbidden = {"api_key", "authorization", "max_results", "query"}
            if any(key.casefold() in forbidden for key in filters):
                raise ProviderError(
                    code="INVALID_REQUEST",
                    provider=self.provider_id,
                    operation="search",
                    public_message="search filters contain a reserved field",
                    retryable=False,
                    usage=_search_usage(),
                )
            payload.update(filters)
        response: httpx.Response | None = None
        failure: ProviderError | None = None
        try:
            response = await await_with_controls(
                self._client.post(
                    self._endpoint,
                    headers={
                        "authorization": f"Bearer {self._api_key.get_secret_value()}",
                        "content-type": "application/json",
                    },
                    json=payload,
                    timeout=max(deadline - time.monotonic(), 0.001),
                ),
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation="search",
            )
        except httpx.TimeoutException:
            failure = ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation="search",
                public_message="search provider timed out",
                retryable=True,
                usage=_search_usage(),
            )
        except httpx.HTTPError:
            failure = ProviderError(
                code="NETWORK",
                provider=self.provider_id,
                operation="search",
                public_message="search provider network request failed",
                retryable=True,
                usage=_search_usage(),
            )
        if failure is not None:
            raise failure
        if response is None:
            raise RuntimeError("search response state is unavailable")
        status_error = self._status_error(response.status_code)
        if status_error is not None:
            retry_after = response.headers.get("retry-after")
            if retry_after is not None:
                try:
                    parsed_retry = float(retry_after)
                except ValueError:
                    parsed_retry = -1
                if isfinite(parsed_retry) and parsed_retry >= 0:
                    status_error.retry_after = parsed_retry
            raise status_error
        decoded: object | None = None
        try:
            decoded = json.loads(response.content)
        except (UnicodeDecodeError, json.JSONDecodeError):
            decoded = None
        if not isinstance(decoded, dict):
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="search",
                public_message="search provider returned an invalid response",
                retryable=False,
                usage=_search_usage(),
            )
        decoded_mapping = cast("dict[str, object]", decoded)
        results_value = decoded_mapping.get("results")
        if not isinstance(results_value, list):
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="search",
                public_message="search provider returned an invalid response",
                retryable=False,
                usage=_search_usage(),
            )
        hits: list[SearchHit] = []
        invalid = False
        try:
            results = cast("list[object]", results_value)
            for rank, raw in enumerate(results[:limit], start=1):
                if not isinstance(raw, dict):
                    raise TypeError
                raw_mapping = cast("dict[str, object]", raw)
                url = raw_mapping.get("url")
                title = raw_mapping.get("title")
                snippet = raw_mapping.get("content")
                if not all(isinstance(item, str) for item in (url, title, snippet)):
                    raise TypeError
                metadata: dict[str, JsonValue] = {}
                score = raw_mapping.get("score")
                if score is not None:
                    if type(score) not in {int, float}:
                        raise ValueError
                    score_number = cast("int | float", score)
                    if not isfinite(score_number):
                        raise ValueError
                    metadata["score"] = score_number
                hits.append(
                    SearchHit.model_validate(
                        {
                            "url": cast("str", url),
                            "title": cast("str", title),
                            "snippet": cast("str", snippet),
                            "rank": rank,
                            "published_at": raw_mapping.get("published_date"),
                            "provider_metadata": metadata,
                        }
                    )
                )
        except (TypeError, ValueError, ValidationError):
            invalid = True
        if invalid:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="search",
                public_message="search provider returned an invalid result shape",
                retryable=False,
                usage=_search_usage(),
            )
        return hits

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation="search",
        )
        before = len(self._executor.attempts)
        started = time.monotonic()
        caught: ProviderError | None = None
        result: list[SearchHit] | None = None

        async def invoke(call_deadline: float) -> list[SearchHit]:
            return await self._search_once(
                query,
                limit,
                filters,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            )

        try:
            result = await self._executor.call(
                "search", invoke, remaining_deadline=deadline
            )
        except ProviderError as error:
            caught = error
        attempts = self._executor.attempts[before:]
        retries = sum(attempt.attempt_index > 0 for attempt in attempts)
        usage = _search_usage(
            retries=retries,
            wall_seconds=max(0, time.monotonic() - started),
        )
        self.last_usage = usage
        if caught is not None:
            failure = ProviderError(
                code=caught.code,
                provider=self.provider_id,
                operation="search",
                public_message=caught.public_message,
                retryable=caught.retryable,
                retry_after=caught.retry_after,
                usage=usage,
            )
            raise failure
        if result is None:
            raise RuntimeError("search executor returned no result")
        return result


__all__ = ["TavilySearchProvider"]
