from __future__ import annotations

import json
import re
import time
from collections.abc import Mapping
from math import isfinite
from typing import cast
from urllib.parse import unquote, urlsplit

import httpx
from pydantic import JsonValue, SecretStr, ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
    ProviderErrorCode,
    ProviderUsageResult,
    SearchHit,
)
from deepresearch.retrieval import URLSecurityError, canonicalize_url
from deepresearch.runtime import CancellationToken

from .httpx_transport import await_with_controls, checkpoint

_RESULT_CREDENTIAL_ASSIGNMENT = re.compile(
    r"(?i)(?<![a-z0-9])(?:api[_ .-]?key|access[_ .-]?key|authorization|auth|"
    r"bearer|cookie|credential|password|passwd|secret|private[_ .-]?key|"
    r"session(?:[_ .-]?(?:id|token))?|access[_ .-]?token|token)"
    r"(?![a-z0-9])\s*(?:=|:)\s*[^\s&,;/#]+"
)
_RESULT_OPAQUE_CREDENTIAL = re.compile(
    r"(?i)(?:^|[^a-z0-9])(?:sk[-_](?:(?:proj|live|test)[-_])?[a-z0-9_-]{6,}|"
    r"(?:gh[opsu]|github_pat)_[a-z0-9_]{6,}|xox[baprs]-[a-z0-9-]{6,})"
    r"(?:$|[^a-z0-9])"
)


def _bounded_decode(value: str) -> str:
    current = value
    for _ in range(6):
        decoded = unquote(current, encoding="utf-8", errors="strict")
        if decoded == current:
            return current
        current = decoded
    raise ValueError("search result URL encoding exceeds the safety bound")


def _result_url_is_safe(url: str) -> bool:
    try:
        canonicalize_url(url)
        split = urlsplit(url)
        decoded = "\n".join(
            _bounded_decode(part) for part in (split.path, split.query, split.fragment)
        )
    except (TypeError, UnicodeError, ValueError, URLSecurityError):
        return False
    return (
        split.username is None
        and split.password is None
        and _RESULT_CREDENTIAL_ASSIGNMENT.search(decoded) is None
        and _RESULT_OPAQUE_CREDENTIAL.search(decoded) is None
    )


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
                if not _result_url_is_safe(cast("str", url)):
                    raise ValueError
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
        return (
            await self.search_with_usage(
                query,
                limit,
                filters,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
        ).value

    async def search_with_usage(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ProviderUsageResult[list[SearchHit]]:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation="search",
        )
        async def invoke(call_deadline: float) -> list[SearchHit]:
            return await self._search_once(
                query,
                limit,
                filters,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            )

        outcome = await self._executor.call_with_trace(
            "search", invoke, remaining_deadline=deadline
        )
        usage = _search_usage(
            retries=outcome.trace.retries,
            wall_seconds=outcome.trace.wall_seconds,
        )
        if outcome.error is not None:
            failure = ProviderError(
                code=outcome.error.code,
                provider=self.provider_id,
                operation="search",
                public_message=outcome.error.public_message,
                retryable=outcome.error.retryable,
                retry_after=outcome.error.retry_after,
                usage=usage,
            )
            raise failure from None
        if outcome.result is None:
            raise RuntimeError("search executor returned no result")
        return ProviderUsageResult(value=outcome.result, usage=usage)


__all__ = ["TavilySearchProvider"]
