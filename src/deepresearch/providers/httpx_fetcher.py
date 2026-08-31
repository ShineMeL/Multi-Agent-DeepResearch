from __future__ import annotations

import asyncio
import ipaddress
import socket
from collections.abc import AsyncGenerator, Awaitable, Callable, Sequence
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from datetime import UTC, datetime
from typing import cast
from urllib.parse import urljoin, urlsplit

import httpcore
import httpx

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
    ProviderErrorCode,
    ProviderUsageResult,
    RawDocument,
)
from deepresearch.retrieval import (
    URLSecurityError,
    canonicalize_url,
    validate_public_http_url,
)
from deepresearch.runtime import CancellationToken

from .httpx_transport import PinnedPeerTransport, await_with_controls, checkpoint

type IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
type Resolver = Callable[[str, int], Awaitable[Sequence[IPAddress]]]
type HostSlot = Callable[[str], AbstractAsyncContextManager[None]]
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ALLOWED_MEDIA_TYPES = frozenset({"text/html", "application/pdf"})


@asynccontextmanager
async def no_op_host_slot(hostname: str) -> AsyncGenerator[None]:
    del hostname
    yield


async def resolve_host(hostname: str, port: int) -> tuple[IPAddress, ...]:
    loop = asyncio.get_running_loop()
    records = await loop.getaddrinfo(
        hostname,
        port,
        family=socket.AF_UNSPEC,
        type=socket.SOCK_STREAM,
        proto=socket.IPPROTO_TCP,
    )
    addresses: list[IPAddress] = []
    for record in records:
        address = ipaddress.ip_address(record[4][0])
        if address not in addresses:
            addresses.append(address)
    return tuple(addresses)


def _fetch_usage(
    *, pages: int = 0, retries: int = 0, wall_seconds: float = 0
) -> ResourceUsage:
    return ResourceUsage.zero().model_copy(
        update={
            "pages": pages,
            "retries": retries,
            "wall_seconds": wall_seconds,
        }
    )


class HttpxFetcher:
    provider_id = "httpx-fetcher"

    def __init__(
        self,
        *,
        transport: PinnedPeerTransport,
        host_slot: HostSlot = no_op_host_slot,
        resolver: Resolver = resolve_host,
        max_redirects: int = 5,
        max_body_bytes: int = 10 * 1024 * 1024,
        executor: ProviderCallExecutor | None = None,
    ) -> None:
        if max_redirects < 0 or max_body_bytes <= 0:
            raise ValueError("fetch limits must be non-negative and positive")
        self._transport = transport
        self._host_slot = host_slot
        self._resolver = resolver
        self._max_redirects = max_redirects
        self._max_body_bytes = max_body_bytes
        self._executor = executor or ProviderCallExecutor(
            policy=ProviderCallPolicy.defaults()
        )

    @staticmethod
    def _private_url_error() -> ProviderError:
        return ProviderError(
            code="INVALID_REQUEST",
            provider="httpx-fetcher",
            operation="fetch",
            public_message="fetch target is private or otherwise non-public",
            retryable=False,
            usage=_fetch_usage(),
        )

    @staticmethod
    def _status_error(status: int) -> ProviderError | None:
        if 200 <= status < 300:
            return None
        if 300 <= status < 400:
            code = "INVALID_RESPONSE"
            retryable = False
        elif status in {401, 403}:
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
            provider="httpx-fetcher",
            operation="fetch",
            public_message=f"fetch origin returned HTTP {status}",
            retryable=retryable,
            usage=_fetch_usage(),
        )

    async def _validated_url(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[str, str, str]:
        canonical: str | None = None
        try:
            canonical = canonicalize_url(url)
        except URLSecurityError:
            pass
        if canonical is None:
            raise self._private_url_error()
        parsed = urlsplit(canonical)
        if parsed.hostname is None:
            raise self._private_url_error()
        port = parsed.port or (443 if parsed.scheme == "https" else 80)
        try:
            addresses = await await_with_controls(
                self._resolver(parsed.hostname, port),
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation="fetch",
            )
        except TimeoutError:
            raise ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation="fetch",
                public_message="fetch hostname resolution timed out",
                retryable=True,
                usage=_fetch_usage(),
            ) from None
        except OSError:
            raise ProviderError(
                code="NETWORK",
                provider=self.provider_id,
                operation="fetch",
                public_message="fetch hostname resolution failed",
                retryable=True,
                usage=_fetch_usage(),
            ) from None
        validated: str | None = None
        try:
            validated = str(validate_public_http_url(canonical, resolved_ips=addresses))
        except URLSecurityError:
            pass
        if validated is None:
            raise self._private_url_error()
        pinned = str(addresses[0])
        return validated, parsed.hostname.rstrip(".").casefold(), pinned

    @staticmethod
    def _peer_matches(response: httpx.Response, pinned_ip: str) -> bool:
        raw_peer = response.extensions.get("peer_ip")
        if not isinstance(raw_peer, str):
            return False
        try:
            peer = ipaddress.ip_address(raw_peer)
            pinned = ipaddress.ip_address(pinned_ip)
        except ValueError:
            return False
        return peer == pinned

    async def _read_body(
        self,
        response: httpx.Response,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> bytes:
        content_length = response.headers.get("content-length")
        if content_length is not None:
            try:
                declared = int(content_length)
            except ValueError:
                declared = -1
            if declared < 0:
                raise ProviderError(
                    code="INVALID_RESPONSE",
                    provider=self.provider_id,
                    operation="fetch",
                    public_message="fetch response has an invalid content length",
                    retryable=False,
                    usage=_fetch_usage(),
                )
            if declared > self._max_body_bytes:
                raise ProviderError(
                    code="INVALID_RESPONSE",
                    provider=self.provider_id,
                    operation="fetch",
                    public_message="fetch response body is too large",
                    retryable=False,
                    usage=_fetch_usage(),
                )
        body = bytearray()
        iterator = response.aiter_bytes()
        while True:
            try:
                chunk = await await_with_controls(
                    anext(iterator),
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                    provider_id=self.provider_id,
                    operation="fetch",
                )
            except StopAsyncIteration:
                break
            body.extend(chunk)
            checkpoint(
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation="fetch",
            )
            if len(body) > self._max_body_bytes:
                raise ProviderError(
                    code="INVALID_RESPONSE",
                    provider=self.provider_id,
                    operation="fetch",
                    public_message="fetch response body is too large",
                    retryable=False,
                    usage=_fetch_usage(),
                )
        return bytes(body)

    async def _fetch_once(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> RawDocument:
        try:
            requested_url = canonicalize_url(url)
        except URLSecurityError:
            raise self._private_url_error() from None
        current_url = requested_url
        for redirect_count in range(self._max_redirects + 1):
            validated_url, hostname, pinned_ip = await self._validated_url(
                current_url,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            redirect_target: str | None = None
            async with self._host_slot(hostname):
                try:
                    async with self._transport.stream(
                        validated_url,
                        pinned_ip=pinned_ip,
                        deadline=deadline,
                        cancellation_token=cancellation_token,
                    ) as response:
                        checkpoint(
                            deadline=deadline,
                            cancellation_token=cancellation_token,
                            provider_id=self.provider_id,
                            operation="fetch",
                        )
                        if not self._peer_matches(response, pinned_ip):
                            raise ProviderError(
                                code="INVALID_RESPONSE",
                                provider=self.provider_id,
                                operation="fetch",
                                public_message=(
                                    "actual peer IP does not match pinned public address"
                                ),
                                retryable=False,
                                usage=_fetch_usage(),
                            )
                        if response.status_code in _REDIRECT_STATUSES:
                            location = response.headers.get("location")
                            if not location:
                                raise ProviderError(
                                    code="INVALID_RESPONSE",
                                    provider=self.provider_id,
                                    operation="fetch",
                                    public_message="redirect response omitted Location",
                                    retryable=False,
                                    usage=_fetch_usage(),
                                )
                            if redirect_count >= self._max_redirects:
                                raise ProviderError(
                                    code="INVALID_RESPONSE",
                                    provider=self.provider_id,
                                    operation="fetch",
                                    public_message="fetch redirect limit exceeded",
                                    retryable=False,
                                    usage=_fetch_usage(),
                                )
                            redirect_target = urljoin(validated_url, location)
                        else:
                            status_error = self._status_error(response.status_code)
                            if status_error is not None:
                                raise status_error
                            media_type = response.headers.get("content-type", "").split(
                                ";", 1
                            )[0].strip().casefold()
                            if media_type not in _ALLOWED_MEDIA_TYPES:
                                raise ProviderError(
                                    code="PARSE_UNSUPPORTED",
                                    provider=self.provider_id,
                                    operation="fetch",
                                    public_message="fetch response media type is unsupported",
                                    retryable=False,
                                    usage=_fetch_usage(),
                                )
                            body = await self._read_body(
                                response,
                                deadline=deadline,
                                cancellation_token=cancellation_token,
                            )
                            return RawDocument.model_validate(
                                {
                                    "requested_url": requested_url,
                                    "final_url": validated_url,
                                    "status": response.status_code,
                                    "headers": dict(response.headers),
                                    "content_type": response.headers.get(
                                        "content-type", media_type
                                    ),
                                    "body_bytes": body,
                                    "retrieved_at": datetime.now(UTC),
                                }
                            )
                except (httpx.ProtocolError, httpcore.ProtocolError):
                    raise ProviderError(
                        code="INVALID_RESPONSE",
                        provider=self.provider_id,
                        operation="fetch",
                        public_message="fetch peer returned invalid HTTP framing",
                        retryable=False,
                        usage=_fetch_usage(),
                    ) from None
                except (httpx.TimeoutException, httpcore.TimeoutException):
                    raise ProviderError(
                        code="TIMEOUT",
                        provider=self.provider_id,
                        operation="fetch",
                        public_message="fetch network operation timed out",
                        retryable=True,
                        usage=_fetch_usage(),
                    ) from None
                except (httpx.HTTPError, httpcore.NetworkError, OSError):
                    raise ProviderError(
                        code="NETWORK",
                        provider=self.provider_id,
                        operation="fetch",
                        public_message="fetch network operation failed",
                        retryable=True,
                        usage=_fetch_usage(),
                    ) from None
            current_url = redirect_target
        raise RuntimeError("fetch redirect loop exited unexpectedly")

    async def fetch(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> RawDocument:
        return (
            await self.fetch_with_usage(
                url,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
        ).value

    async def fetch_with_usage(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ProviderUsageResult[RawDocument]:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation="fetch",
        )
        async def invoke(call_deadline: float) -> RawDocument:
            return await self._fetch_once(
                url,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            )

        outcome = await self._executor.call_with_trace(
            "fetch", invoke, remaining_deadline=deadline
        )
        usage = _fetch_usage(
            pages=int(outcome.result is not None),
            retries=outcome.trace.retries,
            wall_seconds=outcome.trace.wall_seconds,
        )
        if outcome.error is not None:
            failure = ProviderError(
                code=outcome.error.code,
                provider=self.provider_id,
                operation="fetch",
                public_message=outcome.error.public_message,
                retryable=outcome.error.retryable,
                retry_after=outcome.error.retry_after,
                usage=usage,
            )
            raise failure from None
        if outcome.result is None:
            raise RuntimeError("fetch executor returned no document")
        return ProviderUsageResult(value=outcome.result, usage=usage)


__all__ = ["HostSlot", "HttpxFetcher", "no_op_host_slot", "resolve_host"]
