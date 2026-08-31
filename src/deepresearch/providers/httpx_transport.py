from __future__ import annotations

import asyncio
import contextlib
import ssl
import time
from collections.abc import AsyncGenerator, AsyncIterator, Awaitable, Iterable
from contextlib import asynccontextmanager
from math import isfinite
from typing import Any, Protocol, cast, override
from urllib.parse import urlsplit

import httpcore
import httpx

from deepresearch.providers import ProviderError
from deepresearch.runtime import CancellationToken

_POLL_SECONDS = 0.05


class _NetworkStream(Protocol):
    def get_extra_info(self, info: str) -> object: ...


def checkpoint(
    *,
    deadline: float,
    cancellation_token: CancellationToken,
    provider_id: str,
    operation: str,
) -> None:
    if not isfinite(deadline):
        raise ValueError("deadline must be finite")
    cancellation_token.raise_if_cancelled()
    if time.monotonic() >= deadline:
        raise ProviderError(
            code="TIMEOUT",
            provider=provider_id,
            operation=operation,
            public_message="provider call deadline exceeded",
            retryable=True,
        )


async def await_with_controls[T](
    awaitable: Awaitable[T],
    *,
    deadline: float,
    cancellation_token: CancellationToken,
    provider_id: str,
    operation: str,
) -> T:
    checkpoint(
        deadline=deadline,
        cancellation_token=cancellation_token,
        provider_id=provider_id,
        operation=operation,
    )
    task = asyncio.ensure_future(awaitable)
    try:
        while not task.done():
            checkpoint(
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=provider_id,
                operation=operation,
            )
            remaining = deadline - time.monotonic()
            await asyncio.wait({task}, timeout=min(_POLL_SECONDS, max(remaining, 0)))
        return await task
    except BaseException:
        if not task.done():
            task.cancel()
            with contextlib.suppress(BaseException):
                await task
        raise


class _PinnedNetworkBackend(httpcore.AsyncNetworkBackend):
    def __init__(self, *, hostname: str, pinned_ip: str) -> None:
        self._hostname = hostname.rstrip(".").casefold()
        self._pinned_ip = pinned_ip
        self._delegate = cast(
            "httpcore.AsyncNetworkBackend", httpcore.AnyIOBackend()
        )

    @override
    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        if host.rstrip(".").casefold() != self._hostname:
            raise httpcore.ConnectError("connection host differs from pinned hostname")
        return await self._delegate.connect_tcp(
            self._pinned_ip,
            port,
            timeout=timeout,
            local_address=local_address,
            socket_options=socket_options,
        )

    @override
    async def connect_unix_socket(
        self,
        path: str,
        timeout: float | None = None,
        socket_options: Iterable[httpcore.SOCKET_OPTION] | None = None,
    ) -> httpcore.AsyncNetworkStream:
        raise httpcore.ConnectError("Unix sockets are disabled for public HTTP fetching")

    @override
    async def sleep(self, seconds: float) -> None:
        await self._delegate.sleep(seconds)


class _ResponseStream(httpx.AsyncByteStream):
    def __init__(self, stream: AsyncIterator[bytes]) -> None:
        self._stream = stream

    @override
    async def __aiter__(self) -> AsyncIterator[bytes]:
        async for chunk in self._stream:
            yield chunk

    @override
    async def aclose(self) -> None:
        close = getattr(self._stream, "aclose", None)
        if callable(close):
            await cast("Awaitable[None]", close())


class _CoreTransport(httpx.AsyncBaseTransport):
    def __init__(self, pool: httpcore.AsyncConnectionPool) -> None:
        self._pool = pool

    @override
    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        core_request = httpcore.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.raw,
            content=request.stream,
            extensions=request.extensions,
        )
        response = await self._pool.handle_async_request(core_request)
        stream = cast("AsyncIterator[bytes]", response.stream)
        return httpx.Response(
            status_code=response.status,
            headers=response.headers,
            stream=_ResponseStream(stream),
            extensions=cast(
                "dict[str, Any]",
                response.extensions,  # pyright: ignore[reportUnknownMemberType]
            ),
        )

    @override
    async def aclose(self) -> None:
        await self._pool.aclose()


class PinnedPeerTransport:
    """Open one proxy-free HTTPX connection pinned to a validated address.

    The HTTP request retains the original hostname. The custom network backend only
    substitutes the TCP destination, so HTTP Host, TLS SNI, and certificate validation
    continue to use the hostname while the socket connects to ``pinned_ip``.
    """

    def __init__(
        self,
        *,
        connect_timeout_seconds: float = 10.0,
        read_timeout_seconds: float = 20.0,
    ) -> None:
        for value in (connect_timeout_seconds, read_timeout_seconds):
            if not isfinite(value) or value <= 0:
                raise ValueError("transport timeouts must be finite and positive")
        self._connect_timeout_seconds = connect_timeout_seconds
        self._read_timeout_seconds = read_timeout_seconds

    @asynccontextmanager
    async def stream(
        self,
        url: str,
        *,
        pinned_ip: str,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncGenerator[httpx.Response]:
        provider_id = "httpx-fetcher"
        operation = "fetch"
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=provider_id,
            operation=operation,
        )
        parsed = urlsplit(url)
        if parsed.hostname is None:
            raise ValueError("URL must contain a hostname")
        remaining = deadline - time.monotonic()
        timeout = httpx.Timeout(
            connect=min(self._connect_timeout_seconds, remaining),
            read=min(self._read_timeout_seconds, remaining),
            write=min(self._connect_timeout_seconds, remaining),
            pool=min(self._connect_timeout_seconds, remaining),
        )
        backend = _PinnedNetworkBackend(
            hostname=parsed.hostname,
            pinned_ip=pinned_ip,
        )
        pool = httpcore.AsyncConnectionPool(
            ssl_context=ssl.create_default_context(),
            max_connections=1,
            max_keepalive_connections=0,
            retries=0,
            network_backend=backend,
        )
        client = httpx.AsyncClient(
            transport=_CoreTransport(pool),
            timeout=timeout,
            follow_redirects=False,
            trust_env=False,
        )
        response_context = client.stream(
            "GET",
            url,
            headers={"accept": "text/html, application/pdf"},
        )
        response: httpx.Response | None = None
        try:
            response = await await_with_controls(
                response_context.__aenter__(),
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=provider_id,
                operation=operation,
            )
            network_stream = cast(
                "_NetworkStream | None", response.extensions.get("network_stream")
            )
            server_address = (
                None
                if network_stream is None
                else network_stream.get_extra_info("server_addr")
            )
            peer_ip: str | None = (
                cast("str", server_address[0])
                if isinstance(server_address, tuple) and server_address
                else None
            )
            response.extensions["peer_ip"] = peer_ip
            yield response
        finally:
            if response is not None:
                with contextlib.suppress(Exception):
                    await response.aclose()
            with contextlib.suppress(Exception):
                await response_context.__aexit__(None, None, None)
            with contextlib.suppress(Exception):
                await client.aclose()


__all__ = ["PinnedPeerTransport"]
