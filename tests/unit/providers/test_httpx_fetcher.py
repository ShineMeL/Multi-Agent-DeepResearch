import ipaddress
import socket
import ssl
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import AbstractAsyncContextManager, asynccontextmanager
from dataclasses import dataclass

import httpcore
import httpx
import pytest

from deepresearch.providers import ProviderError
from deepresearch.providers.httpx_fetcher import HttpxFetcher
from deepresearch.providers.httpx_transport import PinnedPeerTransport
from deepresearch.runtime import CancellationToken, OperationCancelled


def _deadline() -> float:
    return time.monotonic() + 10.0


class ChunkStream(httpx.AsyncByteStream):
    def __init__(
        self,
        chunks: tuple[bytes, ...],
        *,
        after_chunk: Callable[[int], None] | None = None,
    ) -> None:
        self.chunks = chunks
        self.after_chunk = after_chunk
        self.closed = False

    async def __aiter__(self) -> AsyncIterator[bytes]:
        for index, chunk in enumerate(self.chunks):
            yield chunk
            if self.after_chunk is not None:
                self.after_chunk(index)

    async def aclose(self) -> None:
        self.closed = True


@dataclass(frozen=True)
class ResponseSpec:
    status: int
    headers: Mapping[str, str]
    chunks: tuple[bytes, ...]
    peer_ip: str
    after_chunk: Callable[[int], None] | None = None


class SequenceTransport:
    def __init__(self, *responses: ResponseSpec) -> None:
        self.responses = list(responses)
        self.urls: list[str] = []
        self.pinned_ips: list[str] = []
        self.streams: list[ChunkStream] = []
        self.proxy_connections = 0

    @asynccontextmanager
    async def stream(
        self,
        url: str,
        *,
        pinned_ip: str,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[httpx.Response]:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.urls.append(url)
        self.pinned_ips.append(pinned_ip)
        spec = self.responses.pop(0)
        stream = ChunkStream(spec.chunks, after_chunk=spec.after_chunk)
        self.streams.append(stream)
        response = httpx.Response(
            spec.status,
            headers=spec.headers,
            stream=stream,
            request=httpx.Request("GET", url),
            extensions={"peer_ip": spec.peer_ip},
        )
        try:
            yield response
        finally:
            await response.aclose()


class ProtocolFailureTransport:
    @asynccontextmanager
    async def stream(
        self,
        url: str,
        *,
        pinned_ip: str,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[httpx.Response]:
        del url, pinned_ip, deadline, cancellation_token
        raise httpcore.RemoteProtocolError("TOP-SECRET malformed peer bytes")
        if False:
            yield httpx.Response(200)


def _resolver(
    values: Mapping[str, tuple[str, ...]],
) -> Callable[[str, int], Awaitable[tuple[ipaddress.IPv4Address, ...]]]:
    async def resolve(
        hostname: str, port: int
    ) -> tuple[ipaddress.IPv4Address, ...]:
        assert port in {80, 443}
        return tuple(ipaddress.ip_address(value) for value in values[hostname])

    return resolve


class RecordingHostSlot:
    def __init__(self) -> None:
        self.hosts: list[str] = []
        self.active: str | None = None

    def __call__(self, hostname: str) -> AbstractAsyncContextManager[None]:
        @asynccontextmanager
        async def slot() -> AsyncIterator[None]:
            assert self.active is None
            self.active = hostname
            self.hosts.append(hostname)
            try:
                yield
            finally:
                self.active = None

        return slot()


class RecordingNetworkStream(httpcore.AsyncMockStream):
    def __init__(
        self,
        response_bytes: bytes = (
            b"HTTP/1.1 200 OK\r\n"
            b"Content-Type: text/html\r\n"
            b"Content-Length: 2\r\n"
            b"Connection: close\r\n\r\nOK"
        ),
    ) -> None:
        super().__init__([response_bytes])
        self.writes: list[bytes] = []
        self.server_hostname: str | None = None

    async def write(self, buffer: bytes, timeout: float | None = None) -> None:
        del timeout
        self.writes.append(buffer)

    async def start_tls(
        self,
        ssl_context: ssl.SSLContext,
        server_hostname: str | None = None,
        timeout: float | None = None,
    ) -> httpcore.AsyncNetworkStream:
        del ssl_context, timeout
        self.server_hostname = server_hostname
        return self

    def get_extra_info(self, info: str) -> object:
        if info == "server_addr":
            return ("93.184.216.34", 443)
        return super().get_extra_info(info)


class RecordingNetworkBackend(httpcore.AsyncMockBackend):
    def __init__(self, stream: RecordingNetworkStream) -> None:
        super().__init__([])
        self.stream = stream
        self.destinations: list[tuple[str, int]] = []

    async def connect_tcp(
        self,
        host: str,
        port: int,
        timeout: float | None = None,
        local_address: str | None = None,
        socket_options: object = None,
    ) -> httpcore.AsyncNetworkStream:
        del timeout, local_address, socket_options
        self.destinations.append((host, port))
        return self.stream


@pytest.mark.asyncio
async def test_pinned_transport_preserves_host_and_tls_sni_while_pinning_tcp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stream = RecordingNetworkStream()
    backend = RecordingNetworkBackend(stream)
    monkeypatch.setattr(httpcore, "AnyIOBackend", lambda: backend)
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")

    async with PinnedPeerTransport().stream(
        "https://public.test/article",
        pinned_ip="93.184.216.34",
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    ) as response:
        assert await response.aread() == b"OK"
        assert response.extensions["peer_ip"] == "93.184.216.34"

    request_bytes = b"".join(stream.writes)
    assert backend.destinations == [("93.184.216.34", 443)]
    assert stream.server_hostname == "public.test"
    assert b"Host: public.test" in request_bytes
    assert b"127.0.0.1:9" not in request_bytes


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "response_bytes",
    (
        b"HTTP/1.1 INVALID\r\nContent-Length: 0\r\n\r\n",
        b"HTTP/1.1 200 OK\r\nMalformed Header\r\n\r\n",
        (
            b"HTTP/1.1 200 OK\r\nContent-Type: text/html\r\n"
            b"Transfer-Encoding: chunked\r\n\r\nNOT-A-CHUNK\r\n"
        ),
    ),
)
async def test_pinned_transport_maps_malformed_http_framing_to_typed_failure(
    monkeypatch: pytest.MonkeyPatch,
    response_bytes: bytes,
) -> None:
    stream = RecordingNetworkStream(response_bytes)
    monkeypatch.setattr(
        httpcore, "AnyIOBackend", lambda: RecordingNetworkBackend(stream)
    )

    with pytest.raises(ProviderError) as error:
        async with PinnedPeerTransport().stream(
            "https://public.test/article",
            pinned_ip="93.184.216.34",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        ) as response:
            await response.aread()

    assert error.value.code == "INVALID_RESPONSE"
    assert "INVALID" not in error.value.public_message
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
async def test_fetcher_revalidates_redirect_target() -> None:
    transport = SequenceTransport(
        ResponseSpec(
            status=302,
            headers={"location": "http://127.0.0.1/private"},
            chunks=(),
            peer_ip="93.184.216.34",
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver(
            {
                "public.test": ("93.184.216.34",),
                "127.0.0.1": ("127.0.0.1",),
            }
        ),
    )

    with pytest.raises(ProviderError, match="private"):
        await fetcher.fetch(
            "https://public.test/start",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert all(stream.closed for stream in transport.streams)


@pytest.mark.asyncio
async def test_fetcher_maps_invalid_initial_url_to_typed_failure() -> None:
    fetcher = HttpxFetcher(transport=SequenceTransport())

    with pytest.raises(ProviderError) as error:
        await fetcher.fetch(
            "https://user:secret@public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert error.value.usage is not None


@pytest.mark.asyncio
async def test_fetcher_rejects_unhandled_redirect_status_as_invalid_response() -> None:
    transport = SequenceTransport(
        ResponseSpec(
            status=304,
            headers={"content-type": "text/html"},
            chunks=(),
            peer_ip="93.184.216.34",
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
    )

    with pytest.raises(ProviderError) as error:
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert transport.streams[0].closed is True


@pytest.mark.asyncio
async def test_fetcher_rejects_dns_rebinding_and_ignores_proxy_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:9")
    transport = SequenceTransport(
        ResponseSpec(
            status=200,
            headers={"content-type": "text/html"},
            chunks=(b"<p>private peer</p>",),
            peer_ip="127.0.0.1",
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
    )

    with pytest.raises(ProviderError, match="peer IP"):
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert transport.proxy_connections == 0
    assert all(stream.closed for stream in transport.streams)


@pytest.mark.asyncio
async def test_fetcher_rejects_resolution_if_any_candidate_is_private() -> None:
    transport = SequenceTransport()
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver(
            {"public.test": ("93.184.216.34", "127.0.0.1")}
        ),
    )

    with pytest.raises(ProviderError, match="private"):
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert transport.urls == []


@pytest.mark.asyncio
async def test_fetcher_maps_dns_failure_to_typed_network_error() -> None:
    async def failing_resolver(
        hostname: str, port: int
    ) -> tuple[ipaddress.IPv4Address, ...]:
        del hostname, port
        raise socket.gaierror("synthetic resolver detail")

    fetcher = HttpxFetcher(
        transport=SequenceTransport(),
        resolver=failing_resolver,
    )

    with pytest.raises(ProviderError) as error:
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "NETWORK"
    assert "synthetic" not in error.value.public_message
    assert error.value.usage is not None


@pytest.mark.asyncio
async def test_fetcher_maps_raw_protocol_failure_to_sanitized_invalid_response() -> None:
    fetcher = HttpxFetcher(
        transport=ProtocolFailureTransport(),  # type: ignore[arg-type]
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
    )

    with pytest.raises(ProviderError) as error:
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert "TOP-SECRET" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
async def test_redirect_reacquires_host_slot_for_each_hostname() -> None:
    transport = SequenceTransport(
        ResponseSpec(
            status=302,
            headers={"location": "https://b.public.test/final"},
            chunks=(),
            peer_ip="93.184.216.34",
        ),
        ResponseSpec(
            status=200,
            headers={"content-type": "application/pdf"},
            chunks=(b"%PDF-synthetic",),
            peer_ip="93.184.216.35",
        ),
    )
    host_slot = RecordingHostSlot()
    fetcher = HttpxFetcher(
        transport=transport,
        host_slot=host_slot,
        resolver=_resolver(
            {
                "a.public.test": ("93.184.216.34",),
                "b.public.test": ("93.184.216.35",),
            }
        ),
    )

    document = await fetcher.fetch(
        "https://a.public.test/start",
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert host_slot.hosts == ["a.public.test", "b.public.test"]
    assert str(document.final_url) == "https://b.public.test/final"
    assert document.body_bytes == b"%PDF-synthetic"
    assert transport.pinned_ips == ["93.184.216.34", "93.184.216.35"]


@pytest.mark.asyncio
async def test_fetcher_streams_exact_limit_and_closes_oversize_response() -> None:
    transport = SequenceTransport(
        ResponseSpec(
            status=200,
            headers={"content-type": "text/html; charset=utf-8"},
            chunks=(b"123", b"456"),
            peer_ip="93.184.216.34",
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
        max_body_bytes=5,
    )

    with pytest.raises(ProviderError, match="body|large") as error:
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert transport.streams[0].closed is True


@pytest.mark.asyncio
async def test_fetcher_accepts_body_at_exact_limit_and_closes_response() -> None:
    transport = SequenceTransport(
        ResponseSpec(
            status=200,
            headers={"content-type": "text/html", "content-length": "5"},
            chunks=(b"12", b"345"),
            peer_ip="93.184.216.34",
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
        max_body_bytes=5,
    )

    outcome = await fetcher.fetch_with_usage(
        "https://public.test/article",
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert outcome.value.body_bytes == b"12345"
    assert outcome.usage.pages == 1
    assert outcome.usage.retries == 0
    assert transport.streams[0].closed is True


@pytest.mark.asyncio
async def test_fetcher_rejects_unsupported_media_before_reading_body() -> None:
    transport = SequenceTransport(
        ResponseSpec(
            status=200,
            headers={"content-type": "application/json"},
            chunks=(b'{"secret":"unused"}',),
            peer_ip="93.184.216.34",
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
    )

    with pytest.raises(ProviderError) as error:
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "PARSE_UNSUPPORTED"
    assert transport.streams[0].closed is True


@pytest.mark.asyncio
async def test_fetcher_cancellation_closes_stream_promptly() -> None:
    token = CancellationToken()
    transport = SequenceTransport(
        ResponseSpec(
            status=200,
            headers={"content-type": "text/html"},
            chunks=(b"first", b"second"),
            peer_ip="93.184.216.34",
            after_chunk=lambda index: token.cancel() if index == 0 else None,
        )
    )
    fetcher = HttpxFetcher(
        transport=transport,
        resolver=_resolver({"public.test": ("93.184.216.34",)}),
    )

    with pytest.raises(OperationCancelled):
        await fetcher.fetch(
            "https://public.test/article",
            deadline=_deadline(),
            cancellation_token=token,
        )

    assert transport.streams[0].closed is True
