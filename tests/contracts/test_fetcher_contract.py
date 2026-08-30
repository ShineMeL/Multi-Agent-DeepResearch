import asyncio
from collections.abc import Callable
from datetime import UTC, datetime

import pytest

from deepresearch.providers import Fetcher, ProviderError
from deepresearch.providers.types import RawDocument
from deepresearch.runtime import CancellationToken, OperationCancelled


class AwaitGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started.set()
        await self.release.wait()


class FakeFetcher:
    provider_id = "fake-fetcher"

    def __init__(self, *, gate: AwaitGate | None = None) -> None:
        self._gate = gate
        self.response_closed = False

    async def fetch(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> RawDocument:
        del deadline
        cancellation_token.raise_if_cancelled()
        if url.endswith("timeout"):
            raise ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation="fetch",
                public_message="fetch timed out",
                retryable=True,
            )
        if url.endswith("invalid"):
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="fetch",
                public_message="malformed upstream response",
                retryable=False,
            )
        try:
            if self._gate is not None:
                await self._gate.wait()
            else:
                await _yield_once()
            cancellation_token.raise_if_cancelled()
            return RawDocument(
                requested_url=url,
                final_url=url,
                status=200,
                headers={"content-type": "text/plain"},
                content_type="text/plain",
                body_bytes=b"evidence",
                retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
            )
        finally:
            self.response_closed = True


async def _yield_once() -> None:
    return None


class FetcherContract:
    fetcher: Fetcher
    fetcher_factory: Callable[[AwaitGate], Fetcher]
    response_closed: Callable[[Fetcher], bool]

    @pytest.mark.asyncio
    async def test_success_preserves_body_for_immediate_artifact_write(self) -> None:
        document = await self.fetcher.fetch(
            "https://example.com/doc",
            deadline=10.0,
            cancellation_token=CancellationToken(),
        )

        assert document.body_bytes == b"evidence"
        assert RawDocument.model_validate_json(document.model_dump_json()) == document

    @pytest.mark.asyncio
    async def test_timeout_invalid_response_and_cancellation_are_typed(self) -> None:
        with pytest.raises(ProviderError) as timeout:
            await self.fetcher.fetch(
                "https://example.com/timeout",
                deadline=10.0,
                cancellation_token=CancellationToken(),
            )
        assert timeout.value.code == "TIMEOUT"
        with pytest.raises(ProviderError) as invalid:
            await self.fetcher.fetch(
                "https://example.com/invalid",
                deadline=10.0,
                cancellation_token=CancellationToken(),
            )
        assert invalid.value.code == "INVALID_RESPONSE"
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            await self.fetcher.fetch(
                "https://example.com/doc", deadline=10.0, cancellation_token=token
            )

    @pytest.mark.asyncio
    async def test_cancellation_after_await_closes_fetch_response(self) -> None:
        gate = AwaitGate()
        fetcher = self.fetcher_factory(gate)
        token = CancellationToken()
        task = asyncio.create_task(
            fetcher.fetch(
                "https://example.com/doc", deadline=10.0, cancellation_token=token
            )
        )
        await gate.started.wait()
        token.cancel()
        gate.release.set()
        with pytest.raises(OperationCancelled):
            await task
        assert self.response_closed(fetcher) is True


class TestFakeFetcher(FetcherContract):
    fetcher = FakeFetcher()
    fetcher_factory = staticmethod(lambda gate: FakeFetcher(gate=gate))
    response_closed = staticmethod(lambda fetcher: fetcher.response_closed)
