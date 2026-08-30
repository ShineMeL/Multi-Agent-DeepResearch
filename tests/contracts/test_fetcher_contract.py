from datetime import UTC, datetime

import pytest

from deepresearch.providers import Fetcher, ProviderError
from deepresearch.providers.types import RawDocument
from deepresearch.runtime import CancellationToken, OperationCancelled


class FakeFetcher:
    provider_id = "fake-fetcher"

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


async def _yield_once() -> None:
    return None


class FetcherContract:
    fetcher: Fetcher

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


class TestFakeFetcher(FetcherContract):
    fetcher = FakeFetcher()
