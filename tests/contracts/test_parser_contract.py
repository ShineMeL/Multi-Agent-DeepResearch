import asyncio
from collections.abc import Callable
from datetime import UTC, datetime
from hashlib import sha256

import pytest

from deepresearch.domain import HtmlLocator
from deepresearch.providers import Parser, ProviderError
from deepresearch.providers.types import ParsedBlock, ParsedDocument, RawDocument
from deepresearch.runtime import CancellationToken, OperationCancelled


class AwaitGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started.set()
        await self.release.wait()


def raw(content_type: str = "text/plain", body: bytes = b"evidence") -> RawDocument:
    return RawDocument(
        requested_url="https://example.com/doc",
        final_url="https://example.com/doc",
        status=200,
        headers={},
        content_type=content_type,
        body_bytes=body,
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )


class FakeParser:
    parser_id = "fake-parser"
    parser_version = "v1"

    def __init__(self, *, gate: AwaitGate | None = None) -> None:
        self._gate = gate

    def supports(self, content_type: str) -> bool:
        return content_type == "text/plain"

    async def parse(
        self,
        raw_document: RawDocument,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ParsedDocument:
        del deadline
        cancellation_token.raise_if_cancelled()
        if not self.supports(raw_document.content_type):
            raise ProviderError(
                code="PARSE_UNSUPPORTED",
                provider=self.parser_id,
                operation="parse",
                public_message="unsupported content type",
                retryable=False,
            )
        try:
            text = raw_document.body_bytes.decode("utf-8")
        except UnicodeDecodeError as error:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.parser_id,
                operation="parse",
                public_message="malformed supported content",
                retryable=False,
            ) from error
        if self._gate is not None:
            await self._gate.wait()
        else:
            await _yield_once()
        cancellation_token.raise_if_cancelled()
        block = ParsedBlock(
            block_id="block-1",
            text=text,
            locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=len(text)),
            text_hash=sha256(text.encode()).hexdigest(),
        )
        return ParsedDocument(
            canonical_url=raw_document.final_url,
            title="Document",
            authors=(),
            normalized_text=text,
            blocks=(block,),
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parsed_content_hash=sha256(text.encode()).hexdigest(),
        )


async def _yield_once() -> None:
    return None


class ParserContract:
    parser: Parser
    parser_factory: Callable[[AwaitGate], Parser]

    @pytest.mark.asyncio
    async def test_success_returns_stable_typed_document(self) -> None:
        parsed = await self.parser.parse(
            raw(), deadline=10.0, cancellation_token=CancellationToken()
        )

        assert parsed.normalized_text == "evidence"
        assert ParsedDocument.model_validate_json(parsed.model_dump_json()) == parsed

    @pytest.mark.asyncio
    async def test_unsupported_and_malformed_content_have_distinct_nonretryable_codes(
        self,
    ) -> None:
        with pytest.raises(ProviderError) as unsupported:
            await self.parser.parse(
                raw("application/zip"),
                deadline=10.0,
                cancellation_token=CancellationToken(),
            )
        assert unsupported.value.code == "PARSE_UNSUPPORTED"
        assert unsupported.value.retryable is False
        with pytest.raises(ProviderError) as malformed:
            await self.parser.parse(
                raw(body=b"\xff"),
                deadline=10.0,
                cancellation_token=CancellationToken(),
            )
        assert malformed.value.code == "INVALID_RESPONSE"
        assert malformed.value.retryable is False

    @pytest.mark.asyncio
    async def test_cancellation_fails_before_parse_work(self) -> None:
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            await self.parser.parse(raw(), deadline=10.0, cancellation_token=token)

    @pytest.mark.asyncio
    async def test_cancellation_after_await_is_enforced(self) -> None:
        gate = AwaitGate()
        parser = self.parser_factory(gate)
        token = CancellationToken()
        task = asyncio.create_task(
            parser.parse(raw(), deadline=10.0, cancellation_token=token)
        )
        await gate.started.wait()
        token.cancel()
        gate.release.set()
        with pytest.raises(OperationCancelled):
            await task


class TestFakeParser(ParserContract):
    parser = FakeParser()
    parser_factory = staticmethod(lambda gate: FakeParser(gate=gate))
