import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deepresearch.providers import ProviderError, RawDocument
from deepresearch.providers.parsers.html import HtmlParser
from deepresearch.retrieval import normalize_text
from deepresearch.runtime import CancellationToken, OperationCancelled

FIXTURE = Path(__file__).parents[2] / "fixtures" / "providers" / "article.html"


def _deadline() -> float:
    return time.monotonic() + 10.0


def _document(body: bytes, *, content_type: str = "text/html") -> RawDocument:
    return RawDocument(
        requested_url="https://example.com/article",
        final_url="https://example.com/article",
        status=200,
        headers={"content-type": content_type},
        content_type=content_type,
        body_bytes=body,
        retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_html_parser_emits_normalized_text_stable_locator_and_exact_hash() -> None:
    parser = HtmlParser()

    parsed = await parser.parse(
        _document(FIXTURE.read_bytes()),
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert parser.supports("text/html; charset=utf-8") is True
    assert parsed.title == "Synthetic Agent Study"
    assert "Agent systems coordinate bounded research tasks." in parsed.normalized_text
    assert "Navigation text" not in parsed.normalized_text
    assert parsed.normalized_text == normalize_text(parsed.normalized_text)
    assert parsed.blocks[0].locator.kind == "html"
    assert parsed.blocks[0].locator.start_char == 0
    assert parsed.blocks[0].locator.end_char == len(parsed.blocks[0].text)
    assert parsed.blocks[0].text_hash == hashlib.sha256(
        parsed.blocks[0].text.encode("utf-8")
    ).hexdigest()
    assert parsed.parsed_content_hash == hashlib.sha256(
        parsed.normalized_text.encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_html_parser_rejects_textless_and_unsupported_media() -> None:
    parser = HtmlParser()

    for document in (
        _document(b"<html><body></body></html>"),
        _document(b"plain", content_type="text/plain"),
    ):
        with pytest.raises(ProviderError) as error:
            await parser.parse(
                document,
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            )
        assert error.value.code == "PARSE_UNSUPPORTED"
        assert error.value.retryable is False


@pytest.mark.asyncio
async def test_html_parser_maps_extractor_failure_to_typed_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_extract(*args: object, **kwargs: object) -> str:
        del args, kwargs
        raise ValueError("synthetic parser diagnostic")

    monkeypatch.setattr("trafilatura.extract", fail_extract)

    with pytest.raises(ProviderError) as error:
        await HtmlParser().parse(
            _document(FIXTURE.read_bytes()),
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert "synthetic" not in error.value.public_message
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
async def test_html_parser_checks_cancellation_before_blocking_parse() -> None:
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        await HtmlParser().parse(
            _document(FIXTURE.read_bytes()),
            deadline=_deadline(),
            cancellation_token=token,
        )
