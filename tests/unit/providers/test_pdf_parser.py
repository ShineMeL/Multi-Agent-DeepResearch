import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import pymupdf
import pytest

from deepresearch.providers import ProviderError, RawDocument
from deepresearch.providers.parsers.pdf import PdfParser
from deepresearch.retrieval import normalize_text
from deepresearch.runtime import CancellationToken

FIXTURE = Path(__file__).parents[2] / "fixtures" / "providers" / "paper.pdf"


def _deadline() -> float:
    return time.monotonic() + 10.0


def _document(body: bytes, *, content_type: str = "application/pdf") -> RawDocument:
    return RawDocument(
        requested_url="https://example.com/paper.pdf",
        final_url="https://example.com/paper.pdf",
        status=200,
        headers={"content-type": content_type},
        content_type=content_type,
        body_bytes=body,
        retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
    )


@pytest.mark.asyncio
async def test_pdf_parser_emits_zero_indexed_locator_and_exact_hashes() -> None:
    parser = PdfParser()

    parsed = await parser.parse(
        _document(FIXTURE.read_bytes()),
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert parser.supports("application/pdf") is True
    assert "Synthetic Offline Research Paper" in parsed.normalized_text
    assert parsed.normalized_text == normalize_text(parsed.normalized_text)
    assert parsed.blocks[0].locator.kind == "pdf"
    assert parsed.blocks[0].locator.page_index == 0
    assert parsed.blocks[0].locator.block_index == 0
    assert parsed.blocks[0].text_hash == hashlib.sha256(
        parsed.blocks[0].text.encode("utf-8")
    ).hexdigest()
    assert parsed.parsed_content_hash == hashlib.sha256(
        parsed.normalized_text.encode("utf-8")
    ).hexdigest()


@pytest.mark.asyncio
async def test_pdf_parser_distinguishes_malformed_from_readable_textless_pdf() -> None:
    parser = PdfParser()
    blank = pymupdf.open()
    blank.new_page()
    blank_bytes = blank.tobytes(garbage=4, deflate=True)
    blank.close()

    with pytest.raises(ProviderError) as textless:
        await parser.parse(
            _document(blank_bytes),
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    with pytest.raises(ProviderError) as malformed:
        await parser.parse(
            _document(b"%PDF-1.7\nnot a readable PDF"),
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert textless.value.code == "PARSE_UNSUPPORTED"
    assert textless.value.retryable is False
    assert malformed.value.code == "INVALID_RESPONSE"
    assert malformed.value.retryable is False


@pytest.mark.asyncio
async def test_pdf_parser_rejects_password_protected_pdf() -> None:
    parser = PdfParser()
    protected = pymupdf.open()
    page = protected.new_page()
    page.insert_text((72, 72), "Synthetic protected text")
    protected_bytes = protected.tobytes(
        encryption=pymupdf.PDF_ENCRYPT_AES_256,
        owner_pw="synthetic-owner",
        user_pw="synthetic-reader",
    )
    protected.close()

    with pytest.raises(ProviderError) as error:
        await parser.parse(
            _document(protected_bytes),
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "PARSE_UNSUPPORTED"
    assert error.value.retryable is False


@pytest.mark.asyncio
async def test_pdf_parser_rejects_unsupported_media_without_opening_pdf() -> None:
    with pytest.raises(ProviderError) as error:
        await PdfParser().parse(
            _document(b"plain", content_type="text/plain"),
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "PARSE_UNSUPPORTED"
