from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, cast

import pymupdf

from deepresearch.domain import PdfLocator
from deepresearch.providers import ParsedBlock, ParsedDocument, ProviderError, RawDocument
from deepresearch.retrieval import normalize_text, sha256_text
from deepresearch.runtime import CancellationToken

from ..httpx_transport import await_with_controls, checkpoint


class _PdfPage(Protocol):
    def get_text(self, option: str, *, sort: bool) -> object: ...


class _PdfDocument(Protocol):
    needs_pass: bool
    page_count: int
    metadata: Mapping[str, object] | None

    def load_page(self, page_index: int) -> _PdfPage: ...

    def close(self) -> None: ...


class PdfParser:
    parser_id = "pymupdf-pdf"
    parser_version = "1.28"

    def supports(self, content_type: str) -> bool:
        return content_type.split(";", 1)[0].strip().casefold() == "application/pdf"

    @staticmethod
    def _unsupported(message: str) -> ProviderError:
        return ProviderError(
            code="PARSE_UNSUPPORTED",
            provider="pymupdf-pdf",
            operation="parse",
            public_message=message,
            retryable=False,
        )

    def _parse_sync(self, raw_document: RawDocument) -> ParsedDocument:
        document: _PdfDocument | None = None
        try:
            document = cast(
                "_PdfDocument",
                pymupdf.open(
                    stream=raw_document.body_bytes,
                    filetype="pdf",
                ),
            )
        except (pymupdf.EmptyFileError, pymupdf.FileDataError, RuntimeError):
            pass
        if document is None:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.parser_id,
                operation="parse",
                public_message="PDF bytes are malformed",
                retryable=False,
            )
        extraction_failure: ProviderError | None = None
        try:
            if document.needs_pass:
                raise self._unsupported("password-protected PDF is unsupported")
            if document.page_count == 0:
                raise self._unsupported("empty PDF is unsupported")
            blocks: list[ParsedBlock] = []
            document_parts: list[str] = []
            for page_index in range(document.page_count):
                page = document.load_page(page_index)
                raw_blocks = cast(
                    "Sequence[Sequence[Any]]", page.get_text("blocks", sort=True)
                )
                page_block_index = 0
                for raw_block in raw_blocks:
                    if len(raw_block) < 7 or raw_block[6] != 0:
                        continue
                    raw_text = raw_block[4]
                    if not isinstance(raw_text, str):
                        continue
                    normalized = normalize_text(raw_text)
                    if not normalized:
                        continue
                    blocks.append(
                        ParsedBlock(
                            block_id=f"pdf-page-{page_index}-block-{page_block_index}",
                            text=normalized,
                            locator=PdfLocator(
                                page_index=page_index,
                                block_index=page_block_index,
                                start_char=0,
                                end_char=len(normalized),
                            ),
                            text_hash=sha256_text(normalized),
                        )
                    )
                    document_parts.append(normalized)
                    page_block_index += 1
            normalized_document = normalize_text("\n\n".join(document_parts))
            if not blocks or not normalized_document:
                raise self._unsupported("textless PDF is unsupported")
            metadata = document.metadata or {}
            author_value = metadata.get("author", "")
            raw_author = author_value if isinstance(author_value, str) else ""
            authors = tuple(
                item.strip()
                for item in raw_author.replace(";", ",").split(",")
                if item.strip()
            )
            return ParsedDocument(
                canonical_url=raw_document.final_url,
                title=(
                    title_value
                    if isinstance((title_value := metadata.get("title", "")), str)
                    else ""
                ),
                authors=authors,
                normalized_text=normalized_document,
                blocks=tuple(blocks),
                parser_id=self.parser_id,
                parser_version=self.parser_version,
                parsed_content_hash=sha256_text(normalized_document),
            )
        except ProviderError:
            raise
        except (OSError, RuntimeError, TypeError, ValueError):
            extraction_failure = ProviderError(
                code="INVALID_RESPONSE",
                provider=self.parser_id,
                operation="parse",
                public_message="PDF extraction failed",
                retryable=False,
            )
        finally:
            document.close()
        raise extraction_failure

    async def parse(
        self,
        raw_document: RawDocument,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ParsedDocument:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.parser_id,
            operation="parse",
        )
        if not self.supports(raw_document.content_type):
            raise self._unsupported("PDF parser does not support this media type")
        parsed = await await_with_controls(
            asyncio.to_thread(self._parse_sync, raw_document),
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.parser_id,
            operation="parse",
        )
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.parser_id,
            operation="parse",
        )
        return parsed


__all__ = ["PdfParser"]
