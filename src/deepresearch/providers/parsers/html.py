from __future__ import annotations

import asyncio
import json
from typing import Any, cast

import trafilatura

from deepresearch.domain import HtmlLocator
from deepresearch.providers import ParsedBlock, ParsedDocument, ProviderError, RawDocument
from deepresearch.retrieval import normalize_text, sha256_text
from deepresearch.runtime import CancellationToken

from ..httpx_transport import await_with_controls, checkpoint


class HtmlParser:
    parser_id = "trafilatura-html"
    parser_version = "2.2"

    def supports(self, content_type: str) -> bool:
        return content_type.split(";", 1)[0].strip().casefold() == "text/html"

    def _parse_sync(self, raw_document: RawDocument) -> ParsedDocument:
        extracted: str | None = None
        extraction_failed = False
        try:
            extracted = trafilatura.extract(
                raw_document.body_bytes,
                url=str(raw_document.final_url),
                output_format="json",
                with_metadata=True,
                include_comments=False,
                include_tables=False,
            )
        except (OSError, RuntimeError, TypeError, ValueError):
            extraction_failed = True
        if extraction_failed:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.parser_id,
                operation="parse",
                public_message="HTML extraction failed",
                retryable=False,
            )
        if extracted is None:
            raise ProviderError(
                code="PARSE_UNSUPPORTED",
                provider=self.parser_id,
                operation="parse",
                public_message="HTML document has no extractable main text",
                retryable=False,
            )
        decoded: object | None = None
        try:
            decoded = json.loads(extracted)
        except json.JSONDecodeError:
            decoded = None
        if not isinstance(decoded, dict):
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.parser_id,
                operation="parse",
                public_message="HTML parser returned invalid structured output",
                retryable=False,
            )
        metadata = cast("dict[str, Any]", decoded)
        raw_text = metadata.get("text")
        if not isinstance(raw_text, str) or not normalize_text(raw_text):
            raise ProviderError(
                code="PARSE_UNSUPPORTED",
                provider=self.parser_id,
                operation="parse",
                public_message="HTML document has no extractable main text",
                retryable=False,
            )
        normalized = normalize_text(raw_text)
        title = metadata.get("title")
        author = metadata.get("author")
        authors = (
            tuple(
                item.strip()
                for item in author.replace(";", ",").split(",")
                if item.strip()
            )
            if isinstance(author, str)
            else ()
        )
        block = ParsedBlock(
            block_id="html-main-0",
            text=normalized,
            locator=HtmlLocator(
                paragraph_id="main-0",
                start_char=0,
                end_char=len(normalized),
            ),
            text_hash=sha256_text(normalized),
        )
        return ParsedDocument(
            canonical_url=raw_document.final_url,
            title=title if isinstance(title, str) else "",
            authors=authors,
            normalized_text=normalized,
            blocks=(block,),
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parsed_content_hash=sha256_text(normalized),
        )

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
            raise ProviderError(
                code="PARSE_UNSUPPORTED",
                provider=self.parser_id,
                operation="parse",
                public_message="HTML parser does not support this media type",
                retryable=False,
            )
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


__all__ = ["HtmlParser"]
