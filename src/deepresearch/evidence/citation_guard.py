from __future__ import annotations

import hashlib
import hmac
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol, cast

from deepresearch.domain import EvidenceSpan, HtmlLocator, PdfLocator, SourceDocument

from .graph import ClaimEvidenceGraph

_CITATION = re.compile(r"\[(E-[A-Za-z0-9_-]+)\]")
_CITATION_START = re.compile(r"\[E")


@dataclass(frozen=True)
class CitationGuardResult:
    valid: bool
    errors: tuple[str, ...]
    checked_citation_ids: tuple[str, ...]


class CitationMaterialResolver(Protocol):
    def raw_bytes_for_source(self, source_id: str) -> bytes: ...

    def normalized_document_text(self, source_id: str) -> str: ...

    def html_paragraph_text(self, source_id: str, paragraph_id: str) -> str: ...

    def pdf_block_text(
        self,
        source_id: str,
        page_index: int,
        block_index: int,
    ) -> str: ...


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _extract_citation_ids(report_markdown: str) -> tuple[tuple[str, ...], tuple[str, ...]]:
    citation_ids: list[str] = []
    seen: set[str] = set()
    errors: list[str] = []
    valid_starts = {match.start() for match in _CITATION.finditer(report_markdown)}
    for match in _CITATION_START.finditer(report_markdown):
        if match.start() in valid_starts:
            continue
        errors.append("malformed citation")
    for match in _CITATION.finditer(report_markdown):
        citation_id = match.group(1)
        if citation_id not in seen:
            seen.add(citation_id)
            citation_ids.append(citation_id)
    return tuple(citation_ids), tuple(dict.fromkeys(errors))


def _append_error(errors: list[str], message: str) -> None:
    if message not in errors:
        errors.append(message)


def _linked_evidence_ids(graph: ClaimEvidenceGraph) -> frozenset[str]:
    payload = graph.to_json()
    raw_links = payload.get("links", ())
    if not isinstance(raw_links, list):
        return frozenset()
    linked: set[str] = set()
    for raw_link in cast("list[object]", raw_links):
        if not isinstance(raw_link, dict):
            continue
        mapping = cast("dict[object, object]", raw_link)
        evidence_id = mapping.get("evidence_id")
        if isinstance(evidence_id, str):
            linked.add(evidence_id)
    return frozenset(linked)


class CitationGuard:
    def verify(
        self,
        report_markdown: str,
        graph: ClaimEvidenceGraph,
        evidence: Mapping[str, EvidenceSpan],
        sources: Mapping[str, SourceDocument],
        *,
        materials: CitationMaterialResolver,
    ) -> CitationGuardResult:
        citation_ids, parse_errors = _extract_citation_ids(report_markdown)
        errors = list(parse_errors)

        graph_result = graph.validate()
        for code in graph_result.error_codes:
            _append_error(errors, f"graph validation error: {code}")
        linked_evidence_ids = _linked_evidence_ids(graph)

        for citation_id in citation_ids:
            span = evidence.get(citation_id)
            if span is None:
                _append_error(errors, f"unknown citation: {citation_id}")
                continue
            if span.evidence_id != citation_id:
                _append_error(errors, f"evidence ID mismatch: {citation_id}")
                continue
            if citation_id not in linked_evidence_ids:
                _append_error(errors, f"citation has no graph link: {citation_id}")

            source = sources.get(span.source_id)
            if source is None:
                _append_error(errors, f"unknown source: {span.source_id}")
                continue
            if source.source_id != span.source_id:
                _append_error(errors, f"source ID mismatch: {span.source_id}")
                continue

            try:
                raw_bytes = materials.raw_bytes_for_source(span.source_id)
            except (LookupError, OSError, RuntimeError, TypeError, ValueError):
                _append_error(errors, f"raw material unavailable: {span.source_id}")
                continue
            if not isinstance(raw_bytes, bytes) or not hmac.compare_digest(  # pyright: ignore[reportUnnecessaryIsInstance]
                _sha256_bytes(raw_bytes), source.content_hash
            ):
                _append_error(errors, f"content hash mismatch: {span.source_id}")
                continue

            try:
                normalized_text = materials.normalized_document_text(span.source_id)
            except (LookupError, OSError, RuntimeError, TypeError, ValueError):
                _append_error(errors, f"parsed material unavailable: {span.source_id}")
                continue
            if not isinstance(normalized_text, str) or not hmac.compare_digest(  # pyright: ignore[reportUnnecessaryIsInstance]
                _sha256_text(normalized_text), source.parsed_content_hash
            ):
                _append_error(errors, f"parsed content hash mismatch: {span.source_id}")
                continue

            container = self._resolve_container(span, materials, errors)
            if container is None:
                continue
            locator = span.locator
            if not (0 <= locator.start_char < locator.end_char <= len(container)):
                _append_error(errors, f"locator out of bounds: {citation_id}")
                continue
            excerpt = container[locator.start_char : locator.end_char]
            if excerpt != span.excerpt or not hmac.compare_digest(
                _sha256_text(excerpt), span.excerpt_hash
            ):
                _append_error(errors, f"excerpt hash mismatch: {citation_id}")

        return CitationGuardResult(
            valid=not errors,
            errors=tuple(errors),
            checked_citation_ids=citation_ids,
        )

    @staticmethod
    def _resolve_container(
        span: EvidenceSpan,
        materials: CitationMaterialResolver,
        errors: list[str],
    ) -> str | None:
        locator = span.locator
        try:
            if isinstance(locator, HtmlLocator):
                container = materials.html_paragraph_text(
                    span.source_id, locator.paragraph_id
                )
            elif isinstance(locator, PdfLocator):  # pyright: ignore[reportUnnecessaryIsInstance]
                container = materials.pdf_block_text(
                    span.source_id, locator.page_index, locator.block_index
                )
            else:
                _append_error(errors, f"unsupported locator: {span.evidence_id}")
                return None
        except (LookupError, OSError, RuntimeError, TypeError, ValueError):
            _append_error(errors, f"locator material unavailable: {span.evidence_id}")
            return None
        if not isinstance(container, str):  # pyright: ignore[reportUnnecessaryIsInstance]
            _append_error(errors, f"locator material unavailable: {span.evidence_id}")
            return None
        return container


__all__ = [
    "CitationGuard",
    "CitationGuardResult",
    "CitationMaterialResolver",
]
