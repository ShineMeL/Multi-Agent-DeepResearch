from __future__ import annotations

import asyncio
import base64
import binascii
import hashlib
import math
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import cast

from pydantic import AnyHttpUrl, JsonValue

from benchmarks.datasets.models import FrozenEvidenceRecord
from deepresearch.domain import EvidenceSpan, HtmlLocator, PdfLocator, SourceDocument
from deepresearch.providers.errors import ProviderError
from deepresearch.providers.protocols import Fetcher, SearchProvider
from deepresearch.providers.types import Deadline, ParsedDocument, RawDocument, SearchHit
from deepresearch.retrieval import canonicalize_url
from deepresearch.runtime import CancellationToken

from .frozen_index import FrozenCorpusSnapshot


def _checkpoint(
    *,
    provider: str,
    operation: str,
    deadline: Deadline,
    cancellation_token: CancellationToken,
) -> None:
    cancellation_token.raise_if_cancelled()
    if not math.isfinite(deadline):
        raise ProviderError(
            code="INVALID_REQUEST",
            provider=provider,
            operation=operation,
            public_message="deadline must be finite",
            retryable=False,
        )
    if time.monotonic() >= deadline:
        raise ProviderError(
            code="TIMEOUT",
            provider=provider,
            operation=operation,
            public_message="frozen corpus deadline exceeded",
            retryable=False,
        )


def _invalid_request(message: str) -> ProviderError:
    return ProviderError(
        code="INVALID_REQUEST",
        provider="frozen-corpus",
        operation="search",
        public_message=message,
        retryable=False,
    )


def _invalid_snapshot(message: str) -> ProviderError:
    return ProviderError(
        code="INVALID_SNAPSHOT",
        provider="frozen-corpus",
        operation="materialize",
        public_message=message,
        retryable=False,
    )


def _parse_published_at_filter(value: JsonValue) -> datetime:
    if type(value) is not str or not value.strip():
        raise _invalid_request("published_at must be an ISO-8601 timestamp")
    normalized = f"{value[:-1]}+00:00" if value.endswith("Z") else value
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        raise _invalid_request("published_at must be an ISO-8601 timestamp") from None
    if parsed.tzinfo is None or parsed.utcoffset() is None:
        raise _invalid_request("published_at must include a timezone")
    return parsed


def _filter_values(value: object) -> dict[str, JsonValue]:
    if not isinstance(value, Mapping):
        raise _invalid_request("filters must be a mapping")
    raw = cast("Mapping[object, object]", value)
    if any(type(key) is not str for key in raw):
        raise _invalid_request("filter keys must be strings")
    return {cast("str", key): cast("JsonValue", item) for key, item in raw.items()}


class FrozenCorpusSearchProvider(SearchProvider):
    provider_id = "frozen-corpus"

    def __init__(self, snapshot: FrozenCorpusSnapshot) -> None:
        if type(snapshot) is not FrozenCorpusSnapshot:
            raise TypeError("snapshot must be a FrozenCorpusSnapshot")
        self.snapshot = snapshot

    @classmethod
    def from_snapshot(
        cls,
        snapshot_dir: Path,
        *,
        task_id: str,
    ) -> FrozenCorpusSearchProvider:
        return cls(FrozenCorpusSnapshot.load(snapshot_dir, task_id=task_id))

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        _checkpoint(
            provider=self.provider_id,
            operation="search",
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        if type(query) is not str or not query.strip():
            raise _invalid_request("query must be non-empty")
        if type(limit) is not int or limit <= 0:
            raise _invalid_request("limit must be positive")
        allowed_filters = {"language", "source_type", "published_at"}
        filter_values = {} if filters is None else _filter_values(filters)
        unknown = set(filter_values) - allowed_filters
        if unknown:
            raise _invalid_request(f"unknown filter: {min(unknown)}")
        published_filter = (
            _parse_published_at_filter(filter_values["published_at"])
            if "published_at" in filter_values
            else None
        )
        filtered = self.snapshot.index.search(query, max(limit, len(self.snapshot.records)))

        def accepts(record: FrozenEvidenceRecord) -> bool:
            language = filter_values.get("language")
            if language is not None and record.language != language:
                return False
            source_type = filter_values.get("source_type")
            if source_type is not None and record.source_type != source_type:
                return False
            return published_filter is None or (
                record.published_at is not None and record.published_at == published_filter
            )

        hits: list[SearchHit] = []
        for frozen_hit in filtered:
            _checkpoint(
                provider=self.provider_id,
                operation="search",
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            record = frozen_hit.record
            if not accepts(record):
                continue
            rank = len(hits) + 1
            hits.append(
                SearchHit(
                    url=record.canonical_url,
                    title=record.title,
                    snippet=record.excerpt,
                    rank=rank,
                    published_at=record.published_at,
                    provider_metadata={
                        "evidence_id": record.evidence_id,
                        "source_id": record.source_id,
                        "source_family_id": record.source_family_id,
                        "language": record.language,
                        "source_type": record.source_type,
                    },
                )
            )
            if len(hits) >= limit:
                break
        _checkpoint(
            provider=self.provider_id,
            operation="search",
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        await asyncio.sleep(0)
        _checkpoint(
            provider=self.provider_id,
            operation="search",
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return hits


class FrozenCorpusFetcher(Fetcher):
    provider_id = "frozen-corpus"

    def __init__(self, snapshot: FrozenCorpusSnapshot) -> None:
        if type(snapshot) is not FrozenCorpusSnapshot:
            raise TypeError("snapshot must be a FrozenCorpusSnapshot")
        self.snapshot = snapshot

    async def fetch(
        self,
        url: str,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> RawDocument:
        _checkpoint(
            provider=self.provider_id,
            operation="fetch",
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        records = self.snapshot.records_for_url(url)
        if not records:
            raise ProviderError(
                code="REPLAY_MISS",
                provider=self.provider_id,
                operation="fetch",
                public_message="URL is not present in the frozen corpus",
                retryable=False,
            )
        record = records[0]
        try:
            body = base64.b64decode(record.raw_body_b64, validate=True)
        except (ValueError, binascii.Error):
            raise _invalid_snapshot("frozen raw body is invalid") from None
        _checkpoint(
            provider=self.provider_id,
            operation="fetch",
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        await asyncio.sleep(0)
        _checkpoint(
            provider=self.provider_id,
            operation="fetch",
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return RawDocument(
            requested_url=cast("AnyHttpUrl", canonicalize_url(url)),
            final_url=record.canonical_url,
            status=200,
            headers={"content-type": record.media_type},
            content_type=record.media_type,
            body_bytes=body,
            retrieved_at=record.retrieved_at,
        )


@dataclass(frozen=True, slots=True)
class FrozenMaterialization:
    source_documents: tuple[SourceDocument, ...]
    evidence_spans: tuple[EvidenceSpan, ...]


class FrozenCorpusMaterializer:
    def __init__(self, snapshot: FrozenCorpusSnapshot) -> None:
        if type(snapshot) is not FrozenCorpusSnapshot:
            raise TypeError("snapshot must be a FrozenCorpusSnapshot")
        self.snapshot = snapshot

    def materialize(
        self,
        *,
        selected_evidence_ids: Sequence[str],
        parsed_documents: Mapping[str, ParsedDocument],
        information_need_ids: tuple[str, ...],
    ) -> FrozenMaterialization:
        if not information_need_ids or any(
            type(need_id) is not str or not need_id.strip() for need_id in information_need_ids
        ):
            raise ValueError("information_need_ids must be non-empty")
        selected = tuple(selected_evidence_ids)
        if len(selected) != len(set(selected)):
            raise ValueError("selected_evidence_ids must be unique")
        spans: list[EvidenceSpan] = []
        source_documents: dict[str, SourceDocument] = {}
        for evidence_id in selected:
            record = self.snapshot.record(evidence_id)
            parsed = parsed_documents.get(record.source_id)
            if parsed is None:
                raise _invalid_snapshot(f"parsed document missing: {record.source_id}")
            if str(parsed.canonical_url) != str(record.canonical_url):
                raise _invalid_snapshot("parsed canonical URL disagrees with snapshot")
            if parsed.parsed_content_hash != record.parsed_content_hash:
                raise _invalid_snapshot("parsed content hash disagrees with snapshot")

            locator = record.locator
            container: str | None = None
            for block in parsed.blocks:
                block_locator = block.locator
                if isinstance(locator, HtmlLocator) and isinstance(block_locator, HtmlLocator):
                    if locator.paragraph_id == block_locator.paragraph_id:
                        container = block.text
                        break
                elif isinstance(locator, PdfLocator) and isinstance(block_locator, PdfLocator) and (
                    locator.page_index == block_locator.page_index
                    and locator.block_index == block_locator.block_index
                ):
                    container = block.text
                    break
            if container is None:
                raise _invalid_snapshot("snapshot locator is absent from parsed document")
            if not 0 <= locator.start_char < locator.end_char <= len(container):
                raise _invalid_snapshot("snapshot locator is out of bounds")
            excerpt = container[locator.start_char : locator.end_char]
            if excerpt != record.excerpt:
                raise _invalid_snapshot("snapshot excerpt disagrees with parsed document")
            if hashlib.sha256(excerpt.encode("utf-8")).hexdigest() != record.excerpt_hash:
                raise _invalid_snapshot("snapshot excerpt hash is invalid")

            source_documents[record.source_id] = SourceDocument(
                source_id=record.source_id,
                canonical_url=record.canonical_url,
                title=parsed.title or record.title,
                authors=parsed.authors,
                published_at=parsed.published_at,
                retrieved_at=record.retrieved_at,
                content_hash=record.content_hash,
                parsed_content_hash=parsed.parsed_content_hash,
                source_type=record.source_type,
                source_family_id=record.source_family_id,
                parser_version=parsed.parser_version,
            )
            spans.append(
                EvidenceSpan(
                    evidence_id=record.evidence_id,
                    source_id=record.source_id,
                    locator=record.locator,
                    excerpt=record.excerpt,
                    excerpt_hash=record.excerpt_hash,
                    language=record.language,
                    information_need_ids=information_need_ids,
                )
            )
        return FrozenMaterialization(
            source_documents=tuple(source_documents[key] for key in sorted(source_documents)),
            evidence_spans=tuple(spans),
        )


__all__ = [
    "FrozenCorpusFetcher",
    "FrozenCorpusMaterializer",
    "FrozenCorpusSearchProvider",
    "FrozenMaterialization",
]
