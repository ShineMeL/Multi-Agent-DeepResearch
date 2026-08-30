from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol, TypeVar, runtime_checkable

from pydantic import JsonValue

from deepresearch.domain import EvidenceSpan, RerankScore
from deepresearch.runtime.cancellation import CancellationToken

from .types import (
    Deadline,
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    ParsedDocument,
    RawDocument,
    SearchHit,
    StructuredModelResult,
)

T = TypeVar("T")


@runtime_checkable
class ModelProvider(Protocol):
    provider_id: str

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]: ...

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[T]: ...

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]: ...


@runtime_checkable
class SearchProvider(Protocol):
    provider_id: str

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]: ...


@runtime_checkable
class Parser(Protocol):
    parser_id: str
    parser_version: str

    def supports(self, content_type: str) -> bool: ...

    async def parse(
        self,
        raw_document: RawDocument,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> ParsedDocument: ...


@runtime_checkable
class Fetcher(Protocol):
    provider_id: str

    async def fetch(
        self,
        url: str,
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> RawDocument: ...


@runtime_checkable
class TextEmbedder(Protocol):
    provider_id: str
    model_id: str
    model_revision: str
    snapshot_sha256: str

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]: ...


@runtime_checkable
class Reranker(Protocol):
    reranker_id: str

    async def score(
        self,
        information_need: str,
        evidence_spans: Sequence[EvidenceSpan],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[RerankScore]: ...


__all__ = [
    "Fetcher",
    "ModelProvider",
    "Parser",
    "Reranker",
    "SearchProvider",
    "TextEmbedder",
]
