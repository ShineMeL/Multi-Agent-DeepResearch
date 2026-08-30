from __future__ import annotations

import asyncio
import base64
import binascii
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import TypeVar, cast

from pydantic import JsonValue, TypeAdapter, ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    ProviderError,
    ProviderErrorCode,
    RawDocument,
    SearchHit,
    StructuredModelResult,
    validate_embeddings,
    validate_model_stream,
)
from deepresearch.runtime import CancellationToken

from .replay_schema import (
    REPLAY_REQUEST_SCHEMA_VERSION,
    ReplayBundle,
    ReplayFailure,
    ReplayOperation,
    ReplayRequestKey,
    ReplaySuccess,
    canonical_request_sha256,
    embed_request_payload,
    fetch_request_payload,
    model_request_payload,
    search_request_payload,
)

T = TypeVar("T")


class _ReplayProvider:
    live_calls = 0

    def __init__(self, bundle: ReplayBundle, *, provider_kind: str) -> None:
        self._bundle = bundle
        self._provider_snapshot = bundle.provider_snapshot(provider_kind)
        self.provider_id = self._provider_snapshot.provider_id
        self.last_usage: ResourceUsage | None = None

    def _key(
        self,
        operation: ReplayOperation,
        request_payload: object,
        *,
        prompt_version: str | None = None,
    ) -> ReplayRequestKey:
        return ReplayRequestKey(
            operation=operation,
            provider_id=self.provider_id,
            request_sha256=canonical_request_sha256(request_payload),
            prompt_version=prompt_version,
            schema_version=REPLAY_REQUEST_SCHEMA_VERSION,
        )

    def _lookup(self, key: ReplayRequestKey) -> ReplaySuccess:
        record = self._bundle.lookup(key)
        self.last_usage = record.usage
        if isinstance(record.outcome, ReplayFailure):
            raise ProviderError(
                code=cast("ProviderErrorCode", record.outcome.code),
                provider=record.key.provider_id,
                operation=record.key.operation,
                public_message=record.outcome.public_message,
                retryable=record.outcome.retryable,
                retry_after=record.outcome.retry_after,
                usage=record.usage,
            )
        return record.outcome

    @staticmethod
    def _invalid_response(provider: str, operation: str, error: Exception) -> ProviderError:
        return ProviderError(
            code="INVALID_SNAPSHOT",
            provider=provider,
            operation=operation,
            public_message="recorded replay response is invalid",
            retryable=False,
        )


class ReplayModelProvider(_ReplayProvider):
    def __init__(self, bundle: ReplayBundle) -> None:
        super().__init__(bundle, provider_kind="model")

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        del deadline
        cancellation_token.raise_if_cancelled()
        key = self._key(
            "model.complete",
            model_request_payload(request),
            prompt_version=request.prompt_version,
        )
        outcome = self._lookup(key)
        await asyncio.sleep(0)
        cancellation_token.raise_if_cancelled()
        try:
            return ModelResult[str].model_validate(outcome.response)
        except ValidationError as error:
            raise self._invalid_response(self.provider_id, key.operation, error) from error

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[T]:
        del deadline
        cancellation_token.raise_if_cancelled()
        key = self._key(
            "model.structured",
            model_request_payload(request),
            prompt_version=request.prompt_version,
        )
        outcome = self._lookup(key)
        await asyncio.sleep(0)
        cancellation_token.raise_if_cancelled()
        try:
            if not isinstance(outcome.response, dict):
                raise TypeError("structured replay response must be an object")
            response: dict[str, object] = dict(outcome.response)
            response["output"] = TypeAdapter(output_schema).validate_python(
                response.get("output")
            )
            return StructuredModelResult[T].model_validate(response)
        except (TypeError, ValueError, ValidationError) as error:
            raise self._invalid_response(self.provider_id, key.operation, error) from error

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        del deadline

        async def replay_chunks() -> AsyncIterator[ModelStreamChunk]:
            cancellation_token.raise_if_cancelled()
            key = self._key(
                "model.stream",
                model_request_payload(request),
                prompt_version=request.prompt_version,
            )
            outcome = self._lookup(key)
            try:
                raw_chunks = cast("list[object] | tuple[object, ...]", outcome.response)
                chunks = validate_model_stream(
                    tuple(ModelStreamChunk.model_validate(item) for item in raw_chunks)
                )
            except (TypeError, ValueError, ValidationError) as error:
                raise self._invalid_response(self.provider_id, key.operation, error) from error
            for chunk in chunks:
                cancellation_token.raise_if_cancelled()
                await asyncio.sleep(0)
                cancellation_token.raise_if_cancelled()
                yield chunk

        return replay_chunks()


class ReplaySearchProvider(_ReplayProvider):
    def __init__(self, bundle: ReplayBundle) -> None:
        super().__init__(bundle, provider_kind="search")

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del deadline
        cancellation_token.raise_if_cancelled()
        key = self._key("search", search_request_payload(query, limit, filters))
        outcome = self._lookup(key)
        await asyncio.sleep(0)
        cancellation_token.raise_if_cancelled()
        try:
            items = cast("list[object] | tuple[object, ...]", outcome.response)
            return [SearchHit.model_validate(item) for item in items]
        except (TypeError, ValidationError) as error:
            raise self._invalid_response(self.provider_id, key.operation, error) from error


class ReplayFetcher(_ReplayProvider):
    def __init__(self, bundle: ReplayBundle) -> None:
        super().__init__(bundle, provider_kind="fetch")

    async def fetch(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> RawDocument:
        del deadline
        cancellation_token.raise_if_cancelled()
        key = self._key("fetch", fetch_request_payload(url))
        outcome = self._lookup(key)
        await asyncio.sleep(0)
        cancellation_token.raise_if_cancelled()
        try:
            response = dict(cast("dict[str, JsonValue]", outcome.response))
            encoded_body = response.pop("body_base64")
            if not isinstance(encoded_body, str):
                raise TypeError("body_base64 must be a string")
            response["body_bytes"] = cast(
                "JsonValue", base64.b64decode(encoded_body, validate=True)
            )
            return RawDocument.model_validate(response)
        except (binascii.Error, TypeError, ValueError, ValidationError) as error:
            raise self._invalid_response(self.provider_id, key.operation, error) from error


class ReplayTextEmbedder(_ReplayProvider):
    def __init__(self, bundle: ReplayBundle) -> None:
        super().__init__(bundle, provider_kind="embed")
        if (
            self._provider_snapshot.model_id is None
            or self._provider_snapshot.model_revision is None
            or self._provider_snapshot.snapshot_sha256 is None
        ):
            raise ProviderError(
                code="INVALID_SNAPSHOT",
                provider=self.provider_id,
                operation="embed",
                public_message="embedding replay metadata is incomplete",
                retryable=False,
            )
        self.model_id = self._provider_snapshot.model_id
        self.model_revision = self._provider_snapshot.model_revision
        self.snapshot_sha256 = self._provider_snapshot.snapshot_sha256

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        del deadline
        cancellation_token.raise_if_cancelled()
        frozen_texts = tuple(texts)
        key = self._key("embed", embed_request_payload(frozen_texts))
        outcome = self._lookup(key)
        await asyncio.sleep(0)
        cancellation_token.raise_if_cancelled()
        try:
            raw_vectors = cast("list[list[float]] | tuple[tuple[float, ...], ...]", outcome.response)
            return validate_embeddings(frozen_texts, raw_vectors)
        except (TypeError, ValueError) as error:
            raise self._invalid_response(self.provider_id, key.operation, error) from error


__all__ = [
    "ReplayFetcher",
    "ReplayModelProvider",
    "ReplaySearchProvider",
    "ReplayTextEmbedder",
]
