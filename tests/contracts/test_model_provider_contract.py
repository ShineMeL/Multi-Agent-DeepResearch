import asyncio
from collections.abc import AsyncIterator, Callable
from decimal import Decimal

import pytest
from pydantic import BaseModel, ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.providers import ModelProvider, ProviderError
from deepresearch.providers.types import (
    ModelMessage,
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    StructuredModelResult,
    validate_model_stream,
)
from deepresearch.runtime import CancellationToken, OperationCancelled

SHA256 = "a" * 64


class AwaitGate:
    def __init__(self) -> None:
        self.started = asyncio.Event()
        self.release = asyncio.Event()

    async def wait(self) -> None:
        self.started.set()
        await self.release.wait()


def usage() -> ResourceUsage:
    return ResourceUsage(
        input_tokens=2,
        output_tokens=1,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=3,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0.01,
        cost_usd=Decimal("0.001"),
    )


def request() -> ModelRequest:
    return ModelRequest(
        model_id="fake-model",
        messages=(ModelMessage(role="user", content="Question?"),),
        temperature=Decimal(0),
        max_output_tokens=20,
        prompt_version="v1",
        system_prompt_hash=SHA256,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )


class Answer(BaseModel):
    answer: str


class InvalidAnswer(BaseModel):
    count: int


class FakeModelProvider:
    provider_id = "fake-model-provider"

    def __init__(self, *, gate: AwaitGate | None = None) -> None:
        self._gate = gate
        self.io_closed = False
        self.stream_closed = False

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        del deadline
        cancellation_token.raise_if_cancelled()
        if request.model_id == "timeout":
            raise ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation="model.complete",
                public_message="model timed out",
                retryable=True,
            )
        try:
            await self._await_io()
            cancellation_token.raise_if_cancelled()
            return ModelResult(
                output="answer",
                usage=usage(),
                provider_id=self.provider_id,
                model_id=request.model_id,
                raw_response_artifact_id="artifact-1",
            )
        finally:
            self.io_closed = True

    async def structured[T: BaseModel](
        self,
        request: ModelRequest,
        output_schema: type[T],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[T]:
        del deadline
        cancellation_token.raise_if_cancelled()
        try:
            output = output_schema.model_validate({"answer": "structured"})
        except ValidationError as error:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="model.structured",
                public_message="invalid structured response",
                retryable=False,
            ) from error
        await self._await_io()
        cancellation_token.raise_if_cancelled()
        return StructuredModelResult(
            output=output,
            usage=usage(),
            provider_id=self.provider_id,
            model_id=request.model_id,
            raw_response_artifact_id="artifact-2",
            output_schema_hash=request.output_schema_hash,
        )

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        del request, deadline

        async def chunks() -> AsyncIterator[ModelStreamChunk]:
            try:
                cancellation_token.raise_if_cancelled()
                yield ModelStreamChunk(index=0, text_delta="answer")
                await self._await_io()
                cancellation_token.raise_if_cancelled()
                yield ModelStreamChunk(index=1, finish_reason="stop", final_usage=usage())
            finally:
                self.stream_closed = True

        return chunks()

    async def _await_io(self) -> None:
        if self._gate is not None:
            await self._gate.wait()
        else:
            await _yield_once()


async def _yield_once() -> None:
    return None


class ModelProviderContract:
    provider: ModelProvider
    provider_factory: Callable[[AwaitGate], ModelProvider]
    io_closed: Callable[[ModelProvider], bool]
    stream_closed: Callable[[ModelProvider], bool]

    @pytest.mark.asyncio
    async def test_complete_returns_typed_usage_with_stable_serialization(self) -> None:
        result = await self.provider.complete(
            request(), deadline=10.0, cancellation_token=CancellationToken()
        )

        assert result.output == "answer"
        assert result.usage.total_tokens == 3
        assert ModelResult[str].model_validate_json(result.model_dump_json()) == result

    @pytest.mark.asyncio
    async def test_structured_success_and_stream_final_usage(self) -> None:
        result = await self.provider.structured(
            request(), Answer, deadline=10.0, cancellation_token=CancellationToken()
        )
        chunks = tuple(
            [
                chunk
                async for chunk in self.provider.stream(
                    request(), deadline=10.0, cancellation_token=CancellationToken()
                )
            ]
        )

        assert result.output.answer == "structured"
        assert validate_model_stream(chunks)[-1].final_usage == usage()

    @pytest.mark.asyncio
    async def test_timeout_mapping_and_pre_call_cancellation(self) -> None:
        timeout_request = request().model_copy(update={"model_id": "timeout"})
        with pytest.raises(ProviderError) as timeout:
            await self.provider.complete(
                timeout_request, deadline=10.0, cancellation_token=CancellationToken()
            )
        assert timeout.value.code == "TIMEOUT"
        token = CancellationToken()
        token.cancel()
        with pytest.raises(OperationCancelled):
            await self.provider.complete(request(), deadline=10.0, cancellation_token=token)

    @pytest.mark.asyncio
    async def test_invalid_structured_response_maps_to_invalid_response(self) -> None:
        with pytest.raises(ProviderError) as invalid:
            await self.provider.structured(
                request(),
                InvalidAnswer,
                deadline=10.0,
                cancellation_token=CancellationToken(),
            )
        assert invalid.value.code == "INVALID_RESPONSE"

    @pytest.mark.asyncio
    async def test_cancellation_after_await_closes_model_response(self) -> None:
        gate = AwaitGate()
        provider = self.provider_factory(gate)
        token = CancellationToken()
        task = asyncio.create_task(
            provider.complete(request(), deadline=10.0, cancellation_token=token)
        )
        await gate.started.wait()

        token.cancel()
        gate.release.set()
        with pytest.raises(OperationCancelled):
            await task
        assert self.io_closed(provider) is True

    @pytest.mark.asyncio
    async def test_stream_cancellation_closes_the_upstream_iterator(self) -> None:
        gate = AwaitGate()
        provider = self.provider_factory(gate)
        token = CancellationToken()
        stream = provider.stream(request(), deadline=10.0, cancellation_token=token)

        assert (await anext(stream)).text_delta == "answer"
        pending_chunk = asyncio.create_task(anext(stream))
        await gate.started.wait()
        token.cancel()
        gate.release.set()
        with pytest.raises(OperationCancelled):
            await pending_chunk
        assert self.stream_closed(provider) is True


class TestFakeModelProvider(ModelProviderContract):
    provider = FakeModelProvider()
    provider_factory = staticmethod(lambda gate: FakeModelProvider(gate=gate))
    io_closed = staticmethod(lambda provider: provider.io_closed)
    stream_closed = staticmethod(lambda provider: provider.stream_closed)
