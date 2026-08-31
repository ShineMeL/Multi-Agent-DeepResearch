import json
import time
from decimal import Decimal

import pytest
import respx
from pydantic import BaseModel, SecretStr

from deepresearch.providers import (
    ModelMessage,
    ModelRequest,
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
)
from deepresearch.providers.openai_compatible import OpenAICompatibleModelProvider
from deepresearch.runtime import CancellationToken, OperationCancelled


class ResearchPlan(BaseModel):
    objective: str


def _deadline() -> float:
    return time.monotonic() + 10.0


def _request() -> ModelRequest:
    return ModelRequest(
        model_id="fixture-model",
        messages=(ModelMessage(role="user", content="make plan"),),
        temperature=Decimal(0),
        max_output_tokens=100,
        prompt_version="planner-v1",
        system_prompt_hash="a" * 64,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )


def _executor() -> ProviderCallExecutor:
    defaults = ProviderCallPolicy.defaults()
    return ProviderCallExecutor(
        policy=ProviderCallPolicy(
            default_timeout_seconds=defaults.default_timeout_seconds,
            max_retries=0,
            base_delay_seconds=0,
            max_delay_seconds=0,
            jitter_ratio=0,
        )
    )


def _provider() -> OpenAICompatibleModelProvider:
    return OpenAICompatibleModelProvider(
        base_url="https://model.test/v1",
        api_key=SecretStr("TOP-SECRET-MODEL"),
        provider_id="fixture-openai",
        model_revision="revision-1",
        executor=_executor(),
    )


def _response(content: str) -> dict[str, object]:
    return {
        "id": "response-1",
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 3,
            "total_tokens": 13,
        },
    }


@pytest.mark.asyncio
@respx.mock
async def test_complete_validates_response_usage_and_sends_secret_only_as_header() -> None:
    route = respx.post("https://model.test/v1/chat/completions").respond(
        200, json=_response("answer")
    )
    provider = _provider()

    result = await provider.complete(
        _request(), deadline=_deadline(), cancellation_token=CancellationToken()
    )

    assert result.output == "answer"
    assert result.usage.total_tokens == 13
    assert result.usage.retries == 0
    assert result.raw_response_artifact_id.startswith("sha256:")
    assert route.calls.last.request.headers["authorization"] == "Bearer TOP-SECRET-MODEL"
    assert "TOP-SECRET-MODEL" not in repr(provider)


@pytest.mark.asyncio
@respx.mock
async def test_complete_accepts_tool_call_with_null_content() -> None:
    payload = _response("unused")
    payload["choices"] = [
        {
            "message": {
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "function": {
                            "name": "search",
                            "arguments": '{"query":"agents"}',
                        },
                    }
                ],
            },
            "finish_reason": "tool_calls",
        }
    ]
    respx.post("https://model.test/v1/chat/completions").respond(200, json=payload)

    result = await _provider().complete(
        _request(), deadline=_deadline(), cancellation_token=CancellationToken()
    )

    assert result.output == ""
    assert result.tool_calls[0].tool_call_id == "call-1"
    assert result.tool_calls[0].name == "search"
    assert result.tool_calls[0].arguments == {"query": "agents"}


@pytest.mark.asyncio
@respx.mock
async def test_openai_structured_rejects_invalid_schema_with_measured_usage() -> None:
    respx.post("https://model.test/v1/chat/completions").respond(
        200, json=_response(json.dumps({"unexpected": True}))
    )

    with pytest.raises(ProviderError) as error:
        await _provider().structured(
            _request(),
            ResearchPlan,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.usage is not None
    assert error.value.usage.total_tokens == 13
    assert "TOP-SECRET" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
async def test_openai_maps_authentication_without_leaking_response_or_key() -> None:
    respx.post("https://model.test/v1/chat/completions").respond(
        401, text="TOP-SECRET-MODEL upstream diagnostic"
    )

    with pytest.raises(ProviderError) as error:
        await _provider().complete(
            _request(), deadline=_deadline(), cancellation_token=CancellationToken()
        )

    assert error.value.code == "AUTHENTICATION"
    assert "TOP-SECRET" not in str(error.value)
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
async def test_openai_rejects_malformed_usage_shape() -> None:
    payload = _response("answer")
    payload["usage"] = {"prompt_tokens": True, "completion_tokens": 3}
    respx.post("https://model.test/v1/chat/completions").respond(200, json=payload)

    with pytest.raises(ProviderError) as error:
        await _provider().complete(
            _request(), deadline=_deadline(), cancellation_token=CancellationToken()
        )

    assert error.value.code == "INVALID_RESPONSE"


@pytest.mark.asyncio
@respx.mock
async def test_openai_stream_is_ordered_call_once_and_has_terminal_usage() -> None:
    route = respx.post("https://model.test/v1/chat/completions").respond(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            b'data: {"choices":[{"delta":{"content":"ordered "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"stream"},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
            b"data: [DONE]\n\n"
        ),
    )

    chunks = tuple(
        [
            chunk
            async for chunk in _provider().stream(
                _request(),
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            )
        ]
    )

    assert "".join(chunk.text_delta for chunk in chunks) == "ordered stream"
    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[-1].finish_reason == "stop"
    assert chunks[-1].final_usage is not None
    assert chunks[-1].final_usage.total_tokens == 13
    assert route.call_count == 1


@pytest.mark.asyncio
@respx.mock
async def test_openai_cancelled_before_call_never_reaches_network() -> None:
    route = respx.post("https://model.test/v1/chat/completions").respond(
        200, json=_response("unused")
    )
    token = CancellationToken()
    token.cancel()

    with pytest.raises(OperationCancelled):
        await _provider().complete(
            _request(), deadline=_deadline(), cancellation_token=token
        )

    assert route.call_count == 0
