import asyncio
import json
import time
from decimal import Decimal

import httpx
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


def _retrying_executor() -> ProviderCallExecutor:
    defaults = ProviderCallPolicy.defaults()
    return ProviderCallExecutor(
        policy=ProviderCallPolicy(
            default_timeout_seconds=defaults.default_timeout_seconds,
            max_retries=1,
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
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content},
                "finish_reason": "stop",
            }
        ],
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
async def test_overlapping_model_calls_measure_only_their_own_retry_attempts() -> None:
    retry_second_entered = asyncio.Event()
    retry_attempts = 0

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal retry_attempts
        payload = json.loads(request.content)
        content = payload["messages"][-1]["content"]
        if content == "retry":
            retry_attempts += 1
            if retry_attempts == 1:
                return httpx.Response(500, text="retry")
            retry_second_entered.set()
            return httpx.Response(200, json=_response("retried"))
        await retry_second_entered.wait()
        return httpx.Response(200, json=_response("clean"))

    async with httpx.AsyncClient(
        transport=httpx.MockTransport(handler), trust_env=False
    ) as client:
        provider = OpenAICompatibleModelProvider(
            base_url="https://model.test/v1",
            api_key=SecretStr("TOP-SECRET-MODEL"),
            provider_id="fixture-openai",
            model_revision="revision-1",
            executor=_retrying_executor(),
            client=client,
        )
        retry_request = _request().model_copy(
            update={"messages": (ModelMessage(role="user", content="retry"),)}
        )
        clean_request = _request().model_copy(
            update={"messages": (ModelMessage(role="user", content="clean"),)}
        )

        retried, clean = await asyncio.gather(
            provider.complete(
                retry_request,
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            ),
            provider.complete(
                clean_request,
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            ),
        )

    assert retried.usage.retries == 1
    assert clean.usage.retries == 0


@pytest.mark.asyncio
@respx.mock
async def test_complete_accepts_tool_call_with_null_content() -> None:
    payload = _response("unused")
    payload["choices"] = [
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "call-1",
                        "type": "function",
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
@pytest.mark.parametrize(
    "choice",
    (
        {"index": 0, "message": {"role": "assistant", "content": "answer"}},
        {
            "index": 0,
            "message": {"role": "user", "content": "answer"},
            "finish_reason": "stop",
        },
        {
            "index": 0,
            "message": {
                "role": "assistant",
                "content": None,
                "tool_calls": [
                    {
                        "id": "",
                        "type": "function",
                        "function": {"name": "", "arguments": "{}"},
                    }
                ],
            },
            "finish_reason": "tool_calls",
        },
        {
            "index": 0,
            "message": {"role": "assistant", "content": None},
            "finish_reason": "stop",
        },
    ),
)
async def test_openai_unary_rejects_nonterminal_or_invalid_assistant_choice(
    choice: dict[str, object],
) -> None:
    payload = _response("unused")
    payload["choices"] = [choice]
    respx.post("https://model.test/v1/chat/completions").respond(200, json=payload)

    with pytest.raises(ProviderError) as error:
        await _provider().complete(
            _request(), deadline=_deadline(), cancellation_token=CancellationToken()
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.usage is not None
    assert error.value.usage.total_tokens == 13
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
async def test_openai_stream_is_ordered_call_once_and_has_terminal_usage() -> None:
    route = respx.post("https://model.test/v1/chat/completions").respond(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant",'
            b'"content":"ordered "}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"stream"},'
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
@pytest.mark.parametrize(
    "content",
    (
        (
            b'data: {"choices":[{"index":0,"delta":{"content":"answer"},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
        ),
        (
            b'data: {"choices":[{"index":0,"delta":{"content":"one"}},'
            b'{"index":1,"delta":{"content":"two"},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
            b'data: [DONE]\n\n'
        ),
        (
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: [DONE]\n\n'
        ),
        (
            b'data: {"choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
            b'data: [DONE]\n\n'
        ),
        (
            b'data: {"choices":[{"index":0,"delta":{},'
            b'"finish_reason":"stop"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"content":"after"}}]}\n\n'
            b'data: [DONE]\n\n'
        ),
    ),
)
async def test_openai_stream_rejects_missing_or_out_of_order_terminal_events(
    content: bytes,
) -> None:
    respx.post("https://model.test/v1/chat/completions").respond(
        200, headers={"content-type": "text/event-stream"}, content=content
    )

    with pytest.raises(ProviderError) as error:
        tuple(
            [
                chunk
                async for chunk in _provider().stream(
                    _request(),
                    deadline=_deadline(),
                    cancellation_token=CancellationToken(),
                )
            ]
        )

    assert error.value.code == "INVALID_RESPONSE"
    if b'"after"' in content:
        assert error.value.usage is not None
        assert error.value.usage.total_tokens == 13
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
@pytest.mark.parametrize(
    "tool_fragment",
    (
        (
            b'{"index":0,"type":"function","function":{"name":"search",'
            b'"arguments":"{}"}}'
        ),
        (
            b'{"index":1,"id":"call-1","type":"function",'
            b'"function":{"name":"search","arguments":"{}"}}'
        ),
        (
            b'{"index":0,"id":"call-1","type":"function",'
            b'"function":{"name":"","arguments":"{}"}}'
        ),
        (
            b'{"index":0,"id":"call-1","type":"function",'
            b'"function":{"name":"search","arguments":"{}"}},'
            b'{"index":1,"id":"call-1","type":"function",'
            b'"function":{"name":"fetch","arguments":"{}"}}'
        ),
    ),
)
async def test_openai_stream_rejects_malformed_tool_fragments(
    tool_fragment: bytes,
) -> None:
    content = (
        b'data: {"choices":[{"index":0,"delta":{"tool_calls":['
        + tool_fragment
        + b']},"finish_reason":"tool_calls"}]}\n\n'
        b'data: {"choices":[],"usage":{"prompt_tokens":10,'
        b'"completion_tokens":3,"total_tokens":13}}\n\n'
        b'data: [DONE]\n\n'
    )
    respx.post("https://model.test/v1/chat/completions").respond(
        200, headers={"content-type": "text/event-stream"}, content=content
    )

    with pytest.raises(ProviderError) as error:
        tuple(
            [
                chunk
                async for chunk in _provider().stream(
                    _request(),
                    deadline=_deadline(),
                    cancellation_token=CancellationToken(),
                )
            ]
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.__cause__ is None
    assert error.value.__context__ is None


@pytest.mark.asyncio
@respx.mock
async def test_openai_stream_accepts_ordered_function_tool_fragments() -> None:
    respx.post("https://model.test/v1/chat/completions").respond(
        200,
        headers={"content-type": "text/event-stream"},
        content=(
            b'data: {"choices":[{"index":0,"delta":{"role":"assistant",'
            b'"tool_calls":[{"index":0,"id":"call-1","type":"function",'
            b'"function":{"name":"search","arguments":"{\\"query\\":'
            b'"}}]}}]}\n\n'
            b'data: {"choices":[{"index":0,"delta":{"tool_calls":'
            b'[{"index":0,"function":{"arguments":"\\"agents\\"}"}}]},'
            b'"finish_reason":"tool_calls"}]}\n\n'
            b'data: {"choices":[],"usage":{"prompt_tokens":10,'
            b'"completion_tokens":3,"total_tokens":13}}\n\n'
            b'data: [DONE]\n\n'
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

    assert [chunk.index for chunk in chunks] == list(range(len(chunks)))
    assert chunks[-1].finish_reason == "tool_calls"
    assert chunks[-1].final_usage is not None
    assert chunks[0].tool_call_delta is not None
    assert chunks[1].tool_call_delta is not None


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
