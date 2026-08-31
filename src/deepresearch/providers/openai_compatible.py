from __future__ import annotations

import contextlib
import hashlib
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from dataclasses import dataclass
from math import isfinite
from typing import TypeVar, cast
from urllib.parse import urlsplit

import httpx
from pydantic import JsonValue, SecretStr, TypeAdapter, ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    ProviderCallExecutor,
    ProviderCallPolicy,
    ProviderError,
    ProviderErrorCode,
    StructuredModelResult,
    ToolCall,
    validate_model_stream,
)
from deepresearch.runtime import CancellationToken

from .httpx_transport import await_with_controls, checkpoint

T = TypeVar("T")
R = TypeVar("R")


@dataclass(frozen=True)
class _DecodedResponse:
    payload: dict[str, object]
    raw_bytes: bytes
    usage: ResourceUsage


def _zero_usage() -> ResourceUsage:
    return ResourceUsage.zero().model_copy(update={"cost_usd": None})


def _integer(mapping: Mapping[str, object], name: str) -> int:
    value = mapping.get(name)
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _usage(payload: object) -> ResourceUsage:
    if not isinstance(payload, dict):
        raise TypeError("usage must be an object")
    values = cast("dict[str, object]", payload)
    prompt_tokens = _integer(values, "prompt_tokens")
    completion_tokens = _integer(values, "completion_tokens")
    total_tokens = values.get("total_tokens")
    if type(total_tokens) is not int or total_tokens != prompt_tokens + completion_tokens:
        raise ValueError("total_tokens does not match prompt and completion tokens")
    prompt_details = values.get("prompt_tokens_details", {})
    completion_details = values.get("completion_tokens_details", {})
    if not isinstance(prompt_details, dict) or not isinstance(completion_details, dict):
        raise TypeError("token details must be objects")
    prompt_values = cast("dict[str, object]", prompt_details)
    completion_values = cast("dict[str, object]", completion_details)
    cached_tokens = prompt_values.get("cached_tokens", 0)
    reasoning_tokens = completion_values.get("reasoning_tokens", 0)
    if type(cached_tokens) is not int or cached_tokens < 0:
        raise ValueError("cached_tokens must be a non-negative integer")
    if type(reasoning_tokens) is not int or not 0 <= reasoning_tokens <= completion_tokens:
        raise ValueError("reasoning_tokens must fit within completion_tokens")
    return ResourceUsage(
        input_tokens=prompt_tokens,
        output_tokens=completion_tokens - reasoning_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        total_tokens=prompt_tokens + completion_tokens,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0,
        cost_usd=None,
    )


class OpenAICompatibleModelProvider:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: SecretStr,
        provider_id: str = "openai-compatible",
        model_revision: str = "provider-managed",
        executor: ProviderCallExecutor | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        parsed = urlsplit(base_url)
        if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
            raise ValueError("base_url must be an absolute HTTP(S) URL")
        if parsed.username is not None or parsed.password is not None:
            raise ValueError("base_url must not contain credentials")
        if not provider_id or not model_revision:
            raise ValueError("provider_id and model_revision must not be empty")
        self.provider_id = provider_id
        self.model_revision = model_revision
        self._endpoint = f"{base_url.rstrip('/')}/chat/completions"
        self._api_key = api_key
        self._executor = executor or ProviderCallExecutor(
            policy=ProviderCallPolicy.defaults()
        )
        self._client = client or httpx.AsyncClient(
            follow_redirects=False,
            trust_env=False,
        )
        self._owns_client = client is None

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(provider_id={self.provider_id!r}, "
            f"model_revision={self.model_revision!r})"
        )

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    def _headers(self) -> dict[str, str]:
        return {
            "authorization": f"Bearer {self._api_key.get_secret_value()}",
            "content-type": "application/json",
        }

    @staticmethod
    def _request_payload(
        request: ModelRequest,
        *,
        stream: bool,
        response_format: dict[str, JsonValue] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "max_tokens": request.max_output_tokens,
            "messages": [message.model_dump(mode="json") for message in request.messages],
            "model": request.model_id,
            "stream": stream,
            "temperature": float(request.temperature),
        }
        if request.seed is not None:
            payload["seed"] = request.seed
        if request.tools:
            payload["tools"] = [dict(tool) for tool in request.tools]
        if stream:
            payload["stream_options"] = {"include_usage": True}
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    @staticmethod
    def _status_error(status: int, *, operation: str) -> ProviderError | None:
        if 200 <= status < 300:
            return None
        if status in {401, 403}:
            code = "AUTHENTICATION"
            retryable = False
        elif status == 429:
            code = "RATE_LIMITED"
            retryable = True
        elif status >= 500:
            code = "UPSTREAM_5XX"
            retryable = True
        else:
            code = "INVALID_REQUEST"
            retryable = False
        return ProviderError(
            code=cast("ProviderErrorCode", code),
            provider="openai-compatible",
            operation=operation,
            public_message=f"model provider returned HTTP {status}",
            retryable=retryable,
            usage=_zero_usage(),
        )

    async def _post_json(
        self,
        payload: dict[str, object],
        *,
        operation: str,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> _DecodedResponse:
        remaining = deadline - time.monotonic()
        response: httpx.Response | None = None
        failure: ProviderError | None = None
        try:
            response = await await_with_controls(
                self._client.post(
                    self._endpoint,
                    headers=self._headers(),
                    json=payload,
                    timeout=max(remaining, 0.001),
                ),
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation=operation,
            )
        except httpx.TimeoutException:
            failure = ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation=operation,
                public_message="model provider timed out",
                retryable=True,
                usage=_zero_usage(),
            )
        except httpx.HTTPError:
            failure = ProviderError(
                code="NETWORK",
                provider=self.provider_id,
                operation=operation,
                public_message="model provider network request failed",
                retryable=True,
                usage=_zero_usage(),
            )
        if failure is not None:
            raise failure
        if response is None:
            raise RuntimeError("model response state is unavailable")
        status_error = self._status_error(response.status_code, operation=operation)
        if status_error is not None:
            retry_after = response.headers.get("retry-after")
            parsed_retry: float | None = None
            if retry_after is not None:
                try:
                    parsed_retry = float(retry_after)
                except ValueError:
                    parsed_retry = None
            if parsed_retry is not None and isfinite(parsed_retry) and parsed_retry >= 0:
                status_error.retry_after = parsed_retry
            status_error.provider = self.provider_id
            raise status_error
        raw_bytes = response.content
        decoded: object | None = None
        decode_failed = False
        try:
            decoded = json.loads(raw_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError):
            decode_failed = True
        if decode_failed or not isinstance(decoded, dict):
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation=operation,
                public_message="model provider returned invalid JSON",
                retryable=False,
                usage=_zero_usage(),
            )
        usage_value: ResourceUsage | None = None
        usage_failed = False
        try:
            decoded_mapping = cast("dict[str, object]", decoded)
            usage_value = _usage(decoded_mapping.get("usage"))
        except (TypeError, ValueError, ValidationError):
            usage_failed = True
        if usage_failed or usage_value is None:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation=operation,
                public_message="model provider returned invalid usage",
                retryable=False,
                usage=_zero_usage(),
            )
        return _DecodedResponse(
            payload=cast("dict[str, object]", decoded),
            raw_bytes=raw_bytes,
            usage=usage_value,
        )

    @staticmethod
    def _content_and_tools(
        response: _DecodedResponse,
        *,
        provider_id: str,
        operation: str,
    ) -> tuple[str, tuple[ToolCall, ...]]:
        content: str | None = None
        tool_calls: list[ToolCall] = []
        invalid = False
        try:
            choices_value = response.payload["choices"]
            if not isinstance(choices_value, list):
                raise TypeError
            choices = cast("list[object]", choices_value)
            if len(choices) != 1:
                raise ValueError
            choice = choices[0]
            if not isinstance(choice, dict):
                raise TypeError
            choice_mapping = cast("dict[str, object]", choice)
            if not isinstance(choice_mapping.get("message"), dict):
                raise TypeError
            message = cast("dict[str, object]", choice_mapping["message"])
            value = message.get("content")
            if value is None:
                content = ""
            elif isinstance(value, str):
                content = value
            else:
                raise TypeError
            raw_tool_calls = message.get("tool_calls", [])
            if not isinstance(raw_tool_calls, list):
                raise TypeError
            for raw_call in cast("list[object]", raw_tool_calls):
                if not isinstance(raw_call, dict):
                    raise TypeError
                raw_call_mapping = cast("dict[str, object]", raw_call)
                if not isinstance(raw_call_mapping.get("function"), dict):
                    raise TypeError
                function = cast("dict[str, object]", raw_call_mapping["function"])
                arguments = function.get("arguments")
                parsed_arguments = json.loads(arguments) if isinstance(arguments, str) else None
                if (
                    not isinstance(raw_call_mapping.get("id"), str)
                    or not isinstance(function.get("name"), str)
                    or not isinstance(parsed_arguments, dict)
                ):
                    raise TypeError
                tool_calls.append(
                    ToolCall(
                        tool_call_id=cast("str", raw_call_mapping["id"]),
                        name=cast("str", function["name"]),
                        arguments=cast("dict[str, JsonValue]", parsed_arguments),
                    )
                )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError, ValidationError):
            invalid = True
        if invalid or content is None:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=provider_id,
                operation=operation,
                public_message="model provider returned an invalid output shape",
                retryable=False,
                usage=response.usage,
            )
        return content, tuple(tool_calls)

    async def _execute(
        self,
        operation: str,
        invoke: Callable[[float], Awaitable[R]],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[R, int, float]:
        checkpoint(
            deadline=deadline,
            cancellation_token=cancellation_token,
            provider_id=self.provider_id,
            operation=operation,
        )
        before = len(self._executor.attempts)
        started = time.monotonic()
        caught: ProviderError | None = None
        result: R | None = None
        try:
            result = await self._executor.call(
                "model",
                invoke,
                remaining_deadline=deadline,
            )
        except ProviderError as error:
            caught = error
        attempts = self._executor.attempts[before:]
        retries = sum(attempt.attempt_index > 0 for attempt in attempts)
        elapsed = max(0, time.monotonic() - started)
        if caught is not None:
            usage = caught.usage or _zero_usage()
            failure = ProviderError(
                code=caught.code,
                provider=self.provider_id,
                operation=operation,
                public_message=caught.public_message,
                retryable=caught.retryable,
                retry_after=caught.retry_after,
                usage=usage.model_copy(
                    update={"retries": retries, "wall_seconds": elapsed}
                ),
            )
            raise failure
        if result is None:
            raise RuntimeError("provider executor returned no result")
        return result, retries, elapsed

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        operation = "model.complete"

        async def invoke(call_deadline: float) -> ModelResult[str]:
            response = await self._post_json(
                self._request_payload(request, stream=False),
                operation=operation,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            )
            content, tool_calls = self._content_and_tools(
                response, provider_id=self.provider_id, operation=operation
            )
            return ModelResult(
                output=content,
                usage=response.usage,
                provider_id=self.provider_id,
                model_id=request.model_id,
                tool_calls=tool_calls,
                raw_response_artifact_id=(
                    f"sha256:{hashlib.sha256(response.raw_bytes).hexdigest()}"
                ),
            )

        result, retries, elapsed = await self._execute(
            operation,
            invoke,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return result.model_copy(
            update={
                "usage": result.usage.model_copy(
                    update={"retries": retries, "wall_seconds": elapsed}
                )
            }
        )

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[T],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[T]:
        operation = "model.structured"
        adapter = TypeAdapter(output_schema)
        schema = cast("dict[str, JsonValue]", adapter.json_schema())
        response_format: dict[str, JsonValue] = {
            "type": "json_schema",
            "json_schema": {
                "name": getattr(output_schema, "__name__", "structured_output"),
                "schema": schema,
                "strict": True,
            },
        }

        async def invoke(call_deadline: float) -> StructuredModelResult[T]:
            response = await self._post_json(
                self._request_payload(
                    request, stream=False, response_format=response_format
                ),
                operation=operation,
                deadline=call_deadline,
                cancellation_token=cancellation_token,
            )
            content, tool_calls = self._content_and_tools(
                response, provider_id=self.provider_id, operation=operation
            )
            output: T | None = None
            invalid = False
            try:
                output = adapter.validate_json(content)
            except (ValueError, ValidationError):
                invalid = True
            if invalid or output is None:
                raise ProviderError(
                    code="INVALID_RESPONSE",
                    provider=self.provider_id,
                    operation=operation,
                    public_message="model provider output did not match the requested schema",
                    retryable=False,
                    usage=response.usage,
                )
            return StructuredModelResult(
                output=output,
                usage=response.usage,
                provider_id=self.provider_id,
                model_id=request.model_id,
                tool_calls=tool_calls,
                raw_response_artifact_id=(
                    f"sha256:{hashlib.sha256(response.raw_bytes).hexdigest()}"
                ),
                output_schema_hash=request.output_schema_hash,
            )

        result, retries, elapsed = await self._execute(
            operation,
            invoke,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return result.model_copy(
            update={
                "usage": result.usage.model_copy(
                    update={"retries": retries, "wall_seconds": elapsed}
                )
            }
        )

    async def _stream_once(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[ModelStreamChunk, ...]:
        operation = "model.stream"
        remaining = deadline - time.monotonic()
        response_context = self._client.stream(
            "POST",
            self._endpoint,
            headers=self._headers(),
            json=self._request_payload(request, stream=True),
            timeout=max(remaining, 0.001),
        )
        response: httpx.Response | None = None
        chunks: list[ModelStreamChunk] = []
        final_usage: ResourceUsage | None = None
        finish_reason: str | None = None
        failure: ProviderError | None = None
        try:
            response = await await_with_controls(
                response_context.__aenter__(),
                deadline=deadline,
                cancellation_token=cancellation_token,
                provider_id=self.provider_id,
                operation=operation,
            )
            status_error = self._status_error(response.status_code, operation=operation)
            if status_error is not None:
                status_error.provider = self.provider_id
                raise status_error
            lines = response.aiter_lines()
            while True:
                try:
                    line = await await_with_controls(
                        anext(lines),
                        deadline=deadline,
                        cancellation_token=cancellation_token,
                        provider_id=self.provider_id,
                        operation=operation,
                    )
                except StopAsyncIteration:
                    break
                if not line or line.startswith(":"):
                    continue
                if not line.startswith("data:"):
                    raise ProviderError(
                        code="INVALID_RESPONSE",
                        provider=self.provider_id,
                        operation=operation,
                        public_message="model stream used invalid SSE framing",
                        retryable=False,
                    )
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                decoded: object | None = None
                try:
                    decoded = json.loads(data)
                except json.JSONDecodeError:
                    decoded = None
                if not isinstance(decoded, dict):
                    raise ProviderError(
                        code="INVALID_RESPONSE",
                        provider=self.provider_id,
                        operation=operation,
                        public_message="model stream contained invalid JSON",
                        retryable=False,
                    )
                decoded_mapping = cast("dict[str, object]", decoded)
                if "usage" in decoded_mapping:
                    try:
                        final_usage = _usage(decoded_mapping["usage"])
                    except (TypeError, ValueError, ValidationError):
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.provider_id,
                            operation=operation,
                            public_message="model stream contained invalid usage",
                            retryable=False,
                        ) from None
                choices_value = decoded_mapping.get("choices", [])
                if not isinstance(choices_value, list):
                    raise ProviderError(
                        code="INVALID_RESPONSE",
                        provider=self.provider_id,
                        operation=operation,
                        public_message="model stream choices were invalid",
                        retryable=False,
                    )
                for choice in cast("list[object]", choices_value):
                    if not isinstance(choice, dict):
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.provider_id,
                            operation=operation,
                            public_message="model stream delta was invalid",
                            retryable=False,
                        )
                    choice_mapping = cast("dict[str, object]", choice)
                    if not isinstance(choice_mapping.get("delta", {}), dict):
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.provider_id,
                            operation=operation,
                            public_message="model stream delta was invalid",
                            retryable=False,
                        )
                    delta = cast(
                        "dict[str, object]", choice_mapping.get("delta", {})
                    )
                    text_delta = delta.get("content", "")
                    if text_delta is None:
                        text_delta = ""
                    if not isinstance(text_delta, str):
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.provider_id,
                            operation=operation,
                            public_message="model stream text delta was invalid",
                            retryable=False,
                        )
                    tool_delta = delta.get("tool_calls")
                    if tool_delta is not None and not isinstance(tool_delta, (dict, list)):
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.provider_id,
                            operation=operation,
                            public_message="model stream tool delta was invalid",
                            retryable=False,
                        )
                    raw_finish = choice_mapping.get("finish_reason")
                    if raw_finish is not None:
                        if not isinstance(raw_finish, str):
                            raise ProviderError(
                                code="INVALID_RESPONSE",
                                provider=self.provider_id,
                                operation=operation,
                                public_message="model stream finish reason was invalid",
                                retryable=False,
                            )
                        finish_reason = raw_finish
                    if text_delta or tool_delta is not None:
                        chunks.append(
                            ModelStreamChunk(
                                index=len(chunks),
                                text_delta=text_delta,
                                tool_call_delta=(
                                    cast("dict[str, JsonValue]", {"calls": tool_delta})
                                    if tool_delta is not None
                                    else None
                                ),
                            )
                        )
        except httpx.TimeoutException:
            failure = ProviderError(
                code="TIMEOUT",
                provider=self.provider_id,
                operation=operation,
                public_message="model stream timed out",
                retryable=True,
            )
        except httpx.HTTPError:
            failure = ProviderError(
                code="NETWORK",
                provider=self.provider_id,
                operation=operation,
                public_message="model stream network request failed",
                retryable=True,
            )
        finally:
            if response is not None:
                with contextlib.suppress(Exception):
                    await response.aclose()
            with contextlib.suppress(Exception):
                await response_context.__aexit__(None, None, None)
        if failure is not None:
            raise failure
        if final_usage is None:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation=operation,
                public_message="model stream ended without terminal usage",
                retryable=False,
            )
        chunks.append(
            ModelStreamChunk(
                index=len(chunks),
                finish_reason=finish_reason or "stop",
                final_usage=final_usage,
            )
        )
        return validate_model_stream(chunks)

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        async def chunks() -> AsyncIterator[ModelStreamChunk]:
            operation = "model.stream"

            async def invoke(call_deadline: float) -> tuple[ModelStreamChunk, ...]:
                return await self._stream_once(
                    request,
                    deadline=call_deadline,
                    cancellation_token=cancellation_token,
                )

            result, retries, elapsed = await self._execute(
                operation,
                invoke,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            terminal = result[-1]
            if terminal.final_usage is None:
                raise RuntimeError("validated stream has no final usage")
            final = terminal.model_copy(
                update={
                    "final_usage": terminal.final_usage.model_copy(
                        update={"retries": retries, "wall_seconds": elapsed}
                    )
                }
            )
            for chunk in (*result[:-1], final):
                checkpoint(
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                    provider_id=self.provider_id,
                    operation=operation,
                )
                yield chunk

        return chunks()


__all__ = ["OpenAICompatibleModelProvider"]
