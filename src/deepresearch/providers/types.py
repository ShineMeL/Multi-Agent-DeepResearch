from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime
from decimal import Decimal
from hashlib import sha256
from math import isfinite
from typing import Annotated, Generic, Literal, TypeAlias, TypeVar, cast

from pydantic import (
    AnyHttpUrl,
    ConfigDict,
    Field,
    JsonValue,
    field_serializer,
    field_validator,
    model_validator,
)

from deepresearch.domain import Locator, ResourceUsage
from deepresearch.domain.locators import _DomainModel  # pyright: ignore[reportPrivateUsage]
from deepresearch.domain.research import (
    _canonical_json,  # pyright: ignore[reportPrivateUsage]
    _freeze_json_mapping,  # pyright: ignore[reportPrivateUsage]
    _freeze_mapping,  # pyright: ignore[reportPrivateUsage]
    _thaw_internal_json,  # pyright: ignore[reportPrivateUsage]
)

Deadline: TypeAlias = float  # noqa: UP040 - exact frozen public contract
T = TypeVar("T")


def _require_timezone(value: datetime | None) -> datetime | None:
    if value is not None and (value.tzinfo is None or value.utcoffset() is None):
        raise ValueError("datetime must be timezone-aware")
    return value


def _require_sha256(value: str) -> str:
    if len(value) != 64 or value != value.lower():
        raise ValueError("value must be a lowercase 64-character SHA-256")
    try:
        int(value, 16)
    except ValueError as error:
        raise ValueError("value must be a lowercase 64-character SHA-256") from error
    return value


class _ProviderModel(_DomainModel):
    model_config = ConfigDict(
        extra="forbid",
        frozen=True,
        allow_inf_nan=False,
        ser_json_bytes="base64",
        val_json_bytes="base64",
    )


class SearchHit(_ProviderModel):
    url: AnyHttpUrl
    title: str
    snippet: str
    rank: Annotated[int, Field(ge=1)]
    published_at: datetime | None = None
    provider_metadata: dict[str, JsonValue] = Field(default_factory=dict)

    _aware_published_at = field_validator("published_at")(_require_timezone)

    @field_validator("provider_metadata", mode="before")
    @classmethod
    def thaw_metadata(cls, value: object) -> object:
        return _thaw_internal_json(value)

    @field_validator("provider_metadata")
    @classmethod
    def freeze_metadata(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)

    @field_serializer("provider_metadata")
    def serialize_metadata(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {key: _canonical_json(value[key]) for key in sorted(value)}


class RawDocument(_ProviderModel):
    requested_url: AnyHttpUrl
    final_url: AnyHttpUrl
    status: Annotated[int, Field(ge=100, le=599)]
    headers: dict[str, str]
    content_type: str
    body_bytes: bytes
    retrieved_at: datetime

    _aware_retrieved_at = field_validator("retrieved_at")(_require_timezone)

    @field_validator("headers")
    @classmethod
    def freeze_headers(cls, value: dict[str, str]) -> dict[str, str]:
        return _freeze_mapping(value)

    @field_serializer("headers", when_used="json")
    def serialize_headers(self, value: dict[str, str]) -> dict[str, str]:
        return {key: value[key] for key in sorted(value)}


class ParsedBlock(_ProviderModel):
    block_id: str
    text: str
    locator: Locator
    text_hash: str

    _sha256_text_hash = field_validator("text_hash")(_require_sha256)

    @model_validator(mode="after")
    def validate_text_integrity(self) -> ParsedBlock:
        expected = sha256(self.text.encode("utf-8")).hexdigest()
        if self.text_hash != expected:
            raise ValueError("text_hash does not match text")
        if self.locator.end_char > len(self.text):
            raise ValueError("locator bounds exceed text length")
        return self


class ParsedDocument(_ProviderModel):
    canonical_url: AnyHttpUrl
    title: str
    authors: tuple[str, ...]
    published_at: datetime | None = None
    normalized_text: str
    blocks: tuple[ParsedBlock, ...]
    parser_id: str
    parser_version: str
    parsed_content_hash: str

    _aware_published_at = field_validator("published_at")(_require_timezone)
    _sha256_content_hash = field_validator("parsed_content_hash")(_require_sha256)

    @model_validator(mode="after")
    def validate_content_hash(self) -> ParsedDocument:
        expected = sha256(self.normalized_text.encode("utf-8")).hexdigest()
        if self.parsed_content_hash != expected:
            raise ValueError("parsed_content_hash does not match normalized_text")
        return self


class ModelMessage(_ProviderModel):
    role: Literal["system", "user", "assistant", "tool"]
    content: str
    tool_call_id: str | None = None


class ModelRequest(_ProviderModel):
    model_id: str
    messages: Annotated[tuple[ModelMessage, ...], Field(min_length=1)]
    tools: tuple[dict[str, JsonValue], ...] = ()
    temperature: Decimal
    seed: int | None = None
    max_output_tokens: Annotated[int, Field(gt=0)]
    prompt_version: str
    system_prompt_hash: str
    tool_schema_hash: str
    output_schema_hash: str

    _sha256_fields = field_validator(
        "system_prompt_hash", "tool_schema_hash", "output_schema_hash"
    )(_require_sha256)

    @field_validator("tools", mode="before")
    @classmethod
    def thaw_tools(cls, value: object) -> object:
        if isinstance(value, list):
            items = cast("list[object]", value)
            return tuple(_thaw_internal_json(item) for item in items)
        if isinstance(value, tuple):
            items = cast("tuple[object, ...]", value)
            return tuple(_thaw_internal_json(item) for item in items)
        return value

    @field_validator("tools")
    @classmethod
    def freeze_tools(
        cls, value: tuple[dict[str, JsonValue], ...]
    ) -> tuple[dict[str, JsonValue], ...]:
        return tuple(_freeze_json_mapping(item) for item in value)

    @field_serializer("tools")
    def serialize_tools(
        self, value: tuple[dict[str, JsonValue], ...]
    ) -> tuple[dict[str, JsonValue], ...]:
        return tuple(
            {key: _canonical_json(item[key]) for key in sorted(item)} for item in value
        )


class ToolCall(_ProviderModel):
    tool_call_id: str
    name: str
    arguments: dict[str, JsonValue]

    @field_validator("arguments", mode="before")
    @classmethod
    def thaw_arguments(cls, value: object) -> object:
        return _thaw_internal_json(value)

    @field_validator("arguments")
    @classmethod
    def freeze_arguments(cls, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return _freeze_json_mapping(value)

    @field_serializer("arguments")
    def serialize_arguments(self, value: dict[str, JsonValue]) -> dict[str, JsonValue]:
        return {key: _canonical_json(value[key]) for key in sorted(value)}


class ModelResult(_ProviderModel, Generic[T]):  # noqa: UP046 - exact public contract
    output: T
    usage: ResourceUsage
    provider_id: str
    model_id: str
    tool_calls: tuple[ToolCall, ...] = ()
    raw_response_artifact_id: str


class StructuredModelResult(
    ModelResult[T], Generic[T]  # noqa: UP046 - exact public contract
):
    output_schema_hash: str

    _sha256_output_schema = field_validator("output_schema_hash")(_require_sha256)


class ModelStreamChunk(_ProviderModel):
    index: Annotated[int, Field(ge=0)]
    text_delta: str = ""
    tool_call_delta: dict[str, JsonValue] | None = None
    finish_reason: str | None = None
    final_usage: ResourceUsage | None = None

    @field_validator("tool_call_delta", mode="before")
    @classmethod
    def thaw_tool_call_delta(cls, value: object) -> object:
        return _thaw_internal_json(value)

    @field_validator("tool_call_delta")
    @classmethod
    def freeze_tool_call_delta(
        cls, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        return None if value is None else _freeze_json_mapping(value)

    @field_serializer("tool_call_delta")
    def serialize_tool_call_delta(
        self, value: dict[str, JsonValue] | None
    ) -> dict[str, JsonValue] | None:
        if value is None:
            return None
        return {key: _canonical_json(value[key]) for key in sorted(value)}


def validate_model_stream(
    chunks: Sequence[ModelStreamChunk],
) -> tuple[ModelStreamChunk, ...]:
    """Validate and freeze a complete provider stream response."""
    frozen = tuple(chunks)
    if not frozen:
        raise ValueError("model stream must not be empty")
    if tuple(chunk.index for chunk in frozen) != tuple(range(len(frozen))):
        raise ValueError("model stream indexes must be sorted and contiguous from zero")
    usage_indexes = [index for index, chunk in enumerate(frozen) if chunk.final_usage is not None]
    if len(usage_indexes) != 1:
        raise ValueError("model stream must contain exactly one final_usage")
    if usage_indexes[0] != len(frozen) - 1:
        raise ValueError("final_usage must appear on the final chunk")
    return frozen


def validate_embeddings(
    texts: Sequence[str], embeddings: Sequence[Sequence[float]]
) -> tuple[tuple[float, ...], ...]:
    """Validate the shape and numeric integrity of an embedding response."""
    if not texts:
        raise ValueError("embedding request texts must not be empty")
    frozen = tuple(tuple(vector) for vector in embeddings)
    if len(frozen) != len(texts):
        raise ValueError("embedding result count must match input text count")
    if not frozen[0]:
        raise ValueError("embedding dimension must be greater than zero")
    dimension = len(frozen[0])
    if any(len(vector) != dimension for vector in frozen):
        raise ValueError("embedding dimension must be consistent")
    if any(not isfinite(value) for vector in frozen for value in vector):
        raise ValueError("embedding values must be finite")
    return frozen


__all__ = [
    "Deadline",
    "ModelMessage",
    "ModelRequest",
    "ModelResult",
    "ModelStreamChunk",
    "ParsedBlock",
    "ParsedDocument",
    "RawDocument",
    "SearchHit",
    "StructuredModelResult",
    "ToolCall",
    "validate_embeddings",
    "validate_model_stream",
]
