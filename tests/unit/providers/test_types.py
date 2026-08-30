from datetime import UTC, datetime
from decimal import Decimal
from hashlib import sha256
from math import inf, nan

import pytest
from pydantic import ValidationError

from deepresearch.domain import HtmlLocator, ResourceUsage
from deepresearch.providers.types import (
    ModelMessage,
    ModelRequest,
    ModelStreamChunk,
    ParsedBlock,
    ParsedDocument,
    RawDocument,
    SearchHit,
    validate_embeddings,
    validate_model_stream,
)

SHA256 = "a" * 64


def usage(*, tokens: int = 0) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=tokens,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=tokens,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0.0,
        cost_usd=Decimal(0),
    )


def test_provider_models_require_aware_timestamps_and_forbid_extra_fields() -> None:
    with pytest.raises(ValidationError, match="timezone"):
        SearchHit(
            url="https://example.com",
            title="Example",
            snippet="Snippet",
            rank=1,
            published_at=datetime(2026, 1, 1),  # noqa: DTZ001 - deliberately naive
        )
    with pytest.raises(ValidationError, match="timezone"):
        RawDocument(
            requested_url="https://example.com",
            final_url="https://example.com",
            status=200,
            headers={},
            content_type="text/plain",
            body_bytes=b"body",
            retrieved_at=datetime(2026, 1, 1),  # noqa: DTZ001 - deliberately naive
        )
    with pytest.raises(ValidationError, match="Extra inputs"):
        SearchHit.model_validate(
            {
                "url": "https://example.com",
                "title": "Example",
                "snippet": "Snippet",
                "rank": 1,
                "usage": usage(),
            }
        )


def test_raw_document_binary_body_round_trips_through_typed_json() -> None:
    document = RawDocument(
        requested_url="https://example.com/file.pdf",
        final_url="https://example.com/file.pdf",
        status=200,
        headers={"content-type": "application/pdf"},
        content_type="application/pdf",
        body_bytes=b"\xff\x00\xfe",
        retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
    )

    assert RawDocument.model_validate_json(document.model_dump_json()) == document


def test_parsed_block_checks_locator_bounds_and_text_hash() -> None:
    text = "evidence"
    block = ParsedBlock(
        block_id="block-1",
        text=text,
        locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=len(text)),
        text_hash=sha256(text.encode()).hexdigest(),
    )

    assert block.text == text
    with pytest.raises(ValidationError, match="text_hash"):
        block.model_copy(update={"text_hash": SHA256})
    with pytest.raises(ValidationError, match="locator"):
        block.model_copy(
            update={
                "locator": HtmlLocator(
                    paragraph_id="p-1", start_char=0, end_char=len(text) + 1
                )
            }
        )


def test_parsed_document_hashes_exact_normalized_text_and_has_only_blocks() -> None:
    text = "evidence"
    block = ParsedBlock(
        block_id="block-1",
        text=text,
        locator=HtmlLocator(paragraph_id="p-1", start_char=0, end_char=len(text)),
        text_hash=sha256(text.encode()).hexdigest(),
    )
    document = ParsedDocument(
        canonical_url="https://example.com",
        title="Example",
        authors=("Researcher",),
        published_at=datetime(2026, 1, 1, tzinfo=UTC),
        normalized_text=text,
        blocks=(block,),
        parser_id="html",
        parser_version="v1",
        parsed_content_hash=sha256(text.encode()).hexdigest(),
    )

    assert "spans" not in ParsedDocument.model_fields
    with pytest.raises(ValidationError, match="parsed_content_hash"):
        document.model_copy(update={"normalized_text": "changed"})


def test_model_request_requires_messages_and_lowercase_sha256_fields() -> None:
    payload: dict[str, object] = {
        "model_id": "model",
        "messages": (ModelMessage(role="user", content="Question?"),),
        "tools": (),
        "temperature": Decimal(0),
        "max_output_tokens": 100,
        "prompt_version": "v1",
        "system_prompt_hash": SHA256,
        "tool_schema_hash": "b" * 64,
        "output_schema_hash": "c" * 64,
    }
    request = ModelRequest.model_validate(payload)

    assert ModelRequest.model_validate_json(request.model_dump_json()) == request
    with pytest.raises(ValidationError, match="messages"):
        request.model_copy(update={"messages": ()})
    with pytest.raises(ValidationError, match="SHA-256"):
        request.model_copy(update={"system_prompt_hash": "A" * 64})


def test_provider_json_serialization_is_stable_for_nested_mappings() -> None:
    first = SearchHit(
        url="https://example.com",
        title="Example",
        snippet="Snippet",
        rank=1,
        provider_metadata={"z": 2, "a": {"y": 1, "b": 2}},
    )
    second = SearchHit(
        url="https://example.com",
        title="Example",
        snippet="Snippet",
        rank=1,
        provider_metadata={"a": {"b": 2, "y": 1}, "z": 2},
    )

    assert first.model_dump_json() == second.model_dump_json()
    with pytest.raises(TypeError, match="immutable"):
        first.provider_metadata["new"] = 3


def test_stream_validation_requires_contiguous_indexes_and_one_final_usage() -> None:
    chunks = (
        ModelStreamChunk(index=0, text_delta="answer"),
        ModelStreamChunk(index=1, finish_reason="stop", final_usage=usage(tokens=3)),
    )

    assert validate_model_stream(chunks) == chunks
    with pytest.raises(ValueError, match="contiguous"):
        validate_model_stream((chunks[1], chunks[0]))
    with pytest.raises(ValueError, match="final_usage"):
        validate_model_stream((chunks[0],))
    with pytest.raises(ValueError, match="final chunk"):
        validate_model_stream(
            (
                ModelStreamChunk(index=0, final_usage=usage()),
                ModelStreamChunk(index=1, finish_reason="stop"),
            )
        )


def test_stream_validation_allows_usage_only_terminal_chunk() -> None:
    chunks = (
        ModelStreamChunk(index=0, text_delta="answer"),
        ModelStreamChunk(index=1, finish_reason="stop"),
        ModelStreamChunk(index=2, final_usage=usage(tokens=3)),
    )

    assert validate_model_stream(chunks) == chunks


@pytest.mark.parametrize("bad", [nan, inf, -inf])
def test_embedding_validation_rejects_shape_mismatch_and_non_finite_values(
    bad: float,
) -> None:
    assert validate_embeddings(("a", "b"), ((1.0, 2.0), (3.0, 4.0))) == (
        (1.0, 2.0),
        (3.0, 4.0),
    )
    with pytest.raises(ValueError, match="result count"):
        validate_embeddings(("a", "b"), ((1.0, 2.0),))
    with pytest.raises(ValueError, match="dimension"):
        validate_embeddings(("a", "b"), ((1.0,), (2.0, 3.0)))
    with pytest.raises(ValueError, match="finite"):
        validate_embeddings(("a",), ((bad,),))
