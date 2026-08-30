import time
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import NoReturn

import pytest
from pydantic import JsonValue

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ModelMessage,
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    ProviderError,
    RawDocument,
    SearchHit,
)
from deepresearch.providers.recording import (
    RecordingFetcher,
    RecordingModelProvider,
    RecordingSearchProvider,
    RecordingTextEmbedder,
    ReplayBundleWriter,
)
from deepresearch.providers.replay import (
    ReplayFetcher,
    ReplayModelProvider,
    ReplaySearchProvider,
    ReplayTextEmbedder,
)
from deepresearch.providers.replay_schema import ReplayBundle
from deepresearch.runtime import CancellationToken

SHA256 = "a" * 64


def _deadline() -> float:
    return time.monotonic() + 100.0


def _usage() -> ResourceUsage:
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


def _request() -> ModelRequest:
    return ModelRequest(
        model_id="fixture-model-v1",
        messages=(ModelMessage(role="user", content="Question?"),),
        temperature=Decimal(0),
        max_output_tokens=20,
        prompt_version="prompt-v1",
        system_prompt_hash=SHA256,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )


class FakeModel:
    provider_id = "record-model"

    def __init__(self) -> None:
        self.calls = 0

    async def complete(self, request: ModelRequest, *, deadline: float, cancellation_token: CancellationToken) -> ModelResult[str]:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        return ModelResult(
            output="answer",
            usage=_usage(),
            provider_id=self.provider_id,
            model_id=request.model_id,
            raw_response_artifact_id="artifact-model",
        )

    async def structured(self, request: ModelRequest, output_schema: type[object], *, deadline: float, cancellation_token: CancellationToken) -> object:
        raise AssertionError("not used")

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        del request, deadline
        self.calls += 1

        async def chunks() -> AsyncIterator[ModelStreamChunk]:
            cancellation_token.raise_if_cancelled()
            yield ModelStreamChunk(index=0, text_delta="ordered ")
            yield ModelStreamChunk(
                index=1,
                text_delta="stream",
                finish_reason="stop",
                final_usage=_usage(),
            )

        return chunks()


class FakeSearch:
    provider_id = "record-search"

    def __init__(self) -> None:
        self.calls = 0

    async def search(self, query: str, limit: int, filters: Mapping[str, JsonValue] | None, *, deadline: float, cancellation_token: CancellationToken) -> list[SearchHit]:
        del filters, deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        if query == "failure":
            raise ProviderError(
                code="RATE_LIMITED",
                provider=self.provider_id,
                operation="search",
                public_message="synthetic rate limit",
                retryable=True,
                retry_after=1.25,
                usage=_usage(),
            )
        return [SearchHit(url="https://example.com", title=query, snippet="hit", rank=1)][:limit]


class FakeFetcher:
    provider_id = "record-fetch"

    def __init__(self) -> None:
        self.calls = 0

    async def fetch(self, url: str, *, deadline: float, cancellation_token: CancellationToken) -> RawDocument:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        return RawDocument(
            requested_url=url,
            final_url=url,
            status=200,
            headers={"content-type": "text/plain", "set-cookie": "secret", "x-api-key": "secret"},
            content_type="text/plain",
            body_bytes=b"evidence",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


class FakeEmbedder:
    provider_id = "record-embed"
    model_id = "embed-v1"
    model_revision = "revision-1"
    snapshot_sha256 = "d" * 64

    def __init__(self) -> None:
        self.calls = 0

    async def embed(self, texts: Sequence[str], *, deadline: float, cancellation_token: CancellationToken) -> tuple[tuple[float, ...], ...]:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        return tuple((float(index), 1.0) for index, _ in enumerate(texts))


@pytest.mark.asyncio
async def test_recorded_model_search_fetch_and_embed_replay_without_delegates(tmp_path: Path) -> None:
    final_root = tmp_path / "bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="recording-run")
    model = FakeModel()
    search = FakeSearch()
    fetcher = FakeFetcher()
    embedder = FakeEmbedder()
    token = CancellationToken()

    model_result = await RecordingModelProvider(model, writer).complete(
        _request(), deadline=_deadline(), cancellation_token=token
    )
    search_result = await RecordingSearchProvider(search, writer).search(
        "multimodal agents", 5, {"language": "en"}, deadline=_deadline(), cancellation_token=token
    )
    fetch_result = await RecordingFetcher(fetcher, writer).fetch(
        "https://example.com/path?b=2&a=1", deadline=_deadline(), cancellation_token=token
    )
    embed_result = await RecordingTextEmbedder(embedder, writer).embed(
        ("one", "two"), deadline=_deadline(), cancellation_token=token
    )
    await writer.finalize()

    assert (model.calls, search.calls, fetcher.calls, embedder.calls) == (1, 1, 1, 1)
    private_text = (final_root / "documents.jsonl").read_text(encoding="utf-8").lower()
    assert "set-cookie" not in private_text
    assert "x-api-key" not in private_text
    bundle = ReplayBundle.load(final_root)
    replay_model = await ReplayModelProvider(bundle).complete(
        _request(), deadline=_deadline(), cancellation_token=token
    )
    replay_search = await ReplaySearchProvider(bundle).search(
        "multimodal agents", 5, {"language": "en"}, deadline=_deadline(), cancellation_token=token
    )
    replay_fetch = await ReplayFetcher(bundle).fetch(
        "https://example.com/path?a=1&b=2", deadline=_deadline(), cancellation_token=token
    )
    replay_embed = await ReplayTextEmbedder(bundle).embed(
        ("one", "two"), deadline=_deadline(), cancellation_token=token
    )

    assert replay_model == model_result
    assert replay_search == search_result
    assert replay_fetch == fetch_result.model_copy(update={"headers": {"content-type": "text/plain"}})
    assert replay_embed == embed_result
    assert sum(provider.live_calls for provider in (ReplayModelProvider(bundle), ReplaySearchProvider(bundle), ReplayFetcher(bundle), ReplayTextEmbedder(bundle))) == 0


@pytest.mark.asyncio
async def test_writer_abort_remains_safe_after_atomic_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="rename-failure")

    def fail_rename(source: Path, destination: Path) -> NoReturn:
        del source, destination
        raise OSError("synthetic rename failure")

    monkeypatch.setattr("deepresearch.providers.recording.os.rename", fail_rename)
    with pytest.raises(OSError, match="synthetic"):
        await writer.finalize()
    await writer.abort()

    assert not final_root.exists()
    assert not list(tmp_path.glob(".bundle.staging.*"))


@pytest.mark.asyncio
async def test_recorded_stream_replays_ordered_chunks_and_terminal_usage(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "stream-bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="stream-run")
    model = FakeModel()
    token = CancellationToken()
    recorded = tuple(
        [
            chunk
            async for chunk in RecordingModelProvider(model, writer).stream(
                _request(), deadline=_deadline(), cancellation_token=token
            )
        ]
    )
    await writer.finalize()

    replayed = tuple(
        [
            chunk
            async for chunk in ReplayModelProvider(ReplayBundle.load(final_root)).stream(
                _request(), deadline=_deadline(), cancellation_token=token
            )
        ]
    )

    assert replayed == recorded
    assert [chunk.index for chunk in replayed] == [0, 1]
    assert replayed[-1].final_usage == _usage()
    assert model.calls == 1


@pytest.mark.asyncio
async def test_recorded_failure_replays_all_public_error_fields_and_usage(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "failure-bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="failure-run")
    search = FakeSearch()
    token = CancellationToken()
    recorder = RecordingSearchProvider(search, writer)

    with pytest.raises(ProviderError) as live_error:
        await recorder.search(
            "failure", 1, None, deadline=_deadline(), cancellation_token=token
        )
    await writer.finalize()
    replay = ReplaySearchProvider(ReplayBundle.load(final_root))
    with pytest.raises(ProviderError) as replay_error:
        await replay.search(
            "failure", 1, None, deadline=_deadline(), cancellation_token=token
        )

    assert search.calls == 1
    assert replay_error.value.code == live_error.value.code
    assert replay_error.value.public_message == live_error.value.public_message
    assert replay_error.value.retryable == live_error.value.retryable
    assert replay_error.value.retry_after == live_error.value.retry_after
    assert replay_error.value.usage == live_error.value.usage


@pytest.mark.asyncio
async def test_recording_expired_deadline_never_invokes_delegate(tmp_path: Path) -> None:
    writer = ReplayBundleWriter.create(tmp_path / "deadline-bundle", run_id="deadline-run")
    search = FakeSearch()
    recorder = RecordingSearchProvider(search, writer)
    try:
        with pytest.raises(ProviderError) as error:
            await recorder.search(
                "too late",
                1,
                None,
                deadline=time.monotonic() - 1,
                cancellation_token=CancellationToken(),
            )
        assert error.value.code == "TIMEOUT"
        assert search.calls == 0
    finally:
        await writer.abort()
