import asyncio
import json
import os
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, NoReturn, TypeVar

import pytest
from pydantic import BaseModel, JsonValue, TypeAdapter

from deepresearch.domain import ResourceUsage
from deepresearch.providers import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResult,
    ModelStreamChunk,
    ProviderError,
    ProviderUsageResult,
    RawDocument,
    SearchHit,
    StructuredModelResult,
)
from deepresearch.providers import recording as recording_module
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
from deepresearch.runtime import CancellationToken, OperationCancelled

SHA256 = "a" * 64
T = TypeVar("T")


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
        self.calls += 1
        return StructuredModelResult(
            output=TypeAdapter(output_schema).validate_python({"answer": "structured"}),
            usage=_usage(),
            provider_id=self.provider_id,
            model_id=request.model_id,
            raw_response_artifact_id="artifact-structured",
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


class MessageFailureSearch(FakeSearch):
    def __init__(self, public_message: str) -> None:
        super().__init__()
        self.public_message = public_message

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del query, limit, filters, deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        raise ProviderError(
            code="INVALID_RESPONSE",
            provider=self.provider_id,
            operation="search",
            public_message=self.public_message,
            retryable=False,
            usage=_usage(),
        )


class ChainedMessageFailureSearch(MessageFailureSearch):
    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del query, limit, filters, deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        try:
            raise RuntimeError("TOP-SECRET internal cause")
        except RuntimeError as cause:
            raise ProviderError(
                code="INVALID_RESPONSE",
                provider=self.provider_id,
                operation="search",
                public_message=self.public_message,
                retryable=False,
                usage=_usage(),
            ) from cause


class SearchHitProvider(FakeSearch):
    def __init__(self, hit: SearchHit) -> None:
        super().__init__()
        self.hit = hit

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del query, filters, deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        return [self.hit][:limit]


class FakeFetcher:
    provider_id = "record-fetch"

    def __init__(
        self,
        *,
        final_url: str | None = None,
        requested_url: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.calls = 0
        self.final_url = final_url
        self.requested_url = requested_url
        self.headers = headers

    async def fetch(self, url: str, *, deadline: float, cancellation_token: CancellationToken) -> RawDocument:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        return RawDocument(
            requested_url=self.requested_url or url,
            final_url=self.final_url or url,
            status=200,
            headers=self.headers or {
                "content-type": "text/plain",
                "location": "https://example.com/next?api_key=TOP-SECRET",
                "set-cookie": "secret",
                "x-api-key": "secret",
            },
            content_type="text/plain",
            body_bytes=b"evidence",
            retrieved_at=datetime(2026, 1, 1, tzinfo=UTC),
        )


class UsageSearch(FakeSearch):
    async def search_with_usage(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ProviderUsageResult[list[SearchHit]]:
        value = await self.search(
            query,
            limit,
            filters,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return ProviderUsageResult(
            value=value,
            usage=ResourceUsage.zero().model_copy(
                update={"search_calls": 1, "retries": 1, "wall_seconds": 0.25}
            ),
        )


class UsageFetcher(FakeFetcher):
    async def fetch_with_usage(
        self,
        url: str,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ProviderUsageResult[RawDocument]:
        value = await self.fetch(
            url, deadline=deadline, cancellation_token=cancellation_token
        )
        return ProviderUsageResult(
            value=value,
            usage=ResourceUsage.zero().model_copy(
                update={"pages": 1, "retries": 1, "wall_seconds": 0.5}
            ),
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


class SlowFailureSearch(FakeSearch):
    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        del query, limit, filters, deadline, cancellation_token
        self.calls += 1
        await asyncio.sleep(0.05)
        raise ProviderError(
            code="INVALID_REQUEST",
            provider=self.provider_id,
            operation="search",
            public_message="synthetic delayed failure",
            retryable=False,
            usage=_usage(),
        )


class FailingStreamModel(FakeModel):
    def __init__(self, *, delay: float = 0) -> None:
        super().__init__()
        self.delay = delay

    def stream(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[ModelStreamChunk]:
        del request, deadline, cancellation_token
        self.calls += 1

        async def failure() -> AsyncIterator[ModelStreamChunk]:
            if self.delay:
                await asyncio.sleep(self.delay)
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="model.stream",
                public_message="synthetic stream failure",
                retryable=False,
                usage=_usage(),
            )
            if False:
                yield ModelStreamChunk(index=0, text_delta="unreachable")

        return failure()


def _recording_model(
    delegate: ModelProvider,
    writer: ReplayBundleWriter,
    *,
    revision: str = "revision-1",
) -> RecordingModelProvider:
    writer.configure_model_provider(
        provider_id=delegate.provider_id, model_revision=revision
    )
    return RecordingModelProvider(delegate, writer)


def test_windows_directory_fsync_uses_native_flush(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    if os.name != "nt":
        pytest.skip("Windows native directory flush boundary")
    flushed: list[Path] = []
    monkeypatch.setattr(
        recording_module,
        "_flush_windows_directory",
        lambda path: flushed.append(path),
        raising=False,
    )

    recording_module._fsync_directory(tmp_path)

    assert flushed == [tmp_path]


@pytest.mark.asyncio
async def test_recorded_model_search_fetch_and_embed_replay_without_delegates(tmp_path: Path) -> None:
    final_root = tmp_path / "bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="recording-run")
    model = FakeModel()
    search = FakeSearch()
    fetcher = FakeFetcher()
    embedder = FakeEmbedder()
    token = CancellationToken()

    assert isinstance(model, ModelProvider)
    model_result = await _recording_model(model, writer).complete(
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
    assert "location" not in private_text
    assert "top-secret" not in private_text
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
async def test_recording_preserves_authoritative_search_and_fetch_retry_usage(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "usage-bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="usage-run")
    token = CancellationToken()

    search_result = await RecordingSearchProvider(UsageSearch(), writer).search(
        "usage",
        1,
        None,
        deadline=_deadline(),
        cancellation_token=token,
    )
    fetch_result = await RecordingFetcher(UsageFetcher(), writer).fetch(
        "https://example.com/usage",
        deadline=_deadline(),
        cancellation_token=token,
    )
    await writer.finalize()

    search_line = json.loads(
        (final_root / "search.jsonl").read_text(encoding="utf-8")
    )
    fetch_line = json.loads(
        (final_root / "documents.jsonl").read_text(encoding="utf-8")
    )
    assert search_line["usage"]["retries"] == 1
    assert search_line["usage"]["search_calls"] == 1
    assert fetch_line["usage"]["retries"] == 1
    assert fetch_line["usage"]["pages"] == 1

    bundle = ReplayBundle.load(final_root)
    replay_search = ReplaySearchProvider(bundle)
    replay_fetch = ReplayFetcher(bundle)
    assert await replay_search.search(
        "usage", 1, None, deadline=_deadline(), cancellation_token=token
    ) == search_result
    assert await replay_fetch.fetch(
        "https://example.com/usage",
        deadline=_deadline(),
        cancellation_token=token,
    ) == fetch_result.model_copy(update={"headers": {"content-type": "text/plain"}})
    assert replay_search.last_usage is not None
    assert replay_search.last_usage.retries == 1
    assert replay_fetch.last_usage is not None
    assert replay_fetch.last_usage.retries == 1


@pytest.mark.asyncio
async def test_writer_abort_remains_safe_after_atomic_rename_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "bundle"
    writer = ReplayBundleWriter.create(final_root, run_id="rename-failure")

    def fail_rename(source: Path, destination: Path) -> NoReturn:
        del source, destination
        raise OSError("synthetic rename failure")

    monkeypatch.setattr(
        "deepresearch.providers.recording._atomic_rename_noreplace",
        fail_rename,
        raising=False,
    )
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
            async for chunk in _recording_model(model, writer).stream(
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
async def test_replay_stream_rechecks_deadline_before_each_yield(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "stream-deadline"
    writer = ReplayBundleWriter.create(root, run_id="stream-deadline")
    token = CancellationToken()
    recorded = _recording_model(FakeModel(), writer).stream(
        _request(), deadline=_deadline(), cancellation_token=token
    )
    _ = tuple([chunk async for chunk in recorded])
    await writer.finalize()

    clock = iter((1.0, 1.0, 1.0, 1.0, 11.0))
    monkeypatch.setattr(
        "deepresearch.providers.replay.time",
        SimpleNamespace(monotonic=lambda: next(clock)),
    )
    replayed = ReplayModelProvider(ReplayBundle.load(root)).stream(
        _request(), deadline=10.0, cancellation_token=token
    )

    assert (await anext(replayed)).index == 0
    with pytest.raises(ProviderError, match="deadline") as error:
        await anext(replayed)
    assert error.value.code == "TIMEOUT"


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


class Answer(BaseModel):
    answer: str


@pytest.mark.asyncio
async def test_recorded_structured_result_replays_typed_output(tmp_path: Path) -> None:
    writer = ReplayBundleWriter.create(tmp_path / "structured", run_id="structured-run")
    token = CancellationToken()
    recorded = await _recording_model(FakeModel(), writer).structured(
        _request(), Answer, deadline=_deadline(), cancellation_token=token
    )
    await writer.finalize()

    replayed = await ReplayModelProvider(ReplayBundle.load(tmp_path / "structured")).structured(
        _request(), Answer, deadline=_deadline(), cancellation_token=token
    )

    assert replayed == recorded
    assert replayed.output.answer == "structured"


@pytest.mark.asyncio
async def test_model_revision_is_required_persisted_and_changes_request_identity(
    tmp_path: Path,
) -> None:
    missing_writer = ReplayBundleWriter.create(
        tmp_path / "missing", run_id="missing-revision"
    )
    with pytest.raises(ValueError, match="revision"):
        RecordingModelProvider(FakeModel(), missing_writer)
    await missing_writer.abort()

    conflicting_writer = ReplayBundleWriter.create(
        tmp_path / "conflicting", run_id="conflicting-revision"
    )
    conflicting_writer.configure_model_provider(
        provider_id=FakeModel.provider_id, model_revision="revision-1"
    )
    with pytest.raises(ValueError, match="conflicting"):
        conflicting_writer.configure_model_provider(
            provider_id=FakeModel.provider_id, model_revision="revision-2"
        )
    await conflicting_writer.abort()

    request_hashes: list[str] = []
    for revision in ("revision-1", "revision-2"):
        root = tmp_path / revision
        writer = ReplayBundleWriter.create(root, run_id=revision)
        await _recording_model(FakeModel(), writer, revision=revision).complete(
            _request(), deadline=_deadline(), cancellation_token=CancellationToken()
        )
        await writer.finalize()
        record = json.loads((root / "model_responses.jsonl").read_text(encoding="utf-8"))
        request_hashes.append(record["key"]["request_sha256"])
        snapshot = json.loads((root / "snapshot.json").read_text(encoding="utf-8"))
        assert snapshot["providers"]["model"]["model_revision"] == revision

    assert request_hashes[0] != request_hashes[1]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_url",
    (
        "https://example.com/redirect?api_key=TOP-SECRET",
        "https://user:TOP-SECRET@example.com/redirect",
        "https://example.com/redirect#api_key=TOP-SECRET",
        "https://example.com/redirect?%2561pi_key=TOP-SECRET",
        "https://example.com/redirect?AuTh%6FrIzAtIoN=TOP-SECRET",
        "https://example.com/api_key/TOP-SECRET",
        "https://example.com/path/sk-proj-ABC123",
    ),
)
async def test_recording_fetch_fails_closed_for_secret_final_url(
    tmp_path: Path, secret_url: str
) -> None:
    writer = ReplayBundleWriter.create(tmp_path / "secret-final", run_id="secret-final")
    recorder = RecordingFetcher(FakeFetcher(final_url=secret_url), writer)

    try:
        with pytest.raises(ValueError) as error:
            await recorder.fetch(
                "https://example.com/start",
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            )
        assert "TOP-SECRET" not in str(error.value)
        assert error.value.__context__ is None
        assert error.value.__cause__ is None
    finally:
        await writer.abort()


@pytest.mark.asyncio
async def test_recording_fetch_fails_closed_for_secret_requested_url(
    tmp_path: Path,
) -> None:
    writer = ReplayBundleWriter.create(
        tmp_path / "secret-requested", run_id="secret-requested"
    )
    recorder = RecordingFetcher(
        FakeFetcher(
            requested_url="https://example.com/start?%2561PI_KEY=TOP-SECRET",
            final_url="https://example.com/final",
        ),
        writer,
    )

    try:
        with pytest.raises(ValueError) as error:
            await recorder.fetch(
                "https://example.com/start",
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            )
        assert "TOP-SECRET" not in str(error.value)
        assert error.value.__cause__ is None
        assert error.value.__context__ is None
    finally:
        await writer.abort()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("header", "value"),
    (
        ("etag", "Bearer TOP-SECRET"),
        ("content-language", "cookie%3DTOP-SECRET"),
        ("cache-control", "private, api%255fkey=TOP-SECRET"),
        ("etag", "sk-proj-ABC123"),
        ("content-language", "session_id=ABC123SECRET"),
    ),
)
async def test_recording_fetch_fails_closed_for_secrets_in_retained_headers(
    tmp_path: Path, header: str, value: str
) -> None:
    writer = ReplayBundleWriter.create(
        tmp_path / f"secret-{header}", run_id=f"secret-{header}"
    )
    recorder = RecordingFetcher(FakeFetcher(headers={header: value}), writer)

    try:
        with pytest.raises(ValueError) as error:
            await recorder.fetch(
                "https://example.com/start",
                deadline=_deadline(),
                cancellation_token=CancellationToken(),
            )
        assert "TOP-SECRET" not in str(error.value)
        assert error.value.__context__ is None
        assert error.value.__cause__ is None
    finally:
        await writer.abort()


@pytest.mark.asyncio
async def test_recording_fetch_retains_safe_response_metadata(tmp_path: Path) -> None:
    final_root = tmp_path / "safe-metadata"
    writer = ReplayBundleWriter.create(final_root, run_id="safe-metadata")
    headers = {
        "etag": 'W/"abc123"',
        "content-language": "en-US",
        "cache-control": "public, max-age=60",
    }

    await RecordingFetcher(
        FakeFetcher(
            final_url="https://example.com/page#section",
            requested_url="https://example.com/start#overview",
            headers=headers,
        ),
        writer,
    ).fetch(
        "https://example.com/start",
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    await writer.finalize()

    record = json.loads((final_root / "documents.jsonl").read_text(encoding="utf-8"))
    assert record["outcome"]["response"]["headers"] == headers
    assert record["outcome"]["response"]["final_url"].endswith("#section")
    assert record["outcome"]["response"]["requested_url"].endswith("#overview")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "secret_path",
    (
        "api_key=TOP-SECRET",
        "api%5fkey%3dTOP-SECRET",
        "AuTh%256FrIzAtIoN%253DTOP-SECRET",
    ),
)
async def test_finalized_bundle_excludes_credential_assignment_in_url_path(
    tmp_path: Path, secret_path: str
) -> None:
    final_root = tmp_path / f"path-assignment-{len(list(tmp_path.iterdir()))}"
    writer = ReplayBundleWriter.create(final_root, run_id="path-assignment")
    recorder = RecordingFetcher(
        FakeFetcher(final_url=f"https://example.com/{secret_path}"), writer
    )
    caught: ValueError | None = None

    try:
        await recorder.fetch(
            "https://example.com/start",
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    except ValueError as error:
        caught = error
    await writer.finalize()
    finalized = (final_root / "documents.jsonl").read_bytes()

    assert caught is not None
    assert b"top-secret" not in finalized.lower()
    assert finalized == b""


@pytest.mark.asyncio
async def test_finalized_bundle_allows_benign_security_words(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "benign-security-words"
    writer = ReplayBundleWriter.create(final_root, run_id="benign-security-words")
    benign_url = (
        "https://example.com/docs/token/usage"
        "?topic=authorization#token-guidance"
    )
    benign_message = "Read /docs/token/usage?topic=authorization#token-guidance"

    await RecordingFetcher(
        FakeFetcher(
            requested_url=benign_url,
            final_url=benign_url,
            headers={
                "content-type": "text/plain",
                "etag": 'W/"authorization-guide"',
            },
        ),
        writer,
    ).fetch(
        benign_url,
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    with pytest.raises(ProviderError):
        await RecordingSearchProvider(
            MessageFailureSearch(benign_message), writer
        ).search(
            "benign-message",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    await writer.finalize()

    document_bytes = (final_root / "documents.jsonl").read_bytes()
    search_bytes = (final_root / "search.jsonl").read_bytes()
    search_record = json.loads(search_bytes)
    assert benign_url.encode() in document_bytes
    assert b"authorization-guide" in document_bytes
    assert search_record["outcome"]["public_message"] == benign_message


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "unsafe_url",
    (
        "https://user:PASS@example.com/result",
        "https://example.com/result?api_key=TOP-SECRET",
        "https://example.com/api%255fkey%253dTOP-SECRET",
    ),
)
async def test_finalized_search_rejects_credential_bearing_hit_url(
    tmp_path: Path, unsafe_url: str
) -> None:
    final_root = tmp_path / f"unsafe-search-url-{len(list(tmp_path.iterdir()))}"
    writer = ReplayBundleWriter.create(final_root, run_id="unsafe-search-url")
    recorder = RecordingSearchProvider(
        SearchHitProvider(
            SearchHit(url=unsafe_url, title="unsafe", snippet="unsafe", rank=1)
        ),
        writer,
    )

    with pytest.raises(ValueError, match="credential"):
        await recorder.search(
            "unsafe",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    await writer.finalize()

    finalized = (final_root / "search.jsonl").read_bytes()
    assert finalized == b""
    assert b"top-secret" not in finalized.lower()
    assert b"user:pass" not in finalized.lower()


@pytest.mark.asyncio
async def test_finalized_search_rejects_credential_bearing_hit_metadata(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "unsafe-search-metadata"
    writer = ReplayBundleWriter.create(final_root, run_id="unsafe-search-metadata")
    recorder = RecordingSearchProvider(
        SearchHitProvider(
            SearchHit(
                url="https://example.com/result",
                title="unsafe",
                snippet="unsafe",
                rank=1,
                provider_metadata={
                    "nested": {"api%255fkey": "TOP-SECRET"},
                },
            )
        ),
        writer,
    )

    with pytest.raises(ValueError, match="credential"):
        await recorder.search(
            "unsafe",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    await writer.finalize()

    finalized = (final_root / "search.jsonl").read_bytes()
    assert finalized == b""
    assert b"top-secret" not in finalized.lower()


@pytest.mark.asyncio
async def test_finalized_search_allows_benign_security_words_in_hit_metadata(
    tmp_path: Path,
) -> None:
    final_root = tmp_path / "benign-search-metadata"
    writer = ReplayBundleWriter.create(final_root, run_id="benign-search-metadata")
    benign_url = (
        "https://example.com/docs/token/usage"
        "?topic=authorization#token-guidance"
    )
    hit = SearchHit(
        url=benign_url,
        title="Authorization guide",
        snippet="Token usage documentation",
        rank=1,
        provider_metadata={"topic": "authorization", "document": "token usage"},
    )

    result = await RecordingSearchProvider(
        SearchHitProvider(hit), writer
    ).search(
        "benign",
        1,
        None,
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    await writer.finalize()

    finalized = (final_root / "search.jsonl").read_bytes()
    assert result == [hit]
    assert benign_url.encode() in finalized
    assert b'"topic":"authorization"' in finalized


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "public_message",
    (
        "upstream rejected sk-proj-ABC123 at https://user:PASS@example.com",
        "upstream rejected api%255fkey%253DTOP-SECRET",
    ),
)
async def test_finalized_failure_redacts_encoded_opaque_and_userinfo_credentials(
    tmp_path: Path, public_message: str
) -> None:
    final_root = tmp_path / f"credential-message-{len(list(tmp_path.iterdir()))}"
    writer = ReplayBundleWriter.create(final_root, run_id="credential-message")
    recorder = RecordingSearchProvider(
        ChainedMessageFailureSearch(public_message), writer
    )

    with pytest.raises(ProviderError) as live_error:
        await recorder.search(
            "credential-message",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    await writer.finalize()
    finalized = (final_root / "search.jsonl").read_bytes()
    record = json.loads(finalized)

    assert b"top-secret" not in finalized.lower()
    assert b"sk-proj-abc123" not in finalized.lower()
    assert b"user:pass" not in finalized.lower()
    assert record["outcome"]["public_message"] == "[REDACTED]"
    assert live_error.value.public_message == "[REDACTED]"
    assert live_error.value.usage == _usage()
    assert live_error.value.__cause__ is None
    assert live_error.value.__context__ is None

    replay = ReplaySearchProvider(ReplayBundle.load(final_root))
    with pytest.raises(ProviderError) as replay_error:
        await replay.search(
            "credential-message",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )
    assert replay_error.value.__cause__ is None
    assert replay_error.value.__context__ is None
    assert replay_error.value.public_message == live_error.value.public_message
    assert replay_error.value.code == live_error.value.code
    assert replay_error.value.retryable == live_error.value.retryable
    assert replay_error.value.retry_after == live_error.value.retry_after
    assert replay_error.value.usage == live_error.value.usage


@pytest.mark.asyncio
async def test_recording_append_failure_never_retains_delegate_error_chain(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ReplayBundleWriter.create(
        tmp_path / "append-failure-chain", run_id="append-failure-chain"
    )

    async def fail_append(**kwargs: Any) -> NoReturn:
        del kwargs
        raise RuntimeError("synthetic safe append failure")

    monkeypatch.setattr(writer, "append", fail_append)
    recorder = RecordingSearchProvider(
        ChainedMessageFailureSearch(
            "upstream rejected api_key=TOP-SECRET-DELEGATE"
        ),
        writer,
    )

    with pytest.raises(RuntimeError, match="synthetic safe append failure") as caught:
        await recorder.search(
            "append-failure-chain",
            1,
            None,
            deadline=_deadline(),
            cancellation_token=CancellationToken(),
        )

    error = caught.value
    assert error.__cause__ is None
    assert error.__context__ is None
    pending: list[BaseException] = [error]
    seen: set[int] = set()
    rendered: list[str] = []
    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))
        rendered.extend((str(current), repr(current)))
        if current.__cause__ is not None:
            pending.append(current.__cause__)
        if current.__context__ is not None:
            pending.append(current.__context__)
    assert "top-secret" not in "\n".join(rendered).lower()

    await writer.abort()


@pytest.mark.asyncio
async def test_create_marker_failure_removes_exact_staging_and_allows_same_run_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "create-marker-failure"
    original_write = recording_module._write_fsynced

    def fail_marker(path: Path, payload: bytes) -> None:
        if path.name == ".replay-writer-owner.json":
            raise OSError("synthetic marker write failure")
        original_write(path, payload)

    monkeypatch.setattr(recording_module, "_write_fsynced", fail_marker)
    with pytest.raises(OSError, match="marker"):
        ReplayBundleWriter.create(final_root, run_id="same-run")
    assert not list(tmp_path.glob(".create-marker-failure.staging.*"))

    monkeypatch.undo()
    replacement = ReplayBundleWriter.create(final_root, run_id="same-run")
    await replacement.abort()


@pytest.mark.asyncio
async def test_create_cleanup_refuses_substituted_staging_but_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "create-substitution"
    substituted: Path | None = None

    def substitute_staging(path: Path, payload: bytes) -> None:
        nonlocal substituted
        del payload
        substituted = path.parent
        substituted.rmdir()
        substituted.mkdir()
        (substituted / "sentinel.txt").write_text("not-owned", encoding="utf-8")
        raise OSError("synthetic substituted marker failure")

    monkeypatch.setattr(recording_module, "_write_fsynced", substitute_staging)
    with pytest.raises(OSError, match="cleanup|marker|substitut"):
        ReplayBundleWriter.create(final_root, run_id="same-run")
    assert substituted is not None
    assert (substituted / "sentinel.txt").read_text(encoding="utf-8") == "not-owned"

    monkeypatch.undo()
    (substituted / "sentinel.txt").unlink()
    substituted.rmdir()
    replacement = ReplayBundleWriter.create(final_root, run_id="same-run")
    await replacement.abort()


@pytest.mark.asyncio
async def test_cleanup_root_rmdir_failure_keeps_owned_tombstone_and_frees_staging_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "cleanup-root-failure"
    writer = ReplayBundleWriter.create(final_root, run_id="same-run")
    original_write = recording_module._write_fsynced
    original_rmdir = Path.rmdir
    failed = False

    def fail_record_write(path: Path, payload: bytes) -> None:
        if path.name.endswith(".jsonl"):
            raise OSError("synthetic primary failure")
        original_write(path, payload)

    def fail_cleanup_root(path: Path) -> None:
        nonlocal failed
        if ".cleanup." in path.name and not failed:
            failed = True
            raise OSError("synthetic cleanup root failure")
        original_rmdir(path)

    monkeypatch.setattr(recording_module, "_write_fsynced", fail_record_write)
    monkeypatch.setattr(Path, "rmdir", fail_cleanup_root)
    with pytest.raises(OSError, match="cleanup"):
        await writer.finalize()

    assert not list(tmp_path.glob(".cleanup-root-failure.staging.*"))
    tombstones = list(tmp_path.glob(".cleanup-root-failure.cleanup.*"))
    assert len(tombstones) == 1
    assert (tombstones[0] / ".replay-writer-owner.json").is_file()

    monkeypatch.undo()
    replacement = ReplayBundleWriter.create(final_root, run_id="same-run")
    await replacement.abort()
    await writer.abort()
    await writer.abort()
    assert not tombstones[0].exists()


async def _leave_owned_tombstone_without_marker(
    final_root: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ReplayBundleWriter, Path]:
    writer = ReplayBundleWriter.create(final_root, run_id="same-run")
    original_write = recording_module._write_fsynced
    original_rmdir = Path.rmdir
    root_failed = False

    def fail_record_and_marker_restore(path: Path, payload: bytes) -> None:
        if path.name.endswith(".jsonl"):
            raise OSError("synthetic primary failure")
        if path.name == ".replay-writer-owner.json" and ".cleanup." in path.parent.name:
            raise OSError("synthetic marker restoration failure")
        original_write(path, payload)

    def fail_cleanup_root(path: Path) -> None:
        nonlocal root_failed
        if ".cleanup." in path.name and not root_failed:
            root_failed = True
            raise OSError("synthetic cleanup root failure")
        original_rmdir(path)

    def fail_marker_proof(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise OSError("synthetic marker restoration failure")

    monkeypatch.setattr(
        recording_module, "_write_fsynced", fail_record_and_marker_restore
    )
    monkeypatch.setattr(
        recording_module,
        "_create_repaired_marker_proof",
        fail_marker_proof,
        raising=False,
    )
    monkeypatch.setattr(Path, "rmdir", fail_cleanup_root)
    with pytest.raises(OSError, match="cleanup"):
        await writer.finalize()

    tombstones = list(final_root.parent.glob(f".{final_root.name}.cleanup.*"))
    assert len(tombstones) == 1
    assert not (tombstones[0] / ".replay-writer-owner.json").exists()
    return writer, tombstones[0]


@pytest.mark.asyncio
async def test_abort_repairs_owned_missing_tombstone_marker_and_retries_cleanup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "missing-cleanup-marker"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )

    replacement = ReplayBundleWriter.create(final_root, run_id="same-run")
    await replacement.abort()
    monkeypatch.undo()
    await writer.abort()
    await writer.abort()

    assert not tombstone.exists()
    second_replacement = ReplayBundleWriter.create(final_root, run_id="same-run")
    await second_replacement.abort()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="exercises Windows handle disposition")
async def test_repaired_marker_proof_never_adopts_post_creation_replacement(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "marker-creation-substitution"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    marker = tombstone / ".replay-writer-owner.json"
    moved_marker = tmp_path / "owned-creation-marker"
    marker_payload = (
        b'{"final_name":"marker-creation-substitution","run_id":"same-run",'
        b'"schema_version":"replay-bundle-v1"}'
    )
    original_fsync_directory = recording_module._fsync_directory
    substituted = False

    def substitute_at_legacy_identity_gap(path: Path) -> None:
        nonlocal substituted
        original_fsync_directory(path)
        if path == tombstone and marker.is_file() and not substituted:
            substituted = True
            marker.rename(moved_marker)
            marker.write_bytes(marker_payload)

    monkeypatch.setattr(
        recording_module, "_fsync_directory", substitute_at_legacy_identity_gap
    )

    with pytest.raises(ValueError, match="marker|identity|substitut|contents"):
        await writer.abort()

    assert substituted
    assert marker.is_file()
    assert tombstone.is_dir()
    with pytest.raises(ValueError, match="marker|identity|substitut|owned"):
        await writer.abort()
    assert marker.is_file()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="exercises POSIX anonymous marker")
async def test_posix_anonymous_marker_rejects_tombstone_path_substitution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "posix-directory-substitution"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    moved_owned = tombstone.with_name(f"{tombstone.name}.owned")
    sentinel = tombstone / "sentinel.txt"
    original_anonymous_marker = recording_module._open_posix_anonymous_marker
    substituted = False

    def substitute_after_anonymous_creation(*args: Any, **kwargs: Any) -> Any:
        nonlocal substituted
        created = original_anonymous_marker(*args, **kwargs)
        if not substituted:
            substituted = True
            tombstone.rename(moved_owned)
            tombstone.mkdir()
            sentinel.write_text("not-owned", encoding="utf-8")
        return created

    monkeypatch.setattr(
        recording_module,
        "_open_posix_anonymous_marker",
        substitute_after_anonymous_creation,
    )

    with pytest.raises(ValueError, match="identity|substitut|owned"):
        await writer.abort()

    assert sentinel.read_text(encoding="utf-8") == "not-owned"
    assert moved_owned.is_dir()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name == "nt", reason="exercises POSIX anonymous marker")
async def test_posix_anonymous_marker_creation_failure_preserves_empty_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "posix-anonymous-unsupported"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()

    def fail_anonymous_creation(*args: Any, **kwargs: Any) -> NoReturn:
        del args, kwargs
        raise OSError("synthetic anonymous marker unsupported")

    monkeypatch.setattr(
        recording_module, "_open_posix_anonymous_marker", fail_anonymous_creation
    )

    with pytest.raises(OSError, match="anonymous marker unsupported"):
        await writer.abort()

    assert tombstone.is_dir()
    assert tuple(tombstone.iterdir()) == ()


@pytest.mark.asyncio
async def test_repaired_tombstone_never_recursively_deletes_post_write_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "post-write-competitor"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    original_create_proof = recording_module._create_repaired_marker_proof
    sentinel = tombstone / "unowned.txt"

    def inject_after_marker_creation(*args: Any, **kwargs: Any) -> Any:
        proof = original_create_proof(*args, **kwargs)
        sentinel.write_text("unowned", encoding="utf-8")
        return proof

    monkeypatch.setattr(
        recording_module, "_create_repaired_marker_proof", inject_after_marker_creation
    )

    with pytest.raises(ValueError, match="contents|changed|owned"):
        await writer.abort()

    assert sentinel.read_text(encoding="utf-8") == "unowned"
    assert tombstone.is_dir()


@pytest.mark.asyncio
async def test_repaired_tombstone_rejects_pre_write_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "pre-write-competitor"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    original_create_proof = recording_module._create_repaired_marker_proof
    sentinel = tombstone / "unowned.txt"

    def inject_before_marker_creation(*args: Any, **kwargs: Any) -> Any:
        sentinel.write_text("unowned", encoding="utf-8")
        return original_create_proof(*args, **kwargs)

    monkeypatch.setattr(
        recording_module, "_create_repaired_marker_proof", inject_before_marker_creation
    )

    with pytest.raises(ValueError, match="contents|changed|owned"):
        await writer.abort()

    assert sentinel.read_text(encoding="utf-8") == "unowned"


@pytest.mark.asyncio
async def test_repaired_tombstone_rmdir_fails_closed_on_late_competitor(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "before-rmdir-competitor"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    original_rmdir = Path.rmdir
    sentinel = tombstone / "unowned.txt"
    injected = False

    def inject_before_rmdir(path: Path) -> None:
        nonlocal injected
        if path == tombstone and not injected:
            injected = True
            sentinel.write_text("unowned", encoding="utf-8")
        original_rmdir(path)

    monkeypatch.setattr(Path, "rmdir", inject_before_rmdir)

    with pytest.raises(OSError):
        await writer.abort()
    with pytest.raises(ValueError, match="contents|owned"):
        await writer.abort()

    assert sentinel.read_text(encoding="utf-8") == "unowned"
    assert tombstone.is_dir()


@pytest.mark.asyncio
@pytest.mark.skipif(os.name != "nt", reason="exercises Windows marker pathname")
async def test_repaired_tombstone_never_unlinks_marker_substituted_after_validation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "marker-unlink-substitution"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    marker = tombstone / ".replay-writer-owner.json"
    moved_marker = tmp_path / "owned-repaired-marker"
    marker_payload = (
        b'{"final_name":"marker-unlink-substitution","run_id":"same-run",'
        b'"schema_version":"replay-bundle-v1"}'
    )
    original_validate = writer._validate_repaired_tombstone
    substituted = False

    def substitute_after_final_validation(
        staging: Path,
        marker_path: Path,
        proof: Any,
    ) -> None:
        nonlocal substituted
        original_validate(staging, marker_path, proof)
        if not proof.marker_removed and not substituted:
            substituted = True
            marker_path.rename(moved_marker)
            marker_path.write_bytes(marker_payload)

    monkeypatch.setattr(
        writer, "_validate_repaired_tombstone", substitute_after_final_validation
    )

    with pytest.raises((OSError, ValueError), match="marker|identity|substitut"):
        await writer.abort()

    assert marker.is_file()
    assert tombstone.is_dir()
    with pytest.raises(ValueError, match="marker|contents|owned"):
        await writer.abort()
    assert marker.is_file()


@pytest.mark.asyncio
async def test_abort_never_repairs_or_deletes_substituted_cleanup_tombstone(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "substituted-cleanup-marker"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    tombstone.rmdir()
    tombstone.mkdir()
    sentinel = tombstone / "sentinel.txt"
    sentinel.write_text("not-owned", encoding="utf-8")

    for _ in range(2):
        with pytest.raises(ValueError, match="substitut|owned"):
            await writer.abort()

    assert sentinel.read_text(encoding="utf-8") == "not-owned"


@pytest.mark.asyncio
async def test_abort_never_overwrites_partial_cleanup_tombstone_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "partial-cleanup-marker"
    writer, tombstone = await _leave_owned_tombstone_without_marker(
        final_root, monkeypatch
    )
    monkeypatch.undo()
    marker = tombstone / ".replay-writer-owner.json"
    marker.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="marker|owned"):
        await writer.abort()

    assert marker.read_text(encoding="utf-8") == "{}"


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("unary", "stream"))
async def test_recorded_failure_rechecks_cancellation_after_append(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    kind: str,
) -> None:
    final_root = tmp_path / f"cancel-failure-{kind}"
    writer = ReplayBundleWriter.create(final_root, run_id=f"cancel-failure-{kind}")
    token = CancellationToken()
    original_append = writer.append

    async def cancel_after_append(**kwargs: Any) -> None:
        await original_append(**kwargs)
        token.cancel()

    monkeypatch.setattr(writer, "append", cancel_after_append)
    if kind == "unary":
        call = RecordingSearchProvider(FakeSearch(), writer).search(
            "failure", 1, None, deadline=_deadline(), cancellation_token=token
        )
    else:
        stream = _recording_model(FailingStreamModel(), writer).stream(
            _request(), deadline=_deadline(), cancellation_token=token
        )
        call = anext(stream)

    with pytest.raises(OperationCancelled) as cancelled:
        await call
    assert cancelled.value.__cause__ is None
    assert cancelled.value.__context__ is None
    await writer.finalize()

    replay_token = CancellationToken()
    bundle = ReplayBundle.load(final_root)
    if kind == "unary":
        replay_call = ReplaySearchProvider(bundle).search(
            "failure", 1, None, deadline=_deadline(), cancellation_token=replay_token
        )
    else:
        replay_stream = ReplayModelProvider(bundle).stream(
            _request(), deadline=_deadline(), cancellation_token=replay_token
        )
        replay_call = anext(replay_stream)
    with pytest.raises(ProviderError) as recorded:
        await replay_call
    assert recorded.value.code in {"RATE_LIMITED", "INVALID_REQUEST"}
    assert recorded.value.usage == _usage()
    assert recorded.value.__cause__ is None
    assert recorded.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize("kind", ("unary", "stream"))
async def test_recorded_failure_after_hard_deadline_surfaces_timeout_but_preserves_record(
    tmp_path: Path, kind: str
) -> None:
    final_root = tmp_path / f"deadline-failure-{kind}"
    writer = ReplayBundleWriter.create(final_root, run_id=f"deadline-failure-{kind}")
    deadline = time.monotonic() + 0.02
    token = CancellationToken()
    if kind == "unary":
        call = RecordingSearchProvider(SlowFailureSearch(), writer).search(
            "delayed", 1, None, deadline=deadline, cancellation_token=token
        )
    else:
        stream = _recording_model(FailingStreamModel(delay=0.05), writer).stream(
            _request(), deadline=deadline, cancellation_token=token
        )
        call = anext(stream)

    with pytest.raises(ProviderError) as live_error:
        await call
    assert live_error.value.code == "TIMEOUT"
    assert live_error.value.__cause__ is None
    assert live_error.value.__context__ is None
    await writer.finalize()

    bundle = ReplayBundle.load(final_root)
    if kind == "unary":
        replay_call = ReplaySearchProvider(bundle).search(
            "delayed", 1, None, deadline=_deadline(), cancellation_token=token
        )
    else:
        replay_stream = ReplayModelProvider(bundle).stream(
            _request(), deadline=_deadline(), cancellation_token=token
        )
        replay_call = anext(replay_stream)
    with pytest.raises(ProviderError) as recorded:
        await replay_call
    assert recorded.value.code == "INVALID_REQUEST"
    assert recorded.value.usage == _usage()
    assert recorded.value.__cause__ is None
    assert recorded.value.__context__ is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "failure_point",
    ("preflight", "record-write", "snapshot-write", "manifest-write", "self-verify", "rename"),
)
async def test_prepublication_finalize_failure_cleans_staging_and_releases_lock(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    final_root = tmp_path / f"prepublish-{failure_point}"
    writer = ReplayBundleWriter.create(final_root, run_id=f"prepublish-{failure_point}")
    original_write = recording_module._write_fsynced

    if failure_point == "preflight":
        final_root.mkdir()
    elif failure_point in {"record-write", "snapshot-write", "manifest-write"}:
        target = {
            "record-write": ".jsonl",
            "snapshot-write": "snapshot.json",
            "manifest-write": "manifest.sha256",
        }[failure_point]

        def fail_write(path: Path, payload: bytes) -> None:
            if path.name.endswith(target):
                raise OSError(f"synthetic {failure_point} failure")
            original_write(path, payload)

        monkeypatch.setattr(recording_module, "_write_fsynced", fail_write)
    elif failure_point == "self-verify":
        def fail_load(root: Path) -> NoReturn:
            del root
            raise OSError("synthetic self-verify failure")

        monkeypatch.setattr(ReplayBundle, "load", staticmethod(fail_load))
    else:
        def fail_rename(source: Path, destination: Path) -> NoReturn:
            del source, destination
            raise OSError("synthetic rename failure")

        monkeypatch.setattr(recording_module, "_atomic_rename_noreplace", fail_rename)

    with pytest.raises((FileExistsError, OSError)):
        await writer.finalize()

    assert not list(tmp_path.glob(f".{final_root.name}.staging.*"))
    with pytest.raises(RuntimeError, match="closed"):
        await writer.finalize()
    await writer.abort()
    await writer.abort()

    monkeypatch.undo()
    if final_root.exists():
        final_root.rmdir()
    replacement = ReplayBundleWriter.create(
        final_root, run_id=f"replacement-{failure_point}"
    )
    await replacement.abort()


@pytest.mark.asyncio
async def test_prepublication_cleanup_failure_still_releases_lock_and_can_be_retried(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    final_root = tmp_path / "cleanup-failure"
    writer = ReplayBundleWriter.create(final_root, run_id="cleanup-failure")
    original_write = recording_module._write_fsynced
    original_unlink = Path.unlink

    def fail_write(path: Path, payload: bytes) -> None:
        if path.name.endswith(".jsonl"):
            raise OSError("synthetic primary failure")
        original_write(path, payload)

    def fail_cleanup(path: Path, *args: object, **kwargs: object) -> None:
        if path.name == ".replay-writer-owner.json":
            raise OSError("synthetic cleanup failure")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(recording_module, "_write_fsynced", fail_write)
    monkeypatch.setattr(Path, "unlink", fail_cleanup)

    with pytest.raises(OSError, match="cleanup") as error:
        await writer.finalize()
    assert isinstance(error.value.__cause__, OSError)
    assert isinstance(error.value.__context__, OSError)
    with pytest.raises(RuntimeError, match="closed"):
        await writer.finalize()

    monkeypatch.undo()
    await writer.abort()
    await writer.abort()
    replacement = ReplayBundleWriter.create(final_root, run_id="cleanup-replacement")
    await replacement.abort()


async def _empty_model_replay(tmp_path: Path) -> ReplayModelProvider:
    writer = ReplayBundleWriter.create(tmp_path / "empty-replay", run_id="empty-replay")
    writer.configure_model_provider(
        provider_id=FakeModel.provider_id, model_revision="revision-1"
    )
    root = await writer.finalize()
    return ReplayModelProvider(ReplayBundle.load(root))


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", (float("nan"), float("inf"), float("-inf")))
async def test_recording_and_replay_reject_nonfinite_deadlines_before_cancellation(
    tmp_path: Path, deadline: float
) -> None:
    writer = ReplayBundleWriter.create(tmp_path / "record", run_id="record-deadline")
    model = FakeModel()
    recorder = _recording_model(model, writer)
    replay = await _empty_model_replay(tmp_path)
    token = CancellationToken()
    token.cancel()

    with pytest.raises(ValueError, match="finite"):
        await recorder.complete(_request(), deadline=deadline, cancellation_token=token)
    with pytest.raises(ValueError, match="finite"):
        await replay.complete(_request(), deadline=deadline, cancellation_token=token)
    recording_stream = recorder.stream(
        _request(), deadline=deadline, cancellation_token=token
    )
    with pytest.raises(ValueError, match="finite"):
        await anext(recording_stream)
    replay_stream = replay.stream(
        _request(), deadline=deadline, cancellation_token=token
    )
    with pytest.raises(ValueError, match="finite"):
        await anext(replay_stream)

    assert model.calls == 0
    await writer.abort()


@pytest.mark.asyncio
async def test_recording_and_replay_finite_expiry_keep_cancellation_first(
    tmp_path: Path,
) -> None:
    writer = ReplayBundleWriter.create(tmp_path / "record", run_id="record-cancelled")
    model = FakeModel()
    recorder = _recording_model(model, writer)
    replay = await _empty_model_replay(tmp_path)
    token = CancellationToken()
    token.cancel()
    expired = time.monotonic() - 1

    with pytest.raises(OperationCancelled):
        await recorder.complete(_request(), deadline=expired, cancellation_token=token)
    with pytest.raises(OperationCancelled):
        await replay.complete(_request(), deadline=expired, cancellation_token=token)
    recording_stream = recorder.stream(
        _request(), deadline=expired, cancellation_token=token
    )
    with pytest.raises(OperationCancelled):
        await anext(recording_stream)
    replay_stream = replay.stream(
        _request(), deadline=expired, cancellation_token=token
    )
    with pytest.raises(OperationCancelled):
        await anext(replay_stream)

    assert model.calls == 0
    await writer.abort()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure_point", ("marker", "final", "parent"))
async def test_post_publish_failures_release_lock_and_abort_never_deletes_final(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, failure_point: str
) -> None:
    final_root = tmp_path / f"published-{failure_point}"
    writer = ReplayBundleWriter.create(final_root, run_id=f"published-{failure_point}")
    original_unlink = Path.unlink

    if failure_point == "marker":
        def fail_marker(path: Path, *args: object, **kwargs: object) -> None:
            del args, kwargs
            if path.parent == final_root and path.name == ".replay-writer-owner.json":
                raise OSError("synthetic marker durability failure")
            original_unlink(path)

        monkeypatch.setattr(Path, "unlink", fail_marker)
    else:
        def fail_fsync(path: Path) -> None:
            target = final_root if failure_point == "final" else final_root.parent
            if path == target:
                raise OSError(f"synthetic {failure_point} durability failure")

        monkeypatch.setattr(
            "deepresearch.providers.recording._fsync_directory", fail_fsync
        )

    with pytest.raises(OSError, match="published|durability"):
        await writer.finalize()
    assert final_root.exists()
    await writer.abort()
    assert final_root.exists()

    monkeypatch.undo()
    moved = tmp_path / f"saved-{failure_point}"
    final_root.rename(moved)
    replacement = ReplayBundleWriter.create(
        final_root, run_id=f"replacement-{failure_point}"
    )
    await replacement.abort()


@pytest.mark.asyncio
async def test_writer_uses_atomic_no_replace_helper_for_destination_race(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    writer = ReplayBundleWriter.create(tmp_path / "race", run_id="race-run")
    called = False

    def collide(source: Path, destination: Path) -> NoReturn:
        nonlocal called
        del source, destination
        called = True
        raise FileExistsError("synthetic destination race")

    monkeypatch.setattr(
        "deepresearch.providers.recording._atomic_rename_noreplace",
        collide,
        raising=False,
    )
    with pytest.raises(FileExistsError, match="destination race"):
        await writer.finalize()
    await writer.abort()

    assert called is True
