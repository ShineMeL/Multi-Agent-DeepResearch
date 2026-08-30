import json
import os
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import NoReturn, TypeVar

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


class FakeFetcher:
    provider_id = "record-fetch"

    def __init__(
        self,
        *,
        final_url: str | None = None,
        headers: Mapping[str, str] | None = None,
    ) -> None:
        self.calls = 0
        self.final_url = final_url
        self.headers = headers

    async def fetch(self, url: str, *, deadline: float, cancellation_token: CancellationToken) -> RawDocument:
        del deadline
        cancellation_token.raise_if_cancelled()
        self.calls += 1
        return RawDocument(
            requested_url=url,
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
@pytest.mark.parametrize(
    ("header", "value"),
    (
        ("etag", "Bearer TOP-SECRET"),
        ("content-language", "cookie%3DTOP-SECRET"),
        ("cache-control", "private, api%255fkey=TOP-SECRET"),
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

    await RecordingFetcher(FakeFetcher(headers=headers), writer).fetch(
        "https://example.com/start",
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    await writer.finalize()

    record = json.loads((final_root / "documents.jsonl").read_text(encoding="utf-8"))
    assert record["outcome"]["response"]["headers"] == headers


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
