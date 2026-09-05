from __future__ import annotations

import asyncio
import contextlib
import hashlib
import json
import os
import stat
import subprocess
import sys
import tempfile
import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import TYPE_CHECKING, Annotated, Any, Literal, Self, cast
from urllib.parse import urlsplit

import typer
from click import ClickException

from deepresearch.domain import (
    FreshnessRequirement,
    ResearchRequest,
    RunBudget,
    RunConfig,
    RunEvent,
    RunResult,
)
from deepresearch.evidence.similarity import SimilarityRanker
from deepresearch.planning import FixedPlanner
from deepresearch.providers import (
    Fetcher,
    ModelProvider,
    Parser,
    SearchProvider,
    StructuredModelResult,
    TextEmbedder,
)
from deepresearch.providers.parsers import HtmlParser, PdfParser
from deepresearch.providers.replay_schema import (
    REPLAY_BUNDLE_SCHEMA_VERSION,
    REPLAY_FILES,
    ReplayBundle,
    ReplayProviderSnapshot,
    canonical_json_bytes,
)
from deepresearch.reporting import MarkdownReportWriter
from deepresearch.retrieval import URLSecurityError, canonicalize_url
from deepresearch.runtime import (
    CancellationToken,
    CheckpointRef,
    open_sqlite_checkpointer,
)
from deepresearch.runtime.checkpoints import checkpoint_ref_from_tuple
from deepresearch.storage import (
    ArtifactIntegrityError,
    FileCache,
    LocalArtifactStore,
    LocalEvidenceStore,
)
from deepresearch.workflow import (
    BaselineNodeHandlers,
    BaselineRuntimeHooks,
    DurableRunEventSink,
    LangGraphResearchRunner,
    build_baseline_graph,
)

if TYPE_CHECKING:
    from deepresearch.config import Settings
    from deepresearch.providers.embeddings import EmbeddingModelLock

from deepresearch import __version__

app = typer.Typer(no_args_is_help=True)


@dataclass(frozen=True)
class _ResearchOptions:
    question: str
    mode: Literal["live", "replay", "hybrid"]
    replay_root: Path | None
    record_replay_root: Path | None
    checkpoint_db: Path
    resume_checkpoint: str | None
    resume_thread_id: str | None
    budget: Literal["low", "medium", "high"]
    output: Path


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0) & reparse_flag
    )


def _option_error(message: str) -> None:
    # Keep validation messages independent from paths, credentials, and exception
    # representations.  This boundary runs before any provider is constructed.
    raise typer.BadParameter(message)


def _validate_research_options(options: _ResearchOptions) -> _ResearchOptions:
    if not options.question.strip():
        _option_error("question must not be empty")

    if options.mode == "replay" and options.record_replay_root is not None:
        _option_error("replay mode forbids --record-replay-root")
    if options.mode == "replay" and options.replay_root is None:
        _option_error("replay mode requires --replay-root")
    if options.mode == "live" and options.replay_root is not None:
        _option_error("live mode forbids --replay-root")
    if options.mode == "hybrid" and options.replay_root is None:
        _option_error("hybrid mode requires --replay-root")

    resume_values = (
        options.resume_checkpoint is not None,
        options.resume_thread_id is not None,
    )
    if any(resume_values) and not all(resume_values):
        _option_error("resume requires --resume-checkpoint and --resume-thread-id")
    if all(resume_values) and options.record_replay_root is not None:
        _option_error("recording cannot be combined with resume")

    replay_root = options.replay_root
    if replay_root is not None and (
        _is_link_or_reparse(replay_root) or not replay_root.exists() or not replay_root.is_dir()
    ):
        _option_error("replay root is invalid")

    record_root = options.record_replay_root
    if record_root is not None and (_is_link_or_reparse(record_root) or record_root.exists()):
        _option_error("recording destination must not exist")

    output = options.output
    if _is_link_or_reparse(output) or output.exists():
        _option_error("output destination must not exist")

    checkpoint_db = options.checkpoint_db.absolute()
    if checkpoint_db == Path(checkpoint_db.anchor) or not checkpoint_db.name:
        _option_error("checkpoint path is invalid")
    if _is_link_or_reparse(checkpoint_db) or (checkpoint_db.exists() and not checkpoint_db.is_file()):
        _option_error("checkpoint path is invalid")
    return _ResearchOptions(
        question=options.question,
        mode=options.mode,
        replay_root=replay_root,
        record_replay_root=record_root,
        checkpoint_db=checkpoint_db,
        resume_checkpoint=options.resume_checkpoint,
        resume_thread_id=options.resume_thread_id,
        budget=options.budget,
        output=output,
    )


_PROFILE_SCHEMA_VERSION = "baseline-provider-profile-v1"
_MODEL_PROVIDER_ID = "openai-compatible"
_MODEL_REVISION = "provider-managed"
_SEARCH_PROVIDER_ID = "tavily"
_SEARCH_ENDPOINT = "https://api.tavily.com/search"
_FETCH_PROVIDER_ID = "httpx-fetcher"
_FETCH_POLICY_VERSION = "pinned-http-fetch-v1"
_FETCH_MAX_REDIRECTS = 5
_FETCH_MAX_BODY_BYTES = 10 * 1024 * 1024
_PARSER_SPECS: tuple[dict[str, str], ...] = (
    {"parser_id": "trafilatura-html", "parser_version": "2.2"},
    {"parser_id": "pymupdf-pdf", "parser_version": "1.28"},
)
_RUNTIME_PROFILE_PLACEHOLDER = "runtime-profile-bound-v1"


def _safe_absolute_endpoint(value: str, *, model_base: bool = False) -> str:
    """Validate and canonicalize an endpoint without exposing its value on error."""
    try:
        parsed = urlsplit(value)
        if (
            parsed.scheme.casefold() not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.query
            or parsed.fragment
        ):
            raise ValueError
        candidate = value.rstrip("/") if model_base else value
        return canonicalize_url(candidate)
    except (URLSecurityError, UnicodeError, ValueError, TypeError):
        raise ValueError("provider endpoint is invalid") from None


def _snapshot_payload(snapshot: ReplayProviderSnapshot) -> dict[str, object]:
    return snapshot.model_dump(mode="json", exclude_none=False)


def _replay_profile(bundle: ReplayBundle, *, mode: Literal["replay", "hybrid"]) -> dict[str, object]:
    verification = bundle.verify()
    if not verification.valid:
        raise ValueError("replay bundle is invalid")
    snapshots = bundle.snapshot.providers
    if mode == "replay":
        file_sha256 = {
            filename: verification.file_sha256[filename]
            for filename in REPLAY_FILES
        }
        providers = {
            operation: _snapshot_payload(snapshots[operation])
            for operation in sorted(snapshots)
        }
    else:
        file_sha256 = {
            filename: verification.file_sha256[filename]
            for filename in ("documents.jsonl", "search.jsonl")
        }
        providers = {
            operation: _snapshot_payload(snapshots[operation])
            for operation in ("fetch", "search")
        }
    return {
        "schema_version": REPLAY_BUNDLE_SCHEMA_VERSION,
        "run_id": bundle.snapshot.run_id,
        "file_sha256": file_sha256,
        "providers": providers,
    }


def _live_profile(
    *,
    settings: Settings,
    embedding_lock: EmbeddingModelLock,
    include_search_fetch: bool = True,
) -> dict[str, object]:
    model_url = _safe_absolute_endpoint(str(settings.model_base_url), model_base=True)
    search_url = _safe_absolute_endpoint(_SEARCH_ENDPOINT)
    live_search: dict[str, object] | None = {
        "provider_id": _SEARCH_PROVIDER_ID,
        "endpoint": search_url,
    } if include_search_fetch else None
    live_fetch: dict[str, object] | None = {
        "provider_id": _FETCH_PROVIDER_ID,
        "policy_version": _FETCH_POLICY_VERSION,
        "max_redirects": _FETCH_MAX_REDIRECTS,
        "max_body_bytes": _FETCH_MAX_BODY_BYTES,
    } if include_search_fetch else None
    return {
        "model": {
            "base_url": model_url,
            "provider_id": _MODEL_PROVIDER_ID,
            "model_id": settings.model_id,
            "model_revision": _MODEL_REVISION,
            "seed_supported": True,
        },
        "search": live_search,
        "fetch": live_fetch,
        "parsers": [dict(item) for item in _PARSER_SPECS],
        "embed": {
            "provider_id": "sentence-transformer",
            "model_id": embedding_lock.model_id,
            "model_revision": embedding_lock.revision,
            "snapshot_sha256": embedding_lock.snapshot_sha256,
        },
        "transport": {
            "connect_timeout_seconds": float(settings.connect_timeout_seconds),
            "read_timeout_seconds": float(settings.read_timeout_seconds),
            "follow_redirects": False,
            "trust_env": False,
        },
    }


def _build_provider_profile(
    *,
    mode: Literal["live", "replay", "hybrid"],
    bundle: ReplayBundle | None = None,
    settings: Settings | None = None,
    embedding_lock: EmbeddingModelLock | None = None,
) -> tuple[dict[str, object], str]:
    if mode == "replay":
        if bundle is None:
            raise ValueError("replay bundle is required")
        replay = _replay_profile(bundle, mode="replay")
        live = None
    elif mode == "hybrid":
        if bundle is None or settings is None or embedding_lock is None:
            raise ValueError("hybrid provider inputs are incomplete")
        replay = _replay_profile(bundle, mode="hybrid")
        live = _live_profile(settings=settings, embedding_lock=embedding_lock, include_search_fetch=False)
    else:
        if settings is None or embedding_lock is None:
            raise ValueError("live provider inputs are incomplete")
        replay = None
        live = _live_profile(settings=settings, embedding_lock=embedding_lock)
    profile = {
        "schema_version": _PROFILE_SCHEMA_VERSION,
        "mode": mode,
        "replay": replay,
        "live": live,
    }
    digest = hashlib.sha256(canonical_json_bytes(profile)).hexdigest()
    return cast("dict[str, object]", profile), f"baseline-{mode}-v1:{digest}"


def _build_request_config(
    *,
    question: str,
    mode: Literal["live", "replay", "hybrid"],
    budget_name: Literal["low", "medium", "high"],
    provider_profile_id: str,
    prompt_versions: dict[str, str],
) -> RunConfig:
    budget = RunBudget.preset(budget_name).model_copy(update={"max_cost_usd": None})
    request = ResearchRequest(
        question=question,
        output_requirements={"answer_shape": "markdown"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode=mode,
        access_profile="local",
        provider_profile_id=provider_profile_id,
        run_purpose="demo",
        budget_preset=budget_name,
    )
    return RunConfig(
        request=request,
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
        budget=budget,
        prompt_versions=dict(prompt_versions),
        ranker_weights_version=None,
        seed=0,
    )


def _repository_metadata(*, root: Path | None = None) -> tuple[str, str]:
    repository_root = (root or Path(__file__).resolve().parents[2]).resolve()
    lock_path = repository_root / "uv.lock"
    try:
        completed = subprocess.run(
            ["git", "-C", str(repository_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip()
        if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
            raise ValueError
        lock_digest = hashlib.sha256(lock_path.read_bytes()).hexdigest()
    except (OSError, subprocess.SubprocessError, UnicodeError, ValueError):
        raise ValueError("repository metadata is unavailable") from None
    return commit, lock_digest


def _bind_runtime_profile_request(request: Any) -> Any:
    """Stabilize planner request identity without leaking a circular profile hash.

    ``provider_profile_id`` is run metadata, not research content.  The profile
    itself signs the replay files, while the planner request is recorded in one
    of those files; passing the digest through the planner prompt would create
    an impossible cryptographic fixed point.  Keep the public RunConfig value
    exact, but use one deterministic marker at the model boundary.
    """
    if getattr(request, "prompt_version", None) != "fixed-planner-v1":
        return request
    messages_value: object = getattr(request, "messages", None)
    if not isinstance(messages_value, tuple) or not messages_value:
        return request
    messages = cast("tuple[Any, ...]", messages_value)
    last: Any = messages[-1]
    content = getattr(last, "content", None)
    if not isinstance(content, str):
        return request
    try:
        payload_value: object = json.loads(
            content,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError):
        return request
    if type(payload_value) is not dict or "provider_profile_id" not in payload_value:
        return request
    payload = cast("dict[str, object]", payload_value)
    payload["provider_profile_id"] = _RUNTIME_PROFILE_PLACEHOLDER
    message_copy = getattr(last, "model_copy", None)
    request_copy = getattr(request, "model_copy", None)
    if not callable(message_copy) or not callable(request_copy):
        return request
    bound_message = message_copy(
        update={
            "content": canonical_json_bytes(payload).decode("utf-8"),
        }
    )
    return request_copy(update={"messages": (*messages[:-1], bound_message)})


class _ModelIdentityAdapter:
    """Immutable public-protocol forwarding adapter with explicit model identity."""

    def __init__(
        self,
        delegate: ModelProvider,
        *,
        model_id: str,
        model_revision: str,
        bind_runtime_profile: bool = False,
    ) -> None:
        if not model_id or not model_revision:
            raise ValueError("model identity is incomplete")
        self._delegate = delegate
        self.provider_id = delegate.provider_id
        self.model_id = model_id
        self.model_revision = model_revision
        self._bind_runtime_profile = bind_runtime_profile

    def _request(self, request: Any) -> Any:
        copy = getattr(request, "model_copy", None)
        if not callable(copy):
            raise TypeError("model request is invalid")
        request_copy = copy(update={"model_id": self.model_id})
        if not self._bind_runtime_profile:
            return request_copy
        return _bind_runtime_profile_request(request_copy)

    async def complete(self, request: Any, *, deadline: float, cancellation_token: CancellationToken) -> Any:
        result = await self._delegate.complete(
            self._request(request),
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return result.model_copy(update={"model_id": self.model_id})

    async def structured(
        self,
        request: Any,
        output_schema: type[Any],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> Any:
        result = await self._delegate.structured(
            self._request(request),
            output_schema,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        # The replay provider's public generic return is parameterized with a
        # runtime TypeVar.  Calling its model_copy would therefore revalidate
        # the typed output as a plain mapping.  Rebind the supplied schema here
        # so FixedPlanner receives the exact structured object it requested.
        return StructuredModelResult[output_schema].model_validate(
            {
                **result.model_dump(mode="json"),
                "output": result.output,
                "model_id": self.model_id,
            }
        )

    def stream(
        self,
        request: Any,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> AsyncIterator[Any]:
        return self._delegate.stream(
            self._request(request),
            deadline=deadline,
            cancellation_token=cancellation_token,
        )

    async def aclose(self) -> None:
        close = getattr(self._delegate, "aclose", None)
        if callable(close):
            result = close()
            if hasattr(result, "__await__"):
                await cast("Any", result)


class _ParserRouter:
    parser_id = "baseline-parser-router"
    parser_version = "baseline-parser-v1"

    def __init__(self, parsers: tuple[Parser, ...]) -> None:
        self._parsers = parsers

    def supports(self, content_type: str) -> bool:
        return any(parser.supports(content_type) for parser in self._parsers)

    async def parse(
        self,
        raw_document: Any,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> Any:
        for parser in self._parsers:
            if parser.supports(raw_document.content_type):
                parsed = await parser.parse(
                    raw_document,
                    deadline=deadline,
                    cancellation_token=cancellation_token,
                )
                return parsed.model_copy(
                    update={"parser_id": self.parser_id, "parser_version": self.parser_version}
                )
        from deepresearch.providers import ProviderError

        raise ProviderError(
            code="PARSE_UNSUPPORTED",
            provider=self.parser_id,
            operation="parse",
            public_message="document media type is unsupported",
            retryable=False,
        )


class _UnknownCostResolver:
    def resolve_cost(
        self,
        *,
        operation: str,
        provider_id: str,
        model_id: str | None,
        outcome: str,
        usage: Any,
    ) -> Decimal | None:
        del operation, provider_id, model_id, outcome, usage
        return None


def _runtime_root(checkpoint_db: Path) -> Path:
    absolute = checkpoint_db.absolute()
    return absolute.parent / f".{absolute.stem}.runtime"


def _runtime_hooks() -> BaselineRuntimeHooks:
    """Pair wall timestamps to one monotonic source for manifest envelopes."""
    origin_monotonic = time.monotonic()
    origin_utc = datetime.now(UTC)
    utc_calls = 0

    def monotonic() -> float:
        return time.monotonic()

    def utc_now() -> datetime:
        nonlocal utc_calls
        utc_calls += 1
        elapsed = max(0.0, time.monotonic() - origin_monotonic)
        # A timestamp is sampled after the monotonic measurement that updates
        # elapsed state.  Keep the wall envelope just ahead of that sample so
        # sub-millisecond clock ordering cannot make a valid run fail strict
        # manifest reconciliation.
        slack = 0.001 if utc_calls > 1 else 0.0
        return origin_utc + timedelta(seconds=elapsed + slack)

    return BaselineRuntimeHooks(monotonic=monotonic, utc_now=utc_now)


def _new_run_id() -> str:
    return str(uuid.uuid4())


def _path_identity(path: Path) -> tuple[int, int]:
    details = path.stat(follow_symlinks=False)
    return details.st_dev, details.st_ino


class _RuntimeLock:
    def __init__(self, root: Path) -> None:
        self.root = root
        self._descriptor: int | None = None
        self._locked = False

    def __enter__(self) -> Self:
        if _is_link_or_reparse(self.root):
            raise ValueError("runtime root is unsafe")
        if self.root.exists() and not self.root.is_dir():
            raise ValueError("runtime root is not a directory")
        self.root.mkdir(parents=True, exist_ok=True)
        if _is_link_or_reparse(self.root) or not self.root.is_dir():
            raise ValueError("runtime root is unsafe")
        lock_path = self.root / ".runtime.lock"
        if _is_link_or_reparse(lock_path):
            raise ValueError("runtime lock is unsafe")
        descriptor = os.open(
            lock_path,
            os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        try:
            if os.fstat(descriptor).st_size == 0:
                os.write(descriptor, b"\0")
                os.fsync(descriptor)
            os.lseek(descriptor, 0, os.SEEK_SET)
            if os.name == "nt":
                import msvcrt

                try:
                    msvcrt.locking(descriptor, msvcrt.LK_NBLCK, 1)
                except OSError as error:
                    raise BlockingIOError("runtime is already in use") from error
            else:
                import fcntl

                try:
                    fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as error:
                    raise BlockingIOError("runtime is already in use") from error
            self._descriptor = descriptor
            self._locked = True
            return self
        except BaseException:
            os.close(descriptor)
            raise

    def __exit__(self, exc_type: object, exc: object, traceback: object) -> None:
        descriptor = self._descriptor
        self._descriptor = None
        if descriptor is None:
            return
        try:
            if self._locked:
                os.lseek(descriptor, 0, os.SEEK_SET)
                if os.name == "nt":
                    import msvcrt

                    msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)
                else:
                    import fcntl

                    fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
            self._locked = False


def _event_path(root: Path, *, run_id: str, seq: int) -> Path:
    key = canonical_json_bytes({"run_id": run_id, "seq": seq})
    digest = hashlib.sha256(key).hexdigest()
    directory = root / "run-events" / digest[:2]
    directory.mkdir(parents=True, exist_ok=True)
    return directory / f"{digest}.json"


class _DurableEventFileSink:
    def __init__(self, root: Path) -> None:
        self._root = root
        events = root / "run-events"
        if _is_link_or_reparse(events):
            raise ValueError("event root is unsafe")
        events.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _canonical_event(event: RunEvent) -> bytes:
        payload = canonical_json_bytes(event.model_dump(mode="json"))
        if len(payload) > 1024 * 1024:
            raise ArtifactIntegrityError("event is too large")
        return payload

    async def __call__(self, event: RunEvent) -> None:
        payload = self._canonical_event(event)
        path = _event_path(self._root, run_id=event.run_id, seq=event.seq)
        try:
            descriptor = os.open(
                path,
                os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
                0o600,
            )
        except FileExistsError:
            existing = path.read_bytes()
            if existing != payload:
                raise ArtifactIntegrityError("durable event conflicts with stored event")
            return
        try:
            with os.fdopen(descriptor, "wb") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
        except BaseException:
            with contextlib.suppress(OSError):
                path.unlink()
            raise

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        path = _event_path(self._root, run_id=run_id, seq=seq)
        try:
            payload = path.read_bytes()
        except FileNotFoundError:
            return None
        try:
            event = RunEvent.model_validate_json(payload, strict=True)
            if canonical_json_bytes(event.model_dump(mode="json")) != payload:
                raise ValueError
            if event.run_id != run_id or event.seq != seq:
                raise ValueError
            return event
        except (TypeError, ValueError):
            raise ArtifactIntegrityError("durable event is corrupt") from None


@dataclass
class _RuntimeComposition:
    model: ModelProvider
    search: SearchProvider
    fetcher: Fetcher
    embedder: TextEmbedder
    parser: Parser
    artifact_store: LocalArtifactStore
    evidence_store: LocalEvidenceStore
    cache: FileCache
    graph: Any
    runner: LangGraphResearchRunner
    config: RunConfig
    replay_bundle: ReplayBundle | None = None
    recording_writer: Any | None = None
    _owned: tuple[Any, ...] = ()

    async def aclose(self) -> None:
        errors: list[Exception] = []
        for resource in reversed(self._owned):
            close = getattr(resource, "aclose", None)
            if not callable(close):
                continue
            try:
                result = close()
                if hasattr(result, "__await__"):
                    await cast("Any", result)
            except Exception as error:  # noqa: BLE001 - close all resources
                errors.append(error)
        if errors:
            raise errors[0]


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_json_constant(value: str) -> object:
    raise ValueError(f"non-finite JSON constant: {value}")


def _validate_public_artifacts(
    composition: _RuntimeComposition,
    result: RunResult,
) -> tuple[bytes, bytes, bytes]:
    if result.status != "completed" or not all(
        isinstance(item, str)
        for item in (
            result.report_artifact_id,
            result.evidence_graph_artifact_id,
            result.manifest_artifact_id,
        )
    ):
        raise ArtifactIntegrityError("completed run is missing public artifacts")
    assert result.report_artifact_id is not None
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    try:
        report_bytes = composition.artifact_store.get_bytes(result.report_artifact_id)
        evidence_bytes = composition.artifact_store.get_bytes(result.evidence_graph_artifact_id)
        manifest_bytes = composition.artifact_store.get_bytes(result.manifest_artifact_id)
        report_bytes.decode("utf-8")
        parsed_evidence: object = json.loads(
            evidence_bytes,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if type(parsed_evidence) is not dict:
            raise ValueError
        parsed_mapping = cast("dict[str, object]", parsed_evidence)
        if set(parsed_mapping) != {"evidence", "sources"}:
            raise ValueError
        from deepresearch.domain import EvidenceSpan, SourceDocument

        evidence_values = parsed_mapping["evidence"]
        source_values = parsed_mapping["sources"]
        if type(evidence_values) is not list or type(source_values) is not list:
            raise ValueError
        for item in cast("list[object]", evidence_values):
            EvidenceSpan.model_validate_json(canonical_json_bytes(item), strict=True)
        for item in cast("list[object]", source_values):
            SourceDocument.model_validate_json(canonical_json_bytes(item), strict=True)
        from deepresearch.runtime.manifest import RunManifest

        RunManifest.model_validate_json(manifest_bytes, strict=True)
    except (FileNotFoundError, TypeError, ValueError, UnicodeDecodeError):
        raise ArtifactIntegrityError("public artifacts are invalid") from None
    return report_bytes, evidence_bytes, manifest_bytes


def _write_fsynced(path: Path, payload: bytes) -> None:
    descriptor = os.open(
        path,
        os.O_CREAT | os.O_EXCL | os.O_WRONLY | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
    except BaseException:
        with contextlib.suppress(OSError):
            path.unlink()
        raise


def _flush_windows_directory(path: Path) -> None:
    """Flush directory metadata with a native backup-semantics handle."""
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.FlushFileBuffers.argtypes = (wintypes.HANDLE,)
    kernel32.FlushFileBuffers.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CloseHandle.restype = wintypes.BOOL
    handle = kernel32.CreateFileW(
        str(path),
        0x40000000,
        0x00000001 | 0x00000002 | 0x00000004,
        None,
        3,
        0x02000000,
        None,
    )
    if handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    flushed = False
    try:
        if not kernel32.FlushFileBuffers(handle):
            raise ctypes.WinError(ctypes.get_last_error())
        flushed = True
    finally:
        if not kernel32.CloseHandle(handle) and flushed:
            raise ctypes.WinError(ctypes.get_last_error())


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        _flush_windows_directory(path)
        return
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_rename_noreplace_impl(source: Path, destination: Path) -> None:
    """Atomically publish a directory without replacing an existing target."""
    if os.name == "nt":
        # ``os.rename`` maps to MoveFileExW without replace semantics on Windows.
        os.rename(source, destination)
        return
    import ctypes
    import errno

    libc = ctypes.CDLL(None, use_errno=True)
    source_bytes = os.fsencode(source)
    destination_bytes = os.fsencode(destination)
    if sys.platform.startswith("linux"):
        renameat2 = getattr(libc, "renameat2", None)
        if renameat2 is None:
            raise NotImplementedError("atomic no-replace rename is unavailable")
        result = renameat2(-100, source_bytes, -100, destination_bytes, 1)
    elif sys.platform == "darwin":
        renamex_np = getattr(libc, "renamex_np", None)
        if renamex_np is None:
            raise NotImplementedError("atomic no-replace rename is unavailable")
        result = renamex_np(source_bytes, destination_bytes, 0x00000004)
    else:
        raise NotImplementedError("atomic no-replace rename is unsupported")
    if result != 0:
        error_number = ctypes.get_errno()
        if error_number == errno.EEXIST:
            raise FileExistsError(error_number, os.strerror(error_number), destination)
        raise OSError(error_number, os.strerror(error_number), destination)


def _atomic_rename_noreplace(source: Path, destination: Path) -> None:
    """Keep the publication primitive injectable for security/fault tests."""
    _atomic_rename_noreplace_impl(source, destination)


def _publish_public_output(
    *,
    output: Path,
    report_bytes: bytes,
    evidence_bytes: bytes,
    manifest_bytes: bytes,
) -> Path:
    final = output.absolute()
    parent = final.parent
    if _is_link_or_reparse(parent):
        raise ArtifactIntegrityError("output parent is unsafe")
    parent.mkdir(parents=True, exist_ok=True)
    if _is_link_or_reparse(parent):
        raise ArtifactIntegrityError("output parent is unsafe")
    parent_identity = _path_identity(parent)
    if _is_link_or_reparse(final) or final.exists():
        raise ArtifactIntegrityError("output destination is occupied")
    staging = Path(tempfile.mkdtemp(prefix=f".{final.name}.staging.", dir=parent))
    staging_identity = _path_identity(staging)
    try:
        if _is_link_or_reparse(staging) or _path_identity(staging) != staging_identity:
            raise ArtifactIntegrityError("output staging object was substituted")
        for name, payload in (
            ("report.md", report_bytes),
            ("evidence.json", evidence_bytes),
            ("run-manifest.json", manifest_bytes),
        ):
            if _is_link_or_reparse(staging) or _path_identity(staging) != staging_identity:
                raise ArtifactIntegrityError("output staging object was substituted")
            _write_fsynced(staging / name, payload)
        _fsync_directory(staging)
        if _is_link_or_reparse(staging) or _path_identity(staging) != staging_identity:
            raise ArtifactIntegrityError("output staging object was substituted")
        if _is_link_or_reparse(parent) or _path_identity(parent) != parent_identity:
            raise ArtifactIntegrityError("output parent was substituted")
        if _is_link_or_reparse(final) or final.exists():
            raise ArtifactIntegrityError("output destination is occupied")
        _atomic_rename_noreplace(staging, final)
        staging = Path()
        try:
            _fsync_directory(final)
            _fsync_directory(parent)
        except (OSError, RuntimeError, ValueError) as error:
            raise OSError(
                "public output was published but durability finalization failed"
            ) from error
        return final
    finally:
        if staging != Path() and staging.exists():
            with contextlib.suppress(OSError):
                if not _is_link_or_reparse(staging) and _path_identity(staging) == staging_identity:
                    for child in staging.iterdir():
                        child.unlink(missing_ok=True)
                    staging.rmdir()
                    _fsync_directory(parent)


def _print_result(*, result: RunResult, output: Path) -> None:
    typer.echo(f"status={result.status}")
    if result.stop_reason is not None:
        typer.echo(f"stop_reason={result.stop_reason}")
    typer.echo(f"report={output / 'report.md'}")
    typer.echo(f"evidence={output / 'evidence.json'}")
    typer.echo(f"manifest={output / 'run-manifest.json'}")


def _replay_model_id(bundle: ReplayBundle) -> str:
    snapshot = bundle.snapshot.providers.get("model")
    if snapshot is None or snapshot.model_id is None:
        raise ValueError("replay model identity is unavailable")
    return snapshot.model_id


def _load_replay_bundle(path: Path) -> ReplayBundle:
    try:
        bundle = ReplayBundle.load(path)
        verification = bundle.verify()
        if not verification.valid:
            raise ValueError("replay bundle is invalid")
        return bundle
    except Exception as error:
        if isinstance(error, MemoryError):
            raise
        raise ValueError("replay bundle is invalid") from None


def _compose_replay_components(
    *,
    bundle: ReplayBundle,
    runtime_root: Path,
    budget_name: Literal["low", "medium", "high"],
    provider_profile_id: str,
    checkpointer: Any,
    question: str,
    mode: Literal["replay", "hybrid"],
) -> _RuntimeComposition:
    from deepresearch.providers.replay import (
        ReplayFetcher,
        ReplayModelProvider,
        ReplaySearchProvider,
        ReplayTextEmbedder,
    )

    artifact_store = LocalArtifactStore(runtime_root)
    evidence_store = LocalEvidenceStore(runtime_root)
    cache = FileCache(runtime_root)
    model_base: ModelProvider = _ModelIdentityAdapter(
        ReplayModelProvider(bundle),
        model_id=_replay_model_id(bundle),
        model_revision=bundle.snapshot.providers["model"].model_revision or "",
        bind_runtime_profile=True,
    )
    search: SearchProvider = ReplaySearchProvider(bundle)
    fetcher: Fetcher = ReplayFetcher(bundle)
    embedder: TextEmbedder = ReplayTextEmbedder(bundle)
    parser: Parser = _ParserRouter((HtmlParser(), PdfParser()))
    budget = RunBudget.preset(budget_name).model_copy(update={"max_cost_usd": None})
    planner = FixedPlanner(
        model=model_base,
        artifact_store=artifact_store,
        budget=budget,
    )
    ranker = SimilarityRanker(embedder)
    writer = MarkdownReportWriter(evidence_store, model=model_base)
    code_commit, lock_digest = _repository_metadata()
    handlers = BaselineNodeHandlers(
        initial_plan_generator=planner,
        ranker=cast("Any", ranker),
        writer=writer,
        search_provider=search,
        fetcher=fetcher,
        parser=parser,
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=cache,
        usage_cost_resolver=_UnknownCostResolver(),
        search_snapshot_id=f"{provider_profile_id}/search",
        fetch_snapshot_id=f"{provider_profile_id}/fetch",
        code_commit=code_commit,
        dependency_lock_sha256=lock_digest,
        provider_profile_configuration_sha256=provider_profile_id.split(":", 1)[1],
        seed_supported=True,
        pricing_status="unknown",
        pricing_snapshots=(),
        replay_parent=bundle.snapshot.run_id,
    )
    config = _build_request_config(
        question=question,
        mode=mode,
        budget_name=budget_name,
        provider_profile_id=provider_profile_id,
        prompt_versions=dict(handlers.prompt_versions),
    )
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer))
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=_runtime_hooks(),
    )
    return _RuntimeComposition(
        model=model_base,
        search=search,
        fetcher=fetcher,
        embedder=embedder,
        parser=parser,
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=cache,
        graph=graph,
        runner=runner,
        config=config,
        replay_bundle=bundle,
        _owned=(model_base, search, fetcher, embedder),
    )


def _compose_live_components(
    *,
    settings: Any,
    embedding_lock: Any,
    runtime_root: Path,
    budget_name: Literal["low", "medium", "high"],
    provider_profile_id: str,
    checkpointer: Any,
    question: str,
    mode: Literal["live", "hybrid"],
    replay_bundle: ReplayBundle | None = None,
) -> _RuntimeComposition:
    from deepresearch.providers.embeddings import SentenceTransformerTextEmbedder
    from deepresearch.providers.httpx_fetcher import HttpxFetcher
    from deepresearch.providers.httpx_transport import PinnedPeerTransport
    from deepresearch.providers.openai_compatible import OpenAICompatibleModelProvider
    from deepresearch.providers.tavily import TavilySearchProvider

    artifact_store = LocalArtifactStore(runtime_root)
    evidence_store = LocalEvidenceStore(runtime_root)
    cache = FileCache(runtime_root)
    transport = PinnedPeerTransport(
        connect_timeout_seconds=float(settings.connect_timeout_seconds),
        read_timeout_seconds=float(settings.read_timeout_seconds),
    )
    raw_model = OpenAICompatibleModelProvider(
        base_url=_safe_absolute_endpoint(str(settings.model_base_url), model_base=True),
        api_key=settings.model_api_key,
        model_revision=_MODEL_REVISION,
    )
    model: ModelProvider = _ModelIdentityAdapter(
        raw_model,
        model_id=settings.model_id,
        model_revision=_MODEL_REVISION,
        bind_runtime_profile=True,
    )
    parser: Parser = _ParserRouter((HtmlParser(), PdfParser()))
    embedder: TextEmbedder = SentenceTransformerTextEmbedder.from_lock(
        embedding_lock,
        model_root=Path(settings.embedding_model_root),
    )
    search: SearchProvider
    fetcher: Fetcher
    # The identity adapter owns and closes the raw model delegate.  Keep only
    # the adapter here so shutdown cannot close the same HTTP client twice.
    owned: list[Any] = [model, embedder]
    if replay_bundle is None:
        search = TavilySearchProvider(api_key=settings.tavily_api_key)
        fetcher = HttpxFetcher(
            transport=transport,
            max_redirects=_FETCH_MAX_REDIRECTS,
            max_body_bytes=_FETCH_MAX_BODY_BYTES,
        )
        owned.extend((search, fetcher))
    else:
        from deepresearch.providers.replay import ReplayFetcher, ReplaySearchProvider

        search = ReplaySearchProvider(replay_bundle)
        fetcher = ReplayFetcher(replay_bundle)
        owned.extend((search, fetcher))
    budget = RunBudget.preset(budget_name).model_copy(update={"max_cost_usd": None})
    planner = FixedPlanner(model=model, artifact_store=artifact_store, budget=budget)
    ranker = SimilarityRanker(embedder)
    writer = MarkdownReportWriter(evidence_store, model=model)
    code_commit, lock_digest = _repository_metadata()
    handlers = BaselineNodeHandlers(
        initial_plan_generator=planner,
        ranker=cast("Any", ranker),
        writer=writer,
        search_provider=search,
        fetcher=fetcher,
        parser=parser,
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=cache,
        usage_cost_resolver=_UnknownCostResolver(),
        search_snapshot_id=f"{provider_profile_id}/search",
        fetch_snapshot_id=f"{provider_profile_id}/fetch",
        code_commit=code_commit,
        dependency_lock_sha256=lock_digest,
        provider_profile_configuration_sha256=provider_profile_id.split(":", 1)[1],
        seed_supported=True,
        pricing_status="unknown",
        pricing_snapshots=(),
        replay_parent=None if replay_bundle is None else replay_bundle.snapshot.run_id,
    )
    config = _build_request_config(
        question=question,
        mode=mode,
        budget_name=budget_name,
        provider_profile_id=provider_profile_id,
        prompt_versions=dict(handlers.prompt_versions),
    )
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer))
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=_runtime_hooks(),
    )
    return _RuntimeComposition(
        model=model,
        search=search,
        fetcher=fetcher,
        embedder=embedder,
        parser=parser,
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=cache,
        graph=graph,
        runner=runner,
        config=config,
        replay_bundle=replay_bundle,
        _owned=tuple(owned),
    )


@app.callback()
def main() -> None:
    pass


@app.command()
def version() -> None:
    typer.echo(__version__)


@app.command()
def research(
    question: Annotated[str, typer.Option(help="The research question.")],
    mode: Annotated[
        Literal["live", "replay", "hybrid"],
        typer.Option(help="Provider execution mode."),
    ],
    replay_root: Annotated[
        Path | None,
        typer.Option(help="Verified replay bundle directory."),
    ] = None,
    record_replay_root: Annotated[
        Path | None,
        typer.Option(help="Destination for a newly recorded replay bundle."),
    ] = None,
    checkpoint_db: Annotated[
        Path,
        typer.Option(help="SQLite checkpoint file."),
    ] = Path("artifacts/checkpoints.sqlite3"),
    resume_checkpoint: Annotated[
        str | None,
        typer.Option(help="Exact checkpoint identity to resume."),
    ] = None,
    resume_thread_id: Annotated[
        str | None,
        typer.Option(help="Thread identity for an exact checkpoint resume."),
    ] = None,
    budget: Annotated[
        Literal["low", "medium", "high"],
        typer.Option(help="Run budget preset."),
    ] = "medium",
    output: Annotated[
        Path,
        typer.Option(help="Final public artifact directory."),
    ] = Path("artifacts/latest"),
) -> None:
    """Run the baseline deep-research workflow."""
    options = _validate_research_options(
        _ResearchOptions(
            question=question,
            mode=mode,
            replay_root=replay_root,
            record_replay_root=record_replay_root,
            checkpoint_db=checkpoint_db,
            resume_checkpoint=resume_checkpoint,
            resume_thread_id=resume_thread_id,
            budget=budget,
            output=output,
        )
    )
    try:
        asyncio.run(_research_async(options))
    except ClickException:
        raise
    except Exception as error:
        raise ClickException("research run failed") from error


async def _research_async(options: _ResearchOptions) -> None:
    replay_bundle: ReplayBundle | None = None
    settings: Any | None = None
    embedding_lock: Any | None = None
    if options.mode in {"replay", "hybrid"}:
        if options.replay_root is None:
            raise ClickException("replay root is required")
        replay_bundle = _load_replay_bundle(options.replay_root)
    if options.mode in {"live", "hybrid"}:
        try:
            from deepresearch.config import Settings
            from deepresearch.providers.embeddings import EmbeddingModelLock

            settings = Settings()
            embedding_lock = EmbeddingModelLock.load(settings.embedding_lock_path)
            embedding_lock.verify(settings.embedding_model_root)
        except Exception as error:
            if isinstance(error, MemoryError):
                raise
            raise ClickException("live provider configuration is unavailable") from None

    # Build identity before the regular request/config objects.  Replay remains a
    # strict offline path: neither Settings nor live provider modules are imported.
    _profile, profile_id = _build_provider_profile(
        mode=options.mode,
        bundle=replay_bundle,
        settings=settings,
        embedding_lock=embedding_lock,
    )
    runtime_root = _runtime_root(options.checkpoint_db)
    try:
        with _RuntimeLock(runtime_root):
            async with open_sqlite_checkpointer(options.checkpoint_db) as saver:
                checkpoint: CheckpointRef | None = None
                run_id: str
                thread_id: str
                if options.resume_checkpoint is not None and options.resume_thread_id is not None:
                    thread_id = options.resume_thread_id
                    run_id = thread_id
                    saved = await cast("Any", saver).aget_tuple(
                        {
                            "configurable": {
                                "thread_id": thread_id,
                                "checkpoint_ns": "",
                                "checkpoint_id": options.resume_checkpoint,
                            }
                        }
                    )
                    if saved is None:
                        raise ClickException("checkpoint mismatch")
                    try:
                        checkpoint = checkpoint_ref_from_tuple(saved)
                    except Exception as error:
                        if isinstance(error, MemoryError):
                            raise
                        raise ClickException("checkpoint mismatch") from None
                    if (
                        checkpoint.thread_id != thread_id
                        or checkpoint.checkpoint_id != options.resume_checkpoint
                    ):
                        raise ClickException("checkpoint mismatch")
                else:
                    run_id = _new_run_id()
                    thread_id = run_id

                if replay_bundle is not None and options.mode == "replay":
                    composition = _compose_replay_components(
                        bundle=replay_bundle,
                        runtime_root=runtime_root,
                        budget_name=options.budget,
                        provider_profile_id=profile_id,
                        checkpointer=saver,
                        question=options.question,
                        mode=cast("Literal['replay', 'hybrid']", options.mode),
                    )
                elif settings is not None and embedding_lock is not None:
                    composition = _compose_live_components(
                        settings=settings,
                        embedding_lock=embedding_lock,
                        runtime_root=runtime_root,
                        budget_name=options.budget,
                        provider_profile_id=profile_id,
                        checkpointer=saver,
                        question=options.question,
                        mode=cast("Literal['live', 'hybrid']", options.mode),
                        replay_bundle=replay_bundle,
                    )
                else:
                    raise ClickException("provider composition is unavailable")
                try:
                    # Task 8 requires a durable sink, not a best-effort display callback.
                    sink: DurableRunEventSink = _DurableEventFileSink(runtime_root)
                    result = await composition.runner.run(
                        run_id=run_id,
                        thread_id=thread_id,
                        config=composition.config,
                        checkpoint=checkpoint,
                        emit=sink,
                        cancellation_token=CancellationToken(),
                    )
                    if result.status != "completed":
                        raise ClickException("research run failed")
                    report_bytes, evidence_bytes, manifest_bytes = _validate_public_artifacts(
                        composition, result
                    )
                finally:
                    await composition.aclose()
                published = _publish_public_output(
                    output=options.output,
                    report_bytes=report_bytes,
                    evidence_bytes=evidence_bytes,
                    manifest_bytes=manifest_bytes,
                )
                _print_result(result=result, output=published)
    except ClickException:
        raise
    except BlockingIOError:
        raise ClickException("runtime is already in use") from None
    except (ArtifactIntegrityError, OSError, ValueError):
        raise ClickException("research run failed") from None
