import asyncio
import hashlib
import json
import os
import stat
import subprocess
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Literal
from urllib.parse import urlsplit

import typer

from deepresearch.config import Settings
from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunBudget, RunConfig
from deepresearch.providers.embeddings import EmbeddingModelLock
from deepresearch.providers.replay_schema import (
    REPLAY_BUNDLE_SCHEMA_VERSION,
    REPLAY_FILES,
    ReplayBundle,
    ReplayProviderSnapshot,
    canonical_json_bytes,
)
from deepresearch.retrieval import URLSecurityError, canonicalize_url

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
    if replay_root is not None:
        if _is_link_or_reparse(replay_root) or not replay_root.exists() or not replay_root.is_dir():
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
    return profile, f"baseline-{mode}-v1:{digest}"


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
    except typer.ClickException:
        raise
    except Exception as error:  # noqa: BLE001 - convert to a stable CLI boundary
        raise typer.ClickException("research run failed") from error


async def _research_async(options: _ResearchOptions) -> None:
    del options
    raise typer.ClickException("research composition is not available")
