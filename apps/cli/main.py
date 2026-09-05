import asyncio
import os
import stat
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Literal

import typer

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
