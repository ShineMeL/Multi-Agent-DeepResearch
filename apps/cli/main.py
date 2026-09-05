from pathlib import Path
from typing import Annotated, Literal

import typer

from deepresearch import __version__

app = typer.Typer(no_args_is_help=True)


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
    del (
        question,
        mode,
        replay_root,
        record_replay_root,
        checkpoint_db,
        resume_checkpoint,
        resume_thread_id,
        budget,
        output,
    )
    raise typer.ClickException("research composition is not available")
