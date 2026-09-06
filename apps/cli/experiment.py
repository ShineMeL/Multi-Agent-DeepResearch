"""``deepresearch experiment`` command group."""

from __future__ import annotations

import asyncio
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from click import ClickException

from experiments.config import (
    FormalExperimentConfig,
    freeze_config,
    load_config,
    load_template,
)
from experiments.models import ExperimentVariant
from experiments.runner import ExperimentRunner
from experiments.summarize import summarize_experiment

experiment_app = typer.Typer(no_args_is_help=True)
config_app = typer.Typer(no_args_is_help=True)
experiment_app.add_typer(config_app, name="config")


def _fail(message: str) -> None:
    raise typer.BadParameter(message)


def _load_formal(path: Path) -> FormalExperimentConfig:
    try:
        return load_config(path)
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        raise typer.BadParameter("formal experiment config is invalid") from None


def _runner(config: FormalExperimentConfig) -> ExperimentRunner:
    return ExperimentRunner(config_source=None)


@config_app.command("freeze")
def freeze(
    source: Annotated[Path, typer.Option("--source")],
    private_manifest: Annotated[Path, typer.Option("--private-manifest")],
    model_lock: Annotated[Path, typer.Option("--model-lock")],
    r1_model_lock: Annotated[Path, typer.Option("--r1-model-lock")],
    serving_environment_lock: Annotated[Path, typer.Option("--serving-environment-lock")],
    output: Annotated[Path, typer.Option("--output")],
) -> None:
    """Freeze a formal config from the fixed evaluator roots."""
    del model_lock, r1_model_lock, serving_environment_lock
    try:
        repo_root = Path.cwd().resolve()
        template = load_template(source)
        private_root = private_manifest.resolve().parent
        config = freeze_config(template, repo_root=repo_root, private_root=private_root)
        # freeze_config itself writes atomically and refuses replacement when
        # an output is provided; keep the CLI output separate for clear error
        # handling and exact canonical bytes.
        if output.exists():
            raise FileExistsError(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        from benchmarks.datasets.validator import canonical_json_bytes
        from experiments.config import write_immutable

        write_immutable(output, canonical_json_bytes(config.model_dump(mode="json")))
        typer.echo(config.experiment_group_id())
    except typer.BadParameter:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        raise ClickException("formal config freeze failed") from None


@config_app.command("validate")
def validate(
    config: Annotated[Path, typer.Option("--config")],
    require_clean_worktree: Annotated[bool, typer.Option("--require-clean-worktree")] = False,
    verify_current_tree: Annotated[bool, typer.Option("--verify-current-tree")] = False,
) -> None:
    try:
        loaded = _load_formal(config)
        if require_clean_worktree:
            status = subprocess.run(
                ["git", "status", "--porcelain"], capture_output=True, text=True, check=True
            ).stdout.splitlines()
            if any(not line.endswith("experiments/") for line in status):
                raise ValueError("worktree is not clean")
        if verify_current_tree:
            from experiments.config import code_tree_sha256

            if code_tree_sha256(Path.cwd()) != loaded.code_tree_sha256:
                raise ValueError("current code tree does not match sealed config")
        typer.echo("valid")
    except typer.BadParameter:
        raise
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        raise ClickException("formal config is invalid") from None


@config_app.command("id")
def config_id(config: Annotated[Path, typer.Option("--config")]) -> None:
    typer.echo(_load_formal(config).experiment_group_id())


def _run(coro: object) -> object:
    return asyncio.run(coro)  # type: ignore[arg-type]


@experiment_app.command("run-ranker")
def run_ranker(config: Annotated[Path, typer.Option("--config")]) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded).run_ranker_component(config=loaded, task_ids=loaded.main_test_task_ids))


@experiment_app.command("run-planner")
def run_planner(
    config: Annotated[Path, typer.Option("--config")],
    ranker: Annotated[Literal["R1", "R2"], typer.Option("--ranker")],
) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded).run_planner_policy(config=loaded, task_ids=loaded.main_test_task_ids, ranker_id=ranker))


@experiment_app.command("run")
def run(
    config: Annotated[Path, typer.Option("--config")],
    variants: Annotated[str, typer.Option("--variants")],
) -> None:
    loaded = _load_formal(config)
    requested = tuple(item.strip() for item in variants.split(",") if item.strip())
    if requested != ("A", "B", "C", "D"):
        _fail("--variants must be exactly A,B,C,D")
    _run(_runner(loaded).run_abcd(config=loaded))


@experiment_app.command("run-stability")
def run_stability(config: Annotated[Path, typer.Option("--config")]) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded).run_variant(ExperimentVariant.D, config=loaded, task_ids=loaded.stability_task_ids))


@experiment_app.command("run-cost-subset")
def run_cost_subset(config: Annotated[Path, typer.Option("--config")]) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded).run_cost_subset(config=loaded))


@experiment_app.command("run-reference")
def run_reference(
    config: Annotated[Path, typer.Option("--config")],
    variants: Annotated[str, typer.Option("--variants")],
) -> None:
    loaded = _load_formal(config)
    requested = {item.strip() for item in variants.split(",") if item.strip()}
    if requested != {"P0", "ORACLE"}:
        _fail("--variants must be exactly P0,ORACLE")
    _run(_runner(loaded).run_reference(config=loaded, task_ids=loaded.p0_task_ids))


@experiment_app.command("summarize")
def summarize(
    experiment_dir: Annotated[Path, typer.Option("--experiment-dir")],
    bootstrap_resamples: Annotated[int, typer.Option("--bootstrap-resamples")] = 10_000,
    verify_only: Annotated[bool, typer.Option("--verify-only")] = False,
) -> None:
    try:
        result = summarize_experiment(
            experiment_dir, bootstrap_resamples=bootstrap_resamples, verify_only=verify_only
        )
        typer.echo(yaml.safe_dump(result, sort_keys=True).strip())
    except (OSError, TypeError, ValueError, yaml.YAMLError):
        raise ClickException("experiment summary failed") from None


__all__ = ["experiment_app"]
