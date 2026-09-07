"""``deepresearch experiment`` command group."""

from __future__ import annotations

import asyncio
import stat
import subprocess
from pathlib import Path
from typing import Annotated, Literal

import typer
import yaml
from click import ClickException

from benchmarks.datasets.models import RuntimeTask, TaskCategory
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


def _require_clean_worktree(repo_root: Path) -> None:
    try:
        status = subprocess.run(
            ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=all"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.splitlines()
    except (OSError, subprocess.SubprocessError) as error:
        raise ValueError("unable to verify clean worktree") from error
    if status:
        raise ValueError("worktree must be clean before freezing formal config")


def _require_regular_file(path: Path, *, label: str) -> Path:
    absolute = path.absolute()
    current = absolute
    while current != Path(current.anchor):
        try:
            details = current.lstat()
        except FileNotFoundError:
            details = None
        if details is not None:
            reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
            if stat.S_ISLNK(details.st_mode) or bool(
                getattr(details, "st_file_attributes", 0) & reparse_flag
            ):
                raise ValueError(f"{label} contains a symlink or reparse point")
        current = current.parent
    if not absolute.is_file():
        raise ValueError(f"{label} must be a regular file")
    return absolute


def _oracle_bindings(config: FormalExperimentConfig, repo_root: Path):
    from benchmarks.datasets.models import AnnotatedQuestion, PrivateDatasetManifest
    from benchmarks.datasets.validator import sha256_bytes
    from benchmarks.evaluators.oracle import OracleEvidenceProvider
    from deepresearch.providers.frozen_index import FrozenCorpusSnapshot

    private_root = repo_root / "benchmarks" / "private" / config.dataset_id
    manifest_path = private_root / "private_manifest.json"
    manifest_path = _require_regular_file(
        manifest_path, label="sealed private manifest"
    )
    private_bytes = manifest_path.read_bytes()
    if sha256_bytes(private_bytes) != config.private_manifest_sha256:
        raise ValueError("sealed private manifest hash mismatch")
    private = PrivateDatasetManifest.model_validate_json(private_bytes, strict=True)
    if (private.dataset_id, private.version) != (config.dataset_id, config.dataset_version):
        raise ValueError("sealed private dataset identity mismatch")
    questions: dict[str, AnnotatedQuestion] = {}
    for relative in private.private_test_runtime_files:
        relative_path = Path(relative)
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ValueError("sealed private runtime path is unsafe")
        path = private_root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError("sealed private runtime input is unavailable")
        for line in path.read_bytes().splitlines():
            question = AnnotatedQuestion.model_validate_json(line, strict=True)
            if question.task_id in config.oracle_task_ids:
                questions[question.task_id] = question
    if set(questions) != set(config.oracle_task_ids):
        raise ValueError("sealed ORACLE task inputs are incomplete")
    snapshots = {
        task_id: FrozenCorpusSnapshot.load(
            repo_root / "benchmarks" / "snapshots" / config.dataset_id / task_id,
            task_id=task_id,
        )
        for task_id in config.oracle_task_ids
    }
    approved = {
        task_id: tuple(sorted({span.evidence_id for span in questions[task_id].gold_evidence_spans}))
        for task_id in config.oracle_task_ids
    }
    records = {
        task_id: {record.evidence_id: record for record in snapshots[task_id].records}
        for task_id in config.oracle_task_ids
    }
    provider = OracleEvidenceProvider(
        approved_ids_by_task=approved,
        dataset_version=config.dataset_version,
        private_manifest_sha256=config.private_manifest_sha256,
        evaluator_version=config.evaluator_version,
        evaluation_timestamp=config.evaluation_timestamp,
        formal=True,
        snapshots_by_task=snapshots,
    )
    return provider, records


def _load_sealed_tasks(
    config: FormalExperimentConfig, repo_root: Path
) -> dict[str, RuntimeTask]:
    """Load only the sealed public RuntimeTask projections for agent runs.

    The private manifest is an authorization boundary, not a task source that
    the CLI may choose from. Every task is selected by the frozen config,
    validated against the manifest's fixed runtime roots, and hash-bound to
    ``internal_runtime_task_hashes`` before it reaches the evaluator runner.
    """
    from benchmarks.datasets.models import PrivateDatasetManifest
    from benchmarks.datasets.validator import sha256_bytes
    from experiments.config import canonical_sha256, validate_base_task

    private_root = repo_root / "benchmarks" / "private" / config.dataset_id
    manifest_path = _require_regular_file(
        private_root / "private_manifest.json", label="sealed private manifest"
    )
    manifest_bytes = manifest_path.read_bytes()
    if sha256_bytes(manifest_bytes) != config.private_manifest_sha256:
        raise ValueError("sealed private manifest hash mismatch")
    manifest = PrivateDatasetManifest.model_validate_json(manifest_bytes, strict=True)
    if (manifest.dataset_id, manifest.version) != (
        config.dataset_id,
        config.dataset_version,
    ):
        raise ValueError("sealed private dataset identity mismatch")
    expected_runtime_files = tuple(
        f"runtime/test/{category.value}.jsonl" for category in TaskCategory
    )
    if (
        tuple(manifest.main_test_task_ids) != tuple(config.main_test_task_ids)
        or tuple(manifest.private_test_runtime_files) != expected_runtime_files
    ):
        raise ValueError("sealed private task coverage disagrees with config")

    expected = set(config.main_test_task_ids)
    tasks: dict[str, RuntimeTask] = {}
    for relative in manifest.private_test_runtime_files:
        relative_path = Path(relative)
        if (
            relative_path.is_absolute()
            or ".." in relative_path.parts
            or "\\" in relative
            or relative_path.as_posix() != relative
        ):
            raise ValueError("sealed private runtime path is unsafe")
        runtime_path = _require_regular_file(
            private_root / relative, label="sealed private runtime input"
        )
        for line in runtime_path.read_bytes().splitlines():
            task = RuntimeTask.model_validate_json(line, strict=True)
            if task.task_id not in expected:
                continue
            if task.task_id in tasks:
                raise ValueError("sealed RuntimeTask coverage contains a duplicate")
            validate_base_task(task, config)
            expected_hash = config.internal_runtime_task_hashes.get(task.task_id)
            if expected_hash != canonical_sha256(task.model_dump(mode="json")):
                raise ValueError("sealed RuntimeTask hash mismatch")
            tasks[task.task_id] = task
    if set(tasks) != expected:
        raise ValueError("sealed RuntimeTask inputs are incomplete")
    return tasks


def _runner(
    config: FormalExperimentConfig,
    source: Path,
    *,
    with_oracle: bool = False,
) -> ExperimentRunner:
    repo_root = Path.cwd().resolve()
    tasks = _load_sealed_tasks(config, repo_root)
    kwargs: dict[str, object] = {
        "config_source": source.absolute(),
        "repo_root": repo_root,
        "private_root": repo_root / "benchmarks" / "private" / config.dataset_id,
        "task_loader": tasks,
    }
    if with_oracle:
        provider, records = _oracle_bindings(config, repo_root)
        kwargs["oracle_provider"] = provider
        kwargs["oracle_records_loader"] = records
    return ExperimentRunner(**kwargs)  # type: ignore[arg-type]


@config_app.command("freeze")
def freeze(
    source: Annotated[Path, typer.Option("--source")],
    private_manifest: Annotated[Path, typer.Option("--private-manifest")],
    model_lock: Annotated[Path, typer.Option("--model-lock")],
    r1_model_lock: Annotated[Path, typer.Option("--r1-model-lock")],
    serving_environment_lock: Annotated[Path, typer.Option("--serving-environment-lock")],
    output: Annotated[Path, typer.Option("--output")],
    external_config: Annotated[Path | None, typer.Option("--external-config")] = None,
    external_lock: Annotated[Path | None, typer.Option("--external-lock")] = None,
) -> None:
    """Freeze a formal config from the fixed evaluator roots."""
    try:
        repo_root = Path.cwd().resolve()
        _require_clean_worktree(repo_root)
        private_manifest = _require_regular_file(
            private_manifest, label="sealed private manifest"
        )
        template = load_template(source)
        expected_locks = {
            "model lock": repo_root / template.model_lock_path,
            "R1 model lock": repo_root / template.r1_model_lock_path,
            "serving environment lock": repo_root / template.serving_environment_lock_path,
        }
        supplied_locks = {
            "model lock": model_lock,
            "R1 model lock": r1_model_lock,
            "serving environment lock": serving_environment_lock,
        }
        for label, supplied in supplied_locks.items():
            supplied_path = _require_regular_file(supplied, label=label)
            expected_path = _require_regular_file(expected_locks[label], label=label)
            if supplied_path != expected_path:
                raise ValueError(f"{label} does not match the sealed template reference")
        private_root = private_manifest.parent
        external_pair = (external_config is not None, external_lock is not None)
        if any(external_pair) and not all(external_pair):
            raise ValueError("external config and lock must be supplied together")
        if external_config is not None:
            external_config = _require_regular_file(external_config, label="external config")
            external_lock = _require_regular_file(external_lock, label="external lock")
        config = freeze_config(
            template,
            repo_root=repo_root,
            private_root=private_root,
            external_config_path=external_config,
            external_lock_path=external_lock,
        )
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
            _require_clean_worktree(Path.cwd().resolve())
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
    _run(_runner(loaded, config).run_ranker_component(config=loaded, task_ids=loaded.main_test_task_ids))


@experiment_app.command("run-planner")
def run_planner(
    config: Annotated[Path, typer.Option("--config")],
    ranker: Annotated[Literal["R1", "R2"], typer.Option("--ranker")],
) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded, config).run_planner_policy(config=loaded, task_ids=loaded.main_test_task_ids, ranker_id=ranker))


@experiment_app.command("run")
def run(
    config: Annotated[Path, typer.Option("--config")],
    variants: Annotated[str, typer.Option("--variants")],
) -> None:
    loaded = _load_formal(config)
    requested = tuple(item.strip() for item in variants.split(",") if item.strip())
    if requested != ("A", "B", "C", "D"):
        _fail("--variants must be exactly A,B,C,D")
    _run(_runner(loaded, config).run_abcd(config=loaded))


@experiment_app.command("run-stability")
def run_stability(config: Annotated[Path, typer.Option("--config")]) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded, config).run_variant(ExperimentVariant.D, config=loaded, task_ids=loaded.stability_task_ids))


@experiment_app.command("run-cost-subset")
def run_cost_subset(config: Annotated[Path, typer.Option("--config")]) -> None:
    loaded = _load_formal(config)
    _run(_runner(loaded, config).run_cost_subset(config=loaded))


@experiment_app.command("run-reference")
def run_reference(
    config: Annotated[Path, typer.Option("--config")],
    variants: Annotated[str, typer.Option("--variants")],
) -> None:
    loaded = _load_formal(config)
    requested = {item.strip() for item in variants.split(",") if item.strip()}
    if requested != {"P0", "ORACLE"}:
        _fail("--variants must be exactly P0,ORACLE")
    _run(
        _runner(loaded, config, with_oracle=True).run_reference(
            config=loaded, task_ids=loaded.p0_task_ids
        )
    )


@experiment_app.command("run-external")
def run_external(
    config: Annotated[Path, typer.Option("--config")],
    external_config: Annotated[Path, typer.Option("--external-config")],
    external_lock: Annotated[Path, typer.Option("--external-lock")],
    benchmarks: Annotated[str, typer.Option("--benchmarks")],
) -> None:
    """Run the evaluator-only Portfolio external benchmark extension."""

    loaded = _load_formal(config)
    requested = tuple(item.strip() for item in benchmarks.split(",") if item.strip())
    allowed = {"livedrbench", "frames", "deepresearchbench"}
    if not requested or len(set(requested)) != len(requested) or not set(requested) <= allowed:
        _fail("--benchmarks must contain unique external benchmark names")
    from experiments.external_runner import ExternalExperimentRunner

    _run(
        ExternalExperimentRunner(
            repo_root=Path.cwd().resolve(),
            config_source=config.absolute(),
        ).run(
            config=loaded,
            external_config_path=external_config,
            external_lock_path=external_lock,
            benchmarks=requested,  # type: ignore[arg-type]
        )
    )


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
