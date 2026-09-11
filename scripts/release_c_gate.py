"""Fail-closed orchestration for the Release C benchmark gates.

This module validates sealed inputs before delegating any experiment command.
It intentionally does not freeze missing configuration, download data, invent
ratings, or rewrite an unsealed results page.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, cast

# ``python scripts/release_c_gate.py`` is a documented invocation.  Python
# otherwise puts only ``scripts/`` on ``sys.path`` for a direct file launch.
if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmarks.scripts.render_results import ResultValidationError, render_results
from scripts.release_preflight import GateReport, assess_gate

CStage = Literal["c1", "c2", "c3", "c4"]
CStepStatus = Literal["ready", "blocked", "failed"]
CRunStatus = Literal["ready", "blocked", "failed"]


@dataclass(frozen=True)
class CStep:
    """One redacted operation in a Release C gate."""

    name: str
    status: CStepStatus
    detail: str


@dataclass(frozen=True)
class CRunResult:
    """Immutable, secret-free result of a Release C stage."""

    status: CRunStatus
    reason: str | None
    steps: tuple[CStep, ...]


Runner = Callable[..., subprocess.CompletedProcess[str] | None]
_FORMAL_CONFIG = Path("benchmarks/configs/formal.yaml")
_PORTFOLIO_CONFIG = Path("benchmarks/configs/formal-portfolio.yaml")
_EXTERNAL_CONFIG = Path("benchmarks/configs/external.yaml")
_EXTERNAL_LOCK = Path("benchmarks/external/external.lock.json")
_EXTERNAL_COUNTS = {"livedrbench": 10, "frames": 20, "deepresearchbench": 10}
_HASH_RE = frozenset("0123456789abcdef")
_PUBLICATION_FILES = (
    Path("results.md"),
    Path("evaluation.md"),
    Path("assets/results/abcd-metrics.svg"),
    Path("assets/results/citation-support-vs-usd.svg"),
    Path("assets/results/completeness-vs-search.svg"),
)


def _blocked(report: GateReport) -> CRunResult:
    steps = tuple(
        CStep(check.name, "ready" if check.present else "blocked", check.detail)
        for check in report.checks
    )
    return CRunResult("blocked", report.reason, steps)


def _invoke(
    runner: Runner,
    command: Sequence[str],
    repository: Path,
) -> subprocess.CompletedProcess[str] | None:
    try:
        return runner(
            list(command),
            cwd=str(repository),
            check=False,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _detail(result: subprocess.CompletedProcess[str] | None) -> str:
    if result is None:
        return "unavailable"
    return "completed" if result.returncode == 0 else f"exit_{result.returncode}"


def _run_c2(repository: Path, experiment_dir: Path, runner: Runner) -> CRunResult:
    commands: tuple[tuple[str, tuple[str, ...]], ...] = (
        (
            "formal_validate",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "config",
                "validate",
                "--config",
                _FORMAL_CONFIG.as_posix(),
                "--require-clean-worktree",
                "--verify-current-tree",
            ),
        ),
        (
            "ranker_execution",
            ("uv", "run", "deepresearch", "experiment", "run-ranker", "--config", _FORMAL_CONFIG.as_posix()),
        ),
        (
            "planner_execution",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "run-planner",
                "--config",
                _FORMAL_CONFIG.as_posix(),
                "--ranker",
                "R1",
            ),
        ),
        (
            "abcd_execution",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "run",
                "--config",
                _FORMAL_CONFIG.as_posix(),
                "--variants",
                "A,B,C,D",
            ),
        ),
        (
            "stability_execution",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "run-stability",
                "--config",
                _FORMAL_CONFIG.as_posix(),
            ),
        ),
        (
            "cost_subset_execution",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "run-cost-subset",
                "--config",
                _FORMAL_CONFIG.as_posix(),
            ),
        ),
        (
            "reference_execution",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "run-reference",
                "--config",
                _FORMAL_CONFIG.as_posix(),
                "--variants",
                "P0,ORACLE",
            ),
        ),
        (
            "formal_summary",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "summarize",
                "--experiment-dir",
                str(experiment_dir),
                "--bootstrap-resamples",
                "10000",
            ),
        ),
        (
            "formal_summary_verify",
            (
                "uv",
                "run",
                "deepresearch",
                "experiment",
                "summarize",
                "--experiment-dir",
                str(experiment_dir),
                "--bootstrap-resamples",
                "10000",
                "--verify-only",
            ),
        ),
    )
    steps: list[CStep] = []
    for name, command in commands:
        result = _invoke(runner, command, repository)
        if result is None or result.returncode != 0:
            steps.append(CStep(name, "failed", _detail(result)))
            return CRunResult("failed", "FORMAL_EXECUTION_FAILED", tuple(steps))
        steps.append(CStep(name, "ready", "completed"))
    return CRunResult("ready", None, tuple(steps))


def _valid_hash(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and value != "0" * 64
        and set(value) <= _HASH_RE
    )


def _read_json(path: Path) -> object | None:
    try:
        return json.loads(path.read_bytes())
    except (OSError, UnicodeError, TypeError, ValueError):
        return None


def _manifest_hash(path: Path, *, allowed_schemas: frozenset[str]) -> bool:
    manifest_path = path.parent / "manifest.sha256"
    manifest = _read_json(manifest_path)
    if not isinstance(manifest, Mapping):
        return False
    raw_manifest = cast(Mapping[object, object], manifest)
    if raw_manifest.get("schema_version") not in allowed_schemas:
        return False
    files = raw_manifest.get("files")
    if not isinstance(files, Mapping):
        return False
    raw_files = cast(Mapping[object, object], files)
    expected = raw_files.get(path.name)
    if not _valid_hash(expected):
        return False
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest() == expected
    except OSError:
        return False


def _validate_external(experiment_dir: Path) -> bool:
    metrics = experiment_dir / "metrics.json"
    if not metrics.is_file() or metrics.is_symlink() or not _manifest_hash(
        metrics, allowed_schemas=frozenset({"external-result-manifest-v1"})
    ):
        return False
    payload = _read_json(metrics)
    if not isinstance(payload, Mapping):
        return False
    raw_payload = cast(Mapping[object, object], payload)
    if raw_payload.get("schema_version") != "external-metrics-v1":
        return False
    counts = raw_payload.get("benchmark_counts")
    if not isinstance(counts, Mapping):
        return False
    normalised: dict[str, int] = {}
    for name, value in cast(Mapping[object, object], counts).items():
        if not isinstance(name, str) or isinstance(value, bool) or not isinstance(value, int) or value < 0:
            return False
        normalised[name] = value
    if normalised != _EXTERNAL_COUNTS:
        return False
    return _valid_hash(raw_payload.get("formal_config_sha256")) and _valid_hash(
        raw_payload.get("external_lock_sha256")
    )


def _human_sidecar_valid(path: Path) -> bool:
    return _manifest_hash(
        path,
        allowed_schemas=frozenset(
            {"benchmark-aggregate-manifest-v1", "external-result-manifest-v1"}
        ),
    )


def _validate_human(path: Path) -> bool:
    if not path.is_file() or path.is_symlink() or not _human_sidecar_valid(path):
        return False
    payload = _read_json(path)
    ratings: object = payload
    if isinstance(payload, Mapping):
        ratings = cast(Mapping[object, object], payload).get("ratings")
    if not isinstance(ratings, Sequence) or isinstance(ratings, (str, bytes, bytearray)):
        return False
    try:
        from benchmarks.evaluators.human import validate_human_ratings

        validation = validate_human_ratings(
            cast(Sequence[Mapping[str, object]], ratings),
            expected_tasks=20,
            raters_per_task=3,
        )
    except (TypeError, ValueError, RuntimeError):
        return False
    return bool(validation.valid)


def _validate_human_seal(path: Path) -> bool:
    """Validate a renderer-facing aggregate without assuming raw ratings shape."""

    if not path.is_file() or path.is_symlink() or not _human_sidecar_valid(path):
        return False
    return _read_json(path) is not None


def _validate_external_inputs(repository: Path) -> bool:
    config = repository / _EXTERNAL_CONFIG
    lock = repository / _EXTERNAL_LOCK
    try:
        text = config.read_text(encoding="utf-8").casefold()
    except (OSError, UnicodeError):
        return False
    if "example.invalid" in text or "pending" in text:
        return False
    if not lock.is_file() or lock.is_symlink():
        return False
    try:
        from benchmarks.external import load_external_config, load_external_lock

        load_external_config(config)
        load_external_lock(lock)
    except (ImportError, OSError, TypeError, ValueError, RuntimeError):
        return False
    return True


def _run_c3(
    repository: Path,
    *,
    external_experiment_dir: Path | None,
    human_summary: Path | None,
) -> CRunResult:
    if not _validate_external_inputs(repository):
        return CRunResult(
            "blocked",
            "PORTFOLIO_INPUT_MISSING",
            (CStep("external_license_and_lock", "blocked", "missing_or_unverified"),),
        )
    if external_experiment_dir is None or not _validate_external(external_experiment_dir):
        return CRunResult(
            "blocked",
            "PORTFOLIO_INPUT_MISSING",
            (CStep("external_10_20_10", "blocked", "missing_or_invalid"),),
        )
    if human_summary is None or not _validate_human(human_summary):
        return CRunResult(
            "blocked",
            "HUMAN_AGGREGATE_INCOMPLETE",
            (CStep("human_20x3", "blocked", "missing_or_invalid"),),
        )
    return CRunResult(
        "ready",
        None,
        (
            CStep("external_10_20_10", "ready", "verified"),
            CStep("human_20x3", "ready", "verified"),
        ),
    )


def _tree_hashes(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for relative in _PUBLICATION_FILES:
        path = root / relative
        if path.is_file() and not path.is_symlink():
            values[relative.as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return values


def _promote_publication(staging: Path, docs_root: Path) -> None:
    for relative in _PUBLICATION_FILES:
        source = staging / relative
        if not source.is_file() or source.is_symlink():
            raise ResultValidationError(f"renderer omitted {relative.as_posix()}")
    docs_root.mkdir(parents=True, exist_ok=True)
    for relative in _PUBLICATION_FILES:
        target = docs_root / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.is_symlink():
            raise ResultValidationError(f"publication target is a symlink: {relative.as_posix()}")
        os.replace(staging / relative, target)


def _run_c4(
    repository: Path,
    *,
    experiment_dir: Path | None,
    external_experiment_dir: Path | None,
    human_summary: Path | None,
) -> CRunResult:
    if experiment_dir is None:
        return CRunResult("blocked", "PUBLICATION_UNSEALED", (CStep("primary_inputs", "blocked", "missing"),))
    summary = experiment_dir / "summary.json"
    manifest = experiment_dir / "manifest.sha256"
    if (
        not summary.is_file()
        or summary.is_symlink()
        or not manifest.is_file()
        or manifest.is_symlink()
    ):
        return CRunResult(
            "blocked",
            "PUBLICATION_UNSEALED",
            (CStep("primary_manifest", "blocked", "missing_or_invalid"),),
        )
    if external_experiment_dir is not None and not _validate_external(external_experiment_dir):
        return CRunResult(
            "blocked",
            "PUBLICATION_UNSEALED",
            (CStep("external_manifest", "blocked", "missing_or_invalid"),),
        )
    if human_summary is not None and not _validate_human_seal(human_summary):
        return CRunResult(
            "blocked",
            "PUBLICATION_UNSEALED",
            (CStep("human_manifest", "blocked", "missing_or_invalid"),),
        )
    docs_root = repository / "docs"
    try:
        with tempfile.TemporaryDirectory(prefix="deepresearch-publication-") as temporary:
            staging_root = Path(temporary)
            first = staging_root / "first"
            second = staging_root / "second"
            render_results(
                experiment_dir=experiment_dir,
                external_experiment_dir=external_experiment_dir,
                human_summary=human_summary,
                docs_dir=first,
            )
            render_results(
                experiment_dir=experiment_dir,
                external_experiment_dir=external_experiment_dir,
                human_summary=human_summary,
                docs_dir=second,
            )
            first_hashes = _tree_hashes(first)
            second_hashes = _tree_hashes(second)
            if first_hashes != second_hashes:
                return CRunResult(
                    "failed",
                    "PUBLICATION_NONDETERMINISTIC",
                    (CStep("render_determinism", "failed", "hash_mismatch"),),
                )
            _promote_publication(first, docs_root)
    except (OSError, TypeError, ValueError, ResultValidationError):
        return CRunResult(
            "failed",
            "PUBLICATION_VALIDATION_FAILED",
            (CStep("render_validation", "failed", "rejected"),),
        )
    return CRunResult(
        "ready",
        None,
        (
            CStep("render_first", "ready", "verified"),
            CStep("render_second", "ready", "verified"),
            CStep("publication_promote", "ready", "promoted"),
        ),
    )


def run_c_gate(
    repository: Path,
    *,
    stage: CStage,
    experiment_dir: Path | None = None,
    external_experiment_dir: Path | None = None,
    human_summary: Path | None = None,
    runner: Runner = subprocess.run,
) -> CRunResult:
    """Run a Release C stage only after its capability preflight is ready."""

    root = repository.resolve()
    if stage == "c2" and experiment_dir is None:
        return CRunResult(
            "blocked",
            "FORMAL_INPUT_MISSING",
            (CStep("experiment_dir", "blocked", "missing"),),
        )
    report = assess_gate(
        root,
        stage,
        experiment_dir=experiment_dir,
        external_experiment_dir=external_experiment_dir,
        human_summary=human_summary,
    )
    if report.status != "ready":
        return _blocked(report)
    if stage == "c1":
        return CRunResult("ready", None, (CStep("formal_inputs", "ready", "verified"),))
    if stage == "c2":
        assert experiment_dir is not None
        return _run_c2(root, experiment_dir, runner)
    if stage == "c3":
        return _run_c3(
            root,
            external_experiment_dir=external_experiment_dir,
            human_summary=human_summary,
        )
    return _run_c4(
        root,
        experiment_dir=experiment_dir,
        external_experiment_dir=external_experiment_dir,
        human_summary=human_summary,
    )


def _payload(result: CRunResult) -> dict[str, object]:
    return cast(dict[str, object], asdict(result))


def _parse_stage(value: str) -> CStage:
    if value not in {"c1", "c2", "c3", "c4"}:
        raise argparse.ArgumentTypeError("stage must be c1, c2, c3, or c4")
    return cast(CStage, value)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--stage", type=_parse_stage, required=True)
    parser.add_argument("--experiment-dir", type=Path)
    parser.add_argument("--external-experiment-dir", type=Path)
    parser.add_argument("--human-summary", type=Path)
    parser.add_argument("--format", choices=("text", "json"), default="text")
    arguments = parser.parse_args(argv)
    result = run_c_gate(
        arguments.repository,
        stage=arguments.stage,
        experiment_dir=arguments.experiment_dir,
        external_experiment_dir=arguments.external_experiment_dir,
        human_summary=arguments.human_summary,
    )
    if arguments.format == "json":
        print(json.dumps(_payload(result), ensure_ascii=False, sort_keys=True, separators=(",", ":")))
    else:
        print(f"release_c stage={arguments.stage} status={result.status} reason={result.reason or 'none'}")
        for step in result.steps:
            print(f"  step={step.name} status={step.status} detail={step.detail}")
    return 0 if result.status == "ready" else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["CRunResult", "CStage", "CStep", "main", "run_c_gate"]
