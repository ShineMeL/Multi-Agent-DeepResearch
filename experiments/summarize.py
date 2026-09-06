"""Safe aggregate outputs for formal experiment groups.

The summarizer consumes only public ``ExperimentTaskRun`` records and
hash-only evaluator reference metadata.  It never opens private gold, prompts,
model responses or credentials.
"""

from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import cast

from experiments.models import ExperimentTaskRun

_OUTPUTS = (
    "summary.json",
    "task_metrics.jsonl",
    "confidence_intervals.json",
    "pareto.json",
    "failures.jsonl",
)


def _canonical(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
        .encode("utf-8")
        + b"\n"
    )


def _write_reusable(path: Path, payload: bytes) -> None:
    if path.is_symlink():
        raise FileExistsError(path)
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.staging")
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
        staging.rename(path)
    finally:
        staging.unlink(missing_ok=True)


def _jsonl(rows: Sequence[Mapping[str, object]]) -> bytes:
    if not rows:
        return b""
    return b"".join(_canonical(row) for row in rows)


def _load_runs(experiment_dir: Path) -> tuple[ExperimentTaskRun, ...]:
    raw = experiment_dir / "raw"
    runs: list[ExperimentTaskRun] = []
    seen: set[str] = set()
    if not raw.is_dir():
        return ()
    for path in sorted(raw.glob("*.json")):
        key = path.stem
        if key in seen:
            raise ValueError("duplicate idempotency key")
        seen.add(key)
        runs.append(ExperimentTaskRun.model_validate_json(path.read_bytes(), strict=True))
    return tuple(runs)


def _safe_run_row(run: ExperimentTaskRun) -> dict[str, object]:
    return {
        "task_id": run.task_id,
        "protocol": run.protocol,
        "variant": run.variant.value,
        "planner_id": run.planner_id,
        "ranker_id": run.ranker_id,
        "budget_preset": run.budget_preset,
        "seed": run.seed,
        "repeat_id": run.repeat_id,
        "status": run.status,
        "validity": run.validity,
        "error_code": run.error_code,
        "metrics": {
            "input_tokens": run.usage.input_tokens,
            "output_tokens": run.usage.output_tokens,
            "total_tokens": run.usage.total_tokens,
            "search_calls": run.usage.search_calls,
            "pages": run.usage.pages,
            "wall_seconds": run.usage.wall_seconds,
            "cost_usd": None if run.usage.cost_usd is None else str(run.usage.cost_usd),
        },
    }


def _manifest_payload(experiment_dir: Path, files: tuple[str, ...]) -> bytes:
    hashes = {
        name: hashlib.sha256((experiment_dir / name).read_bytes()).hexdigest()
        for name in files
        if (experiment_dir / name).is_file()
    }
    return _canonical({"schema_version": "experiment-artifact-manifest-v1", "files": hashes})


def _verify_manifest(experiment_dir: Path) -> None:
    manifest_path = experiment_dir / "manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError("experiment manifest is missing")
    payload = json.loads(manifest_path.read_bytes())
    if not isinstance(payload, dict):
        raise TypeError("experiment manifest is invalid")
    raw_manifest = cast(dict[str, object], payload)
    if raw_manifest.get("schema_version") != "experiment-artifact-manifest-v1":
        raise ValueError("experiment manifest is invalid")
    files = raw_manifest.get("files")
    if not isinstance(files, dict):
        raise TypeError("experiment manifest file list is invalid")
    typed_files = cast(dict[object, object], files)
    if set(typed_files) != set(_OUTPUTS):
        raise ValueError("experiment manifest is incomplete")
    for name, expected in typed_files.items():
        if not isinstance(name, str) or Path(name).name != name or not isinstance(expected, str):
            raise ValueError("experiment manifest contains unsafe entries")
        path = experiment_dir / name
        if not path.is_file() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError("experiment artifact hash mismatch")


def summarize_experiment(
    experiment_dir: Path,
    *,
    bootstrap_resamples: int = 10_000,
    verify_only: bool = False,
) -> dict[str, object]:
    """Build or verify the safe aggregate files for one group."""
    if bootstrap_resamples <= 0:
        raise ValueError("bootstrap_resamples must be positive")
    root = Path(experiment_dir).resolve()
    group_path = root / "group.json"
    if not group_path.is_file():
        raise ValueError("experiment group metadata is missing")
    group = json.loads(group_path.read_bytes())
    if not isinstance(group, dict):
        raise TypeError("experiment group metadata is invalid")
    raw_group = cast(dict[str, object], group)
    if not isinstance(raw_group.get("group_id"), str):
        raise TypeError("experiment group metadata is invalid")
    group_id = cast(str, group["group_id"])
    if verify_only:
        _verify_manifest(root)
        # Verification intentionally does not parse private/evaluator-only
        # files or rewrite any output.
        _load_runs(root)
        return {"group_id": group_id, "verified": True}

    runs = _load_runs(root)
    sections: dict[str, dict[str, object]] = {}
    grouped: defaultdict[str, list[ExperimentTaskRun]] = defaultdict(list)
    for run in runs:
        grouped[run.protocol].append(run)
    for protocol in ("ranker_component", "planner_policy", "end_to_end", "reference"):
        items = grouped.get(protocol, [])
        sections[protocol] = {
            "run_count": len(items),
            "task_count": len({item.task_id for item in items}),
            "variants": dict(sorted(Counter(item.variant.value for item in items).items())),
            "budgets": dict(sorted(Counter(item.budget_preset for item in items).items())),
            "statuses": dict(sorted(Counter(item.status for item in items).items())),
        }
    task_rows = [_safe_run_row(run) for run in runs]
    failures = [
        {
            "task_id": run.task_id,
            "protocol": run.protocol,
            "variant": run.variant.value,
            "error_code": run.error_code,
            "status": run.status,
        }
        for run in runs
        if run.status != "completed" or run.validity != "valid"
    ]
    summary: dict[str, object] = {
        "schema_version": "experiment-summary-v1",
        "group_id": group_id,
        "bootstrap_resamples": bootstrap_resamples,
        "sections": sections,
        "run_count": len(runs),
        "reference_note": "ORACLE is evaluator-only and incomparable to agent cost/latency traces.",
    }
    outputs = {
        "summary.json": _canonical(summary),
        "task_metrics.jsonl": _jsonl(task_rows),
        "confidence_intervals.json": _canonical({"schema_version": "confidence-intervals-v1", "sections": {}}),
        "pareto.json": _canonical({"schema_version": "pareto-v1", "sections": {}}),
        "failures.jsonl": _jsonl(failures),
    }
    for name, payload in outputs.items():
        _write_reusable(root / name, payload)
    manifest_payload = _manifest_payload(root, _OUTPUTS)
    _write_reusable(root / "manifest.sha256", manifest_payload)
    return summary


__all__ = ["summarize_experiment"]
