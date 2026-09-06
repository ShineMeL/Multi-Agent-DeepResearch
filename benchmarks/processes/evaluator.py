from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import uuid
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from benchmarks.datasets.isolation import GoldAccessViolation
from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import sha256_bytes
from experiments.config import FormalExperimentConfig, authorized_staged_task
from experiments.models import SealedModel, Sha256, canonical_sha256


class StagedRuntimeTask(SealedModel):
    """Hash binding for the immutable task copy passed to an agent child."""

    runtime_task_path: str
    runtime_task_sha256: Sha256
    base_runtime_task_sha256: Sha256


def _publish_no_replace_bytes(path: Path, payload: bytes) -> Path:
    """Publish bytes without replacing an existing artifact."""
    path = Path(path)
    if path.is_symlink():
        raise GoldAccessViolation("artifact destination cannot be a symlink")
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return path
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.{uuid.uuid4().hex}.staging")
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        from benchmarks.scripts.build_snapshot import (  # pyright: ignore[reportPrivateUsage]
            _publish_no_replace,  # pyright: ignore[reportPrivateUsage]
        )

        _publish_no_replace(staging, path)  # pyright: ignore[reportPrivateUsage]
        return path
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def materialize_agent_runtime_task(
    task: RuntimeTask,
    *,
    agent_input_root: Path,
    request_id: str,
    forbidden_private_root: Path,
) -> Path:
    if type(task) is not RuntimeTask:
        raise TypeError("task must be a RuntimeTask")
    if type(request_id) is not str or not request_id.strip():
        raise ValueError("request_id must be non-empty")
    if Path(request_id).name != request_id or request_id in {".", ".."}:
        raise GoldAccessViolation("request_id must not contain path separators")

    private_root = Path(forbidden_private_root).resolve()
    input_root = Path(agent_input_root).resolve()
    if input_root == private_root or private_root in input_root.parents:
        raise GoldAccessViolation("agent input root must be outside private benchmark path")
    input_root.mkdir(parents=True, exist_ok=True)
    target = input_root / f"{request_id}.json"
    if private_root == target or private_root in target.parents:
        raise GoldAccessViolation("agent task destination is evaluator-only")
    payload = _canonical_json(task.model_dump(mode="json"))
    _publish_no_replace_bytes(target, payload)
    with target.open("rb") as handle:
        if handle.read() != payload:
            raise GoldAccessViolation("materialized runtime task failed byte validation")
    restored = RuntimeTask.model_validate_json(payload, strict=True)
    if _canonical_json(restored.model_dump(mode="json")) != payload:
        raise GoldAccessViolation("materialized runtime task failed schema validation")
    return target


def stage_sealed_config(
    source_path: Path,
    *,
    expected_sha256: str,
    group_run_root: Path,
) -> Path:
    """Copy a verified repository seal into the ignored group input root."""
    source = Path(source_path).resolve(strict=True)
    if not source.is_file() or source.is_symlink():
        raise GoldAccessViolation("sealed config source is not a regular file")
    payload = source.read_bytes()
    if sha256_bytes(payload) != expected_sha256:
        raise GoldAccessViolation("sealed config source hash mismatch")
    destination = Path(group_run_root).resolve() / "config" / "formal.yaml"
    return _publish_no_replace_bytes(destination, payload).resolve()


def stage_authorized_runtime_task(
    base_task: RuntimeTask,
    *,
    config: FormalExperimentConfig,
    budget_preset: str,
    agent_input_root: Path,
    request_id: str,
    forbidden_private_root: Path,
) -> StagedRuntimeTask:
    """Create the selected budget copy and bind it to the sealed base hash."""
    if budget_preset not in config.budget_sensitivity_presets:
        raise GoldAccessViolation("budget preset is not sealed for this experiment")
    base_hash = canonical_sha256(base_task.model_dump(mode="json"))
    authorized = config.internal_runtime_task_hashes.get(base_task.task_id)
    if authorized is None:
        authorized = config.external_runtime_task_hashes.get(base_task.task_id)
    if authorized != base_hash:
        raise GoldAccessViolation("base RuntimeTask is not authorized by the sealed config")
    selected = base_task.model_copy(
        update={"request": base_task.request.model_copy(update={"budget_preset": budget_preset})}
    )
    authorized_staged_task(
        config,
        selected,
        staged_sha256=canonical_sha256(selected.model_dump(mode="json")),
        budget_preset=budget_preset,  # type: ignore[arg-type]
    )
    staged_path = materialize_agent_runtime_task(
        selected,
        agent_input_root=agent_input_root,
        request_id=request_id,
        forbidden_private_root=forbidden_private_root,
    )
    return StagedRuntimeTask(
        runtime_task_path=str(staged_path.resolve()),
        runtime_task_sha256=sha256_bytes(staged_path.read_bytes()),
        base_runtime_task_sha256=base_hash,
    )


def _safe_environment(*, root: Path, runtime_root: Path, snapshot_root: Path, run_root: Path) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SystemRoot", "TEMP", "TMP", "PYTHONIOENCODING")
        if key in os.environ
    }
    environment["PYTHONPATH"] = os.pathsep.join([str(root), str(root / "src")])
    environment["DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT"] = str(runtime_root)
    environment["DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT"] = str(snapshot_root)
    environment["DEEPRESEARCH_BENCHMARK_RUN_ROOT"] = str(run_root)
    environment.pop("DEEPRESEARCH_BENCHMARK_GOLD_ROOT", None)
    for key in tuple(os.environ):
        if key.startswith("DEEPRESEARCH_PROVIDER_"):
            environment[key] = os.environ[key]
    return environment


def _launch_probe(
    *,
    task_path: Path,
    snapshot_root: Path,
    run_root: Path,
    root: Path,
) -> Path:
    output = run_root / "probe.json"
    command = [
        sys.executable,
        "-m",
        "benchmarks.processes.agent",
        "--probe-runtime-task",
        str(task_path),
        "--runtime-root",
        str(task_path.parent),
        "--snapshot-dir",
        str(snapshot_root),
        "--run-root",
        str(run_root),
        "--output",
        str(output),
    ]
    completed = subprocess.run(
        command,
        cwd=root,
        env=_safe_environment(
            root=root,
            runtime_root=task_path.parent,
            snapshot_root=snapshot_root,
            run_root=run_root,
        ),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError("agent probe failed")
    if not output.is_file():
        raise RuntimeError("agent probe did not produce an output")
    output_bytes = output.read_bytes()
    try:
        output_payload = json.loads(output_bytes)
    except json.JSONDecodeError as error:
        raise RuntimeError("agent output hash validation failed") from error
    if not isinstance(output_payload, dict):
        raise TypeError("agent output hash validation failed")
    checked_payload = cast("dict[str, object]", output_payload)
    if set(checked_payload) != {"task_id", "snapshot_id", "probe_status"}:
        raise RuntimeError("agent output hash validation failed")
    if _canonical_json(checked_payload) != output_bytes:
        raise RuntimeError("agent output hash validation failed")
    return output


def probe_agent(
    *,
    private_task: Path,
    snapshot_dir: Path,
    agent_input_root: Path,
    run_root: Path,
    private_root: Path,
) -> Path:
    private_task_path = Path(private_task).resolve()
    private_root_path = Path(private_root).resolve()
    if private_task_path != private_root_path and private_root_path not in private_task_path.parents:
        raise GoldAccessViolation("private task must be below private benchmark root")
    task = RuntimeTask.model_validate_json(private_task_path.read_bytes(), strict=True)
    staged = materialize_agent_runtime_task(
        task,
        agent_input_root=agent_input_root,
        request_id=private_task_path.stem,
        forbidden_private_root=private_root_path,
    )
    root = Path(__file__).resolve().parents[2]
    return _launch_probe(
        task_path=staged,
        snapshot_root=Path(snapshot_dir).resolve(),
        run_root=Path(run_root).resolve(),
        root=root,
    )


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.processes.evaluator")
    subparsers = parser.add_subparsers(dest="command", required=True)
    probe = subparsers.add_parser("probe-agent")
    probe.add_argument("--private-task", type=Path, required=True)
    probe.add_argument("--private-root", type=Path)
    probe.add_argument("--snapshot-dir", type=Path, required=True)
    probe.add_argument("--agent-input-root", type=Path, required=True)
    probe.add_argument("--run-root", type=Path, required=True)
    probe.add_argument("--output", type=Path)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "probe-agent":
        return 2
    private_root_value = args.private_root or os.environ.get("DEEPRESEARCH_BENCHMARK_GOLD_ROOT")
    if not private_root_value:
        print("PRIVATE_ROOT_REQUIRED", file=sys.stderr)
        return 3
    private_root = Path(private_root_value)
    try:
        output = probe_agent(
            private_task=args.private_task,
            snapshot_dir=args.snapshot_dir,
            agent_input_root=args.agent_input_root,
            run_root=args.run_root,
            private_root=private_root,
        )
        if args.output is not None:
            requested = Path(args.output).resolve()
            if requested != output.resolve():
                raise GoldAccessViolation("output path does not match probe run root")
        print(json.dumps({"probe_status": "ok", "output_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}, sort_keys=True))
        return 0
    except GoldAccessViolation as error:
        print(error.code, file=sys.stderr)
        return 3
    except (OSError, RuntimeError, ValueError, TypeError):
        print("PROBE_FAILED", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "StagedRuntimeTask",
    "main",
    "materialize_agent_runtime_task",
    "probe_agent",
    "stage_authorized_runtime_task",
    "stage_sealed_config",
]
