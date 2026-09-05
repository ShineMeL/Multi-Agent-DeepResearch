from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast

from benchmarks.datasets.isolation import GoldAccessViolation
from benchmarks.datasets.models import RuntimeTask


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
    if target.exists():
        raise FileExistsError("agent task destination already exists")

    payload = _canonical_json(task.model_dump(mode="json"))
    staging = target.with_name(f".{target.name}.{os.getpid()}.staging")
    try:
        with staging.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, target)
        with target.open("rb") as handle:
            if handle.read() != payload:
                raise GoldAccessViolation("materialized runtime task failed byte validation")
        restored = RuntimeTask.model_validate_json(payload, strict=True)
        if _canonical_json(restored.model_dump(mode="json")) != payload:
            raise GoldAccessViolation("materialized runtime task failed schema validation")
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass
    return target


def _safe_environment(*, root: Path, runtime_root: Path, snapshot_root: Path, run_root: Path) -> dict[str, str]:
    environment = {
        key: os.environ[key]
        for key in ("PATH", "SystemRoot", "TEMP", "TMP", "PYTHONIOENCODING")
        if key in os.environ
    }
    environment["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), os.environ.get("PYTHONPATH", "")]
    )
    environment["DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT"] = str(runtime_root)
    environment["DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT"] = str(snapshot_root)
    environment["DEEPRESEARCH_BENCHMARK_RUN_ROOT"] = str(run_root)
    environment.pop("DEEPRESEARCH_BENCHMARK_GOLD_ROOT", None)
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


__all__ = ["main", "materialize_agent_runtime_task", "probe_agent"]
