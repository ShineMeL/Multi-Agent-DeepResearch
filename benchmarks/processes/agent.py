from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import cast


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _verify_snapshot(snapshot_dir: Path) -> None:
    manifest_path = snapshot_dir / "manifest.sha256"
    if not manifest_path.is_file():
        raise ValueError("snapshot manifest is missing")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise TypeError("snapshot manifest is invalid")
    raw_manifest = cast("dict[object, object]", manifest)
    file_hashes = raw_manifest.get("file_sha256")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("snapshot manifest has no file hashes")
    for filename, expected in cast("dict[object, object]", file_hashes).items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("snapshot manifest contains an unsafe filename")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or expected != expected.lower()
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("snapshot manifest contains an invalid hash")
        path = snapshot_dir / filename
        if not path.is_file():
            raise ValueError(f"snapshot file is missing: {filename}")
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"snapshot hash mismatch: {filename}")


def _write_probe(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.staging")
    data = _canonical_json(payload)
    try:
        with staging.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.processes.agent")
    parser.add_argument("--probe-runtime-task", type=Path, required=True)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    # This is intentionally the first non-stdlib operation: an agent process
    # must reject a gold environment before importing benchmark/domain code.
    from benchmarks.datasets.isolation import (
        AgentRuntimeGuard,
        GoldAccessViolation,
        assert_agent_environment,
    )
    from benchmarks.datasets.models import RuntimeTask

    try:
        assert_agent_environment(os.environ)
    except GoldAccessViolation as error:
        print(error.code, file=sys.stderr)
        return 3

    try:
        args = _parser().parse_args(argv)
        guard = AgentRuntimeGuard(
            runtime_root=args.runtime_root,
            snapshot_root=args.snapshot_dir,
            run_root=args.run_root,
        )
        task_path = guard.resolve_runtime_task(args.probe_runtime_task)
        snapshot_dir = guard.resolve_snapshot(args.snapshot_dir)
        output_path = guard.resolve_output(args.output)
        task = RuntimeTask.model_validate_json(task_path.read_bytes(), strict=True)
        guard.validate_payload(task.model_dump(mode="json"))
        _verify_snapshot(snapshot_dir)
        _write_probe(
            output_path,
            {
                "probe_status": "ok",
                "snapshot_id": task.snapshot_id,
                "task_id": task.task_id,
            },
        )
        return 0
    except GoldAccessViolation as error:
        print(error.code, file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print("SNAPSHOT_OR_RUNTIME_INVALID", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["main"]
