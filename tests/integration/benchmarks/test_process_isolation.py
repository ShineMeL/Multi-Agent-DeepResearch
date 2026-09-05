from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from benchmarks.datasets.isolation import GoldIsolationGuard
from benchmarks.datasets.models import AnnotatedQuestion, RuntimeTask

EXAMPLE = Path(__file__).parents[2] / ".." / "benchmarks" / "datasets" / "templates" / "question.example.json"


def _base_env(root: Path) -> dict[str, str]:
    env = {
        key: os.environ[key]
        for key in ("PATH", "SystemRoot", "TEMP", "TMP", "PYTHONIOENCODING")
        if key in os.environ
    }
    env["PYTHONPATH"] = os.pathsep.join(
        [str(root), str(root / "src"), os.environ.get("PYTHONPATH", "")]
    )
    return env


def _snapshot(root: Path) -> Path:
    snapshot = root / "snapshots"
    snapshot.mkdir()
    payload = b"{}\n"
    (snapshot / "snapshot.json").write_bytes(payload)
    (snapshot / "manifest.sha256").write_text(
        json.dumps(
            {"file_sha256": {"snapshot.json": hashlib.sha256(payload).hexdigest()}},
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n",
        encoding="utf-8",
    )
    return snapshot


def test_agent_subprocess_refuses_gold_environment(tmp_path: Path) -> None:
    root = Path(__file__).parents[2].parents[1].resolve()
    runtime = tmp_path / "runtime"
    runtime.mkdir()
    task = runtime / "task.json"
    task.write_bytes(EXAMPLE.resolve().read_bytes())
    output = tmp_path / "run" / "probe.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.processes.agent",
            "--probe-runtime-task",
            str(task),
            "--runtime-root",
            str(runtime),
            "--snapshot-dir",
            str(_snapshot(tmp_path)),
            "--run-root",
            str(tmp_path / "run"),
            "--output",
            str(output),
        ],
        cwd=root,
        env={**_base_env(root), "DEEPRESEARCH_BENCHMARK_GOLD_ROOT": str(tmp_path / "private")},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 3
    assert "GOLD_ROOT_FORBIDDEN" in result.stderr


def test_evaluator_stages_sanitized_task_and_agent_never_sees_private_root(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[2].parents[1].resolve()
    private = tmp_path / "private"
    private.mkdir()
    sentinel = private / "sentinel.txt"
    sentinel.write_text("must remain evaluator-only", encoding="utf-8")
    task = private / "task.json"
    question = AnnotatedQuestion.model_validate_json(EXAMPLE.resolve().read_bytes())
    runtime = GoldIsolationGuard.runtime_view(question)
    task.write_bytes(
        (
            json.dumps(runtime.model_dump(mode="json"), sort_keys=True, separators=(",", ":"))
            + "\n"
        ).encode("utf-8")
    )
    snapshot = _snapshot(tmp_path)
    agent_inputs = tmp_path / "agent-inputs"
    run_root = tmp_path / "run"
    output = run_root / "probe.json"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.processes.evaluator",
            "probe-agent",
            "--private-task",
            str(task),
            "--snapshot-dir",
            str(snapshot),
            "--agent-input-root",
            str(agent_inputs),
            "--run-root",
            str(run_root),
            "--output",
            str(output),
        ],
        cwd=root,
        env={**_base_env(root), "DEEPRESEARCH_BENCHMARK_GOLD_ROOT": str(private)},
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert sentinel.read_text(encoding="utf-8") == "must remain evaluator-only"
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["probe_status"] == "ok"
    assert set(payload) == {"task_id", "snapshot_id", "probe_status"}
    assert str(private) not in result.stdout
    assert str(private) not in result.stderr
    assert RuntimeTask.model_validate_json(task.read_bytes()).task_id == payload["task_id"]
