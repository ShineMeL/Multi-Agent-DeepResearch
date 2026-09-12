"""Docker-engine verification for exclusions from the build context."""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest


def _docker_server_is_available() -> bool:
    if shutil.which("docker") is None:
        return False
    try:
        result = subprocess.run(
            ["docker", "info", "--format", "{{.ServerVersion}}"],
            capture_output=True,
            check=False,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return False
    return result.returncode == 0


def test_docker_build_context_excludes_nested_secrets_caches_and_runtime_state(
    tmp_path: Path,
):
    if not _docker_server_is_available():
        pytest.skip("Docker CLI and engine are required to verify build-context exclusions")

    context = tmp_path / "context"
    output = tmp_path / "output"
    context.mkdir()
    (context / ".dockerignore").write_bytes(Path(".dockerignore").read_bytes())
    (context / "Dockerfile").write_text(
        "# syntax=docker/dockerfile:1\nFROM scratch\nCOPY . /\n",
        encoding="utf-8",
    )

    fixtures = {
        "safe/source.py": "safe",
        "nested/.env": "secret",
        "nested/.env.local": "secret",
        "secrets/signing.key": "secret",
        "cache/__pycache__/module.cpython-312.pyc": "cache",
        "runtime/checkpoints.sqlite3": "state",
        "runtime/checkpoints.sqlite3-wal": "state",
        "runtime/service.log": "log",
    }
    for relative_path, contents in fixtures.items():
        fixture = context / relative_path
        fixture.parent.mkdir(parents=True, exist_ok=True)
        fixture.write_text(contents, encoding="utf-8")

    result = subprocess.run(
        [
            "docker",
            "build",
            "--no-cache",
            "--output",
            f"type=local,dest={output}",
            str(context),
        ],
        capture_output=True,
        check=False,
        env={**os.environ, "DOCKER_BUILDKIT": "1"},
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (output / "safe" / "source.py").read_text(encoding="utf-8") == "safe"
    for excluded in fixtures.keys() - {"safe/source.py"}:
        assert not (output / excluded).exists(), excluded
