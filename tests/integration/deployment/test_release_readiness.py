"""Subprocess contract for the secret-free Release A readiness command."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from pathlib import Path


def _release_repository(tmp_path: Path) -> Path:
    repository = tmp_path / "repository"
    catalog_root = repository / "deploy" / "replay"
    catalog_root.mkdir(parents=True)
    for name in ("profiles.json", "pricing.json"):
        shutil.copyfile(Path("deploy/replay") / name, catalog_root / name)

    source_bundle = Path("tests/fixtures/replay/baseline")
    bundle_root = repository / source_bundle
    bundle_root.mkdir(parents=True)
    for source in source_bundle.iterdir():
        if source.is_file():
            bundle_root.joinpath(source.name).write_bytes(
                source.read_bytes().replace(b"\r\n", b"\n")
            )
    return repository


def _run_readiness(repository: Path) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MODEL_API_KEY"] = "MODEL-SECRET"
    return subprocess.run(
        [
            sys.executable,
            "scripts/release_readiness.py",
            "--repository",
            str(repository),
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def test_release_readiness_reports_stable_secret_free_success(tmp_path: Path) -> None:
    repository = _release_repository(tmp_path)

    result = _run_readiness(repository)

    assert result.returncode == 0, result.stderr
    assert re.fullmatch(
        r"release_a version=0\.1\.0 status=pass profile_count=1 route_count=5 "
        r"pricing_count=6 bundle_sha256=[0-9a-f]{64}\n",
        result.stdout,
    )
    assert result.stderr == ""
    assert "MODEL-SECRET" not in result.stdout
    assert "MODEL-SECRET" not in result.stderr


def test_release_readiness_sanitizes_bundle_verification_failure(tmp_path: Path) -> None:
    repository = _release_repository(tmp_path)
    manifest = repository / "tests" / "fixtures" / "replay" / "baseline" / "manifest.sha256"
    manifest.write_bytes(manifest.read_bytes().replace(b"a", b"b", 1))

    result = _run_readiness(repository)

    assert result.returncode != 0
    assert result.stdout == (
        "release_a version=0.1.0 status=fail profile_count=0 route_count=0 "
        f"pricing_count=0 bundle_sha256={'0' * 64}\n"
    )
    assert result.stderr == "readiness_error=bundle_verification\n"
    assert str(repository) not in result.stdout
    assert str(repository) not in result.stderr
    assert "MODEL-SECRET" not in result.stdout
    assert "MODEL-SECRET" not in result.stderr
