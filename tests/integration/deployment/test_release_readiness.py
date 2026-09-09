"""Subprocess contract for the secret-free Release A readiness command."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path

FAILURE_STDOUT = (
    "release_a version=0.1.0 status=fail profile_count=0 route_count=0 "
    f"pricing_count=0 bundle_sha256={'0' * 64}\n"
)
CATALOG_SECRET = "CATALOG-SYNTHETIC-SECRET"
PROVIDER_SENTINEL = "provider-synthetic-sentinel"


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


def _run_readiness(
    repository: Path, *extra_arguments: str
) -> subprocess.CompletedProcess[str]:
    environment = os.environ.copy()
    environment["MODEL_API_KEY"] = "MODEL-SECRET"
    return subprocess.run(
        [
            sys.executable,
            "scripts/release_readiness.py",
            "--repository",
            str(repository),
            *extra_arguments,
        ],
        capture_output=True,
        check=False,
        env=environment,
        text=True,
    )


def _assert_sanitized_failure(
    result: subprocess.CompletedProcess[str],
    repository: Path,
    code: str,
    *,
    forbidden: tuple[str, ...] = (),
) -> None:
    assert result.returncode == 1
    assert result.stdout == FAILURE_STDOUT
    assert result.stderr == f"readiness_error={code}\n"
    combined = result.stdout + result.stderr
    assert str(repository) not in combined
    assert "Traceback" not in combined
    assert "MODEL-SECRET" not in combined
    for value in forbidden:
        assert value not in combined


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

    _assert_sanitized_failure(result, repository, "bundle_verification")


def test_release_readiness_sanitizes_rejected_cli_arguments(tmp_path: Path) -> None:
    repository = _release_repository(tmp_path)
    cli_secret = "CLI-SYNTHETIC-SECRET"

    result = _run_readiness(repository, "--unexpected", cli_secret)

    _assert_sanitized_failure(
        result,
        repository,
        "cli_arguments",
        forbidden=(cli_secret,),
    )


def test_release_readiness_rejects_coordinated_route_topology_change(
    tmp_path: Path,
) -> None:
    repository = _release_repository(tmp_path)
    profiles_path = repository / "deploy" / "replay" / "profiles.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    parse_route = next(
        route
        for route in profiles["profiles"]["replay-default"]["routes"]
        if route["operation"] == "parse"
    )
    parse_route.update(
        operation="search",
        provider_id=PROVIDER_SENTINEL,
        endpoint_type="search",
        fallback_rank=1,
        parameters={"bundle_path": "tests/fixtures/replay/baseline"},
    )
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")

    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    parse_price = next(
        item
        for item in pricing["profiles"]["replay-default"]
        if item["endpoint_type"] == "parse"
    )
    parse_price.update(
        snapshot_id="synthetic-search",
        provider_id=PROVIDER_SENTINEL,
        endpoint_type="search",
        model_id="search",
    )
    pricing_path.write_text(json.dumps(pricing), encoding="utf-8")

    result = _run_readiness(repository)

    _assert_sanitized_failure(
        result,
        repository,
        "route_topology",
        forbidden=(PROVIDER_SENTINEL,),
    )


def test_release_readiness_rejects_coordinated_route_identity_change(
    tmp_path: Path,
) -> None:
    repository = _release_repository(tmp_path)
    profiles_path = repository / "deploy" / "replay" / "profiles.json"
    profiles = json.loads(profiles_path.read_text(encoding="utf-8"))
    model_route = next(
        route
        for route in profiles["profiles"]["replay-default"]["routes"]
        if route["operation"] == "model"
    )
    model_route.update(
        provider_id=PROVIDER_SENTINEL,
        model_id="synthetic-model",
        model_revision="d" * 40,
    )
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")

    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    for item in pricing["profiles"]["replay-default"]:
        if item["provider_id"] == "baseline-model":
            item["provider_id"] = PROVIDER_SENTINEL
            item["model_id"] = "synthetic-model"
    pricing_path.write_text(json.dumps(pricing), encoding="utf-8")

    result = _run_readiness(repository)

    _assert_sanitized_failure(
        result,
        repository,
        "route_identity",
        forbidden=(PROVIDER_SENTINEL,),
    )


def test_release_readiness_rejects_missing_and_extra_pricing_keys(
    tmp_path: Path,
) -> None:
    repository = _release_repository(tmp_path)
    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    snapshots = pricing["profiles"]["replay-default"]
    extra = dict(snapshots.pop())
    extra.update(
        snapshot_id=CATALOG_SECRET,
        provider_id=PROVIDER_SENTINEL,
        endpoint_type="synthetic",
        model_id="synthetic-model",
    )
    snapshots.append(extra)
    pricing_path.write_text(json.dumps(pricing), encoding="utf-8")

    result = _run_readiness(repository)

    _assert_sanitized_failure(
        result,
        repository,
        "pricing_coverage",
        forbidden=(CATALOG_SECRET, PROVIDER_SENTINEL),
    )


def test_release_readiness_rejects_extra_pricing_key(tmp_path: Path) -> None:
    repository = _release_repository(tmp_path)
    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    extra = dict(pricing["profiles"]["replay-default"][-1])
    extra.update(
        snapshot_id=CATALOG_SECRET,
        provider_id=PROVIDER_SENTINEL,
        endpoint_type="synthetic",
        model_id="synthetic-model",
    )
    pricing["profiles"]["replay-default"].append(extra)
    pricing_path.write_text(json.dumps(pricing), encoding="utf-8")

    result = _run_readiness(repository)

    _assert_sanitized_failure(
        result,
        repository,
        "pricing_shape",
        forbidden=(CATALOG_SECRET, PROVIDER_SENTINEL),
    )


def test_release_readiness_rejects_missing_pricing_key(tmp_path: Path) -> None:
    repository = _release_repository(tmp_path)
    pricing_path = repository / "deploy" / "replay" / "pricing.json"
    pricing = json.loads(pricing_path.read_text(encoding="utf-8"))
    pricing["profiles"]["replay-default"].pop()
    pricing_path.write_text(json.dumps(pricing), encoding="utf-8")

    result = _run_readiness(repository)

    _assert_sanitized_failure(result, repository, "pricing_shape")
