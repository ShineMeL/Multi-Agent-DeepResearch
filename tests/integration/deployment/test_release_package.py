"""Contract for the packaged Replay catalog and public baseline fixture."""

import asyncio
import json
import os
import re
import secrets
import shutil
import socket
import subprocess
from pathlib import Path

import httpx
import pytest

from apps.api.main import create_app
from apps.api.settings import ServiceSettings
from apps.ui.replay import replay_payload
from deepresearch.providers.replay import ReplayBundle
from deepresearch.runtime.manifest import RunManifest
from deepresearch.runtime.runner_factory import (
    FilePricingCatalog,
    FileProviderRouteCatalog,
    default_provider_constructors,
)


def _lf_copy(source: Path, destination: Path) -> Path:
    destination.mkdir()
    for item in source.iterdir():
        if item.is_file():
            destination.joinpath(item.name).write_bytes(item.read_bytes().replace(b"\r\n", b"\n"))
    return destination


def _catalog_with_bundle(source: Path, bundle: Path, tmp_path: Path) -> Path:
    payload = json.loads(source.read_text(encoding="utf-8"))
    for route in payload["profiles"]["replay-default"]["routes"]:
        if route["operation"] != "parse":
            route["parameters"]["bundle_path"] = str(bundle)
    destination = tmp_path / "profiles.json"
    destination.write_text(json.dumps(payload), encoding="utf-8")
    return destination


def test_packaged_replay_catalog_is_complete(tmp_path: Path) -> None:
    bundle = _lf_copy(Path("tests/fixtures/replay/baseline"), tmp_path / "bundle")
    profiles = _catalog_with_bundle(Path("deploy/replay/profiles.json"), bundle, tmp_path)
    pricing = FilePricingCatalog.load(Path("deploy/replay/pricing.json"))
    routes = FileProviderRouteCatalog.load(profiles).resolve("replay-default")
    snapshots = pricing.resolve("replay-default")

    assert routes.execution_mode == "replay"
    assert {route.operation for route in routes.routes} == {
        "model",
        "search",
        "fetch",
        "parse",
        "embed",
    }
    assert ReplayBundle.load(bundle).verify().valid
    required = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured") if route.operation == "model" else (route.operation,)
        )
    }
    available = {(item.provider_id, item.endpoint_type, item.model_id) for item in snapshots}
    assert required <= available
    assert len(snapshots) == 6

    checked_in = json.loads(Path("deploy/replay/profiles.json").read_text(encoding="utf-8"))
    non_parse_routes = (
        route
        for route in checked_in["profiles"]["replay-default"]["routes"]
        if route["operation"] != "parse"
    )
    assert all(
        route["parameters"]["bundle_path"] == "tests/fixtures/replay/baseline"
        for route in non_parse_routes
    )


def test_operator_documentation_describes_release_boundaries() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")
    results = Path("docs/results.md").read_text(encoding="utf-8")

    assert "uv run python scripts/release_readiness.py" in readme
    assert "/app/deploy/replay/profiles.json" in deployment
    assert "research-v1" in readme and "RESEARCH_GRAPH_UNAVAILABLE" in deployment
    assert "not sealed" in results
    assert "MODEL_API_KEY" in deployment and "KIMI" in deployment
    assert "applies the `local` replay policy" in deployment
    ui = Path("apps/ui/app.py").read_text(encoding="utf-8")
    assert "Compare planner strategies" in ui
    assert "Custom API deployments" in ui
    assert "catalog entry is empty" not in ui


async def test_documented_sqlite_launch_completes_packaged_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Consume the actual runbook settings and packaged prices, rather than a
    # separately generated test catalog that could hide runtime incompatibility.
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")
    section = deployment.split("## Local SQLite", 1)[1].split("## Compose Postgres", 1)[0]
    block = section.split("```powershell", 1)[1].split("```", 1)[0]
    bundle = tmp_path / "tests/fixtures/replay/baseline"
    bundle.parent.mkdir(parents=True)
    _lf_copy(Path("tests/fixtures/replay/baseline"), bundle)
    catalog = tmp_path / "deploy/replay"
    catalog.mkdir(parents=True)
    for name in ("profiles.json", "pricing.json"):
        shutil.copyfile(Path("deploy/replay") / name, catalog / name)
    expected_parent = ReplayBundle.load(bundle).snapshot.run_id
    setting_names = {name.upper() for name in ServiceSettings.model_fields}
    for name in os.environ:
        if name.upper() in setting_names or name.upper() in {"MODEL_API_KEY", "SEARCH_API_KEY"}:
            monkeypatch.delenv(name)
    for name, value in re.findall(r"^\$env:([A-Z_]+) = '([^']*)'$", block, re.MULTILINE):
        monkeypatch.setenv(name, value)
    # The runbook RNG output stays in memory, just as in its shell assignment.
    monkeypatch.setenv("SESSION_SIGNING_KEY", secrets.token_urlsafe(32))
    monkeypatch.chdir(tmp_path)

    network_attempts: list[str] = []

    def forbid_network(*args: object, **kwargs: object) -> None:
        network_attempts.append("attempt")
        raise AssertionError("Packaged replay attempted network access")

    monkeypatch.setattr(socket.socket, "connect", forbid_network)
    monkeypatch.setattr(socket, "getaddrinfo", forbid_network)
    app = create_app()
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="http://localhost"
        ) as client,
    ):
        accepted = await client.post(
            "/runs",
            json=replay_payload("Compare planner strategies"),
            headers={"Idempotency-Key": "packaged-sqlite"},
        )
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run_id"]
        await asyncio.wait_for(app.state.manager.wait(run_id), 30)
        final = await client.get(f"/runs/{run_id}")
        assert final.json()["status"] == "completed", final.text
        manifest_response = await client.get(f"/runs/{run_id}/artifacts/manifest")
        assert manifest_response.status_code == 200
        manifest = RunManifest.model_validate_json(manifest_response.content)
        assert manifest.replay_parent == expected_parent
        assert manifest.provider_profiles[0].execution_mode == "replay"
        for artifact in ("report", "evidence"):
            assert (await client.get(f"/runs/{run_id}/artifacts/{artifact}")).status_code == 200
        events = await client.get(f"/runs/{run_id}/events")
        assert "run_completed" in events.text
    assert network_attempts == []


def test_live_search_documentation_matches_registered_provider() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    deployment = Path("docs/deployment.md").read_text(encoding="utf-8")
    constructors = default_provider_constructors()

    assert "Tavily search route" in readme
    assert "Tavily search route" in deployment
    assert "Serper" not in readme and "Serper" not in deployment
    assert "tavily" in constructors and "serper" not in constructors


def test_readme_distinguishes_shipped_service_from_follow_on_work() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "Creating new recordings and resumable Core checkpoints remain follow-on work" in readme
    assert "richer service composition" not in readme


def test_checked_in_files_do_not_contain_synthetic_secret() -> None:
    sentinel = "synthetic-" + "secret-sentinel"
    tracked = subprocess.run(
        ["git", "ls-files", "-z"],
        check=True,
        capture_output=True,
    ).stdout.split(b"\0")

    offenders = [
        path.decode("utf-8")
        for path in tracked
        if path and sentinel.encode() in Path(path.decode("utf-8")).read_bytes()
    ]
    assert offenders == []
