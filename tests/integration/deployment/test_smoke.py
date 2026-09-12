"""Secret-free deployment gates and explicitly configured live-provider smoke."""

import asyncio
import json
import os
from pathlib import Path
from typing import Any

import httpx
import pytest
import yaml
from pydantic import SecretStr

from apps.api.main import create_app
from apps.api.schemas import CreateRunRequest
from apps.api.settings import ServiceSettings
from deepresearch.runtime.manifest import RunManifest
from deepresearch.runtime.runner_factory import FilePricingCatalog, FileProviderRouteCatalog
from tests.integration.replay.test_baseline_graph import request
from tests.unit.runtime.test_runner_factory_execution import composition


def test_ci_keeps_provider_secrets_out_of_verification_and_gates_online():
    path = Path(".github/workflows/ci.yml")
    assert path.is_file(), "the service verification workflow is missing"
    workflow: dict[str, Any] = yaml.safe_load(path.read_text(encoding="utf-8"))
    verify = workflow["jobs"]["verify"]
    assert "secrets" not in json.dumps(verify)
    commands = "\n".join(step.get("run", "") for step in verify["steps"])
    assert ' -m "not online"' in commands
    for suite in (
        "tests/unit",
        "tests/contracts",
        "tests/integration/replay",
        "tests/integration/api",
        "tests/integration/deployment",
    ):
        assert suite in commands
    assert "uv sync --all-extras --locked" in commands
    assert "ruff check ." in commands and "pyright src apps benchmarks experiments\n" in commands
    assert "docker compose config" in commands
    steps = verify["steps"]

    def command_index(fragment: str) -> int:
        return next(index for index, step in enumerate(steps) if fragment in step.get("run", ""))

    build_index = command_index("docker compose build --build-arg")
    up_index = command_index("docker compose up -d --wait --wait-timeout 180")
    replay_index = command_index("python -m scripts.smoke_replay")
    cleanup_index = command_index("docker compose down --volumes --remove-orphans")
    assert build_index < up_index < replay_index < cleanup_index
    assert "timeout 240s" in steps[up_index]["run"]
    assert steps[cleanup_index]["if"] == "always()"
    assert verify["env"]["COMPOSE_PROJECT_NAME"] == (
        "deepresearch-ci-${{ github.run_id }}-${{ github.run_attempt }}"
    )

    test_step = steps[command_index(' -m "not online"')]
    assert "DATABASE_URL" not in test_step["env"]
    assert test_step["env"]["DEEPRESEARCH_TEST_POSTGRES_URL"].startswith(
        "postgresql+asyncpg://deepresearch:ci-only@127.0.0.1:5432/"
    )
    smoke = workflow["jobs"]["online-smoke"]
    assert smoke["if"] == (
        "github.event_name == 'workflow_dispatch' || github.event_name == 'schedule'"
    )
    online_step = next(step for step in smoke["steps"] if " -m online" in step.get("run", ""))
    for name in (
        "MODEL_API_KEY",
        "SEARCH_API_KEY",
        "PRICING_CATALOG_JSON",
        "PROVIDER_PROFILE_CATALOG_JSON",
        "SESSION_SIGNING_KEY",
    ):
        assert f"env.{name} != ''" in online_step["if"]


@pytest.mark.parametrize("missing_credentials", [False, True])
async def test_online_smoke_skips_unconfigured_inputs_before_service_start(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, missing_credentials: bool
):
    # Complete route shape but no prices. Neither skip case may construct an app,
    # open a database or call an adapter, even if credentials happen to be set.
    _, _, routes, _, _, _ = composition(tmp_path / "fixture")
    providers = tmp_path / "providers.json"
    providers.write_text(
        json.dumps(
            {
                "profiles": {
                    "online-smoke": {
                        "execution_mode": "live",
                        "routes": [item.model_dump(mode="json") for item in routes.routes],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    prices = tmp_path / "prices.json"
    prices.write_text('{"profiles": {"online-smoke": []}}', encoding="utf-8")
    for name in ("MODEL_API_KEY", "SEARCH_API_KEY", "SESSION_SIGNING_KEY"):
        monkeypatch.setenv(name, "ci-only-never-used-secret-at-least-32-bytes")
    monkeypatch.setenv("PROVIDER_PROFILE_CATALOG_PATH", str(providers))
    monkeypatch.setenv("PRICING_CATALOG_PATH", str(prices))
    if missing_credentials:
        monkeypatch.delenv("MODEL_API_KEY")

    def forbidden_start(*args: Any, **kwargs: Any) -> Any:
        pytest.fail("unconfigured smoke attempted to start the service")

    monkeypatch.setattr("tests.integration.deployment.test_smoke.create_app", forbidden_start)
    reason = "credentials" if missing_credentials else "complete PricingSnapshot"
    with pytest.raises(pytest.skip.Exception, match=reason):
        await test_authorized_live_baseline_health_artifacts_and_redaction(tmp_path)


@pytest.mark.online
async def test_authorized_live_baseline_health_artifacts_and_redaction(tmp_path: Path):
    required = (
        "MODEL_API_KEY",
        "SEARCH_API_KEY",
        "SESSION_SIGNING_KEY",
        "PRICING_CATALOG_PATH",
        "PROVIDER_PROFILE_CATALOG_PATH",
    )
    if any(not os.environ.get(name) for name in required):
        pytest.skip("online smoke requires explicitly supplied credentials and both catalogs")
    settings = ServiceSettings(
        session_signing_key=SecretStr(os.environ["SESSION_SIGNING_KEY"]),
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
        artifact_root=tmp_path / "artifacts",
        checkpoint_sqlite_path=tmp_path / "artifacts" / "checkpoints.sqlite",
        deployment_access_profile="public_live",
        cookie_secure=True,
        langgraph_strict_msgpack=True,
        allowed_execution_modes=("live",),
        allowed_provider_profile_ids=("online-smoke",),
        allowed_run_purposes=("test",),
        allowed_budget_presets=("medium",),
    )
    routes = FileProviderRouteCatalog.load(settings.provider_profile_catalog_path).resolve(
        "online-smoke"
    )
    snapshots = FilePricingCatalog.load(settings.pricing_catalog_path).resolve("online-smoke")
    required_keys = {
        (route.provider_id, endpoint, route.model_id or route.operation)
        for route in routes.routes
        for endpoint in (
            ("complete", "structured") if route.operation == "model" else (route.operation,)
        )
    }
    available = {(item.provider_id, item.endpoint_type, item.model_id) for item in snapshots}
    if not required_keys or not required_keys <= available:
        pytest.skip("online smoke requires a complete PricingSnapshot set before provider calls")
    body = CreateRunRequest(
        request=request().model_copy(
            update={
                "question": "What is the Python programming language? Give a brief sourced answer.",
                "execution_mode": "live",
                "access_profile": "public_live",
                "provider_profile_id": "online-smoke",
                "budget_preset": "medium",
            }
        ),
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
        seed=0,
    )
    app = create_app(settings)
    async with (
        app.router.lifespan_context(app),
        httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app), base_url="https://smoke.test"
        ) as client,
    ):
        assert (await client.get("/health/live")).status_code == 200
        assert (await client.get("/health/ready")).status_code == 200
        accepted = await client.post("/runs", json=body.model_dump(mode="json"))
        assert accepted.status_code == 202, accepted.text
        run_id = accepted.json()["run_id"]
        timeout = settings.deployment_policy().budget_presets["medium"].max_wall_time_seconds + 30
        await asyncio.wait_for(app.state.manager.wait(run_id), timeout=timeout)
        final = await client.get(f"/runs/{run_id}")
        assert final.json()["status"] == "completed", final.text
        public_outputs = [final.text]
        for kind in ("report", "evidence", "manifest"):
            artifact = await client.get(f"/runs/{run_id}/artifacts/{kind}")
            assert artifact.status_code == 200
            public_outputs.append(artifact.text)
            if kind == "manifest":
                manifest = RunManifest.model_validate_json(artifact.content)
                assert manifest.pricing_snapshots
                assert manifest.pricing_status == "estimated"
        events = await client.get(f"/runs/{run_id}/events")
        assert events.status_code == 200 and '"status":"completed"' in events.text
        public_outputs.append(events.text)
        for secret in settings.loaded_secret_values():
            assert all(secret not in output for output in public_outputs)
