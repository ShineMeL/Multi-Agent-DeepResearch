import hashlib
import json
from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from apps.api.main import create_app
from apps.api.schemas import CreateRunRequest
from apps.api.settings import ServiceSettings
from apps.ui.replay import research_replay_payload
from deepresearch.providers.replay_schema import ReplayBundle


def settings(tmp_path, **changes):
    values = {
        "database_url": f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
        "artifact_root": tmp_path,
        "checkpoint_sqlite_path": tmp_path / "checkpoints.sqlite",
        "session_signing_key": "demo-test-signing-value-at-least-32-bytes",
        "langgraph_strict_msgpack": True,
        "deployment_access_profile": "local",
        "provider_profile_catalog_path": Path("deploy/replay/profiles.json"),
        "pricing_catalog_path": Path("deploy/replay/pricing.json"),
    }
    values.update(changes)
    return ServiceSettings(**values)


def test_capabilities_provide_only_server_selected_modes_and_fixed_replay_example(tmp_path):
    with TestClient(create_app(settings(tmp_path)), client=("127.0.0.1", 1234)) as client:
        response = client.get("/capabilities")
        assert response.status_code == 200
        data = response.json()
        assert data["profiles"] == [
            {
                "profile_id": "replay-default",
                "execution_mode": "replay",
                "available": True,
                "reason": None,
                "workflow_id": "research-v1",
                "planner_id": "P1",
                "ranker_id": "R1",
            }
        ]
        assert data["replay_example"]["question"] == "Compare planner strategies"
        assert data["replay_example"]["provider_profile_id"] == "replay-default"
        assert data["replay_example"]["budget_preset"] == "medium"
        assert data["replay_example"]["report_language"] == "en"
        assert "bundle_path" not in response.text
        assert "signing-value" not in response.text


def _custom_replay_bundle(tmp_path: Path) -> Path:
    source = Path("tests/fixtures/replay/baseline")
    bundle = tmp_path / "custom-recording"
    bundle.mkdir()
    for item in source.iterdir():
        if item.is_file():
            bundle.joinpath(item.name).write_bytes(item.read_bytes())
    snapshot = json.loads(bundle.joinpath("snapshot.json").read_text(encoding="utf-8"))
    snapshot["providers"]["embed"]["snapshot_sha256"] = "a" * 64
    snapshot_bytes = json.dumps(snapshot, sort_keys=True, separators=(",", ":")).encode()
    bundle.joinpath("snapshot.json").write_bytes(snapshot_bytes)
    manifest = json.loads(bundle.joinpath("manifest.sha256").read_text(encoding="utf-8"))
    manifest["file_sha256"]["snapshot.json"] = hashlib.sha256(snapshot_bytes).hexdigest()
    bundle.joinpath("manifest.sha256").write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )
    assert ReplayBundle.load(bundle).verify().valid
    return bundle


def _catalog_with_custom_replay_first(tmp_path: Path) -> Path:
    payload = json.loads(Path("deploy/replay/profiles.json").read_text(encoding="utf-8"))
    builtin = payload["profiles"]["replay-default"]
    custom = deepcopy(builtin)
    custom_bundle = _custom_replay_bundle(tmp_path)
    for route in custom["routes"]:
        if "bundle_path" in route["parameters"]:
            route["parameters"]["bundle_path"] = str(custom_bundle)
    payload["profiles"] = {"custom-replay": custom, "replay-default": builtin}
    path = tmp_path / "profiles.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def test_capabilities_bind_the_builtin_example_when_a_custom_replay_profile_is_first(tmp_path):
    current = settings(
        tmp_path,
        provider_profile_catalog_path=_catalog_with_custom_replay_first(tmp_path),
        allowed_provider_profile_ids=("custom-replay", "replay-default"),
    )

    with TestClient(create_app(current), client=("127.0.0.1", 1234)) as client:
        data = client.get("/capabilities").json()

    assert [profile["profile_id"] for profile in data["profiles"]] == [
        "custom-replay",
        "replay-default",
    ]
    assert data["replay_example"]["provider_profile_id"] == "replay-default"


def test_capabilities_do_not_infer_the_builtin_example_from_a_custom_model_id(tmp_path):
    current = settings(
        tmp_path,
        provider_profile_catalog_path=_catalog_with_custom_replay_first(tmp_path),
        allowed_provider_profile_ids=("custom-replay",),
    )

    with TestClient(create_app(current), client=("127.0.0.1", 1234)) as client:
        data = client.get("/capabilities").json()

    assert data["replay_example"] is None


def test_capabilities_reject_a_shipped_bundle_with_mismatched_frozen_routes(tmp_path):
    payload = json.loads(Path("deploy/replay/profiles.json").read_text(encoding="utf-8"))
    model_route = next(
        route
        for route in payload["profiles"]["replay-default"]["routes"]
        if route["operation"] == "model"
    )
    model_route["provider_id"] = "route-does-not-match-bundle"
    catalog = tmp_path / "mismatched-profiles.json"
    catalog.write_text(json.dumps(payload), encoding="utf-8")
    current = settings(tmp_path, provider_profile_catalog_path=catalog)

    with TestClient(create_app(current), client=("127.0.0.1", 1234)) as client:
        response = client.get("/capabilities")

    assert response.status_code == 200
    assert response.json()["profiles"][0]["available"] is False
    assert response.json()["profiles"][0]["reason"] == "PROVIDER_PROFILE_DRIFT"
    assert response.json()["replay_example"] is None


def test_local_unpriced_live_policy_keeps_replay_budget_identity(tmp_path):
    current = settings(
        tmp_path, local_unpriced_live=True, allowed_execution_modes=("replay", "live")
    )
    policy = current.deployment_policy()
    offline = CreateRunRequest.model_validate(research_replay_payload("Compare planner strategies"))
    assert offline.to_run_config(policy).budget.max_cost_usd is not None
    live_request = offline.request.model_copy(update={"execution_mode": "live"})
    online = offline.model_copy(update={"request": live_request})
    conf = online.to_run_config(policy)
    assert conf.budget.max_cost_usd is None
    assert conf.budget.max_total_tokens > 0
    assert conf.budget.max_search_calls > 0
    policy.validate_config(conf)


@pytest.mark.parametrize("access,purposes", [("public_live", ("demo",)), ("local", ("benchmark",))])
def test_unpriced_demo_policy_cannot_be_enabled_for_public_or_benchmark(tmp_path, access, purposes):
    values = settings(tmp_path).model_dump()
    values.update(
        local_unpriced_live=True,
        deployment_access_profile=access,
        allowed_run_purposes=purposes,
        cookie_secure=True,
    )
    with pytest.raises(ValueError, match="unpriced"):
        ServiceSettings.model_validate(values)


def test_missing_live_credential_is_unavailable_without_exposing_secrets(tmp_path, monkeypatch):
    from apps.api.demo import prepare_demo

    prepared = prepare_demo(Path.cwd(), tmp_path, environ={"MODEL_API_KEY": "fixture-model-secret"})
    monkeypatch.setenv("MODEL_API_KEY", "fixture-model-secret")
    monkeypatch.delenv("SEARCH_API_KEY", raising=False)
    with TestClient(create_app(prepared.settings), client=("127.0.0.1", 1234)) as client:
        response = client.get("/capabilities")
        live = next(p for p in response.json()["profiles"] if p["execution_mode"] == "live")
        assert live["available"] is False
        assert live["reason"] == "PROVIDER_NOT_CONFIGURED"
        assert "fixture-model-secret" not in response.text
        assert str(tmp_path) not in response.text


def test_capabilities_use_the_same_startup_catalog_as_the_run_manager(tmp_path):
    from apps.api.demo import prepare_demo

    prepared = prepare_demo(Path.cwd(), tmp_path, environ={})
    with TestClient(create_app(prepared.settings), client=("127.0.0.1", 1234)) as client:
        before = client.get("/capabilities").json()
        # A file edit takes effect only after restart, just like runner routes.
        prepared.settings.provider_profile_catalog_path.write_text("{}", encoding="utf-8")
        assert client.get("/capabilities").json() == before


def test_demo_discovery_omits_budgets_that_the_thin_client_cannot_submit(tmp_path):
    from apps.ui.api_client import DemoCapabilities

    current = settings(tmp_path, allowed_budget_presets=("low", "medium", "high"))
    with TestClient(create_app(current), client=("127.0.0.1", 1234)) as client:
        discovered = DemoCapabilities.model_validate_json(client.get("/capabilities").content)
        assert discovered.budget_presets == ("low", "medium")


@pytest.mark.parametrize(
    ("budgets", "purposes"),
    [(("high",), ("demo",)), (("low",), ("demo",)), (("medium",), ("test",))],
)
def test_replay_capability_is_unavailable_when_the_fixed_request_is_forbidden(
    tmp_path, budgets, purposes
):
    current = settings(tmp_path, allowed_budget_presets=budgets, allowed_run_purposes=purposes)
    with TestClient(create_app(current), client=("127.0.0.1", 1234)) as client:
        data = client.get("/capabilities").json()
        assert data["profiles"][0]["available"] is False
        assert data["profiles"][0]["reason"] == "DEPLOYMENT_POLICY_VIOLATION"
        assert data["replay_example"] is None
        rejected = client.post("/runs", json=research_replay_payload("Compare planner strategies"))
        assert rejected.status_code == 422
        assert rejected.json()["code"] == "DEPLOYMENT_POLICY_VIOLATION"


@pytest.mark.parametrize(
    ("budgets", "purposes", "available", "reason"),
    [
        (("high",), ("demo",), False, "DEPLOYMENT_POLICY_VIOLATION"),
        (("medium",), ("test",), False, "DEPLOYMENT_POLICY_VIOLATION"),
        (("low",), ("demo",), True, None),
    ],
)
def test_live_capability_requires_both_provider_configuration_and_demo_policy(
    tmp_path, monkeypatch, budgets, purposes, available, reason
):
    from apps.api.demo import prepare_demo

    monkeypatch.setenv("MODEL_API_KEY", "fixture-model-key")
    monkeypatch.setenv("SEARCH_API_KEY", "fixture-search-key")
    prepared = prepare_demo(Path.cwd(), tmp_path, environ={})
    values = prepared.settings.model_dump()
    values.update(allowed_budget_presets=budgets, allowed_run_purposes=purposes)
    with TestClient(
        create_app(ServiceSettings.model_validate(values)), client=("127.0.0.1", 1234)
    ) as client:
        response = client.get("/capabilities")
        assert response.status_code == 200
        data = response.json()
        live = next(profile for profile in data["profiles"] if profile["execution_mode"] == "live")
        assert live["available"] is available
        assert live["reason"] == reason
        # Configuration discovery must never make an upstream request or expose
        # the dummy credentials. This test deliberately does not submit live.
        assert "fixture-model-key" not in str(data)
        assert "fixture-search-key" not in str(data)
