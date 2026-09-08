from __future__ import annotations

import base64
import hashlib
import hmac
from collections.abc import Iterator
from dataclasses import dataclass, replace
from ipaddress import ip_network
from pathlib import Path
from typing import Annotated, Any

import pytest
from fastapi import Depends, FastAPI, Request
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from apps.api import create_app
from apps.api.dependencies import get_owned_run
from apps.api.error_handlers import APIError
from apps.api.identity import OwnerIdentity, TrustedClientIpResolver
from apps.api.schemas import CreateRunRequest, RunAccepted, RunViewResponse
from deepresearch.runtime.checkpoints import checkpoint_serializer
from deepresearch.runtime.manager import RunManager, owner_scope_sha256
from deepresearch.runtime.runner_factory import (
    FilePricingCatalog,
    ProviderProfileDrift,
    ResearchGraphUnavailable,
)
from deepresearch.storage import LocalArtifactStore
from deepresearch.storage.protocols import RunView
from tests.fakes.service_store import FakeRunStore, make_record
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_manager import Factory, policy

SECRET = b"server-only-signing-key-32-bytes!!"


@dataclass
class Rig:
    app: FastAPI
    client: TestClient
    manager: RunManager
    store: FakeRunStore
    factory: Any
    artifacts: LocalArtifactStore


def payload() -> dict[str, Any]:
    return {
        "request": config().request.model_dump(mode="json"),
        "workflow_id": "baseline-v1",
        "planner_id": "P1",
        "ranker_id": "R1",
    }


@pytest.fixture
def rig(tmp_path: Path) -> Iterator[Rig]:
    conf = config()
    factory = Factory(conf)
    store = FakeRunStore()
    deployment_policy = policy(conf)
    manager = RunManager(
        runner_factory=factory,
        store=store,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        pricing_catalog=FilePricingCatalog({}),
        deployment_policy=deployment_policy,
    )
    artifacts = LocalArtifactStore(tmp_path)
    app = create_app(
        manager=manager,
        deployment_policy=deployment_policy,
        artifact_store=artifacts,
        session_secret=SECRET,
    )

    # Task 5 owns production SSE. Exercise its shared owned-lookup dependency
    # here so a bare framework 404 cannot falsely prove event ownership.
    @app.get("/runs/{run_id}/events")
    async def events(owned: Annotated[RunView, Depends(get_owned_run)]) -> dict[str, str]:
        return {"run_id": owned.run_id}

    @app.get("/_identity")
    async def identity(request: Request) -> OwnerIdentity:
        return request.state.owner

    with TestClient(app, base_url="https://testserver", client=("127.0.0.1", 12345)) as client:
        try:
            yield Rig(app, client, manager, store, factory, artifacts)
        finally:
            assert client.portal is not None
            client.portal.call(manager.shutdown, 0)


def seed(rig: Rig, status: str = "completed", **changes: Any) -> str:
    identity = rig.client.get("/_identity").json()
    conf = CreateRunRequest.model_validate(payload()).to_run_config(rig.manager.deployment_policy)
    record = replace(
        make_record("owned", status=status),
        owner_scope_sha256=identity["owner_scope_sha256"],
        config_json=conf.model_dump(mode="json"),
        provider_profile_json=rig.factory.routes.model_dump(mode="json"),
        provider_profile_sha256=rig.factory.routes.configuration_sha256,
        **changes,
    )
    assert rig.client.portal is not None
    rig.client.portal.call(rig.store.create_run, record)
    return record.run_id


def test_create_returns_typed_202_with_events_url_and_safe_view(rig: Rig) -> None:
    response = rig.client.post("/runs", json=payload())
    assert response.status_code == 202
    accepted = RunAccepted.model_validate(response.json())
    assert accepted.status == "queued"
    assert accepted.events_url == f"/runs/{accepted.run_id}/events"
    view = rig.client.get(f"/runs/{accepted.run_id}")
    assert view.status_code == 200
    assert RunViewResponse.model_validate(view.json()).run_id == accepted.run_id
    assert "config_json" not in view.json() and "owner_scope_sha256" not in view.json()
    assert rig.factory.creates[0]["config"].prompt_versions["planner"] == "fixed-planner-v1"


@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("field", ["provider_key", "budget", "prompt_versions", "execution_mode"])
def test_extra_fields_and_secrets_are_rejected_without_echo(
    rig: Rig,
    nested: bool,
    field: str,
) -> None:
    body = payload()
    target = body["request"] if nested else body
    if nested and field == "execution_mode":
        field = "owner_scope_sha256"
    target[field] = "secret-that-must-not-appear"
    response = rig.client.post("/runs", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
    assert "secret-that-must-not-appear" not in response.text
    assert not rig.factory.creates


@pytest.mark.parametrize("planner,ranker", [("P0", "R1"), ("P1", "R2"), ("P2", "R2")])
def test_baseline_rejects_incompatible_strategy_before_manager(
    rig: Rig,
    planner: str,
    ranker: str,
) -> None:
    response = rig.client.post(
        "/runs", json={**payload(), "planner_id": planner, "ranker_id": ranker}
    )
    assert response.status_code == 422
    assert response.json()["code"] == "INVALID_REQUEST"
    assert not rig.factory.creates


@pytest.mark.parametrize("claimed", ["local", "showcase"])
def test_public_profile_forced_before_pricing_without_runner_work(rig: Rig, claimed: str) -> None:
    public = replace(rig.manager.deployment_policy, forced_access_profile="public_live")
    rig.manager.deployment_policy = public
    rig.app.state.deployment_policy = public
    body = payload()
    body["request"]["access_profile"] = claimed
    response = rig.client.post("/runs", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "PRICING_REQUIRED"
    assert not rig.factory.creates


@pytest.mark.parametrize(
    "field,value",
    [
        ("execution_mode", "hybrid"),
        ("provider_profile_id", "unapproved-provider"),
        ("run_purpose", "benchmark"),
        ("budget_preset", "high"),
    ],
)
def test_policy_rejects_choices_before_manager(rig: Rig, field: str, value: str) -> None:
    body = payload()
    body["request"][field] = value
    # Also invalid at RunConfig construction: allowlist rejection must win.
    rig.app.state.deployment_policy = replace(
        rig.manager.deployment_policy,
        forced_access_profile="public_live",
        budget_presets={"low": config().budget.model_copy(update={"max_cost_usd": None})},
    )
    response = rig.client.post("/runs", json=body)
    assert response.status_code == 422
    assert response.json()["code"] == "DEPLOYMENT_POLICY_VIOLATION"
    assert not rig.factory.creates


def test_config_uses_server_budget_and_strategy_versions(rig: Rig) -> None:
    server = replace(
        rig.manager.deployment_policy,
        forced_access_profile="local",
        budget_presets={"low": config().budget.model_copy(update={"max_pages": 2})},
    )
    result = CreateRunRequest.model_validate(payload()).to_run_config(server)
    assert result.request.access_profile == "local"
    assert result.budget.max_pages == 2
    assert result.prompt_versions["writer"] == "baseline-writer-v1"
    assert result.ranker_weights_version is None
    research = CreateRunRequest.model_validate({"request": payload()["request"]}).to_run_config(
        server
    )
    assert (research.workflow_id, research.planner_id, research.ranker_id) == (
        "research-v1",
        "P2",
        "R2",
    )
    assert research.prompt_versions["planner"] == "adaptive-planner-v1"
    assert research.prompt_versions["ranker"] == "r2-utility-v1"


def test_idempotency_is_scoped_and_changed_config_conflicts(rig: Rig) -> None:
    headers = {"Idempotency-Key": "k"}
    first = rig.client.post("/runs", json=payload(), headers=headers)
    second = rig.client.post("/runs", json=payload(), headers=headers)
    assert first.status_code == second.status_code == 202
    assert first.json()["run_id"] == second.json()["run_id"]
    changed = rig.client.post("/runs", json={**payload(), "seed": 99}, headers=headers)
    assert changed.status_code == 409
    assert changed.json()["code"] == "IDEMPOTENCY_CONFLICT"
    # Each TestClient owns an event loop. Finish the first runner on its own
    # loop before creating a run in the second client's loop.
    assert rig.client.portal is not None
    rig.client.portal.call(rig.factory.runner.finish.set)
    rig.client.portal.call(rig.manager.wait, first.json()["run_id"])
    with TestClient(rig.app, base_url="https://testserver", client=("127.0.0.1", 12346)) as other:
        created = other.post("/runs", json=payload(), headers=headers)
        assert created.status_code == 202
        assert created.json()["run_id"] != first.json()["run_id"]
        assert other.portal is not None
        other.portal.call(rig.manager.shutdown, 0)


def test_signed_owner_isolation_for_every_resource_and_forged_claim(rig: Rig) -> None:
    run_id = seed(rig)
    owner = rig.client.get("/_identity").json()
    probes = [("GET", ""), ("GET", "/events"), ("POST", "/resume"), ("POST", "/cancel")]
    probes += [("GET", f"/artifacts/{kind}") for kind in ("report", "evidence", "manifest")]
    with TestClient(rig.app, base_url="https://testserver", client=("127.0.0.1", 12346)) as other:
        for method, suffix in probes:
            foreign = other.request(
                method,
                f"/runs/{run_id}{suffix}",
                headers={
                    "X-Session-ID": owner["session_id"],
                    "X-Owner-Scope-Sha256": owner["owner_scope_sha256"],
                },
                params={
                    "owner_scope_sha256": owner["owner_scope_sha256"],
                    "session_id": owner["session_id"],
                },
            )
            missing = other.request(method, f"/runs/absent{suffix}")
            assert foreign.status_code == missing.status_code == 404
            assert (
                foreign.json()
                == missing.json()
                == {
                    "code": "RUN_NOT_FOUND",
                    "message": "Run not found.",
                    "run_id": None,
                    "retry_after": None,
                }
            )
    assert rig.client.get(f"/runs/{run_id}/events").status_code == 200


def test_cookie_signature_flags_rotation_and_ip_binding(rig: Rig) -> None:
    response = rig.client.get("/_identity")
    value = rig.client.cookies["dr_session"]
    session, signature = value.split(".")
    assert len(base64.urlsafe_b64decode(session + "=")) == 32
    assert hmac.compare_digest(
        signature, hmac.new(SECRET, session.encode(), hashlib.sha256).hexdigest()
    )
    assert "HttpOnly" in response.headers["set-cookie"]
    assert "SameSite=lax" in response.headers["set-cookie"]
    assert rig.client.get("/_identity").json() == response.json()
    run_id = seed(rig)
    with TestClient(rig.app, base_url="https://testserver", client=("203.0.113.8", 12346)) as other:
        other.cookies.set("dr_session", value)
        assert other.get(f"/runs/{run_id}").status_code == 404
    rig.client.cookies.clear()
    rig.client.cookies.set("dr_session", session + "." + "0" * 64)
    forged = rig.client.get(f"/runs/{run_id}")
    assert forged.status_code == 404
    assert rig.client.cookies.get("dr_session", domain="testserver.local") != value


def test_public_cookie_always_secure(rig: Rig) -> None:
    app = create_app(
        manager=rig.manager,
        deployment_policy=replace(
            rig.manager.deployment_policy, forced_access_profile="public_live"
        ),
        artifact_store=rig.artifacts,
        session_secret=SECRET,
    )
    with TestClient(app, base_url="https://testserver", client=("127.0.0.1", 12345)) as client:
        assert "Secure" in client.get("/runs/absent").headers["set-cookie"]


@pytest.mark.parametrize("status", ["completed", "failed", "cancelled", "queued"])
def test_illegal_resume_is_409(rig: Rig, status: str) -> None:
    run_id = seed(rig, status)
    response = rig.client.post(f"/runs/{run_id}/resume")
    assert response.status_code == 409
    assert response.json()["code"] == "INVALID_RUN_STATE"


def test_interrupted_resume_reports_unavailable_without_external_work(rig: Rig) -> None:
    run_id = seed(rig, "interrupted")
    response = rig.client.post(f"/runs/{run_id}/resume")
    assert response.status_code == 409
    assert response.json()["code"] == "CHECKPOINT_RESUME_UNAVAILABLE"
    assert not rig.factory.creates
    assert rig.client.get(f"/runs/{run_id}").json()["status"] == "interrupted"


def test_running_resume_and_inactive_cancel(rig: Rig) -> None:
    run_id = seed(rig, "interrupted")
    response = rig.client.post(f"/runs/{run_id}/cancel")
    assert response.status_code == 200
    assert response.json()["status"] == "cancelled"
    assert rig.client.post(f"/runs/{run_id}/cancel").json()["status"] == "cancelled"


def test_running_resume_is_idempotent_and_terminal_cancel_conflicts(rig: Rig) -> None:
    run_id = seed(rig, "running")
    response = rig.client.post(f"/runs/{run_id}/resume")
    assert response.status_code == 200
    assert response.json()["status"] == "running"
    assert not rig.factory.creates


def test_completed_cancel_returns_409(rig: Rig) -> None:
    run_id = seed(rig)
    response = rig.client.post(f"/runs/{run_id}/cancel")
    assert response.status_code == 409
    assert response.json()["code"] == "INVALID_RUN_STATE"


@pytest.mark.parametrize(
    "kind,field,data,media,filename",
    [
        ("report", "report_artifact_id", b"# Report", "text/markdown", "report.md"),
        (
            "evidence",
            "evidence_graph_artifact_id",
            b'{"nodes":[]}',
            "application/json",
            "evidence.json",
        ),
        ("manifest", "manifest_artifact_id", b'{"version":1}', "application/json", "manifest.json"),
    ],
)
def test_artifact_is_selected_from_owned_view(
    rig: Rig,
    kind: str,
    field: str,
    data: bytes,
    media: str,
    filename: str,
) -> None:
    ref = rig.artifacts.put_bytes(data, media_type=media)
    run_id = seed(rig, **{field: ref.artifact_id})
    response = rig.client.get(
        f"/runs/{run_id}/artifacts/{kind}", params={"artifact_id": "../../.env"}
    )
    assert response.status_code == 200
    assert response.content == data
    assert response.headers["content-type"].startswith(media)
    assert response.headers["content-disposition"] == f'attachment; filename="{filename}"'
    assert response.headers["x-content-type-options"] == "nosniff"
    assert rig.client.get(f"/runs/{run_id}/artifacts/../../.env").status_code in {404, 422}
    assert rig.client.get(f"/runs/{run_id}/artifacts/{ref.artifact_id}").status_code == 422


def test_unavailable_artifact_is_safe_404(rig: Rig) -> None:
    run_id = seed(rig, manifest_artifact_id="sha256:" + "0" * 64)
    for kind in ("report", "manifest"):
        response = rig.client.get(f"/runs/{run_id}/artifacts/{kind}")
        assert response.status_code == 404
        assert response.json()["code"] == "ARTIFACT_NOT_FOUND"


def make_request(peer: str, headers: list[tuple[str, str]]) -> Request:
    return Request(
        {
            "type": "http",
            "client": (peer, 12345),
            "headers": [(key.lower().encode(), value.encode()) for key, value in headers],
        }
    )


@pytest.mark.parametrize(
    "headers",
    [
        [("X-Forwarded-For", "198.51.100.7")],
        [("Forwarded", "for=198.51.100.7")],
        [("X-Forwarded-For", "malformed")],
    ],
)
def test_untrusted_peer_ignores_forwarded_headers(headers: list[tuple[str, str]]) -> None:
    resolver = TrustedClientIpResolver((ip_network("10.0.0.0/8"),))
    assert resolver.resolve(make_request("203.0.113.9", headers)) == "203.0.113.9"


@pytest.mark.parametrize(
    "headers,expected",
    [
        ([("X-Forwarded-For", "192.0.2.99, 198.51.100.7, 10.1.1.1")], "198.51.100.7"),
        ([("X-Forwarded-For", "198.51.100.7"), ("X-Forwarded-For", "10.1.1.1")], "198.51.100.7"),
        ([("Forwarded", 'for="[2001:db8::7]";proto=https, for=10.1.1.1')], "2001:db8::7"),
        ([], "10.0.0.2"),
    ],
)
def test_trusted_chain_is_walked_from_right(
    headers: list[tuple[str, str]],
    expected: str,
) -> None:
    resolver = TrustedClientIpResolver((ip_network("10.0.0.0/8"),))
    assert resolver.resolve(make_request("10.0.0.2", headers)) == expected


@pytest.mark.parametrize(
    "headers",
    [
        [("X-Forwarded-For", "unknown, 198.51.100.7")],
        [("X-Forwarded-For", "198.51.100.7,")],
        [("X-Forwarded-For", ",".join(["198.51.100.7"] * 17))],
        [("Forwarded", "for=_hidden")],
        [("Forwarded", "by=10.0.0.1")],
        [("Forwarded", "for=198.51.100.7;for=198.51.100.8")],
    ],
)
def test_malformed_trusted_chain_has_stable_400(headers: list[tuple[str, str]]) -> None:
    resolver = TrustedClientIpResolver((ip_network("10.0.0.0/8"),))
    with pytest.raises(APIError) as raised:
        resolver.resolve(make_request("10.0.0.2", headers))
    assert raised.value.code == "INVALID_FORWARDED_HEADER"


def test_middleware_maps_bad_proxy_chain_without_issuing_work(rig: Rig) -> None:
    app = create_app(
        manager=rig.manager,
        deployment_policy=rig.manager.deployment_policy,
        artifact_store=rig.artifacts,
        session_secret=SECRET,
        trusted_proxy_cidrs=(ip_network("10.0.0.0/8"),),
    )
    with TestClient(app, client=("10.0.0.2", 12345)) as client:
        response = client.post("/runs", json=payload(), headers={"X-Forwarded-For": "bad"})
        assert response.status_code == 400
        assert response.json()["code"] == "INVALID_FORWARDED_HEADER"
    assert not rig.factory.creates


@pytest.mark.parametrize(
    "error,status,code",
    [
        (RuntimeError("secret-provider-token /private/path"), 500, "INTERNAL_ERROR"),
        (ValueError("secret-provider-token"), 500, "INTERNAL_ERROR"),
        (ProviderProfileDrift(), 409, "PROVIDER_PROFILE_DRIFT"),
        (ResearchGraphUnavailable(), 422, "RESEARCH_GRAPH_UNAVAILABLE"),
        (APIError("RATE_LIMITED", retry_after=7), 429, "RATE_LIMITED"),
        (APIError("INVALID_LAST_EVENT_ID"), 422, "INVALID_LAST_EVENT_ID"),
    ],
)
def test_public_error_mapping_never_echoes_exception(
    rig: Rig,
    error: Exception,
    status: int,
    code: str,
) -> None:
    @rig.app.get("/_error")
    async def fail() -> None:
        raise error

    with TestClient(rig.app, raise_server_exceptions=False, client=("127.0.0.1", 12345)) as client:
        response = client.get("/_error")
    assert response.status_code == status
    assert response.json()["code"] == code
    assert set(response.json()) == {"code", "message", "run_id", "retry_after"}
    assert "secret-provider-token" not in response.text
    assert "dr_session=" in response.headers["set-cookie"]
    if status == 429:
        assert response.json()["retry_after"] == 7
        assert response.headers["retry-after"] == "7"


def test_owner_scope_matches_manager_boundary(rig: Rig) -> None:
    identity = rig.client.get("/_identity").json()
    assert identity["owner_scope_sha256"] == owner_scope_sha256(
        client_ip="127.0.0.1",
        session_id=identity["session_id"],
    )
