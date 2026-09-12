"""Deployment acceptance for the bounded, replay-only HTTP smoke client."""

from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path

import httpx
import pytest
from fastapi import FastAPI
from pydantic import SecretStr
from starlette.types import ASGIApp

from apps.api.main import create_app
from apps.api.settings import ServiceSettings
from deepresearch.workflow.runner import BaselineRuntimeHooks
from scripts.smoke_replay import SmokeError, main, replay_smoke
from tests.integration.replay.test_baseline_graph import ControlledSegmentClock


def _replay_app(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> FastAPI:
    source = Path("tests/fixtures/replay/baseline").resolve()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for item in source.iterdir():
        if item.is_file():
            bundle.joinpath(item.name).write_bytes(item.read_bytes().replace(b"\r\n", b"\n"))

    profiles = json.loads(Path("deploy/replay/profiles.json").read_text(encoding="utf-8"))
    for route in profiles["profiles"]["replay-default"]["routes"]:
        if "bundle_path" in route["parameters"]:
            route["parameters"]["bundle_path"] = str(bundle)
    profiles_path = tmp_path / "profiles.json"
    profiles_path.write_text(json.dumps(profiles), encoding="utf-8")

    def runtime_clock() -> BaselineRuntimeHooks:
        clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
        return BaselineRuntimeHooks(monotonic=clock.monotonic, utc_now=clock.utc_now)

    monkeypatch.setattr("deepresearch.runtime.runner_factory.paired_runtime_hooks", runtime_clock)
    return create_app(
        ServiceSettings(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
            artifact_root=tmp_path,
            checkpoint_sqlite_path=tmp_path / "checkpoints.sqlite",
            provider_profile_catalog_path=profiles_path,
            pricing_catalog_path=Path("deploy/replay/pricing.json"),
            deployment_access_profile="local",
            allowed_execution_modes=("replay",),
            allowed_provider_profile_ids=("replay-default",),
            allowed_run_purposes=("demo",),
            allowed_budget_presets=("medium",),
            session_signing_key=SecretStr("s" * 32),
            langgraph_strict_msgpack=True,
        )
    )


class _CorruptingTransport(httpx.AsyncBaseTransport):
    def __init__(self, app: ASGIApp) -> None:
        self._inner = httpx.ASGITransport(app=app)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        response = await self._inner.handle_async_request(request)
        if request.url.path.endswith("/artifacts/report"):
            content = await response.aread()
            await response.aclose()
            return httpx.Response(
                response.status_code,
                headers=response.headers,
                content=content + b"corrupt",
                request=request,
            )
        return response

    async def aclose(self) -> None:
        await self._inner.aclose()


async def test_real_service_replay_proves_terminal_artifacts_and_zero_cost(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    app = _replay_app(tmp_path, monkeypatch)
    async with app.router.lifespan_context(app):
        result = await replay_smoke(
            "http://testserver",
            timeout=30,
            api_transport=httpx.ASGITransport(app=app),
        )

    assert result["ok"] is True
    assert result["mode"] == "replay"
    assert result["status"] == "completed"
    assert result["stop_reason"] == "SUFFICIENT"
    assert result["is_partial"] is False
    assert result["event_count"] > 0
    assert result["cost_usd"] == "0"
    assert set(result["artifact_sha256"]) == {"report", "evidence", "manifest"}
    assert all(len(value) == 64 for value in result["artifact_sha256"].values())


@pytest.mark.parametrize(
    ("capabilities", "code"),
    [
        ({"profiles": [], "replay_example": None, "budget_presets": [], "unpriced_live": False},
         "NO_REPLAY_CAPABILITY"),
        ({"profiles": "not-a-list"}, "MALFORMED_RESPONSE"),
    ],
)
async def test_capability_rejection_is_explicit_and_never_creates_a_run(
    capabilities: dict[str, object], code: str
):
    requested: list[str] = []

    async def handler(request: httpx.Request) -> httpx.Response:
        requested.append(request.url.path)
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ok", "checks": {"database": "ok"}})
        if request.url.path == "/capabilities":
            return httpx.Response(200, json=capabilities)
        raise AssertionError(f"unexpected request: {request.url.path}")

    with pytest.raises(SmokeError) as caught:
        await replay_smoke(
            "http://testserver", timeout=1, api_transport=httpx.MockTransport(handler)
        )

    assert caught.value.code == code
    assert "/runs" not in requested


async def test_download_hash_mismatch_fails_after_real_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    app = _replay_app(tmp_path, monkeypatch)
    async with app.router.lifespan_context(app):
        with pytest.raises(SmokeError) as caught:
            await replay_smoke(
                "http://testserver",
                timeout=30,
                api_transport=_CorruptingTransport(app),
            )

    assert caught.value.code == "ARTIFACT_HASH_MISMATCH"
    assert caught.value.status == "completed"
    assert caught.value.cancel_attempted is False


async def test_accepted_run_timeout_attempts_bounded_owner_scoped_cancel():
    cancelled = False
    never = asyncio.Event()
    run_id = "accepted-run"

    def view(status: str) -> dict[str, object]:
        return {
            "run_id": run_id,
            "thread_id": "thread-1",
            "status": status,
            "stop_reason": None,
            "is_partial": status != "completed",
            "report_artifact_id": None,
            "evidence_graph_artifact_id": None,
            "manifest_artifact_id": None,
            "final_usage": None,
            "error_code": "CANCELLED_BY_USER" if status == "cancelled" else None,
        }

    async def handler(request: httpx.Request) -> httpx.Response:
        nonlocal cancelled
        if request.url.path == "/health/ready":
            return httpx.Response(200, json={"status": "ok", "checks": {"database": "ok"}})
        if request.url.path == "/capabilities":
            return httpx.Response(
                200,
                json={
                    "profiles": [{
                        "profile_id": "replay-default",
                        "execution_mode": "replay",
                        "available": True,
                        "reason": None,
                        "workflow_id": "research-v1",
                        "planner_id": "P1",
                        "ranker_id": "R1",
                    }],
                    "replay_example": {
                        "question": "Compare planner strategies",
                        "report_language": "en",
                        "source_languages": ["en"],
                        "budget_preset": "medium",
                        "seed": 0,
                    },
                    "budget_presets": ["medium"],
                    "unpriced_live": False,
                },
            )
        if request.method == "POST" and request.url.path == "/runs":
            return httpx.Response(
                202,
                json={
                    "run_id": run_id,
                    "thread_id": "thread-1",
                    "status": "queued",
                    "events_url": "https://attacker.invalid/steal",
                },
            )
        if request.method == "GET" and request.url.path.endswith("/events"):
            await never.wait()
        if request.method == "POST" and request.url.path.endswith("/cancel"):
            cancelled = True
            return httpx.Response(200, json=view("cancelled"))
        raise AssertionError(f"unexpected request: {request.method} {request.url.path}")

    with pytest.raises(SmokeError) as caught:
        await replay_smoke(
            "http://testserver", timeout=0.05, api_transport=httpx.MockTransport(handler)
        )

    assert caught.value.code == "TIMEOUT"
    assert caught.value.run_id == run_id
    assert caught.value.cancel_attempted is True
    assert caught.value.cancel_status == "cancelled"
    assert cancelled is True


@pytest.mark.parametrize(
    "url",
    [
        "http://user:super-secret@127.0.0.1:8000",
        "http://127.0.0.1:8000?token=super-secret",
        "http://127.0.0.1:8000/#super-secret",
    ],
)
def test_cli_rejects_credential_query_and_fragment_without_echoing_them(
    url: str, capsys: pytest.CaptureFixture[str]
):
    exit_code = main(["--api-url", url])
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 2
    assert output == {"ok": False, "code": "INVALID_BASE_URL"}
    assert "super-secret" not in json.dumps(output)
