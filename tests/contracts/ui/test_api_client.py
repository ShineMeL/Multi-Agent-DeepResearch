import ast
import json
from pathlib import Path

import httpx
import pytest

from apps.ui.api_client import DemoCapabilities, ResearchApiClient, StreamReconnectExhausted
from apps.ui.replay import live_payload, replay_payload
from deepresearch.domain import ResourceUsage


def view(status="completed"):
    return {
        "run_id": "r1",
        "thread_id": "r1",
        "status": status,
        "stop_reason": None,
        "is_partial": False,
        "report_artifact_id": None,
        "evidence_graph_artifact_id": None,
        "manifest_artifact_id": None,
        "final_usage": ResourceUsage.zero().model_dump(mode="json"),
        "error_code": None,
    }


def frame(seq, status="running", *, run_id="r1", kind="node_completed"):
    event = {
        "seq": seq,
        "run_id": run_id,
        "timestamp": "2026-08-29T00:00:00Z",
        "node": "Plan",
        "kind": kind,
        "status": status,
        "public_payload": {},
        "usage_delta": ResourceUsage.zero().model_dump(mode="json"),
        "artifact_ids": [],
    }
    return f"id: {seq}\nevent: {kind}\ndata: {json.dumps(event)}\n\n".encode()


def capabilities(*, live_available=True, unpriced_live=True):
    return {
        "profiles": [
            {
                "profile_id": "replay-default",
                "execution_mode": "replay",
                "available": True,
                "reason": None,
                "workflow_id": "research-v1",
                "planner_id": "P1",
                "ranker_id": "R1",
            },
            {
                "profile_id": "live-default",
                "execution_mode": "live",
                "available": live_available,
                "reason": None if live_available else "PROVIDER_NOT_CONFIGURED",
                "workflow_id": "baseline-v1",
                "planner_id": "P1",
                "ranker_id": "R1",
            },
        ],
        "replay_example": {
            "provider_profile_id": "replay-default",
            "question": "Compare planner strategies",
            "report_language": "en",
            "source_languages": ["en"],
            "budget_preset": "medium",
            "seed": 0,
        },
        "budget_presets": ["low", "medium"],
        "unpriced_live": unpriced_live,
    }


class BrokenStream(httpx.SyncByteStream):
    def __iter__(self):
        yield frame(1) + frame(2)
        raise httpx.ReadError("disconnected")


def test_ui_has_no_provider_storage_runtime_or_server_imports():
    paths = list(Path("apps/ui").glob("*.py"))
    assert paths
    forbidden = (
        "deepresearch.providers",
        "deepresearch.storage",
        "deepresearch.runtime",
        "deepresearch.workflow",
        "apps.api",
        "sqlite3",
        "sqlalchemy",
    )
    for path in paths:
        for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
            names = (
                [item.name for item in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            assert not any(name.startswith(forbidden) for name in names), path


def test_replay_request_matches_strict_api_and_retains_cookie_for_all_routes():
    from apps.api.schemas import CreateRunRequest

    seen = []

    def respond(request):
        seen.append((request.method, request.url.path))
        if request.url.path == "/runs":
            assert request.headers["Idempotency-Key"] == "showcase-1"
            body = json.loads(request.content)
            parsed = CreateRunRequest.model_validate(body)
            assert parsed.request.execution_mode == "replay"
            assert parsed.request.report_language == "zh"
            assert (parsed.workflow_id, parsed.planner_id, parsed.ranker_id) == (
                "baseline-v1",
                "P1",
                "R1",
            )
            assert set(body) == {"request", "workflow_id", "planner_id", "ranker_id", "seed"}
            return httpx.Response(
                202,
                json={
                    "run_id": "r1",
                    "thread_id": "r1",
                    "status": "queued",
                    "events_url": "/runs/r1/events",
                },
                headers={"Set-Cookie": "dr_session=signed; Path=/; HttpOnly"},
            )
        assert request.headers["Cookie"] == "dr_session=signed"
        if request.url.path.endswith("/events"):
            return httpx.Response(200, content=frame(1, "completed"))
        if "/artifacts/" in request.url.path:
            return httpx.Response(200, content=b'{"schema_version":"run-manifest-v1"}')
        return httpx.Response(200, json=view())

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        assert (
            client.create_run(replay_payload("Question", report_language="zh"), "showcase-1").status
            == "queued"
        )
        client.get_run("r1")
        client.resume("r1")
        client.cancel("r1")
        assert [event.seq for event in client.events("r1")] == [1]
        for kind in ("report", "evidence", "manifest"):
            assert client.download_artifact("r1", kind) == b'{"schema_version":"run-manifest-v1"}'
    assert ("POST", "/runs/r1/resume") in seen
    assert ("POST", "/runs/r1/cancel") in seen
    assert seen[-1] == ("GET", "/runs/r1/artifacts/manifest")


def test_capabilities_are_typed_and_fetched_with_the_session_owned_http_client():
    def respond(request):
        assert request.method == "GET"
        assert request.url.path == "/capabilities"
        return httpx.Response(200, json=capabilities(live_available=False))

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        result = client.get_capabilities()

    assert result.replay_example is not None
    assert result.replay_example.provider_profile_id == "replay-default"
    assert result.replay_example.question == "Compare planner strategies"
    assert result.profiles[1].available is False
    assert result.profiles[1].reason == "PROVIDER_NOT_CONFIGURED"


def test_replay_profile_uses_the_example_binding_instead_of_profile_order():
    data = capabilities()
    builtin = data["profiles"][0]
    data["profiles"].insert(0, {**builtin, "profile_id": "custom-replay"})

    discovered = DemoCapabilities.model_validate(data)

    assert discovered.replay_profile() is not None
    assert discovered.replay_profile().profile_id == "replay-default"


def test_legacy_replay_example_is_safe_with_exactly_one_eligible_profile():
    data = capabilities()
    data["replay_example"].pop("provider_profile_id")

    discovered = DemoCapabilities.model_validate(data)

    assert discovered.replay_profile() is not None
    assert discovered.replay_profile().profile_id == "replay-default"


@pytest.mark.parametrize(
    "case",
    ["missing", "unavailable", "wrong-mode", "ambiguous-legacy"],
)
def test_replay_profile_fails_closed_for_an_unsafe_example_binding(case):
    data = capabilities()
    if case == "missing":
        data["replay_example"]["provider_profile_id"] = "missing-replay"
    elif case == "unavailable":
        data["profiles"][0].update(available=False, reason="PROVIDER_PROFILE_DRIFT")
    elif case == "wrong-mode":
        data["replay_example"]["provider_profile_id"] = "live-default"
    else:
        data["replay_example"].pop("provider_profile_id")
        data["profiles"].insert(0, {**data["profiles"][0], "profile_id": "custom-replay"})

    discovered = DemoCapabilities.model_validate(data)

    assert discovered.replay_profile() is None


def test_live_payload_posts_server_selected_baseline_profile_without_seed():
    payload = live_payload(
        "What changed?",
        report_language="zh",
        budget_preset="low",
        provider_profile_id="live-default",
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
    )

    assert payload == {
        "request": {
            "question": "What changed?",
            "output_requirements": {"answer_shape": "markdown"},
            "report_language": "zh",
            "source_languages": ["en"],
            "freshness_requirement": {
                "kind": "none",
                "published_after": None,
                "retrieved_within_days": None,
            },
            "execution_mode": "live",
            "access_profile": "showcase",
            "provider_profile_id": "live-default",
            "run_purpose": "demo",
            "budget_preset": "low",
        },
        "workflow_id": "baseline-v1",
        "planner_id": "P1",
        "ranker_id": "R1",
        "seed": None,
    }


def test_client_reconnects_from_last_durable_sequence(monkeypatch):
    monkeypatch.setattr(ResearchApiClient, "_wait_for_reconnect", lambda self, _: False)
    cursors = []

    def respond(request):
        if request.url.path.endswith("events"):
            cursors.append(request.headers["Last-Event-ID"])
            if len(cursors) == 1:
                return httpx.Response(
                    200, stream=BrokenStream(), headers={"Set-Cookie": "dr_session=signed; Path=/"}
                )
            assert request.headers["Cookie"] == "dr_session=signed"
            # Repeated rows cannot regress the durable cursor or duplicate output.
            return httpx.Response(200, content=frame(2) + frame(3, "completed"))
        return httpx.Response(200, json=view())

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        assert [event.seq for event in client.events("r1")] == [1, 2, 3]
    assert cursors == ["0", "2"]


@pytest.mark.parametrize("historical_status", ["interrupted", "completed", "failed", "cancelled"])
def test_historical_node_status_does_not_override_current_run(monkeypatch, historical_status):
    monkeypatch.setattr(ResearchApiClient, "_wait_for_reconnect", lambda self, _: False)
    cursors, statuses = [], []

    def respond(request):
        if request.url.path.endswith("events"):
            cursors.append(request.headers["Last-Event-ID"])
            data = (
                frame(1, historical_status)
                if len(cursors) == 1
                else frame(2) + frame(3, "completed", kind="run_completed")
            )
            return httpx.Response(200, content=data)
        status = "running" if len(cursors) == 1 else "completed"
        statuses.append(status)
        return httpx.Response(200, json=view(status))

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        assert [event.seq for event in client.events("r1")] == [1, 2, 3]
    assert cursors == ["0", "1"]
    assert statuses == ["running", "completed"]


@pytest.mark.parametrize("status", ["interrupted", "completed", "failed", "cancelled"])
def test_empty_eof_checks_current_terminal_run(status):
    seen = []

    def respond(request):
        seen.append(request.url.path)
        return (
            httpx.Response(200, content=b": heartbeat\n\n")
            if request.url.path.endswith("events")
            else httpx.Response(200, json=view(status))
        )

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        assert list(client.events("r1", 8)) == []
    assert seen == ["/runs/r1/events", "/runs/r1"]


def test_unterminated_frame_never_advances_cursor(monkeypatch):
    monkeypatch.setattr(ResearchApiClient, "_wait_for_reconnect", lambda self, _: False)
    cursors = []

    def respond(request):
        if request.url.path.endswith("events"):
            cursors.append(request.headers["Last-Event-ID"])
            return httpx.Response(
                200,
                content=(frame(1).rstrip(b"\n") if len(cursors) == 1 else frame(1, "completed")),
            )
        return httpx.Response(200, json=view("running" if len(cursors) == 1 else "completed"))

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        assert [event.seq for event in client.events("r1")] == [1]
    assert cursors == ["0", "0"]


def test_bounded_reconnect_exposes_cursor(monkeypatch):
    monkeypatch.setattr(ResearchApiClient, "_wait_for_reconnect", lambda self, _: False)
    cursors = []

    def respond(request):
        if request.url.path.endswith("events"):
            cursors.append(request.headers["Last-Event-ID"])
            raise httpx.ReadTimeout("no heartbeat")
        return httpx.Response(200, json=view("running"))

    with (
        ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client,
        pytest.raises(StreamReconnectExhausted) as error,
    ):
        list(client.events("r1", 7))
    assert error.value.run_id == "r1" and error.value.last_event_id == 7
    assert cursors == ["7"] * 6


def test_client_rejects_foreign_event():
    with (
        ResearchApiClient(
            "http://api",
            transport=httpx.MockTransport(
                lambda _: httpx.Response(200, content=frame(1, run_id="foreign"))
            ),
        ) as client,
        pytest.raises(ValueError, match="run"),
    ):
        list(client.events("r1"))


def test_http_failure_is_not_retried_or_converted_to_success():
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(404, json={"error": {"code": "RUN_NOT_FOUND"}})

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        with pytest.raises(httpx.HTTPStatusError):
            list(client.events("r1"))
        with pytest.raises(ValueError):
            client.download_artifact("r1", "../../secret")
    assert len(calls) == 1
