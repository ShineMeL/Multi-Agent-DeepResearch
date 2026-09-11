import json
import logging
from pathlib import Path
from threading import Event

import httpx
import pytest
from fastapi.testclient import TestClient
from streamlit.testing.v1 import AppTest

from apps.api.main import create_app
from apps.api.settings import ServiceSettings
from apps.ui.api_client import ResearchApiClient, StreamReconnectExhausted
from apps.ui.replay import ShowcaseSession, replay_payload, research_replay_payload
from deepresearch.domain import RunEvent
from tests.contracts.ui.test_api_client import frame, view
from tests.unit.runtime.test_manager import ControlledRunner


@pytest.fixture(autouse=True)
def restore_streamlit_server_logging():
    # AppTest config initialization calls Streamlit init_uvicorn_logs(), which
    # sets propagate=False and installs console handlers on these shared names.
    # Preserve the embedding test process's loggers for later API tests.
    names = ("uvicorn", "uvicorn.access", "uvicorn.asgi", "uvicorn.error", "websockets")
    before = [
        (logger, logger.level, logger.propagate, list(logger.handlers))
        for name in names
        for logger in (logging.getLogger(name),)
    ]
    yield
    for logger, level, propagate, handlers in before:
        logger.setLevel(level)
        logger.propagate = propagate
        logger.handlers[:] = handlers


def test_showcase_uses_hosted_api_owner_cookie_durable_events_and_artifact_routes(
    tmp_path, monkeypatch
):
    app = create_app(
        ServiceSettings(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
            artifact_root=tmp_path / "artifacts",
            checkpoint_sqlite_path=tmp_path / "artifacts" / "checkpoints.sqlite",
            session_signing_key="showcase-test-signing-key-at-least-32-bytes",
            langgraph_strict_msgpack=True,
        )
    )
    with TestClient(app, client=("127.0.0.1", 1234)) as hosted:
        runner = ControlledRunner()
        runner.finish.set()
        artifact_bytes = {
            "report": b"# Hosted report",
            "evidence": b'{"sources":[],"evidence":[]}',
            "manifest": b'{"schema_version":"run-manifest-v1"}',
        }
        artifact_ids = {
            kind: app.state.artifact_store.put_bytes(
                data, media_type="text/markdown" if kind == "report" else "application/json"
            ).artifact_id
            for kind, data in artifact_bytes.items()
        }
        original_run = runner.run

        async def run_with_artifacts(**kwargs):
            result = await original_run(**kwargs)
            return result.model_copy(
                update={
                    "report_artifact_id": artifact_ids["report"],
                    "evidence_graph_artifact_id": artifact_ids["evidence"],
                    "manifest_artifact_id": artifact_ids["manifest"],
                }
            )

        runner.run = run_with_artifacts
        # Boundary integration: the real Task 8 composition/store/SSE surrounds
        # a controlled runner; this is not evidence of production research-v1.
        monkeypatch.setattr(app.state.manager.runner_factory, "create", lambda **_: runner)
        with ResearchApiClient("http://testserver", transport=hosted._transport) as client:
            accepted = client.create_run(replay_payload("Showcase"), "showcase-1")
            hosted.portal.call(app.state.manager.wait, accepted.run_id)
            events = list(client.events(accepted.run_id))
            assert events[-1].kind == "run_completed"
            assert client.get_run(accepted.run_id).status == "completed"
            assert list(client.events(accepted.run_id, events[-1].seq)) == []
            assert client.client.cookies.get("dr_session")
            for kind, expected in artifact_bytes.items():
                assert client.download_artifact(accepted.run_id, kind) == expected
        with ResearchApiClient("http://testserver", transport=hosted._transport) as stranger:
            with pytest.raises(httpx.HTTPStatusError) as error:
                stranger.get_run(accepted.run_id)
            assert error.value.response.status_code == 404


def test_empty_default_profile_failure_is_displayed_without_fake_success(tmp_path):
    app = create_app(
        ServiceSettings(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
            artifact_root=tmp_path / "artifacts",
            checkpoint_sqlite_path=tmp_path / "artifacts" / "checkpoints.sqlite",
            session_signing_key="showcase-test-signing-key-at-least-32-bytes",
            langgraph_strict_msgpack=True,
        )
    )
    with (
        TestClient(app, client=("127.0.0.1", 1234)) as hosted,
        ResearchApiClient("http://testserver", transport=hosted._transport) as client,
    ):
        session = ShowcaseSession(client)
        with pytest.raises(httpx.HTTPStatusError) as error:
            session.submit(replay_payload("Showcase"))
        assert error.value.response.json()["code"] == "PROVIDER_PROFILE_DRIFT"
        assert session.run_id is None


def test_replay_payload_uses_the_baseline_fixture_output_shape():
    payload = replay_payload("Compare planner strategies")

    assert payload["request"]["output_requirements"] == {"answer_shape": "markdown"}
    assert payload["request"]["budget_preset"] == "medium"


def test_research_replay_payload_selects_supported_production_composition():
    payload = research_replay_payload("Compare planner strategies")

    assert payload["workflow_id"] == "research-v1"
    assert payload["planner_id"] == "P1"
    assert payload["ranker_id"] == "R1"
    assert payload["request"]["execution_mode"] == "replay"


def test_app_submits_replay_shows_downloads_metrics_and_preserves_session():
    seen, keys = [], []
    manifest = {
        "schema_version": "run-manifest-v1",
        "provider_calls": [
            {"operation": "search", "normalized_query": "durable replay", "outcome": "success"}
        ],
    }
    evidence = {
        "sources": [{"source_id": "s1", "title": "Source"}],
        "evidence": [{"evidence_id": "e1", "source_id": "s1", "excerpt": "Evidence excerpt"}],
    }
    current = view()
    current.update(
        report_artifact_id="report-id",
        evidence_graph_artifact_id="evidence-id",
        manifest_artifact_id="manifest-id",
    )
    current["final_usage"].update(input_tokens=42, total_tokens=42, wall_seconds=1.5)

    def respond(request):
        seen.append(request.url.path)
        if request.url.path == "/runs":
            keys.append(request.headers["Idempotency-Key"])
            assert json.loads(request.content)["request"]["execution_mode"] == "replay"
            return httpx.Response(
                202,
                json={
                    "run_id": "r1",
                    "thread_id": "r1",
                    "status": "queued",
                    "events_url": "/runs/r1/events",
                },
                headers={"Set-Cookie": "dr_session=owner; Path=/"},
            )
        assert request.headers["Cookie"] == "dr_session=owner"
        if request.url.path.endswith("/events"):
            return httpx.Response(200, content=frame(1) + frame(2, "completed"))
        if request.url.path.endswith("/report"):
            return httpx.Response(200, content=b"# API report\nA supported finding.")
        if request.url.path.endswith("/manifest"):
            return httpx.Response(200, json=manifest)
        if request.url.path.endswith("/evidence"):
            return httpx.Response(200, json=evidence)
        return httpx.Response(200, json=current)

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        session = ShowcaseSession(client)
        app = AppTest.from_file(str(Path("apps/ui/app.py").resolve()), default_timeout=5)
        app.session_state["showcase"] = session
        app.run()
        assert not app.exception
        assert any("research-v1" in info.value for info in app.info)
        assert next(item for item in app.selectbox if item.label == "Budget").value == "medium"
        app.text_area(key="question").input("Showcase question")
        app.button(key="FormSubmitter:replay_request-Start replay").click().run()
        assert session.finished.wait(3)
        app.run()
        assert not app.exception
        assert session.cursor == 2
        assert any(metric.label == "Tokens" and metric.value == "42" for metric in app.metric)
        assert any(
            metric.label == "Cost (USD)" and metric.value == "Unknown" for metric in app.metric
        )
        assert any("API report" in block.value for block in app.markdown)
        assert len(app.get("download_button")) == 3
        assert len(app.get("graphviz_chart")) == 1
        assert any("not exposed" in info.value.lower() for info in app.info)
        assert len(keys) == 1
        assert app.session_state["showcase"] is session
        assert {f"/runs/r1/artifacts/{kind}" for kind in ("report", "evidence", "manifest")} <= set(
            seen
        )
        status_reads = seen.count("/runs/r1")
        for _ in range(3):
            app.run()
        assert seen.count("/runs/r1") == status_reads


def test_submit_retry_keeps_idempotency_key_and_payload():
    keys, payloads = [], []

    def respond(request):
        keys.append(request.headers["Idempotency-Key"])
        payloads.append(json.loads(request.content))
        if len(keys) == 1:
            raise httpx.ReadTimeout("reply lost after create")
        return httpx.Response(
            202,
            json={
                "run_id": "r1",
                "thread_id": "r1",
                "status": "queued",
                "events_url": "/runs/r1/events",
            },
        )

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        session = ShowcaseSession(client)
        with pytest.raises(httpx.ReadTimeout):
            session.submit(replay_payload("Original"))
        session.submit(replay_payload("Edited"))
        assert keys[0] == keys[1]
        assert payloads[0] == payloads[1]
        assert payloads[1]["request"]["question"] == "Original"


def test_background_watch_restarts_from_persisted_cursor(monkeypatch):
    monkeypatch.setattr(ResearchApiClient, "_wait_for_reconnect", lambda self, _: False)
    cursors = []
    fail = True

    def respond(request):
        if request.url.path.endswith("events"):
            cursors.append(request.headers["Last-Event-ID"])
            if fail and len(cursors) > 1:
                raise httpx.ReadError("disconnected")
            return httpx.Response(200, content=frame(1) if fail else frame(2, "completed"))
        return httpx.Response(200, json=view("running" if fail else "completed"))

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        session = ShowcaseSession(client)
        session.run_id = "r1"
        session.watch()
        assert session.finished.wait(3)
        session.drain()
        assert session.cursor == 1 and session.stream_error is not None
        fail = False
        session.watch()
        assert session.finished.wait(3)
        session.drain()
        assert [event.seq for event in session.events] == [1, 2]
        assert cursors[-1] == "1"
        assert session.stream_error is None


def test_app_resume_refusal_keeps_run_and_cancel_remains_available():
    seen = []
    status = "interrupted"

    def respond(request):
        nonlocal status
        seen.append((request.method, request.url.path))
        if request.url.path.endswith("/resume"):
            return httpx.Response(409, json={"code": "CHECKPOINT_RESUME_UNAVAILABLE"})
        if request.url.path.endswith("/cancel"):
            status = "cancelled"
        return httpx.Response(200, json=view(status))

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        session = ShowcaseSession(client, run_id="r1")
        app = AppTest.from_file(str(Path("apps/ui/app.py").resolve()), default_timeout=5)
        app.session_state["showcase"] = session
        app.run()
        assert session.poll_finished.wait(2)
        app.run()
        next(button for button in app.button if button.label == "Resume").click().run()
        assert not app.exception
        assert any("Checkpoint continuation is unavailable" in error.value for error in app.error)
        app.run()
        next(button for button in app.button if button.label == "Cancel").click().run()
        assert not app.exception
        assert ("POST", "/runs/r1/resume") in seen
        assert ("POST", "/runs/r1/cancel") in seen
        assert session.run_id == "r1"


def test_active_run_metrics_use_durable_usage_before_final_usage_is_published():
    event_json = json.loads(frame(1).decode().split("data: ")[1].strip())
    event_json["usage_delta"].update(input_tokens=15, total_tokens=15, wall_seconds=2)
    with ResearchApiClient(
        "http://api",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, json=view("running"))),
    ) as client:
        session = ShowcaseSession(
            client, run_id="r1", events=[RunEvent.model_validate(event_json)], cursor=1
        )
        app = AppTest.from_file(str(Path("apps/ui/app.py").resolve()), default_timeout=5)
        app.session_state["showcase"] = session
        app.run()
        assert session.poll_finished.wait(2)
        app.run()
        assert not app.exception
        assert any(metric.label == "Tokens" and metric.value == "15" for metric in app.metric)
        assert any(
            metric.label == "Time (seconds)" and metric.value == "2.00" for metric in app.metric
        )
        assert not next(button for button in app.button if button.label == "Cancel").disabled


def test_fragment_completion_refreshes_form_outside_fragment():
    with ResearchApiClient(
        "http://api", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=view()))
    ) as client:
        session = ShowcaseSession(client, run_id="r1")
        app = AppTest.from_string(
            """
import streamlit as st
from apps.ui.app import _run_panel
if not st.session_state.get("rendered"):
    st.session_state["rendered"] = True
    st.button("Start replay", disabled=True)
    _run_panel(st.session_state["showcase"], watching_at_render=True)
else:
    st.button("Start replay", disabled=False)
""",
            default_timeout=5,
        )
        app.session_state["showcase"] = session
        app.run()
        assert not app.exception
        assert not next(button for button in app.button if button.label == "Start replay").disabled


@pytest.mark.parametrize("status", ["completed", "cancelled", "failed", "interrupted"])
def test_final_run_stops_status_reads_even_after_many_refresh_ticks(status):
    calls = []
    with ResearchApiClient(
        "http://api",
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, json=view(status))
        ),
    ) as client:
        session = ShowcaseSession(client, run_id="r1")
        session.refresh(now=0)
        assert session.poll_finished.wait(2)
        session.drain(now=0)
        for tick in range(1, 3600):
            session.refresh(now=tick)
        assert len(calls) == 1
        assert not session.automatic_refresh
        assert session.view.status == status


def test_poll_transport_failures_back_off_then_require_explicit_retry():
    calls = []
    fail = True

    def respond(request):
        calls.append(request)
        if fail:
            raise httpx.ConnectError("offline")
        return httpx.Response(200, json=view())

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        session = ShowcaseSession(client, run_id="r1")
        for tick, expected in ((0, 1), (1, 1), (2, 2), (5, 2), (6, 3), (3600, 3)):
            session.refresh(now=tick)
            assert session.poll_finished.wait(2)
            session.drain(now=tick)
            assert len(calls) == expected
        assert not session.automatic_refresh
        assert session.poll_error is not None
        fail = False
        session.retry_status()
        session.refresh(now=3601)
        assert session.poll_finished.wait(2)
        session.drain(now=3601)
        assert len(calls) == 4
        assert session.view.status == "completed"


def test_missing_run_stops_after_one_status_read():
    calls = []
    with ResearchApiClient(
        "http://api",
        transport=httpx.MockTransport(lambda request: calls.append(request) or httpx.Response(404)),
    ) as client:
        session = ShowcaseSession(client, run_id="r1")
        session.refresh(now=0)
        assert session.poll_finished.wait(2)
        session.drain(now=0)
        session.refresh(now=3600)
        assert len(calls) == 1
        assert not session.automatic_refresh


def test_status_request_is_background_and_only_one_is_in_flight():
    entered, release = Event(), Event()
    calls = []

    def respond(request):
        calls.append(request)
        entered.set()
        assert release.wait(3)
        return httpx.Response(200, json=view())

    with ResearchApiClient("http://api", transport=httpx.MockTransport(respond)) as client:
        session = ShowcaseSession(client, run_id="r1")
        try:
            session.refresh(now=0)
            assert entered.wait(1)
            assert not session.poll_finished.is_set()
            session.refresh(now=100)
            assert len(calls) == 1
        finally:
            release.set()
            assert session.poll_finished.wait(2)


def test_exhausted_stream_pauses_automatic_status_reads():
    calls = []
    with ResearchApiClient(
        "http://api",
        transport=httpx.MockTransport(
            lambda request: calls.append(request) or httpx.Response(200, json=view("running"))
        ),
    ) as client:
        session = ShowcaseSession(client, run_id="r1")
        session.stream_error = StreamReconnectExhausted("r1", 8)
        session.refresh(now=0)
        assert not calls
        assert not session.automatic_refresh


def test_session_close_stops_reader_and_closes_client():
    entered, released = Event(), Event()

    class WaitingStream(httpx.SyncByteStream):
        def __iter__(self):
            entered.set()
            assert released.wait(3)
            yield b": closed\n\n"

        def close(self):
            released.set()

    client = ResearchApiClient(
        "http://api",
        transport=httpx.MockTransport(lambda _: httpx.Response(200, stream=WaitingStream())),
    )
    session = ShowcaseSession(client, run_id="r1")
    session.watch()
    assert entered.wait(1)
    session.close()
    assert session.finished.wait(2)
    session.close()
    assert client.client.is_closed
    assert session.closed
    assert not session.automatic_refresh


def test_streamlit_session_resource_release_closes_actual_ui_client(monkeypatch):
    from streamlit.runtime.caching import clear_session_resource_cache

    from apps.ui import app as ui

    client = ResearchApiClient(
        "http://api", transport=httpx.MockTransport(lambda _: httpx.Response(200, json=view()))
    )
    monkeypatch.setattr(ui, "ResearchApiClient", lambda _: client)
    app = AppTest.from_string("""
import streamlit as st
from streamlit.runtime.scriptrunner_utils.script_run_context import get_script_run_ctx
from apps.ui.app import _session_resource
st.session_state["resource"] = _session_resource("http://api")
st.session_state["session_id"] = get_script_run_ctx().session_id
""")
    app.run()
    assert not app.exception
    session = app.session_state["resource"]
    clear_session_resource_cache(app.session_state["session_id"])
    assert session.closed
    assert client.client.is_closed
