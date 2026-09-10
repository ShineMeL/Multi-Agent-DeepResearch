from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.domain import ResourceUsage, RunEvent, RunStatus
from deepresearch.runtime.manager import RunManager, RunNotFound
from deepresearch.runtime.runner_factory import FilePricingCatalog
from deepresearch.storage.protocols import RunFinalization, RunStore, TerminalEventDraft
from tests.contracts.api.test_runs_api import Rig, seed
from tests.contracts.api.test_runs_api import (
    rig as rig,  # noqa: PLC0414 -- pytest fixture re-export
)
from tests.fakes.service_store import FakeRunStore, make_record
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_manager import Factory, policy

OWNER = "c" * 64


def make_event(seq: int, *, run_id: str = "r1", status: RunStatus = "running") -> RunEvent:
    return RunEvent(
        run_id=run_id,
        seq=seq,
        timestamp=datetime(2026, 8, 29, tzinfo=UTC),
        node="Search",
        kind="node_finished",
        status=status,
        public_payload={"message": "progress"},
        usage_delta=ResourceUsage.zero(),
        artifact_ids=(),
        error_code=None,
    )


def make_manager(store: RunStore) -> RunManager:
    conf = config()
    return RunManager(
        runner_factory=Factory(conf),
        store=store,
        checkpointer=InMemorySaver(),
        pricing_catalog=FilePricingCatalog({}),
        deployment_policy=policy(conf),
    )


async def finalize(
    store: RunStore, run_id: str = "r1", status: RunStatus = "completed"
) -> RunEvent:
    _, event = await store.finalize_run(
        run_id,
        "running",
        RunFinalization(status, None, False, None, None, None, ResourceUsage.zero(), None),
        TerminalEventDraft(datetime.now(UTC), "service", f"run_{status}", {}),
    )
    return event


def sse_frames(body: str) -> list[dict[str, str]]:
    return [
        dict(
            line.split(": ", 1) for line in block.splitlines() if line and not line.startswith(":")
        )
        for block in body.replace("\r\n", "\n").split("\n\n")
        if block and not block.startswith(":")
    ]


@pytest.mark.parametrize(
    "cursor,want", [(None, ["1", "2"]), ("0", ["1", "2"]), ("1", ["2"]), ("2", []), ("99", [])]
)
def test_last_event_id_replays_only_later_durable_events(
    rig: Rig, cursor: str | None, want: list[str]
) -> None:
    run_id = seed(rig, "running")
    assert rig.client.portal is not None
    rig.client.portal.call(rig.store.append_event, make_event(1, run_id=run_id))
    rig.client.portal.call(finalize, rig.store, run_id)
    headers = {} if cursor is None else {"Last-Event-ID": cursor}
    response = rig.client.get(f"/runs/{run_id}/events", headers=headers)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    frames = sse_frames(response.text)
    assert [frame["id"] for frame in frames] == want
    if frames:
        assert frames[-1]["event"] == "run_completed"
        assert json.loads(frames[-1]["data"])["run_id"] == run_id
    assert not rig.manager._subscribers


@pytest.mark.parametrize("value", ["-1", "abc", "1.5", "", "+1", " 1", "1 ", "1_0"])
def test_malformed_last_event_id_returns_stable_422(rig: Rig, value: str) -> None:
    run_id = seed(rig)
    response = rig.client.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": value})
    assert response.status_code == 422
    assert response.json() == {
        "code": "INVALID_LAST_EVENT_ID",
        "message": "Invalid event cursor.",
        "run_id": None,
        "retry_after": None,
    }
    assert response.headers["cache-control"] == "no-store"
    assert not rig.manager._subscribers


@pytest.mark.parametrize("cursor", [None, "0", "99", "invalid"])
def test_foreign_and_missing_streams_are_identical_before_headers(rig: Rig, cursor: str | None):
    run_id = seed(rig)
    identity = rig.client.get("/_identity").json()
    headers = {
        "X-Session-ID": identity["session_id"],
        "X-Owner-Scope-Sha256": identity["owner_scope_sha256"],
    }
    if cursor is not None:
        headers["Last-Event-ID"] = cursor
    with TestClient(rig.app, base_url="https://testserver", client=("127.0.0.1", 12346)) as other:
        foreign = other.get(f"/runs/{run_id}/events", headers=headers)
        missing = other.get("/runs/missing/events", headers=headers)
    assert foreign.status_code == missing.status_code == 404
    assert foreign.json() == missing.json()
    assert foreign.json()["code"] == "RUN_NOT_FOUND"
    assert "dr_session=" in foreign.headers["set-cookie"]
    assert not rig.manager._subscribers


@pytest.mark.parametrize(
    "value,want",
    [
        (None, 0),
        ("0", 0),
        ("1", 1),
        ("0012", 12),
        pytest.param("9" * 5000, 10**5000 - 1, id="large-valid-decimal"),
    ],
)
def test_cursor_parser(value: str | None, want: int):
    from apps.api.sse import parse_last_event_id

    assert parse_last_event_id(value) == want


@pytest.mark.parametrize("value", ["١", "²", "１２", "\n1", "1\n"])
def test_cursor_rejects_non_ascii_decimal(value: str):
    from apps.api.sse import InvalidLastEventId, parse_last_event_id

    with pytest.raises(InvalidLastEventId):
        parse_last_event_id(value)


class ControlledStore(FakeRunStore):
    def __init__(self) -> None:
        super().__init__()
        self.pause_snapshot = False
        self.snapshot_taken = asyncio.Event()
        self.release_snapshot = asyncio.Event()

    async def list_events_after(self, run_id: str, seq: int) -> list[RunEvent]:
        snapshot = await super().list_events_after(run_id, seq)
        if self.pause_snapshot:
            self.pause_snapshot = False
            self.snapshot_taken.set()
            await self.release_snapshot.wait()
        return snapshot


@pytest.fixture
async def stream_rig() -> AsyncIterator[tuple[RunManager, ControlledStore]]:
    store = ControlledStore()
    await store.create_run(make_record("r1", status="running"))
    manager = make_manager(store)
    yield manager, store
    await manager.shutdown(0)


@pytest.mark.parametrize("terminal", [False, True])
async def test_snapshot_race_drains_committed_event(stream_rig, terminal: bool):
    from apps.api.sse import event_stream

    manager, store = stream_rig
    await store.append_event(make_event(1))
    store.pause_snapshot = True
    stream = event_stream(
        "r1", last_event_id=1, owner_scope_sha256=OWNER, store=store, manager=manager
    )
    pending = asyncio.create_task(anext(stream))
    try:
        await asyncio.wait_for(store.snapshot_taken.wait(), 1)
        if terminal:
            # Atomic durable finalization, deliberately without a manager notification.
            await finalize(store)
        else:
            await manager.emit(make_event(2))
        store.release_snapshot.set()
        assert (await asyncio.wait_for(pending, 1))["id"] == "2"
    finally:
        store.release_snapshot.set()
        pending.cancel()
        await asyncio.gather(pending, return_exceptions=True)
        await stream.aclose()
    assert not manager._subscribers


async def test_historical_interruption_does_not_hide_resumed_terminal(stream_rig):
    from apps.api.sse import event_stream

    manager, store = stream_rig
    await finalize(store, status="interrupted")
    await store.transition("r1", "interrupted", "running")
    await store.append_event(make_event(2))
    await finalize(store)
    frames = [
        frame
        async for frame in event_stream(
            "r1", last_event_id=0, owner_scope_sha256=OWNER, store=store, manager=manager
        )
    ]
    assert [frame["id"] for frame in frames] == ["1", "2", "3"]
    assert not manager._subscribers


@pytest.mark.parametrize("node_status", ["failed", "cancelled", "completed"])
async def test_node_status_does_not_close_running_service(stream_rig, node_status: RunStatus):
    from apps.api.sse import event_stream

    manager, store = stream_rig
    await manager.emit(make_event(1, status=node_status))
    stream = event_stream(
        "r1", last_event_id=0, owner_scope_sha256=OWNER, store=store, manager=manager
    )
    assert (await anext(stream))["id"] == "1"
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    terminal = await finalize(store)
    await manager.broadcast_persisted(terminal)
    assert (await asyncio.wait_for(pending, 1))["id"] == "2"
    with pytest.raises(StopAsyncIteration):
        await anext(stream)
    assert not manager._subscribers


async def test_coalesced_wakeups_replay_all_durable_events_without_duplicates(stream_rig):
    from apps.api.sse import event_stream

    manager, store = stream_rig
    await manager.emit(make_event(1))
    stream = event_stream(
        "r1", last_event_id=0, owner_scope_sha256=OWNER, store=store, manager=manager
    )
    assert (await anext(stream))["id"] == "1"
    await manager.emit(make_event(2))
    await manager.emit(make_event(3))
    await manager.broadcast_persisted(make_event(3))
    await finalize(store)
    assert [frame["id"] async for frame in stream] == ["2", "3", "4"]


async def test_heartbeat_has_no_cursor_and_cancel_releases_subscription(stream_rig, monkeypatch):
    from apps.api import sse

    monkeypatch.setattr(sse, "HEARTBEAT_SECONDS", 0.01)
    manager, store = stream_rig
    stream = sse.event_stream(
        "r1", last_event_id=0, owner_scope_sha256=OWNER, store=store, manager=manager
    )
    assert await asyncio.wait_for(anext(stream), 1) == {"comment": "heartbeat"}
    await manager.emit(make_event(1))
    assert (await anext(stream))["id"] == "1"
    monkeypatch.setattr(sse, "HEARTBEAT_SECONDS", 15.0)
    pending = asyncio.create_task(anext(stream))
    await asyncio.sleep(0)
    pending.cancel()
    with pytest.raises(asyncio.CancelledError):
        await pending
    assert not manager._subscribers


async def test_direct_stream_authorizes_before_reading_events(stream_rig):
    from apps.api.sse import event_stream

    manager, store = stream_rig
    store.pause_snapshot = True
    stream = event_stream(
        "r1", last_event_id=0, owner_scope_sha256="wrong", store=store, manager=manager
    )
    with pytest.raises(RunNotFound):
        await anext(stream)
    assert not store.snapshot_taken.is_set()
    assert not manager._subscribers


def test_http_disconnect_releases_live_subscription(rig: Rig):
    run_id = seed(rig, "running")
    assert rig.client.portal is not None
    rig.client.portal.call(rig.store.append_event, make_event(1, run_id=run_id))

    async def disconnect_after_first_frame():
        received_frame = asyncio.Event()
        messages = []

        async def send(message):
            messages.append(message)
            if message["type"] == "http.response.body" and b"id: 1" in message.get("body", b""):
                received_frame.set()

        async def receive():
            await received_frame.wait()
            return {"type": "http.disconnect"}

        await asyncio.wait_for(
            rig.app(
                {
                    "type": "http",
                    "asgi": {"version": "3.0"},
                    "method": "GET",
                    "scheme": "https",
                    "http_version": "1.1",
                    "path": f"/runs/{run_id}/events",
                    "query_string": b"",
                    "root_path": "",
                    "server": ("testserver", 443),
                    "client": ("127.0.0.1", 12345),
                    "headers": [
                        (b"cookie", f"dr_session={rig.client.cookies['dr_session']}".encode())
                    ],
                },
                receive,
                send,
            ),
            1,
        )
        assert messages[0]["status"] == 200
        assert received_frame.is_set()
        assert not rig.manager._subscribers

    rig.client.portal.call(disconnect_after_first_frame)
