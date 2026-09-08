"""Public wire contracts and one session-owned HTTP client, including durable SSE."""

from __future__ import annotations

from collections.abc import Iterable, Iterator
from threading import Event
from typing import Literal, Self
from urllib.parse import quote

import httpx
from pydantic import BaseModel, ConfigDict

from deepresearch.domain import ResourceUsage, RunEvent, RunStatus, StopReason

HTTPError = httpx.HTTPError
HTTPStatusError = httpx.HTTPStatusError
TransportError = httpx.TransportError

ArtifactKind = Literal["report", "evidence", "manifest"]
_TERMINAL = {"interrupted", "completed", "failed", "cancelled"}


class RunAccepted(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    thread_id: str
    status: RunStatus
    events_url: str


class RunView(BaseModel):
    model_config = ConfigDict(extra="forbid")

    run_id: str
    thread_id: str
    status: RunStatus
    stop_reason: StopReason | None
    is_partial: bool
    report_artifact_id: str | None
    evidence_graph_artifact_id: str | None
    manifest_artifact_id: str | None
    final_usage: ResourceUsage | None
    error_code: str | None


class StreamReconnectExhausted(RuntimeError):
    def __init__(self, run_id: str, last_event_id: int) -> None:
        self.run_id = run_id
        self.last_event_id = last_event_id
        super().__init__(f"Event connection exhausted at sequence {last_event_id}")


def _sse_data(lines: Iterable[str]) -> Iterator[str]:
    """Dispatch only blank-line-terminated frames; discard an incomplete EOF frame."""
    data: list[str] = []
    for line in lines:
        if not line:
            if data:
                yield "\n".join(data)
            data = []
        elif not line.startswith(":"):
            field, separator, value = line.partition(":")
            if field == "data":
                data.append(value.removeprefix(" ") if separator else "")


class ResearchApiClient:
    def __init__(self, base_url: str, *, transport: httpx.BaseTransport | None = None) -> None:
        self.client = httpx.Client(base_url=base_url, timeout=30, transport=transport)
        self._stream_stop = Event()
        self._stream_response: httpx.Response | None = None

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def close(self) -> None:
        self.stop_events()
        self.client.close()

    def stop_events(self) -> None:
        self._stream_stop.set()
        response = self._stream_response
        if response is not None:
            response.close()

    def _wait_for_reconnect(self, seconds: float) -> bool:
        return self._stream_stop.wait(seconds)

    @staticmethod
    def _run_path(run_id: str) -> str:
        if not run_id or run_id in {".", ".."}:
            raise ValueError("invalid run ID")
        return f"/runs/{quote(run_id, safe='')}"

    def create_run(self, payload: dict[str, object], idempotency_key: str) -> RunAccepted:
        response = self.client.post(
            "/runs", json=payload, headers={"Idempotency-Key": idempotency_key}
        )
        response.raise_for_status()
        return RunAccepted.model_validate_json(response.content)

    def get_run(self, run_id: str) -> RunView:
        response = self.client.get(self._run_path(run_id), timeout=5)
        response.raise_for_status()
        return RunView.model_validate_json(response.content)

    def resume(self, run_id: str) -> RunView:
        response = self.client.post(f"{self._run_path(run_id)}/resume")
        response.raise_for_status()
        return RunView.model_validate_json(response.content)

    def cancel(self, run_id: str) -> RunView:
        response = self.client.post(f"{self._run_path(run_id)}/cancel")
        response.raise_for_status()
        return RunView.model_validate_json(response.content)

    def events(self, run_id: str, last_event_id: int = 0) -> Iterator[RunEvent]:
        if type(last_event_id) is not int or last_event_id < 0:
            raise ValueError("last_event_id must be a nonnegative integer")
        if self.client.is_closed:
            return
        self._stream_stop.clear()
        cursor = last_event_id
        for attempt in range(6):
            if self._stream_stop.is_set() or self.client.is_closed:
                return
            try:
                with self.client.stream(
                    "GET",
                    f"{self._run_path(run_id)}/events",
                    headers={"Last-Event-ID": str(cursor), "Accept": "text/event-stream"},
                ) as response:
                    self._stream_response = response
                    if self._stream_stop.is_set():
                        return
                    response.raise_for_status()
                    for data in _sse_data(response.iter_lines()):
                        if self._stream_stop.is_set():
                            return
                        event = RunEvent.model_validate_json(data)
                        if event.run_id != run_id:
                            raise ValueError("event belongs to another run")
                        if event.seq <= cursor:
                            continue
                        cursor = event.seq
                        yield event
                # Node status is not service status, and an interrupted run may
                # already have resumed. Drain durable frames before checking EOF.
                if self._stream_stop.is_set() or self.get_run(run_id).status in _TERMINAL:
                    return
            except httpx.TransportError:
                pass
            finally:
                self._stream_response = None
            if attempt < 5 and self._wait_for_reconnect(min(2 ** (attempt + 1), 8)):
                return
        if self._stream_stop.is_set():
            return
        raise StreamReconnectExhausted(run_id, cursor)

    def download_artifact(self, run_id: str, kind: ArtifactKind) -> bytes:
        if kind not in {"report", "evidence", "manifest"}:
            raise ValueError("unsupported artifact kind")
        response = self.client.get(f"{self._run_path(run_id)}/artifacts/{kind}")
        response.raise_for_status()
        return response.content
