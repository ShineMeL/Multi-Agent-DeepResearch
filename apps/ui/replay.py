"""Replay request construction using only public research contracts."""

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from typing import Literal
from uuid import uuid4

import httpx

from apps.ui.api_client import ResearchApiClient, RunAccepted, StreamReconnectExhausted
from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunEvent


def replay_payload(
    question: str,
    *,
    report_language: str = "en",
    source_languages: tuple[str, ...] = ("en",),
    budget_preset: Literal["low", "medium"] = "low",
    provider_profile_id: str = "replay-default",
    seed: int = 0,
) -> dict[str, object]:
    if not question.strip() or not provider_profile_id.strip():
        raise ValueError("Question and replay profile are required")
    request = ResearchRequest(
        question=question.strip(),
        output_requirements={},
        report_language=report_language,
        source_languages=source_languages,
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id=provider_profile_id.strip(),
        run_purpose="demo",
        budget_preset=budget_preset,
    )
    return {
        "request": request.model_dump(mode="json"),
        "workflow_id": "baseline-v1",
        "planner_id": "P1",
        "ranker_id": "R1",
        "seed": seed,
    }


@dataclass
class ShowcaseSession:
    """One browser session owns its cookie jar, cursor, and background event reader."""

    api: ResearchApiClient
    run_id: str | None = None
    events: list[RunEvent] = field(default_factory=list[RunEvent])
    cursor: int = 0
    stream_error: Exception | None = None
    finished: Event = field(default_factory=Event)
    downloads: dict[tuple[str, str], bytes] = field(default_factory=dict[tuple[str, str], bytes])
    pending: tuple[dict[str, object], str] | None = None
    _queue: Queue[RunEvent | Exception] = field(default_factory=Queue[RunEvent | Exception])
    _worker: Thread | None = None

    @property
    def watching(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def submit(self, payload: dict[str, object]) -> RunAccepted:
        if self.watching:
            raise ValueError("Wait for the current event reader before starting another run")
        if self.pending is None:
            self.pending = (payload, str(uuid4()))
        try:
            accepted = self.api.create_run(*self.pending)
        except httpx.HTTPStatusError as error:
            # An explicit client rejection can be edited. An ambiguous timeout
            # or server failure retains the exact key and payload for retry.
            if error.response.status_code < 500:
                self.pending = None
            raise
        self.pending = None
        self.run_id = accepted.run_id
        self.events.clear()
        self.cursor = 0
        self.downloads.clear()
        self.stream_error = None
        self._queue = Queue()
        return accepted

    def watch(self) -> None:
        if self.run_id is None or self.watching:
            return
        self.drain()
        self.stream_error = None
        self.finished.clear()
        run_id, cursor = self.run_id, self.cursor

        def read_events() -> None:
            try:
                for event in self.api.events(run_id, cursor):
                    self._queue.put(event)
            except (httpx.HTTPError, ValueError, StreamReconnectExhausted) as error:
                self._queue.put(error)
            finally:
                self.finished.set()

        self._worker = Thread(target=read_events, name="showcase-events", daemon=True)
        self._worker.start()

    def drain(self) -> None:
        while True:
            try:
                item = self._queue.get_nowait()
            except Empty:
                return
            if isinstance(item, Exception):
                self.stream_error = item
            elif item.seq > self.cursor:
                self.events.append(item)
                self.cursor = item.seq
