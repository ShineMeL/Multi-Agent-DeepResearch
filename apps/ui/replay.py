"""Replay request construction using only public research contracts."""

from dataclasses import dataclass, field
from queue import Empty, Queue
from threading import Event, Thread
from time import monotonic
from typing import Literal
from uuid import uuid4

from apps.ui.api_client import HTTPError, HTTPStatusError, ResearchApiClient, RunAccepted, RunView
from deepresearch.domain import FreshnessRequirement, ResearchRequest, RunEvent


def replay_payload(
    question: str,
    *,
    report_language: str = "en",
    source_languages: tuple[str, ...] = ("en",),
    # The shipped baseline fixture was recorded with the medium planner
    # budget; replay request identity includes this field.
    budget_preset: Literal["low", "medium"] = "medium",
    provider_profile_id: str = "replay-default",
    seed: int = 0,
    workflow_id: Literal["baseline-v1", "research-v1"] = "baseline-v1",
) -> dict[str, object]:
    if not question.strip() or not provider_profile_id.strip():
        raise ValueError("Question and replay profile are required")
    if workflow_id not in {"baseline-v1", "research-v1"}:
        raise ValueError("Unsupported replay workflow")
    request = ResearchRequest(
        question=question.strip(),
        # Keep the Showcase request identity aligned with the shipped baseline
        # recording.  Replay is exact by design; even this output-shape field is
        # part of the planner request hash.
        output_requirements={"answer_shape": "markdown"},
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
        "workflow_id": workflow_id,
        "planner_id": "P1",
        "ranker_id": "R1",
        "seed": seed,
    }


def research_replay_payload(
    question: str,
    *,
    report_language: str = "en",
    source_languages: tuple[str, ...] = ("en",),
    budget_preset: Literal["low", "medium"] = "medium",
    provider_profile_id: str = "replay-default",
    seed: int = 0,
) -> dict[str, object]:
    """Build the supported deterministic research-v1 showcase request.

    The production research graph currently exposes the audited P1/R1
    composition only.  Keeping this helper separate from ``replay_payload``
    preserves the baseline fixture contract used by existing clients/tests.
    """
    return replay_payload(
        question,
        report_language=report_language,
        source_languages=source_languages,
        budget_preset=budget_preset,
        provider_profile_id=provider_profile_id,
        seed=seed,
        workflow_id="research-v1",
    )


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
    view: RunView | None = None
    poll_error: Exception | None = None
    poll_finished: Event = field(default_factory=Event)
    closed: bool = False
    _queue: Queue[RunEvent | Exception] = field(default_factory=Queue[RunEvent | Exception])
    _worker: Thread | None = None
    _poll_queue: Queue[RunView | Exception] = field(default_factory=Queue[RunView | Exception])
    _poll_worker: Thread | None = None
    _poll_failures: int = 0
    _next_poll: float = 0
    _force_refresh: bool = False
    _paused: bool = False

    @property
    def _poll_allowed(self) -> bool:
        return (
            not self.closed
            and not self._paused
            and self.run_id is not None
            and self.stream_error is None
            and self._poll_failures < 3
            and (
                self._force_refresh
                or self.view is None
                or self.view.status in {"queued", "running"}
            )
        )

    @property
    def automatic_refresh(self) -> bool:
        return not self.closed and (
            self.watching
            or self._poll_allowed
            or (self._poll_worker is not None and self._poll_worker.is_alive())
        )

    def stop(self) -> None:
        """Stop observation without cancelling the service-owned run."""
        self._poll_failures = 3
        self._paused = True
        self.api.stop_events()

    def close(self) -> None:
        if not self.closed:
            self.closed = True
            self.stop()
            self.api.close()

    def retry_status(self) -> None:
        self._paused = False
        self._poll_failures = 0
        self._next_poll = 0
        self._force_refresh = True
        self.poll_error = None
        self.stream_error = None

    def update_view(self, view: RunView, *, now: float | None = None) -> None:
        self.view = view
        self.poll_error = None
        self._poll_failures = 0
        self._force_refresh = False
        self._next_poll = (monotonic() if now is None else now) + 5

    def refresh(self, *, now: float | None = None) -> None:
        """Schedule at most one nonblocking status read, respecting bounded backoff."""
        current = monotonic() if now is None else now
        self.drain(now=current)
        if (
            not self._poll_allowed
            or current < self._next_poll
            or (self._poll_worker is not None and self._poll_worker.is_alive())
        ):
            return
        run_id = self.run_id
        if run_id is None:
            return
        self.poll_finished.clear()
        api, queue, finished = self.api, self._poll_queue, self.poll_finished

        def read_status() -> None:
            try:
                queue.put(api.get_run(run_id))
            except (HTTPError, ValueError, RuntimeError) as error:
                queue.put(error)
            finally:
                finished.set()

        self._poll_worker = Thread(target=read_status, name="showcase-status", daemon=True)
        self._poll_worker.start()

    @property
    def watching(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def submit(self, payload: dict[str, object]) -> RunAccepted:
        if self.watching:
            raise ValueError("Wait for the current event reader before starting another run")
        if self._poll_worker is not None and self._poll_worker.is_alive():
            raise ValueError("Wait for the current status request before starting another run")
        if self.pending is None:
            self.pending = (payload, str(uuid4()))
        try:
            accepted = self.api.create_run(*self.pending)
        except HTTPStatusError as error:
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
        self.view = None
        self.retry_status()
        self._queue = Queue()
        self._poll_queue = Queue()
        return accepted

    def watch(self) -> None:
        if self.closed or self.run_id is None or self.watching:
            return
        self.drain()
        self.stream_error = None
        self.finished.clear()
        run_id, cursor = self.run_id, self.cursor
        api, queue, finished = self.api, self._queue, self.finished

        def read_events() -> None:
            try:
                for event in api.events(run_id, cursor):
                    queue.put(event)
            except (HTTPError, ValueError, RuntimeError) as error:
                queue.put(error)
            finally:
                finished.set()

        self._worker = Thread(target=read_events, name="showcase-events", daemon=True)
        self._worker.start()

    def drain(self, *, now: float | None = None) -> None:
        current = monotonic() if now is None else now
        while True:
            try:
                result = self._poll_queue.get_nowait()
            except Empty:
                break
            if isinstance(result, Exception):
                self.poll_error = result
                self._poll_failures += 1
                if isinstance(result, HTTPStatusError) and result.response.status_code in {
                    401,
                    403,
                    404,
                }:
                    self._poll_failures = 3
                self._next_poll = current + 2**self._poll_failures
                if self._poll_failures >= 3:
                    self.stop()
            else:
                self.update_view(result, now=current)
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
