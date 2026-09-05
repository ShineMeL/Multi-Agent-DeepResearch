from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import re
import time
import uuid
from collections import Counter
from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Self, cast

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, StateGraph
from pydantic import BaseModel

from deepresearch.domain import (
    FreshnessRequirement,
    HtmlLocator,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
    RunConfig,
    RunEvent,
)
from deepresearch.evidence import SimilarityRanker
from deepresearch.planning import FixedPlanner
from deepresearch.providers import (
    ModelMessage,
    ModelRequest,
    ModelResult,
    ParsedBlock,
    ParsedDocument,
    ProviderError,
    ProviderUsageResult,
    RawDocument,
    SearchHit,
    StructuredModelResult,
)
from deepresearch.reporting import MarkdownReportWriter
from deepresearch.runtime import (
    BudgetAccountant,
    BudgetExceeded,
    CancellationToken,
    CheckpointRef,
    OperationCancelled,
    ResourceEstimate,
)
from deepresearch.runtime.checkpoints import (
    checkpoint_config,
    checkpoint_ref_from_tuple,
    checkpoint_serializer,
    open_sqlite_checkpointer,
)
from deepresearch.runtime.manifest import (
    CostCalculator,
    NodeExecutionRecord,
    PricingSnapshot,
    ProviderCallRecord,
    ProviderProfileRecord,
    RunManifest,
)
from deepresearch.storage import (
    ArtifactIntegrityError,
    CacheEntry,
    CacheIntegrityError,
    EmbedCacheKey,
    FetchCacheKey,
    FileCache,
    LocalArtifactStore,
    LocalEvidenceStore,
    ModelCacheKey,
    ParseCacheKey,
    SearchCacheKey,
    cache_key_sha256,
)
from deepresearch.workflow import baseline_graph as baseline_graph_module
from deepresearch.workflow.baseline_graph import (
    BaselineNodeHandlers,
    BaselineRuntimeContext,
    UsageIntegrityError,
    build_baseline_graph,
)
from deepresearch.workflow.runner import BaselineRuntimeHooks, LangGraphResearchRunner
from deepresearch.workflow.state import BaselineState


def request() -> ResearchRequest:
    return ResearchRequest(
        question="What does the offline baseline prove?",
        output_requirements={"answer_shape": "brief"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="showcase",
        provider_profile_id="offline",
        run_purpose="test",
        budget_preset="low",
    )


def config(**updates: object) -> RunConfig:
    values: dict[str, object] = {
        "request": request(),
        "workflow_id": "baseline-v1",
        "planner_id": "P1",
        "ranker_id": "R1",
        "budget": RunBudget.preset("low"),
        "prompt_versions": {"planner": "p1", "writer": "w1"},
        "seed": 0,
    }
    values.update(updates)
    return RunConfig.model_validate(values)


def _canonical_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def test_run_header_binds_request_config_and_profile_identity(tmp_path: Path) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    composition = cast("Any", handlers)._audit_composition
    current = config()
    header_id = composition.create_run_header(
        run_id="run-header-binding",
        thread_id="thread-header-binding",
        config=current,
        started_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers.artifact_store,
        header_id,
    )

    assert receipt is not None
    assert receipt.payload["request_sha256"] == _canonical_sha256(
        current.request.model_dump(mode="json")
    )
    assert receipt.payload["config_sha256"] == _canonical_sha256(
        current.model_dump(mode="json")
    )
    assert receipt.payload["provider_profile_id"] == "offline"
    assert receipt.payload["execution_mode"] == "replay"


@pytest.mark.parametrize("binding", ["config", "profile", "execution-mode"])
def test_load_run_header_rejects_current_identity_rebinding(
    tmp_path: Path,
    binding: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    composition = cast("Any", handlers)._audit_composition
    original = config()
    header_id = composition.create_run_header(
        run_id="run-header-rebind",
        thread_id="thread-header-rebind",
        config=original,
        started_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    if binding == "config":
        rebound = original.model_copy(
            update={"prompt_versions": {"planner": "changed", "writer": "w1"}}
        )
    elif binding == "profile":
        rebound = original.model_copy(
            update={
                "request": original.request.model_copy(
                    update={"provider_profile_id": "changed"}
                )
            }
        )
    else:
        rebound = original.model_copy(
            update={
                "request": original.request.model_copy(
                    update={"execution_mode": "live"}
                )
            }
        )

    with pytest.raises(ArtifactIntegrityError, match="composition|identity"):
        composition.load_run_header(
            (header_id,),
            run_id="run-header-rebind",
            thread_id="thread-header-rebind",
            config=rebound,
        )


@pytest.mark.parametrize("kind", ["parsed-artifact", "evidence-hash"])
def test_receipt_topology_rejects_orphan_typed_child(kind: str) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    record: object
    if kind == "parsed-artifact":
        record = {
            "source_id": "S-orphan",
            "raw_content_hash": "1" * 64,
            "parsed_content_hash": "2" * 64,
            "parser_id": "parser",
            "parser_version": "v1",
            "normalization_version": "v1",
            "artifact_id": "sha256:" + "3" * 64,
        }
    else:
        record = {
            "evidence_id": "E-orphan",
            "source_id": "S-orphan",
            "locator_sha256": "1" * 64,
            "excerpt_hash": "2" * 64,
            "artifact_id": "sha256:" + "3" * 64,
        }
    orphan = cast("Any", baseline_graph_module)._AuditReceiptEnvelope.model_validate(
        {
            "schema_version": "baseline-audit-receipt-v1",
            "kind": kind,
            "run_id": "run-orphan",
            "thread_id": "thread-orphan",
            "receipt_key": f"orphan:{kind}",
            "payload": {"record": record},
        }
    )

    with pytest.raises(ArtifactIntegrityError, match="ownership|orphan|incomplete"):
        cast("Any", baseline_graph_module)._validate_complete_receipt_ownership(
            indexed_receipts={"sha256:" + "4" * 64: orphan},
            consumed_node_receipts=set(),
            consumed_child_receipts=set(),
        )


class FakeSaver:
    def __init__(self) -> None:
        self.saved: object | None = None

    async def aget_tuple(self, value: object) -> object | None:
        return self.saved


class FakeGraph:
    def __init__(self) -> None:
        self.checkpointer = FakeSaver()
        self.calls: list[tuple[object, object, object]] = []

    async def ainvoke(
        self,
        value: object,
        graph_config: object,
        *,
        context: object,
    ) -> dict[str, object]:
        self.calls.append((value, graph_config, context))
        current = cast("dict[str, object]", value)
        return {
            **current,
            "stop_reason": "SUFFICIENT",
            "report_artifact_id": "sha256:" + "1" * 64,
            "evidence_graph_artifact_id": "sha256:" + "2" * 64,
            "manifest_artifact_id": "sha256:" + "3" * 64,
        }


class MemoryEventSink:
    def __init__(self) -> None:
        self.events: dict[tuple[str, int], RunEvent] = {}
        self.calls: list[RunEvent] = []

    async def __call__(self, event: RunEvent) -> None:
        self.calls.append(event)
        key = (event.run_id, event.seq)
        previous = self.events.get(key)
        if previous is not None and previous != event:
            raise ValueError("conflicting durable event")
        self.events[key] = event

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        return self.events.get((run_id, seq))


class ConcurrentPlanBarrierSink(MemoryEventSink):
    def __init__(self) -> None:
        super().__init__()
        self.plan_runs: set[str] = set()
        self.both_plans = asyncio.Event()

    async def __call__(self, event: RunEvent) -> None:
        await super().__call__(event)
        if event.node != "Plan":
            return
        self.plan_runs.add(event.run_id)
        if len(self.plan_runs) == 2:
            self.both_plans.set()
        await asyncio.wait_for(self.both_plans.wait(), timeout=5.0)


_emit = MemoryEventSink()


class CrashAfterDurableEventSink(MemoryEventSink):
    def __init__(self, *, crash_seq: int) -> None:
        super().__init__()
        self.crash_seq = crash_seq
        self.crashed = False

    async def __call__(self, event: RunEvent) -> None:
        await super().__call__(event)
        if event.seq == self.crash_seq and not self.crashed:
            self.crashed = True
            raise MemoryError("crash after durable event publication")


class OneShotReadbackFailureSink(MemoryEventSink):
    def __init__(self, *, target_node: str) -> None:
        super().__init__()
        self.target_node = target_node
        self.target_key: tuple[str, int] | None = None
        self.failed = False

    async def __call__(self, event: RunEvent) -> None:
        await super().__call__(event)
        if event.node == self.target_node:
            self.target_key = (event.run_id, event.seq)

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        if self.target_key == (run_id, seq) and not self.failed:
            self.failed = True
            raise OSError("one-shot durable event readback failure")
        return await super().get_event(run_id=run_id, seq=seq)


class OneShotPreWriteFailureSink(MemoryEventSink):
    def __init__(
        self,
        *,
        target_node: str,
        target_occurrence: int = 1,
        invocations: Counter[str] | None = None,
        target_success_only: bool = True,
    ) -> None:
        super().__init__()
        self.target_node = target_node
        self.target_occurrence = target_occurrence
        self.invocations = invocations
        self.target_success_only = target_success_only
        self.attempts: list[RunEvent] = []
        self.target_key: tuple[str, int] | None = None
        self.authoritative_reads: list[RunEvent | None] = []
        self.provider_calls_at_failure: int | None = None
        self.successful_target_events = 0
        self.failed = False

    async def __call__(self, event: RunEvent) -> None:
        self.attempts.append(event)
        matches_target = event.node == self.target_node and (
            not self.target_success_only or event.error_code is None
        )
        if matches_target:
            self.successful_target_events += 1
        if (
            matches_target
            and self.successful_target_events == self.target_occurrence
            and not self.failed
        ):
            self.target_key = (event.run_id, event.seq)
            self.provider_calls_at_failure = (
                None if self.invocations is None else sum(self.invocations.values())
            )
            self.failed = True
            raise OSError("one-shot durable event pre-write failure")
        await super().__call__(event)

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        event = await super().get_event(run_id=run_id, seq=seq)
        if self.failed and self.target_key == (run_id, seq):
            self.authoritative_reads.append(event)
        return event


class ClassifiedPreWriteFailureSink(OneShotPreWriteFailureSink):
    def __init__(
        self,
        *,
        target_node: str,
        mode: str,
        primary: BaseException | None = None,
    ) -> None:
        super().__init__(target_node=target_node)
        self.mode = mode
        self.primary = primary
        self.boundary_reads = 0

    async def __call__(self, event: RunEvent) -> None:
        if (
            event.node == self.target_node
            and event.error_code is None
            and not self.failed
            and self.mode == "initial-call-hard"
        ):
            self.attempts.append(event)
            assert self.primary is not None
            raise self.primary
        if (
            self.failed
            and self.target_key == (event.run_id, event.seq)
            and event.error_code == "DATA_CORRUPTION"
        ):
            if self.mode == "replacement-call-hard":
                self.attempts.append(event)
                assert self.primary is not None
                raise self.primary
            if self.mode == "replacement-call-ordinary-stored":
                self.attempts.append(event)
                await MemoryEventSink.__call__(self, event)
                raise OSError("replacement failed after durable write")
            if self.mode == "replacement-call-ordinary-absent":
                self.attempts.append(event)
                raise OSError("replacement failed before durable write")
        await super().__call__(event)

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        if self.failed and self.target_key == (run_id, seq):
            self.boundary_reads += 1
            if self.boundary_reads == 1:
                if self.mode == "conflict":
                    attempted = self.attempts[-1]
                    return attempted.model_copy(
                        update={"timestamp": attempted.timestamp + timedelta(seconds=1)}
                    )
                if self.mode == "constructed-corruption":
                    corrupt_usage = cast(
                        "ResourceUsage",
                        BaseModel.model_copy(
                            self.attempts[-1].usage_delta,
                            update={"wall_seconds": 0},
                        ),
                    )
                    return cast(
                        "RunEvent",
                        BaseModel.model_copy(
                            self.attempts[-1],
                            update={"usage_delta": corrupt_usage},
                        ),
                    )
                if self.mode == "unreadable":
                    raise OSError("authoritative durable read failed")
                if self.mode == "authoritative-read-hard":
                    assert self.primary is not None
                    raise self.primary
            if self.boundary_reads == 2:
                if self.mode == "replacement-readback-hard":
                    assert self.primary is not None
                    raise self.primary
                if self.mode == "replacement-readback-ordinary":
                    raise OSError("replacement immediate readback failed")
        return await super().get_event(run_id=run_id, seq=seq)


class FailedEventPreWriteFailureSink(MemoryEventSink):
    def __init__(
        self,
        *,
        target_node: str,
        error_code: str,
        store_before_failure: bool,
    ) -> None:
        super().__init__()
        self.target_node = target_node
        self.error_code = error_code
        self.store_before_failure = store_before_failure
        self.attempts: list[RunEvent] = []
        self.target_key: tuple[str, int] | None = None
        self.authoritative_reads: list[RunEvent | None] = []
        self.failed = False

    async def __call__(self, event: RunEvent) -> None:
        if event.node == self.target_node:
            self.attempts.append(event)
        if (
            event.node == self.target_node
            and event.error_code == self.error_code
            and not self.failed
        ):
            self.target_key = (event.run_id, event.seq)
            self.failed = True
            if self.store_before_failure:
                await super().__call__(event)
            raise OSError("failed event did not reach durable storage")
        await super().__call__(event)

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        event = await super().get_event(run_id=run_id, seq=seq)
        if self.failed and self.target_key == (run_id, seq):
            self.authoritative_reads.append(event)
        return event


class ImmediateConstructedReadbackSink(MemoryEventSink):
    def __init__(self, *, target_node: str, wall_seconds: object) -> None:
        super().__init__()
        self.target_node = target_node
        self.wall_seconds = wall_seconds
        self.target_key: tuple[str, int] | None = None
        self.target_event: RunEvent | None = None
        self.boundary_reads = 0

    async def __call__(self, event: RunEvent) -> None:
        await super().__call__(event)
        if event.node == self.target_node and event.error_code is None:
            self.target_key = (event.run_id, event.seq)
            self.target_event = event

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        if self.target_key == (run_id, seq):
            self.boundary_reads += 1
            if self.boundary_reads == 1:
                assert self.target_event is not None
                corrupt_usage = cast(
                    "ResourceUsage",
                    BaseModel.model_copy(
                        self.target_event.usage_delta,
                        update={"wall_seconds": self.wall_seconds},
                    ),
                )
                return cast(
                    "RunEvent",
                    BaseModel.model_copy(
                        self.target_event,
                        update={"usage_delta": corrupt_usage},
                    ),
                )
        return await super().get_event(run_id=run_id, seq=seq)


class FilesystemEventSink:
    def __init__(self, root: Path, *, crash_after_node: str | None = None) -> None:
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.crash_after_node = crash_after_node
        self.crashed = False

    @staticmethod
    def _bytes(event: RunEvent) -> bytes:
        return json.dumps(
            event.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()

    def _path(self, *, run_id: str, seq: int) -> Path:
        run_hash = hashlib.sha256(run_id.encode()).hexdigest()
        return self.root / run_hash / f"{seq:020d}.json"

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
        path = self._path(run_id=run_id, seq=seq)
        try:
            raw = path.read_bytes()
        except FileNotFoundError:
            return None
        event = RunEvent.model_validate_json(raw, strict=True)
        if (
            event.run_id != run_id
            or event.seq != seq
            or self._bytes(event) != raw
        ):
            raise ValueError("durable event is corrupt")
        return event

    async def __call__(self, event: RunEvent) -> None:
        payload = self._bytes(event)
        path = self._path(run_id=event.run_id, seq=event.seq)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with temporary.open("xb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            try:
                os.link(temporary, path)
            except FileExistsError:
                if path.read_bytes() != payload:
                    raise ValueError("conflicting durable event") from None
        finally:
            temporary.unlink(missing_ok=True)
        verified = await self.get_event(run_id=event.run_id, seq=event.seq)
        if verified != event:
            raise ValueError("durable event readback mismatch")
        if self.crash_after_node == event.node and not self.crashed:
            self.crashed = True
            raise MemoryError("crash after filesystem event publication")


async def test_runner_has_frozen_constructor_and_builds_isolated_new_run_context() -> None:
    graph = FakeGraph()
    ticks = iter((10.0, 12.5))
    hooks = BaselineRuntimeHooks(
        monotonic=lambda: next(ticks),
        utc_now=lambda: datetime(2026, 9, 1, tzinfo=UTC),
        new_id=lambda prefix: f"{prefix}-fixed",
    )
    runner = LangGraphResearchRunner(
        baseline_graph=cast("Any", graph),
        runtime_hooks=hooks,
    )

    result = await runner.run(
        run_id="run-1",
        thread_id="thread-1",
        config=config(),
        checkpoint=None,
        emit=_emit,
        cancellation_token=CancellationToken(),
    )

    assert inspect.signature(LangGraphResearchRunner.__init__).parameters.keys() == {
        "self",
        "baseline_graph",
        "runtime_hooks",
        "research_graph",
    }
    assert runner._baseline_graph is graph
    assert result.status == "completed"
    assert result.stop_reason == "SUFFICIENT"
    assert result.is_partial is False
    assert result.final_usage == ResourceUsage.zero(cost_known=True)
    initial, graph_config, context = graph.calls[0]
    assert cast("dict[str, object]", initial)["run_id"] == "run-1"
    assert graph_config == {"configurable": {"thread_id": "thread-1", "checkpoint_ns": ""}}
    runtime_context = cast("BaselineRuntimeContext", context)
    assert runtime_context.deadline == 190.0
    assert runtime_context.run_started_monotonic == 10.0


async def test_runner_rejects_invalid_baseline_configuration_before_graph_work() -> None:
    graph = FakeGraph()
    runner = LangGraphResearchRunner(baseline_graph=cast("Any", graph))

    result = await runner.run(
        run_id="run-1",
        thread_id="thread-1",
        config=config(planner_id="P0"),
        checkpoint=None,
        emit=_emit,
        cancellation_token=CancellationToken(),
    )

    assert result.status == "failed"
    assert result.error_code == "INVALID_WORKFLOW_CONFIG"
    assert graph.calls == []


async def test_runner_rejects_non_durable_event_sink_before_graph_work() -> None:
    graph = FakeGraph()
    runner = LangGraphResearchRunner(baseline_graph=cast("Any", graph))

    async def callback(_event: RunEvent) -> None:
        return None

    result = await runner.run(
        run_id="run-1",
        thread_id="thread-1",
        config=config(),
        checkpoint=None,
        emit=callback,
        cancellation_token=CancellationToken(),
    )

    assert result.status == "failed"
    assert result.error_code == "INVALID_EVENT_SINK"
    assert graph.calls == []


async def test_hard_event_readback_failure_preserves_primary_identity(
    tmp_path: Path,
) -> None:
    handlers = _owned_handlers(tmp_path)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    runner = LangGraphResearchRunner(baseline_graph=graph)
    primary = MemoryError("hard durable event readback failure")

    class HardReadbackSink(MemoryEventSink):
        def __init__(self) -> None:
            super().__init__()
            self.reads = 0

        async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
            self.reads += 1
            if self.reads == 2:
                raise primary
            return await super().get_event(run_id=run_id, seq=seq)

    sink = HardReadbackSink()
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    current_config = config(
        budget=unpriced,
        prompt_versions=dict(handlers.prompt_versions),
        ranker_weights_version=handlers.ranker_weights_version,
    )

    with pytest.raises(MemoryError) as caught:
        await runner.run(
            run_id="run-hard-event-readback",
            thread_id="thread-hard-event-readback",
            config=current_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=CancellationToken(),
        )

    assert caught.value is primary


async def test_hard_resume_event_lookup_preserves_primary_identity() -> None:
    from deepresearch.workflow import runner as runner_module

    current_config = config()
    accountant = BudgetAccountant(current_config.budget, run_scope="run-hard-resume")
    state = cast("Any", runner_module)._initial_state(
        run_id="run-hard-resume",
        thread_id="thread-hard-resume",
        config=current_config,
        accountant=accountant,
    )
    created = datetime(2026, 9, 1, tzinfo=UTC)
    saved = SimpleNamespace(
        config={
            "configurable": {
                "thread_id": "thread-hard-resume",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-hard-resume",
            }
        },
        checkpoint={
            "id": "cp-hard-resume",
            "ts": created.isoformat(),
            "channel_values": state,
        },
    )
    graph = FakeGraph()
    graph.checkpointer.saved = saved
    graph._baseline_audit_composition = SimpleNamespace(load_run_header=lambda *_args, **_kwargs: {"started_at": created.isoformat()})
    runner = LangGraphResearchRunner(baseline_graph=cast("Any", graph))
    primary = MemoryError("hard resume event lookup failure")

    class HardLookupSink(MemoryEventSink):
        async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
            del run_id, seq
            raise primary

    with pytest.raises(MemoryError) as caught:
        await runner.run(
            run_id="run-hard-resume",
            thread_id="thread-hard-resume",
            config=current_config,
            checkpoint=checkpoint_ref_from_tuple(saved),
            emit=HardLookupSink(),
            cancellation_token=CancellationToken(),
        )

    assert caught.value is primary
    assert graph.calls == []


async def test_sqlite_resume_reports_cumulative_elapsed_wall_time(tmp_path: Path) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module
    from deepresearch.workflow import runner as runner_module

    current_config = config()
    accountant = BudgetAccountant(current_config.budget, run_scope="run-wall-resume")
    initial = cast("Any", runner_module)._initial_state(
        run_id="run-wall-resume",
        thread_id="thread-wall-resume",
        config=current_config,
        accountant=accountant,
    )
    initial["elapsed_wall_seconds"] = 7.0
    path = (tmp_path / "wall-resume.sqlite3").resolve()
    async with open_sqlite_checkpointer(path) as saver:
        builder = StateGraph(cast("Any", baseline_graph_module.BaselineState))

        async def freeze(_state: object) -> dict[str, object]:
            return {}

        builder.add_node("freeze", freeze)
        builder.set_entry_point("freeze")
        builder.add_edge("freeze", END)
        graph = builder.compile(checkpointer=saver)
        await graph.ainvoke(
            initial,
            {"configurable": {"thread_id": "thread-wall-resume"}},
        )
        saved = await saver.aget_tuple(
            {
                "configurable": {
                    "thread_id": "thread-wall-resume",
                    "checkpoint_ns": "",
                }
            }
        )
        assert saved is not None
        checkpoint = checkpoint_ref_from_tuple(saved)

    deadlines: list[float] = []
    async with open_sqlite_checkpointer(path) as reopened:

        class ResumedGraph:
            checkpointer = reopened

            async def ainvoke(
                self,
                value: object,
                graph_config: object,
                *,
                context: BaselineRuntimeContext,
            ) -> dict[str, object]:
                assert value is None
                del graph_config
                deadlines.append(context.deadline)
                return {
                    **initial,
                    "elapsed_wall_seconds": 12.0,
                    "stop_reason": "SUFFICIENT",
                    "report_artifact_id": "sha256:" + "1" * 64,
                    "evidence_graph_artifact_id": "sha256:" + "2" * 64,
                    "manifest_artifact_id": "sha256:" + "3" * 64,
                }

        runner = LangGraphResearchRunner(
            baseline_graph=cast("Any", ResumedGraph()),
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=lambda: 100.0,
                utc_now=lambda: datetime(2026, 9, 1, tzinfo=UTC),
            ),
        )
        result = await runner.run(
            run_id="run-wall-resume",
            thread_id="thread-wall-resume",
            config=current_config,
            checkpoint=checkpoint,
            emit=MemoryEventSink(),
            cancellation_token=CancellationToken(),
        )

    assert result.status == "completed", result.error_code
    assert deadlines == [
        100.0 + float(current_config.budget.max_wall_time_seconds) - 7.0
    ]
    assert result.final_usage.wall_seconds == 12.0


async def test_runner_requires_exact_checkpoint_identity_on_resume() -> None:
    graph = FakeGraph()
    created = datetime(2026, 9, 1, tzinfo=UTC)
    checkpoint = CheckpointRef(
        checkpoint_id="cp-1",
        thread_id="other-thread",
        created_at=created,
    )
    runner = LangGraphResearchRunner(baseline_graph=cast("Any", graph))

    result = await runner.run(
        run_id="run-1",
        thread_id="thread-1",
        config=config(),
        checkpoint=checkpoint,
        emit=_emit,
        cancellation_token=CancellationToken(),
    )

    assert result.status == "failed"
    assert result.error_code == "CHECKPOINT_MISMATCH"
    assert graph.calls == []


async def test_runner_maps_partial_stop_without_report_to_report_missing() -> None:
    graph = FakeGraph()

    async def missing_report(
        value: object,
        graph_config: object,
        *,
        context: object,
    ) -> dict[str, object]:
        del graph_config, context
        return {
            **cast("dict[str, object]", value),
            "stop_reason": "PLATEAU",
            "is_partial": True,
        }

    graph.ainvoke = missing_report  # type: ignore[method-assign]
    runner = LangGraphResearchRunner(baseline_graph=cast("Any", graph))

    result = await runner.run(
        run_id="run-1",
        thread_id="thread-1",
        config=config(),
        checkpoint=None,
        emit=_emit,
        cancellation_token=CancellationToken(),
    )

    assert result.status == "failed"
    assert result.stop_reason == "PLATEAU"
    assert result.error_code == "REPORT_MISSING"


def test_fake_checkpoint_tuple_shape_documents_exact_resume_contract() -> None:
    created = datetime(2026, 9, 1, tzinfo=UTC)
    saved = SimpleNamespace(
        config={
            "configurable": {
                "thread_id": "thread-1",
                "checkpoint_ns": "",
                "checkpoint_id": "cp-1",
            }
        },
        checkpoint={"id": "cp-1", "ts": created.isoformat()},
    )
    assert saved.checkpoint["id"] == "cp-1"


def test_manifest_accepts_one_typed_call_with_aggregate_retry_usage() -> None:
    started = datetime(2026, 9, 1, tzinfo=UTC)
    usage = ResourceUsage.zero().model_copy(
        update={"retries": 2, "search_calls": 1, "wall_seconds": 1.0}
    )
    call = ProviderCallRecord(
        operation="search",
        node="Tool",
        provider_id="offline-search",
        endpoint_type="search",
        request_sha256="1" * 64,
        snapshot_id="search-snapshot-v1",
        normalized_query="offline query",
        locale="en",
        complete_parameters={"filters": None, "limit": 10},
        time_policy="frozen",
        started_at=started,
        finished_at=started + timedelta(seconds=1),
        latency_ms=1_000,
        attempt=1,
        cache_hit=False,
        outcome_code="SUCCESS",
        usage=usage,
    )
    RunManifest.create(
        {
            "schema_version": "run-manifest-v1",
            "run_id": "run-retries",
            "thread_id": "thread-retries",
            "code_commit": "a" * 40,
            "dependency_lock_sha256": "2" * 64,
            "request_sha256": "3" * 64,
            "config_sha256": "4" * 64,
            "workflow_id": "baseline-v1",
            "graph_version": "baseline-graph-v1",
            "planner_id": "P1",
            "provider_profiles": (
                ProviderProfileRecord(
                    profile_id="offline",
                    execution_mode="replay",
                    provider_ids=("offline-search",),
                    configuration_sha256="5" * 64,
                ),
            ),
            "model_ids": (),
            "prompt_versions": {"planner": "p1"},
            "parser_versions": {"html": "v1"},
            "ranker_id": "R1",
            "ranker_weights_version": "r1-v1",
            "budget": RunBudget.preset("low"),
            "usage": usage,
            "usage_by_node": {"Tool": usage},
            "pricing_status": "unknown",
            "pricing_snapshots": (),
            "provider_calls": (call,),
            "node_executions": (
                NodeExecutionRecord(
                    node="Tool",
                    attempt=1,
                    started_at=started,
                    finished_at=started + timedelta(seconds=1),
                    latency_ms=1_000,
                    status="completed",
                    input_artifact_ids=(),
                    output_artifact_ids=(),
                    usage=usage,
                ),
            ),
            "parsed_artifacts": (),
            "evidence_hashes": (),
            "source_snapshot_ids": ("search-snapshot-v1",),
            "artifact_ids": (),
            "run_event_count": 1,
            "run_events_sha256": "6" * 64,
            "seed": 0,
            "seed_supported": True,
            "cache_hit_count": 0,
            "stop_reason": "SUFFICIENT",
            "is_partial": False,
            "failure_codes": (),
            "replay_parent": "parent-manifest-v1",
            "started_at": started,
            "finished_at": started + timedelta(seconds=1),
        }
    )


class OfflineModel:
    provider_id = "offline-model"
    model_id = "offline-model-v1"
    model_revision = "offline-model-revision-v1"

    def __init__(self, plan_payload: dict[str, object]) -> None:
        self.plan_payload = plan_payload
        self.last_usage: ResourceUsage | None = None
        self.requests: list[ModelRequest] = []

    @staticmethod
    def _usage() -> ResourceUsage:
        return ResourceUsage(
            input_tokens=10,
            output_tokens=10,
            reasoning_tokens=0,
            cached_tokens=0,
            total_tokens=20,
            search_calls=0,
            pages=0,
            retries=0,
            wall_seconds=0.0,
            cost_usd=Decimal(0),
        )

    async def complete(self, request: ModelRequest, **kwargs: object) -> ModelResult[str]:
        del kwargs
        self.requests.append(request)
        usage = self._usage()
        self.last_usage = usage
        if request.prompt_version == "fixed-planner-v1":
            output = json.dumps(self.plan_payload, sort_keys=True)
        else:
            evidence_ids = sorted(set(re.findall(r"E-[0-9a-f]+", request.messages[-1].content)))
            output = (
                "Offline baseline result " + " ".join(f"[{item}]" for item in evidence_ids) + "."
            )
        return ModelResult(
            output=output,
            usage=usage,
            provider_id=self.provider_id,
            model_id=self.model_id,
            raw_response_artifact_id="sha256:" + "a" * 64,
        )

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[Any],
        **kwargs: object,
    ) -> StructuredModelResult[Any]:
        del kwargs
        self.requests.append(request)
        usage = self._usage()
        self.last_usage = usage
        return StructuredModelResult(
            output=output_schema.model_validate({"queries": ["offline baseline query"]}),
            usage=usage,
            provider_id=self.provider_id,
            model_id=self.model_id,
            raw_response_artifact_id="sha256:" + "b" * 64,
            output_schema_hash=request.output_schema_hash,
        )

    def stream(self, request: ModelRequest, **kwargs: object):  # type: ignore[no-untyped-def]
        del request, kwargs
        raise AssertionError("streaming is not used")


class OfflineSearch:
    provider_id = "offline-search"

    async def search(self, query: str, limit: int, filters: object, **kwargs: object):  # type: ignore[no-untyped-def]
        raise AssertionError("typed search usage path must be used")

    async def search_with_usage(self, query: str, limit: int, filters: object, **kwargs: object):  # type: ignore[no-untyped-def]
        del query, limit, filters, kwargs
        return ProviderUsageResult(
            value=[
                SearchHit(
                    url=f"https://source{index}.example.test/doc",
                    title=f"Offline source {index}",
                    snippet="UNTRUSTED replay snippet",
                    rank=index,
                )
                for index in (1, 2)
            ],
            usage=ResourceUsage.zero(cost_known=True).model_copy(
                update={"search_calls": 1, "wall_seconds": 0.25}
            ),
        )


class OfflineFetcher:
    provider_id = "offline-fetch"

    async def fetch(self, url: str, **kwargs: object) -> RawDocument:
        raise AssertionError("typed fetch usage path must be used")

    async def fetch_with_usage(
        self, url: str, **kwargs: object
    ) -> ProviderUsageResult[RawDocument]:
        del kwargs
        body = f"UNTRUSTED offline document from {url}".encode()
        return ProviderUsageResult(
            value=RawDocument(
                requested_url=url,
                final_url=url,
                status=200,
                headers={"content-type": "text/html"},
                content_type="text/html",
                body_bytes=body,
                retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
            ),
            usage=ResourceUsage.zero(cost_known=True).model_copy(
                update={"pages": 1, "wall_seconds": 0.5}
            ),
        )


class OfflineParser:
    parser_id = "offline-parser"
    parser_version = "offline-parser-v1"

    def supports(self, content_type: str) -> bool:
        return content_type == "text/html"

    async def parse(self, raw_document: RawDocument, **kwargs: object) -> ParsedDocument:
        del kwargs
        text = raw_document.body_bytes.decode()
        return ParsedDocument(
            canonical_url=raw_document.final_url,
            title=f"Parsed {raw_document.final_url.host}",
            authors=(),
            normalized_text=text,
            blocks=(
                ParsedBlock(
                    block_id="p-1",
                    text=text,
                    locator=HtmlLocator(
                        paragraph_id="p-1",
                        start_char=0,
                        end_char=len(text),
                    ),
                    text_hash=hashlib.sha256(text.encode()).hexdigest(),
                ),
            ),
            parser_id=self.parser_id,
            parser_version=self.parser_version,
            parsed_content_hash=hashlib.sha256(text.encode()).hexdigest(),
        )


class OfflineEmbedder:
    provider_id = "offline-embed"
    model_id = "offline-embed-v1"
    model_revision = "revision-1"
    snapshot_sha256 = "c" * 64

    async def embed(self, texts: object, **kwargs: object):  # type: ignore[no-untyped-def]
        del kwargs
        return tuple((1.0, 0.0) for _ in cast("tuple[str, ...]", tuple(texts)))


class ZeroCostResolver:
    def __init__(self, cost: Decimal | None = Decimal(0)) -> None:
        self.cost = cost

    def resolve_cost(self, **kwargs: object) -> Decimal | None:
        del kwargs
        return self.cost


class CountingOfflineModel(OfflineModel):
    def __init__(
        self,
        plan_payload: dict[str, object],
        invocations: Counter[str],
    ) -> None:
        super().__init__(plan_payload)
        self.invocations = invocations

    async def complete(self, request: ModelRequest, **kwargs: object) -> ModelResult[str]:
        self.invocations[f"model:{request.prompt_version}"] += 1
        return await super().complete(request, **kwargs)

    async def structured(
        self,
        request: ModelRequest,
        output_schema: type[Any],
        **kwargs: object,
    ) -> StructuredModelResult[Any]:
        self.invocations[f"model:{request.prompt_version}"] += 1
        return await super().structured(request, output_schema, **kwargs)


class CountingOfflineSearch(OfflineSearch):
    def __init__(self, invocations: Counter[str]) -> None:
        self.invocations = invocations

    async def search_with_usage(
        self,
        query: str,
        limit: int,
        filters: object,
        **kwargs: object,
    ) -> ProviderUsageResult[list[SearchHit]]:
        self.invocations[f"search:{query}"] += 1
        return await super().search_with_usage(query, limit, filters, **kwargs)


class CountingOfflineFetcher(OfflineFetcher):
    def __init__(self, invocations: Counter[str]) -> None:
        self.invocations = invocations

    async def fetch_with_usage(
        self,
        url: str,
        **kwargs: object,
    ) -> ProviderUsageResult[RawDocument]:
        self.invocations[f"fetch:{url}"] += 1
        return await super().fetch_with_usage(url, **kwargs)


class CountingOfflineParser(OfflineParser):
    def __init__(self, invocations: Counter[str]) -> None:
        self.invocations = invocations

    async def parse(self, raw_document: RawDocument, **kwargs: object) -> ParsedDocument:
        digest = hashlib.sha256(raw_document.body_bytes).hexdigest()
        self.invocations[f"parse:{digest}"] += 1
        return await super().parse(raw_document, **kwargs)


class CountingOfflineEmbedder(OfflineEmbedder):
    def __init__(self, invocations: Counter[str]) -> None:
        self.invocations = invocations

    async def embed(self, texts: object, **kwargs: object):  # type: ignore[no-untyped-def]
        values = tuple(cast("Any", texts))
        digest = _canonical_sha256(list(values))
        self.invocations[f"embed:{digest}"] += 1
        return await super().embed(values, **kwargs)


def _matches_recovery_case(case: str, key: object) -> bool:
    if case == "p1-model":
        return type(key) is ModelCacheKey and key.prompt_version == "fixed-planner-v1"
    if case == "query-model":
        return (
            type(key) is ModelCacheKey
            and key.prompt_version == "fixed-planner-v1-queries"
        )
    if case == "search":
        return (
            type(key) is SearchCacheKey
            and key.normalized_query == "offline baseline query"
        )
    if case == "fetch":
        return (
            type(key) is FetchCacheKey
            and str(key.canonical_url) == "https://source1.example.test/doc"
        )
    if case == "parse":
        expected_body = (
            b"UNTRUSTED offline document from https://source1.example.test/doc"
        )
        return (
            type(key) is ParseCacheKey
            and key.raw_content_hash == hashlib.sha256(expected_body).hexdigest()
        )
    if case == "embed":
        return type(key) is EmbedCacheKey
    if case == "writer":
        return type(key) is ModelCacheKey and key.prompt_version == "baseline-writer-v1"
    return False


class CrashOnceFileCache(FileCache):
    def __init__(self, root: Path, *, case: str, probe: dict[str, object]) -> None:
        super().__init__(root)
        self.case = case
        self.probe = probe
        self.crashed = False

    def put_if_absent(self, key: Any, value: CacheEntry) -> CacheEntry:
        returned = super().put_if_absent(key, value)
        if not self.crashed and _matches_recovery_case(self.case, key):
            self.crashed = True
            self.probe["key"] = key
            self.probe["entry"] = returned
            raise MemoryError(f"crash after {self.case} cache publication")
        return returned


class CrashOnceManifestStore(LocalArtifactStore):
    def __init__(self, root: Path, *, probe: dict[str, object]) -> None:
        super().__init__(root)
        self.probe = probe
        self.crashed = False

    def put_bytes(self, data: bytes, *, media_type: str):  # type: ignore[no-untyped-def]
        reference = super().put_bytes(data, media_type=media_type)
        if (
            not self.crashed
            and media_type == "application/vnd.deepresearch.run-manifest+json"
        ):
            self.crashed = True
            self.probe["manifest_id"] = reference.artifact_id
            self.probe["manifest_bytes"] = data
            raise MemoryError("crash after manifest publication")
        return reference


class SegmentClock:
    def __init__(self, segment: int) -> None:
        self._monotonic = time.monotonic()
        self._utc = datetime(2026, 9, 1, tzinfo=UTC) + timedelta(
            seconds=5 * segment
        )

    def monotonic(self) -> float:
        self._monotonic += 0.01
        return self._monotonic

    def utc_now(self) -> datetime:
        self._utc += timedelta(milliseconds=10)
        return self._utc


class ControlledSegmentClock:
    def __init__(
        self,
        *,
        monotonic_start: float,
        utc_offset_seconds: float,
    ) -> None:
        self.monotonic_start = monotonic_start
        self.value = monotonic_start
        self.utc_offset_seconds = utc_offset_seconds
        self.utc_calls = 0

    def advance(self, seconds: float) -> None:
        self.value += seconds

    def monotonic(self) -> float:
        return self.value

    def utc_now(self) -> datetime:
        self.utc_calls += 1
        return datetime(2026, 9, 1, tzinfo=UTC) + timedelta(
            seconds=(
                self.utc_offset_seconds + self.value - self.monotonic_start
            ),
            microseconds=self.utc_calls,
        )


def _usage(*, tokens: int) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=tokens,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=tokens,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0.0,
        cost_usd=Decimal(0),
    )


def _bare_handlers() -> BaselineNodeHandlers:
    value = object.__new__(BaselineNodeHandlers)
    value.usage_cost_resolver = ZeroCostResolver()
    value._locks_guard = asyncio.Lock()  # type: ignore[attr-defined]
    value._locks = {}  # type: ignore[attr-defined]
    return value


def _runtime_context(
    *,
    ticks: object,
    run_config: RunConfig | None = None,
    run_id: str = "run-usage",
    thread_id: str = "thread-usage",
) -> BaselineRuntimeContext:
    current = run_config or config()
    tick_iterator = cast("Iterator[float]", ticks)
    last_tick = 0.0

    def monotonic() -> float:
        nonlocal last_tick
        try:
            last_tick = next(tick_iterator)
        except StopIteration:
            pass
        return last_tick

    return BaselineRuntimeContext(
        run_id=run_id,
        thread_id=thread_id,
        config=current,
        emit=cast("Any", _emit),
        cancellation_token=CancellationToken(),
        budget_accountant=BudgetAccountant(current.budget, run_scope="usage-test"),
        deadline=999.0,
        run_started_monotonic=0.0,
        run_started_at=datetime(2026, 9, 1, tzinfo=UTC),
        elapsed_base_seconds=0.0,
        monotonic=monotonic,
        utc_now=lambda: datetime(2026, 9, 1, tzinfo=UTC),
        new_id=lambda prefix: prefix,
    )


class MutableMonotonic:
    def __init__(self, value: float) -> None:
        self.value = value

    def __call__(self) -> float:
        return self.value


class InstrumentedAsyncLock:
    def __init__(self, inner: asyncio.Lock) -> None:
        self.inner = inner
        self.queued = asyncio.Event()
        self.acquired = asyncio.Event()

    async def __aenter__(self) -> Self:
        self.queued.set()
        await self.inner.acquire()
        self.acquired.set()
        return self

    async def __aexit__(self, *_args: object) -> None:
        self.inner.release()


def _model_request(*, content: str) -> ModelRequest:
    return ModelRequest(
        model_id="offline-model-v1",
        messages=(ModelMessage(role="user", content=content),),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=1,
        prompt_version="fixed-planner-v1",
        system_prompt_hash=hashlib.sha256(b"system").hexdigest(),
        tool_schema_hash=hashlib.sha256(b"[]").hexdigest(),
        output_schema_hash=hashlib.sha256(b"").hexdigest(),
    )


def _model_cache_key(provider: object, request_value: ModelRequest) -> ModelCacheKey:
    return ModelCacheKey(
        provider_id=cast("Any", provider).provider_id,
        endpoint_type="complete",
        model_id=request_value.model_id,
        prompt_version=request_value.prompt_version,
        system_prompt_hash=request_value.system_prompt_hash,
        tool_schema_hash=request_value.tool_schema_hash,
        output_schema_hash=request_value.output_schema_hash,
        temperature=request_value.temperature,
        seed=request_value.seed,
        canonical_request_hash=_canonical_sha256(
            request_value.model_dump(mode="json")
        ),
    )


@pytest.mark.parametrize("gate", ["cache", "provider"])
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_waiting_provider_rechecks_stop_before_any_second_side_effect(
    tmp_path: Path,
    gate: str,
    stop: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    run_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )

    class ObservedProvider(OfflineModel):
        def __init__(self) -> None:
            super().__init__(_offline_plan())
            self.observer_consumptions = 0

        def consume_invocation_usage(self) -> ResourceUsage | None:
            self.observer_consumptions += 1
            return None

    provider = ObservedProvider()
    first_request = _model_request(content="first cache identity")
    second_request = (
        first_request
        if gate == "cache"
        else _model_request(content="different cache identity")
    )
    first_clock = MutableMonotonic(0.0)
    second_clock = MutableMonotonic(0.0)

    def make_context(
        *,
        run_id: str,
        thread_id: str,
        clock: MutableMonotonic,
        deadline: float,
    ) -> BaselineRuntimeContext:
        base = _runtime_context(
            ticks=iter(()),
            run_config=run_config,
            run_id=run_id,
            thread_id=thread_id,
        )
        context = replace(
            base,
            budget_accountant=BudgetAccountant(
                run_config.budget,
                run_scope=run_id,
            ),
            deadline=deadline,
            monotonic=clock,
        )
        context.audit.begin(graph_node="Plan", node_attempt=1)
        return context

    first_context = make_context(
        run_id="run-lock-first",
        thread_id="thread-lock-first",
        clock=first_clock,
        deadline=100.0,
    )
    second_context = make_context(
        run_id="run-lock-second",
        thread_id="thread-lock-second",
        clock=second_clock,
        deadline=1.0,
    )
    second_before = second_context.budget_accountant.snapshot()
    task_contexts: dict[object, BaselineRuntimeContext] = {}
    cast("Any", handlers)._runtime = lambda: task_contexts[asyncio.current_task()]
    provider_calls: Counter[str] = Counter()
    first_entered = asyncio.Event()
    release_first = asyncio.Event()

    async def invoke_provider(
        *,
        label: str,
        request_value: ModelRequest,
    ) -> ModelResult[str]:
        provider_calls[label] += 1
        if label == "first":
            first_entered.set()
            await asyncio.wait_for(release_first.wait(), timeout=5.0)
        return await provider.complete(request_value)

    async def run_call(
        *,
        context: BaselineRuntimeContext,
        label: str,
        request_value: ModelRequest,
    ) -> object:
        task_contexts[asyncio.current_task()] = context
        return await cast("Any", handlers)._cached_model_call(
            provider=provider,
            request=request_value,
            output_schema=None,
            invoke=lambda: invoke_provider(label=label, request_value=request_value),
        )

    first_task = asyncio.create_task(
        run_call(
            context=first_context,
            label="first",
            request_value=first_request,
        )
    )
    await asyncio.wait_for(first_entered.wait(), timeout=5.0)
    lock_digest = (
        cache_key_sha256(_model_cache_key(provider, first_request))
        if gate == "cache"
        else cast("Any", baseline_graph_module)._hash_json(
            {"provider": provider.provider_id}
        )
    )
    original_lock = cast("Any", handlers)._locks[lock_digest]
    instrumented_lock = InstrumentedAsyncLock(original_lock)
    cast("Any", handlers)._locks[lock_digest] = instrumented_lock
    second_task = asyncio.create_task(
        run_call(
            context=second_context,
            label="second",
            request_value=second_request,
        )
    )
    await asyncio.wait_for(instrumented_lock.queued.wait(), timeout=5.0)
    assert not second_task.done()
    assert not instrumented_lock.acquired.is_set()
    if stop == "timeout":
        second_clock.value = 1.0
    else:
        second_context.cancellation_token.cancel()
    release_first.set()

    first_result = await first_task
    expected_error = ProviderError if stop == "timeout" else OperationCancelled
    with pytest.raises(expected_error) as caught:
        await second_task

    if stop == "timeout":
        assert cast("ProviderError", caught.value).code == "TIMEOUT"
    assert instrumented_lock.acquired.is_set()
    assert first_result.output
    assert provider_calls == Counter({"first": 1})
    assert provider.observer_consumptions == 2
    assert second_context.budget_accountant.snapshot() == second_before
    assert second_context.audit.provider_calls == []
    assert second_context.audit.provider_receipt_ids == []
    assert second_context.audit.result_artifact_ids == []

    second_key = _model_cache_key(provider, second_request)
    entry = handlers.cache.get(second_key)
    if gate == "cache":
        assert entry is not None
        assert entry.metadata["producer_run_id"] == first_context.run_id
    else:
        assert entry is None


@pytest.mark.parametrize("cache_state", ["miss", "hit"])
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_model_cache_lock_waiter_rechecks_stop_for_hit_and_miss(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    cache_state: str,
    stop: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    clock = MutableMonotonic(0.0)
    run_id = f"run-model-{cache_state}-cache-waiter"
    context = replace(
        _runtime_context(
            ticks=iter(()),
            run_config=current_config,
            run_id=run_id,
            thread_id=f"thread-model-{cache_state}-cache-waiter",
        ),
        budget_accountant=BudgetAccountant(
            current_config.budget,
            run_scope=run_id,
        ),
        deadline=1.0,
        monotonic=clock,
    )
    _bind_handler_runtime(handlers, context, graph_node="Plan")
    provider = OfflineModel(_offline_plan())
    request_value = _model_request(content="model cache waiter")
    provider_calls = 0

    async def invoke_provider() -> ModelResult[str]:
        nonlocal provider_calls
        provider_calls += 1
        return await provider.complete(request_value)

    if cache_state == "hit":
        await cast("Any", handlers)._cached_model_call(
            provider=provider,
            request=request_value,
            output_schema=None,
            invoke=invoke_provider,
        )
        context.audit.reset()
        context.audit.begin(graph_node="Plan", node_attempt=1)
    key = _model_cache_key(provider, request_value)
    seeded_entry = handlers.cache.get(key)
    assert (seeded_entry is not None) is (cache_state == "hit")
    provider_calls_before = provider_calls
    before = context.budget_accountant.snapshot()
    inner_lock = asyncio.Lock()
    await inner_lock.acquire()
    instrumented_lock = InstrumentedAsyncLock(inner_lock)
    cast("Any", handlers)._locks[cache_key_sha256(key)] = instrumented_lock
    cache_gets = 0
    original_cache_get = handlers.cache.get

    def observed_cache_get(key_value: object) -> CacheEntry | None:
        nonlocal cache_gets
        cache_gets += 1
        return original_cache_get(cast("Any", key_value))

    monkeypatch.setattr(handlers.cache, "get", observed_cache_get)
    cached_observations = 0
    original_observe_cached = cast("Any", handlers)._observe_cached

    async def observed_cached(*args: object, **kwargs: object) -> object:
        nonlocal cached_observations
        cached_observations += 1
        return await original_observe_cached(*args, **kwargs)

    monkeypatch.setattr(handlers, "_observe_cached", observed_cached)
    task = asyncio.create_task(
        cast("Any", handlers)._cached_model_call(
            provider=provider,
            request=request_value,
            output_schema=None,
            invoke=invoke_provider,
        )
    )
    await asyncio.wait_for(instrumented_lock.queued.wait(), timeout=5.0)
    if stop == "timeout":
        clock.value = 1.0
    else:
        context.cancellation_token.cancel()
    inner_lock.release()

    expected_error = ProviderError if stop == "timeout" else OperationCancelled
    with pytest.raises(expected_error) as caught:
        await task

    if stop == "timeout":
        assert cast("ProviderError", caught.value).code == "TIMEOUT"
    assert instrumented_lock.acquired.is_set()
    assert cache_gets == 0
    assert cached_observations == 0
    assert provider_calls == provider_calls_before
    assert context.budget_accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []
    assert original_cache_get(key) == seeded_entry


@pytest.mark.parametrize("operation", ["embed", "search", "fetch", "parse"])
@pytest.mark.parametrize("stop", ["timeout", "cancel"])
@pytest.mark.parametrize("cache_state", ["miss", "hit"])
async def test_non_model_cache_lock_waiter_rechecks_stop_before_cache_access(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
    stop: str,
    cache_state: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    clock = MutableMonotonic(0.0)
    context = replace(
        _runtime_context(
            ticks=iter(()),
            run_config=current_config,
            run_id=f"run-{operation}-{cache_state}-cache-waiter",
            thread_id=f"thread-{operation}-{cache_state}-cache-waiter",
        ),
        budget_accountant=BudgetAccountant(
            current_config.budget,
            run_scope=f"run-{operation}-{cache_state}-cache-waiter",
        ),
        deadline=1.0,
        monotonic=clock,
    )
    graph_node = {
        "embed": "RankEvidence",
        "search": "Search",
        "fetch": "Fetch",
        "parse": "ParseAndNormalize",
    }[operation]
    _bind_handler_runtime(
        handlers,
        context,
        graph_node=graph_node,
    )
    inner_lock = asyncio.Lock()
    await inner_lock.acquire()
    instrumented_lock = InstrumentedAsyncLock(inner_lock)

    async def fixed_lock(_digest: str) -> InstrumentedAsyncLock:
        return instrumented_lock

    original_lock_for = handlers._lock_for
    monkeypatch.setattr(handlers, "_lock_for", fixed_lock)
    cache_gets = 0
    original_cache_get = handlers.cache.get

    def observed_cache_get(key: object) -> CacheEntry | None:
        nonlocal cache_gets
        cache_gets += 1
        return original_cache_get(cast("Any", key))

    monkeypatch.setattr(handlers.cache, "get", observed_cache_get)

    async def invoke_operation() -> object:
        if operation == "embed":
            proxy = cast("Any", handlers.ranker).embedder
            provider = proxy._inner
            return await cast("Any", handlers)._cached_embed_call(
                provider=provider,
                texts=("offline proof",),
                invoke=lambda: provider.embed(
                    ("offline proof",),
                    deadline=999.0,
                    cancellation_token=CancellationToken(),
                ),
            )
        if operation == "search":
            async def fixed_queries(
                _context: object,
                _invoke: object,
            ) -> tuple[str, ...]:
                return ("offline baseline query",)

            monkeypatch.setattr(
                handlers,
                "_isolated_planner_call",
                fixed_queries,
            )
            plan = handlers.artifact_store.put_bytes(
                json.dumps(
                    _offline_plan(),
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                media_type="application/vnd.deepresearch.plan+json",
            )
            return await handlers.search(
                cast(
                    "Any",
                    {
                        "run_id": context.run_id,
                        "plan_artifact_id": plan.artifact_id,
                        "active_subquestion_id": "sq-offline",
                        "baseline_work_artifact_ids": (),
                    },
                )
            )
        if operation == "fetch":
            search_work = handlers.artifact_store.put_bytes(
                json.dumps(
                    {
                        "hits": [
                            SearchHit(
                                url="https://cache-wait.example.test/doc",
                                title="Cache wait",
                                snippet="offline",
                                rank=1,
                            ).model_dump(mode="json")
                        ],
                        "kind": "search",
                    },
                    ensure_ascii=False,
                    allow_nan=False,
                    sort_keys=True,
                    separators=(",", ":"),
                ).encode(),
                media_type="application/vnd.deepresearch.baseline-work+json",
            )
            return await handlers.fetch(
                cast(
                    "Any",
                    {
                        "run_id": context.run_id,
                        "baseline_work_artifact_ids": (search_work.artifact_id,),
                    },
                )
            )
        raw = RawDocument(
            requested_url="https://cache-wait.example.test/doc",
            final_url="https://cache-wait.example.test/doc",
            status=200,
            headers={"content-type": "text/html"},
            content_type="text/html",
            body_bytes=b"offline cache wait",
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        )
        raw_work = handlers.artifact_store.put_bytes(
            json.dumps(
                {
                    "documents": [raw.model_dump(mode="json")],
                    "kind": "raw",
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            media_type="application/vnd.deepresearch.baseline-work+json",
        )
        return await handlers.parse_and_normalize(
            cast(
                "Any",
                {
                    "run_id": context.run_id,
                    "baseline_work_artifact_ids": (raw_work.artifact_id,),
                },
            )
        )

    if cache_state == "hit":
        inner_lock.release()
        monkeypatch.setattr(handlers, "_lock_for", original_lock_for)
        await invoke_operation()
        monkeypatch.setattr(handlers, "_lock_for", fixed_lock)
        context.audit.reset()
        context.audit.begin(graph_node=graph_node, node_attempt=1)
        await inner_lock.acquire()
        instrumented_lock = InstrumentedAsyncLock(inner_lock)
        cache_gets = 0
    cached_observations = 0
    original_observe_cached = cast("Any", handlers)._observe_cached

    async def observed_cached(*args: object, **kwargs: object) -> object:
        nonlocal cached_observations
        cached_observations += 1
        return await original_observe_cached(*args, **kwargs)

    monkeypatch.setattr(handlers, "_observe_cached", observed_cached)
    provider_invocations = 0
    if operation == "embed":
        provider_owner = cast("Any", handlers.ranker).embedder._inner
        provider_method_name = "embed"
    elif operation == "search":
        provider_owner = handlers.search_provider
        provider_method_name = "search_with_usage"
    elif operation == "fetch":
        provider_owner = handlers.fetcher
        provider_method_name = "fetch_with_usage"
    else:
        provider_owner = handlers.parser
        provider_method_name = "parse"
    original_provider_invoke = getattr(provider_owner, provider_method_name)

    async def observed_provider(*args: object, **kwargs: object) -> object:
        nonlocal provider_invocations
        provider_invocations += 1
        return await original_provider_invoke(*args, **kwargs)

    monkeypatch.setattr(
        provider_owner,
        provider_method_name,
        observed_provider,
    )
    cache_files_before = tuple(
        sorted(
            path.relative_to(tmp_path).as_posix()
            for path in (tmp_path / "cache").rglob("*.json")
        )
    )
    assert bool(cache_files_before) is (cache_state == "hit")
    before = context.budget_accountant.snapshot()
    task = asyncio.create_task(invoke_operation())
    await asyncio.wait_for(instrumented_lock.queued.wait(), timeout=5.0)
    assert not task.done()
    assert not instrumented_lock.acquired.is_set()
    if stop == "timeout":
        clock.value = 1.0
    else:
        context.cancellation_token.cancel()
    inner_lock.release()

    expected_error = ProviderError if stop == "timeout" else OperationCancelled
    with pytest.raises(expected_error) as caught:
        await task

    if stop == "timeout":
        assert cast("ProviderError", caught.value).code == "TIMEOUT"
    assert instrumented_lock.acquired.is_set()
    assert cache_gets == 0
    assert cached_observations == 0
    assert provider_invocations == 0
    assert context.budget_accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []
    cache_files_after = tuple(
        sorted(
            path.relative_to(tmp_path).as_posix()
            for path in (tmp_path / "cache").rglob("*.json")
        )
    )
    assert cache_files_after == cache_files_before


@pytest.mark.parametrize("stop", ["timeout", "cancel"])
async def test_waiting_planner_rechecks_stop_before_mutating_shared_state(
    tmp_path: Path,
    stop: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    first_clock = MutableMonotonic(0.0)
    second_clock = MutableMonotonic(0.0)
    first_context = replace(
        _runtime_context(ticks=iter(()), run_config=current_config),
        deadline=100.0,
        monotonic=first_clock,
    )
    second_context = replace(
        _runtime_context(
            ticks=iter(()),
            run_config=current_config,
            run_id="run-planner-waiter",
            thread_id="thread-planner-waiter",
        ),
        budget_accountant=BudgetAccountant(
            current_config.budget,
            run_scope="run-planner-waiter",
        ),
        deadline=1.0,
        monotonic=second_clock,
    )
    planner = cast("Any", handlers.initial_plan_generator)
    planner_before = (
        planner.budget,
        planner._initial_model_tokens,
        planner._model_tokens_used,
        planner._model_tokens_reserved,
        planner._query_cache,
        planner._query_locks,
    )
    first_entered = asyncio.Event()
    release_first = asyncio.Event()
    invocations: Counter[str] = Counter()

    async def first_invoke() -> str:
        invocations["first"] += 1
        first_entered.set()
        await asyncio.wait_for(release_first.wait(), timeout=5.0)
        return "first"

    async def second_invoke() -> str:
        invocations["second"] += 1
        return "second"

    first_task = asyncio.create_task(
        cast("Any", handlers)._isolated_planner_call(first_context, first_invoke)
    )
    await asyncio.wait_for(first_entered.wait(), timeout=5.0)
    planner_lock = InstrumentedAsyncLock(cast("Any", handlers)._planner_lock)
    cast("Any", handlers)._planner_lock = planner_lock
    second_task = asyncio.create_task(
        cast("Any", handlers)._isolated_planner_call(second_context, second_invoke)
    )
    await asyncio.wait_for(planner_lock.queued.wait(), timeout=5.0)
    assert not second_task.done()
    assert not planner_lock.acquired.is_set()
    if stop == "timeout":
        second_clock.value = 1.0
    else:
        second_context.cancellation_token.cancel()
    release_first.set()

    assert await first_task == "first"
    expected_error = ProviderError if stop == "timeout" else OperationCancelled
    with pytest.raises(expected_error) as caught:
        await second_task
    if stop == "timeout":
        assert cast("ProviderError", caught.value).code == "TIMEOUT"
    assert planner_lock.acquired.is_set()
    assert invocations == Counter({"first": 1})
    planner_after = (
        planner.budget,
        planner._initial_model_tokens,
        planner._model_tokens_used,
        planner._model_tokens_reserved,
        planner._query_cache,
        planner._query_locks,
    )
    assert planner_after == planner_before
    assert planner_after[4] is planner_before[4]
    assert planner_after[5] is planner_before[5]
    assert second_context.audit.provider_calls == []
    assert second_context.audit.provider_receipt_ids == []


@pytest.mark.parametrize("stop", ["timeout", "cancel", "cancel-and-timeout"])
async def test_final_preinvoke_gate_releases_reservation_without_a_call(
    monkeypatch: pytest.MonkeyPatch,
    stop: str,
) -> None:
    handlers = _bare_handlers()
    clock = MutableMonotonic(0.0)
    context = replace(
        _runtime_context(ticks=iter(())),
        deadline=1.0,
        monotonic=clock,
    )
    context.audit.begin(graph_node="ParseAndNormalize", node_attempt=1)
    before = context.budget_accountant.snapshot()
    original_reserve = context.budget_accountant.reserve
    original_release = context.budget_accountant.release
    release_calls = 0

    def stop_after_reserve(*args: Any, **kwargs: Any) -> object:
        reservation = original_reserve(*args, **kwargs)
        if stop in {"timeout", "cancel-and-timeout"}:
            clock.value = 1.0
        if stop in {"cancel", "cancel-and-timeout"}:
            context.cancellation_token.cancel()
        return reservation

    def observed_release(reservation: object) -> None:
        nonlocal release_calls
        release_calls += 1
        original_release(cast("Any", reservation))

    monkeypatch.setattr(context.budget_accountant, "reserve", stop_after_reserve)
    monkeypatch.setattr(context.budget_accountant, "release", observed_release)
    provider_calls = 0

    async def invoke() -> str:
        nonlocal provider_calls
        provider_calls += 1
        return "unexpected"

    expected_error = ProviderError if stop == "timeout" else OperationCancelled
    with pytest.raises(expected_error) as caught:
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="post-reservation-provider",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id="post-reservation-timeout",
            invoke=invoke,
            result_usage=lambda _result: ResourceUsage.zero(cost_known=True),
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    if stop == "timeout":
        assert cast("ProviderError", caught.value).code == "TIMEOUT"
    assert provider_calls == 0
    assert release_calls == 1
    assert context.budget_accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []


async def test_final_preinvoke_gate_propagates_hard_release_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _bare_handlers()
    clock = MutableMonotonic(0.0)
    context = replace(
        _runtime_context(ticks=iter(())),
        deadline=1.0,
        monotonic=clock,
    )
    context.audit.begin(graph_node="ParseAndNormalize", node_attempt=1)
    before = context.budget_accountant.snapshot()
    original_reserve = context.budget_accountant.reserve
    original_release = context.budget_accountant.release
    primary = MemoryError("hard reservation release failure")
    release_calls = 0

    def stop_after_reserve(*args: Any, **kwargs: Any) -> object:
        reservation = original_reserve(*args, **kwargs)
        clock.value = 1.0
        return reservation

    def hard_release(reservation: object) -> None:
        nonlocal release_calls
        release_calls += 1
        original_release(cast("Any", reservation))
        raise primary

    monkeypatch.setattr(context.budget_accountant, "reserve", stop_after_reserve)
    monkeypatch.setattr(context.budget_accountant, "release", hard_release)
    provider_calls = 0

    async def invoke() -> str:
        nonlocal provider_calls
        provider_calls += 1
        return "unexpected"

    with pytest.raises(MemoryError) as caught:
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="post-reservation-provider",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id="post-reservation-hard-release",
            invoke=invoke,
            result_usage=lambda _result: ResourceUsage.zero(cost_known=True),
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    assert caught.value is primary
    assert provider_calls == 0
    assert release_calls == 1
    assert context.budget_accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []


async def test_final_preinvoke_gate_runs_after_timestamp_hooks_at_deadline() -> None:
    handlers = _bare_handlers()
    context = replace(
        _runtime_context(ticks=iter((0.0, 1.0))),
        deadline=1.0,
    )
    context.audit.begin(graph_node="ParseAndNormalize", node_attempt=1)
    before = context.budget_accountant.snapshot()
    provider_calls = 0

    async def invoke() -> str:
        nonlocal provider_calls
        provider_calls += 1
        return "unexpected"

    with pytest.raises(ProviderError) as caught:
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="timestamp-boundary-provider",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id="timestamp-boundary-timeout",
            invoke=invoke,
            result_usage=lambda _result: ResourceUsage.zero(cost_known=True),
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    assert caught.value.code == "TIMEOUT"
    assert provider_calls == 0
    assert context.budget_accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []


async def test_final_preinvoke_gate_now_starts_provider_wall_measurement() -> None:
    handlers = _bare_handlers()

    class HookAdvancingClock:
        def __init__(self) -> None:
            self.value = 0.0

        def monotonic(self) -> float:
            return self.value

        def utc_now(self) -> datetime:
            self.value += 1.0
            return datetime(2026, 9, 1, tzinfo=UTC) + timedelta(
                seconds=self.value
            )

    clock = HookAdvancingClock()
    context = replace(
        _runtime_context(ticks=iter(())),
        deadline=10.0,
        monotonic=clock.monotonic,
        utc_now=clock.utc_now,
    )
    context.audit.begin(graph_node="ParseAndNormalize", node_attempt=1)

    async def invoke() -> str:
        clock.value += 1.0
        return "ok"

    result, usage = await cast("Any", handlers)._invoke_metered(
        context=context,
        provider=object(),
        provider_id="provider-wall-boundary",
        model_id=None,
        operation="parse",
        node="Tool",
        operation_id="provider-wall-boundary",
        invoke=invoke,
        result_usage=lambda _result: None,
        fallback_usage=ResourceUsage.zero(cost_known=True),
    )

    assert result == "ok"
    assert usage.wall_seconds == 1.0


async def test_cached_model_final_gate_releases_once_without_cache_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    clock = MutableMonotonic(0.0)
    run_id = "run-cached-final-gate"
    context = replace(
        _runtime_context(
            ticks=iter(()),
            run_config=current_config,
            run_id=run_id,
            thread_id="thread-cached-final-gate",
        ),
        budget_accountant=BudgetAccountant(
            current_config.budget,
            run_scope=run_id,
        ),
        deadline=1.0,
        monotonic=clock,
    )
    _bind_handler_runtime(handlers, context, graph_node="Plan")
    provider = OfflineModel(_offline_plan())
    request_value = _model_request(content="cached final gate")
    key = _model_cache_key(provider, request_value)
    assert handlers.cache.get(key) is None
    original_reserve = context.budget_accountant.reserve
    original_release = context.budget_accountant.release
    release_calls = 0

    def expire_after_reserve(*args: Any, **kwargs: Any) -> object:
        reservation = original_reserve(*args, **kwargs)
        clock.value = 1.0
        return reservation

    def observed_release(reservation: object) -> None:
        nonlocal release_calls
        release_calls += 1
        original_release(cast("Any", reservation))

    monkeypatch.setattr(context.budget_accountant, "reserve", expire_after_reserve)
    monkeypatch.setattr(context.budget_accountant, "release", observed_release)
    provider_calls = 0

    async def invoke_provider() -> ModelResult[str]:
        nonlocal provider_calls
        provider_calls += 1
        return await provider.complete(request_value)

    with pytest.raises(ProviderError) as caught:
        await cast("Any", handlers)._cached_model_call(
            provider=provider,
            request=request_value,
            output_schema=None,
            invoke=invoke_provider,
        )

    assert caught.value.code == "TIMEOUT"
    assert release_calls == 1
    assert provider_calls == 0
    assert handlers.cache.get(key) is None
    assert tuple((tmp_path / "cache").rglob("*.json")) == ()
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []


async def test_safe_node_prefers_cancellation_when_deadline_is_also_expired(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch.workflow import runner as runner_module

    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    run_id = "run-cancel-timeout-precedence"
    thread_id = "thread-cancel-timeout-precedence"
    accountant = BudgetAccountant(current_config.budget, run_scope=run_id)
    token = CancellationToken()
    token.cancel()
    sink = MemoryEventSink()
    fixed_now = datetime(2026, 9, 1, tzinfo=UTC)
    context = BaselineRuntimeContext(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        emit=sink,
        cancellation_token=token,
        budget_accountant=accountant,
        deadline=1.0,
        run_started_monotonic=0.0,
        run_started_at=fixed_now,
        elapsed_base_seconds=0.0,
        monotonic=lambda: 1.0,
        utc_now=lambda: fixed_now,
        new_id=lambda prefix: f"{prefix}-fixed",
    )
    state = cast("Any", runner_module)._initial_state(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        accountant=accountant,
    )
    before = accountant.snapshot()
    monkeypatch.setattr(
        baseline_graph_module,
        "get_runtime",
        lambda _schema: SimpleNamespace(context=context),
    )
    handler_calls = 0

    async def forbidden_handler(_state: BaselineState) -> dict[str, object]:
        nonlocal handler_calls
        handler_calls += 1
        return {}

    safe = cast("Any", baseline_graph_module)._safe_node(
        "ValidateRequest",
        forbidden_handler,
        audit_composition=None,
    )
    update = await safe(state)
    event = await sink.get_event(run_id=run_id, seq=1)

    assert handler_calls == 0
    assert update["error_code"] == "CANCELLED"
    assert update["failed_node"] == "ValidateRequest"
    assert update["next_event_seq"] == 2
    assert event is not None
    assert event.error_code == "CANCELLED"
    assert event.status == "cancelled"
    assert accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []


async def test_provider_gate_uses_recovered_offset_at_exact_effective_deadline() -> None:
    handlers = _bare_handlers()
    context = replace(
        _runtime_context(ticks=iter(())),
        deadline=273.0,
        monotonic=MutableMonotonic(253.0),
    )
    context.elapsed_tracker.recover(
        elapsed_wall_seconds=20.0,
        elapsed_base_seconds=0.0,
    )
    context.audit.begin(graph_node="ParseAndNormalize", node_attempt=1)
    before = context.budget_accountant.snapshot()
    provider_calls = 0

    async def invoke() -> str:
        nonlocal provider_calls
        provider_calls += 1
        return "unexpected"

    with pytest.raises(ProviderError) as caught:
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="recovered-offset-provider",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id="recovered-offset-timeout",
            invoke=invoke,
            result_usage=lambda _result: ResourceUsage.zero(cost_known=True),
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    assert caught.value.code == "TIMEOUT"
    assert provider_calls == 0
    assert context.budget_accountant.snapshot() == before
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.result_artifact_ids == []


@pytest.mark.parametrize("operation", ["search", "fetch"])
async def test_untyped_provider_receives_recovered_effective_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    context = replace(
        _runtime_context(
            ticks=iter(()),
            run_config=current_config,
            run_id=f"run-untyped-{operation}-deadline",
            thread_id=f"thread-untyped-{operation}-deadline",
        ),
        budget_accountant=BudgetAccountant(
            current_config.budget,
            run_scope=f"run-untyped-{operation}-deadline",
        ),
        deadline=273.0,
        monotonic=MutableMonotonic(252.0),
    )
    context.elapsed_tracker.recover(
        elapsed_wall_seconds=20.0,
        elapsed_base_seconds=0.0,
    )
    _bind_handler_runtime(
        handlers,
        context,
        graph_node="Search" if operation == "search" else "Fetch",
    )
    observed_deadlines: list[float] = []

    if operation == "search":
        class UntypedSearch:
            provider_id = "untyped-search"

            async def search(
                self,
                _query: str,
                _limit: int,
                _filters: object,
                *,
                deadline: float,
                cancellation_token: CancellationToken,
            ) -> list[SearchHit]:
                cancellation_token.raise_if_cancelled()
                observed_deadlines.append(deadline)
                return [
                    SearchHit(
                        url="https://untyped.example.test/doc",
                        title="Untyped search",
                        snippet="offline",
                        rank=1,
                    )
                ]

        handlers.search_provider = cast("Any", UntypedSearch())

        async def fixed_queries(
            _context: object,
            _invoke: object,
        ) -> tuple[str, ...]:
            return ("offline baseline query",)

        monkeypatch.setattr(handlers, "_isolated_planner_call", fixed_queries)
        plan = handlers.artifact_store.put_bytes(
            json.dumps(
                _offline_plan(),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            media_type="application/vnd.deepresearch.plan+json",
        )
        await handlers.search(
            cast(
                "Any",
                {
                    "run_id": context.run_id,
                    "plan_artifact_id": plan.artifact_id,
                    "active_subquestion_id": "sq-offline",
                    "baseline_work_artifact_ids": (),
                },
            )
        )
    else:
        class UntypedFetcher:
            provider_id = "untyped-fetch"

            async def fetch(
                self,
                url: str,
                *,
                deadline: float,
                cancellation_token: CancellationToken,
            ) -> RawDocument:
                cancellation_token.raise_if_cancelled()
                observed_deadlines.append(deadline)
                return RawDocument(
                    requested_url=url,
                    final_url=url,
                    status=200,
                    headers={"content-type": "text/html"},
                    content_type="text/html",
                    body_bytes=b"untyped fetch",
                    retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
                )

        handlers.fetcher = cast("Any", UntypedFetcher())
        search_work = handlers.artifact_store.put_bytes(
            json.dumps(
                {
                    "hits": [
                        SearchHit(
                            url="https://untyped.example.test/doc",
                            title="Untyped fetch",
                            snippet="offline",
                            rank=1,
                        ).model_dump(mode="json")
                    ],
                    "kind": "search",
                },
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode(),
            media_type="application/vnd.deepresearch.baseline-work+json",
        )
        await handlers.fetch(
            cast(
                "Any",
                {
                    "run_id": context.run_id,
                    "baseline_work_artifact_ids": (search_work.artifact_id,),
                },
            )
        )

    assert observed_deadlines == [253.0]
    assert len(context.audit.provider_calls) == 1
    assert context.audit.provider_calls[0].outcome_code == "SUCCESS"


async def test_usage_observer_mismatch_settles_inner_usage_once_then_fails() -> None:
    observed = _usage(tokens=7)
    carried = _usage(tokens=9)

    class Observer:
        provider_id = "observer"

        def __init__(self) -> None:
            self.values = [None, observed]

        def consume_invocation_usage(self) -> ResourceUsage | None:
            return self.values.pop(0)

    context = _runtime_context(ticks=iter((1.0, 2.0)))
    handlers = _bare_handlers()

    with pytest.raises(UsageIntegrityError):
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=Observer(),
            provider_id="observer",
            model_id="model",
            operation="model",
            node="Planner",
            operation_id="mismatch",
            invoke=lambda: _async_value(object()),
            result_usage=lambda _result: carried,
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    assert context.budget_accountant.snapshot().used_tokens == 7


async def _async_value(value: object) -> object:
    return value


async def test_recorder_failure_settles_observed_usage_and_preserves_exception_object() -> None:
    observed = _usage(tokens=5)
    failure = RuntimeError("recorder storage failed")

    class Observer:
        provider_id = "observer"

        def __init__(self) -> None:
            self.values = [None, observed]

        def consume_invocation_usage(self) -> ResourceUsage | None:
            return self.values.pop(0)

    async def fail() -> object:
        raise failure

    context = _runtime_context(ticks=iter((1.0, 2.0)))
    with pytest.raises(RuntimeError) as caught:
        await cast("Any", _bare_handlers())._invoke_metered(
            context=context,
            provider=Observer(),
            provider_id="observer",
            model_id="model",
            operation="model",
            node="Planner",
            operation_id="recorder-failure",
            invoke=fail,
            result_usage=lambda _result: None,
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    assert caught.value is failure
    assert context.budget_accountant.snapshot().used_tokens == 5


async def test_hard_primary_is_not_masked_by_observer_cleanup_failure() -> None:
    primary = MemoryError("hard primary")
    cleanup = RuntimeError("observer cleanup failed")

    class Observer:
        provider_id = "observer"

        def __init__(self) -> None:
            self.calls = 0

        def consume_invocation_usage(self) -> ResourceUsage | None:
            self.calls += 1
            if self.calls == 1:
                return None
            raise cleanup

    async def fail() -> object:
        raise primary

    context = _runtime_context(ticks=iter((1.0,)))
    with pytest.raises(MemoryError) as caught:
        await cast("Any", _bare_handlers())._invoke_metered(
            context=context,
            provider=Observer(),
            provider_id="observer",
            model_id="model",
            operation="model",
            node="Planner",
            operation_id="hard-primary",
            invoke=fail,
            result_usage=lambda _result: None,
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    assert caught.value is primary
    assert context.budget_accountant.snapshot().used_tokens == 0


async def test_untyped_usage_measures_injected_elapsed_time() -> None:
    context = _runtime_context(ticks=iter((0.0, 4.0, 6.5)))
    result, usage = await cast("Any", _bare_handlers())._invoke_metered(
        context=context,
        provider=object(),
        provider_id="fallback",
        model_id=None,
        operation="search",
        node="Tool",
        operation_id="elapsed",
        invoke=lambda: _async_value("ok"),
        result_usage=lambda _result: None,
        fallback_usage=ResourceUsage.zero(cost_known=True).model_copy(update={"search_calls": 1}),
        search_calls=1,
    )

    assert result == "ok"
    assert usage.wall_seconds == 2.5
    assert context.budget_accountant.snapshot().used_wall_seconds == 2.5


async def test_model_token_reservation_stops_provider_before_hard_ceiling() -> None:
    low_budget = RunBudget.preset("low").model_copy(update={"max_total_tokens": 5})
    context = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=low_budget),
    )
    called = False

    async def invoke() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(BudgetExceeded):
        await cast("Any", _bare_handlers())._invoke_metered(
            context=context,
            provider=object(),
            provider_id="model",
            model_id="model-v1",
            operation="model",
            node="Writer",
            operation_id="hard-token-ceiling",
            invoke=invoke,
            result_usage=lambda _result: None,
            fallback_usage=ResourceUsage.zero(cost_known=True),
            tokens=6,
        )

    assert called is False
    assert context.budget_accountant.snapshot().used_tokens == 0


async def test_cost_preflight_prices_conservative_usage_before_provider_call() -> None:
    class LinearCostResolver:
        def resolve_cost(self, *, usage: ResourceUsage, **_kwargs: object) -> Decimal:
            return Decimal(usage.total_tokens) / Decimal(100)

    low_cost = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": Decimal("0.01")}
    )
    context = _runtime_context(ticks=iter((1.0, 2.0)), run_config=config(budget=low_cost))
    handlers = _bare_handlers()
    handlers.usage_cost_resolver = LinearCostResolver()
    called = False

    async def invoke() -> object:
        nonlocal called
        called = True
        return object()

    with pytest.raises(BudgetExceeded):
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="priced-model",
            model_id="model-v1",
            operation="model",
            node="Writer",
            operation_id="priced-preflight",
            invoke=invoke,
            result_usage=lambda _result: _usage(tokens=100),
            fallback_usage=ResourceUsage.zero(cost_known=True),
            tokens=100,
        )

    assert called is False


async def test_model_preflight_prices_output_tokens_before_provider_call(
    tmp_path: Path,
) -> None:
    class OutputCostResolver:
        def resolve_cost(self, *, usage: ResourceUsage, **_kwargs: object) -> Decimal:
            return Decimal(usage.output_tokens) / Decimal(100)

    handlers = _owned_handlers(tmp_path)
    handlers.usage_cost_resolver = OutputCostResolver()
    request_value = _cache_test_model_request()
    capped = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": Decimal("0.01")}
    )
    context = _runtime_context(ticks=iter((1.0, 2.0)), run_config=config(budget=capped))
    _bind_handler_runtime(handlers, context)
    provider = cast("Any", handlers.initial_plan_generator.model)._inner

    with pytest.raises(BudgetExceeded):
        await _invoke_cached_test_model(handlers, request_value)

    assert provider.requests == []


@pytest.mark.parametrize("dimension", ["retries", "wall_seconds", "cost_usd"])
async def test_success_overrun_is_charged_but_not_published(
    dimension: str,
) -> None:
    class ActualCostResolver:
        def resolve_cost(self, *, outcome: str, **_kwargs: object) -> Decimal:
            return Decimal(0) if outcome == "preflight" else Decimal("0.02")

    updates: dict[str, object]
    usage_value = ResourceUsage.zero(cost_known=True)
    if dimension == "retries":
        updates = {"max_retries": 0}
        usage_value = usage_value.model_copy(update={"retries": 1})
    elif dimension == "wall_seconds":
        updates = {"max_wall_time_seconds": 1}
        usage_value = usage_value.model_copy(update={"wall_seconds": 2.0})
    else:
        updates = {"max_cost_usd": Decimal("0.01")}
    current = config(budget=RunBudget.preset("low").model_copy(update=updates))
    context = _runtime_context(ticks=iter((1.0, 2.0)), run_config=current)
    handlers = _bare_handlers()
    if dimension == "cost_usd":
        handlers.usage_cost_resolver = ActualCostResolver()
    published = False

    async def invoke() -> object:
        nonlocal published
        published = True
        return object()

    with pytest.raises(BudgetExceeded):
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="overrun",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id=f"overrun-{dimension}",
            invoke=invoke,
            result_usage=lambda _result: usage_value,
            fallback_usage=ResourceUsage.zero(cost_known=True),
        )

    snapshot = context.budget_accountant.snapshot()
    assert published is True
    assert snapshot.reserved_retries == 0
    if dimension == "retries":
        assert snapshot.used_retries == 1
    elif dimension == "wall_seconds":
        assert snapshot.used_wall_seconds == 2.0
    else:
        assert snapshot.used_cost_usd == Decimal("0.02")


async def test_success_at_exact_limit_may_publish_once() -> None:
    current = config(
        budget=RunBudget.preset("low").model_copy(update={"max_retries": 1})
    )
    context = _runtime_context(ticks=iter((1.0, 2.0)), run_config=current)
    actual = ResourceUsage.zero(cost_known=True).model_copy(update={"retries": 1})

    result, settled = await cast("Any", _bare_handlers())._invoke_metered(
        context=context,
        provider=object(),
        provider_id="exact-limit",
        model_id=None,
        operation="parse",
        node="Tool",
        operation_id="exact-limit",
        invoke=lambda: _async_value("published"),
        result_usage=lambda _result: actual,
        fallback_usage=ResourceUsage.zero(cost_known=True),
    )

    assert result == "published"
    assert settled.retries == 1
    with pytest.raises(BudgetExceeded):
        context.budget_accountant.reserve(
            ResourceEstimate(retries=1, cost_usd=Decimal(0)),
            node="Tool",
            idempotency_key="after-exact-limit",
        )


async def test_failed_usage_is_costed_with_the_real_outcome_code() -> None:
    class OutcomeCostResolver:
        def resolve_cost(
            self,
            *,
            outcome: str,
            **_kwargs: object,
        ) -> Decimal:
            return Decimal("0.02") if outcome == "RATE_LIMITED" else Decimal(0)

    failure_usage = _usage(tokens=5)
    context = _runtime_context(ticks=iter((1.0, 2.0)))
    handlers = _bare_handlers()
    handlers.usage_cost_resolver = OutcomeCostResolver()

    async def fail() -> object:
        from deepresearch.providers import ProviderError

        raise ProviderError(
            code="RATE_LIMITED",
            provider="provider",
            operation="model",
            public_message="rate limited",
            retryable=True,
            usage=failure_usage,
        )

    with pytest.raises(Exception, match="rate limited"):
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="provider",
            model_id="model-v1",
            operation="model",
            node="Planner",
            operation_id="failed-outcome",
            invoke=fail,
            result_usage=lambda _result: None,
            fallback_usage=ResourceUsage.zero(cost_known=True),
            tokens=5,
        )

    assert context.budget_accountant.snapshot().used_cost_usd == Decimal("0.02")


async def test_failed_untyped_call_reads_post_call_last_usage_inside_provider_lock() -> None:
    exact = ResourceUsage.zero(cost_known=True).model_copy(
        update={"search_calls": 1, "retries": 2, "wall_seconds": 0.5}
    )

    class FailingProvider:
        provider_id = "untyped-search"
        last_usage = ResourceUsage.zero(cost_known=True)

        async def invoke(self) -> object:
            self.last_usage = exact
            raise RuntimeError("search failed")

    provider = FailingProvider()
    context = _runtime_context(ticks=iter((1.0, 2.0)))

    with pytest.raises(RuntimeError, match="search failed"):
        await cast("Any", _bare_handlers())._invoke_metered(
            context=context,
            provider=provider,
            provider_id=provider.provider_id,
            model_id=None,
            operation="search",
            node="Tool",
            operation_id="failed-last-usage",
            invoke=provider.invoke,
            result_usage=lambda _result: None,
            fallback_usage=ResourceUsage.zero(cost_known=True).model_copy(
                update={"search_calls": 1}
            ),
            search_calls=1,
        )

    snapshot = context.budget_accountant.snapshot()
    assert snapshot.used_retries == 2
    assert snapshot.used_search_calls == 1


async def test_post_success_observer_error_settles_carried_usage_and_stays_primary() -> None:
    carried = _usage(tokens=5)
    failure = OSError("observer read failed")

    class Observer:
        provider_id = "observer"

        def __init__(self) -> None:
            self.calls = 0

        def consume_invocation_usage(self) -> ResourceUsage | None:
            self.calls += 1
            if self.calls == 1:
                return None
            raise failure

    context = _runtime_context(ticks=iter((1.0, 2.0)))
    with pytest.raises(OSError) as caught:
        await cast("Any", _bare_handlers())._invoke_metered(
            context=context,
            provider=Observer(),
            provider_id="observer",
            model_id="model-v1",
            operation="model",
            node="Planner",
            operation_id="observer-read-failure",
            invoke=lambda: _async_value(object()),
            result_usage=lambda _result: carried,
            fallback_usage=ResourceUsage.zero(cost_known=True),
            tokens=5,
        )

    assert caught.value is failure
    snapshot = context.budget_accountant.snapshot()
    assert snapshot.used_tokens == 5
    assert snapshot.reserved_tokens == 0


def _cache_test_model_request() -> ModelRequest:
    return ModelRequest(
        model_id="offline-model-v1",
        messages=(ModelMessage(role="user", content="cache publication boundary"),),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=64,
        prompt_version="fixed-planner-v1",
        system_prompt_hash="a" * 64,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )


def _cache_test_model_key(request_value: ModelRequest) -> ModelCacheKey:
    return ModelCacheKey(
        provider_id="offline-model",
        endpoint_type="complete",
        model_id=request_value.model_id,
        prompt_version=request_value.prompt_version,
        system_prompt_hash=request_value.system_prompt_hash,
        tool_schema_hash=request_value.tool_schema_hash,
        output_schema_hash=request_value.output_schema_hash,
        temperature=request_value.temperature,
        seed=request_value.seed,
        canonical_request_hash=hashlib.sha256(
            json.dumps(
                request_value.model_dump(mode="json"),
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )


def _bind_handler_runtime(
    handlers: BaselineNodeHandlers,
    context: BaselineRuntimeContext,
    *,
    graph_node: str = "Plan",
) -> None:
    context.audit.begin(graph_node=graph_node, node_attempt=1)
    cast("Any", handlers)._runtime = lambda: context


async def _invoke_cached_test_model(
    handlers: BaselineNodeHandlers,
    request_value: ModelRequest,
) -> ModelResult[str]:
    proxy = cast("Any", handlers.initial_plan_generator.model)
    provider = proxy._inner
    return await cast("Any", handlers)._cached_model_call(
        provider=provider,
        request=request_value,
        output_schema=None,
        invoke=lambda: provider.complete(
            request_value,
            deadline=999.0,
            cancellation_token=CancellationToken(),
        ),
    )


@pytest.mark.parametrize(
    "corruption",
    [
        "thread",
        "receipt-key",
        "budget-node",
        "decision",
        "usage",
        "result",
        "release-success",
        "empty-success",
        "success-cache-hit",
        "cache-hit-charge",
        "cache-hit-empty",
    ],
)
async def test_provider_receipt_closure_rejects_incoherent_facts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    unpriced = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    context = _runtime_context(
        ticks=iter((1.0, 1.1, 1.2, 1.3)),
        run_config=unpriced,
        run_id="run-provider-closure",
        thread_id="thread-provider-closure",
    )
    _bind_handler_runtime(handlers, context)
    await _invoke_cached_test_model(handlers, _cache_test_model_request())
    receipt_id = context.audit.provider_receipt_ids[0]
    result_ids = tuple(context.audit.result_artifact_ids)
    receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers.artifact_store,
        receipt_id,
    )
    assert receipt is not None
    tampered = receipt.model_dump(mode="json")
    payload = cast("dict[str, object]", tampered["payload"])
    budget_replay = cast("dict[str, object]", payload["budget_replay"])
    if corruption == "thread":
        tampered["thread_id"] = "other-thread"
    elif corruption == "receipt-key":
        tampered["receipt_key"] = "call:wrong:1"
    elif corruption == "budget-node":
        budget_replay["budget_node"] = "Tool"
    elif corruption == "decision":
        budget_replay["decision"] = "observe"
    elif corruption == "usage":
        budget_replay["actual"] = _usage(tokens=21).model_dump(mode="json")
    elif corruption == "release-success":
        budget_replay["actual"] = None
        budget_replay["decision"] = "release"
    elif corruption == "empty-success":
        payload["result_artifact_ids"] = []
    elif corruption == "success-cache-hit":
        cast("dict[str, object]", payload["record"])["cache_hit"] = True
    elif corruption in {"cache-hit-charge", "cache-hit-empty"}:
        record = cast("dict[str, object]", payload["record"])
        record["cache_hit"] = True
        record["outcome_code"] = "CACHE_HIT"
        if corruption == "cache-hit-empty":
            budget_replay["decision"] = "observe"
            payload["result_artifact_ids"] = []
    else:
        unrelated = handlers.artifact_store.put_bytes(
            b"{}",
            media_type="application/json",
        )
        payload["result_artifact_ids"] = [unrelated.artifact_id]
    encoded = json.dumps(
        tampered,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()
    original_get = handlers.artifact_store.get_bytes

    def tampered_get(artifact_id: str) -> bytes:
        return encoded if artifact_id == receipt_id else original_get(artifact_id)

    monkeypatch.setattr(handlers.artifact_store, "get_bytes", tampered_get)

    with pytest.raises(ArtifactIntegrityError, match="provider|receipt|closure"):
        cast("Any", baseline_graph_module)._provider_calls_from_receipt(
            cast("Any", handlers)._audit_composition,
            run_id=context.run_id,
            thread_id=context.thread_id,
            owning_node="Plan",
            owning_output_artifact_ids=result_ids,
            receipt_ids=[receipt_id],
        )


def _tamper_cached_artifact_read(
    handlers: BaselineNodeHandlers,
    monkeypatch: pytest.MonkeyPatch,
    *,
    artifact_id: str,
    payload: object,
) -> None:
    original_get = handlers.artifact_store.get_bytes
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()

    def get_bytes(current_artifact_id: str) -> bytes:
        if current_artifact_id == artifact_id:
            return encoded
        return original_get(current_artifact_id)

    monkeypatch.setattr(handlers.artifact_store, "get_bytes", get_bytes)


async def test_success_receipt_is_published_only_after_result_artifact(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
    )
    _bind_handler_runtime(handlers, context)
    captured_result_ids: list[tuple[str, ...]] = []
    original_put = cast("Any", baseline_graph_module)._put_audit_receipt

    def capture_put(*args: object, **kwargs: object) -> str:
        if kwargs.get("kind") == "provider-call":
            payload = cast("dict[str, object]", kwargs["payload"])
            record = cast("dict[str, object]", payload["record"])
            if record["outcome_code"] == "SUCCESS":
                captured_result_ids.append(
                    tuple(cast("list[str]", payload["result_artifact_ids"]))
                )
        return cast("str", original_put(*args, **kwargs))

    monkeypatch.setattr(baseline_graph_module, "_put_audit_receipt", capture_put)

    await _invoke_cached_test_model(handlers, _cache_test_model_request())

    assert len(captured_result_ids) == 1
    assert len(captured_result_ids[0]) == 1


async def test_crash_before_completion_cache_may_repeat_public_provider(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    proxy = cast("Any", handlers.initial_plan_generator.model)
    provider = proxy._inner
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    original_put = handlers.cache.put_if_absent

    def crash_before_cache(*_args: object, **_kwargs: object) -> None:
        raise MemoryError("crash before completion cache publication")

    monkeypatch.setattr(handlers.cache, "put_if_absent", crash_before_cache)
    first_context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
    )
    _bind_handler_runtime(handlers, first_context)
    with pytest.raises(MemoryError, match="before completion cache"):
        await _invoke_cached_test_model(handlers, request_value)

    monkeypatch.setattr(handlers.cache, "put_if_absent", original_put)
    resumed_context = _runtime_context(
        ticks=iter((3.0, 4.0)),
        run_config=config(budget=unpriced),
    )
    _bind_handler_runtime(handlers, resumed_context)
    await _invoke_cached_test_model(handlers, request_value)

    assert len(provider.requests) == 2


async def test_crash_after_completion_cache_reuses_exact_provider_result(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    proxy = cast("Any", handlers.initial_plan_generator.model)
    provider = proxy._inner
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    original_put = handlers.cache.put_if_absent

    def crash_after_cache(key: object, entry: object) -> None:
        original_put(cast("Any", key), cast("Any", entry))
        raise MemoryError("crash after completion cache publication")

    monkeypatch.setattr(handlers.cache, "put_if_absent", crash_after_cache)
    first_context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
    )
    _bind_handler_runtime(handlers, first_context)
    with pytest.raises(MemoryError, match="after completion cache"):
        await _invoke_cached_test_model(handlers, request_value)

    monkeypatch.setattr(handlers.cache, "put_if_absent", original_put)
    resumed_context = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
    )
    _bind_handler_runtime(handlers, resumed_context)
    result = await _invoke_cached_test_model(handlers, request_value)

    assert len(provider.requests) == 1
    assert result.output == json.dumps(_offline_plan(), sort_keys=True)
    assert resumed_context.budget_accountant.snapshot().used_tokens == 20


async def test_completion_cache_ack_without_readable_entry_fails_before_return(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    provider = cast("Any", handlers.initial_plan_generator.model)._inner
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    original_put = handlers.cache.put_if_absent

    def acknowledge_without_store(_key: object, candidate: CacheEntry) -> CacheEntry:
        return candidate

    monkeypatch.setattr(
        handlers.cache,
        "put_if_absent",
        acknowledge_without_store,
    )
    first = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-unreadable-cache",
        thread_id="thread-unreadable-cache",
    )
    _bind_handler_runtime(handlers, first)
    with pytest.raises(CacheIntegrityError):
        await _invoke_cached_test_model(handlers, request_value)
    assert len(provider.requests) == 1

    monkeypatch.setattr(handlers.cache, "put_if_absent", original_put)
    retry = _runtime_context(
        ticks=iter((3.0, 4.0)),
        run_config=config(budget=unpriced),
        run_id=first.run_id,
        thread_id=first.thread_id,
    )
    _bind_handler_runtime(handlers, retry)
    await _invoke_cached_test_model(handlers, request_value)
    assert len(provider.requests) == 2


async def test_result_store_failure_reconciles_settled_pending_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-result-store-failure",
        thread_id="thread-result-store-failure",
    )
    _bind_handler_runtime(handlers, context)
    original_put = handlers.artifact_store.put_bytes

    def fail_model_result(payload: bytes, *, media_type: str):
        if media_type == "application/vnd.deepresearch.model-result+json":
            raise OSError("model result persistence failed")
        return original_put(payload, media_type=media_type)

    monkeypatch.setattr(handlers.artifact_store, "put_bytes", fail_model_result)

    with pytest.raises(OSError, match="model result persistence failed"):
        await _invoke_cached_test_model(handlers, request_value)

    assert context.budget_accountant.snapshot().used_tokens == 20
    assert context.audit.pending_provider_call is None
    assert len(context.audit.provider_calls) == 1
    assert context.audit.provider_calls[0].outcome_code == "INTERNAL_ERROR"
    assert len(context.audit.provider_receipt_ids) == 1
    assert context.audit.result_artifact_ids == []


async def test_model_result_projection_failure_reconciles_pending_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    failure = OSError("model result projection failed")

    def fail_projection(self: ModelResult[object], **_kwargs: object) -> object:
        del self
        raise failure

    monkeypatch.setattr(ModelResult, "model_dump", fail_projection)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-model-projection-failure",
        thread_id="thread-model-projection-failure",
    )
    _bind_handler_runtime(handlers, context)

    with pytest.raises(OSError) as caught:
        await _invoke_cached_test_model(handlers, _cache_test_model_request())

    assert caught.value is failure
    assert context.audit.pending_provider_call is None
    assert context.budget_accountant.snapshot().used_tokens == 20
    assert len(context.audit.provider_calls) == 1
    assert context.audit.provider_calls[0].outcome_code == "INTERNAL_ERROR"
    assert len(context.audit.provider_receipt_ids) == 1


async def test_hard_model_result_projection_preserves_primary_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    primary = MemoryError("hard model result projection failure")

    def fail_projection(self: ModelResult[object], **_kwargs: object) -> object:
        del self
        raise primary

    monkeypatch.setattr(ModelResult, "model_dump", fail_projection)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-hard-model-projection",
        thread_id="thread-hard-model-projection",
    )
    _bind_handler_runtime(handlers, context)

    with pytest.raises(MemoryError) as caught:
        await _invoke_cached_test_model(handlers, _cache_test_model_request())

    assert caught.value is primary
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.pending_provider_call is not None


async def test_search_result_projection_failure_reconciles_pending_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    failure = OSError("search result projection failed")

    async def fixed_queries(_context: object, _invoke: object) -> tuple[str, ...]:
        return ("offline baseline query",)

    monkeypatch.setattr(handlers, "_isolated_planner_call", fixed_queries)
    plan = handlers.artifact_store.put_bytes(
        json.dumps(
            _offline_plan(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.plan+json",
    )
    state_value = cast(
        "Any",
        {
            "run_id": "run-search-projection-failure",
            "plan_artifact_id": plan.artifact_id,
            "active_subquestion_id": "sq-offline",
            "baseline_work_artifact_ids": (),
        },
    )

    def fail_projection(self: SearchHit, **_kwargs: object) -> object:
        del self
        raise failure

    monkeypatch.setattr(SearchHit, "model_dump", fail_projection)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=state_value["run_id"],
        thread_id="thread-search-projection-failure",
    )
    _bind_handler_runtime(handlers, context, graph_node="Search")

    with pytest.raises(OSError) as caught:
        await handlers.search(state_value)

    assert caught.value is failure
    assert context.audit.pending_provider_call is None
    assert context.budget_accountant.snapshot().used_search_calls == 1
    assert len(context.audit.provider_calls) == 1
    assert context.audit.provider_calls[0].outcome_code == "INTERNAL_ERROR"
    assert len(context.audit.provider_receipt_ids) == 1


async def test_success_receipt_write_failure_reconciles_attempt_one(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-receipt-write-failure",
        thread_id="thread-receipt-write-failure",
    )
    _bind_handler_runtime(handlers, context)
    original_put = cast("Any", baseline_graph_module)._put_audit_receipt
    failures = 0

    def fail_first_provider_receipt(*args: object, **kwargs: object) -> str:
        nonlocal failures
        if kwargs.get("kind") == "provider-call" and failures == 0:
            failures += 1
            raise OSError("provider receipt persistence failed")
        return cast("str", original_put(*args, **kwargs))

    monkeypatch.setattr(
        baseline_graph_module,
        "_put_audit_receipt",
        fail_first_provider_receipt,
    )

    with pytest.raises(OSError, match="provider receipt persistence failed"):
        await _invoke_cached_test_model(handlers, request_value)

    assert context.audit.pending_provider_call is None
    assert len(context.audit.provider_calls) == 1
    assert context.audit.provider_calls[0].attempt == 1
    assert context.audit.provider_calls[0].outcome_code == "INTERNAL_ERROR"
    assert len(context.audit.provider_receipt_ids) == 1
    receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers.artifact_store,
        context.audit.provider_receipt_ids[0],
    )
    assert receipt.payload["result_artifact_ids"] == context.audit.result_artifact_ids


async def test_stale_pending_rejected_before_next_provider_or_budget_work() -> None:
    context = _runtime_context(ticks=iter((1.0, 2.0, 3.0, 4.0)))
    context.audit.begin(graph_node="Plan", node_attempt=1)
    handlers = _bare_handlers()
    await cast("Any", handlers)._invoke_metered(
        context=context,
        provider=object(),
        provider_id="first-provider",
        model_id=None,
        operation="parse",
        node="Tool",
        operation_id="first-pending",
        invoke=lambda: _async_value("first"),
        result_usage=lambda _result: ResourceUsage.zero(cost_known=True),
        fallback_usage=ResourceUsage.zero(cost_known=True),
        call_fields={"operation": "parse"},
    )
    assert context.audit.pending_provider_call is not None
    budget_before = context.budget_accountant.snapshot()
    provider_calls = 0

    async def second_provider() -> object:
        nonlocal provider_calls
        provider_calls += 1
        return object()

    with pytest.raises(ArtifactIntegrityError):
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="second-provider",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id="second-pending",
            invoke=second_provider,
            result_usage=lambda _result: ResourceUsage.zero(cost_known=True),
            fallback_usage=ResourceUsage.zero(cost_known=True),
            call_fields={"operation": "parse"},
        )

    assert provider_calls == 0
    assert context.budget_accountant.snapshot() == budget_before


@pytest.mark.parametrize("retries", [0, 1])
async def test_usage_bearing_provider_failure_with_receipt_error_stays_fail_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    retries: int,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=f"run-failed-receipt-{retries}",
        thread_id=f"thread-failed-receipt-{retries}",
    )
    _bind_handler_runtime(handlers, context, graph_node="ParseAndNormalize")
    usage = ResourceUsage.zero(cost_known=False).model_copy(
        update={"retries": retries, "wall_seconds": 0.5}
    )
    primary = ProviderError(
        code="UPSTREAM_5XX",
        provider="failing-parser",
        operation="parse",
        public_message="upstream parser failed",
        retryable=True,
        usage=usage,
    )

    async def fail() -> object:
        raise primary

    def fail_provider_receipt(*_args: object, **_kwargs: object) -> str:
        raise OSError("provider failure receipt write failed")

    monkeypatch.setattr(
        baseline_graph_module,
        "_put_audit_receipt",
        fail_provider_receipt,
    )
    call_fields = {
        "operation": "parse",
        "provider_id": "failing-parser",
        "endpoint_type": "parse",
        "request_sha256": "a" * 64,
        "snapshot_id": "parse-source-v1-" + "b" * 64,
        "complete_parameters": {
            "raw_content_hash": "c" * 64,
            "parser_id": "failing-parser",
            "parser_version": "v1",
            "normalization_version": "baseline-normalization-v1",
        },
    }

    with pytest.raises(ProviderError) as caught:
        await cast("Any", handlers)._invoke_metered(
            context=context,
            provider=object(),
            provider_id="failing-parser",
            model_id=None,
            operation="parse",
            node="Tool",
            operation_id=f"failed-provider-receipt-{retries}",
            invoke=fail,
            result_usage=lambda _result: None,
            fallback_usage=ResourceUsage.zero(cost_known=False),
            call_fields=call_fields,
        )

    assert caught.value is primary
    assert context.budget_accountant.snapshot().used_retries == retries
    assert context.audit.provider_calls == []
    assert context.audit.provider_receipt_ids == []
    assert context.audit.pending_provider_call is not None


def _estimated_handlers(tmp_path: Path) -> BaselineNodeHandlers:
    handlers = _owned_handlers(tmp_path)
    snapshot = PricingSnapshot(
        snapshot_id="offline-model-pricing-v1",
        provider_id="offline-model",
        endpoint_type="complete",
        model_id="offline-model-v1",
        effective_at=datetime(2026, 9, 1, tzinfo=UTC),
        currency="USD",
        input_tokens_per_million_usd=Decimal(1),
        output_tokens_per_million_usd=Decimal(2),
        cached_tokens_per_million_usd=Decimal("0.5"),
        reasoning_tokens_per_million_usd=Decimal(3),
    )

    class SnapshotCostResolver:
        def resolve_cost(
            self,
            *,
            outcome: str,
            usage: ResourceUsage,
            **_kwargs: object,
        ) -> Decimal:
            if outcome == "CACHE_HIT":
                return Decimal(0)
            return CostCalculator.estimate(usage, snapshot).total_usd

    handlers.usage_cost_resolver = SnapshotCostResolver()
    handlers.pricing_status = "estimated"
    handlers.pricing_snapshots = (snapshot,)
    cast("Any", handlers)._audit_composition = cast(
        "Any", handlers._audit_composition
    ).__class__(
        **{
            **cast("Any", handlers._audit_composition).__dict__,
            "pricing_status": "estimated",
            "pricing_snapshots": (snapshot,),
        }
    )
    return handlers


async def test_cross_run_estimated_model_cache_hit_observes_zero_cost(
    tmp_path: Path,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _estimated_handlers(tmp_path)
    request_value = _cache_test_model_request()
    provider = cast("Any", handlers.initial_plan_generator.model)._inner
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_id="run-cache-producer",
        thread_id="thread-cache-producer",
    )
    _bind_handler_runtime(handlers, producer)
    await _invoke_cached_test_model(handlers, request_value)
    producer_record = producer.audit.provider_calls[-1]
    producer_receipt_id = producer.audit.provider_receipt_ids[-1]
    assert producer_record.usage.cost_usd is not None
    assert producer_record.usage.cost_usd > 0

    consumer = _runtime_context(
        ticks=iter(()),
        run_id="run-cache-consumer",
        thread_id="thread-cache-consumer",
    )
    _bind_handler_runtime(handlers, consumer)
    result = await _invoke_cached_test_model(handlers, request_value)

    assert result.output == json.dumps(_offline_plan(), sort_keys=True)
    assert len(provider.requests) == 1
    assert consumer.budget_accountant.snapshot().used_cost_usd == Decimal(0)
    assert consumer.budget_accountant.snapshot().last_observed_usage.total_tokens == 20
    current = consumer.audit.provider_calls[-1]
    assert current.cache_hit is True
    assert current.usage.total_tokens == 20
    assert current.usage.cost_usd == Decimal(0)
    assert consumer.audit.provider_receipt_ids[-1] != producer_receipt_id
    receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers.artifact_store,
        consumer.audit.provider_receipt_ids[-1],
    )
    assert receipt.payload["budget_replay"]["decision"] == "observe"
    assert receipt.payload["budget_replay"]["actual"] == current.usage.model_dump(
        mode="json"
    )


async def test_same_run_estimated_model_cache_resume_reuses_positive_receipt(
    tmp_path: Path,
) -> None:
    handlers = _estimated_handlers(tmp_path)
    request_value = _cache_test_model_request()
    provider = cast("Any", handlers.initial_plan_generator.model)._inner
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_id="run-cache-same",
        thread_id="thread-cache-same",
    )
    _bind_handler_runtime(handlers, producer)
    await _invoke_cached_test_model(handlers, request_value)
    original_call = producer.audit.provider_calls[-1]
    original_receipt_id = producer.audit.provider_receipt_ids[-1]

    resumed = _runtime_context(
        ticks=iter(()),
        run_id="run-cache-same",
        thread_id="thread-cache-same",
    )
    _bind_handler_runtime(handlers, resumed)
    await _invoke_cached_test_model(handlers, request_value)

    assert len(provider.requests) == 1
    assert resumed.audit.provider_calls == [original_call]
    assert resumed.audit.provider_receipt_ids == [original_receipt_id]
    assert resumed.audit.provider_calls[0].cache_hit is False
    assert resumed.budget_accountant.snapshot().used_cost_usd == (
        original_call.usage.cost_usd
    )


async def test_completion_cache_records_exact_producer_provenance(
    tmp_path: Path,
) -> None:
    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-cache-provenance",
        thread_id="thread-cache-provenance",
    )
    _bind_handler_runtime(handlers, producer)
    await _invoke_cached_test_model(handlers, request_value)

    entry = handlers.cache.get(_cache_test_model_key(request_value))
    assert entry is not None
    assert set(entry.metadata) == {
        "artifact_ids",
        "operation_id",
        "outcome",
        "producer_run_id",
        "producer_thread_id",
        "provider_call",
        "provider_call_receipt_id",
        "schema_version",
    }
    assert entry.metadata["producer_run_id"] == producer.run_id
    assert entry.metadata["producer_thread_id"] == producer.thread_id


@pytest.mark.parametrize(
    "corruption",
    [
        "missing-receipt",
        "missing-result",
        "wrong-producer-run",
        "wrong-replay-operation",
        "wrong-artifact-metadata",
        "wrong-call-key",
    ],
)
async def test_corrupt_model_completion_cache_fails_before_budget_mutation(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    key = _cache_test_model_key(request_value)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-cache-integrity",
        thread_id="thread-cache-integrity",
    )
    _bind_handler_runtime(handlers, producer)
    await _invoke_cached_test_model(handlers, request_value)
    entry = handlers.cache.get(key)
    assert entry is not None
    metadata = dict(entry.metadata)
    metadata["producer_run_id"] = producer.run_id
    metadata["producer_thread_id"] = producer.thread_id
    receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers.artifact_store,
        metadata["provider_call_receipt_id"],
    )
    assert receipt is not None
    value_artifact_id = entry.value_artifact_id

    def publish_receipt(
        *,
        payload: dict[str, object],
        receipt_key: str = receipt.receipt_key,
    ) -> str:
        return cast("Any", baseline_graph_module)._put_audit_receipt(
            handlers.artifact_store,
            kind="provider-call",
            run_id=producer.run_id,
            thread_id=producer.thread_id,
            receipt_key=receipt_key,
            payload=payload,
        )

    if corruption == "missing-receipt":
        del metadata["provider_call_receipt_id"]
    elif corruption == "missing-result":
        value_artifact_id = "sha256:" + "f" * 64
        metadata["artifact_ids"] = [value_artifact_id]
        payload = deepcopy(receipt.payload)
        payload["result_artifact_ids"] = [value_artifact_id]
        metadata["provider_call_receipt_id"] = publish_receipt(payload=payload)
    elif corruption == "wrong-producer-run":
        metadata["producer_run_id"] = "run-cache-impostor"
    elif corruption == "wrong-replay-operation":
        payload = deepcopy(receipt.payload)
        replay = cast("dict[str, object]", payload["budget_replay"])
        replay["operation_id"] = "wrong-operation"
        metadata["provider_call_receipt_id"] = publish_receipt(
            payload=payload,
            receipt_key="call:wrong-operation:1",
        )
    elif corruption == "wrong-artifact-metadata":
        metadata["artifact_ids"] = ["sha256:" + "e" * 64]
    else:
        payload = deepcopy(receipt.payload)
        call = cast("dict[str, object]", payload["record"])
        call["request_sha256"] = "d" * 64
        metadata["provider_call"] = call
        metadata["provider_call_receipt_id"] = publish_receipt(payload=payload)
    tampered = CacheEntry.model_validate(
        {
            **entry.model_dump(mode="python"),
            "value_artifact_id": value_artifact_id,
            "metadata": metadata,
        }
    )
    monkeypatch.setattr(handlers.cache, "get", lambda _key: tampered)
    provider = cast("Any", handlers.initial_plan_generator.model)._inner
    provider_calls_before = len(provider.requests)
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed)
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises((CacheIntegrityError, ArtifactIntegrityError)):
        await _invoke_cached_test_model(handlers, request_value)

    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []
    assert resumed.audit.provider_receipt_ids == []
    assert len(provider.requests) == provider_calls_before


@pytest.mark.parametrize("boundary", ["result-artifact", "provider-receipt"])
async def test_hard_cache_validation_failure_preserves_primary_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    boundary: str,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-hard-cache-validation",
        thread_id="thread-hard-cache-validation",
    )
    _bind_handler_runtime(handlers, producer)
    await _invoke_cached_test_model(handlers, request_value)
    entry = handlers.cache.get(_cache_test_model_key(request_value))
    assert entry is not None
    primary = MemoryError(f"hard cache {boundary} failure")
    if boundary == "result-artifact":
        original_get = handlers.artifact_store.get_bytes

        def fail_result_read(artifact_id: str) -> bytes:
            if artifact_id == entry.value_artifact_id:
                raise primary
            return original_get(artifact_id)

        monkeypatch.setattr(handlers.artifact_store, "get_bytes", fail_result_read)
    else:

        def fail_receipt_read(*_args: object, **_kwargs: object) -> object:
            raise primary

        monkeypatch.setattr(
            baseline_graph_module,
            "_load_audit_receipt",
            fail_receipt_read,
        )
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed)
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises(MemoryError) as caught:
        await _invoke_cached_test_model(handlers, request_value)

    assert caught.value is primary
    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []
    assert resumed.audit.provider_receipt_ids == []


@pytest.mark.parametrize("corruption", ["provider", "carried-usage"])
async def test_model_cache_result_must_match_typed_key_and_receipt_usage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    request_value = _cache_test_model_request()
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-model-result-binding",
        thread_id="thread-model-result-binding",
    )
    _bind_handler_runtime(handlers, producer)
    await _invoke_cached_test_model(handlers, request_value)
    entry = handlers.cache.get(_cache_test_model_key(request_value))
    assert entry is not None
    payload = cast(
        "dict[str, object]",
        json.loads(handlers.artifact_store.get_bytes(entry.value_artifact_id)),
    )
    if corruption == "provider":
        payload["provider_id"] = "impostor-model-provider"
    else:
        usage = cast("dict[str, object]", payload["usage"])
        usage["input_tokens"] = 11
        usage["total_tokens"] = 21
    _tamper_cached_artifact_read(
        handlers,
        monkeypatch,
        artifact_id=entry.value_artifact_id,
        payload=payload,
    )
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed)
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises(CacheIntegrityError):
        await _invoke_cached_test_model(handlers, request_value)

    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []
    assert resumed.audit.provider_receipt_ids == []


async def test_embed_cache_result_must_match_exact_input_shape(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    proxy = cast("Any", handlers.ranker.embedder)
    provider = proxy._inner
    texts = ("one evidence passage",)
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-embed-result-binding",
        thread_id="thread-embed-result-binding",
    )
    _bind_handler_runtime(handlers, producer, graph_node="RankEvidence")
    await proxy.embed(
        texts,
        deadline=999.0,
        cancellation_token=CancellationToken(),
    )
    key = EmbedCacheKey(
        model_id=provider.model_id,
        model_revision=provider.model_revision,
        snapshot_sha256=provider.snapshot_sha256,
        normalize_embeddings=True,
        canonical_texts_hash=hashlib.sha256(
            json.dumps(
                {"texts": texts},
                ensure_ascii=False,
                allow_nan=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode()
        ).hexdigest(),
    )
    entry = handlers.cache.get(key)
    assert entry is not None
    _tamper_cached_artifact_read(
        handlers,
        monkeypatch,
        artifact_id=entry.value_artifact_id,
        payload=[[1.0, 0.0], [0.0, 1.0]],
    )
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed, graph_node="RankEvidence")
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises(CacheIntegrityError):
        await proxy.embed(
            texts,
            deadline=999.0,
            cancellation_token=CancellationToken(),
        )

    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []


async def test_search_cache_result_must_respect_key_limit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers = _owned_handlers(tmp_path)
    query = "offline baseline query"

    async def fixed_queries(_context: object, _invoke: object) -> tuple[str, ...]:
        return (query,)

    monkeypatch.setattr(handlers, "_isolated_planner_call", fixed_queries)
    plan = handlers.artifact_store.put_bytes(
        json.dumps(
            _offline_plan(),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.plan+json",
    )
    state_value = cast(
        "Any",
        {
            "run_id": "run-search-result-binding",
            "plan_artifact_id": plan.artifact_id,
            "active_subquestion_id": "sq-offline",
            "baseline_work_artifact_ids": (),
        },
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=state_value["run_id"],
        thread_id="thread-search-result-binding",
    )
    _bind_handler_runtime(handlers, producer, graph_node="Search")
    await handlers.search(state_value)
    key = SearchCacheKey(
        snapshot_id=handlers.search_snapshot_id,
        normalized_query=query,
        provider_id=handlers.search_provider.provider_id,
        endpoint_type="search",
        locale="und",
        complete_parameters={"filters": None, "limit": 10},
        time_policy="frozen",
    )
    entry = handlers.cache.get(key)
    assert entry is not None
    _tamper_cached_artifact_read(
        handlers,
        monkeypatch,
        artifact_id=entry.value_artifact_id,
        payload=[
            SearchHit(
                url=f"https://limit-{rank}.example.test/doc",
                title=f"Limit source {rank}",
                snippet="offline",
                rank=rank,
            ).model_dump(mode="json")
            for rank in range(1, 12)
        ],
    )
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed, graph_node="Search")
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises(CacheIntegrityError):
        await handlers.search(state_value)

    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []


@pytest.mark.parametrize("corruption", ["requested-url", "content-type"])
async def test_fetch_cache_result_must_match_key_policy(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    url = "https://fetch-binding.example.test/doc"
    work = handlers.artifact_store.put_bytes(
        json.dumps(
            {
                "hits": [
                    SearchHit(
                        url=url,
                        title="Fetch binding",
                        snippet="offline",
                        rank=1,
                    ).model_dump(mode="json")
                ],
                "kind": "search",
            },
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.baseline-work+json",
    )
    state_value = cast(
        "Any",
        {
            "run_id": "run-fetch-result-binding",
            "baseline_work_artifact_ids": (work.artifact_id,),
        },
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=state_value["run_id"],
        thread_id="thread-fetch-result-binding",
    )
    _bind_handler_runtime(handlers, producer, graph_node="Fetch")
    await handlers.fetch(state_value)
    key = FetchCacheKey(
        snapshot_id=handlers.fetch_snapshot_id,
        canonical_url=url,
        fetch_policy="baseline",
        accepted_content_types=("text/html", "application/pdf"),
    )
    entry = handlers.cache.get(key)
    assert entry is not None
    payload = cast(
        "dict[str, object]",
        json.loads(handlers.artifact_store.get_bytes(entry.value_artifact_id)),
    )
    if corruption == "requested-url":
        payload["requested_url"] = "https://impostor.example.test/doc"
    else:
        payload["content_type"] = "application/json"
        payload["headers"] = {"content-type": "application/json"}
    _tamper_cached_artifact_read(
        handlers,
        monkeypatch,
        artifact_id=entry.value_artifact_id,
        payload=payload,
    )
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed, graph_node="Fetch")
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises(CacheIntegrityError):
        await handlers.fetch(state_value)

    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []


@pytest.mark.parametrize("corruption", ["parser-id", "canonical-url"])
async def test_parse_cache_result_must_match_parser_and_raw_document(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    corruption: str,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    raw = RawDocument(
        requested_url="https://parse-binding.example.test/document",
        final_url="https://parse-binding.example.test/document",
        status=200,
        headers={"content-type": "text/html"},
        content_type="text/html",
        body_bytes=b"parse binding body",
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    work = handlers.artifact_store.put_bytes(
        json.dumps(
            {"documents": [raw.model_dump(mode="json")], "kind": "raw"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.baseline-work+json",
    )
    state_value = cast(
        "Any",
        {
            "run_id": "run-parse-result-binding",
            "baseline_work_artifact_ids": (work.artifact_id,),
        },
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    producer = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=state_value["run_id"],
        thread_id="thread-parse-result-binding",
    )
    _bind_handler_runtime(handlers, producer, graph_node="ParseAndNormalize")
    await handlers.parse_and_normalize(state_value)
    key = ParseCacheKey(
        snapshot_id=cast("Any", baseline_graph_module)._parse_source_snapshot_id(
            fetch_snapshot_id=handlers.fetch_snapshot_id,
            final_url=raw.final_url,
            content_type=raw.content_type,
        ),
        raw_content_hash=hashlib.sha256(raw.body_bytes).hexdigest(),
        parser_id=handlers.parser.parser_id,
        parser_version=handlers.parser.parser_version,
        normalization_version=handlers.normalization_version,
    )
    entry = handlers.cache.get(key)
    assert entry is not None
    payload = cast(
        "dict[str, object]",
        json.loads(handlers.artifact_store.get_bytes(entry.value_artifact_id)),
    )
    if corruption == "parser-id":
        payload["parser_id"] = "impostor-parser"
    else:
        payload["canonical_url"] = "https://impostor.example.test/document"
    _tamper_cached_artifact_read(
        handlers,
        monkeypatch,
        artifact_id=entry.value_artifact_id,
        payload=payload,
    )
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=producer.run_id,
        thread_id=producer.thread_id,
    )
    _bind_handler_runtime(handlers, resumed, graph_node="ParseAndNormalize")
    budget_before = resumed.budget_accountant.snapshot()

    with pytest.raises(CacheIntegrityError):
        await handlers.parse_and_normalize(state_value)

    assert resumed.budget_accountant.snapshot() == budget_before
    assert resumed.audit.provider_calls == []


@pytest.mark.parametrize("corruption", ["parser-id", "canonical-url"])
async def test_parse_provider_result_is_bound_before_success_publication(
    tmp_path: Path,
    corruption: str,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    raw = RawDocument(
        requested_url="https://parse-provider-binding.example.test/document",
        final_url="https://parse-provider-binding.example.test/document",
        status=200,
        headers={"content-type": "text/html"},
        content_type="text/html",
        body_bytes=b"parse provider binding body",
        retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
    )
    original_parse = handlers.parser.parse

    async def corrupt_parse(
        raw_document: RawDocument,
        **kwargs: object,
    ) -> ParsedDocument:
        result = await original_parse(raw_document, **kwargs)
        if corruption == "parser-id":
            return result.model_copy(update={"parser_id": "impostor-parser"})
        return result.model_copy(
            update={"canonical_url": "https://impostor.example.test/document"}
        )

    handlers.parser.parse = cast("Any", corrupt_parse)
    work = handlers.artifact_store.put_bytes(
        json.dumps(
            {"documents": [raw.model_dump(mode="json")], "kind": "raw"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.baseline-work+json",
    )
    state_value = cast(
        "Any",
        {
            "run_id": "run-parse-provider-binding",
            "baseline_work_artifact_ids": (work.artifact_id,),
        },
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=state_value["run_id"],
        thread_id="thread-parse-provider-binding",
    )
    _bind_handler_runtime(handlers, context, graph_node="ParseAndNormalize")
    key = ParseCacheKey(
        snapshot_id=cast("Any", baseline_graph_module)._parse_source_snapshot_id(
            fetch_snapshot_id=handlers.fetch_snapshot_id,
            final_url=raw.final_url,
            content_type=raw.content_type,
        ),
        raw_content_hash=hashlib.sha256(raw.body_bytes).hexdigest(),
        parser_id=handlers.parser.parser_id,
        parser_version=handlers.parser.parser_version,
        normalization_version=handlers.normalization_version,
    )

    with pytest.raises(ProviderError) as caught:
        await handlers.parse_and_normalize(state_value)

    assert caught.value.code == "INVALID_RESPONSE"
    assert handlers.cache.get(key) is None
    assert context.audit.pending_provider_call is None
    assert len(context.audit.provider_calls) == 1
    assert context.audit.provider_calls[0].outcome_code == "INVALID_RESPONSE"


async def test_identical_parse_bodies_use_distinct_operation_ordinals(
    tmp_path: Path,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    parser_calls = 0
    original_parse = handlers.parser.parse

    async def counted_parse(raw_document: RawDocument, **kwargs: object) -> ParsedDocument:
        nonlocal parser_calls
        parser_calls += 1
        return await original_parse(raw_document, **kwargs)

    handlers.parser.parse = cast("Any", counted_parse)
    body = b"identical parse cache body"
    documents = [
        RawDocument(
            requested_url=f"https://parse-{index}.example.test/document",
            final_url="https://canonical-parse.example.test/document",
            status=200,
            headers={"content-type": "text/html"},
            content_type="text/html",
            body_bytes=body,
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        ).model_dump(mode="json")
        for index in (1, 2)
    ]
    raw_work = handlers.artifact_store.put_bytes(
        json.dumps(
            {"documents": documents, "kind": "raw"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.baseline-work+json",
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id="run-parse-ordinal",
        thread_id="thread-parse-ordinal",
    )
    _bind_handler_runtime(handlers, context)
    context.audit.begin(graph_node="ParseAndNormalize", node_attempt=1)
    state_value = cast(
        "Any",
        {
            "run_id": context.run_id,
            "baseline_work_artifact_ids": (raw_work.artifact_id,),
        },
    )

    await handlers.parse_and_normalize(state_value)

    assert parser_calls == 1
    assert [call.cache_hit for call in context.audit.provider_calls] == [False, True]
    assert [call.attempt for call in context.audit.provider_calls] == [1, 2]
    assert len(set(context.audit.provider_receipt_ids)) == 2
    receipts = [
        cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            receipt_id,
        )
        for receipt_id in context.audit.provider_receipt_ids
    ]
    replay_facts = [receipt.payload["budget_replay"] for receipt in receipts]
    assert [fact["decision"] for fact in replay_facts] == ["charge", "observe"]
    assert len({fact["operation_id"] for fact in replay_facts}) == 2


async def test_identical_parse_bodies_with_different_final_urls_use_distinct_entries(
    tmp_path: Path,
) -> None:
    handlers = _owned_handlers(tmp_path)
    parser_calls = 0
    original_parse = handlers.parser.parse

    async def counted_parse(raw_document: RawDocument, **kwargs: object) -> ParsedDocument:
        nonlocal parser_calls
        parser_calls += 1
        return await original_parse(raw_document, **kwargs)

    handlers.parser.parse = cast("Any", counted_parse)
    body = b"identical bytes with distinct canonical identities"
    documents = [
        RawDocument(
            requested_url=f"https://requested-{index}.example.test/document",
            final_url=f"https://canonical-{index}.example.test/document",
            status=200,
            headers={"content-type": "text/html; charset=utf-8"},
            content_type="text/html; charset=utf-8",
            body_bytes=body,
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        ).model_dump(mode="json")
        for index in (1, 2)
    ]
    raw_work = handlers.artifact_store.put_bytes(
        json.dumps(
            {"documents": documents, "kind": "raw"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.baseline-work+json",
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    context = _runtime_context(
        ticks=iter((1.0, 2.0, 3.0, 4.0)),
        run_config=config(budget=unpriced),
        run_id="run-parse-distinct-final",
        thread_id="thread-parse-distinct-final",
    )
    _bind_handler_runtime(handlers, context, graph_node="ParseAndNormalize")
    update = await handlers.parse_and_normalize(
        cast(
            "Any",
            {
                "run_id": context.run_id,
                "baseline_work_artifact_ids": (raw_work.artifact_id,),
            },
        )
    )
    parsed_work_id = cast("tuple[str, ...]", update["baseline_work_artifact_ids"])[-1]
    parsed_work = cast(
        "dict[str, object]",
        json.loads(handlers.artifact_store.get_bytes(parsed_work_id)),
    )
    parsed_documents = cast("list[dict[str, object]]", parsed_work["documents"])

    assert parser_calls == 2
    assert [call.cache_hit for call in context.audit.provider_calls] == [False, False]
    assert len({call.snapshot_id for call in context.audit.provider_calls}) == 2
    assert len(set(cast("list[str]", parsed_work["parsed_artifact_ids"]))) == 2
    assert [item["canonical_url"] for item in parsed_documents] == [
        "https://canonical-1.example.test/document",
        "https://canonical-2.example.test/document",
    ]


async def test_parse_ordinals_are_stable_after_first_completion_cache_crash(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    parser_calls = 0
    original_parse = handlers.parser.parse

    async def counted_parse(raw_document: RawDocument, **kwargs: object) -> ParsedDocument:
        nonlocal parser_calls
        parser_calls += 1
        return await original_parse(raw_document, **kwargs)

    handlers.parser.parse = cast("Any", counted_parse)
    body = b"parse crash ordinal body"
    final_url = "https://canonical-crash.example.test/document"
    documents = [
        RawDocument(
            requested_url=f"https://parse-crash-{index}.example.test/document",
            final_url=final_url,
            status=200,
            headers={"content-type": "text/html"},
            content_type="text/html",
            body_bytes=body,
            retrieved_at=datetime(2026, 9, 1, tzinfo=UTC),
        ).model_dump(mode="json")
        for index in (1, 2)
    ]
    raw_work = handlers.artifact_store.put_bytes(
        json.dumps(
            {"documents": documents, "kind": "raw"},
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode(),
        media_type="application/vnd.deepresearch.baseline-work+json",
    )
    state_value = cast(
        "Any",
        {
            "run_id": "run-parse-cache-crash",
            "baseline_work_artifact_ids": (raw_work.artifact_id,),
        },
    )
    unpriced = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    original_put = handlers.cache.put_if_absent
    crashed = False

    def crash_after_durable_put(key: object, entry: CacheEntry) -> CacheEntry:
        nonlocal crashed
        returned = original_put(cast("Any", key), entry)
        if not crashed:
            crashed = True
            raise MemoryError("crash after first parse completion cache")
        return returned

    monkeypatch.setattr(handlers.cache, "put_if_absent", crash_after_durable_put)
    first = _runtime_context(
        ticks=iter((1.0, 2.0)),
        run_config=config(budget=unpriced),
        run_id=state_value["run_id"],
        thread_id="thread-parse-cache-crash",
    )
    _bind_handler_runtime(handlers, first, graph_node="ParseAndNormalize")
    with pytest.raises(MemoryError, match="first parse completion cache"):
        await handlers.parse_and_normalize(state_value)
    original_receipt_id = first.audit.provider_receipt_ids[0]

    monkeypatch.setattr(handlers.cache, "put_if_absent", original_put)
    resumed = _runtime_context(
        ticks=iter(()),
        run_config=config(budget=unpriced),
        run_id=first.run_id,
        thread_id=first.thread_id,
    )
    _bind_handler_runtime(handlers, resumed, graph_node="ParseAndNormalize")
    await handlers.parse_and_normalize(state_value)

    producer_call = resumed.audit.provider_calls[0]
    assert producer_call.snapshot_id is not None
    parse_key = ParseCacheKey(
        snapshot_id=producer_call.snapshot_id,
        raw_content_hash=hashlib.sha256(body).hexdigest(),
        parser_id=handlers.parser.parser_id,
        parser_version=handlers.parser.parser_version,
        normalization_version=handlers.normalization_version,
    )
    op0 = cast("Any", baseline_graph_module).stable_operation_id(
        run_id=resumed.run_id,
        workflow_id=resumed.config.workflow_id,
        node="ParseAndNormalize",
        logical_input=parse_key.model_dump(mode="json"),
        ordinal=0,
    )
    op1 = cast("Any", baseline_graph_module).stable_operation_id(
        run_id=resumed.run_id,
        workflow_id=resumed.config.workflow_id,
        node="ParseAndNormalize",
        logical_input=parse_key.model_dump(mode="json"),
        ordinal=1,
    )
    receipts = [
        cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            receipt_id,
        )
        for receipt_id in resumed.audit.provider_receipt_ids
    ]
    replay_facts = [receipt.payload["budget_replay"] for receipt in receipts]

    assert parser_calls == 1
    assert [call.cache_hit for call in resumed.audit.provider_calls] == [False, True]
    assert [call.attempt for call in resumed.audit.provider_calls] == [1, 2]
    assert [fact["operation_id"] for fact in replay_facts] == [
        op0,
        f"{op1}:cache-hit",
    ]
    assert [fact["decision"] for fact in replay_facts] == ["charge", "observe"]
    assert resumed.audit.provider_receipt_ids.count(original_receipt_id) == 1
    assert resumed.budget_accountant.snapshot().used_by_node["Tool"] == (
        producer_call.usage
    )


def _offline_plan() -> dict[str, object]:
    return {
        "plan_id": "plan-offline",
        "scope": {
            "included_topics": ["offline baseline"],
            "excluded_topics": [],
            "date_range": None,
            "answer_shape": "brief",
        },
        "subquestions": [
            {
                "id": "sq-offline",
                "question": "What does the offline baseline prove?",
                "rationale_code": "coverage",
                "importance": 1.0,
                "dependencies": [],
                "information_needs": [
                    {"need_id": "need-offline", "text": "offline proof", "importance": 1.0}
                ],
                "evidence_requirements": {
                    "min_independent_sources": 2,
                    "allowed_source_types": ["paper"],
                    "must_include_primary": False,
                    "freshness": None,
                },
                "status": "pending",
            }
        ],
        "created_by_model": "offline-model-v1",
        "prompt_version": "fixed-planner-v1",
    }


def _two_loop_plan() -> dict[str, object]:
    payload = deepcopy(_offline_plan())
    first = cast("dict[str, object]", cast("list[object]", payload["subquestions"])[0])
    first["id"] = "sq-first"
    first_need = cast("dict[str, object]", cast("list[object]", first["information_needs"])[0])
    first_need["need_id"] = "need-first"
    second = deepcopy(first)
    second["id"] = "sq-second"
    second["dependencies"] = ["sq-first"]
    second_need = cast(
        "dict[str, object]",
        cast("list[object]", second["information_needs"])[0],
    )
    second_need["need_id"] = "need-second"
    payload["subquestions"] = [first, second]
    return payload


def _recovery_composition(
    root: Path,
    *,
    invocations: Counter[str],
    persist_entries: Counter[str],
    crash_case: str | None = None,
    probe: dict[str, object] | None = None,
    plan_payload: dict[str, object] | None = None,
) -> BaselineNodeHandlers:
    current_probe = probe if probe is not None else {}
    artifact_store: LocalArtifactStore
    if crash_case == "persist-results":
        artifact_store = CrashOnceManifestStore(root, probe=current_probe)
    else:
        artifact_store = LocalArtifactStore(root)
    evidence_store = LocalEvidenceStore(root)
    unpriced_budget = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": None}
    )

    def boundary(value: str) -> str:
        return value.replace("UNTRUSTED", "BOUNDARY_APPLIED")

    current_plan = _offline_plan() if plan_payload is None else plan_payload
    planner_model = CountingOfflineModel(current_plan, invocations)
    writer_model = CountingOfflineModel(current_plan, invocations)
    planner = FixedPlanner(
        model=planner_model,
        artifact_store=artifact_store,
        budget=unpriced_budget,
        search_depth=1,
        content_boundary=boundary,
    )
    cache: FileCache
    if crash_case is not None and crash_case != "persist-results":
        cache = CrashOnceFileCache(root, case=crash_case, probe=current_probe)
    else:
        cache = FileCache(root)
    handlers = BaselineNodeHandlers(
        initial_plan_generator=planner,
        ranker=SimilarityRanker(CountingOfflineEmbedder(invocations)),
        writer=MarkdownReportWriter(
            evidence_store,
            model=writer_model,
            content_boundary=boundary,
        ),
        search_provider=CountingOfflineSearch(invocations),
        fetcher=CountingOfflineFetcher(invocations),
        parser=CountingOfflineParser(invocations),
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=cache,
        usage_cost_resolver=ZeroCostResolver(cost=None),
        search_snapshot_id="search-snapshot-v1",
        fetch_snapshot_id="fetch-snapshot-v1",
        code_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        provider_profile_configuration_sha256="c" * 64,
        seed_supported=True,
        pricing_status="unknown",
        pricing_snapshots=(),
        replay_parent="offline-parent-manifest",
    )
    original_persist = handlers.persist_results

    async def counted_persist(state: BaselineState) -> object:
        persist_entries["persist-results"] += 1
        return await original_persist(state)

    handlers.persist_results = cast("Any", counted_persist)
    return handlers


def _recovery_config(handlers: BaselineNodeHandlers) -> RunConfig:
    return RunConfig(
        request=request(),
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None}),
        prompt_versions=dict(handlers.prompt_versions),
        ranker_weights_version=handlers.ranker_weights_version,
        seed=0,
    )


async def _run_failure_after_evidence(
    tmp_path: Path,
    *,
    mutate: object | None = None,
    sink: MemoryEventSink | None = None,
    plan_payload: dict[str, object] | None = None,
    prewrite_target: tuple[str, int] | None = None,
) -> tuple[
    object,
    BaselineNodeHandlers,
    Counter[str],
    Counter[str],
    MemoryEventSink,
]:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
        plan_payload=plan_payload,
    )
    if mutate is not None:
        cast("Any", mutate)(handlers)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    )
    if sink is not None:
        event_sink = sink
    elif prewrite_target is not None:
        event_sink = OneShotPreWriteFailureSink(
            target_node=prewrite_target[0],
            target_occurrence=prewrite_target[1],
            invocations=invocations,
        )
    else:
        event_sink = MemoryEventSink()
    result = await runner.run(
        run_id="run-failure-after-evidence",
        thread_id="thread-failure-after-evidence",
        config=_recovery_config(handlers),
        checkpoint=None,
        emit=event_sink,
        cancellation_token=CancellationToken(),
    )
    return result, handlers, invocations, persist_entries, event_sink


async def test_failure_after_evidence_exports_successfully_persisted_artifacts(
    tmp_path: Path,
) -> None:
    def fail_draft(handlers: BaselineNodeHandlers) -> None:
        def fail_render_prompt(**_kwargs: object) -> str:
            raise OSError("draft dependency failed")

        handlers.writer.render_prompt = cast("Any", fail_render_prompt)

    result_value, handlers, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(tmp_path, mutate=fail_draft)
    )
    result = cast("Any", result_value)

    assert result.status == "failed"
    assert result.error_code == "INTERNAL_ERROR"
    assert sum(invocations.values()) == 8
    assert persist_entries == Counter({"persist-results": 1})
    assert result.report_artifact_id is None
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    assert handlers.artifact_store.get_bytes(result.evidence_graph_artifact_id)
    manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    assert manifest.run_id == "run-failure-after-evidence"


async def test_failure_after_evidence_event_readback_routes_once_to_persistence(
    tmp_path: Path,
) -> None:
    sink = OneShotReadbackFailureSink(target_node="DraftReport")
    result_value, handlers, invocations, persist_entries, returned_sink = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    assert returned_sink is sink
    assert sink.failed is True
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert persist_entries == Counter({"persist-results": 1})
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    assert handlers.artifact_store.get_bytes(result.evidence_graph_artifact_id)
    RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )


@pytest.mark.parametrize(
    ("target_node", "target_seq", "keeps_report"),
    [
        ("DraftReport", 10, False),
        ("FinalizeCitations", 11, True),
    ],
)
async def test_prewrite_event_failure_after_evidence_persists_once_at_same_seq(
    tmp_path: Path,
    target_node: str,
    target_seq: int,
    keeps_report: bool,
) -> None:
    sink = OneShotPreWriteFailureSink(target_node=target_node)
    result_value, handlers, invocations, persist_entries, returned_sink = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    assert returned_sink is sink
    assert sink.failed is True
    attempted_success = next(
        event
        for event in sink.attempts
        if event.node == target_node and event.error_code is None
    )
    assert sink.authoritative_reads
    assert sink.authoritative_reads[0] is None
    stored_target_events = [
        event for event in sink.events.values() if event.node == target_node
    ]
    assert len(stored_target_events) == 1
    failure_event = stored_target_events[0]
    assert attempted_success.seq == target_seq
    assert failure_event.seq == attempted_success.seq
    assert failure_event.status == "failed"
    assert failure_event.error_code == "DATA_CORRUPTION"
    assert not any(
        event.node == target_node and event.error_code is None
        for event in sink.events.values()
    )
    assert sink.authoritative_reads[-1] == failure_event

    node_receipts = [
        (artifact_id, receipt)
        for artifact_id in failure_event.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    ]
    assert len(node_receipts) == 1
    failure_receipt_id, failure_receipt = node_receipts[0]
    receipt_state = cast("Any", baseline_graph_module)._state_from_payload(
        failure_receipt.payload["state"]
    )
    execution = NodeExecutionRecord.model_validate_json(
        json.dumps(
            failure_receipt.payload["record"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        strict=True,
    )
    receipt_event = cast("Any", baseline_graph_module)._run_event_from_receipt_payload(
        failure_receipt.payload["event"]
    )
    assert receipt_state["next_event_seq"] == target_seq
    assert receipt_state["error_code"] == "DATA_CORRUPTION"
    assert receipt_state["failed_node"] == target_node
    assert execution.attempt == 1
    assert execution.status == "failed"
    assert execution.error_code == "DATA_CORRUPTION"
    assert execution.output_artifact_ids == receipt_event.artifact_ids
    assert failure_event.artifact_ids == tuple(
        dict.fromkeys((*receipt_event.artifact_ids, failure_receipt_id))
    )
    provider_receipt_ids = failure_receipt.payload["provider_receipt_ids"]
    child_receipt_ids = failure_receipt.payload["child_receipt_ids"]
    assert type(provider_receipt_ids) is list
    assert type(child_receipt_ids) is list
    assert len(provider_receipt_ids) == (1 if target_node == "DraftReport" else 0)
    assert len(set(cast("list[str]", child_receipt_ids))) == len(child_receipt_ids)
    assert set(cast("list[str]", child_receipt_ids)).issubset(
        receipt_state["baseline_work_artifact_ids"]
    )
    terminal_receipts = [
        receipt
        for artifact_id in cast("list[str]", child_receipt_ids)
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "terminal"
    ]
    assert len(terminal_receipts) == 1
    terminal = terminal_receipts[0]
    assert terminal.payload == {
        "error_code": "DATA_CORRUPTION",
        "elapsed_wall_seconds": receipt_state["elapsed_wall_seconds"],
        "finished_at": execution.finished_at.isoformat(),
        "terminal_event_seq": target_seq,
    }
    assert receipt_event.seq == target_seq
    assert receipt_event.timestamp == execution.finished_at
    assert receipt_event.error_code == execution.error_code
    assert receipt_event.usage_delta == attempted_success.usage_delta
    provisional_receipt_ids = [
        artifact_id
        for artifact_id in attempted_success.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    ]
    assert len(provisional_receipt_ids) == 1
    assert provisional_receipt_ids[0] != failure_receipt_id
    assert provisional_receipt_ids[0] not in receipt_state["baseline_work_artifact_ids"]
    provisional_receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers.artifact_store,
        provisional_receipt_ids[0],
    )
    assert provisional_receipt is not None
    provisional_terminal_ids = [
        artifact_id
        for artifact_id in provisional_receipt.payload["child_receipt_ids"]
        if (
            child := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and child.kind == "terminal"
    ]
    assert len(provisional_terminal_ids) == (
        1 if target_node == "FinalizeCitations" else 0
    )
    assert set(provisional_terminal_ids).isdisjoint(child_receipt_ids)
    assert set(provisional_terminal_ids).isdisjoint(
        receipt_state["baseline_work_artifact_ids"]
    )
    for provider_receipt_id in cast("list[str]", provider_receipt_ids):
        provider_receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            provider_receipt_id,
        )
        assert provider_receipt is not None
        assert provider_receipt.kind == "provider-call"
        for result_id in provider_receipt.payload["result_artifact_ids"]:
            assert result_id in execution.output_artifact_ids
            assert result_id in receipt_event.artifact_ids
            assert result_id in failure_event.artifact_ids

    persist_events = [
        event for event in sink.events.values() if event.node == "PersistResults"
    ]
    assert len(persist_events) == 1
    assert persist_events[0].seq == target_seq + 1
    persist_receipt = next(
        receipt
        for artifact_id in persist_events[0].artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    )
    persist_state = cast("Any", baseline_graph_module)._state_from_payload(
        persist_receipt.payload["state"]
    )
    assert set(provisional_terminal_ids).isdisjoint(
        persist_state["baseline_work_artifact_ids"]
    )

    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert invocations["model:baseline-writer-v1"] == 1
    assert persist_entries == Counter({"persist-results": 1})
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    assert (result.report_artifact_id is not None) is keeps_report
    handlers.artifact_store.get_bytes(result.evidence_graph_artifact_id)
    if result.report_artifact_id is not None:
        handlers.artifact_store.get_bytes(result.report_artifact_id)

    manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    executions = [
        execution
        for execution in manifest.node_executions
        if execution.node == target_node
    ]
    assert len(executions) == 1
    assert executions[0].attempt == 1
    assert executions[0].status == "failed"
    assert executions[0].error_code == "DATA_CORRUPTION"
    assert manifest.failure_codes == ("DATA_CORRUPTION",)
    assert manifest.run_event_count == target_seq
    assert set(provisional_terminal_ids).isdisjoint(manifest.artifact_ids)
    assert len(
        [call for call in manifest.provider_calls if call.node == "DraftReport"]
    ) == 1
    assert len(
        [call for call in manifest.provider_calls if call.node == target_node]
    ) == (1 if target_node == "DraftReport" else 0)
    durable_events = [
        sink.events[("run-failure-after-evidence", seq)]
        for seq in range(1, target_seq + 1)
    ]
    assert FilesystemEventSink._bytes(durable_events[-1]) == (
        FilesystemEventSink._bytes(failure_event)
    )
    assert manifest.run_events_sha256 == hashlib.sha256(
        json.dumps(
            [event.model_dump(mode="json") for event in durable_events],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    graph_payload = json.loads(
        handlers.artifact_store.get_bytes(result.evidence_graph_artifact_id)
    )
    assert {
        item["evidence_id"] for item in graph_payload["evidence"]
    } == {item.evidence_id for item in manifest.evidence_hashes}
    assert {item["source_id"] for item in graph_payload["sources"]} == {
        item.source_id for item in manifest.parsed_artifacts
    }


@pytest.mark.parametrize(
    ("target_node", "target_occurrence", "two_loop"),
    [
        ("StoreEvidence", 1, False),
        ("RankEvidence", 1, False),
        ("DecideNext", 2, True),
        ("Search", 2, True),
        ("Fetch", 2, True),
        ("ParseAndNormalize", 2, True),
        ("StoreEvidence", 2, True),
        ("RankEvidence", 2, True),
    ],
)
async def test_prewrite_event_failure_after_merged_evidence_persists_once(
    tmp_path: Path,
    target_node: str,
    target_occurrence: int,
    two_loop: bool,
) -> None:
    result_value, handlers, invocations, persist_entries, returned_sink = (
        await _run_failure_after_evidence(
            tmp_path,
            plan_payload=_two_loop_plan() if two_loop else None,
            prewrite_target=(target_node, target_occurrence),
        )
    )
    result = cast("Any", result_value)
    sink = cast("OneShotPreWriteFailureSink", returned_sink)

    assert sink.failed is True
    assert sink.authoritative_reads[0] is None
    assert sink.provider_calls_at_failure is not None
    assert sum(invocations.values()) == sink.provider_calls_at_failure
    attempted_success = [
        event
        for event in sink.attempts
        if event.node == target_node and event.error_code is None
    ][target_occurrence - 1]
    target_events = [
        event for event in sink.events.values() if event.node == target_node
    ]
    failure_events = [
        event for event in target_events if event.error_code == "DATA_CORRUPTION"
    ]
    assert len(failure_events) == 1
    failure_event = failure_events[0]
    assert failure_event.seq == attempted_success.seq
    assert failure_event.status == "failed"
    assert not any(
        event.seq == attempted_success.seq and event.error_code is None
        for event in target_events
    )

    persist_events = [
        event for event in sink.events.values() if event.node == "PersistResults"
    ]
    assert len(persist_events) == 1
    assert persist_events[0].seq == failure_event.seq + 1
    assert sorted(seq for run_id, seq in sink.events if run_id == failure_event.run_id) == list(
        range(1, persist_events[0].seq + 1)
    )
    assert persist_entries == Counter({"persist-results": 1})

    node_receipts = [
        (artifact_id, receipt)
        for artifact_id in failure_event.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    ]
    assert len(node_receipts) == 1
    _, node_receipt = node_receipts[0]
    receipt_state = cast("Any", baseline_graph_module)._state_from_payload(
        node_receipt.payload["state"]
    )
    execution = NodeExecutionRecord.model_validate_json(
        json.dumps(
            node_receipt.payload["record"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        strict=True,
    )
    receipt_event = cast("Any", baseline_graph_module)._run_event_from_receipt_payload(
        node_receipt.payload["event"]
    )
    assert receipt_state["next_event_seq"] == failure_event.seq
    assert receipt_state["evidence_ids"]
    assert receipt_state["error_code"] == "DATA_CORRUPTION"
    assert receipt_state["failed_node"] == target_node
    assert execution.attempt == target_occurrence
    assert execution.status == "failed"
    assert execution.error_code == "DATA_CORRUPTION"
    assert execution.output_artifact_ids == receipt_event.artifact_ids
    child_receipts = [
        cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            artifact_id,
        )
        for artifact_id in node_receipt.payload["child_receipt_ids"]
    ]
    assert all(receipt is not None for receipt in child_receipts)
    assert len([receipt for receipt in child_receipts if receipt.kind == "terminal"]) == 1
    for artifact_id, receipt in zip(
        node_receipt.payload["child_receipt_ids"],
        child_receipts,
        strict=True,
    ):
        if receipt.kind in {"parsed-artifact", "evidence-hash"}:
            assert artifact_id in execution.output_artifact_ids
            assert artifact_id in receipt_event.artifact_ids

    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert result.report_artifact_id is None
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    graph_payload = json.loads(
        handlers.artifact_store.get_bytes(result.evidence_graph_artifact_id)
    )
    manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    assert graph_payload["evidence"]
    assert manifest.failure_codes == ("DATA_CORRUPTION",)
    assert manifest.run_event_count == failure_event.seq
    target_executions = [
        record for record in manifest.node_executions if record.node == target_node
    ]
    assert len(target_executions) == target_occurrence
    assert target_executions[-1].attempt == target_occurrence
    assert target_executions[-1].status == "failed"
    assert target_executions[-1].error_code == "DATA_CORRUPTION"


async def test_persist_results_prewrite_absence_is_terminal_without_second_emit(
    tmp_path: Path,
) -> None:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    )
    sink = OneShotPreWriteFailureSink(
        target_node="PersistResults",
        invocations=invocations,
    )
    run_id = "run-persist-prewrite-absent"
    thread_id = "thread-persist-prewrite-absent"

    result = await runner.run(
        run_id=run_id,
        thread_id=thread_id,
        config=_recovery_config(handlers),
        checkpoint=None,
        emit=sink,
        cancellation_token=CancellationToken(),
    )

    assert sink.failed is True
    assert sink.authoritative_reads == [None]
    assert sink.provider_calls_at_failure is not None
    assert sum(invocations.values()) == sink.provider_calls_at_failure
    assert persist_entries == Counter({"persist-results": 1})
    persist_attempts = [
        event for event in sink.attempts if event.node == "PersistResults"
    ]
    assert len(persist_attempts) == 1
    attempted_success = persist_attempts[0]
    assert attempted_success.seq == 12
    assert attempted_success.error_code is None
    assert not any(event.node == "PersistResults" for event in sink.events.values())
    assert await sink.get_event(run_id=run_id, seq=12) is None
    provisional_receipt_ids = [
        artifact_id
        for artifact_id in attempted_success.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    ]
    assert len(provisional_receipt_ids) == 1

    saved = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
    )
    assert saved is not None
    values = cast("Any", saved).checkpoint["channel_values"]
    terminal_state = cast("Any", baseline_graph_module).validate_baseline_state(
        {
            field: values[field]
            for field in cast("Any", baseline_graph_module).BaselineState.__annotations__
        }
    )
    assert terminal_state["next_event_seq"] == 12
    assert terminal_state["error_code"] == "PERSIST_RESULTS_FAILED"
    assert terminal_state["failed_node"] == "PersistResults"
    assert provisional_receipt_ids[0] not in terminal_state[
        "baseline_work_artifact_ids"
    ]

    assert result.status == "failed"
    assert result.error_code == "PERSIST_RESULTS_FAILED"
    assert result.report_artifact_id == terminal_state["report_artifact_id"]
    assert (
        result.evidence_graph_artifact_id
        == terminal_state["evidence_graph_artifact_id"]
    )
    assert result.manifest_artifact_id == terminal_state["manifest_artifact_id"]
    assert result.report_artifact_id is not None
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    report_bytes = handlers.artifact_store.get_bytes(result.report_artifact_id)
    graph_bytes = handlers.artifact_store.get_bytes(
        result.evidence_graph_artifact_id
    )
    manifest_bytes = handlers.artifact_store.get_bytes(result.manifest_artifact_id)
    assert report_bytes
    assert json.loads(graph_bytes)["evidence"]
    manifest = RunManifest.model_validate_json(
        manifest_bytes,
        strict=True,
    )
    assert manifest.run_event_count == 11
    assert manifest.failure_codes == ()
    assert [
        sink.events[(run_id, seq)].node for seq in range(1, 12)
    ][-1] == "FinalizeCitations"

    terminal_checkpoint = checkpoint_ref_from_tuple(saved)
    resumed = await runner.run(
        run_id=run_id,
        thread_id=thread_id,
        config=_recovery_config(handlers),
        checkpoint=terminal_checkpoint,
        emit=sink,
        cancellation_token=CancellationToken(),
    )
    assert resumed == result
    assert persist_entries == Counter({"persist-results": 1})
    assert sum(invocations.values()) == sink.provider_calls_at_failure
    assert len(
        [event for event in sink.attempts if event.node == "PersistResults"]
    ) == 1
    assert await sink.get_event(run_id=run_id, seq=12) is None


async def test_persist_results_recovered_exact_is_success_and_resumes_identically(
    tmp_path: Path,
) -> None:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    current_config = _recovery_config(handlers)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    )
    sink = OneShotReadbackFailureSink(target_node="PersistResults")
    run_id = "run-persist-recovered-exact"
    thread_id = "thread-persist-recovered-exact"

    first = await runner.run(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        checkpoint=None,
        emit=sink,
        cancellation_token=CancellationToken(),
    )
    provider_calls_after_first = sum(invocations.values())
    persist_event = await sink.get_event(run_id=run_id, seq=12)
    assert persist_event is not None
    persist_event_bytes = FilesystemEventSink._bytes(persist_event)
    pre_persist_checkpoint: CheckpointRef | None = None
    async for saved in saver.alist(
        {
            "configurable": {
                "thread_id": thread_id,
                "checkpoint_ns": "",
            }
        }
    ):
        values = cast("Any", saved).checkpoint["channel_values"]
        if (
            values.get("next_event_seq") == 12
            and values.get("manifest_artifact_id") is None
        ):
            pre_persist_checkpoint = checkpoint_ref_from_tuple(saved)
            break
    assert pre_persist_checkpoint is not None

    second = await runner.run(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        checkpoint=pre_persist_checkpoint,
        emit=sink,
        cancellation_token=CancellationToken(),
    )

    assert sink.failed is True
    assert first.status == "completed", first.error_code
    assert second.status == "completed", second.error_code
    assert second == first
    assert persist_entries == Counter({"persist-results": 1})
    assert sum(invocations.values()) == provider_calls_after_first == 9
    assert FilesystemEventSink._bytes(
        cast("RunEvent", await sink.get_event(run_id=run_id, seq=12))
    ) == persist_event_bytes
    assert first.report_artifact_id is not None
    assert first.evidence_graph_artifact_id is not None
    assert first.manifest_artifact_id is not None
    handlers.artifact_store.get_bytes(first.report_artifact_id)
    handlers.artifact_store.get_bytes(first.evidence_graph_artifact_id)
    RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(first.manifest_artifact_id),
        strict=True,
    )


@pytest.mark.parametrize(
    ("mode", "expected_code", "keeps_ids"),
    [
        ("conflict", "DATA_CORRUPTION", False),
        ("constructed-corruption", "DATA_CORRUPTION", False),
        ("unreadable", "PERSIST_RESULTS_FAILED", True),
    ],
)
async def test_persist_results_publication_uncertainty_does_not_reenter_handler(
    tmp_path: Path,
    mode: str,
    expected_code: str,
    keeps_ids: bool,
) -> None:
    sink = ClassifiedPreWriteFailureSink(
        target_node="PersistResults",
        mode=mode,
    )
    result_value, handlers, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    assert sink.failed is True
    assert result.status == "failed"
    assert result.error_code == expected_code
    assert (result.report_artifact_id is not None) is keeps_ids
    assert (result.evidence_graph_artifact_id is not None) is keeps_ids
    assert (result.manifest_artifact_id is not None) is keeps_ids
    if keeps_ids:
        assert result.report_artifact_id is not None
        assert result.evidence_graph_artifact_id is not None
        assert result.manifest_artifact_id is not None
        handlers.artifact_store.get_bytes(result.report_artifact_id)
        assert json.loads(
            handlers.artifact_store.get_bytes(result.evidence_graph_artifact_id)
        )["evidence"]
        RunManifest.model_validate_json(
            handlers.artifact_store.get_bytes(result.manifest_artifact_id),
            strict=True,
        )
    assert persist_entries == Counter({"persist-results": 1})
    assert sum(invocations.values()) == 9
    assert len(
        [event for event in sink.attempts if event.node == "PersistResults"]
    ) == 1
    assert not any(event.node == "PersistResults" for event in sink.events.values())


async def test_persist_publication_failure_replaces_prior_primary_once(
    tmp_path: Path,
) -> None:
    def fail_draft(handlers: BaselineNodeHandlers) -> None:
        def fail_render_prompt(**_kwargs: object) -> str:
            raise OSError("draft dependency failed")

        handlers.writer.render_prompt = cast("Any", fail_render_prompt)

    sink = OneShotPreWriteFailureSink(
        target_node="PersistResults",
        target_success_only=False,
    )
    result_value, handlers, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(
            tmp_path,
            mutate=fail_draft,
            sink=sink,
        )
    )
    result = cast("Any", result_value)

    assert sink.authoritative_reads == [None]
    assert len(
        [event for event in sink.attempts if event.node == "PersistResults"]
    ) == 1
    assert not any(event.node == "PersistResults" for event in sink.events.values())
    assert persist_entries == Counter({"persist-results": 1})
    assert sum(invocations.values()) == 8
    assert result.status == "failed"
    assert result.error_code == "PERSIST_RESULTS_FAILED"
    assert result.report_artifact_id is None
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    assert manifest.failure_codes == ("INTERNAL_ERROR",)
    assert manifest.run_event_count == 10
    assert sink.events[("run-failure-after-evidence", 10)].error_code == "INTERNAL_ERROR"


@pytest.mark.parametrize(
    "mode",
    ["initial-call-hard", "authoritative-read-hard"],
)
@pytest.mark.parametrize(
    "hard_type",
    [asyncio.CancelledError, MemoryError],
)
async def test_persist_results_publication_preserves_hard_exception_identity(
    tmp_path: Path,
    mode: str,
    hard_type: type[BaseException],
) -> None:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    )
    primary = hard_type(f"hard PersistResults publication {mode}")
    sink = ClassifiedPreWriteFailureSink(
        target_node="PersistResults",
        mode=mode,
        primary=primary,
    )

    with pytest.raises(hard_type) as caught:
        await runner.run(
            run_id=f"run-persist-hard-{mode}-{hard_type.__name__}",
            thread_id=f"thread-persist-hard-{mode}-{hard_type.__name__}",
            config=_recovery_config(handlers),
            checkpoint=None,
            emit=sink,
            cancellation_token=CancellationToken(),
        )

    assert caught.value is primary
    assert persist_entries == Counter({"persist-results": 1})
    assert sum(invocations.values()) == 9
    assert len(
        [event for event in sink.attempts if event.node == "PersistResults"]
    ) == 1
    assert not any(event.node == "PersistResults" for event in sink.events.values())


@pytest.mark.parametrize(
    "mode",
    ["conflict", "constructed-corruption", "unreadable"],
)
async def test_prewrite_event_failure_requires_authoritative_exact_absence(
    tmp_path: Path,
    mode: str,
) -> None:
    sink = ClassifiedPreWriteFailureSink(
        target_node="DraftReport",
        mode=mode,
    )
    result_value, _, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    target_attempts = [
        event for event in sink.attempts if event.node == "DraftReport"
    ]
    assert len(target_attempts) == 1
    assert target_attempts[0].error_code is None
    assert sink.boundary_reads == 1
    assert not any(event.node == "DraftReport" for event in sink.events.values())
    assert not any(
        event.error_code == "DATA_CORRUPTION" for event in sink.attempts
    )
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert persist_entries == Counter()
    assert result.evidence_graph_artifact_id is None
    assert result.manifest_artifact_id is None


@pytest.mark.parametrize(
    ("mode", "expected_boundary_reads"),
    [
        ("replacement-call-ordinary-stored", 3),
        ("replacement-readback-ordinary", 4),
    ],
)
async def test_prewrite_replacement_ordinary_failure_accepts_bounded_exact_reread(
    tmp_path: Path,
    mode: str,
    expected_boundary_reads: int,
) -> None:
    sink = ClassifiedPreWriteFailureSink(
        target_node="DraftReport",
        mode=mode,
    )
    result_value, _, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    assert sink.boundary_reads == expected_boundary_reads
    target_attempts = [
        event for event in sink.attempts if event.node == "DraftReport"
    ]
    assert len(target_attempts) == 2
    assert target_attempts[0].error_code is None
    assert target_attempts[1].error_code == "DATA_CORRUPTION"
    assert sink.events[(target_attempts[1].run_id, target_attempts[1].seq)] == (
        target_attempts[1]
    )
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert persist_entries == Counter({"persist-results": 1})
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None


async def test_prewrite_replacement_ordinary_failure_without_durable_event_fails_closed(
    tmp_path: Path,
) -> None:
    sink = ClassifiedPreWriteFailureSink(
        target_node="DraftReport",
        mode="replacement-call-ordinary-absent",
    )
    result_value, _, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    assert sink.boundary_reads == 2
    target_attempts = [
        event for event in sink.attempts if event.node == "DraftReport"
    ]
    assert len(target_attempts) == 2
    assert target_attempts[1].error_code == "DATA_CORRUPTION"
    assert not any(event.node == "DraftReport" for event in sink.events.values())
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert persist_entries == Counter()
    assert result.evidence_graph_artifact_id is None
    assert result.manifest_artifact_id is None


@pytest.mark.parametrize(
    ("store_before_failure", "expected_attempts"),
    [(False, 2), (True, 1)],
)
async def test_prewrite_existing_primary_accepts_exact_failed_event_at_same_slot(
    tmp_path: Path,
    store_before_failure: bool,
    expected_attempts: int,
) -> None:
    def fail_draft(handlers: BaselineNodeHandlers) -> None:
        def fail_render_prompt(**_kwargs: object) -> str:
            raise OSError("draft dependency failed")

        handlers.writer.render_prompt = cast("Any", fail_render_prompt)

    sink = FailedEventPreWriteFailureSink(
        target_node="DraftReport",
        error_code="INTERNAL_ERROR",
        store_before_failure=store_before_failure,
    )
    result_value, _, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(
            tmp_path,
            mutate=fail_draft,
            sink=sink,
        )
    )
    result = cast("Any", result_value)

    assert (sink.authoritative_reads[0] is None) is not store_before_failure
    assert len(sink.attempts) == expected_attempts
    assert all(event == sink.attempts[0] for event in sink.attempts)
    assert sink.attempts[0].seq == 10
    assert sink.attempts[0].error_code == "INTERNAL_ERROR"
    assert sink.events[(sink.attempts[0].run_id, 10)] == sink.attempts[0]
    assert result.status == "failed"
    assert result.error_code == "INTERNAL_ERROR"
    assert sum(invocations.values()) == 8
    assert persist_entries == Counter({"persist-results": 1})
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None


@pytest.mark.parametrize(
    "wall_seconds",
    [pytest.param(0, id="coercion-equal-int"), pytest.param(-0.0, id="negative-zero")],
)
async def test_immediate_non_none_constructed_event_fails_closed_without_reread(
    tmp_path: Path,
    wall_seconds: object,
) -> None:
    sink = ImmediateConstructedReadbackSink(
        target_node="DraftReport",
        wall_seconds=wall_seconds,
    )
    result_value, _, invocations, persist_entries, _ = (
        await _run_failure_after_evidence(tmp_path, sink=sink)
    )
    result = cast("Any", result_value)

    assert sink.boundary_reads == 1
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert persist_entries == Counter()
    assert result.evidence_graph_artifact_id is None
    assert result.manifest_artifact_id is None


@pytest.mark.parametrize(
    "mode",
    [
        "initial-call-hard",
        "authoritative-read-hard",
        "replacement-call-hard",
        "replacement-readback-hard",
    ],
)
async def test_prewrite_failure_transition_preserves_hard_exception_identity(
    tmp_path: Path,
    mode: str,
) -> None:
    primary = MemoryError(f"hard {mode}")
    sink = ClassifiedPreWriteFailureSink(
        target_node="DraftReport",
        mode=mode,
        primary=primary,
    )

    with pytest.raises(MemoryError) as caught:
        await _run_failure_after_evidence(tmp_path, sink=sink)

    assert caught.value is primary
    target_attempts = [
        event for event in sink.attempts if event.node == "DraftReport"
    ]
    assert len(target_attempts) == (
        1
        if mode in {"initial-call-hard", "authoritative-read-hard"}
        else 2
    )
    assert target_attempts[0].error_code is None
    assert not any(event.node == "PersistResults" for event in sink.events.values())
    if mode == "replacement-readback-hard":
        assert sink.events[(target_attempts[-1].run_id, target_attempts[-1].seq)] == (
            target_attempts[-1]
        )
    else:
        assert not any(
            event.node == "DraftReport" for event in sink.events.values()
        )


@pytest.mark.parametrize(
    "mode",
    [
        "initial-call-hard",
        "authoritative-read-hard",
        "replacement-call-hard",
        "replacement-readback-hard",
    ],
)
async def test_prewrite_failure_transition_preserves_cancelled_identity_at_runner(
    tmp_path: Path,
    mode: str,
) -> None:
    primary = asyncio.CancelledError(f"hard {mode}")
    sink = ClassifiedPreWriteFailureSink(
        target_node="DraftReport",
        mode=mode,
        primary=primary,
    )

    with pytest.raises(asyncio.CancelledError) as caught:
        await _run_failure_after_evidence(tmp_path, sink=sink)

    assert caught.value is primary
    assert not any(event.node == "PersistResults" for event in sink.events.values())


@pytest.mark.parametrize(
    "boundary",
    ["call", "immediate-read", "authoritative-read"],
)
@pytest.mark.parametrize(
    "hard_type",
    [asyncio.CancelledError, MemoryError, KeyboardInterrupt, SystemExit],
)
async def test_event_publication_classifier_preserves_hard_exception_identity(
    boundary: str,
    hard_type: type[BaseException],
) -> None:
    primary = hard_type(f"hard publication {boundary}")

    class HardBoundarySink:
        async def __call__(self, _event: RunEvent) -> None:
            if boundary == "call":
                raise primary
            if boundary == "authoritative-read":
                raise OSError("ordinary create failure")

        async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
            del run_id, seq
            if boundary == "immediate-read":
                raise primary
            if boundary == "authoritative-read":
                raise primary
            raise AssertionError("unexpected durable event read")

    event = cast("Any", baseline_graph_module)._EVENT_PAYLOAD_PROBE
    with pytest.raises(hard_type) as caught:
        await cast("Any", baseline_graph_module)._publish_event_with_bounded_recovery(
            sink=HardBoundarySink(),
            event=event,
        )

    assert caught.value is primary


@pytest.mark.parametrize("receipt_boundary", ["terminal", "node-execution"])
@pytest.mark.parametrize(
    "hard_type",
    [asyncio.CancelledError, MemoryError, KeyboardInterrupt, SystemExit],
)
async def test_replacement_receipt_storage_preserves_hard_exception_identity(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    receipt_boundary: str,
    hard_type: type[BaseException],
) -> None:
    sink = OneShotPreWriteFailureSink(target_node="DraftReport")
    captured: dict[str, object] = {}
    capture_primary = MemoryError("capture replacement transition inputs")
    original_transition = cast(
        "Any",
        baseline_graph_module,
    )._publish_missing_event_failure_transition

    async def capture_transition(**kwargs: object) -> dict[str, object]:
        captured.update(kwargs)
        raise capture_primary

    monkeypatch.setattr(
        baseline_graph_module,
        "_publish_missing_event_failure_transition",
        capture_transition,
    )
    with pytest.raises(MemoryError) as captured_error:
        await _run_failure_after_evidence(tmp_path, sink=sink)
    assert captured_error.value is capture_primary
    assert captured
    audit = cast("Any", captured["audit"])
    provider_calls_before = tuple(audit.provider_calls)
    provider_receipts_before = tuple(audit.provider_receipt_ids)
    assert len(provider_calls_before) == 1
    assert len(provider_receipts_before) == 1
    monkeypatch.setattr(
        baseline_graph_module,
        "_publish_missing_event_failure_transition",
        original_transition,
    )
    original_put = cast("Any", baseline_graph_module)._put_audit_receipt
    primary = hard_type(f"hard replacement {receipt_boundary} receipt")

    def hard_receipt_put(*args: object, **kwargs: object) -> str:
        kind = kwargs.get("kind")
        payload = cast("Mapping[str, object]", kwargs.get("payload"))
        record = payload.get("record")
        is_failure_node = (
            kind == "node-execution"
            and isinstance(record, dict)
            and record.get("error_code") == "DATA_CORRUPTION"
        )
        if (
            receipt_boundary == "terminal"
            and kind == "terminal"
            and payload.get("error_code") == "DATA_CORRUPTION"
        ) or (receipt_boundary == "node-execution" and is_failure_node):
            raise primary
        return cast("str", original_put(*args, **kwargs))

    monkeypatch.setattr(
        baseline_graph_module,
        "_put_audit_receipt",
        hard_receipt_put,
    )
    with pytest.raises(hard_type) as caught:
        await original_transition(**captured)

    assert caught.value is primary
    assert tuple(audit.provider_calls) == provider_calls_before
    assert tuple(audit.provider_receipt_ids) == provider_receipts_before
    assert not any(
        event.error_code == "DATA_CORRUPTION" for event in sink.attempts
    )
    assert not any(event.node == "PersistResults" for event in sink.events.values())
    restored = cast("Mapping[str, object]", captured["restored"])
    assert restored["next_event_seq"] == 10


async def test_sqlite_resume_reuses_prewrite_replacement_without_writer_repeat(
    tmp_path: Path,
) -> None:
    root = (tmp_path / "prewrite-replacement-resume").resolve()
    event_root = (root / "events").resolve()
    checkpoint_path = (root / "checkpoints.sqlite3").resolve()
    run_id = "run-prewrite-replacement-resume"
    thread_id = "thread-prewrite-replacement-resume"
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    primary = MemoryError("crash after replacement event publication")

    class CrashAfterReplacementSink(FilesystemEventSink):
        def __init__(self, path: Path) -> None:
            super().__init__(path)
            self.target_key: tuple[str, int] | None = None
            self.prewrite_failed = False
            self.absence_reads = 0
            self.replacement: RunEvent | None = None

        async def __call__(self, event: RunEvent) -> None:
            if (
                event.node == "DraftReport"
                and event.error_code is None
                and not self.prewrite_failed
            ):
                self.target_key = (event.run_id, event.seq)
                self.prewrite_failed = True
                raise OSError("draft event failed before write")
            await super().__call__(event)
            if (
                self.target_key == (event.run_id, event.seq)
                and event.error_code == "DATA_CORRUPTION"
                and self.replacement is None
            ):
                self.replacement = event
                raise primary

        async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None:
            event = await super().get_event(run_id=run_id, seq=seq)
            if (
                self.prewrite_failed
                and self.target_key == (run_id, seq)
                and event is None
            ):
                self.absence_reads += 1
            return event

    handlers_a = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    current_config = _recovery_config(handlers_a)
    sink_a = CrashAfterReplacementSink(event_root)
    clock_a = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    async with open_sqlite_checkpointer(checkpoint_path) as saver_a:
        graph_a = build_baseline_graph(
            handlers_a.as_dependencies(checkpointer=saver_a)
        )
        runner_a = LangGraphResearchRunner(
            baseline_graph=graph_a,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_a.monotonic,
                utc_now=clock_a.utc_now,
            ),
        )
        with pytest.raises(MemoryError) as caught:
            await runner_a.run(
                run_id=run_id,
                thread_id=thread_id,
                config=current_config,
                checkpoint=None,
                emit=sink_a,
                cancellation_token=CancellationToken(),
            )
        assert caught.value is primary
        saved_a = await saver_a.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        assert saved_a is not None
        checkpoint = checkpoint_ref_from_tuple(saved_a)
        saved_values = cast("Any", saved_a).checkpoint["channel_values"]
        assert saved_values["next_event_seq"] == 10

    assert sink_a.absence_reads == 1
    assert sink_a.replacement is not None
    replacement = sink_a.replacement
    assert replacement.seq == 10
    assert replacement.error_code == "DATA_CORRUPTION"
    replacement_bytes = FilesystemEventSink._bytes(replacement)
    assert sum(invocations.values()) == 9
    assert invocations["model:baseline-writer-v1"] == 1
    assert persist_entries == Counter()
    replacement_receipt = next(
        receipt
        for artifact_id in replacement.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers_a.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    )
    replacement_state = cast("Any", baseline_graph_module)._state_from_payload(
        replacement_receipt.payload["state"]
    )
    assert replacement_state["next_event_seq"] == 10

    handlers_b = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    sink_b = FilesystemEventSink(event_root)
    clock_b = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=10.0,
    )
    async with open_sqlite_checkpointer(checkpoint_path) as saver_b:
        exact = await saver_b.aget_tuple(checkpoint_config(checkpoint))
        assert exact is not None
        assert checkpoint_ref_from_tuple(exact) == checkpoint
        graph_b = build_baseline_graph(
            handlers_b.as_dependencies(checkpointer=saver_b)
        )
        result = await LangGraphResearchRunner(
            baseline_graph=graph_b,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_b.monotonic,
                utc_now=clock_b.utc_now,
            ),
        ).run(
            run_id=run_id,
            thread_id=thread_id,
            config=current_config,
            checkpoint=checkpoint,
            emit=sink_b,
            cancellation_token=CancellationToken(),
        )

    reread = await sink_b.get_event(run_id=run_id, seq=10)
    assert reread == replacement
    assert FilesystemEventSink._bytes(cast("RunEvent", reread)) == replacement_bytes
    persist_event = await sink_b.get_event(run_id=run_id, seq=11)
    assert persist_event is not None
    assert persist_event.node == "PersistResults"
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert sum(invocations.values()) == 9
    assert invocations["model:baseline-writer-v1"] == 1
    assert persist_entries == Counter({"persist-results": 1})
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    manifest = RunManifest.model_validate_json(
        handlers_b.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    assert manifest.failure_codes == ("DATA_CORRUPTION",)
    assert len(
        [call for call in manifest.provider_calls if call.node == "DraftReport"]
    ) == 1


async def _clean_ledger_checkpoint(
    tmp_path: Path,
    *,
    next_event_seq: int = 12,
) -> tuple[BaselineNodeHandlers, RunConfig, MemoryEventSink, BaselineState]:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    current_config = _recovery_config(handlers)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    sink = MemoryEventSink()
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    result = await LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    ).run(
        run_id="run-ledger-owner",
        thread_id="thread-ledger-owner",
        config=current_config,
        checkpoint=None,
        emit=sink,
        cancellation_token=CancellationToken(),
    )
    assert result.status == "completed", result.error_code
    pre_persist: BaselineState | None = None
    async for saved in saver.alist(
        {
            "configurable": {
                "thread_id": "thread-ledger-owner",
                "checkpoint_ns": "",
            }
        }
    ):
        values = cast("Any", saved).checkpoint["channel_values"]
        if (
            values.get("next_event_seq") == next_event_seq
            and values.get("manifest_artifact_id") is None
        ):
            pre_persist = cast("Any", baseline_graph_module).validate_baseline_state(
                {
                    name: values[name]
                    for name in cast(
                        "Any", baseline_graph_module
                    ).BaselineState.__annotations__
                }
            )
            break
    assert pre_persist is not None
    return handlers, current_config, sink, pre_persist


def _event_node_receipt(
    handlers: BaselineNodeHandlers,
    sink: MemoryEventSink,
    *,
    node: str,
) -> tuple[RunEvent, str, object]:
    event = next(item for item in sink.events.values() if item.node == node)
    for artifact_id in event.artifact_ids:
        receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            artifact_id,
        )
        if receipt is not None and receipt.kind == "node-execution":
            return event, artifact_id, receipt
    raise AssertionError(f"missing node receipt for {node}")


def _replace_event_node_receipt(
    handlers: BaselineNodeHandlers,
    sink: MemoryEventSink,
    state: BaselineState,
    *,
    event: RunEvent,
    old_receipt_id: str,
    receipt: object,
    payload: dict[str, object],
) -> BaselineState:
    new_receipt_id = cast("Any", baseline_graph_module)._put_audit_receipt(
        handlers.artifact_store,
        kind="node-execution",
        run_id=cast("Any", receipt).run_id,
        thread_id=cast("Any", receipt).thread_id,
        receipt_key=cast("Any", receipt).receipt_key,
        payload=payload,
    )
    base_event = RunEvent.model_validate(payload["event"])
    replacement_event = base_event.model_copy(
        update={
            "artifact_ids": tuple(
                dict.fromkeys((*base_event.artifact_ids, new_receipt_id))
            )
        }
    )
    sink.events[(event.run_id, event.seq)] = replacement_event
    replaced = dict(state)
    replaced["baseline_work_artifact_ids"] = tuple(
        new_receipt_id if item == old_receipt_id else item
        for item in state["baseline_work_artifact_ids"]
    )
    return cast("Any", baseline_graph_module).validate_baseline_state(replaced)


def _apply_node_record_representation_corruption(
    record: dict[str, object],
    corruption: str,
) -> None:
    usage = cast("dict[str, object]", record["usage"])
    if corruption == "record-attempt-bool":
        record["attempt"] = True
    elif corruption == "record-latency-float":
        record["latency_ms"] = float(cast("int", record["latency_ms"]))
    elif corruption == "record-usage-wall-int":
        usage["wall_seconds"] = 0
    elif corruption == "record-usage-cached-bool":
        usage["cached_tokens"] = False
    elif corruption == "record-usage-wall-negative-zero":
        usage["wall_seconds"] = -0.0
    else:
        raise AssertionError(f"unknown node record corruption: {corruption}")


@pytest.mark.parametrize(
    "corruption",
    [
        "record-attempt-bool",
        "record-latency-float",
        "record-usage-wall-int",
        "record-usage-cached-bool",
        "record-usage-wall-negative-zero",
    ],
)
async def test_attempt_and_manifest_reject_noncanonical_node_execution_record(
    tmp_path: Path,
    corruption: str,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(tmp_path)
    composition = handlers.as_dependencies(
        checkpointer=InMemorySaver(serde=checkpoint_serializer())
    )._audit_composition
    assert composition is not None
    assert (
        cast("Any", baseline_graph_module)._next_node_attempt(
            composition,
            state,
            "Plan",
        )
        == 2
    )
    event, receipt_id, receipt = _event_node_receipt(
        handlers,
        sink,
        node="Plan",
    )
    payload = deepcopy(cast("Any", receipt).payload)
    record = cast("dict[str, object]", payload["record"])
    _apply_node_record_representation_corruption(record, corruption)
    state = _replace_event_node_receipt(
        handlers,
        sink,
        state,
        event=event,
        old_receipt_id=receipt_id,
        receipt=receipt,
        payload=payload,
    )

    with pytest.raises(ArtifactIntegrityError):
        cast("Any", baseline_graph_module)._next_node_attempt(
            composition,
            state,
            "Plan",
        )

    _bind_direct_runtime(handlers, current_config, sink, state)

    with pytest.raises(ArtifactIntegrityError):
        await handlers.persist_results(state)


def _bind_direct_runtime(
    handlers: BaselineNodeHandlers,
    current_config: RunConfig,
    sink: MemoryEventSink,
    state: BaselineState,
) -> BaselineRuntimeContext:
    clock = SegmentClock(1)
    context = BaselineRuntimeContext(
        run_id=state["run_id"],
        thread_id=state["thread_id"],
        config=current_config,
        emit=sink,
        cancellation_token=CancellationToken(),
        budget_accountant=BudgetAccountant.from_snapshot(
            current_config.budget,
            state["budget_snapshot"],
            run_scope=state["run_id"],
        ),
        deadline=clock.monotonic() + 100.0,
        run_started_monotonic=clock.monotonic(),
        run_started_at=datetime(2026, 9, 1, tzinfo=UTC),
        elapsed_base_seconds=state["elapsed_wall_seconds"],
        monotonic=clock.monotonic,
        utc_now=clock.utc_now,
        new_id=lambda prefix: prefix,
    )
    cast("Any", handlers)._runtime = lambda: context
    return context


@pytest.mark.parametrize("corruption", ["remove-child", "typed-reparent", "terminal-reparent"])
async def test_persist_rejects_real_ledger_child_ownership_corruption(
    tmp_path: Path,
    corruption: str,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(tmp_path)
    store_event, store_id, store_receipt = _event_node_receipt(
        handlers,
        sink,
        node="StoreEvidence",
    )
    store_payload = deepcopy(cast("Any", store_receipt).payload)
    original_store_payload = deepcopy(store_payload)
    store_children = cast("list[str]", store_payload["child_receipt_ids"])
    typed_id = next(
        item
        for item in store_children
        if cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            item,
        ).kind
        == "parsed-artifact"
    )
    store_children.remove(typed_id)
    if corruption == "typed-reparent":
        store_record = cast("dict[str, object]", store_payload["record"])
        store_record["output_artifact_ids"] = [
            item
            for item in cast("list[str]", store_record["output_artifact_ids"])
            if item != typed_id
        ]
        store_base_event = cast("dict[str, object]", store_payload["event"])
        store_base_event["artifact_ids"] = [
            item
            for item in cast("list[str]", store_base_event["artifact_ids"])
            if item != typed_id
        ]
    state = _replace_event_node_receipt(
        handlers,
        sink,
        state,
        event=store_event,
        old_receipt_id=store_id,
        receipt=store_receipt,
        payload=store_payload,
    )

    if corruption == "terminal-reparent":
        current_store_event, current_store_id, current_store_receipt = (
            _event_node_receipt(handlers, sink, node="StoreEvidence")
        )
        state = _replace_event_node_receipt(
            handlers,
            sink,
            state,
            event=current_store_event,
            old_receipt_id=current_store_id,
            receipt=current_store_receipt,
            payload=original_store_payload,
        )

    if corruption == "typed-reparent":
        plan_event, plan_id, plan_receipt = _event_node_receipt(
            handlers,
            sink,
            node="Plan",
        )
        plan_payload = deepcopy(cast("Any", plan_receipt).payload)
        cast("list[str]", plan_payload["child_receipt_ids"]).append(typed_id)
        cast("dict[str, object]", plan_payload["record"])[
            "output_artifact_ids"
        ] = [
            *cast(
                "list[str]",
                cast("dict[str, object]", plan_payload["record"])[
                    "output_artifact_ids"
                ],
            ),
            typed_id,
        ]
        cast("dict[str, object]", plan_payload["event"])["artifact_ids"] = [
            *cast(
                "list[str]",
                cast("dict[str, object]", plan_payload["event"])["artifact_ids"],
            ),
            typed_id,
        ]
        state = _replace_event_node_receipt(
            handlers,
            sink,
            state,
            event=plan_event,
            old_receipt_id=plan_id,
            receipt=plan_receipt,
            payload=plan_payload,
        )
    elif corruption == "terminal-reparent":
        finalize_event, finalize_id, finalize_receipt = _event_node_receipt(
            handlers,
            sink,
            node="FinalizeCitations",
        )
        finalize_payload = deepcopy(cast("Any", finalize_receipt).payload)
        terminal_id = next(
            item
            for item in cast("list[str]", finalize_payload["child_receipt_ids"])
            if cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                item,
            ).kind
            == "terminal"
        )
        cast("list[str]", finalize_payload["child_receipt_ids"]).remove(terminal_id)
        state = _replace_event_node_receipt(
            handlers,
            sink,
            state,
            event=finalize_event,
            old_receipt_id=finalize_id,
            receipt=finalize_receipt,
            payload=finalize_payload,
        )
        plan_event, plan_id, plan_receipt = _event_node_receipt(
            handlers,
            sink,
            node="Plan",
        )
        plan_payload = deepcopy(cast("Any", plan_receipt).payload)
        cast("list[str]", plan_payload["child_receipt_ids"]).append(terminal_id)
        state = _replace_event_node_receipt(
            handlers,
            sink,
            state,
            event=plan_event,
            old_receipt_id=plan_id,
            receipt=plan_receipt,
            payload=plan_payload,
        )

    _bind_direct_runtime(handlers, current_config, sink, state)
    with pytest.raises(ArtifactIntegrityError):
        await handlers.persist_results(state)


async def test_store_evidence_exact_repeat_has_one_typed_owner(
    tmp_path: Path,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(
        tmp_path,
        next_event_seq=7,
    )
    context = _bind_direct_runtime(handlers, current_config, sink, state)
    context.audit.begin(graph_node="StoreEvidence", node_attempt=1)
    first = dict(await handlers.store_evidence(state))
    first_children = tuple(context.audit.child_receipt_ids)
    first_results = tuple(context.audit.result_artifact_ids)
    merged = cast("Any", baseline_graph_module).validate_baseline_state(
        {**state, **first}
    )

    context.audit.begin(graph_node="StoreEvidence", node_attempt=2)
    second = dict(await handlers.store_evidence(merged))

    assert len(first_children) == 4
    assert len(first_results) == 6
    assert context.audit.child_receipt_ids == []
    assert context.audit.result_artifact_ids == []
    assert second["source_ids"] == first["source_ids"]
    assert second["evidence_ids"] == first["evidence_ids"]
    assert second["baseline_work_artifact_ids"] == first[
        "baseline_work_artifact_ids"
    ]


async def test_store_evidence_same_locator_different_need_has_distinct_evidence_ids(
    tmp_path: Path,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(
        tmp_path,
        next_event_seq=7,
    )
    plan = cast("Any", handlers)._plan_from_state(state)
    original = plan.subquestions[0]
    second_need = original.information_needs[0].model_copy(
        update={"need_id": "need-second"}
    )
    second_subquestion = original.model_copy(
        update={
            "id": "sq-second",
            "information_needs": (second_need,),
        }
    )
    expanded = plan.model_copy(
        update={"subquestions": (original, second_subquestion)}
    )
    cast("Any", handlers)._plan_from_state = lambda _state: expanded
    context = _bind_direct_runtime(handlers, current_config, sink, state)

    context.audit.begin(graph_node="StoreEvidence", node_attempt=1)
    first = dict(await handlers.store_evidence(state))
    first_state = cast("Any", baseline_graph_module).validate_baseline_state(
        {**state, **first, "active_subquestion_id": "sq-second"}
    )
    first_evidence_ids = set(cast("tuple[str, ...]", first["evidence_ids"]))

    context.audit.begin(graph_node="StoreEvidence", node_attempt=2)
    second = dict(await handlers.store_evidence(first_state))
    second_evidence_ids = set(cast("tuple[str, ...]", second["evidence_ids"]))
    new_evidence_ids = second_evidence_ids - first_evidence_ids

    assert len(first_evidence_ids) == 2
    assert len(new_evidence_ids) == 2
    assert second["source_ids"] == first["source_ids"]
    assert len(context.audit.child_receipt_ids) == 2
    assert {
        handlers.evidence_store.get_evidence(item).information_need_ids
        for item in first_evidence_ids
    } == {("need-offline",)}
    assert {
        handlers.evidence_store.get_evidence(item).information_need_ids
        for item in new_evidence_ids
    } == {("need-second",)}


async def test_store_evidence_reuses_receipts_after_pre_checkpoint_crash(
    tmp_path: Path,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(
        tmp_path,
        next_event_seq=7,
    )
    context = _bind_direct_runtime(handlers, current_config, sink, state)

    context.audit.begin(graph_node="StoreEvidence", node_attempt=1)
    first = dict(await handlers.store_evidence(state))
    first_children = tuple(context.audit.child_receipt_ids)
    first_results = tuple(context.audit.result_artifact_ids)

    context.audit.begin(graph_node="StoreEvidence", node_attempt=2)
    resumed = dict(await handlers.store_evidence(state))

    assert tuple(context.audit.child_receipt_ids) == first_children
    assert tuple(context.audit.result_artifact_ids) == first_results
    assert resumed["baseline_work_artifact_ids"] == first[
        "baseline_work_artifact_ids"
    ]
    assert resumed["source_ids"] == first["source_ids"]
    assert resumed["evidence_ids"] == first["evidence_ids"]


async def test_store_evidence_repeat_conflict_is_corruption(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(
        tmp_path,
        next_event_seq=7,
    )
    context = _bind_direct_runtime(handlers, current_config, sink, state)
    context.audit.begin(graph_node="StoreEvidence", node_attempt=1)
    first = dict(await handlers.store_evidence(state))
    merged = cast("Any", baseline_graph_module).validate_baseline_state(
        {**state, **first}
    )
    source_id = cast("tuple[str, ...]", first["source_ids"])[0]
    source = handlers.evidence_store.get_source(source_id)
    original_get_source = handlers.evidence_store.get_source

    def conflicting_source(candidate_id: str):  # type: ignore[no-untyped-def]
        candidate = original_get_source(candidate_id)
        if candidate_id == source_id:
            return candidate.model_copy(update={"title": f"{source.title} changed"})
        return candidate

    monkeypatch.setattr(
        handlers.evidence_store,
        "get_source",
        conflicting_source,
    )
    context.audit.begin(graph_node="StoreEvidence", node_attempt=2)

    with pytest.raises(ArtifactIntegrityError, match="stored source"):
        await handlers.store_evidence(merged)

    assert context.audit.child_receipt_ids == []


async def test_manifest_two_loop_shared_sources_have_one_typed_owner_per_identity(
    tmp_path: Path,
) -> None:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
        plan_payload=_two_loop_plan(),
    )
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    sink = MemoryEventSink()
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    result = await LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    ).run(
        run_id="run-two-loop-ownership",
        thread_id="thread-two-loop-ownership",
        config=_recovery_config(handlers),
        checkpoint=None,
        emit=sink,
        cancellation_token=CancellationToken(),
    )

    assert result.status == "completed", (
        result.error_code,
        ";".join(
            f"{event.seq}:{event.node}:{event.error_code}"
            for event in sorted(sink.events.values(), key=lambda item: item.seq)
        ),
    )
    assert result.manifest_artifact_id is not None
    manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    store_events = sorted(
        (event for event in sink.events.values() if event.node == "StoreEvidence"),
        key=lambda event: event.seq,
    )
    assert len(store_events) == 2
    typed_children_by_loop: list[dict[str, list[str]]] = []
    for event in store_events:
        node_receipt = next(
            receipt
            for artifact_id in event.artifact_ids
            if (
                receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                    handlers.artifact_store,
                    artifact_id,
                )
            )
            is not None
            and receipt.kind == "node-execution"
        )
        typed_children: dict[str, list[str]] = {
            "parsed-artifact": [],
            "evidence-hash": [],
        }
        for child_id in node_receipt.payload["child_receipt_ids"]:
            child = cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                child_id,
            )
            if child is not None and child.kind in typed_children:
                typed_children[child.kind].append(child_id)
        typed_children_by_loop.append(typed_children)

    first, second = typed_children_by_loop
    assert len(first["parsed-artifact"]) == 2
    assert second["parsed-artifact"] == []
    assert len(first["evidence-hash"]) == 2
    assert len(second["evidence-hash"]) == 2
    assert set(first["evidence-hash"]).isdisjoint(second["evidence-hash"])
    assert len(manifest.parsed_artifacts) == 2
    assert len(manifest.evidence_hashes) == 4
    assert {
        handlers.evidence_store.get_evidence(record.evidence_id).information_need_ids
        for record in manifest.evidence_hashes
    } == {("need-first",), ("need-second",)}
    assert persist_entries == Counter({"persist-results": 1})


@pytest.mark.parametrize("failure", ["missing", "changed", "membership"])
async def test_evidence_graph_requires_exact_immediate_readback_and_membership(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    handlers, current_config, sink, state = await _clean_ledger_checkpoint(tmp_path)
    original_put = handlers.artifact_store.put_bytes
    original_get = handlers.artifact_store.get_bytes
    target_id: str | None = None
    failed = False
    manifest_puts = 0

    def selective_put(data: bytes, *, media_type: str):  # type: ignore[no-untyped-def]
        nonlocal target_id, manifest_puts
        if media_type == "application/vnd.deepresearch.run-manifest+json":
            manifest_puts += 1
        reference = original_put(data, media_type=media_type)
        if media_type == "application/vnd.deepresearch.evidence-graph+json":
            if failure == "membership":
                value = json.loads(data)
                cast("list[object]", value["evidence"]).pop()
                divergent = original_put(
                    json.dumps(
                        value,
                        ensure_ascii=False,
                        allow_nan=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode(),
                    media_type=media_type,
                )
                target_id = divergent.artifact_id
                return divergent
            target_id = reference.artifact_id
        return reference

    def selective_get(artifact_id: str) -> bytes:
        nonlocal failed
        if artifact_id == target_id and not failed and failure != "membership":
            failed = True
            if failure == "missing":
                raise FileNotFoundError(artifact_id)
            return original_get(artifact_id) + b"\n"
        return original_get(artifact_id)

    monkeypatch.setattr(handlers.artifact_store, "put_bytes", selective_put)
    monkeypatch.setattr(handlers.artifact_store, "get_bytes", selective_get)
    _bind_direct_runtime(handlers, current_config, sink, state)

    with pytest.raises(ArtifactIntegrityError):
        await handlers.persist_results(state)

    assert target_id is not None
    assert manifest_puts == 0


@pytest.mark.parametrize("failure", ["missing", "changed"])
async def test_manifest_missing_or_changed_immediate_readback_is_not_published(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failure: str,
) -> None:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    original_put = handlers.artifact_store.put_bytes
    original_get = handlers.artifact_store.get_bytes
    target_id: str | None = None
    failed = False

    def selective_put(data: bytes, *, media_type: str):  # type: ignore[no-untyped-def]
        nonlocal target_id
        reference = original_put(data, media_type=media_type)
        if media_type == "application/vnd.deepresearch.run-manifest+json":
            target_id = reference.artifact_id
        return reference

    def selective_get(artifact_id: str) -> bytes:
        nonlocal failed
        if artifact_id == target_id and not failed:
            failed = True
            if failure == "missing":
                raise FileNotFoundError(artifact_id)
            return original_get(artifact_id) + b"\n"
        return original_get(artifact_id)

    monkeypatch.setattr(handlers.artifact_store, "put_bytes", selective_put)
    monkeypatch.setattr(handlers.artifact_store, "get_bytes", selective_get)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    result = await LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    ).run(
        run_id=f"run-manifest-readback-{failure}",
        thread_id=f"thread-manifest-readback-{failure}",
        config=_recovery_config(handlers),
        checkpoint=None,
        emit=MemoryEventSink(),
        cancellation_token=CancellationToken(),
    )

    assert target_id is not None
    assert failed is True
    assert result.status == "failed"
    assert result.error_code == "PERSIST_RESULTS_FAILED"
    assert result.manifest_artifact_id is None
    assert persist_entries == Counter({"persist-results": 1})
    assert sum(invocations.values()) == 9


@pytest.mark.parametrize(
    ("case", "event_node", "event_seq"),
    [
        ("p1-model", "Plan", 2),
        ("query-model", "Search", 4),
        ("search", "Search", 4),
        ("fetch", "Fetch", 5),
        ("parse", "ParseAndNormalize", 6),
        ("embed", "RankEvidence", 8),
        ("writer", "DraftReport", 10),
        ("persist-results", "PersistResults", 12),
    ],
)
async def test_real_sqlite_filesystem_recovery_reuses_operation_and_event(
    tmp_path: Path,
    case: str,
    event_node: str,
    event_seq: int,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    root = (tmp_path / case).resolve()
    event_root = (root / "events").resolve()
    checkpoint_path = (root / "checkpoints.sqlite3").resolve()
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    probe: dict[str, object] = {}
    run_id = f"run-recovery-{case}"
    thread_id = f"thread-recovery-{case}"

    handlers_a = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
        crash_case=case,
        probe=probe,
    )
    config_a = _recovery_config(handlers_a)
    sink_a = FilesystemEventSink(event_root)
    clock_a = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    async with open_sqlite_checkpointer(checkpoint_path) as saver_a:
        graph_a = build_baseline_graph(
            handlers_a.as_dependencies(checkpointer=saver_a)
        )
        runner_a = LangGraphResearchRunner(
            baseline_graph=graph_a,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_a.monotonic,
                utc_now=clock_a.utc_now,
            ),
        )
        try:
            result_a = await runner_a.run(
                run_id=run_id,
                thread_id=thread_id,
                config=config_a,
                checkpoint=None,
                emit=sink_a,
                cancellation_token=CancellationToken(),
            )
        except MemoryError as error:
            assert "crash after" in str(error)
        else:
            pytest.fail(f"operation failpoint did not fire: {result_a}")
        saved_a = await saver_a.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        assert saved_a is not None
        ref_a = checkpoint_ref_from_tuple(saved_a)

    reopened_store = LocalArtifactStore(root)
    if case == "persist-results":
        manifest_id = cast("str", probe["manifest_id"])
        manifest_bytes = cast("bytes", probe["manifest_bytes"])
        assert reopened_store.get_bytes(manifest_id) == manifest_bytes
    else:
        captured_key = probe["key"]
        captured_entry = cast("CacheEntry", probe["entry"])
        reopened_entry = FileCache(root).get(cast("Any", captured_key))
        assert reopened_entry == captured_entry
        assert reopened_store.get_bytes(captured_entry.value_artifact_id)
        receipt_id = cast("str", captured_entry.metadata["provider_call_receipt_id"])
        assert reopened_store.get_bytes(receipt_id)

    handlers_b = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    config_b = _recovery_config(handlers_b)
    assert config_b == config_a
    sink_b = FilesystemEventSink(event_root, crash_after_node=event_node)
    clock_b = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=5.0,
    )
    async with open_sqlite_checkpointer(checkpoint_path) as saver_b:
        exact_a = await saver_b.aget_tuple(checkpoint_config(ref_a))
        assert exact_a is not None
        assert checkpoint_ref_from_tuple(exact_a) == ref_a
        graph_b = build_baseline_graph(
            handlers_b.as_dependencies(checkpointer=saver_b)
        )
        runner_b = LangGraphResearchRunner(
            baseline_graph=graph_b,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_b.monotonic,
                utc_now=clock_b.utc_now,
            ),
        )
        with pytest.raises(
            MemoryError,
            match="crash after filesystem event publication",
        ):
            await runner_b.run(
                run_id=run_id,
                thread_id=thread_id,
                config=config_b,
                checkpoint=ref_a,
                emit=sink_b,
                cancellation_token=CancellationToken(),
            )
        event_b = await sink_b.get_event(run_id=run_id, seq=event_seq)
        assert event_b is not None
        latest_b = await saver_b.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        assert latest_b is not None
        fork_values = cast("Any", latest_b).checkpoint["channel_values"]
        fork_state = {
            name: fork_values[name]
            for name in cast("Any", baseline_graph_module).BaselineState.__annotations__
        }
        cast("Any", baseline_graph_module).validate_baseline_state(fork_state)
        node_receipt_id = event_b.artifact_ids[-1]
        node_receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
            reopened_store,
            node_receipt_id,
        )
        assert node_receipt is not None
        child_receipt_ids = tuple(node_receipt.payload["child_receipt_ids"])
        fork_work_ids = tuple(fork_state["baseline_work_artifact_ids"])
        assert fork_state["next_event_seq"] == event_seq
        assert node_receipt_id not in fork_work_ids
        assert set(child_receipt_ids).isdisjoint(fork_work_ids)

    handlers_c = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    sink_c = FilesystemEventSink(event_root)
    clock_c = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=10.0,
    )
    async with open_sqlite_checkpointer(checkpoint_path) as saver_c:
        exact_c = await saver_c.aget_tuple(checkpoint_config(ref_a))
        assert exact_c is not None
        assert checkpoint_ref_from_tuple(exact_c) == ref_a
        graph_c = build_baseline_graph(
            handlers_c.as_dependencies(checkpointer=saver_c)
        )
        runner_c = LangGraphResearchRunner(
            baseline_graph=graph_c,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_c.monotonic,
                utc_now=clock_c.utc_now,
            ),
        )
        result = await runner_c.run(
            run_id=run_id,
            thread_id=thread_id,
            config=_recovery_config(handlers_c),
            checkpoint=ref_a,
            emit=sink_c,
            cancellation_token=CancellationToken(),
        )

    assert result.status == "completed", result.error_code
    assert result.stop_reason == "SUFFICIENT"
    event_c = await sink_c.get_event(run_id=run_id, seq=event_seq)
    assert event_c == event_b
    assert FilesystemEventSink._bytes(event_c) == FilesystemEventSink._bytes(event_b)
    durable_events = [
        await sink_c.get_event(run_id=run_id, seq=seq) for seq in range(1, 13)
    ]
    assert all(event is not None for event in durable_events)
    assert [cast("RunEvent", event).seq for event in durable_events] == list(
        range(1, 13)
    )
    assert all(count == 1 for count in invocations.values())
    assert sum(invocations.values()) == 9
    assert invocations["model:fixed-planner-v1"] == 1
    assert invocations["model:fixed-planner-v1-queries"] == 1
    assert invocations["search:offline baseline query"] == 1
    assert invocations["fetch:https://source1.example.test/doc"] == 1
    assert invocations["fetch:https://source2.example.test/doc"] == 1
    assert len([key for key in invocations if key.startswith("parse:")]) == 2
    assert len([key for key in invocations if key.startswith("embed:")]) == 1
    assert invocations["model:baseline-writer-v1"] == 1
    assert persist_entries["persist-results"] == (
        2 if case == "persist-results" else 1
    )
    assert result.manifest_artifact_id is not None
    manifest_bytes = reopened_store.get_bytes(result.manifest_artifact_id)
    manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
    if case == "persist-results":
        first_manifest_id = cast("str", probe["manifest_id"])
        first_manifest_bytes = cast("bytes", probe["manifest_bytes"])
        assert result.manifest_artifact_id == first_manifest_id
        assert manifest_bytes == first_manifest_bytes
        assert RunManifest.model_validate_json(first_manifest_bytes, strict=True) == manifest
    assert len(manifest.provider_calls) == 9
    assert manifest.cache_hit_count == 0
    expected_event_hash = hashlib.sha256(
        json.dumps(
            [
                cast("RunEvent", event).model_dump(mode="json")
                for event in durable_events[:11]
            ],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert manifest.run_events_sha256 == expected_event_hash
    for artifact_id in (
        result.report_artifact_id,
        result.evidence_graph_artifact_id,
        result.manifest_artifact_id,
    ):
        assert artifact_id is not None
        assert reopened_store.get_bytes(artifact_id)


@pytest.mark.parametrize(
    ("second_segment_seconds", "expect_completed"),
    [(20.0, True), (173.0, False)],
)
async def test_recovered_durable_wall_shortens_every_provider_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    second_segment_seconds: float,
    expect_completed: bool,
) -> None:
    root = (tmp_path / f"wall-{second_segment_seconds}").resolve()
    event_root = (root / "events").resolve()
    checkpoint_path = (root / "checkpoints.sqlite3").resolve()
    run_id = f"run-wall-{second_segment_seconds}"
    thread_id = f"thread-wall-{second_segment_seconds}"
    first_segment_seconds = 7.0
    max_wall_seconds = 180.0
    recovered_active_wall = first_segment_seconds + second_segment_seconds
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    observed_deadlines: dict[str, list[float]] = {}

    def capture_deadline(
        owner: object,
        method_name: str,
        label: str,
        *,
        advance_clock: ControlledSegmentClock | None = None,
        advance_seconds: float = 0.0,
    ) -> None:
        original = getattr(owner, method_name)

        async def wrapped(*args: Any, **kwargs: Any) -> object:
            deadline = kwargs.get("deadline")
            assert type(deadline) is float
            observed_deadlines.setdefault(label, []).append(deadline)
            if advance_clock is not None:
                advance_clock.advance(advance_seconds)
            return await original(*args, **kwargs)

        setattr(owner, method_name, wrapped)

    handlers_a = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    current_config = _recovery_config(handlers_a)
    assert float(current_config.budget.max_wall_time_seconds) == max_wall_seconds
    clock_a = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    original_validate = handlers_a.validate_request

    async def advance_validate(state: BaselineState) -> object:
        update = await original_validate(state)
        clock_a.advance(first_segment_seconds)
        return update

    handlers_a.validate_request = cast("Any", advance_validate)
    sink_a = FilesystemEventSink(
        event_root,
        crash_after_node="ValidateRequest",
    )
    async with open_sqlite_checkpointer(checkpoint_path) as saver_a:
        graph_a = build_baseline_graph(
            handlers_a.as_dependencies(checkpointer=saver_a)
        )
        runner_a = LangGraphResearchRunner(
            baseline_graph=graph_a,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_a.monotonic,
                utc_now=clock_a.utc_now,
            ),
        )
        with pytest.raises(
            MemoryError,
            match="crash after filesystem event publication",
        ):
            await runner_a.run(
                run_id=run_id,
                thread_id=thread_id,
                config=current_config,
                checkpoint=None,
                emit=sink_a,
                cancellation_token=CancellationToken(),
            )
        saved_a = await saver_a.aget_tuple(
            {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
        )
        assert saved_a is not None
        ref_a = checkpoint_ref_from_tuple(saved_a)
        values_a = cast("Any", saved_a).checkpoint["channel_values"]
        assert values_a["next_event_seq"] == 1
        assert values_a["elapsed_wall_seconds"] == 0.0

    validate_event_a = await sink_a.get_event(run_id=run_id, seq=1)
    assert validate_event_a is not None
    validate_bytes_a = FilesystemEventSink._bytes(validate_event_a)

    handlers_b = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    assert _recovery_config(handlers_b) == current_config
    clock_b = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=first_segment_seconds + 50.0,
    )
    planner_model_b = cast(
        "Any",
        cast("Any", handlers_b.initial_plan_generator.model)._inner,
    )
    capture_deadline(
        planner_model_b,
        "complete",
        "plan-model",
        advance_clock=clock_b,
        advance_seconds=second_segment_seconds,
    )
    sink_b = FilesystemEventSink(event_root, crash_after_node="Plan")
    async with open_sqlite_checkpointer(checkpoint_path) as saver_b:
        exact_b = await saver_b.aget_tuple(checkpoint_config(ref_a))
        assert exact_b is not None
        assert checkpoint_ref_from_tuple(exact_b) == ref_a
        graph_b = build_baseline_graph(
            handlers_b.as_dependencies(checkpointer=saver_b)
        )
        runner_b = LangGraphResearchRunner(
            baseline_graph=graph_b,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_b.monotonic,
                utc_now=clock_b.utc_now,
            ),
        )
        with pytest.raises(
            MemoryError,
            match="crash after filesystem event publication",
        ):
            await runner_b.run(
                run_id=run_id,
                thread_id=thread_id,
                config=current_config,
                checkpoint=ref_a,
                emit=sink_b,
                cancellation_token=CancellationToken(),
            )

    validate_event_b = await sink_b.get_event(run_id=run_id, seq=1)
    plan_event_b = await sink_b.get_event(run_id=run_id, seq=2)
    assert validate_event_b == validate_event_a
    assert FilesystemEventSink._bytes(cast("RunEvent", validate_event_b)) == validate_bytes_a
    assert plan_event_b is not None
    plan_receipt = next(
        receipt
        for artifact_id in plan_event_b.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers_b.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    )
    assert plan_receipt.payload["state"]["elapsed_wall_seconds"] == (
        recovered_active_wall
    ), (
        observed_deadlines,
        invocations,
        clock_b.value,
        plan_event_b.error_code,
    )
    provider_receipt_ids_before_c = tuple(
        cast("list[str]", plan_receipt.payload["provider_receipt_ids"])
    )
    assert len(provider_receipt_ids_before_c) == 1
    provider_result_ids_before_c: set[str] = set()
    plan_provider_receipt = cast("Any", baseline_graph_module)._load_audit_receipt(
        handlers_b.artifact_store,
        provider_receipt_ids_before_c[0],
    )
    assert plan_provider_receipt is not None
    assert plan_provider_receipt.kind == "provider-call"
    plan_provider_record = ProviderCallRecord.model_validate_json(
        json.dumps(
            plan_provider_receipt.payload["record"],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ),
        strict=True,
    )
    assert plan_provider_record.usage.wall_seconds == 0.0
    provider_result_ids_before_c.update(
        cast("list[str]", plan_provider_receipt.payload["result_artifact_ids"])
    )
    assert len(provider_result_ids_before_c) == 1
    for result_id in provider_result_ids_before_c:
        handlers_b.artifact_store.get_bytes(result_id)
    state_before_c = cast("Any", baseline_graph_module)._state_from_payload(
        plan_receipt.payload["state"]
    )
    budget_before_c = state_before_c["budget_snapshot"]
    assert budget_before_c.reserved_search_calls == 0
    assert budget_before_c.reserved_pages == 0
    assert budget_before_c.reserved_tokens == 0
    assert budget_before_c.reserved_wall_seconds == 0.0
    assert budget_before_c.reserved_retries == 0
    assert budget_before_c.reserved_cost_usd in {None, Decimal(0)}
    calls_before_c = invocations.copy()
    cache_files_before_c = tuple(
        sorted(
            path.relative_to(root).as_posix()
            for path in (root / "cache").rglob("*")
            if path.is_file()
        )
    )

    handlers_c = _recovery_composition(
        root,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    provider_receipt_writes_c = 0
    original_put_receipt = cast("Any", baseline_graph_module)._put_audit_receipt

    def observed_put_receipt(*args: object, **kwargs: object) -> str:
        nonlocal provider_receipt_writes_c
        if kwargs.get("kind") == "provider-call":
            provider_receipt_writes_c += 1
        return cast("str", original_put_receipt(*args, **kwargs))

    monkeypatch.setattr(
        baseline_graph_module,
        "_put_audit_receipt",
        observed_put_receipt,
    )
    provider_result_media_types = {
        "application/vnd.deepresearch.model-result+json",
        "application/vnd.deepresearch.embeddings+json",
        "application/vnd.deepresearch.search-results+json",
        "application/vnd.deepresearch.raw-document+json",
        "application/vnd.deepresearch.parsed-document+json",
        "application/vnd.deepresearch.evidence-span+json",
    }
    provider_result_writes_c: list[str] = []
    original_put_bytes_c = handlers_c.artifact_store.put_bytes

    def observed_put_bytes_c(data: bytes, *, media_type: str):  # type: ignore[no-untyped-def]
        if media_type in provider_result_media_types:
            provider_result_writes_c.append(media_type)
        return original_put_bytes_c(data, media_type=media_type)

    monkeypatch.setattr(
        handlers_c.artifact_store,
        "put_bytes",
        observed_put_bytes_c,
    )
    clock_c = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=recovered_active_wall + 100.0,
    )
    planner_model_c = cast(
        "Any",
        cast("Any", handlers_c.initial_plan_generator.model)._inner,
    )
    writer_model_c = cast("Any", cast("Any", handlers_c.writer.model)._inner)
    embedder_c = cast("Any", cast("Any", handlers_c.ranker).embedder)
    inner_embedder_c = cast("Any", embedder_c._inner)
    capture_deadline(planner_model_c, "structured", "query-model")
    capture_deadline(handlers_c.search_provider, "search_with_usage", "search")
    capture_deadline(handlers_c.fetcher, "fetch_with_usage", "fetch")
    capture_deadline(handlers_c.parser, "parse", "parse")
    capture_deadline(handlers_c.ranker, "score", "ranker")
    capture_deadline(inner_embedder_c, "embed", "embed")
    capture_deadline(writer_model_c, "complete", "writer-model")
    sink_c = FilesystemEventSink(event_root)
    async with open_sqlite_checkpointer(checkpoint_path) as saver_c:
        exact_c = await saver_c.aget_tuple(checkpoint_config(ref_a))
        assert exact_c is not None
        assert checkpoint_ref_from_tuple(exact_c) == ref_a
        graph_c = build_baseline_graph(
            handlers_c.as_dependencies(checkpointer=saver_c)
        )
        runner_c = LangGraphResearchRunner(
            baseline_graph=graph_c,
            runtime_hooks=BaselineRuntimeHooks(
                monotonic=clock_c.monotonic,
                utc_now=clock_c.utc_now,
            ),
        )
        result = await runner_c.run(
            run_id=run_id,
            thread_id=thread_id,
            config=current_config,
            checkpoint=ref_a,
            emit=sink_c,
            cancellation_token=CancellationToken(),
        )

    assert await sink_c.get_event(run_id=run_id, seq=1) == validate_event_a
    assert await sink_c.get_event(run_id=run_id, seq=2) == plan_event_b
    expected_plan_deadline = (
        clock_b.monotonic_start
        + max_wall_seconds
        - first_segment_seconds
    )
    assert observed_deadlines["plan-model"] == [expected_plan_deadline]
    expected_c_deadline = (
        clock_c.monotonic_start + max_wall_seconds - recovered_active_wall
    )

    assert result.final_usage.wall_seconds == recovered_active_wall
    assert result.manifest_artifact_id is not None
    manifest = RunManifest.model_validate_json(
        handlers_c.artifact_store.get_bytes(result.manifest_artifact_id),
        strict=True,
    )
    assert manifest.usage.wall_seconds == recovered_active_wall
    expected_provider_wall = 1.25 if expect_completed else 0.0
    assert sum(
        call.usage.wall_seconds for call in manifest.provider_calls
    ) == expected_provider_wall
    assert (manifest.finished_at - manifest.started_at).total_seconds() > (
        manifest.usage.wall_seconds
    )
    assert len(
        [call for call in manifest.provider_calls if call.node == "Plan"]
    ) == 1

    if expect_completed:
        assert result.status == "completed", result.error_code
        expected_deadlines = {
            "query-model": [expected_c_deadline],
            "search": [expected_c_deadline],
            "fetch": [expected_c_deadline, expected_c_deadline],
            "parse": [expected_c_deadline, expected_c_deadline],
            "ranker": [expected_c_deadline],
            "embed": [expected_c_deadline],
            "writer-model": [expected_c_deadline],
        }
        assert {
            key: observed_deadlines[key] for key in expected_deadlines
        } == expected_deadlines
        assert sum(invocations.values()) == 9
    else:
        assert recovered_active_wall == max_wall_seconds
        assert result.status == "failed"
        assert result.error_code == "TIMEOUT"
        assert invocations == calls_before_c
        assert set(observed_deadlines) == {"plan-model"}
        cache_files_after_c = tuple(
            sorted(
                path.relative_to(root).as_posix()
                for path in (root / "cache").rglob("*")
                if path.is_file()
            )
        )
        assert cache_files_after_c == cache_files_before_c
        assert len(manifest.provider_calls) == 1
        assert provider_receipt_writes_c == 0
        assert provider_result_writes_c == []
        reachable_provider_receipt_ids: set[str] = set()
        for seq in range(1, manifest.run_event_count + 1):
            event = await sink_c.get_event(run_id=run_id, seq=seq)
            assert event is not None
            node_receipt = next(
                receipt
                for artifact_id in event.artifact_ids
                if (
                    receipt := cast(
                        "Any",
                        baseline_graph_module,
                    )._load_audit_receipt(
                        handlers_c.artifact_store,
                        artifact_id,
                    )
                )
                is not None
                and receipt.kind == "node-execution"
            )
            reachable_provider_receipt_ids.update(
                cast("list[str]", node_receipt.payload["provider_receipt_ids"])
            )
        assert reachable_provider_receipt_ids == set(
            provider_receipt_ids_before_c
        )
        reachable_provider_result_ids: set[str] = set()
        for receipt_id in reachable_provider_receipt_ids:
            provider_receipt = cast(
                "Any",
                baseline_graph_module,
            )._load_audit_receipt(handlers_c.artifact_store, receipt_id)
            assert provider_receipt is not None
            assert provider_receipt.kind == "provider-call"
            reachable_provider_result_ids.update(
                cast("list[str]", provider_receipt.payload["result_artifact_ids"])
            )
        assert reachable_provider_result_ids == provider_result_ids_before_c
        persist_event = await sink_c.get_event(
            run_id=run_id,
            seq=manifest.run_event_count + 1,
        )
        assert persist_event is not None
        assert persist_event.node == "PersistResults"
        persist_receipt = next(
            receipt
            for artifact_id in persist_event.artifact_ids
            if (
                receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                    handlers_c.artifact_store,
                    artifact_id,
                )
            )
            is not None
            and receipt.kind == "node-execution"
        )
        state_after_c = cast("Any", baseline_graph_module)._state_from_payload(
            persist_receipt.payload["state"]
        )
        budget_after_c = state_after_c["budget_snapshot"]
        assert budget_after_c == budget_before_c
        assert budget_after_c.reserved_search_calls == 0
        assert budget_after_c.reserved_pages == 0
        assert budget_after_c.reserved_tokens == 0
        assert budget_after_c.reserved_wall_seconds == 0.0
        assert budget_after_c.reserved_retries == 0
        assert budget_after_c.reserved_cost_usd in {None, Decimal(0)}


def _owned_handlers(tmp_path: Path) -> BaselineNodeHandlers:
    artifact_store = LocalArtifactStore(tmp_path)
    evidence_store = LocalEvidenceStore(tmp_path)
    model = OfflineModel(_offline_plan())
    unpriced_budget = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": None}
    )

    def boundary(value: str) -> str:
        return value

    return BaselineNodeHandlers(
        initial_plan_generator=FixedPlanner(
            model=model,
            artifact_store=artifact_store,
            budget=unpriced_budget,
            search_depth=1,
            content_boundary=boundary,
        ),
        ranker=SimilarityRanker(OfflineEmbedder()),
        writer=MarkdownReportWriter(
            evidence_store,
            model=model,
            content_boundary=boundary,
        ),
        search_provider=OfflineSearch(),
        fetcher=OfflineFetcher(),
        parser=OfflineParser(),
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=FileCache(tmp_path),
        usage_cost_resolver=ZeroCostResolver(cost=None),
        search_snapshot_id="search-snapshot-v1",
        fetch_snapshot_id="fetch-snapshot-v1",
        code_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        provider_profile_configuration_sha256="c" * 64,
        seed_supported=True,
        pricing_status="unknown",
        pricing_snapshots=(),
        replay_parent="offline-parent-manifest",
    )


async def test_resume_reuses_durable_event_receipt_after_pre_checkpoint_crash(
    tmp_path: Path,
) -> None:
    from deepresearch.workflow import baseline_graph as baseline_graph_module
    handlers = _owned_handlers(tmp_path)
    run_plan_calls = 0
    original_run_plan = handlers.run_plan

    async def counted_run_plan(state: object) -> object:
        nonlocal run_plan_calls
        run_plan_calls += 1
        return await original_run_plan(cast("Any", state))

    handlers.run_plan = cast("Any", counted_run_plan)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    runner = LangGraphResearchRunner(baseline_graph=graph)
    sink = CrashAfterDurableEventSink(crash_seq=2)
    unpriced_budget = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": None}
    )
    current_config = config(
        budget=unpriced_budget,
        prompt_versions=dict(handlers.prompt_versions),
        ranker_weights_version=handlers.ranker_weights_version,
    )

    with pytest.raises(MemoryError, match="crash after durable"):
        await runner.run(
            run_id="run-event-crash",
            thread_id="thread-event-crash",
            config=current_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=CancellationToken(),
        )

    original_event = sink.events[("run-event-crash", 2)]
    assert original_event.usage_delta.total_tokens == 20
    assert original_event.error_code is None
    completion_receipts = [
        cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            artifact_id,
        )
        for artifact_id in original_event.artifact_ids
    ]
    completion = next(
        item for item in completion_receipts if item is not None and item.kind == "node-execution"
    )
    assert len(completion.payload["provider_receipt_ids"]) == 1
    saved = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": "thread-event-crash",
                "checkpoint_ns": "",
            }
        }
    )
    assert saved is not None
    saved_values = cast("Any", saved).checkpoint["channel_values"]
    state_values = {
        key: saved_values[key]
        for key in cast("Any", baseline_graph_module).BaselineState.__annotations__
    }
    cast("Any", baseline_graph_module).validate_baseline_state(state_values)
    cast("Any", handlers)._audit_composition.load_run_header(
        state_values["baseline_work_artifact_ids"],
        run_id="run-event-crash",
        thread_id="thread-event-crash",
        config=current_config,
    )
    checkpoint = checkpoint_ref_from_tuple(saved)
    result = await runner.run(
        run_id="run-event-crash",
        thread_id="thread-event-crash",
        config=current_config,
        checkpoint=checkpoint,
        emit=sink,
        cancellation_token=CancellationToken(),
    )

    assert result.status == "completed", result.error_code
    assert run_plan_calls == 1
    assert [event for event in sink.calls if event.seq == 2] == [
        original_event,
        original_event,
    ]
    assert result.final_usage.total_tokens == 60


@pytest.mark.parametrize(
    "receipt_corruption",
    [
        "state-extra-field",
        "event-seq-float",
        "record-attempt-bool",
        "record-latency-float",
        "record-usage-wall-int",
        "record-usage-cached-bool",
        "record-usage-wall-negative-zero",
    ],
)
async def test_resume_rejects_corrupt_durable_state_receipt_without_handler_reexecution(
    tmp_path: Path,
    receipt_corruption: str,
) -> None:
    from copy import deepcopy

    from deepresearch.workflow import baseline_graph as baseline_graph_module

    handlers = _owned_handlers(tmp_path)
    run_plan_calls = 0
    original_run_plan = handlers.run_plan

    async def counted_run_plan(state: object) -> object:
        nonlocal run_plan_calls
        run_plan_calls += 1
        return await original_run_plan(cast("Any", state))

    handlers.run_plan = cast("Any", counted_run_plan)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    runner = LangGraphResearchRunner(baseline_graph=graph)
    sink = CrashAfterDurableEventSink(crash_seq=2)
    unpriced_budget = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": None}
    )
    current_config = config(
        budget=unpriced_budget,
        prompt_versions=dict(handlers.prompt_versions),
        ranker_weights_version=handlers.ranker_weights_version,
    )

    with pytest.raises(MemoryError, match="crash after durable"):
        await runner.run(
            run_id="run-corrupt-state-receipt",
            thread_id="thread-corrupt-state-receipt",
            config=current_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=CancellationToken(),
        )

    original_event = sink.events[("run-corrupt-state-receipt", 2)]
    completion = next(
        receipt
        for artifact_id in original_event.artifact_ids
        if (
            receipt := cast("Any", baseline_graph_module)._load_audit_receipt(
                handlers.artifact_store,
                artifact_id,
            )
        )
        is not None
        and receipt.kind == "node-execution"
    )
    corrupt_payload = deepcopy(completion.payload)
    if receipt_corruption == "state-extra-field":
        cast("dict[str, object]", corrupt_payload["state"])[
            "ignored_collision"
        ] = True
    elif receipt_corruption == "event-seq-float":
        cast("dict[str, object]", corrupt_payload["event"])["seq"] = 2.0
    else:
        record = cast("dict[str, object]", corrupt_payload["record"])
        _apply_node_record_representation_corruption(record, receipt_corruption)
    corrupt_receipt_id = cast("Any", baseline_graph_module)._put_audit_receipt(
        handlers.artifact_store,
        kind="node-execution",
        run_id=completion.run_id,
        thread_id=completion.thread_id,
        receipt_key=completion.receipt_key,
        payload=corrupt_payload,
    )
    original_completion_id = next(
        artifact_id
        for artifact_id in original_event.artifact_ids
        if cast("Any", baseline_graph_module)._load_audit_receipt(
            handlers.artifact_store,
            artifact_id,
        )
        == completion
    )
    corrupt_event = original_event.model_copy(
        update={
            "artifact_ids": tuple(
                corrupt_receipt_id if item == original_completion_id else item
                for item in original_event.artifact_ids
            )
        }
    )
    sink.events[(corrupt_event.run_id, corrupt_event.seq)] = corrupt_event
    saved = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": "thread-corrupt-state-receipt",
                "checkpoint_ns": "",
            }
        }
    )
    assert saved is not None
    checkpoint = checkpoint_ref_from_tuple(saved)

    result = await runner.run(
        run_id="run-corrupt-state-receipt",
        thread_id="thread-corrupt-state-receipt",
        config=current_config,
        checkpoint=checkpoint,
        emit=sink,
        cancellation_token=CancellationToken(),
    )

    after = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": "thread-corrupt-state-receipt",
                "checkpoint_ns": "",
            }
        }
    )
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert result.manifest_artifact_id is None
    assert run_plan_calls == 1
    assert sink.events[(corrupt_event.run_id, corrupt_event.seq)] == corrupt_event
    assert after is not None
    if receipt_corruption in {"state-extra-field", "event-seq-float"}:
        assert checkpoint_ref_from_tuple(after) == checkpoint
    else:
        original_values = cast("Any", saved).checkpoint["channel_values"]
        after_values = cast("Any", after).checkpoint["channel_values"]
        assert after_values["next_event_seq"] == original_values["next_event_seq"]


@pytest.mark.parametrize(
    "corruption",
    ["seq-bool", "usage-wall-int", "payload-map-subclass", "payload-str-subclass"],
)
async def test_resume_rejects_constructed_durable_event_before_handler(
    tmp_path: Path,
    corruption: str,
) -> None:
    handlers = _owned_handlers(tmp_path)
    validate_calls = 0
    original_validate = handlers.validate_request

    async def counted_validate(state: object) -> object:
        nonlocal validate_calls
        validate_calls += 1
        return await original_validate(cast("Any", state))

    handlers.validate_request = cast("Any", counted_validate)
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    fixed_now = datetime(2026, 9, 1, tzinfo=UTC)
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=lambda: 0.0,
            utc_now=lambda: fixed_now,
        ),
    )
    sink = CrashAfterDurableEventSink(crash_seq=1)
    unpriced_budget = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": None}
    )
    current_config = config(
        budget=unpriced_budget,
        prompt_versions=dict(handlers.prompt_versions),
        ranker_weights_version=handlers.ranker_weights_version,
    )
    run_id = f"run-corrupt-event-{corruption}"
    thread_id = f"thread-corrupt-event-{corruption}"

    with pytest.raises(MemoryError, match="crash after durable"):
        await runner.run(
            run_id=run_id,
            thread_id=thread_id,
            config=current_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=CancellationToken(),
        )

    original_event = sink.events[(run_id, 1)]
    payload_observations: list[str] = []
    if corruption == "seq-bool":
        corrupt_event = cast(
            "RunEvent",
            BaseModel.model_copy(original_event, update={"seq": True}),
        )
    elif corruption == "usage-wall-int":
        corrupt_usage = cast(
            "ResourceUsage",
            BaseModel.model_copy(
                original_event.usage_delta,
                update={"wall_seconds": 0},
            ),
        )
        corrupt_event = cast(
            "RunEvent",
            BaseModel.model_copy(
                original_event,
                update={"usage_delta": corrupt_usage},
            ),
        )
    elif corruption == "payload-map-subclass":
        class ObservedPayload(dict[str, object]):
            def __iter__(self) -> Iterator[str]:
                payload_observations.append("iter")
                return super().__iter__()

        corrupt_event = cast(
            "RunEvent",
            BaseModel.model_copy(
                original_event,
                update={"public_payload": ObservedPayload({"message": "corrupt"})},
            ),
        )
    else:
        class StringSubclass(str):
            pass

        payload_type = type(original_event.public_payload)
        corrupt_event = cast(
            "RunEvent",
            BaseModel.model_copy(
                original_event,
                update={
                    "public_payload": payload_type(
                        {"message": StringSubclass("corrupt")}
                    )
                },
            ),
        )
    sink.events[(run_id, 1)] = corrupt_event
    saved = await saver.aget_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert saved is not None
    checkpoint = checkpoint_ref_from_tuple(saved)

    result = await runner.run(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        checkpoint=checkpoint,
        emit=sink,
        cancellation_token=CancellationToken(),
    )

    after = await saver.aget_tuple(
        {"configurable": {"thread_id": thread_id, "checkpoint_ns": ""}}
    )
    assert result.status == "failed"
    assert result.error_code == "DATA_CORRUPTION"
    assert validate_calls == 1
    assert payload_observations == []
    assert after is not None
    assert checkpoint_ref_from_tuple(after) == checkpoint


def test_handlers_reject_rebinding_components_owned_by_another_handler(
    tmp_path: Path,
) -> None:
    handlers = _owned_handlers(tmp_path)

    with pytest.raises(TypeError, match="already bound"):
        BaselineNodeHandlers(
            initial_plan_generator=handlers.initial_plan_generator,
            ranker=handlers.ranker,
            writer=handlers.writer,
            search_provider=handlers.search_provider,
            fetcher=handlers.fetcher,
            parser=handlers.parser,
            artifact_store=handlers.artifact_store,
            evidence_store=handlers.evidence_store,
            cache=handlers.cache,
            usage_cost_resolver=handlers.usage_cost_resolver,
            search_snapshot_id=handlers.search_snapshot_id,
            fetch_snapshot_id=handlers.fetch_snapshot_id,
            code_commit=handlers.code_commit,
            dependency_lock_sha256=handlers.dependency_lock_sha256,
            provider_profile_configuration_sha256=(
                handlers.provider_profile_configuration_sha256
            ),
            seed_supported=handlers.seed_supported,
            pricing_status=handlers.pricing_status,
            pricing_snapshots=handlers.pricing_snapshots,
            replay_parent=handlers.replay_parent,
        )


async def test_planner_state_is_restored_after_hard_cancellation(tmp_path: Path) -> None:
    handlers = _owned_handlers(tmp_path)
    planner = cast("Any", handlers.initial_plan_generator)
    before = (
        planner.budget,
        planner._initial_model_tokens,
        planner._model_tokens_used,
        planner._model_tokens_reserved,
        planner._query_cache,
        planner._query_locks,
    )

    async def cancel() -> None:
        raise asyncio.CancelledError

    with pytest.raises(asyncio.CancelledError):
        await cast("Any", handlers)._isolated_planner_call(
            _runtime_context(ticks=iter(())),
            cancel,
        )

    after = (
        planner.budget,
        planner._initial_model_tokens,
        planner._model_tokens_used,
        planner._model_tokens_reserved,
        planner._query_cache,
        planner._query_locks,
    )
    assert after == before
    assert after[4] is before[4]
    assert after[5] is before[5]


async def test_invocation_audit_uses_only_each_runtime_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch.workflow import runner as runner_module

    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    sink = MemoryEventSink()
    fixed_now = datetime(2026, 9, 1, tzinfo=UTC)
    accountants: dict[str, BudgetAccountant] = {}

    def make_context(run_id: str, thread_id: str) -> BaselineRuntimeContext:
        accountant = BudgetAccountant(current_config.budget, run_scope=run_id)
        accountants[run_id] = accountant
        return BaselineRuntimeContext(
            run_id=run_id,
            thread_id=thread_id,
            config=current_config,
            emit=sink,
            cancellation_token=CancellationToken(),
            budget_accountant=accountant,
            deadline=100.0,
            run_started_monotonic=0.0,
            run_started_at=fixed_now,
            elapsed_base_seconds=0.0,
            monotonic=lambda: 0.0,
            utc_now=lambda: fixed_now,
            new_id=lambda prefix: f"{prefix}-fixed",
        )

    context_a = make_context("run-audit-hard", "thread-audit-hard")
    context_b = make_context("run-audit-survivor", "thread-audit-survivor")
    assert context_a.audit is not context_b.audit
    for context in (context_a, context_b):
        context.audit.provider_receipt_ids.append("stale-receipt")
        context.audit.result_artifact_ids.append("sha256:" + "f" * 64)
        context.audit.identity_counts["stale"] = 1
        context.audit.operation_counts["stale"] = 1

    state_a = cast("Any", runner_module)._initial_state(
        run_id=context_a.run_id,
        thread_id=context_a.thread_id,
        config=current_config,
        accountant=accountants[context_a.run_id],
    )
    state_b = cast("Any", runner_module)._initial_state(
        run_id=context_b.run_id,
        thread_id=context_b.thread_id,
        config=current_config,
        accountant=accountants[context_b.run_id],
    )
    task_contexts: dict[object, BaselineRuntimeContext] = {}

    def fake_runtime(_schema: object) -> object:
        return SimpleNamespace(context=task_contexts[asyncio.current_task()])

    monkeypatch.setattr(baseline_graph_module, "get_runtime", fake_runtime)
    arrivals = 0
    both_arrived = asyncio.Event()
    observations: dict[str, tuple[bool, tuple[object, ...]]] = {}
    hard_primary = MemoryError("hard audit isolation probe")

    async def handler(_state: BaselineState) -> dict[str, object]:
        nonlocal arrivals
        context = task_contexts[asyncio.current_task()]
        arrivals += 1
        if arrivals == 2:
            both_arrived.set()
        await asyncio.wait_for(both_arrived.wait(), timeout=5.0)
        resolved = cast("Any", baseline_graph_module)._audit_buffer(context)
        observations[context.run_id] = (
            resolved is context.audit,
            (
                *resolved.provider_receipt_ids,
                *resolved.result_artifact_ids,
                *resolved.identity_counts,
                *resolved.operation_counts,
            ),
        )
        if context is context_a:
            raise hard_primary
        return {}

    safe = cast("Any", baseline_graph_module)._safe_node(
        "ValidateRequest",
        handler,
        audit_composition=None,
    )
    accountant_attributes_before = {
        run_id: set(vars(accountant)) for run_id, accountant in accountants.items()
    }

    async def invoke(
        context: BaselineRuntimeContext,
        current_state: BaselineState,
    ) -> object:
        task_contexts[asyncio.current_task()] = context
        return await safe(current_state)

    results = await asyncio.gather(
        invoke(context_a, state_a),
        invoke(context_b, state_b),
        return_exceptions=True,
    )

    assert results[0] is hard_primary
    assert cast("dict[str, object]", results[1])["next_event_seq"] == 2
    assert observations == {
        "run-audit-hard": (True, ()),
        "run-audit-survivor": (True, ()),
    }
    assert {
        run_id: set(vars(accountant)) for run_id, accountant in accountants.items()
    } == accountant_attributes_before
    assert cast("Any", baseline_graph_module)._audit_buffer(context_a) is context_a.audit
    assert cast("Any", baseline_graph_module)._audit_buffer(context_b) is context_b.audit


async def test_invocation_audit_resets_before_existing_event_recovery(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from deepresearch.workflow import runner as runner_module

    current_config = config(
        budget=RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    )
    run_id = "run-audit-recovery"
    thread_id = "thread-audit-recovery"
    accountant = BudgetAccountant(current_config.budget, run_scope=run_id)
    sink = MemoryEventSink()
    fixed_now = datetime(2026, 9, 1, tzinfo=UTC)
    context = BaselineRuntimeContext(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        emit=sink,
        cancellation_token=CancellationToken(),
        budget_accountant=accountant,
        deadline=100.0,
        run_started_monotonic=0.0,
        run_started_at=fixed_now,
        elapsed_base_seconds=0.0,
        monotonic=lambda: 0.0,
        utc_now=lambda: fixed_now,
        new_id=lambda prefix: f"{prefix}-fixed",
    )
    state = cast("Any", runner_module)._initial_state(
        run_id=run_id,
        thread_id=thread_id,
        config=current_config,
        accountant=accountant,
    )
    monkeypatch.setattr(
        baseline_graph_module,
        "get_runtime",
        lambda _schema: SimpleNamespace(context=context),
    )

    normal_calls = 0

    async def normal_handler(_state: BaselineState) -> dict[str, object]:
        nonlocal normal_calls
        normal_calls += 1
        return {}

    normal_safe = cast("Any", baseline_graph_module)._safe_node(
        "ValidateRequest",
        normal_handler,
        audit_composition=None,
    )
    first_update = await normal_safe(state)
    recovered_input = cast("Any", baseline_graph_module).validate_baseline_state(
        {**state, **first_update}
    )
    assert recovered_input["next_event_seq"] == 2
    assert normal_calls == 1

    existing = RunEvent(
        seq=2,
        run_id=run_id,
        timestamp=fixed_now,
        node="Plan",
        kind="node_completed",
        status="running",
        public_payload={"is_partial": False, "stop_reason": None},
        usage_delta=ResourceUsage.zero(cost_known=True),
        artifact_ids=(),
        error_code=None,
    )
    await sink(existing)
    context.audit.graph_node = "stale-node"
    context.audit.node_attempt = 99
    context.audit.provider_calls.append(cast("Any", object()))
    context.audit.provider_receipt_ids.append("stale-provider")
    context.audit.result_artifact_ids.append("stale-result")
    context.audit.child_receipt_ids.append("stale-child")
    context.audit.identity_counts["stale"] = 1
    context.audit.operation_counts["stale"] = 1
    context.audit.pending_provider_call = cast("Any", object())
    recovery_observations: list[tuple[object, ...]] = []

    async def inspect_recovery(**kwargs: object) -> dict[str, object]:
        assert kwargs["event"] == existing
        audit = context.audit
        recovery_observations.append(
            (
                audit.graph_node,
                audit.node_attempt,
                tuple(audit.provider_calls),
                tuple(audit.provider_receipt_ids),
                tuple(audit.result_artifact_ids),
                tuple(audit.child_receipt_ids),
                dict(audit.identity_counts),
                dict(audit.operation_counts),
                audit.pending_provider_call,
            )
        )
        return {"next_event_seq": 3}

    monkeypatch.setattr(
        baseline_graph_module,
        "_recover_durable_node_event",
        inspect_recovery,
    )
    recovered_handler_calls = 0

    async def forbidden_handler(_state: BaselineState) -> dict[str, object]:
        nonlocal recovered_handler_calls
        recovered_handler_calls += 1
        return {}

    recovery_safe = cast("Any", baseline_graph_module)._safe_node(
        "Plan",
        forbidden_handler,
        audit_composition=cast("Any", object()),
    )
    recovered_update = await recovery_safe(recovered_input)

    assert recovered_update == {"next_event_seq": 3}
    assert recovered_handler_calls == 0
    assert recovery_observations == [
        (None, 0, (), (), (), (), {}, {}, None)
    ]


async def test_concurrent_runs_share_graph_without_context_or_receipt_leakage(
    tmp_path: Path,
) -> None:
    invocations: Counter[str] = Counter()
    persist_entries: Counter[str] = Counter()
    handlers = _recovery_composition(
        tmp_path,
        invocations=invocations,
        persist_entries=persist_entries,
    )
    planner = cast("Any", handlers.initial_plan_generator)
    planner_before = (
        planner.budget,
        planner._initial_model_tokens,
        planner._model_tokens_used,
        planner._model_tokens_reserved,
        planner._query_cache,
        planner._query_locks,
    )
    cancelled_token = CancellationToken()
    surviving_token = CancellationToken()
    planner_model = cast("Any", handlers.initial_plan_generator.model)._inner
    original_complete = planner_model.complete

    async def cancel_after_first_model(
        request_value: ModelRequest,
        **kwargs: object,
    ) -> ModelResult[str]:
        result = await original_complete(request_value, **kwargs)
        if any(
            "cancelled concurrent request" in message.content
            for message in request_value.messages
        ):
            cancelled_token.cancel()
        return result

    planner_model.complete = cancel_after_first_model
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    sink = ConcurrentPlanBarrierSink()
    clock = ControlledSegmentClock(
        monotonic_start=time.monotonic(),
        utc_offset_seconds=0.0,
    )
    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(
            monotonic=clock.monotonic,
            utc_now=clock.utc_now,
        ),
    )
    base_config = _recovery_config(handlers)
    cancelled_config = base_config.model_copy(
        update={
            "request": base_config.request.model_copy(
                update={"question": "cancelled concurrent request"}
            )
        }
    )
    surviving_config = base_config.model_copy(
        update={
            "request": base_config.request.model_copy(
                update={"question": "surviving concurrent request"}
            )
        }
    )

    cancelled_result, surviving_result = await asyncio.gather(
        runner.run(
            run_id="run-concurrent-cancelled",
            thread_id="thread-concurrent-cancelled",
            config=cancelled_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=cancelled_token,
        ),
        runner.run(
            run_id="run-concurrent-surviving",
            thread_id="thread-concurrent-surviving",
            config=surviving_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=surviving_token,
        ),
    )

    assert cancelled_result.status == "cancelled"
    assert cancelled_result.error_code == "CANCELLED"
    assert cancelled_result.final_usage.total_tokens == 20
    assert surviving_result.status == "completed", surviving_result.error_code
    assert surviving_result.final_usage.total_tokens == 60
    assert cancelled_result.manifest_artifact_id is not None
    assert surviving_result.manifest_artifact_id is not None
    assert (
        cancelled_result.manifest_artifact_id
        != surviving_result.manifest_artifact_id
    )
    assert sink.plan_runs == {
        "run-concurrent-cancelled",
        "run-concurrent-surviving",
    }
    assert {
        event.seq
        for event in sink.events.values()
        if event.node == "Plan"
    } == {2}
    for run_id in sink.plan_runs:
        run_events = sorted(
            event.seq
            for (event_run_id, _), event in sink.events.items()
            if event_run_id == run_id
        )
        assert run_events == list(range(1, len(run_events) + 1))
    cancelled_manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(cancelled_result.manifest_artifact_id),
        strict=True,
    )
    surviving_manifest = RunManifest.model_validate_json(
        handlers.artifact_store.get_bytes(surviving_result.manifest_artifact_id),
        strict=True,
    )
    assert cancelled_manifest.run_id == "run-concurrent-cancelled"
    assert cancelled_manifest.thread_id == "thread-concurrent-cancelled"
    assert cancelled_manifest.usage.total_tokens == 20
    assert surviving_manifest.run_id == "run-concurrent-surviving"
    assert surviving_manifest.thread_id == "thread-concurrent-surviving"
    assert surviving_manifest.usage.total_tokens == 60
    assert persist_entries == Counter({"persist-results": 2})
    planner_after = (
        planner.budget,
        planner._initial_model_tokens,
        planner._model_tokens_used,
        planner._model_tokens_reserved,
        planner._query_cache,
        planner._query_locks,
    )
    assert planner_after == planner_before
    assert planner_after[4] is planner_before[4]
    assert planner_after[5] is planner_before[5]


async def test_fully_offline_baseline_runs_p1_r1_writer_and_persists_stable_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact_store = LocalArtifactStore(tmp_path)
    durable_reads: dict[str, int] = {}
    durable_get = artifact_store.get_bytes

    def observe_durable_read(artifact_id: str) -> bytes:
        durable_reads[artifact_id] = durable_reads.get(artifact_id, 0) + 1
        return durable_get(artifact_id)

    monkeypatch.setattr(artifact_store, "get_bytes", observe_durable_read)
    evidence_store = LocalEvidenceStore(tmp_path)
    model = OfflineModel(_offline_plan())
    writer_model = OfflineModel(_offline_plan())
    unpriced_budget = RunBudget.preset("low").model_copy(
        update={"max_cost_usd": None}
    )

    def boundary(value: str) -> str:
        return value.replace("UNTRUSTED", "BOUNDARY_APPLIED")

    planner = FixedPlanner(
        model=model,
        artifact_store=artifact_store,
        budget=unpriced_budget,
        search_depth=1,
        content_boundary=boundary,
    )
    handlers = BaselineNodeHandlers(
        initial_plan_generator=planner,
        ranker=SimilarityRanker(OfflineEmbedder()),
        writer=MarkdownReportWriter(
            evidence_store,
            model=writer_model,
            content_boundary=boundary,
        ),
        search_provider=OfflineSearch(),
        fetcher=OfflineFetcher(),
        parser=OfflineParser(),
        artifact_store=artifact_store,
        evidence_store=evidence_store,
        cache=FileCache(tmp_path),
        usage_cost_resolver=ZeroCostResolver(cost=None),
        search_snapshot_id="search-snapshot-v1",
        fetch_snapshot_id="fetch-snapshot-v1",
        code_commit="a" * 40,
        dependency_lock_sha256="b" * 64,
        provider_profile_configuration_sha256="c" * 64,
        seed_supported=True,
        pricing_status="unknown",
        pricing_snapshots=(),
        replay_parent="offline-parent-manifest",
    )
    saver = InMemorySaver(serde=checkpoint_serializer())
    graph = build_baseline_graph(handlers.as_dependencies(checkpointer=saver))
    monotonic_value = time.monotonic()
    monotonic_origin = monotonic_value
    utc_origin = datetime(2026, 8, 29, tzinfo=UTC)

    def monotonic() -> float:
        nonlocal monotonic_value
        monotonic_value += 0.5
        return monotonic_value

    def utc_now() -> datetime:
        return utc_origin + timedelta(seconds=monotonic_value - monotonic_origin)

    runner = LangGraphResearchRunner(
        baseline_graph=graph,
        runtime_hooks=BaselineRuntimeHooks(monotonic=monotonic, utc_now=utc_now),
    )
    sink = MemoryEventSink()

    current_config = RunConfig(
        request=request(),
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
        budget=unpriced_budget,
        prompt_versions=dict(handlers.prompt_versions),
        ranker_weights_version=handlers.ranker_weights_version,
        seed=0,
    )
    result = await runner.run(
        run_id="run-offline",
        thread_id="thread-offline",
        config=current_config,
        checkpoint=None,
        emit=sink,
        cancellation_token=CancellationToken(),
    )
    events = {
        seq: event
        for (event_run_id, seq), event in sink.events.items()
        if event_run_id == "run-offline"
    }

    assert result.status == "completed", result.error_code
    assert result.stop_reason == "SUFFICIENT"
    assert result.is_partial is False
    assert result.report_artifact_id is not None
    report = artifact_store.get_bytes(result.report_artifact_id).decode()
    assert "## References" in report
    assert "UNTRUSTED" not in writer_model.requests[-1].messages[-1].content
    assert "BOUNDARY_APPLIED" in writer_model.requests[-1].messages[-1].content
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    assert artifact_store.exists(result.evidence_graph_artifact_id)
    assert artifact_store.exists(result.manifest_artifact_id)
    assert durable_reads.get(result.manifest_artifact_id, 0) >= 1
    manifest = RunManifest.model_validate_json(
        artifact_store.get_bytes(result.manifest_artifact_id)
    )
    assert manifest.run_id == "run-offline"
    assert manifest.thread_id == "thread-offline"
    assert manifest.workflow_id == "baseline-v1"
    assert manifest.stop_reason == "SUFFICIENT"
    assert manifest.run_event_count == 11
    assert manifest.manifest_sha256 == manifest.canonical_sha256()
    assert result.evidence_graph_artifact_id in manifest.artifact_ids
    assert result.report_artifact_id in manifest.artifact_ids
    assert len(manifest.provider_calls) == 9
    assert [call.operation for call in manifest.provider_calls] == [
        "model",
        "model",
        "search",
        "fetch",
        "fetch",
        "parse",
        "parse",
        "embed",
        "model",
    ]
    assert [call.node for call in manifest.provider_calls] == [
        "Plan",
        "Search",
        "Search",
        "Fetch",
        "Fetch",
        "ParseAndNormalize",
        "ParseAndNormalize",
        "RankEvidence",
        "DraftReport",
    ]
    assert len(manifest.node_executions) == 11
    assert len(manifest.parsed_artifacts) == 2
    assert len(manifest.evidence_hashes) == 2
    assert manifest.model_ids == ("offline-model-v1",)
    assert manifest.cache_hit_count == 0
    assert manifest.replay_parent == "offline-parent-manifest"
    assert manifest.pricing_status == "unknown"
    assert manifest.pricing_snapshots == ()
    assert manifest.budget.max_cost_usd is None
    assert manifest.usage.cost_usd is None
    expected_event_hash = hashlib.sha256(
        json.dumps(
            [cast("Any", events[index]).model_dump(mode="json") for index in range(1, 12)],
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    assert manifest.run_events_sha256 == expected_event_hash
    assert all(artifact_store.exists(item) for item in manifest.artifact_ids)
    assert tuple(events) == tuple(range(1, 13))
    assert result.final_usage.search_calls == 1
    assert result.final_usage.pages == 2
    assert result.final_usage.total_tokens == 60
    provider_wall_seconds = sum(
        call.usage.wall_seconds for call in manifest.provider_calls
    )
    manifest_wall_seconds = (
        manifest.finished_at - manifest.started_at
    ).total_seconds()
    assert provider_wall_seconds == 2.75
    assert result.final_usage.wall_seconds == manifest.usage.wall_seconds
    assert manifest.usage.wall_seconds == manifest_wall_seconds
    assert result.final_usage.wall_seconds != provider_wall_seconds
    saved = await saver.aget_tuple(
        {
            "configurable": {
                "thread_id": "thread-offline",
                "checkpoint_ns": "",
            }
        }
    )
    assert saved is not None
    from deepresearch.workflow import baseline_graph as baseline_graph_module

    work_ids = cast("Any", saved).checkpoint["channel_values"][
        "baseline_work_artifact_ids"
    ]
    receipts = {
        artifact_id: cast("Any", baseline_graph_module)._load_audit_receipt(
            artifact_store,
            artifact_id,
        )
        for artifact_id in work_ids
    }
    store_receipt = next(
        receipt
        for receipt in receipts.values()
        if receipt is not None
        and receipt.kind == "node-execution"
        and receipt.payload["record"]["node"] == "StoreEvidence"
    )
    typed_receipt_ids = {
        artifact_id
        for artifact_id, receipt in receipts.items()
        if receipt is not None
        and receipt.kind in {"parsed-artifact", "evidence-hash"}
    }
    assert set(store_receipt.payload["child_receipt_ids"]) == typed_receipt_ids
    state_values = {
        name: cast("Any", saved).checkpoint["channel_values"][name]
        for name in cast("Any", baseline_graph_module).BaselineState.__annotations__
    }
    final_state = cast("Any", baseline_graph_module).validate_baseline_state(
        state_values
    )
    typed_receipts = [receipts[item] for item in typed_receipt_ids]
    original_get = artifact_store.get_bytes
    parsed_receipts = [
        item for item in typed_receipts if item.kind == "parsed-artifact"
    ]
    evidence_receipts = [
        item for item in typed_receipts if item.kind == "evidence-hash"
    ]
    assert len(parsed_receipts) == 2
    assert len(evidence_receipts) == 2
    for target, replacement in (
        (parsed_receipts[0], evidence_receipts[0]),
        (evidence_receipts[0], parsed_receipts[0]),
        (parsed_receipts[0], parsed_receipts[1]),
        (evidence_receipts[0], evidence_receipts[1]),
    ):
        target_artifact_id = target.payload["record"]["artifact_id"]
        replacement_artifact_id = replacement.payload["record"]["artifact_id"]

        def corrupt_typed_artifact(
            artifact_id: str,
            *,
            target_id: str = target_artifact_id,
            replacement_id: str = replacement_artifact_id,
        ) -> bytes:
            return (
                original_get(replacement_id)
                if artifact_id == target_id
                else original_get(artifact_id)
            )

        monkeypatch.setattr(artifact_store, "get_bytes", corrupt_typed_artifact)
        with pytest.raises(ArtifactIntegrityError, match="typed|parsed|evidence"):
            cast("Any", baseline_graph_module)._validate_typed_receipt_closure(
                artifact_store=artifact_store,
                evidence_store=evidence_store,
                receipts=typed_receipts,
                state=final_state,
                normalization_version=handlers.normalization_version,
            )
        monkeypatch.setattr(artifact_store, "get_bytes", original_get)
    assert handlers.plan.initial_plan_generator is handlers.initial_plan_generator
    assert handlers.model_ids["planner"] == "offline-model-v1"
