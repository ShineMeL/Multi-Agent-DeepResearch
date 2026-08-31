from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, cast

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import (  # pyright: ignore[reportMissingTypeStubs]
    CompiledStateGraph,
)

from deepresearch.domain import ResourceUsage, RunConfig, RunEvent, RunResult
from deepresearch.planning import PlanGenerationError
from deepresearch.providers import ProviderError
from deepresearch.runtime import (
    BudgetAccountant,
    CancellationToken,
    CheckpointRef,
    OperationCancelled,
)
from deepresearch.runtime.checkpoints import (
    CheckpointIdentityError,
    checkpoint_config,
    checkpoint_ref_from_tuple,
)
from deepresearch.storage import (
    ArtifactIntegrityError,
    CacheIntegrityError,
    EvidenceIntegrityError,
)

from .baseline_graph import (
    BaselineRuntimeContext,
    DurableRunEventSink,
    WorkflowInvariantError,
)
from .state import BaselineState, StateValidationError, validate_baseline_state


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4()}"


@dataclass(frozen=True)
class BaselineRuntimeHooks:
    monotonic: Callable[[], float] = time.monotonic
    utc_now: Callable[[], datetime] = lambda: datetime.now(UTC)
    new_id: Callable[[str], str] = _new_id


def _config_sha256(config: RunConfig) -> str:
    payload = json.dumps(
        config.model_dump(mode="json"),
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _usage_from_accountant(
    accountant: BudgetAccountant,
    *,
    run_wall_seconds: float,
) -> ResourceUsage:
    if (
        type(run_wall_seconds) not in {int, float}
        or not math.isfinite(float(run_wall_seconds))
        or run_wall_seconds < 0.0
    ):
        raise ValueError("run wall must be finite and non-negative")
    snapshot = accountant.snapshot()
    input_tokens = sum(item.input_tokens for item in snapshot.used_by_node.values())
    output_tokens = sum(item.output_tokens for item in snapshot.used_by_node.values())
    reasoning_tokens = sum(item.reasoning_tokens for item in snapshot.used_by_node.values())
    cached_tokens = sum(item.cached_tokens for item in snapshot.used_by_node.values())
    return ResourceUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_tokens=cached_tokens,
        total_tokens=snapshot.used_tokens,
        search_calls=snapshot.used_search_calls,
        pages=snapshot.used_pages,
        retries=snapshot.used_retries,
        wall_seconds=float(run_wall_seconds),
        cost_usd=snapshot.used_cost_usd,
    )


def _cumulative_wall(*, base: float, start: float, now: float) -> float:
    delta = now - start
    if not math.isfinite(delta) or delta < 0.0:
        delta = 0.0
    total = base + delta
    return total if math.isfinite(total) else base


def _initial_state(
    *,
    run_id: str,
    thread_id: str,
    config: RunConfig,
    accountant: BudgetAccountant,
    baseline_work_artifact_ids: tuple[str, ...] = (),
) -> BaselineState:
    return BaselineState(
        run_id=run_id,
        thread_id=thread_id,
        request=config.request,
        config_sha256=_config_sha256(config),
        plan_id=None,
        plan_artifact_id=None,
        pending_subquestion_ids=(),
        active_subquestion_id=None,
        query_ids=(),
        source_ids=(),
        evidence_ids=(),
        selected_evidence_ids=(),
        coverage_ledger=(),
        high_priority_unresolved_conflict_ids=(),
        blocked_needs=(),
        recent_marginal_gains=(),
        baseline_work_artifact_ids=baseline_work_artifact_ids,
        budget_snapshot=accountant.snapshot(),
        stop_reason=None,
        is_partial=False,
        draft_artifact_id=None,
        report_artifact_id=None,
        evidence_graph_artifact_id=None,
        manifest_artifact_id=None,
        next_event_seq=1,
        failed_node=None,
        elapsed_wall_seconds=0.0,
        error_code=None,
    )


def _failed_result(
    *,
    run_id: str,
    thread_id: str,
    usage: ResourceUsage,
    error_code: str,
    stop_reason: str | None = None,
    is_partial: bool = False,
    report_artifact_id: str | None = None,
    evidence_graph_artifact_id: str | None = None,
    manifest_artifact_id: str | None = None,
) -> RunResult:
    return RunResult(
        run_id=run_id,
        thread_id=thread_id,
        status="cancelled" if error_code == "CANCELLED" else "failed",
        stop_reason=cast("Any", stop_reason),
        is_partial=is_partial,
        report_artifact_id=report_artifact_id,
        evidence_graph_artifact_id=evidence_graph_artifact_id,
        manifest_artifact_id=manifest_artifact_id,
        final_usage=usage,
        error_code=error_code,
    )


def _result_from_state(state: BaselineState, usage: ResourceUsage) -> RunResult:
    error_code = state["error_code"]
    stop_reason = state["stop_reason"]
    report_id = state["report_artifact_id"]
    if error_code is not None:
        return _failed_result(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            usage=usage,
            error_code=error_code,
            is_partial=state["is_partial"],
            report_artifact_id=report_id,
            evidence_graph_artifact_id=state["evidence_graph_artifact_id"],
            manifest_artifact_id=state["manifest_artifact_id"],
        )
    if stop_reason is not None and report_id is None:
        return _failed_result(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            usage=usage,
            error_code="REPORT_MISSING",
            stop_reason=stop_reason,
        )
    if stop_reason == "SUFFICIENT" and report_id is not None:
        partial = False
    elif stop_reason in {"PLATEAU", "BUDGET_EXHAUSTED", "BLOCKED"} and report_id:
        partial = True
    else:
        return _failed_result(
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            usage=usage,
            error_code="NO_LEGAL_CONTINUATION",
        )
    return RunResult(
        run_id=state["run_id"],
        thread_id=state["thread_id"],
        status="completed",
        stop_reason=stop_reason,
        is_partial=partial,
        report_artifact_id=report_id,
        evidence_graph_artifact_id=state["evidence_graph_artifact_id"],
        manifest_artifact_id=state["manifest_artifact_id"],
        final_usage=usage,
    )


def _exception_code(error: Exception) -> str:
    if isinstance(error, OperationCancelled):
        return "CANCELLED"
    if isinstance(error, PlanGenerationError):
        return "PLAN_INVALID"
    if isinstance(error, ProviderError):
        return error.code
    if isinstance(error, WorkflowInvariantError):
        return error.code
    if isinstance(error, CheckpointIdentityError):
        return "CHECKPOINT_MISMATCH"
    if isinstance(
        error,
        (
            StateValidationError,
            ArtifactIntegrityError,
            CacheIntegrityError,
            EvidenceIntegrityError,
        ),
    ):
        return "DATA_CORRUPTION"
    return "INTERNAL_ERROR"


class LangGraphResearchRunner:
    def __init__(
        self,
        *,
        baseline_graph: CompiledStateGraph[
            BaselineState,
            BaselineRuntimeContext,
            BaselineState,
            BaselineState,
        ],
        runtime_hooks: BaselineRuntimeHooks | None = None,
    ) -> None:
        self._baseline_graph = baseline_graph
        self._runtime_hooks = runtime_hooks or BaselineRuntimeHooks()

    async def run(
        self,
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig,
        checkpoint: CheckpointRef | None,
        emit: Callable[[RunEvent], Awaitable[None]],
        cancellation_token: CancellationToken,
    ) -> RunResult:
        if (
            config.workflow_id != "baseline-v1"
            or config.planner_id != "P1"
            or config.ranker_id != "R1"
        ):
            accountant = BudgetAccountant(config.budget, run_scope=run_id or "invalid")
            return _failed_result(
                run_id=run_id,
                thread_id=thread_id,
                usage=_usage_from_accountant(accountant, run_wall_seconds=0.0),
                error_code="INVALID_WORKFLOW_CONFIG",
            )
        if not isinstance(emit, DurableRunEventSink):
            accountant = BudgetAccountant(config.budget, run_scope=run_id or "invalid")
            return _failed_result(
                run_id=run_id,
                thread_id=thread_id,
                usage=_usage_from_accountant(accountant, run_wall_seconds=0.0),
                error_code="INVALID_EVENT_SINK",
            )

        start = self._runtime_hooks.monotonic()
        run_started_at = self._runtime_hooks.utc_now()
        effective_budget = config.budget
        accountant = BudgetAccountant(config.budget, run_scope=run_id or "invalid")
        input_state: BaselineState | None
        graph_config: dict[str, dict[str, str]]
        elapsed_before_resume = 0.0
        context: BaselineRuntimeContext | None = None
        try:
            cancellation_token.raise_if_cancelled()
            graph = cast("Any", self._baseline_graph)
            audit_composition = getattr(
                graph,
                "_baseline_audit_composition",
                None,
            )
            saver = cast("BaseCheckpointSaver[str] | bool | None", graph.checkpointer)
            if saver is None or isinstance(saver, bool):
                raise CheckpointIdentityError()
            saver_api = cast("Any", saver)
            if checkpoint is None:
                existing = await saver_api.aget_tuple(
                    {
                        "configurable": {
                            "thread_id": thread_id,
                            "checkpoint_ns": "",
                        }
                    }
                )
                if existing is not None:
                    raise CheckpointIdentityError()
                accountant = BudgetAccountant(effective_budget, run_scope=run_id)
                audit_artifact_ids: tuple[str, ...] = ()
                if audit_composition is not None:
                    audit_artifact_ids = (
                        audit_composition.create_run_header(
                            run_id=run_id,
                            thread_id=thread_id,
                            config=config,
                            started_at=run_started_at,
                        ),
                    )
                input_state = _initial_state(
                    run_id=run_id,
                    thread_id=thread_id,
                    config=config,
                    accountant=accountant,
                    baseline_work_artifact_ids=audit_artifact_ids,
                )
                graph_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                    }
                }
            else:
                if checkpoint.thread_id != thread_id:
                    raise CheckpointIdentityError()
                graph_config = checkpoint_config(checkpoint)
                saved = await saver_api.aget_tuple(graph_config)
                if saved is None or checkpoint_ref_from_tuple(saved) != checkpoint:
                    raise CheckpointIdentityError()
                raw_checkpoint = cast("Mapping[str, object]", saved.checkpoint)
                channel_values = raw_checkpoint.get("channel_values")
                if not isinstance(channel_values, Mapping):
                    raise CheckpointIdentityError()
                try:
                    state_values: dict[str, object] = {
                        name: channel_values[name]
                        for name in BaselineState.__annotations__
                    }
                except KeyError:
                    raise CheckpointIdentityError() from None
                restored = validate_baseline_state(
                    cast("Mapping[str, object]", state_values)
                )
                if (
                    restored["run_id"] != run_id
                    or restored["thread_id"] != thread_id
                    or restored["config_sha256"] != _config_sha256(config)
                ):
                    raise CheckpointIdentityError()
                accountant = BudgetAccountant.from_snapshot(
                    config.budget,
                    restored["budget_snapshot"],
                    run_scope=run_id,
                )
                input_state = None
                elapsed_before_resume = restored["elapsed_wall_seconds"]
                if audit_composition is not None:
                    header = audit_composition.load_run_header(
                        restored["baseline_work_artifact_ids"],
                        run_id=run_id,
                        thread_id=thread_id,
                        config=config,
                    )
                    stored_start = header.get("started_at")
                    if type(stored_start) is not str:
                        raise ArtifactIntegrityError(
                            "run audit header is missing or corrupt"
                        )
                    run_started_at = datetime.fromisoformat(stored_start)
                    if (
                        run_started_at.tzinfo is None
                        or run_started_at.utcoffset() is None
                    ):
                        raise ArtifactIntegrityError(
                            "run audit header is missing or corrupt"
                        )
                    try:
                        existing_event = await emit.get_event(
                            run_id=run_id,
                            seq=restored["next_event_seq"],
                        )
                    except (
                        asyncio.CancelledError,
                        KeyboardInterrupt,
                        MemoryError,
                        SystemExit,
                    ):
                        raise
                    except Exception:  # noqa: BLE001 - durable sink integrity boundary
                        raise ArtifactIntegrityError(
                            "durable event lookup failed"
                        ) from None
                    if existing_event is not None:
                        audit_composition.validate_durable_event_state(
                            restored,
                            existing_event,
                        )

            remaining_seconds = max(
                0.0,
                float(config.budget.max_wall_time_seconds) - elapsed_before_resume,
            )
            context = BaselineRuntimeContext(
                run_id=run_id,
                thread_id=thread_id,
                config=config,
                emit=emit,
                cancellation_token=cancellation_token,
                budget_accountant=accountant,
                deadline=start + remaining_seconds,
                run_started_monotonic=start,
                run_started_at=run_started_at,
                elapsed_base_seconds=elapsed_before_resume,
                monotonic=self._runtime_hooks.monotonic,
                utc_now=self._runtime_hooks.utc_now,
                new_id=self._runtime_hooks.new_id,
            )
            output = await graph.ainvoke(
                input_state,
                graph_config,
                context=context,
            )
            state = validate_baseline_state(cast("Mapping[str, object]", output))
            return _result_from_state(
                state,
                _usage_from_accountant(
                    accountant,
                    run_wall_seconds=state["elapsed_wall_seconds"],
                ),
            )
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception as error:  # noqa: BLE001 - stable workflow result boundary
            code = _exception_code(error)
            return _failed_result(
                run_id=run_id,
                thread_id=thread_id,
                usage=_usage_from_accountant(
                    accountant,
                    run_wall_seconds=_cumulative_wall(
                        base=(
                            elapsed_before_resume
                            + (
                                context.elapsed_tracker.recovered_offset_seconds
                                if context is not None
                                else 0.0
                            )
                        ),
                        start=start,
                        now=self._runtime_hooks.monotonic(),
                    ),
                ),
                error_code=code,
            )


__all__ = ["BaselineRuntimeHooks", "LangGraphResearchRunner"]
