from __future__ import annotations

import asyncio
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from decimal import Decimal
from types import MappingProxyType
from typing import Any, Literal, Protocol, TypeAlias, cast, runtime_checkable
from urllib.parse import urlsplit

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph import END, StateGraph  # pyright: ignore[reportMissingTypeStubs]
from langgraph.graph.state import (  # pyright: ignore[reportMissingTypeStubs]
    CompiledStateGraph,
)
from langgraph.runtime import get_runtime  # pyright: ignore[reportMissingTypeStubs]
from pydantic import BaseModel, ConfigDict, TypeAdapter, model_validator

from deepresearch.domain import (
    CoverageLedgerEntry,
    EvidenceSpan,
    FreshnessRequirement,
    RerankScore,
    ResearchPlan,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
    RunConfig,
    RunEvent,
    SourceDocument,
    SourceType,
    StopReason,
)
from deepresearch.planning import FixedPlanner, PlanGenerationError
from deepresearch.providers import (
    Fetcher,
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResult,
    ParsedDocument,
    Parser,
    ProviderError,
    ProviderUsageResult,
    RawDocument,
    Reranker,
    SearchHit,
    SearchProvider,
    StructuredModelResult,
    TextEmbedder,
    UsageReportingFetcher,
    UsageReportingSearchProvider,
    validate_embeddings,
)
from deepresearch.reporting import ContentBoundary, MarkdownReportWriter
from deepresearch.runtime import (
    BudgetAccountant,
    BudgetExceeded,
    BudgetReservation,
    BudgetSnapshot,
    CancellationToken,
    OperationCancelled,
    ResourceEstimate,
)
from deepresearch.runtime.manifest import (
    CostCalculator,
    EvidenceHashRecord,
    NodeExecutionRecord,
    ParsedArtifactRecord,
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
    EvidenceIntegrityError,
    FetchCacheKey,
    FileCache,
    LocalArtifactStore,
    LocalEvidenceStore,
    ModelCacheKey,
    ParseCacheKey,
    SearchCacheKey,
    cache_key_sha256,
)

from .state import (
    BaselineBlockedNeed,
    BaselineState,
    StateValidationError,
    validate_baseline_state,
)


class WorkflowInvariantError(RuntimeError):
    def __init__(self, *, code: str) -> None:
        super().__init__("workflow invariant violated")
        self.code = code


StateUpdate: TypeAlias = Mapping[str, object]  # noqa: UP040 - approved public contract
BaselineNode: TypeAlias = Callable[  # noqa: UP040 - approved public contract
    [BaselineState], Awaitable[StateUpdate]
]
BaselineRoute: TypeAlias = Literal[  # noqa: UP040 - stable route labels
    "Search", "DraftReport", "PersistResults"
]


class UsageCostResolver(Protocol):
    def resolve_cost(
        self,
        *,
        operation: str,
        provider_id: str,
        model_id: str | None,
        outcome: str,
        usage: ResourceUsage,
    ) -> Decimal | None: ...


@runtime_checkable
class InvocationUsageObserver(Protocol):
    def consume_invocation_usage(self) -> ResourceUsage | None: ...


@runtime_checkable
class DurableRunEventSink(Protocol):
    async def __call__(self, event: RunEvent) -> None: ...

    async def get_event(self, *, run_id: str, seq: int) -> RunEvent | None: ...


class UsageIntegrityError(RuntimeError):
    code = "USAGE_INTEGRITY"

    def __init__(self) -> None:
        super().__init__("provider invocation usage is inconsistent")


@dataclass(frozen=True)
class _PendingProviderCall:
    call_fields: Mapping[str, object]
    usage: ResourceUsage
    started_at: datetime
    finished_at: datetime
    budget_replay: _BudgetReplayFact


@dataclass
class _AuditBuffer:
    graph_node: str | None = None
    node_attempt: int = 0
    provider_calls: list[ProviderCallRecord] = field(
        default_factory=lambda: cast("list[ProviderCallRecord]", [])
    )
    provider_receipt_ids: list[str] = field(
        default_factory=lambda: cast("list[str]", [])
    )
    child_receipt_ids: list[str] = field(
        default_factory=lambda: cast("list[str]", [])
    )
    result_artifact_ids: list[str] = field(
        default_factory=lambda: cast("list[str]", [])
    )
    identity_counts: dict[str, int] = field(
        default_factory=lambda: cast("dict[str, int]", {})
    )
    operation_counts: dict[str, int] = field(
        default_factory=lambda: cast("dict[str, int]", {})
    )
    pending_provider_call: _PendingProviderCall | None = None

    def reset(self) -> None:
        self.graph_node = None
        self.node_attempt = 0
        self.provider_calls.clear()
        self.provider_receipt_ids.clear()
        self.child_receipt_ids.clear()
        self.result_artifact_ids.clear()
        self.identity_counts.clear()
        self.operation_counts.clear()
        self.pending_provider_call = None

    def begin(self, *, graph_node: str, node_attempt: int) -> None:
        self.reset()
        self.graph_node = graph_node
        self.node_attempt = node_attempt


def _audit_buffer(context: BaselineRuntimeContext) -> _AuditBuffer:
    return context.audit


@dataclass
class _ElapsedTracker:
    recovered_offset_seconds: float = 0.0

    def recover(self, *, elapsed_wall_seconds: float, elapsed_base_seconds: float) -> None:
        self.recovered_offset_seconds = max(
            self.recovered_offset_seconds,
            elapsed_wall_seconds - elapsed_base_seconds,
        )


@dataclass(frozen=True)
class BaselineRuntimeContext:
    run_id: str
    thread_id: str
    config: RunConfig
    emit: Callable[[RunEvent], Awaitable[None]]
    cancellation_token: CancellationToken
    budget_accountant: BudgetAccountant
    deadline: float
    run_started_monotonic: float
    run_started_at: datetime
    elapsed_base_seconds: float
    monotonic: Callable[[], float]
    utc_now: Callable[[], datetime]
    new_id: Callable[[str], str]
    audit: _AuditBuffer = field(default_factory=_AuditBuffer)
    elapsed_tracker: _ElapsedTracker = field(default_factory=_ElapsedTracker)


def _effective_deadline(context: BaselineRuntimeContext) -> float:
    effective = context.deadline - context.elapsed_tracker.recovered_offset_seconds
    if not math.isfinite(effective):
        raise ProviderError(
            code="TIMEOUT",
            provider="workflow",
            operation="deadline",
            public_message="workflow deadline expired",
            retryable=False,
        )
    return effective


def _ensure_operation_active(
    context: BaselineRuntimeContext,
    *,
    operation: str,
) -> float:
    context.cancellation_token.raise_if_cancelled()
    effective = _effective_deadline(context)
    now = context.monotonic()
    if not math.isfinite(now) or now >= effective:
        raise ProviderError(
            code="TIMEOUT",
            provider="workflow",
            operation=operation,
            public_message="workflow deadline expired",
            retryable=False,
        )
    return now


_AUDIT_SCHEMA = "baseline-audit-receipt-v1"
_BASELINE_STATE_ADAPTER = TypeAdapter(BaselineState)
_JSON_OBJECT_ADAPTER = TypeAdapter(dict[str, object])
_SEARCH_HITS_ADAPTER = TypeAdapter(list[SearchHit])
_MODEL_RESULT_ADAPTER = TypeAdapter(ModelResult[str])
_BUDGET_USAGE_MAP_TYPE = type(
    BudgetAccountant(RunBudget.preset("low")).snapshot().used_by_node
)
_EVENT_PAYLOAD_PROBE = RunEvent(
    seq=0,
    run_id="type-probe",
    timestamp=datetime(1970, 1, 1, tzinfo=UTC),
    node="type-probe",
    kind="type-probe",
    status="running",
    public_payload={"array": []},
    usage_delta=ResourceUsage.zero(),
    artifact_ids=(),
)
_EVENT_PAYLOAD_MAP_TYPE = type(_EVENT_PAYLOAD_PROBE.public_payload)
_EVENT_PAYLOAD_ARRAY_TYPE = type(_EVENT_PAYLOAD_PROBE.public_payload["array"])
_CACHE_METADATA_MAP_TYPE = type(
    CacheEntry(
        key_sha256="0" * 64,
        value_artifact_id="sha256:" + "0" * 64,
        producer_version="type-probe",
        usage=ResourceUsage.zero(),
        created_at=datetime(1970, 1, 1, tzinfo=UTC),
        metadata={},
    ).metadata
)


class _AuditReceiptEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["baseline-audit-receipt-v1"]
    kind: Literal[
        "run-header",
        "provider-call",
        "node-execution",
        "parsed-artifact",
        "evidence-hash",
        "terminal",
    ]
    run_id: str
    thread_id: str
    receipt_key: str
    payload: dict[str, object]

    @model_validator(mode="after")
    def validate_exact_payload(self) -> _AuditReceiptEnvelope:
        if not self.run_id or not self.thread_id or not self.receipt_key:
            raise ValueError("audit receipt identity must not be empty")
        expected = {
            "run-header": {
                "code_commit",
                "config_sha256",
                "dependency_lock_sha256",
                "execution_mode",
                "graph_version",
                "pricing_snapshots",
                "pricing_status",
                "provider_ids",
                "provider_profile_configuration_sha256",
                "provider_profile_id",
                "replay_parent",
                "request_sha256",
                "seed_supported",
                "started_at",
                "workflow_id",
            },
            "provider-call": {"record", "budget_replay", "result_artifact_ids"},
            "node-execution": {
                "event",
                "input_state_sha256",
                "provider_receipt_ids",
                "child_receipt_ids",
                "record",
                "state",
            },
            "parsed-artifact": {"record"},
            "evidence-hash": {"record"},
            "terminal": {
                "elapsed_wall_seconds",
                "error_code",
                "finished_at",
                "terminal_event_seq",
            },
        }[self.kind]
        if set(self.payload) != expected:
            raise ValueError("audit receipt payload shape is invalid")
        return self


class _BudgetReplayFact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    operation_id: str
    budget_node: Literal["Planner", "Ranker", "Writer", "Judge", "Tool"]
    estimate: ResourceEstimate
    actual: ResourceUsage | None
    decision: Literal["charge", "observe", "release"]

    @model_validator(mode="after")
    def validate_decision(self) -> _BudgetReplayFact:
        if not self.operation_id:
            raise ValueError("budget replay operation ID must not be empty")
        if self.decision == "release" and self.actual is not None:
            raise ValueError("released budget operations must not have actual usage")
        if self.decision != "release" and self.actual is None:
            raise ValueError("settled budget operations require actual usage")
        return self


def _state_payload_unchecked(state: BaselineState) -> dict[str, object]:
    value = _BASELINE_STATE_ADAPTER.dump_python(state, mode="json")
    if type(value) is not dict:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    return cast("dict[str, object]", value)


_USAGE_INTEGER_FIELDS = (
    "input_tokens",
    "output_tokens",
    "reasoning_tokens",
    "cached_tokens",
    "total_tokens",
    "search_calls",
    "pages",
    "retries",
)


def _require_exact_usage(value: object, *, label: str) -> ResourceUsage:
    if type(value) is not ResourceUsage:
        raise ArtifactIntegrityError(f"{label} is corrupt")
    usage = value
    raw = dict(usage.__dict__)
    if (
        set(raw) != set(ResourceUsage.model_fields)
        or any(type(raw[field]) is not int for field in _USAGE_INTEGER_FIELDS)
        or any(raw[field] < 0 for field in _USAGE_INTEGER_FIELDS)
        or type(raw["wall_seconds"]) is not float
        or (raw["cost_usd"] is not None and type(raw["cost_usd"]) is not Decimal)
    ):
        raise ArtifactIntegrityError(f"{label} is corrupt")
    wall_seconds = raw["wall_seconds"]
    cost_usd = raw["cost_usd"]
    if (
        not math.isfinite(wall_seconds)
        or wall_seconds < 0.0
        or (
            cost_usd is not None
            and (not cost_usd.is_finite() or cost_usd < Decimal(0))
        )
    ):
        raise ArtifactIntegrityError(f"{label} is corrupt")
    try:
        restored = ResourceUsage.model_validate(raw, strict=True)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError(f"{label} is corrupt") from None
    if restored != usage:
        raise ArtifactIntegrityError(f"{label} is corrupt")
    return usage


def _require_exact_budget_snapshot(value: object) -> BudgetSnapshot:
    if type(value) is not BudgetSnapshot:
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    snapshot = value
    raw = dict(snapshot.__dict__)
    integer_fields = (
        "used_search_calls",
        "used_pages",
        "used_tokens",
        "used_retries",
        "reserved_search_calls",
        "reserved_pages",
        "reserved_tokens",
        "reserved_retries",
    )
    if (
        set(raw) != set(BudgetSnapshot.model_fields)
        or any(type(raw[field]) is not int for field in integer_fields)
        or type(raw["used_wall_seconds"]) is not float
        or type(raw["reserved_wall_seconds"]) is not float
        or (
            raw["used_cost_usd"] is not None
            and type(raw["used_cost_usd"]) is not Decimal
        )
        or (
            raw["reserved_cost_usd"] is not None
            and type(raw["reserved_cost_usd"]) is not Decimal
        )
        or type(raw["exhausted"]) is not frozenset
        or any(type(item) is not str for item in cast("frozenset[object]", raw["exhausted"]))
        or type(raw["used_by_node"]) is not _BUDGET_USAGE_MAP_TYPE
    ):
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    usage_by_node = cast("dict[object, object]", raw["used_by_node"])
    canonical_nodes = {"Planner", "Ranker", "Writer", "Judge", "Tool"}
    if (
        any(type(key) is not str for key in usage_by_node)
        or set(usage_by_node) != canonical_nodes
    ):
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    for usage in usage_by_node.values():
        _require_exact_usage(usage, label="baseline state budget usage")
    _require_exact_usage(
        raw["last_observed_usage"],
        label="baseline state observed usage",
    )
    if (
        any(
            raw[field] != 0
            for field in (
                "reserved_search_calls",
                "reserved_pages",
                "reserved_tokens",
                "reserved_retries",
                "reserved_wall_seconds",
            )
        )
        or raw["reserved_cost_usd"] not in (None, Decimal(0))
    ):
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    rows = tuple(
        cast("ResourceUsage", usage_by_node[node])
        for node in ("Planner", "Ranker", "Writer", "Judge", "Tool")
    )
    expected_totals: tuple[tuple[str, int | float], ...] = (
        ("used_search_calls", sum(item.search_calls for item in rows)),
        ("used_pages", sum(item.pages for item in rows)),
        ("used_tokens", sum(item.total_tokens for item in rows)),
        ("used_retries", sum(item.retries for item in rows)),
        ("used_wall_seconds", sum(item.wall_seconds for item in rows)),
    )
    if any(raw[field] != expected for field, expected in expected_totals):
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    used_cost = raw["used_cost_usd"]
    reserved_cost = raw["reserved_cost_usd"]
    if used_cost is None:
        if reserved_cost is not None:
            raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    else:
        row_costs = tuple(item.cost_usd for item in rows)
        if (
            not used_cost.is_finite()
            or used_cost < Decimal(0)
            or reserved_cost != Decimal(0)
            or any(item is None for item in row_costs)
            or used_cost
            != sum((cast("Decimal", item) for item in row_costs), Decimal(0))
        ):
            raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    try:
        restored = BudgetSnapshot.model_validate(raw, strict=True)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt") from None
    if type(raw["used_by_node"]) is not type(restored.used_by_node) or restored != snapshot:
        raise ArtifactIntegrityError("baseline state budget snapshot is corrupt")
    return snapshot


def _same_runtime_shape(left: object, right: object) -> bool:
    if type(left) is not type(right):
        return False
    if isinstance(left, BaseModel) and isinstance(right, BaseModel):
        return _same_runtime_shape(left.__dict__, right.__dict__)
    if isinstance(left, dict) and isinstance(right, dict):
        left_mapping = cast("dict[object, object]", left)
        right_mapping = cast("dict[object, object]", right)
        left_keys: list[object] = list(left_mapping)
        right_keys: list[object] = list(right_mapping)
        if len(left_keys) != len(right_keys):
            return False
        for left_key in left_keys:
            matches = [
                right_key
                for right_key in right_keys
                if type(left_key) is type(right_key) and left_key == right_key
            ]
            if len(matches) != 1 or not _same_runtime_shape(
                left_mapping[left_key], right_mapping[matches[0]]
            ):
                return False
        return True
    if isinstance(left, tuple) and isinstance(right, tuple):
        left_tuple = cast("tuple[object, ...]", left)
        right_tuple = cast("tuple[object, ...]", right)
        return len(left_tuple) == len(right_tuple) and all(
            _same_runtime_shape(a, b)
            for a, b in zip(left_tuple, right_tuple, strict=True)
        )
    if isinstance(left, frozenset) and isinstance(right, frozenset):
        return cast("frozenset[object]", left) == cast("frozenset[object]", right)
    if isinstance(left, float):
        right_float = cast("float", right)
        return left == right_float and math.copysign(1.0, left) == math.copysign(
            1.0, right_float
        )
    return left == right


def _require_exact_event_json_value(value: object) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactIntegrityError("durable run event is corrupt")
        return
    if type(value) is _EVENT_PAYLOAD_ARRAY_TYPE:
        for item in cast("tuple[object, ...]", value):
            _require_exact_event_json_value(item)
        return
    if type(value) is _EVENT_PAYLOAD_MAP_TYPE:
        mapping = cast("dict[object, object]", value)
        for key, item in mapping.items():
            if type(key) is not str:
                raise ArtifactIntegrityError("durable run event is corrupt")
            _require_exact_event_json_value(item)
        return
    raise ArtifactIntegrityError("durable run event is corrupt")


def _require_exact_state_json_value(value: object) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactIntegrityError("baseline state audit encoding failed")
        return
    if type(value) is _EVENT_PAYLOAD_ARRAY_TYPE:
        for item in cast("tuple[object, ...]", value):
            _require_exact_state_json_value(item)
        return
    if type(value) is _EVENT_PAYLOAD_MAP_TYPE:
        mapping = cast("dict[object, object]", value)
        for key, item in mapping.items():
            if type(key) is not str:
                raise ArtifactIntegrityError("baseline state audit encoding failed")
            _require_exact_state_json_value(item)
        return
    raise ArtifactIntegrityError("baseline state audit encoding failed")


def _require_exact_freshness(value: object) -> FreshnessRequirement:
    if type(value) is not FreshnessRequirement:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    freshness = value
    raw = dict(freshness.__dict__)
    if (
        set(raw) != set(FreshnessRequirement.model_fields)
        or type(raw["kind"]) is not str
        or (
            raw["published_after"] is not None
            and type(raw["published_after"]) is not date
        )
        or (
            raw["retrieved_within_days"] is not None
            and type(raw["retrieved_within_days"]) is not int
        )
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    try:
        restored = FreshnessRequirement.model_validate(raw, strict=True)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline state audit encoding failed") from None
    if restored != freshness:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    return freshness


def _require_exact_research_request(value: object) -> ResearchRequest:
    if type(value) is not ResearchRequest:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    request = value
    raw = dict(request.__dict__)
    string_fields = (
        "question",
        "report_language",
        "execution_mode",
        "access_profile",
        "provider_profile_id",
        "run_purpose",
        "budget_preset",
    )
    if (
        set(raw) != set(ResearchRequest.model_fields)
        or any(type(raw[field]) is not str for field in string_fields)
        or type(raw["source_languages"]) is not tuple
        or any(
            type(item) is not str
            for item in cast("tuple[object, ...]", raw["source_languages"])
        )
        or type(raw["output_requirements"]) is not _EVENT_PAYLOAD_MAP_TYPE
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    _require_exact_state_json_value(raw["output_requirements"])
    _require_exact_freshness(raw["freshness_requirement"])
    try:
        restored = ResearchRequest.model_validate(raw, strict=True)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline state audit encoding failed") from None
    if restored != request or not _same_runtime_shape(restored, request):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    return request


def _require_exact_coverage_entry(value: object) -> CoverageLedgerEntry:
    if type(value) is not CoverageLedgerEntry:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    entry = value
    raw = dict(entry.__dict__)
    string_fields = ("subquestion_id", "last_decision_code")
    integer_fields = ("independent_source_count", "attempt_count")
    float_fields = ("coverage_score", "uncertainty_score", "last_marginal_gain")
    tuple_fields = ("unresolved_conflict_ids", "evidence_ids")
    if (
        set(raw) != set(CoverageLedgerEntry.model_fields)
        or any(type(raw[field]) is not str for field in string_fields)
        or any(type(raw[field]) is not int for field in integer_fields)
        or any(type(raw[field]) is not float for field in float_fields)
        or any(not math.isfinite(raw[field]) for field in float_fields)
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    for field_name in tuple_fields:
        members = raw[field_name]
        if type(members) is not tuple or any(
            type(item) is not str for item in cast("tuple[object, ...]", members)
        ):
            raise ArtifactIntegrityError("baseline state audit encoding failed")
    try:
        restored = CoverageLedgerEntry.model_validate(raw, strict=True)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline state audit encoding failed") from None
    if restored != entry or not _same_runtime_shape(restored, entry):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    return entry


def _validate_exact_live_state_shape(value: object) -> BaselineState:
    if type(value) is not dict:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    raw_state = cast("dict[object, object]", value)
    if any(type(key) is not str for key in raw_state) or set(raw_state) != set(
        BaselineState.__annotations__
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    state = cast("BaselineState", value)
    required_strings = ("run_id", "thread_id", "config_sha256")
    optional_strings = (
        "plan_id",
        "plan_artifact_id",
        "active_subquestion_id",
        "draft_artifact_id",
        "report_artifact_id",
        "evidence_graph_artifact_id",
        "manifest_artifact_id",
        "failed_node",
        "error_code",
    )
    string_tuples = (
        "pending_subquestion_ids",
        "query_ids",
        "source_ids",
        "evidence_ids",
        "selected_evidence_ids",
        "high_priority_unresolved_conflict_ids",
        "baseline_work_artifact_ids",
    )
    if any(type(state[field]) is not str for field in required_strings) or any(
        state[field] is not None and type(state[field]) is not str
        for field in optional_strings
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    for field_name in string_tuples:
        members = state[field_name]
        if type(members) is not tuple or any(
            type(item) is not str for item in cast("tuple[object, ...]", members)
        ):
            raise ArtifactIntegrityError("baseline state audit encoding failed")
    _require_exact_research_request(state["request"])
    ledger = state["coverage_ledger"]
    if type(ledger) is not tuple:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    for item in cast("tuple[object, ...]", ledger):
        _require_exact_coverage_entry(item)
    blocked = state["blocked_needs"]
    blocked_fields = set(BaselineBlockedNeed.__annotations__)
    if type(blocked) is not tuple:
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    for item in cast("tuple[object, ...]", blocked):
        if type(item) is not dict:
            raise ArtifactIntegrityError("baseline state audit encoding failed")
        blocked_mapping = cast("dict[object, object]", item)
        if any(type(key) is not str for key in blocked_mapping) or set(
            blocked_mapping
        ) != blocked_fields:
            raise ArtifactIntegrityError("baseline state audit encoding failed")
        need = cast("dict[str, object]", item)
        if (
            type(need["need_id"]) is not str
            or type(need["required_source_unavailable"]) is not bool
            or type(need["alternative_strategies_exhausted"]) is not bool
            or type(need["retry_count"]) is not int
            or type(need["max_retries"]) is not int
        ):
            raise ArtifactIntegrityError("baseline state audit encoding failed")
    gains = state["recent_marginal_gains"]
    if type(gains) is not tuple or any(
        type(item) is not float for item in cast("tuple[object, ...]", gains)
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    _require_exact_budget_snapshot(state["budget_snapshot"])
    if (
        (state["stop_reason"] is not None and type(state["stop_reason"]) is not str)
        or type(state["is_partial"]) is not bool
        or type(state["next_event_seq"]) is not int
        or type(state["elapsed_wall_seconds"]) is not float
    ):
        raise ArtifactIntegrityError("baseline state audit encoding failed")
    return state


def _strict_run_event(value: object) -> RunEvent:
    if type(value) is not RunEvent:
        raise ArtifactIntegrityError("durable run event is corrupt")
    event = value
    raw = dict(event.__dict__)
    if (
        set(raw) != set(RunEvent.model_fields)
        or type(raw["seq"]) is not int
        or any(
            type(raw[field]) is not str
            for field in ("run_id", "node", "kind", "status")
        )
        or type(raw["timestamp"]) is not datetime
        or type(raw["artifact_ids"]) is not tuple
        or any(
            type(item) is not str
            for item in cast("tuple[object, ...]", raw["artifact_ids"])
        )
        or (raw["error_code"] is not None and type(raw["error_code"]) is not str)
    ):
        raise ArtifactIntegrityError("durable run event is corrupt")
    _require_exact_usage(raw["usage_delta"], label="durable run event usage")
    if type(raw["public_payload"]) is not _EVENT_PAYLOAD_MAP_TYPE:
        raise ArtifactIntegrityError("durable run event is corrupt")
    _require_exact_event_json_value(raw["public_payload"])
    try:
        encoded = _canonical_bytes(event.model_dump(mode="json"))
        restored = RunEvent.model_validate_json(encoded, strict=True)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("durable run event is corrupt") from None
    if not _same_runtime_shape(event, restored):
        raise ArtifactIntegrityError("durable run event is corrupt")
    return event


def _run_events_match_exact(value: object, expected: RunEvent) -> bool:
    try:
        exact = _strict_run_event(value)
        exact_expected = _strict_run_event(expected)
    except ArtifactIntegrityError:
        return False
    return _canonical_bytes(exact.model_dump(mode="json")) == _canonical_bytes(
        exact_expected.model_dump(mode="json")
    )


def _run_event_from_receipt_payload(value: object) -> RunEvent:
    try:
        payload_bytes = _canonical_bytes(value)
        event = RunEvent.model_validate_json(payload_bytes, strict=True)
        exact = _strict_run_event(event)
    except (TypeError, ValueError, ArtifactIntegrityError):
        raise ArtifactIntegrityError("durable event receipt is corrupt") from None
    if payload_bytes != _canonical_bytes(exact.model_dump(mode="json")):
        raise ArtifactIntegrityError("durable event receipt is corrupt")
    return exact


def _node_execution_from_receipt_payload(value: object) -> NodeExecutionRecord:
    """Decode a node receipt only when its JSON is strict and canonical."""
    try:
        payload_bytes = _canonical_bytes(value)
        record = NodeExecutionRecord.model_validate_json(
            payload_bytes,
            strict=True,
        )
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("node execution receipt is corrupt") from None
    if payload_bytes != _canonical_bytes(record.model_dump(mode="json")):
        raise ArtifactIntegrityError("node execution receipt is not canonical")
    if (
        record.usage.wall_seconds == 0.0
        and math.copysign(1.0, record.usage.wall_seconds) < 0
    ):
        raise ArtifactIntegrityError("node execution receipt is not canonical")
    return record


def _validate_exact_json_value(value: object) -> None:
    if value is None or type(value) in (str, int, bool):
        return
    if type(value) is float:
        if not math.isfinite(value):
            raise ArtifactIntegrityError("baseline state audit receipt is corrupt")
        return
    if type(value) is list:
        for item in cast("list[object]", value):
            _validate_exact_json_value(item)
        return
    if type(value) is dict:
        mapping = cast("dict[object, object]", value)
        if any(type(key) is not str for key in mapping):
            raise ArtifactIntegrityError("baseline state audit receipt is corrupt")
        for item in mapping.values():
            _validate_exact_json_value(item)
        return
    raise ArtifactIntegrityError("baseline state audit receipt is corrupt")


def _validate_exact_json_state_shape(value: object) -> dict[str, object]:
    if type(value) is not dict:
        raise ArtifactIntegrityError("baseline state audit receipt is corrupt")
    state = cast("dict[str, object]", value)
    if set(state) != set(BaselineState.__annotations__):
        raise ArtifactIntegrityError("baseline state audit receipt is corrupt")
    blocked = state.get("blocked_needs")
    if type(blocked) is not list:
        raise ArtifactIntegrityError("baseline state audit receipt is corrupt")
    blocked_fields = set(BaselineBlockedNeed.__annotations__)
    for item in cast("list[object]", blocked):
        if type(item) is not dict or set(cast("dict[object, object]", item)) != blocked_fields:
            raise ArtifactIntegrityError("baseline state audit receipt is corrupt")
    _validate_exact_json_value(state)
    return state


def _state_payload(state: BaselineState) -> dict[str, object]:
    try:
        _validate_exact_live_state_shape(state)
        validated = validate_baseline_state(cast("Mapping[str, object]", state))
        payload = _validate_exact_json_state_shape(_state_payload_unchecked(validated))
        state_bytes = _canonical_bytes(payload)
        restored = _BASELINE_STATE_ADAPTER.validate_json(state_bytes, strict=True)
        round_trip = validate_baseline_state(cast("Mapping[str, object]", restored))
        if round_trip != validated or not _same_runtime_shape(validated, round_trip):
            raise ArtifactIntegrityError("baseline state audit encoding failed")
        if _canonical_bytes(_state_payload_unchecked(round_trip)) != state_bytes:
            raise ArtifactIntegrityError("baseline state audit encoding failed")
        return payload
    except (StateValidationError, TypeError, ValueError):
        raise ArtifactIntegrityError("baseline state audit encoding failed") from None


def _state_from_payload(value: object) -> BaselineState:
    try:
        payload = _validate_exact_json_state_shape(value)
        state_bytes = _canonical_bytes(payload)
        restored = _BASELINE_STATE_ADAPTER.validate_json(state_bytes, strict=True)
        _validate_exact_live_state_shape(restored)
        validated = validate_baseline_state(cast("Mapping[str, object]", restored))
        if _canonical_bytes(_state_payload_unchecked(validated)) != state_bytes:
            raise ArtifactIntegrityError("baseline state audit receipt is not canonical")
        return validated
    except (StateValidationError, TypeError, ValueError):
        raise ArtifactIntegrityError("baseline state audit receipt is corrupt") from None


def _unique_json_object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError("duplicate JSON object member")
        value[key] = item
    return value


def _reject_json_constant(_value: str) -> object:
    raise ValueError("non-finite JSON constant")


def _put_audit_receipt(
    store: LocalArtifactStore,
    *,
    kind: str,
    run_id: str,
    thread_id: str,
    receipt_key: str,
    payload: Mapping[str, object],
) -> str:
    try:
        envelope = _AuditReceiptEnvelope.model_validate(
            {
                "schema_version": _AUDIT_SCHEMA,
                "kind": kind,
                "run_id": run_id,
                "thread_id": thread_id,
                "receipt_key": receipt_key,
                "payload": dict(payload),
            }
        )
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline audit receipt is invalid") from None
    ref = store.put_bytes(
        _canonical_bytes(json.loads(envelope.model_dump_json())),
        media_type="application/vnd.deepresearch.baseline-audit+json",
    )
    return ref.artifact_id


def _load_audit_receipt(
    store: LocalArtifactStore,
    artifact_id: str,
) -> _AuditReceiptEnvelope | None:
    try:
        raw = store.get_bytes(artifact_id)
    except FileNotFoundError:
        return None
    try:
        value: object = json.loads(
            raw,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
    except (TypeError, ValueError):
        try:
            claimed_pairs: object = json.loads(
                raw,
                object_pairs_hook=lambda pairs: pairs,
                parse_constant=lambda value: value,
            )
        except (TypeError, ValueError):
            return None
        pairs = cast("list[object]", claimed_pairs) if type(claimed_pairs) is list else []
        if any(
            type(pair) is tuple
            and len(cast("tuple[object, ...]", pair)) == 2
            and cast("tuple[object, ...]", pair)[0] == "schema_version"
            and type(cast("tuple[object, ...]", pair)[1]) is str
            and cast("str", cast("tuple[object, ...]", pair)[1]).startswith(
                "baseline-audit-receipt-"
            )
            for pair in pairs
        ):
            raise ArtifactIntegrityError("baseline audit receipt is corrupt") from None
        return None
    if type(value) is not dict:
        return None
    receipt = cast("dict[str, object]", value)
    claimed_schema = receipt.get("schema_version")
    if claimed_schema is None:
        return None
    if claimed_schema != _AUDIT_SCHEMA:
        if type(claimed_schema) is str and claimed_schema.startswith(
            "baseline-audit-receipt-"
        ):
            raise ArtifactIntegrityError("baseline audit receipt is corrupt")
        return None
    try:
        if _canonical_bytes(receipt) != raw:
            raise ArtifactIntegrityError("baseline audit receipt is corrupt")
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline audit receipt is corrupt") from None
    try:
        return _AuditReceiptEnvelope.model_validate(receipt)
    except (TypeError, ValueError):
        raise ArtifactIntegrityError("baseline audit receipt is corrupt") from None


@dataclass(frozen=True)
class _AuditComposition:
    artifact_store: LocalArtifactStore
    code_commit: str
    dependency_lock_sha256: str
    graph_version: str
    provider_profile_configuration_sha256: str
    provider_ids: tuple[str, ...]
    seed_supported: bool
    pricing_status: Literal["estimated", "unknown"]
    pricing_snapshots: tuple[PricingSnapshot, ...]
    replay_parent: str | None

    def create_run_header(
        self,
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig,
        started_at: datetime,
    ) -> str:
        return _put_audit_receipt(
            self.artifact_store,
            kind="run-header",
            run_id=run_id,
            thread_id=thread_id,
            receipt_key="header",
            payload={
                "code_commit": self.code_commit,
                "config_sha256": _hash_json(config.model_dump(mode="json")),
                "dependency_lock_sha256": self.dependency_lock_sha256,
                "execution_mode": config.request.execution_mode,
                "graph_version": self.graph_version,
                "pricing_snapshots": [
                    item.model_dump(mode="json") for item in self.pricing_snapshots
                ],
                "pricing_status": self.pricing_status,
                "provider_ids": list(self.provider_ids),
                "provider_profile_configuration_sha256": self.provider_profile_configuration_sha256,
                "provider_profile_id": config.request.provider_profile_id,
                "replay_parent": self.replay_parent,
                "request_sha256": _hash_json(
                    config.request.model_dump(mode="json")
                ),
                "seed_supported": self.seed_supported,
                "started_at": started_at.isoformat(),
                "workflow_id": config.workflow_id,
            },
        )

    def load_run_header(
        self,
        artifact_ids: Sequence[str],
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig | None = None,
    ) -> dict[str, object]:
        headers = [
            receipt.payload
            for artifact_id in artifact_ids
            if (receipt := _load_audit_receipt(self.artifact_store, artifact_id))
            is not None
            and receipt.kind == "run-header"
            and receipt.run_id == run_id
            and receipt.thread_id == thread_id
        ]
        if len(headers) != 1:
            raise ArtifactIntegrityError("run audit header is missing or corrupt")
        header = headers[0]
        expected: dict[str, object] = {
            "code_commit": self.code_commit,
            "dependency_lock_sha256": self.dependency_lock_sha256,
            "graph_version": self.graph_version,
            "pricing_snapshots": [
                item.model_dump(mode="json") for item in self.pricing_snapshots
            ],
            "pricing_status": self.pricing_status,
            "provider_ids": list(self.provider_ids),
            "provider_profile_configuration_sha256": (
                self.provider_profile_configuration_sha256
            ),
            "replay_parent": self.replay_parent,
            "seed_supported": self.seed_supported,
        }
        if config is not None:
            expected.update(
                {
                    "config_sha256": _hash_json(config.model_dump(mode="json")),
                    "execution_mode": config.request.execution_mode,
                    "provider_profile_id": config.request.provider_profile_id,
                    "request_sha256": _hash_json(
                        config.request.model_dump(mode="json")
                    ),
                    "workflow_id": config.workflow_id,
                }
            )
        if any(header.get(key) != value for key, value in expected.items()):
            raise ArtifactIntegrityError("run audit composition changed across resume")
        return header

    def validate_durable_event_state(
        self,
        state: BaselineState,
        event: RunEvent,
    ) -> None:
        exact_event = _strict_run_event(event)
        if (
            type(event) is not RunEvent
            or exact_event != event
            or event.run_id != state["run_id"]
            or event.seq != state["next_event_seq"]
        ):
            raise ArtifactIntegrityError("durable run event conflicts with graph state")
        input_sha256 = _hash_json(_state_payload(state))
        matches: list[tuple[str, _AuditReceiptEnvelope]] = []
        for artifact_id in event.artifact_ids:
            receipt = _load_audit_receipt(self.artifact_store, artifact_id)
            if (
                receipt is not None
                and receipt.kind == "node-execution"
                and receipt.run_id == state["run_id"]
                and receipt.thread_id == state["thread_id"]
                and receipt.payload.get("input_state_sha256") == input_sha256
            ):
                matches.append((artifact_id, receipt))
        if len(matches) != 1:
            raise ArtifactIntegrityError(
                "durable event node receipt is missing or ambiguous"
            )
        receipt_id, receipt = matches[0]
        base_event = _run_event_from_receipt_payload(receipt.payload.get("event"))
        expected_event = base_event.model_copy(
            update={
                "artifact_ids": tuple(
                    dict.fromkeys((*base_event.artifact_ids, receipt_id))
                )
            }
        )
        if not _run_events_match_exact(event, expected_event):
            raise ArtifactIntegrityError("durable event bytes conflict with node receipt")
        recovered = _state_from_payload(receipt.payload.get("state"))
        if (
            recovered["run_id"] != state["run_id"]
            or recovered["thread_id"] != state["thread_id"]
            or recovered["next_event_seq"] != state["next_event_seq"]
        ):
            raise ArtifactIntegrityError("durable event state receipt is corrupt")


@dataclass(frozen=True)
class BaselineDependencies:
    checkpointer: BaseCheckpointSaver[str]
    validate_request: BaselineNode
    plan: BaselineNode
    decide_next: BaselineNode
    search: BaselineNode
    fetch: BaselineNode
    parse_and_normalize: BaselineNode
    store_evidence: BaselineNode
    rank_evidence: BaselineNode
    content_boundary: ContentBoundary
    draft_report: BaselineNode
    finalize_citations: BaselineNode
    persist_results: BaselineNode
    _audit_composition: _AuditComposition | None = field(
        default=None,
        init=False,
        repr=False,
        compare=False,
    )


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _put_verified_evidence_graph(
    artifact_store: LocalArtifactStore,
    *,
    evidence: Sequence[EvidenceSpan],
    sources: Sequence[SourceDocument],
) -> tuple[str, tuple[EvidenceSpan, ...], tuple[SourceDocument, ...]]:
    candidate_evidence = tuple(evidence)
    candidate_sources = tuple(sources)
    candidate_bytes = _canonical_bytes(
        {
            "evidence": [item.model_dump(mode="json") for item in candidate_evidence],
            "sources": [item.model_dump(mode="json") for item in candidate_sources],
        }
    )
    reference = artifact_store.put_bytes(
        candidate_bytes,
        media_type="application/vnd.deepresearch.evidence-graph+json",
    )
    try:
        stored_bytes = artifact_store.get_bytes(reference.artifact_id)
    except FileNotFoundError:
        raise ArtifactIntegrityError("evidence graph acknowledgement is missing") from None
    if stored_bytes != candidate_bytes:
        raise ArtifactIntegrityError("evidence graph bytes conflict")
    try:
        value: object = json.loads(
            stored_bytes,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if type(value) is not dict:
            raise ValueError("evidence graph shape is invalid")
        payload = cast("dict[str, object]", value)
        if set(payload) != {"evidence", "sources"}:
            raise ValueError("evidence graph shape is invalid")
        evidence_values = payload["evidence"]
        source_values = payload["sources"]
        if type(evidence_values) is not list or type(source_values) is not list:
            raise ValueError("evidence graph members must be arrays")
        evidence_items = cast("list[object]", evidence_values)
        source_items = cast("list[object]", source_values)
        decoded_evidence = tuple(
            EvidenceSpan.model_validate_json(_canonical_bytes(item), strict=True)
            for item in evidence_items
        )
        decoded_sources = tuple(
            SourceDocument.model_validate_json(_canonical_bytes(item), strict=True)
            for item in source_items
        )
    except (KeyError, TypeError, ValueError):
        raise ArtifactIntegrityError("evidence graph is corrupt") from None
    if decoded_evidence != candidate_evidence or decoded_sources != candidate_sources:
        raise ArtifactIntegrityError("evidence graph membership conflicts")
    return reference.artifact_id, decoded_evidence, decoded_sources


def _hash_json(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def stable_operation_id(
    *,
    run_id: str,
    workflow_id: str,
    node: str,
    logical_input: object,
    node_attempt: int = 1,
    ordinal: int = 0,
) -> str:
    return _hash_json(
        {
            "logical_input_hash": _hash_json(logical_input),
            "node": node,
            "node_attempt": node_attempt,
            "ordinal": ordinal,
            "run_id": run_id,
            "workflow_id": workflow_id,
        }
    )


def _parse_source_snapshot_id(
    *,
    fetch_snapshot_id: str,
    final_url: object,
    content_type: str,
) -> str:
    media_type = content_type.split(";", 1)[0].strip().casefold()
    digest = _hash_json(
        {
            "schema": "baseline-parse-source-v1",
            "fetch_snapshot_id": fetch_snapshot_id,
            "final_url": str(final_url),
            "media_type": media_type,
        }
    )
    return f"parse-source-v1-{digest}"


def _zero_usage(*, cost_known: bool) -> ResourceUsage:
    return ResourceUsage.zero(cost_known=cost_known)


def _reported_usage[Value](result: ProviderUsageResult[Value]) -> ResourceUsage:
    return result.usage


def _no_reported_usage(_result: object) -> None:
    return None


def _model_identity(model: ModelProvider) -> tuple[str, str]:
    model_id = getattr(model, "model_id", model.provider_id)
    return model.provider_id, model_id if isinstance(model_id, str) else model.provider_id


class _CachedModelProvider:
    def __init__(self, owner: BaselineNodeHandlers, inner: ModelProvider) -> None:
        self._owner = owner
        self._inner = inner
        self.provider_id = inner.provider_id
        self.model_id = _model_identity(inner)[1]

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        result = await self._owner._cached_model_call(  # pyright: ignore[reportPrivateUsage]
            provider=self._inner,
            request=request,
            output_schema=None,
            invoke=lambda: self._inner.complete(
                request,
                deadline=deadline,
                cancellation_token=cancellation_token,
            ),
        )
        return result

    async def structured[Output](
        self,
        request: ModelRequest,
        output_schema: type[Output],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> StructuredModelResult[Output]:
        result = await self._owner._cached_model_call(  # pyright: ignore[reportPrivateUsage]
            provider=self._inner,
            request=request,
            output_schema=output_schema,
            invoke=lambda: self._inner.structured(
                request,
                output_schema,
                deadline=deadline,
                cancellation_token=cancellation_token,
            ),
        )
        return result

    def stream(self, request: ModelRequest, **kwargs: Any) -> Any:
        return self._inner.stream(request, **kwargs)


class _CachedTextEmbedder:
    def __init__(self, owner: BaselineNodeHandlers, inner: TextEmbedder) -> None:
        self._owner = owner
        self._inner = inner
        self.provider_id = inner.provider_id
        self.model_id = inner.model_id
        self.model_revision = inner.model_revision
        self.snapshot_sha256 = inner.snapshot_sha256

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        return await self._owner._cached_embed_call(  # pyright: ignore[reportPrivateUsage]
            provider=self._inner,
            texts=tuple(texts),
            invoke=lambda: self._inner.embed(
                texts,
                deadline=deadline,
                cancellation_token=cancellation_token,
            ),
        )


class _PlanNode:
    def __init__(self, owner: BaselineNodeHandlers) -> None:
        self._owner = owner
        self.initial_plan_generator = owner.initial_plan_generator

    async def __call__(self, state: BaselineState) -> StateUpdate:
        return await self._owner.run_plan(state)


class BaselineNodeHandlers:
    def __init__(
        self,
        *,
        initial_plan_generator: FixedPlanner,
        ranker: Reranker,
        writer: MarkdownReportWriter,
        search_provider: SearchProvider,
        fetcher: Fetcher,
        parser: Parser,
        artifact_store: LocalArtifactStore,
        evidence_store: LocalEvidenceStore,
        cache: FileCache,
        usage_cost_resolver: UsageCostResolver,
        search_snapshot_id: str,
        fetch_snapshot_id: str,
        code_commit: str,
        dependency_lock_sha256: str,
        provider_profile_configuration_sha256: str,
        seed_supported: bool,
        pricing_status: Literal["estimated", "unknown"],
        pricing_snapshots: Sequence[PricingSnapshot],
        replay_parent: str | None,
        normalization_version: str = "normalize-v1",
        writer_prompt_version: str = "baseline-writer-v1",
        ranker_weights_version: str = "r1-cosine-v1",
        graph_version: str = "baseline-graph-v1",
    ) -> None:
        if not all(
            item.strip()
            for item in (
                search_snapshot_id,
                fetch_snapshot_id,
                normalization_version,
                writer_prompt_version,
                ranker_weights_version,
                graph_version,
                code_commit,
                dependency_lock_sha256,
                provider_profile_configuration_sha256,
            )
        ):
            raise ValueError("baseline version metadata must not be empty")
        if len(code_commit) not in (40, 64) or any(
            item not in "0123456789abcdef" for item in code_commit
        ):
            raise ValueError("code_commit must be a lowercase Git object hash")
        for digest in (
            dependency_lock_sha256,
            provider_profile_configuration_sha256,
        ):
            if len(digest) != 64 or any(
                item not in "0123456789abcdef" for item in digest
            ):
                raise ValueError("audit configuration hashes must be SHA-256")
        frozen_pricing = tuple(pricing_snapshots)
        if pricing_status == "unknown" and frozen_pricing:
            raise ValueError("unknown pricing must not supply pricing snapshots")
        if pricing_status == "estimated" and not frozen_pricing:
            raise ValueError("estimated pricing requires pricing snapshots")
        if writer.content_boundary is not initial_plan_generator.content_boundary:
            raise ValueError("P1 and Writer must share the identical content boundary")
        original_model = initial_plan_generator.model
        writer_model = writer.model
        ranker_embedder = getattr(ranker, "embedder", None)
        if (
            isinstance(original_model, _CachedModelProvider)
            or isinstance(writer_model, _CachedModelProvider)
            or isinstance(ranker_embedder, _CachedTextEmbedder)
        ):
            raise TypeError("baseline components are already bound to handlers")
        models = (
            original_model,
            *((writer_model,) if isinstance(writer_model, ModelProvider) else ()),
        )
        for model in models:
            model_value = cast("Any", model)
            try:
                model_id = model_value.model_id
                model_revision = model_value.model_revision
            except AttributeError:
                raise ValueError(
                    "model identity must be supplied by the provider profile"
                ) from None
            if type(model_id) is not str or not model_id:
                raise ValueError("model_id must be supplied by the provider profile")
            if type(model_revision) is not str or not model_revision:
                raise ValueError(
                    "model_revision must be supplied by the provider profile"
                )
        self.initial_plan_generator = initial_plan_generator
        self.ranker = ranker
        self.writer = writer
        self.search_provider = search_provider
        self.fetcher = fetcher
        self.parser = parser
        self.artifact_store = artifact_store
        self.evidence_store = evidence_store
        self.cache = cache
        self.usage_cost_resolver = usage_cost_resolver
        self.search_snapshot_id = search_snapshot_id
        self.fetch_snapshot_id = fetch_snapshot_id
        self.normalization_version = normalization_version
        self.writer_prompt_version = writer_prompt_version
        self.ranker_weights_version = ranker_weights_version
        self.graph_version = graph_version
        self.code_commit = code_commit
        self.dependency_lock_sha256 = dependency_lock_sha256
        self.provider_profile_configuration_sha256 = (
            provider_profile_configuration_sha256
        )
        self.seed_supported = seed_supported
        self.pricing_status = pricing_status
        self.pricing_snapshots = frozen_pricing
        self.replay_parent = replay_parent
        self.content_boundary = initial_plan_generator.content_boundary
        self.plan = _PlanNode(self)
        self._locks_guard = asyncio.Lock()
        self._locks: dict[str, asyncio.Lock] = {}
        self._planner_lock = asyncio.Lock()
        metered_model = _CachedModelProvider(self, original_model)
        initial_plan_generator.model = metered_model
        if writer_model is original_model:
            writer.model = metered_model
        elif isinstance(writer_model, ModelProvider):
            writer.model = _CachedModelProvider(self, writer_model)
        if isinstance(ranker_embedder, TextEmbedder):
            cast("Any", ranker).embedder = _CachedTextEmbedder(self, ranker_embedder)
        provider_ids = tuple(
            dict.fromkeys(
                (
                    original_model.provider_id,
                    *(
                        (writer_model.provider_id,)
                        if isinstance(writer_model, ModelProvider)
                        else ()
                    ),
                    search_provider.provider_id,
                    fetcher.provider_id,
                    parser.parser_id,
                    *(
                        (ranker_embedder.provider_id,)
                        if isinstance(ranker_embedder, TextEmbedder)
                        else ()
                    ),
                )
            )
        )
        self._audit_composition = _AuditComposition(
            artifact_store=artifact_store,
            code_commit=code_commit,
            dependency_lock_sha256=dependency_lock_sha256,
            graph_version=graph_version,
            provider_profile_configuration_sha256=provider_profile_configuration_sha256,
            provider_ids=provider_ids,
            seed_supported=seed_supported,
            pricing_status=pricing_status,
            pricing_snapshots=frozen_pricing,
            replay_parent=replay_parent,
        )

    @property
    def prompt_versions(self) -> Mapping[str, str]:
        version = self.initial_plan_generator.prompt_version
        return MappingProxyType(
            {
                "planner": version,
                "planner_queries": f"{version}-queries",
                "writer": self.writer_prompt_version,
            }
        )

    @property
    def version_metadata(self) -> Mapping[str, str]:
        return MappingProxyType(
            {
                "fetch_snapshot_id": self.fetch_snapshot_id,
                "graph_version": self.graph_version,
                "normalization_version": self.normalization_version,
                "parser_id": self.parser.parser_id,
                "parser_version": self.parser.parser_version,
                "ranker_weights_version": self.ranker_weights_version,
                "search_snapshot_id": self.search_snapshot_id,
            }
        )

    @property
    def model_ids(self) -> Mapping[str, str]:
        planner_provider, planner_model = _model_identity(self.initial_plan_generator.model)
        writer_model = self.writer.model
        writer_id = (
            _model_identity(writer_model)[1] if isinstance(writer_model, ModelProvider) else "none"
        )
        ranker_model = getattr(getattr(self.ranker, "embedder", None), "model_id", None)
        ranker_id = getattr(
            self.ranker,
            "reranker_id",
            getattr(self.ranker, "ranker_id", "R1"),
        )
        return MappingProxyType(
            {
                "planner": planner_model,
                "planner_provider": planner_provider,
                "ranker": ranker_model if isinstance(ranker_model, str) else ranker_id,
                "writer": writer_id,
            }
        )

    def as_dependencies(
        self,
        checkpointer: BaseCheckpointSaver[str],
    ) -> BaselineDependencies:
        dependencies = BaselineDependencies(
            checkpointer=checkpointer,
            validate_request=self.validate_request,
            plan=self.plan,
            decide_next=self.decide_next,
            search=self.search,
            fetch=self.fetch,
            parse_and_normalize=self.parse_and_normalize,
            store_evidence=self.store_evidence,
            rank_evidence=self.rank_evidence,
            content_boundary=self.content_boundary,
            draft_report=self.draft_report,
            finalize_citations=self.finalize_citations,
            persist_results=self.persist_results,
        )
        object.__setattr__(dependencies, "_audit_composition", self._audit_composition)
        return dependencies

    async def _lock_for(self, digest: str) -> asyncio.Lock:
        async with self._locks_guard:
            return self._locks.setdefault(digest, asyncio.Lock())

    async def _isolated_planner_call[Result](
        self,
        context: BaselineRuntimeContext,
        invoke: Callable[[], Awaitable[Result]],
    ) -> Result:
        async with self._planner_lock:
            _ensure_operation_active(context, operation="planner")
            planner = cast("Any", self.initial_plan_generator)
            saved = (
                planner.budget,
                planner._initial_model_tokens,
                planner._model_tokens_used,
                planner._model_tokens_reserved,
                planner._query_cache,
                planner._query_locks,
            )
            snapshot = context.budget_accountant.snapshot()
            planner.budget = context.config.budget.model_copy(
                update={"used_by_node": snapshot.used_by_node}
            )
            planner._initial_model_tokens = sum(
                item.total_tokens for item in snapshot.used_by_node.values()
            )
            planner._model_tokens_used = 0
            planner._model_tokens_reserved = 0
            planner._query_cache = {}
            planner._query_locks = {}
            try:
                _ensure_operation_active(context, operation="planner")
                return await invoke()
            finally:
                (
                    planner.budget,
                    planner._initial_model_tokens,
                    planner._model_tokens_used,
                    planner._model_tokens_reserved,
                    planner._query_cache,
                    planner._query_locks,
                ) = saved

    @staticmethod
    def _runtime() -> BaselineRuntimeContext:
        return get_runtime(BaselineRuntimeContext).context

    def _plan_from_state(self, state: BaselineState) -> ResearchPlan:
        artifact_id = state["plan_artifact_id"]
        if artifact_id is None:
            raise WorkflowInvariantError(code="PLAN_MISSING")
        try:
            value: object = json.loads(self.artifact_store.get_bytes(artifact_id))
            return ResearchPlan.model_validate(value)
        except (FileNotFoundError, TypeError, ValueError):
            raise ArtifactIntegrityError("persisted plan artifact is corrupt") from None

    def _latest_work(self, state: BaselineState, kind: str) -> dict[str, object]:
        for artifact_id in reversed(state["baseline_work_artifact_ids"]):
            try:
                value: object = json.loads(self.artifact_store.get_bytes(artifact_id))
            except (FileNotFoundError, TypeError, ValueError):
                continue
            if isinstance(value, dict):
                mapping = cast("dict[str, object]", value)
                if mapping.get("kind") == kind:
                    return mapping
        raise ArtifactIntegrityError("required baseline work artifact is missing")

    def _cache_entry(
        self,
        *,
        key: Any,
        artifact_id: str,
        operation_id: str,
        usage: ResourceUsage,
        context: BaselineRuntimeContext,
        provider_call: ProviderCallRecord | None = None,
        provider_call_receipt_id: str | None = None,
    ) -> CacheEntry:
        if provider_call is None or provider_call_receipt_id is None:
            raise ArtifactIntegrityError("completion cache requires provider audit")
        metadata: dict[str, object] = {
            "schema_version": "baseline-cache-completion-v1",
            "artifact_ids": [artifact_id],
            "operation_id": operation_id,
            "outcome": "success",
            "producer_run_id": context.run_id,
            "producer_thread_id": context.thread_id,
            "provider_call": provider_call.model_dump(mode="json"),
            "provider_call_receipt_id": provider_call_receipt_id,
        }
        return CacheEntry(
            key_sha256=cache_key_sha256(key),
            value_artifact_id=artifact_id,
            producer_version="baseline-v1",
            usage=usage,
            created_at=context.utc_now(),
            metadata=cast("Any", metadata),
        )

    def _record_provider_call(
        self,
        *,
        context: BaselineRuntimeContext,
        call_fields: Mapping[str, object] | None,
        usage: ResourceUsage,
        started_at: datetime,
        finished_at: datetime,
        cache_hit: bool,
        outcome_code: str,
        budget_replay: _BudgetReplayFact,
        result_artifact_ids: Sequence[str] = (),
    ) -> ProviderCallRecord | None:
        audit = _audit_buffer(context)
        if call_fields is None or audit.graph_node is None:
            return None
        composition = getattr(self, "_audit_composition", None)
        if not isinstance(composition, _AuditComposition):
            return None
        identity = _hash_json(
            json.loads(_JSON_OBJECT_ADAPTER.dump_json(dict(call_fields)))
        )
        attempt = audit.identity_counts.get(identity, 0) + 1
        payload = {
            **dict(call_fields),
            "node": audit.graph_node,
            "started_at": started_at,
            "finished_at": finished_at,
            "latency_ms": round((finished_at - started_at).total_seconds() * 1000),
            "attempt": attempt,
            "cache_hit": cache_hit,
            "outcome_code": outcome_code,
            "usage": usage,
        }
        if payload.get("operation") == "model":
            if composition.pricing_status == "unknown":
                if usage.cost_usd is not None:
                    raise UsageIntegrityError
                payload["pricing_snapshot_id"] = None
                payload["estimated_cost_usd"] = None
            else:
                matches = tuple(
                    item
                    for item in composition.pricing_snapshots
                    if item.provider_id == payload.get("provider_id")
                    and item.endpoint_type == payload.get("endpoint_type")
                    and item.model_id == payload.get("model_id")
                )
                if len(matches) != 1:
                    raise UsageIntegrityError
                snapshot = matches[0]
                estimated = (
                    Decimal(0)
                    if cache_hit
                    else CostCalculator.estimate(usage, snapshot).total_usd
                )
                if usage.cost_usd != estimated:
                    raise UsageIntegrityError
                payload["pricing_snapshot_id"] = snapshot.snapshot_id
                payload["estimated_cost_usd"] = estimated
        elif (
            usage.cost_usd is not None
            and usage.cost_usd != Decimal(0)
        ):
            raise UsageIntegrityError
        record = ProviderCallRecord.model_validate(payload)
        receipt_id = _put_audit_receipt(
            composition.artifact_store,
            kind="provider-call",
            run_id=context.run_id,
            thread_id=context.thread_id,
            receipt_key=f"call:{budget_replay.operation_id}:{attempt}",
            payload={
                "record": record.model_dump(mode="json"),
                "budget_replay": budget_replay.model_dump(mode="json"),
                "result_artifact_ids": list(result_artifact_ids),
            },
        )
        audit.identity_counts[identity] = attempt
        audit.provider_calls.append(record)
        audit.provider_receipt_ids.append(receipt_id)
        return record

    def _attach_provider_result(
        self,
        context: BaselineRuntimeContext,
        artifact_id: str,
    ) -> None:
        audit = _audit_buffer(context)
        pending = audit.pending_provider_call
        if pending is None:
            raise ArtifactIntegrityError("provider result has no pending audit")
        record = self._record_provider_call(
            context=context,
            call_fields=pending.call_fields,
            usage=pending.usage,
            started_at=pending.started_at,
            finished_at=pending.finished_at,
            cache_hit=False,
            outcome_code="SUCCESS",
            budget_replay=pending.budget_replay,
            result_artifact_ids=(artifact_id,),
        )
        if record is None:
            raise ArtifactIntegrityError("provider result audit was not published")
        audit.pending_provider_call = None

    def _finalize_pending_failure(
        self,
        context: BaselineRuntimeContext,
        *,
        outcome_code: str,
        result_artifact_ids: Sequence[str] = (),
    ) -> None:
        audit = _audit_buffer(context)
        pending = audit.pending_provider_call
        if pending is None:
            return
        record = self._record_provider_call(
            context=context,
            call_fields=pending.call_fields,
            usage=pending.usage,
            started_at=pending.started_at,
            finished_at=pending.finished_at,
            cache_hit=False,
            outcome_code=outcome_code,
            budget_replay=pending.budget_replay,
            result_artifact_ids=result_artifact_ids,
        )
        if record is None:
            raise ArtifactIntegrityError("provider failure audit was not published")
        audit.pending_provider_call = None

    def _persist_provider_result(
        self,
        context: BaselineRuntimeContext,
        *,
        payload_factory: Callable[[], object],
        media_type: str,
    ) -> tuple[str, object]:
        artifact_id: str | None = None
        try:
            payload = payload_factory()
            ref = self.artifact_store.put_bytes(
                _canonical_bytes(payload),
                media_type=media_type,
            )
            artifact_id = ref.artifact_id
            self._audit_result_artifact(context, artifact_id)
            self._attach_provider_result(context, artifact_id)
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception as error:
            try:
                self._finalize_pending_failure(
                    context,
                    outcome_code=(
                        error.code
                        if isinstance(error, ProviderError)
                        else "INTERNAL_ERROR"
                    ),
                    result_artifact_ids=(
                        () if artifact_id is None else (artifact_id,)
                    ),
                )
            except BaseException:  # noqa: BLE001, S110 - preserve store primary
                pass
            raise
        return artifact_id, payload

    def _publish_completion_cache(
        self,
        key: Any,
        candidate: CacheEntry,
        *,
        context: BaselineRuntimeContext,
        operation_id: str,
        expected_call_fields: Mapping[str, object],
        budget_node: str,
        validate_result: Callable[[CacheEntry], object],
    ) -> CacheEntry:
        returned = self.cache.put_if_absent(key, candidate)
        if (
            type(returned) is not CacheEntry
            or returned != candidate
            or not _same_runtime_shape(returned, candidate)
        ):
            raise CacheIntegrityError("completion cache acknowledgement is corrupt")
        validate_result(returned)
        self._validate_cached_completion(
            returned,
            context=context,
            operation_id=operation_id,
            expected_key=key,
            expected_call_fields=expected_call_fields,
            budget_node=budget_node,
        )
        readback = self.cache.get(key)
        if (
            type(readback) is not CacheEntry
            or readback != candidate
            or not _same_runtime_shape(readback, candidate)
        ):
            raise CacheIntegrityError("completion cache publication is not readable")
        validate_result(readback)
        self._validate_cached_completion(
            readback,
            context=context,
            operation_id=operation_id,
            expected_key=key,
            expected_call_fields=expected_call_fields,
            budget_node=budget_node,
        )
        return readback

    @staticmethod
    def _budget_replay(
        reservation: BudgetReservation,
        *,
        actual: ResourceUsage | None,
        decision: Literal["charge", "observe", "release"],
    ) -> _BudgetReplayFact:
        return _BudgetReplayFact(
            operation_id=reservation.idempotency_key,
            budget_node=reservation.node,
            estimate=reservation.estimate,
            actual=actual,
            decision=decision,
        )

    def _last_provider_audit(
        self,
        context: BaselineRuntimeContext,
    ) -> tuple[ProviderCallRecord | None, str | None]:
        audit = _audit_buffer(context)
        if not audit.provider_calls:
            return None, None
        return (
            audit.provider_calls[-1],
            audit.provider_receipt_ids[-1],
        )

    def _audit_result_artifact(
        self,
        context: BaselineRuntimeContext,
        artifact_id: str,
    ) -> None:
        audit = _audit_buffer(context)
        if audit.graph_node is not None:
            audit.result_artifact_ids.append(artifact_id)

    def _load_cached_json(self, entry: CacheEntry, *, label: str) -> object:
        try:
            raw = self.artifact_store.get_bytes(entry.value_artifact_id)
            value: object = json.loads(
                raw,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
            if _canonical_bytes(value) != raw:
                raise CacheIntegrityError(f"cached {label} result is not canonical")
            return value
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError(f"cached {label} result is corrupt") from None

    @staticmethod
    def _require_content_artifact_id(value: object) -> None:
        if (
            type(value) is not str
            or not value.startswith("sha256:")
            or len(value) != 71
            or any(character not in "0123456789abcdef" for character in value[7:])
        ):
            raise CacheIntegrityError("cached model result artifact identity is corrupt")

    def _load_cached_model_result(
        self,
        entry: CacheEntry,
        *,
        key: ModelCacheKey,
        output_schema: type[Any] | None,
    ) -> object:
        try:
            value = self._load_cached_json(entry, label="model")
            if type(value) is not dict:
                raise CacheIntegrityError("cached model result is corrupt")
            payload = cast("dict[str, object]", value)
            if output_schema is None:
                if key.endpoint_type != "complete":
                    raise CacheIntegrityError("cached model endpoint is corrupt")
                result: ModelResult[str] | StructuredModelResult[Any] = (
                    _MODEL_RESULT_ADAPTER.validate_json(
                        _canonical_bytes(payload),
                        strict=True,
                    )
                )
            else:
                if key.endpoint_type != "structured":
                    raise CacheIntegrityError("cached model endpoint is corrupt")
                typed_output = output_schema.model_validate_json(
                    _canonical_bytes(payload.get("output")),
                    strict=True,
                )
                result = TypeAdapter(StructuredModelResult[Any]).validate_json(
                    _canonical_bytes(payload),
                    strict=True,
                ).model_copy(
                    update={"output": typed_output},
                )
                if result.output_schema_hash != key.output_schema_hash:
                    raise CacheIntegrityError("cached model schema is corrupt")
            if (
                result.provider_id != key.provider_id
                or result.model_id != key.model_id
            ):
                raise CacheIntegrityError("cached model identity is corrupt")
            self._require_content_artifact_id(result.raw_response_artifact_id)
            _require_exact_usage(result.usage, label="cached model carried usage")
            normalized_cost = self.usage_cost_resolver.resolve_cost(
                operation="model",
                provider_id=key.provider_id,
                model_id=key.model_id,
                outcome="success",
                usage=result.usage,
            )
            normalized_usage = result.usage.model_copy(
                update={"cost_usd": normalized_cost}
            )
            if (
                normalized_usage != entry.usage
                or _canonical_bytes(result.model_dump(mode="json"))
                != _canonical_bytes(payload)
            ):
                raise CacheIntegrityError("cached model result is inconsistent")
            return result
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError("cached model result is corrupt") from None

    def _load_cached_embeddings(
        self,
        entry: CacheEntry,
        *,
        texts: tuple[str, ...],
    ) -> tuple[tuple[float, ...], ...]:
        try:
            raw = self._load_cached_json(entry, label="embedding")
            if type(raw) is not list:
                raise CacheIntegrityError("cached embeddings are corrupt")
            raw_vectors = cast("list[object]", raw)
            vectors: list[tuple[float, ...]] = []
            for vector in raw_vectors:
                if type(vector) is not list:
                    raise CacheIntegrityError("cached embeddings are corrupt")
                values: list[float] = []
                for item in cast("list[object]", vector):
                    if type(item) is not float or not math.isfinite(item):
                        raise CacheIntegrityError("cached embeddings are corrupt")
                    values.append(item)
                vectors.append(tuple(values))
            validated = validate_embeddings(texts, vectors)
            if _canonical_bytes(validated) != _canonical_bytes(raw_vectors):
                raise CacheIntegrityError("cached embeddings are not canonical")
            return validated
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError("cached embeddings are corrupt") from None

    def _load_cached_search_hits(
        self,
        entry: CacheEntry,
        *,
        key: SearchCacheKey,
    ) -> list[SearchHit]:
        try:
            raw = self._load_cached_json(entry, label="search")
            validated = _SEARCH_HITS_ADAPTER.validate_json(
                _canonical_bytes(raw),
                strict=True,
            )
            limit = key.complete_parameters.get("limit")
            if type(limit) is not int or len(validated) > limit:
                raise CacheIntegrityError("cached search result exceeds its key limit")
            if (
                _canonical_bytes([item.model_dump(mode="json") for item in validated])
                != _canonical_bytes(raw)
            ):
                raise CacheIntegrityError("cached search result is not canonical")
            return validated
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError("cached search result is corrupt") from None

    def _load_cached_raw_document(
        self,
        entry: CacheEntry,
        *,
        key: FetchCacheKey,
    ) -> RawDocument:
        try:
            raw = self._load_cached_json(entry, label="fetch")
            validated = RawDocument.model_validate_json(
                _canonical_bytes(raw),
                strict=True,
            )
            if (
                str(validated.requested_url) != str(key.canonical_url)
                or validated.content_type not in key.accepted_content_types
                or _canonical_bytes(validated.model_dump(mode="json"))
                != _canonical_bytes(raw)
            ):
                raise CacheIntegrityError("cached fetch result does not match its key")
            return validated
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError("cached fetch result is corrupt") from None

    def _load_cached_parsed_document(
        self,
        entry: CacheEntry,
        *,
        key: ParseCacheKey,
        raw_document: RawDocument,
    ) -> ParsedDocument:
        try:
            raw = self._load_cached_json(entry, label="parse")
            validated = ParsedDocument.model_validate_json(
                _canonical_bytes(raw),
                strict=True,
            )
            if (
                validated.parser_id != key.parser_id
                or validated.parser_version != key.parser_version
                or str(validated.canonical_url) != str(raw_document.final_url)
                or _canonical_bytes(validated.model_dump(mode="json"))
                != _canonical_bytes(raw)
            ):
                raise CacheIntegrityError("cached parse result does not match its key")
            return validated
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError("cached parse result is corrupt") from None

    @staticmethod
    def _call_fields_from_record(record: ProviderCallRecord) -> dict[str, object]:
        return record.model_dump(
            mode="python",
            exclude={
                "attempt",
                "cache_hit",
                "estimated_cost_usd",
                "finished_at",
                "latency_ms",
                "node",
                "outcome_code",
                "pricing_snapshot_id",
                "started_at",
                "usage",
            },
        )

    def _validate_cached_completion(
        self,
        entry: CacheEntry,
        *,
        context: BaselineRuntimeContext,
        operation_id: str,
        expected_key: Any,
        expected_call_fields: Mapping[str, object],
        budget_node: str,
    ) -> tuple[ProviderCallRecord, str, bool]:
        try:
            if type(entry) is not CacheEntry:
                raise CacheIntegrityError("cache entry uses an invalid runtime type")
            if (
                entry.key_sha256 != cache_key_sha256(expected_key)
                or entry.producer_version != "baseline-v1"
                or type(entry.created_at) is not datetime
                or entry.created_at.tzinfo is None
                or entry.created_at.utcoffset() is None
            ):
                raise CacheIntegrityError("cache entry identity is corrupt")
            _require_exact_usage(entry.usage, label="completion cache usage")
            metadata_value = entry.metadata
            expected_metadata = {
                "artifact_ids",
                "schema_version",
                "operation_id",
                "outcome",
                "producer_run_id",
                "producer_thread_id",
                "provider_call",
                "provider_call_receipt_id",
            }
            if type(metadata_value) is not _CACHE_METADATA_MAP_TYPE:
                raise CacheIntegrityError("cache metadata uses an invalid runtime type")
            metadata = cast("dict[str, object]", metadata_value)
            if set(metadata) != expected_metadata:
                raise CacheIntegrityError("completion cache metadata is incomplete")
            artifact_ids = metadata["artifact_ids"]
            producer_operation_id = metadata["operation_id"]
            producer_run_id = metadata["producer_run_id"]
            producer_thread_id = metadata["producer_thread_id"]
            receipt_id = metadata["provider_call_receipt_id"]
            call_value = metadata["provider_call"]
            if (
                type(artifact_ids) is not tuple
                or artifact_ids != (entry.value_artifact_id,)
                or metadata["schema_version"] != "baseline-cache-completion-v1"
                or metadata["outcome"] != "success"
                or type(producer_operation_id) is not str
                or not producer_operation_id
                or type(producer_run_id) is not str
                or not producer_run_id
                or type(producer_thread_id) is not str
                or not producer_thread_id
                or type(receipt_id) is not str
                or not receipt_id
                or type(call_value) is not type(metadata)
            ):
                raise CacheIntegrityError("completion cache metadata is corrupt")
            stored_call = ProviderCallRecord.model_validate_json(
                _canonical_bytes(call_value),
                strict=True,
            )
            if (
                _canonical_bytes(stored_call.model_dump(mode="json"))
                != _canonical_bytes(call_value)
                or stored_call.cache_hit
                or stored_call.outcome_code != "SUCCESS"
                or stored_call.usage != entry.usage
                or stored_call.node != _audit_buffer(context).graph_node
            ):
                raise CacheIntegrityError("cached provider call is corrupt")
            expected_json: object = json.loads(
                _JSON_OBJECT_ADAPTER.dump_json(dict(expected_call_fields))
            )
            actual_json_value: object = json.loads(
                _JSON_OBJECT_ADAPTER.dump_json(
                    self._call_fields_from_record(stored_call)
                )
            )
            if type(expected_json) is not dict or type(actual_json_value) is not dict:
                raise CacheIntegrityError("cached provider call fields are corrupt")
            expected_mapping = cast("dict[str, object]", expected_json)
            actual_mapping = cast("dict[str, object]", actual_json_value)
            if any(
                key not in actual_mapping or actual_mapping[key] != expected
                for key, expected in expected_mapping.items()
            ) or any(
                value is not None
                for key, value in actual_mapping.items()
                if key not in expected_mapping
            ):
                raise CacheIntegrityError("cached provider call does not match its key")
            receipt = _load_audit_receipt(self.artifact_store, receipt_id)
            if (
                receipt is None
                or receipt.kind != "provider-call"
                or receipt.run_id != producer_run_id
                or receipt.thread_id != producer_thread_id
            ):
                raise CacheIntegrityError("cached provider receipt is corrupt")
            receipt_call = ProviderCallRecord.model_validate_json(
                _canonical_bytes(receipt.payload["record"]),
                strict=True,
            )
            replay = _BudgetReplayFact.model_validate_json(
                _canonical_bytes(receipt.payload["budget_replay"]),
                strict=True,
            )
            result_ids = receipt.payload["result_artifact_ids"]
            if (
                receipt_call != stored_call
                or type(result_ids) is not list
                or result_ids != [entry.value_artifact_id]
                or replay.operation_id != producer_operation_id
                or replay.budget_node != budget_node
                or replay.decision != "charge"
                or replay.actual != entry.usage
                or receipt.receipt_key
                != f"call:{producer_operation_id}:{stored_call.attempt}"
            ):
                raise CacheIntegrityError("cached provider receipt is inconsistent")
            same_identity = (
                producer_run_id == context.run_id
                and producer_thread_id == context.thread_id
            )
            same_operation = producer_operation_id == operation_id
            if same_operation and not same_identity:
                raise CacheIntegrityError("cache producer identity is inconsistent")
            return stored_call, receipt_id, same_identity and same_operation
        except CacheIntegrityError:
            raise
        except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
            raise
        except Exception:  # noqa: BLE001 - public-safe cache integrity boundary
            raise CacheIntegrityError("completion cache is corrupt") from None

    async def _observe_cached(
        self,
        entry: CacheEntry,
        *,
        context: BaselineRuntimeContext,
        operation_id: str,
        operation: str,
        provider_id: str,
        model_id: str | None,
        node: str,
        expected_key: Any,
        expected_call_fields: Mapping[str, object],
        search_calls: int = 0,
        pages: int = 0,
        tokens: int = 0,
    ) -> bool:
        stored_call, stored_receipt_id, produced_by_this_operation = (
            self._validate_cached_completion(
                entry,
                context=context,
                operation_id=operation_id,
                expected_key=expected_key,
                expected_call_fields=expected_call_fields,
                budget_node=node,
            )
        )
        reservation = self._preflight(
            context=context,
            operation=operation,
            provider_id=provider_id,
            model_id=model_id,
            node=node,
            operation_id=(
                operation_id if produced_by_this_operation else f"{operation_id}:cache-hit"
            ),
            search_calls=search_calls if produced_by_this_operation else 0,
            pages=pages if produced_by_this_operation else 0,
            tokens=tokens if produced_by_this_operation else 0,
        )
        observed_usage = self._settle(
            reservation,
            context=context,
            operation=operation,
            provider_id=provider_id,
            model_id=model_id,
            usage=entry.usage,
            charge=produced_by_this_operation,
            outcome="success" if produced_by_this_operation else "CACHE_HIT",
        )
        if produced_by_this_operation:
            audit = _audit_buffer(context)
            audit.provider_calls.append(stored_call)
            audit.provider_receipt_ids.append(stored_receipt_id)
            identity = _hash_json(
                json.loads(
                    _JSON_OBJECT_ADAPTER.dump_json(dict(expected_call_fields))
                )
            )
            audit.identity_counts[identity] = max(
                audit.identity_counts.get(identity, 0),
                stored_call.attempt,
            )
        else:
            fields = dict(expected_call_fields)
            now = context.utc_now()
            self._record_provider_call(
                context=context,
                call_fields=fields,
                usage=observed_usage,
                started_at=now,
                finished_at=now,
                cache_hit=True,
                outcome_code="CACHE_HIT",
                budget_replay=self._budget_replay(
                    reservation,
                    actual=observed_usage,
                    decision="observe",
                ),
                result_artifact_ids=(entry.value_artifact_id,),
            )
        return produced_by_this_operation

    def _preflight(
        self,
        *,
        context: BaselineRuntimeContext,
        operation: str,
        provider_id: str,
        model_id: str | None,
        node: str,
        operation_id: str,
        search_calls: int = 0,
        pages: int = 0,
        tokens: int = 0,
        input_tokens: int | None = None,
        output_tokens: int = 0,
        retries: int = 0,
        wall_seconds: float = 0.0,
    ) -> BudgetReservation:
        known_cost = context.config.budget.max_cost_usd is not None
        conservative_input_tokens = tokens if input_tokens is None else input_tokens
        total_tokens = conservative_input_tokens + output_tokens
        conservative_usage = ResourceUsage(
            input_tokens=conservative_input_tokens,
            output_tokens=output_tokens,
            reasoning_tokens=0,
            cached_tokens=0,
            total_tokens=total_tokens,
            search_calls=search_calls,
            pages=pages,
            retries=retries,
            wall_seconds=wall_seconds,
            cost_usd=Decimal(0) if known_cost else None,
        )
        cost = self.usage_cost_resolver.resolve_cost(
            operation=operation,
            provider_id=provider_id,
            model_id=model_id,
            outcome="preflight",
            usage=conservative_usage,
        )
        if known_cost and cost is None:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=provider_id,
                operation=operation,
                public_message="operation pricing is unavailable",
                retryable=False,
            )
        return context.budget_accountant.reserve(
            ResourceEstimate(
                search_calls=search_calls,
                pages=pages,
                tokens=total_tokens,
                retries=retries,
                wall_seconds=wall_seconds,
                cost_usd=cost,
            ),
            node=node,
            idempotency_key=operation_id,
        )

    def _settle(
        self,
        reservation: BudgetReservation,
        *,
        context: BaselineRuntimeContext,
        operation: str,
        provider_id: str,
        model_id: str | None,
        usage: ResourceUsage,
        charge: bool = True,
        outcome: str = "success",
    ) -> ResourceUsage:
        cost = self.usage_cost_resolver.resolve_cost(
            operation=operation,
            provider_id=provider_id,
            model_id=model_id,
            outcome=outcome,
            usage=usage,
        )
        actual = usage.model_copy(update={"cost_usd": cost})
        context.budget_accountant.settle(reservation, actual=actual, charge=charge)
        return actual

    @staticmethod
    def _post_settlement_overages(
        context: BaselineRuntimeContext,
        snapshot: BudgetSnapshot,
    ) -> frozenset[str]:
        budget = context.config.budget
        overages: set[str] = set()
        checks: tuple[tuple[str, int | float, int | float], ...] = (
            ("search_calls", snapshot.used_search_calls, budget.max_search_calls),
            ("pages", snapshot.used_pages, budget.max_pages),
            ("tokens", snapshot.used_tokens, budget.max_total_tokens),
            (
                "wall_seconds",
                snapshot.used_wall_seconds,
                budget.max_wall_time_seconds,
            ),
            ("retries", snapshot.used_retries, budget.max_retries),
        )
        for dimension, used, limit in checks:
            if used > limit:
                overages.add(dimension)
        if (
            budget.max_cost_usd is not None
            and snapshot.used_cost_usd is not None
            and snapshot.used_cost_usd > budget.max_cost_usd
        ):
            overages.add("cost_usd")
        return frozenset(overages)

    def _usage(self, provider: object, *, search_calls: int = 0, pages: int = 0) -> ResourceUsage:
        reported = getattr(provider, "last_usage", None)
        if isinstance(reported, ResourceUsage):
            return reported
        return ResourceUsage(
            input_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            cached_tokens=0,
            total_tokens=0,
            search_calls=search_calls,
            pages=pages,
            retries=0,
            wall_seconds=0.0,
            cost_usd=None,
        )

    async def _invoke_metered[Result](
        self,
        *,
        context: BaselineRuntimeContext,
        provider: object,
        provider_id: str,
        model_id: str | None,
        operation: str,
        node: str,
        operation_id: str,
        invoke: Callable[[], Awaitable[Result]],
        result_usage: Callable[[Result], ResourceUsage | None],
        fallback_usage: ResourceUsage,
        search_calls: int = 0,
        pages: int = 0,
        tokens: int = 0,
        input_tokens: int | None = None,
        output_tokens: int = 0,
        call_fields: Mapping[str, object] | None = None,
    ) -> tuple[Result, ResourceUsage]:
        if _audit_buffer(context).pending_provider_call is not None:
            raise ArtifactIntegrityError(
                "provider result audit publication is incomplete"
            )
        provider_lock = await self._lock_for(_hash_json({"provider": provider_id}))
        async with provider_lock:
            _ensure_operation_active(context, operation=operation)
            observer = provider if isinstance(provider, InvocationUsageObserver) else None
            if observer is not None and observer.consume_invocation_usage() is not None:
                raise UsageIntegrityError
            before = context.budget_accountant.snapshot()
            retries = max(
                0,
                context.config.budget.max_retries
                - before.used_retries
                - before.reserved_retries,
            )
            wall_seconds = max(
                0.0,
                float(context.config.budget.max_wall_time_seconds)
                - before.used_wall_seconds
                - before.reserved_wall_seconds,
            )
            reservation = self._preflight(
                context=context,
                operation=operation,
                provider_id=provider_id,
                model_id=model_id,
                node=node,
                operation_id=operation_id,
                search_calls=search_calls,
                pages=pages,
                tokens=tokens,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                retries=retries,
                wall_seconds=wall_seconds,
            )
            try:
                started_at = context.utc_now()
                started = _ensure_operation_active(context, operation=operation)
            except Exception:
                context.budget_accountant.release(reservation)
                raise
            except BaseException:
                try:
                    context.budget_accountant.release(reservation)
                except BaseException:  # noqa: BLE001, S110 - gate primary must win
                    pass
                raise
            try:
                result = await invoke()
            except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
                try:
                    observed = observer.consume_invocation_usage() if observer is not None else None
                    if observed is None:
                        context.budget_accountant.release(reservation)
                    else:
                        self._settle(
                            reservation,
                            context=context,
                            operation=operation,
                            provider_id=provider_id,
                            model_id=model_id,
                            usage=observed,
                            outcome="CANCELLED",
                        )
                except BaseException:  # noqa: BLE001, S110 - hard primary must win
                    pass
                raise
            except Exception as error:
                try:
                    finished = context.monotonic()
                    finished_at = context.utc_now()
                    observed = observer.consume_invocation_usage() if observer is not None else None
                    carried = error.usage if isinstance(error, ProviderError) else None
                    if observed is not None and carried is not None and observed != carried:
                        settled = self._settle(
                            reservation,
                            context=context,
                            operation=operation,
                            provider_id=provider_id,
                            model_id=model_id,
                            usage=observed,
                            outcome="USAGE_INTEGRITY",
                        )
                        self._record_provider_call(
                            context=context,
                            call_fields=call_fields,
                            usage=settled,
                            started_at=started_at,
                            finished_at=finished_at,
                            cache_hit=False,
                            outcome_code="USAGE_INTEGRITY",
                            budget_replay=self._budget_replay(
                                reservation,
                                actual=settled,
                                decision="charge",
                            ),
                        )
                        raise UsageIntegrityError from None
                    actual = observed or carried
                    if actual is None:
                        reported = getattr(provider, "last_usage", None)
                        actual = reported if isinstance(reported, ResourceUsage) else None
                    if actual is None and (search_calls or pages):
                        elapsed = finished - started
                        if not math.isfinite(elapsed):
                            context.budget_accountant.release(reservation)
                            raise UsageIntegrityError from None
                        actual = fallback_usage.model_copy(
                            update={"wall_seconds": max(0.0, elapsed)}
                        )
                    if actual is None:
                        context.budget_accountant.release(reservation)
                        budget_replay = self._budget_replay(
                            reservation,
                            actual=None,
                            decision="release",
                        )
                        audit_usage = fallback_usage.model_copy(
                            update={
                                "wall_seconds": max(0.0, finished - started),
                                "cost_usd": (
                                    None
                                    if context.config.budget.max_cost_usd is None
                                    else Decimal(0)
                                ),
                            }
                        )
                    else:
                        audit_usage = self._settle(
                            reservation,
                            context=context,
                            operation=operation,
                            provider_id=provider_id,
                            model_id=model_id,
                            usage=actual,
                            outcome=(
                                error.code
                                if isinstance(error, ProviderError)
                                else "INTERNAL_ERROR"
                            ),
                        )
                        budget_replay = self._budget_replay(
                            reservation,
                            actual=audit_usage,
                            decision="charge",
                        )
                    try:
                        self._record_provider_call(
                            context=context,
                            call_fields=call_fields,
                            usage=audit_usage,
                            started_at=started_at,
                            finished_at=finished_at,
                            cache_hit=False,
                            outcome_code=(
                                error.code
                                if isinstance(error, ProviderError)
                                else "INTERNAL_ERROR"
                            ),
                            budget_replay=budget_replay,
                        )
                    except UsageIntegrityError:
                        raise
                    except BaseException:
                        audit = _audit_buffer(context)
                        if call_fields is not None and audit.graph_node is not None:
                            audit.pending_provider_call = _PendingProviderCall(
                                call_fields=dict(call_fields),
                                usage=audit_usage,
                                started_at=started_at,
                                finished_at=finished_at,
                                budget_replay=budget_replay,
                            )
                        raise
                except UsageIntegrityError:
                    raise
                except BaseException:  # noqa: BLE001 - preserve recorder primary
                    raise error
                raise
            finished = context.monotonic()
            finished_at = context.utc_now()
            try:
                observed = (
                    observer.consume_invocation_usage() if observer is not None else None
                )
            except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
                raise
            except Exception:
                carried = result_usage(result)
                try:
                    if carried is None:
                        context.budget_accountant.release(reservation)
                    else:
                        settled = self._settle(
                            reservation,
                            context=context,
                            operation=operation,
                            provider_id=provider_id,
                            model_id=model_id,
                            usage=carried,
                            outcome="INTERNAL_ERROR",
                        )
                        self._record_provider_call(
                            context=context,
                            call_fields=call_fields,
                            usage=settled,
                            started_at=started_at,
                            finished_at=finished_at,
                            cache_hit=False,
                            outcome_code="INTERNAL_ERROR",
                            budget_replay=self._budget_replay(
                                reservation,
                                actual=settled,
                                decision="charge",
                            ),
                        )
                except BaseException:  # noqa: BLE001, S110 - preserve observer primary
                    pass
                raise
            carried = result_usage(result)
            if observed is not None and carried is not None and observed != carried:
                settled = self._settle(
                    reservation,
                    context=context,
                    operation=operation,
                    provider_id=provider_id,
                    model_id=model_id,
                    usage=observed,
                    outcome="USAGE_INTEGRITY",
                )
                self._record_provider_call(
                    context=context,
                    call_fields=call_fields,
                    usage=settled,
                    started_at=started_at,
                    finished_at=finished_at,
                    cache_hit=False,
                    outcome_code="USAGE_INTEGRITY",
                    budget_replay=self._budget_replay(
                        reservation,
                        actual=settled,
                        decision="charge",
                    ),
                )
                raise UsageIntegrityError
            actual = observed or carried
            if actual is None and observer is None:
                reported = getattr(provider, "last_usage", None)
                actual = reported if isinstance(reported, ResourceUsage) else None
            if actual is None:
                elapsed = finished - started
                if not math.isfinite(elapsed):
                    context.budget_accountant.release(reservation)
                    raise UsageIntegrityError
                actual = fallback_usage.model_copy(update={"wall_seconds": max(0.0, elapsed)})
            actual = self._settle(
                reservation,
                context=context,
                operation=operation,
                provider_id=provider_id,
                model_id=model_id,
                usage=actual,
            )
            snapshot = context.budget_accountant.snapshot()
            overages = self._post_settlement_overages(context, snapshot)
            if overages:
                self._record_provider_call(
                    context=context,
                    call_fields=call_fields,
                    usage=actual,
                    started_at=started_at,
                    finished_at=finished_at,
                    cache_hit=False,
                    outcome_code="BUDGET_EXCEEDED",
                    budget_replay=self._budget_replay(
                        reservation,
                        actual=actual,
                        decision="charge",
                    ),
                )
                raise BudgetExceeded(cast("Any", overages), snapshot)
            audit = _audit_buffer(context)
            if call_fields is not None and audit.graph_node is not None:
                if audit.pending_provider_call is not None:
                    raise ArtifactIntegrityError(
                        "provider result audit publication is incomplete"
                    )
                audit.pending_provider_call = _PendingProviderCall(
                    call_fields=dict(call_fields),
                    usage=actual,
                    started_at=started_at,
                    finished_at=finished_at,
                    budget_replay=self._budget_replay(
                        reservation,
                        actual=actual,
                        decision="charge",
                    ),
                )
            return result, actual

    async def _cached_model_call[Result](
        self,
        *,
        provider: ModelProvider,
        request: ModelRequest,
        output_schema: type[Any] | None,
        invoke: Callable[[], Awaitable[Result]],
    ) -> Result:
        context = self._runtime()
        budget_node = (
            "Planner"
            if request.prompt_version.startswith(self.initial_plan_generator.prompt_version)
            else "Writer"
        )
        graph_node = _audit_buffer(context).graph_node or (
            "Plan" if budget_node == "Planner" else "DraftReport"
        )
        operation_id = stable_operation_id(
            run_id=context.run_id,
            workflow_id=context.config.workflow_id,
            node=graph_node,
            logical_input=request.model_dump(mode="json"),
            node_attempt=_audit_buffer(context).node_attempt or 1,
        )
        key = ModelCacheKey(
            provider_id=provider.provider_id,
            endpoint_type="structured" if output_schema is not None else "complete",
            model_id=request.model_id,
            prompt_version=request.prompt_version,
            system_prompt_hash=request.system_prompt_hash,
            tool_schema_hash=request.tool_schema_hash,
            output_schema_hash=request.output_schema_hash,
            temperature=request.temperature,
            seed=request.seed,
            canonical_request_hash=_hash_json(request.model_dump(mode="json")),
        )
        input_token_estimate = 512 + sum(
            len(message.content.encode("utf-8")) for message in request.messages
        )
        token_estimate = input_token_estimate + request.max_output_tokens
        call_fields: dict[str, object] = {
            "operation": "model",
            "provider_id": provider.provider_id,
            "endpoint_type": key.endpoint_type,
            "model_id": request.model_id,
            "model_revision": cast("Any", provider).model_revision,
            "request_sha256": key.canonical_request_hash,
            "complete_parameters": {"seed_supported": self.seed_supported},
            "prompt_version": request.prompt_version,
            "system_prompt_hash": request.system_prompt_hash,
            "tool_schema_hash": request.tool_schema_hash,
            "output_schema_hash": request.output_schema_hash,
            "temperature": request.temperature,
            "seed": request.seed,
        }

        def load_result(entry: CacheEntry) -> object:
            return self._load_cached_model_result(
                entry,
                key=key,
                output_schema=output_schema,
            )

        lock = await self._lock_for(cache_key_sha256(key))
        async with lock:
            _ensure_operation_active(context, operation="model")
            cached = self.cache.get(key)
            if cached is not None:
                cached_result = load_result(cached)
                await self._observe_cached(
                    cached,
                    context=context,
                    operation_id=operation_id,
                    operation="model",
                    provider_id=provider.provider_id,
                    model_id=request.model_id,
                    node=budget_node,
                    expected_key=key,
                    expected_call_fields=call_fields,
                    tokens=token_estimate,
                )
                self._audit_result_artifact(context, cached.value_artifact_id)
                return cast("Result", cached_result)
            result, usage = await self._invoke_metered(
                context=context,
                provider=provider,
                provider_id=provider.provider_id,
                model_id=request.model_id,
                operation="model",
                node=budget_node,
                operation_id=operation_id,
                invoke=invoke,
                result_usage=lambda item: getattr(item, "usage", None),
                fallback_usage=_zero_usage(
                    cost_known=context.config.budget.max_cost_usd is not None
                ),
                input_tokens=input_token_estimate,
                output_tokens=request.max_output_tokens,
                call_fields=call_fields,
            )
            result_model = cast("Any", result)
            artifact_id, _ = self._persist_provider_result(
                context,
                payload_factory=lambda: result_model.model_dump(mode="json"),
                media_type="application/vnd.deepresearch.model-result+json",
            )
            provider_call, provider_receipt_id = self._last_provider_audit(context)
            self._publish_completion_cache(
                key,
                self._cache_entry(
                    key=key,
                    artifact_id=artifact_id,
                    operation_id=operation_id,
                    usage=usage,
                    context=context,
                    provider_call=provider_call,
                    provider_call_receipt_id=provider_receipt_id,
                ),
                context=context,
                operation_id=operation_id,
                expected_call_fields=call_fields,
                budget_node=budget_node,
                validate_result=load_result,
            )
            return result

    async def _cached_embed_call(
        self,
        *,
        provider: TextEmbedder,
        texts: tuple[str, ...],
        invoke: Callable[[], Awaitable[tuple[tuple[float, ...], ...]]],
    ) -> tuple[tuple[float, ...], ...]:
        context = self._runtime()
        logical = {"texts": texts}
        audit = _audit_buffer(context)
        ordinal = audit.operation_counts.get("embed", 0)
        audit.operation_counts["embed"] = ordinal + 1
        operation_id = stable_operation_id(
            run_id=context.run_id,
            workflow_id=context.config.workflow_id,
            node="RankEvidence",
            logical_input=logical,
            node_attempt=audit.node_attempt or 1,
            ordinal=ordinal,
        )
        key = EmbedCacheKey(
            model_id=provider.model_id,
            model_revision=provider.model_revision,
            snapshot_sha256=provider.snapshot_sha256,
            normalize_embeddings=True,
            canonical_texts_hash=_hash_json(logical),
        )
        call_fields: dict[str, object] = {
            "operation": "embed",
            "provider_id": provider.provider_id,
            "endpoint_type": "embed",
            "model_id": provider.model_id,
            "model_revision": provider.model_revision,
            "request_sha256": key.canonical_texts_hash,
            "complete_parameters": {
                "canonical_texts_hash": key.canonical_texts_hash,
                "normalize_embeddings": True,
                "snapshot_sha256": provider.snapshot_sha256,
            },
        }

        def load_result(entry: CacheEntry) -> tuple[tuple[float, ...], ...]:
            return self._load_cached_embeddings(entry, texts=texts)

        token_estimate = sum(len(item.encode("utf-8")) for item in texts)
        lock = await self._lock_for(cache_key_sha256(key))
        async with lock:
            _ensure_operation_active(context, operation="embed")
            cached = self.cache.get(key)
            if cached is not None:
                cached_vectors = load_result(cached)
                await self._observe_cached(
                    cached,
                    context=context,
                    operation_id=operation_id,
                    operation="embed",
                    provider_id=provider.provider_id,
                    model_id=provider.model_id,
                    node="Ranker",
                    expected_key=key,
                    expected_call_fields=call_fields,
                    tokens=token_estimate,
                )
                self._audit_result_artifact(context, cached.value_artifact_id)
                return cached_vectors
            vectors, usage = await self._invoke_metered(
                context=context,
                provider=provider,
                provider_id=provider.provider_id,
                model_id=provider.model_id,
                operation="embed",
                node="Ranker",
                operation_id=operation_id,
                invoke=invoke,
                result_usage=lambda _result: None,
                fallback_usage=_zero_usage(
                    cost_known=context.config.budget.max_cost_usd is not None
                ),
                tokens=token_estimate,
                call_fields=call_fields,
            )
            artifact_id, _ = self._persist_provider_result(
                context,
                payload_factory=lambda: vectors,
                media_type="application/vnd.deepresearch.embeddings+json",
            )
            provider_call, provider_receipt_id = self._last_provider_audit(context)
            self._publish_completion_cache(
                key,
                self._cache_entry(
                    key=key,
                    artifact_id=artifact_id,
                    operation_id=operation_id,
                    usage=usage,
                    context=context,
                    provider_call=provider_call,
                    provider_call_receipt_id=provider_receipt_id,
                ),
                context=context,
                operation_id=operation_id,
                expected_call_fields=call_fields,
                budget_node="Ranker",
                validate_result=load_result,
            )
            return vectors

    async def validate_request(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        if state["request"] != context.config.request:
            raise WorkflowInvariantError(code="INVALID_WORKFLOW_CONFIG")
        if dict(context.config.prompt_versions) != dict(self.prompt_versions):
            raise WorkflowInvariantError(code="INVALID_WORKFLOW_CONFIG")
        if (
            context.config.ranker_weights_version is not None
            and context.config.ranker_weights_version != self.ranker_weights_version
        ):
            raise WorkflowInvariantError(code="INVALID_WORKFLOW_CONFIG")
        if self.pricing_status == "unknown" and context.config.budget.max_cost_usd is not None:
            raise WorkflowInvariantError(code="INVALID_WORKFLOW_CONFIG")
        if context.config.seed is not None and not self.seed_supported:
            raise WorkflowInvariantError(code="INVALID_WORKFLOW_CONFIG")
        if (
            context.config.request.execution_mode in {"replay", "hybrid"}
            and self.replay_parent is None
        ):
            raise WorkflowInvariantError(code="INVALID_WORKFLOW_CONFIG")
        return {}

    async def run_plan(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        plan = await self._isolated_planner_call(
            context,
            lambda: self.initial_plan_generator.create_plan(
                state["request"],
                deadline=_effective_deadline(context),
                cancellation_token=context.cancellation_token,
            ),
        )
        ref = self.artifact_store.put_bytes(
            _canonical_bytes(plan.model_dump(mode="json")),
            media_type="application/vnd.deepresearch.plan+json",
        )
        self._audit_result_artifact(context, ref.artifact_id)
        return {
            "plan_id": plan.plan_id,
            "plan_artifact_id": ref.artifact_id,
            "pending_subquestion_ids": tuple(item.id for item in plan.subquestions),
        }

    async def decide_next(self, state: BaselineState) -> StateUpdate:
        plan = self._plan_from_state(state)
        stop = decide_baseline_stop(state, plan)
        if stop is not None:
            return {"stop_reason": stop, "is_partial": stop != "SUFFICIENT"}
        active = state["active_subquestion_id"] or state["pending_subquestion_ids"][0]
        return {"active_subquestion_id": active}

    async def search(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        plan = self._plan_from_state(state)
        active_id = state["active_subquestion_id"]
        subquestion = next(item for item in plan.subquestions if item.id == active_id)
        queries = await self._isolated_planner_call(
            context,
            lambda: self.initial_plan_generator.queries_for(
                subquestion,
                plan_id=plan.plan_id,
                deadline=_effective_deadline(context),
                cancellation_token=context.cancellation_token,
            ),
        )
        hits: list[object] = []
        query_ids: list[str] = []
        for ordinal, query in enumerate(queries):
            logical = {"limit": 10, "query": query}
            operation_id = stable_operation_id(
                run_id=state["run_id"],
                workflow_id=context.config.workflow_id,
                node="Search",
                logical_input=logical,
                node_attempt=_audit_buffer(context).node_attempt or 1,
                ordinal=ordinal,
            )
            query_ids.append(f"Q-{_hash_json(logical)}")
            key = SearchCacheKey(
                snapshot_id=self.search_snapshot_id,
                normalized_query=" ".join(query.split()),
                provider_id=self.search_provider.provider_id,
                endpoint_type="search",
                locale="und",
                complete_parameters={"filters": None, "limit": 10},
                time_policy="frozen",
            )
            call_fields: dict[str, object] = {
                "operation": "search",
                "provider_id": key.provider_id,
                "endpoint_type": key.endpoint_type,
                "request_sha256": _hash_json(key.model_dump(mode="json")),
                "snapshot_id": key.snapshot_id,
                "normalized_query": key.normalized_query,
                "locale": key.locale,
                "complete_parameters": dict(key.complete_parameters),
                "time_policy": key.time_policy,
            }

            def load_result(
                entry: CacheEntry,
                cache_key: SearchCacheKey = key,
            ) -> list[SearchHit]:
                return self._load_cached_search_hits(entry, key=cache_key)

            lock = await self._lock_for(cache_key_sha256(key))
            async with lock:
                _ensure_operation_active(context, operation="search")
                cached = self.cache.get(key)
                if cached is not None:
                    validated_hits = load_result(cached)
                    await self._observe_cached(
                        cached,
                        context=context,
                        operation_id=operation_id,
                        operation="search",
                        provider_id=self.search_provider.provider_id,
                        model_id=None,
                        node="Tool",
                        expected_key=key,
                        expected_call_fields=call_fields,
                        search_calls=1,
                    )
                    self._audit_result_artifact(context, cached.value_artifact_id)
                    hits.extend(
                        item.model_dump(mode="json") for item in validated_hits
                    )
                    continue
                search_provider = self.search_provider
                if isinstance(search_provider, UsageReportingSearchProvider):
                    typed_provider = search_provider

                    async def invoke_typed_search(
                        query: str = query,
                        provider: UsageReportingSearchProvider = typed_provider,
                    ) -> ProviderUsageResult[list[SearchHit]]:
                        return await provider.search_with_usage(
                            query,
                            10,
                            None,
                            deadline=_effective_deadline(context),
                            cancellation_token=context.cancellation_token,
                        )

                    envelope, usage = await self._invoke_metered(
                        context=context,
                        provider=typed_provider,
                        operation="search",
                        provider_id=typed_provider.provider_id,
                        model_id=None,
                        node="Tool",
                        operation_id=operation_id,
                        invoke=invoke_typed_search,
                        result_usage=_reported_usage,
                        fallback_usage=self._usage(typed_provider, search_calls=1),
                        search_calls=1,
                        call_fields=call_fields,
                    )
                    result = envelope.value
                else:

                    async def invoke_search(
                        query: str = query,
                        provider: SearchProvider = search_provider,
                    ) -> list[SearchHit]:
                        return await provider.search(
                            query,
                            10,
                            None,
                            deadline=_effective_deadline(context),
                            cancellation_token=context.cancellation_token,
                        )

                    result, usage = await self._invoke_metered(
                        context=context,
                        provider=search_provider,
                        operation="search",
                        provider_id=search_provider.provider_id,
                        model_id=None,
                        node="Tool",
                        operation_id=operation_id,
                        invoke=invoke_search,
                        result_usage=_no_reported_usage,
                        fallback_usage=self._usage(search_provider, search_calls=1),
                        search_calls=1,
                        call_fields=call_fields,
                    )
                artifact_id, encoded_value = self._persist_provider_result(
                    context,
                    payload_factory=lambda search_result=result: [
                        item.model_dump(mode="json") for item in search_result
                    ],
                    media_type="application/vnd.deepresearch.search-results+json",
                )
                encoded_hits = cast("list[object]", encoded_value)
                provider_call, provider_receipt_id = self._last_provider_audit(context)
                self._publish_completion_cache(
                    key,
                    self._cache_entry(
                        key=key,
                        artifact_id=artifact_id,
                        operation_id=operation_id,
                        usage=usage,
                        context=context,
                        provider_call=provider_call,
                        provider_call_receipt_id=provider_receipt_id,
                    ),
                    context=context,
                    operation_id=operation_id,
                    expected_call_fields=call_fields,
                    budget_node="Tool",
                    validate_result=load_result,
                )
                hits.extend(encoded_hits)
        work = self.artifact_store.put_bytes(
            _canonical_bytes({"hits": hits, "kind": "search"}),
            media_type="application/vnd.deepresearch.baseline-work+json",
        )
        return {
            "query_ids": tuple(query_ids),
            "baseline_work_artifact_ids": (
                *state["baseline_work_artifact_ids"],
                work.artifact_id,
            ),
        }

    async def fetch(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        search_work = self._latest_work(state, "search")
        raw_documents: list[object] = []
        urls = sorted(
            {
                str(cast("dict[str, object]", item)["url"])
                for item in cast("list[object]", search_work["hits"])
            }
        )
        for ordinal, url in enumerate(urls):
            operation_id = stable_operation_id(
                run_id=state["run_id"],
                workflow_id=context.config.workflow_id,
                node="Fetch",
                logical_input={"url": url},
                node_attempt=_audit_buffer(context).node_attempt or 1,
                ordinal=ordinal,
            )
            key = FetchCacheKey.model_validate(
                {
                    "snapshot_id": self.fetch_snapshot_id,
                    "canonical_url": url,
                    "fetch_policy": "baseline",
                    "accepted_content_types": ("text/html", "application/pdf"),
                }
            )
            call_fields: dict[str, object] = {
                "operation": "fetch",
                "provider_id": self.fetcher.provider_id,
                "endpoint_type": "fetch",
                "request_sha256": _hash_json(key.model_dump(mode="json")),
                "snapshot_id": key.snapshot_id,
                "complete_parameters": {
                    "canonical_url": str(key.canonical_url),
                    "fetch_policy": key.fetch_policy,
                    "accepted_content_types": list(key.accepted_content_types),
                },
            }

            def load_result(
                entry: CacheEntry,
                cache_key: FetchCacheKey = key,
            ) -> RawDocument:
                return self._load_cached_raw_document(entry, key=cache_key)

            lock = await self._lock_for(cache_key_sha256(key))
            async with lock:
                _ensure_operation_active(context, operation="fetch")
                cached = self.cache.get(key)
                if cached is not None:
                    validated_raw = load_result(cached)
                    await self._observe_cached(
                        cached,
                        context=context,
                        operation_id=operation_id,
                        operation="fetch",
                        provider_id=self.fetcher.provider_id,
                        model_id=None,
                        node="Tool",
                        expected_key=key,
                        expected_call_fields=call_fields,
                        pages=1,
                    )
                    self._audit_result_artifact(context, cached.value_artifact_id)
                    raw_documents.append(validated_raw.model_dump(mode="json"))
                    continue
                fetcher = self.fetcher
                if isinstance(fetcher, UsageReportingFetcher):
                    typed_fetcher = fetcher

                    async def invoke_typed_fetch(
                        url: str = url,
                        provider: UsageReportingFetcher = typed_fetcher,
                    ) -> ProviderUsageResult[RawDocument]:
                        return await provider.fetch_with_usage(
                            url,
                            deadline=_effective_deadline(context),
                            cancellation_token=context.cancellation_token,
                        )

                    envelope, usage = await self._invoke_metered(
                        context=context,
                        provider=typed_fetcher,
                        operation="fetch",
                        provider_id=typed_fetcher.provider_id,
                        model_id=None,
                        node="Tool",
                        operation_id=operation_id,
                        invoke=invoke_typed_fetch,
                        result_usage=_reported_usage,
                        fallback_usage=self._usage(typed_fetcher, pages=1),
                        pages=1,
                        call_fields=call_fields,
                    )
                    raw = envelope.value
                else:

                    async def invoke_fetch(
                        url: str = url,
                        provider: Fetcher = fetcher,
                    ) -> RawDocument:
                        return await provider.fetch(
                            url,
                            deadline=_effective_deadline(context),
                            cancellation_token=context.cancellation_token,
                        )

                    raw, usage = await self._invoke_metered(
                        context=context,
                        provider=fetcher,
                        operation="fetch",
                        provider_id=fetcher.provider_id,
                        model_id=None,
                        node="Tool",
                        operation_id=operation_id,
                        invoke=invoke_fetch,
                        result_usage=_no_reported_usage,
                        fallback_usage=self._usage(fetcher, pages=1),
                        pages=1,
                        call_fields=call_fields,
                    )
                artifact_id, payload_value = self._persist_provider_result(
                    context,
                    payload_factory=lambda raw_document=raw: raw_document.model_dump(
                        mode="json"
                    ),
                    media_type="application/vnd.deepresearch.raw-document+json",
                )
                payload = cast("dict[str, object]", payload_value)
                provider_call, provider_receipt_id = self._last_provider_audit(context)
                self._publish_completion_cache(
                    key,
                    self._cache_entry(
                        key=key,
                        artifact_id=artifact_id,
                        operation_id=operation_id,
                        usage=usage,
                        context=context,
                        provider_call=provider_call,
                        provider_call_receipt_id=provider_receipt_id,
                    ),
                    context=context,
                    operation_id=operation_id,
                    expected_call_fields=call_fields,
                    budget_node="Tool",
                    validate_result=load_result,
                )
                raw_documents.append(payload)
        work = self.artifact_store.put_bytes(
            _canonical_bytes({"documents": raw_documents, "kind": "raw"}),
            media_type="application/vnd.deepresearch.baseline-work+json",
        )
        return {
            "baseline_work_artifact_ids": (
                *state["baseline_work_artifact_ids"],
                work.artifact_id,
            )
        }

    async def parse_and_normalize(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        raw_work = self._latest_work(state, "raw")
        parsed_documents: list[object] = []
        parsed_artifact_ids: list[str] = []
        for ordinal, raw_value in enumerate(
            cast("list[object]", raw_work["documents"])
        ):
            raw = RawDocument.model_validate(raw_value)
            key = ParseCacheKey(
                snapshot_id=_parse_source_snapshot_id(
                    fetch_snapshot_id=self.fetch_snapshot_id,
                    final_url=raw.final_url,
                    content_type=raw.content_type,
                ),
                raw_content_hash=hashlib.sha256(raw.body_bytes).hexdigest(),
                parser_id=self.parser.parser_id,
                parser_version=self.parser.parser_version,
                normalization_version=self.normalization_version,
            )
            operation_id = stable_operation_id(
                run_id=state["run_id"],
                workflow_id=context.config.workflow_id,
                node="ParseAndNormalize",
                logical_input=key.model_dump(mode="json"),
                node_attempt=_audit_buffer(context).node_attempt or 1,
                ordinal=ordinal,
            )
            call_fields: dict[str, object] = {
                "operation": "parse",
                "provider_id": self.parser.parser_id,
                "endpoint_type": "parse",
                "request_sha256": _hash_json(key.model_dump(mode="json")),
                "snapshot_id": key.snapshot_id,
                "complete_parameters": {
                    "raw_content_hash": key.raw_content_hash,
                    "parser_id": key.parser_id,
                    "parser_version": key.parser_version,
                    "normalization_version": key.normalization_version,
                },
            }

            def load_result(
                entry: CacheEntry,
                cache_key: ParseCacheKey = key,
                raw_document: RawDocument = raw,
            ) -> ParsedDocument:
                return self._load_cached_parsed_document(
                    entry,
                    key=cache_key,
                    raw_document=raw_document,
                )

            lock = await self._lock_for(cache_key_sha256(key))
            async with lock:
                _ensure_operation_active(context, operation="parse")
                cached = self.cache.get(key)
                if cached is not None:
                    validated_parsed = load_result(cached)
                    await self._observe_cached(
                        cached,
                        context=context,
                        operation_id=operation_id,
                        operation="parse",
                        provider_id=self.parser.parser_id,
                        model_id=None,
                        node="Tool",
                        expected_key=key,
                        expected_call_fields=call_fields,
                    )
                    self._audit_result_artifact(context, cached.value_artifact_id)
                    parsed_documents.append(validated_parsed.model_dump(mode="json"))
                    parsed_artifact_ids.append(cached.value_artifact_id)
                    continue
                parser = self.parser

                async def invoke_parse(
                    raw_document: RawDocument = raw,
                    provider: Parser = parser,
                ) -> ParsedDocument:
                    return await provider.parse(
                        raw_document,
                        deadline=_effective_deadline(context),
                        cancellation_token=context.cancellation_token,
                    )

                parsed, usage = await self._invoke_metered(
                    context=context,
                    provider=parser,
                    provider_id=parser.parser_id,
                    model_id=None,
                    operation="parse",
                    node="Tool",
                    operation_id=operation_id,
                    invoke=invoke_parse,
                    result_usage=_no_reported_usage,
                    fallback_usage=_zero_usage(
                        cost_known=context.config.budget.max_cost_usd is not None
                    ),
                    call_fields=call_fields,
                )
                def encode_parsed_result(
                    parsed_result: ParsedDocument = parsed,
                    cache_key: ParseCacheKey = key,
                    raw_document: RawDocument = raw,
                ) -> object:
                    if type(parsed_result) is not ParsedDocument:
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.parser.parser_id,
                            operation="parse",
                            public_message="parser result is invalid",
                            retryable=False,
                        )
                    payload = parsed_result.model_dump(mode="json")
                    exact_parsed = ParsedDocument.model_validate_json(
                        _canonical_bytes(payload),
                        strict=True,
                    )
                    if (
                        exact_parsed != parsed_result
                        or exact_parsed.parser_id != cache_key.parser_id
                        or exact_parsed.parser_version != cache_key.parser_version
                        or str(exact_parsed.canonical_url)
                        != str(raw_document.final_url)
                    ):
                        raise ProviderError(
                            code="INVALID_RESPONSE",
                            provider=self.parser.parser_id,
                            operation="parse",
                            public_message="parser result identity is invalid",
                            retryable=False,
                        )
                    return payload

                artifact_id, payload_value = self._persist_provider_result(
                    context,
                    payload_factory=encode_parsed_result,
                    media_type="application/vnd.deepresearch.parsed-document+json",
                )
                payload = cast("dict[str, object]", payload_value)
                provider_call, provider_receipt_id = self._last_provider_audit(context)
                self._publish_completion_cache(
                    key,
                    self._cache_entry(
                        key=key,
                        artifact_id=artifact_id,
                        operation_id=operation_id,
                        usage=usage,
                        context=context,
                        provider_call=provider_call,
                        provider_call_receipt_id=provider_receipt_id,
                    ),
                    context=context,
                    operation_id=operation_id,
                    expected_call_fields=call_fields,
                    budget_node="Tool",
                    validate_result=load_result,
                )
                parsed_documents.append(payload)
                parsed_artifact_ids.append(artifact_id)
        work = self.artifact_store.put_bytes(
            _canonical_bytes(
                {
                    "documents": parsed_documents,
                    "kind": "parsed",
                    "parsed_artifact_ids": parsed_artifact_ids,
                    "raw_documents": raw_work["documents"],
                }
            ),
            media_type="application/vnd.deepresearch.baseline-work+json",
        )
        return {
            "baseline_work_artifact_ids": (
                *state["baseline_work_artifact_ids"],
                work.artifact_id,
            )
        }

    async def store_evidence(self, state: BaselineState) -> StateUpdate:
        from deepresearch.providers import ParsedDocument, RawDocument

        context = self._runtime()
        plan = self._plan_from_state(state)
        active_id = state["active_subquestion_id"]
        subquestion = next(item for item in plan.subquestions if item.id == active_id)
        work = self._latest_work(state, "parsed")
        checkpoint_source_ids = set(state["source_ids"])
        checkpoint_evidence_ids = set(state["evidence_ids"])
        source_ids = set(checkpoint_source_ids)
        evidence_ids = set(checkpoint_evidence_ids)
        introduced_sources: dict[
            str, tuple[SourceDocument, ParsedDocument, str]
        ] = {}
        introduced_evidence: dict[str, EvidenceSpan] = {}
        pairs = zip(
            cast("list[object]", work["raw_documents"]),
            cast("list[object]", work["documents"]),
            cast("list[object]", work["parsed_artifact_ids"]),
            strict=True,
        )
        audit_receipt_ids: list[str] = []
        source_type = cast(
            "SourceType",
            next(
                iter(sorted(subquestion.evidence_requirements.allowed_source_types)),
                "unknown",
            ),
        )
        for raw_value, parsed_value, parsed_artifact_id_value in pairs:
            raw = RawDocument.model_validate(raw_value)
            parsed = ParsedDocument.model_validate(parsed_value)
            if type(parsed_artifact_id_value) is not str:
                raise ArtifactIntegrityError("parsed artifact association is corrupt")
            parsed_artifact_id = parsed_artifact_id_value
            if not self.artifact_store.exists(parsed_artifact_id):
                raise ArtifactIntegrityError("parsed artifact association is corrupt")
            source_id = f"S-{_hash_json({'url': str(parsed.canonical_url)})}"
            family = (
                urlsplit(str(parsed.canonical_url)).hostname or str(parsed.canonical_url)
            ).casefold()
            source = SourceDocument(
                source_id=source_id,
                canonical_url=parsed.canonical_url,
                title=parsed.title,
                authors=parsed.authors,
                published_at=parsed.published_at,
                retrieved_at=raw.retrieved_at,
                content_hash=hashlib.sha256(raw.body_bytes).hexdigest(),
                parsed_content_hash=parsed.parsed_content_hash,
                source_type=source_type,
                source_family_id=family,
                parser_version=parsed.parser_version,
            )
            stored_source = self.evidence_store.put_source(
                source,
                normalized_text=parsed.normalized_text,
            )
            stored_parsed = self.evidence_store.put_parsed_document(source_id, parsed)
            if stored_source != source or stored_parsed != parsed:
                raise ArtifactIntegrityError("stored source content is inconsistent")
            prior_source = introduced_sources.get(source_id)
            first_source_introduction = (
                source_id not in checkpoint_source_ids and prior_source is None
            )
            if source_id in checkpoint_source_ids:
                if self.evidence_store.get_source(source_id) != source:
                    raise ArtifactIntegrityError("stored source content is inconsistent")
            elif prior_source is not None and prior_source != (
                source,
                parsed,
                parsed_artifact_id,
            ):
                raise ArtifactIntegrityError("repeated source content is inconsistent")
            elif first_source_introduction:
                introduced_sources[source_id] = (
                    source,
                    parsed,
                    parsed_artifact_id,
                )
            source_ids.add(source_id)
            if first_source_introduction:
                parsed_record = ParsedArtifactRecord(
                    source_id=source_id,
                    raw_content_hash=hashlib.sha256(raw.body_bytes).hexdigest(),
                    parsed_content_hash=parsed.parsed_content_hash,
                    parser_id=self.parser.parser_id,
                    parser_version=parsed.parser_version,
                    normalization_version=self.normalization_version,
                    artifact_id=parsed_artifact_id,
                )
                audit_receipt_ids.append(
                    _put_audit_receipt(
                        self.artifact_store,
                        kind="parsed-artifact",
                        run_id=state["run_id"],
                        thread_id=state["thread_id"],
                        receipt_key=f"parsed:{source_id}:{parsed_artifact_id}",
                        payload={"record": parsed_record.model_dump(mode="json")},
                    )
                )
            for block in parsed.blocks:
                if not block.text:
                    continue
                locator = block.locator.model_copy(
                    update={"start_char": 0, "end_char": len(block.text)}
                )
                excerpt_hash = hashlib.sha256(block.text.encode("utf-8")).hexdigest()
                information_need_ids = tuple(
                    sorted({need.need_id for need in subquestion.information_needs})
                )
                evidence_id = f"E-{_hash_json({'source_id': source_id, 'locator': locator.model_dump(mode='json'), 'excerpt_hash': excerpt_hash, 'language': state['request'].report_language, 'information_need_ids': list(information_need_ids)})}"
                evidence = EvidenceSpan(
                    evidence_id=evidence_id,
                    source_id=source_id,
                    locator=locator,
                    excerpt=block.text,
                    excerpt_hash=excerpt_hash,
                    language=state["request"].report_language,
                    information_need_ids=information_need_ids,
                )
                stored_evidence = self.evidence_store.put_evidence(evidence)
                if stored_evidence != evidence:
                    raise ArtifactIntegrityError("stored evidence content is inconsistent")
                prior_evidence = introduced_evidence.get(evidence_id)
                first_evidence_introduction = (
                    evidence_id not in checkpoint_evidence_ids
                    and prior_evidence is None
                )
                if evidence_id in checkpoint_evidence_ids:
                    if self.evidence_store.get_evidence(evidence_id) != evidence:
                        raise ArtifactIntegrityError(
                            "stored evidence content is inconsistent"
                        )
                elif prior_evidence is not None and prior_evidence != evidence:
                    raise ArtifactIntegrityError(
                        "repeated evidence content is inconsistent"
                    )
                elif first_evidence_introduction:
                    introduced_evidence[evidence_id] = evidence
                evidence_ids.add(evidence_id)
                evidence_ref = self.artifact_store.put_bytes(
                    _canonical_bytes(evidence.model_dump(mode="json")),
                    media_type="application/vnd.deepresearch.evidence-span+json",
                )
                if first_evidence_introduction:
                    evidence_record = EvidenceHashRecord(
                        evidence_id=evidence_id,
                        source_id=source_id,
                        locator_sha256=_hash_json(locator.model_dump(mode="json")),
                        excerpt_hash=evidence.excerpt_hash,
                        artifact_id=evidence_ref.artifact_id,
                    )
                    audit_receipt_ids.append(
                        _put_audit_receipt(
                            self.artifact_store,
                            kind="evidence-hash",
                            run_id=state["run_id"],
                            thread_id=state["thread_id"],
                            receipt_key=f"evidence:{evidence_id}",
                            payload={"record": evidence_record.model_dump(mode="json")},
                        )
                    )
                    self._audit_result_artifact(context, evidence_ref.artifact_id)
        for receipt_id in audit_receipt_ids:
            self._audit_result_artifact(context, receipt_id)
        _audit_buffer(context).child_receipt_ids.extend(audit_receipt_ids)
        return {
            "baseline_work_artifact_ids": tuple(
                dict.fromkeys(
                    (*state["baseline_work_artifact_ids"], *audit_receipt_ids)
                )
            ),
            "source_ids": tuple(sorted(source_ids)),
            "evidence_ids": tuple(sorted(evidence_ids)),
        }

    async def rank_evidence(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        plan = self._plan_from_state(state)
        evidence = tuple(self.evidence_store.get_evidence(item) for item in state["evidence_ids"])
        sources = tuple(self.evidence_store.get_source(item) for item in state["source_ids"])
        scores: dict[str, Sequence[RerankScore]] = {}
        for subquestion in plan.subquestions:
            for need in subquestion.information_needs:
                ranked = await self.ranker.score(
                    need.text,
                    evidence,
                    deadline=_effective_deadline(context),
                    cancellation_token=context.cancellation_token,
                )
                scores[need.need_id] = ranked
        selected, ledger = rank_baseline_coverage(
            plan,
            evidence,
            sources,
            scores,
            previous_ledger=state["coverage_ledger"],
        )
        active_id = state["active_subquestion_id"]
        active_ledger = next(item for item in ledger if item.subquestion_id == active_id)
        pending = tuple(item for item in state["pending_subquestion_ids"] if item != active_id)
        return {
            "active_subquestion_id": None,
            "coverage_ledger": ledger,
            "pending_subquestion_ids": pending,
            "recent_marginal_gains": (
                *state["recent_marginal_gains"],
                active_ledger.last_marginal_gain,
            ),
            "selected_evidence_ids": selected,
        }

    async def draft_report(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        if state["stop_reason"] == "BUDGET_EXHAUSTED":
            if not state["selected_evidence_ids"]:
                raise WorkflowInvariantError(code="REPORT_MISSING")
            local_draft = (
                "Available evidence "
                + " ".join(f"[{item}]" for item in state["selected_evidence_ids"])
                + "."
            )
            ref = self.artifact_store.put_bytes(
                local_draft.encode("utf-8"),
                media_type="text/markdown; charset=utf-8",
            )
            return {"draft_artifact_id": ref.artifact_id}
        candidate_model = self.writer.model
        if not isinstance(candidate_model, ModelProvider):
            raise WorkflowInvariantError(code="WRITER_MODEL_MISSING")
        model = candidate_model
        prompt = self.writer.render_prompt(
            selected_evidence_ids=state["selected_evidence_ids"],
            user_strings=(state["request"].question,),
        )
        system = "Write one evidence-grounded Markdown report with inline evidence IDs."
        request = ModelRequest(
            model_id=_model_identity(model)[1],
            messages=(
                ModelMessage(role="system", content=system),
                ModelMessage(role="user", content=prompt),
            ),
            temperature=Decimal(0),
            seed=context.config.seed,
            max_output_tokens=4_000,
            prompt_version=self.writer_prompt_version,
            system_prompt_hash=hashlib.sha256(system.encode()).hexdigest(),
            tool_schema_hash=hashlib.sha256(b"[]").hexdigest(),
            output_schema_hash=hashlib.sha256(b"").hexdigest(),
        )
        result = await model.complete(
            request,
            deadline=_effective_deadline(context),
            cancellation_token=context.cancellation_token,
        )
        ref = self.artifact_store.put_bytes(
            result.output.encode("utf-8"),
            media_type="text/markdown; charset=utf-8",
        )
        return {"draft_artifact_id": ref.artifact_id}

    async def finalize_citations(self, state: BaselineState) -> StateUpdate:
        draft_id = state["draft_artifact_id"]
        if draft_id is None:
            raise WorkflowInvariantError(code="REPORT_MISSING")
        draft = self.artifact_store.get_bytes(draft_id).decode("utf-8")
        plan = self._plan_from_state(state)
        covered = {
            item.subquestion_id for item in state["coverage_ledger"] if item.coverage_score >= 0.85
        }
        uncovered = tuple(
            need.text
            for subquestion in plan.subquestions
            if subquestion.id not in covered
            for need in subquestion.information_needs
        )
        report = self.writer.finalize_report(
            draft,
            selected_evidence_ids=state["selected_evidence_ids"],
            is_partial=state["is_partial"],
            stop_reason=state["stop_reason"] if state["is_partial"] else None,
            uncovered_information_needs=uncovered,
        )
        ref = self.artifact_store.put_bytes(
            report.encode("utf-8"), media_type="text/markdown; charset=utf-8"
        )
        return {"report_artifact_id": ref.artifact_id}

    async def persist_results(self, state: BaselineState) -> StateUpdate:
        context = self._runtime()
        evidence = tuple(
            self.evidence_store.get_evidence(item)
            for item in state["evidence_ids"]
        )
        sources = tuple(
            self.evidence_store.get_source(item)
            for item in state["source_ids"]
        )
        graph_artifact_id, graph_evidence, graph_sources = (
            _put_verified_evidence_graph(
                self.artifact_store,
                evidence=evidence,
                sources=sources,
            )
        )
        composition = self._audit_composition
        indexed_receipts: dict[str, _AuditReceiptEnvelope] = {}
        work_kinds = {"search", "raw", "parsed"}
        for artifact_id in state["baseline_work_artifact_ids"]:
            receipt = _load_audit_receipt(self.artifact_store, artifact_id)
            if receipt is not None:
                if (
                    receipt.run_id != state["run_id"]
                    or receipt.thread_id != state["thread_id"]
                    or receipt.receipt_key in {
                        item.receipt_key for item in indexed_receipts.values()
                    }
                ):
                    raise ArtifactIntegrityError("indexed audit receipt identity conflicts")
                indexed_receipts[artifact_id] = receipt
                continue
            try:
                work: object = json.loads(self.artifact_store.get_bytes(artifact_id))
            except (FileNotFoundError, TypeError, ValueError):
                raise ArtifactIntegrityError("indexed baseline work artifact is corrupt") from None
            if not isinstance(work, dict):
                raise ArtifactIntegrityError("indexed baseline work artifact is unknown")
            work_mapping = cast("dict[object, object]", work)
            if work_mapping.get("kind") not in work_kinds:
                raise ArtifactIntegrityError("indexed baseline work artifact is unknown")

        header = composition.load_run_header(
            state["baseline_work_artifact_ids"],
            run_id=state["run_id"],
            thread_id=state["thread_id"],
            config=context.config,
        )
        if (
            header["config_sha256"] != state["config_sha256"]
            or header["request_sha256"]
            != _hash_json(state["request"].model_dump(mode="json"))
        ):
            raise ArtifactIntegrityError("run audit identity conflicts with state")
        terminal_receipts = tuple(
            (artifact_id, item)
            for artifact_id, item in indexed_receipts.items()
            if item.kind == "terminal"
        )
        if len(terminal_receipts) != 1:
            raise ArtifactIntegrityError("terminal audit receipt is missing or ambiguous")
        terminal_receipt_id, terminal = terminal_receipts[0]
        try:
            started_at = datetime.fromisoformat(cast("str", header["started_at"]))
            recorded_finished_at = datetime.fromisoformat(
                cast("str", terminal.payload["finished_at"])
            )
            terminal_event_seq = terminal.payload["terminal_event_seq"]
            terminal_elapsed = terminal.payload["elapsed_wall_seconds"]
            terminal_error_code = terminal.payload["error_code"]
        except (KeyError, TypeError, ValueError):
            raise ArtifactIntegrityError("run audit envelope is corrupt") from None
        if (
            started_at.tzinfo is None
            or started_at.utcoffset() is None
            or recorded_finished_at.tzinfo is None
            or recorded_finished_at.utcoffset() is None
            or type(terminal_event_seq) is not int
            or terminal_event_seq != state["next_event_seq"] - 1
            or type(terminal_elapsed) is not float
            or not math.isfinite(terminal_elapsed)
            or terminal_elapsed < 0.0
            or terminal_elapsed != state["elapsed_wall_seconds"]
            or not (
                terminal_error_code is None or type(terminal_error_code) is str
            )
        ):
            raise ArtifactIntegrityError("run audit envelope is inconsistent")
        finished_at = recorded_finished_at

        sink = context.emit
        if not isinstance(sink, DurableRunEventSink):
            raise WorkflowInvariantError(code="INVALID_EVENT_SINK")
        events: list[RunEvent] = []
        node_executions: list[NodeExecutionRecord] = []
        provider_calls: list[ProviderCallRecord] = []
        consumed_node_receipts: set[str] = set()
        consumed_child_receipts: set[str] = set()
        owned_typed_receipt_ids: list[str] = []
        for seq in range(1, state["next_event_seq"]):
            event = await sink.get_event(run_id=state["run_id"], seq=seq)
            if event is None:
                raise ArtifactIntegrityError("durable event sequence has a gap")
            exact_event = _strict_run_event(event)
            if exact_event != event or event.seq != seq:
                raise ArtifactIntegrityError("durable event is corrupt")
            candidates = [
                artifact_id
                for artifact_id in event.artifact_ids
                if (
                    (receipt := indexed_receipts.get(artifact_id)) is not None
                    and receipt.kind == "node-execution"
                )
            ]
            if len(candidates) != 1:
                raise ArtifactIntegrityError("durable event completion receipt is ambiguous")
            node_receipt_id = candidates[0]
            if node_receipt_id in consumed_node_receipts:
                raise ArtifactIntegrityError("node completion receipt was reused")
            consumed_node_receipts.add(node_receipt_id)
            receipt = indexed_receipts[node_receipt_id]
            try:
                execution = _node_execution_from_receipt_payload(
                    receipt.payload.get("record")
                )
            except ArtifactIntegrityError:
                raise ArtifactIntegrityError(
                    "node completion receipt is corrupt"
                ) from None
            base_event = _run_event_from_receipt_payload(receipt.payload.get("event"))
            expected_event = base_event.model_copy(
                update={
                    "artifact_ids": tuple(
                        dict.fromkeys((*base_event.artifact_ids, node_receipt_id))
                    )
                }
            )
            if (
                not _run_events_match_exact(event, expected_event)
                or execution.node != event.node
            ):
                raise ArtifactIntegrityError("event and node completion receipt disagree")
            provider_ids_value = receipt.payload.get("provider_receipt_ids")
            child_ids_value = receipt.payload.get("child_receipt_ids")
            if (
                type(provider_ids_value) is not list
                or type(child_ids_value) is not list
            ):
                raise ArtifactIntegrityError("node child receipt ownership is corrupt")
            provider_id_items = cast("list[object]", provider_ids_value)
            child_id_items = cast("list[object]", child_ids_value)
            if (
                any(type(item) is not str for item in provider_id_items)
                or any(type(item) is not str for item in child_id_items)
            ):
                raise ArtifactIntegrityError("node child receipt ownership is corrupt")
            provider_id_list = cast("list[str]", provider_id_items)
            child_id_list = cast("list[str]", child_ids_value)
            if len(set(child_id_list)) != len(child_id_list):
                raise ArtifactIntegrityError("node child receipt ownership is corrupt")
            child_receipts: list[_AuditReceiptEnvelope] = []
            for child_receipt_id in child_id_list:
                child = indexed_receipts.get(child_receipt_id)
                if (
                    child is None
                    or child.kind in {"run-header", "node-execution"}
                    or child.run_id != state["run_id"]
                    or child.thread_id != state["thread_id"]
                    or child_receipt_id in consumed_child_receipts
                ):
                    raise ArtifactIntegrityError(
                        "node child receipt ownership is corrupt"
                    )
                if child.kind in {"parsed-artifact", "evidence-hash"} and (
                    child_receipt_id not in execution.output_artifact_ids
                    or child_receipt_id not in base_event.artifact_ids
                ):
                    raise ArtifactIntegrityError(
                        "typed receipt is disconnected from its node output"
                    )
                _validate_child_receipt_owner(
                    child_id=child_receipt_id,
                    child=child,
                    execution=execution,
                    base_event=base_event,
                    durable_event=event,
                    state=state,
                    terminal_receipt_id=terminal_receipt_id,
                    terminal_event_seq=terminal_event_seq,
                    terminal_finished_at=recorded_finished_at,
                    terminal_error_code=terminal_error_code,
                )
                consumed_child_receipts.add(child_receipt_id)
                child_receipts.append(child)
                if child.kind in {"parsed-artifact", "evidence-hash"}:
                    owned_typed_receipt_ids.append(child_receipt_id)
            if provider_id_list != [
                child_id
                for child_id, child in zip(
                    child_id_list,
                    child_receipts,
                    strict=True,
                )
                if child.kind == "provider-call"
            ]:
                raise ArtifactIntegrityError("provider receipt ownership is corrupt")
            calls, _facts = _provider_calls_from_receipt(
                composition,
                run_id=state["run_id"],
                thread_id=state["thread_id"],
                owning_node=execution.node,
                owning_output_artifact_ids=execution.output_artifact_ids,
                receipt_ids=provider_id_list,
            )
            if any(call.node != execution.node for call in calls):
                raise ArtifactIntegrityError("provider call and node receipt disagree")
            events.append(event)
            node_executions.append(execution)
            provider_calls.extend(calls)
        _validate_complete_receipt_ownership(
            indexed_receipts=indexed_receipts,
            consumed_node_receipts=consumed_node_receipts,
            consumed_child_receipts=consumed_child_receipts,
        )

        parsed_records_value, evidence_records_value = _validate_typed_receipt_closure(
            artifact_store=self.artifact_store,
            evidence_store=self.evidence_store,
            receipts=tuple(
                indexed_receipts[receipt_id]
                for receipt_id in owned_typed_receipt_ids
            ),
            state=state,
            normalization_version=self.normalization_version,
        )
        parsed_by_source = {item.source_id: item for item in parsed_records_value}
        evidence_by_id = {item.evidence_id: item for item in evidence_records_value}
        try:
            parsed_records = [parsed_by_source[item] for item in state["source_ids"]]
            evidence_records = [evidence_by_id[item] for item in state["evidence_ids"]]
        except KeyError:
            raise ArtifactIntegrityError("typed audit record order is incomplete") from None
        if (
            len(parsed_by_source) != len(parsed_records_value)
            or len(evidence_by_id) != len(evidence_records_value)
            or tuple(item.source_id for item in graph_sources) != state["source_ids"]
            or tuple(item.evidence_id for item in graph_evidence)
            != state["evidence_ids"]
            or tuple(item.source_id for item in parsed_records) != state["source_ids"]
            or tuple(item.evidence_id for item in evidence_records)
            != state["evidence_ids"]
        ):
            raise ArtifactIntegrityError("evidence graph and typed audit closure disagree")

        executions_by_node: dict[str, list[NodeExecutionRecord]] = {}
        for execution in node_executions:
            executions_by_node.setdefault(execution.node, []).append(execution)
        usage_by_node: dict[str, ResourceUsage] = {}
        for node, executions in executions_by_node.items():
            node_calls = tuple(call for call in provider_calls if call.node == node)
            node_cost = (
                None
                if composition.pricing_status == "unknown"
                else sum(
                    (
                        call.estimated_cost_usd or Decimal(0)
                        for call in node_calls
                        if call.operation == "model" and not call.cache_hit
                    ),
                    Decimal(0),
                )
            )
            usage_by_node[node] = _manifest_usage(
                tuple(item.usage for item in executions),
                wall_seconds=max(item.usage.wall_seconds for item in executions),
                cost_usd=node_cost,
            )
        run_cost = (
            None
            if composition.pricing_status == "unknown"
            else sum(
                (
                    call.estimated_cost_usd or Decimal(0)
                    for call in provider_calls
                    if call.operation == "model" and not call.cache_hit
                ),
                Decimal(0),
            )
        )
        run_usage = _manifest_usage(
            tuple(usage_by_node.values()),
            wall_seconds=terminal_elapsed,
            cost_usd=run_cost,
        )
        artifacts = tuple(
            dict.fromkeys(
                (
                    *state["baseline_work_artifact_ids"],
                    *(
                        (state["plan_artifact_id"],)
                        if state["plan_artifact_id"] is not None
                        else ()
                    ),
                    *(
                        (state["draft_artifact_id"],)
                        if state["draft_artifact_id"] is not None
                        else ()
                    ),
                    *(
                        (state["report_artifact_id"],)
                        if state["report_artifact_id"] is not None
                        else ()
                    ),
                    graph_artifact_id,
                    *(item for record in node_executions for item in record.input_artifact_ids),
                    *(item for record in node_executions for item in record.output_artifact_ids),
                    *(item for event in events for item in event.artifact_ids),
                    *(item.artifact_id for item in parsed_records),
                    *(item.artifact_id for item in evidence_records),
                )
            )
        )
        if any(not self.artifact_store.exists(item) for item in artifacts):
            raise ArtifactIntegrityError("manifest references a missing artifact")
        source_snapshot_ids = tuple(
            dict.fromkeys(
                call.snapshot_id
                for call in provider_calls
                if call.operation in {"search", "fetch"} and call.snapshot_id is not None
            )
        )
        parser_versions = {
            item.parser_id: item.parser_version for item in parsed_records
        }
        model_ids = tuple(
            dict.fromkeys(
                call.model_id
                for call in provider_calls
                if call.operation == "model" and call.model_id is not None
            )
        )
        manifest = RunManifest.create(
            {
                "schema_version": "run-manifest-v1",
                "run_id": state["run_id"],
                "thread_id": state["thread_id"],
                "code_commit": header["code_commit"],
                "dependency_lock_sha256": header["dependency_lock_sha256"],
                "request_sha256": header["request_sha256"],
                "config_sha256": header["config_sha256"],
                "workflow_id": header["workflow_id"],
                "graph_version": header["graph_version"],
                "planner_id": context.config.planner_id,
                "provider_profiles": (
                    ProviderProfileRecord(
                        profile_id=cast("str", header["provider_profile_id"]),
                        execution_mode=cast("Any", header["execution_mode"]),
                        provider_ids=tuple(cast("list[str]", header["provider_ids"])),
                        configuration_sha256=cast(
                            "str", header["provider_profile_configuration_sha256"]
                        ),
                    ),
                ),
                "model_ids": model_ids,
                "prompt_versions": dict(context.config.prompt_versions),
                "parser_versions": parser_versions,
                "ranker_id": context.config.ranker_id,
                "ranker_weights_version": context.config.ranker_weights_version,
                "budget": context.config.budget,
                "usage": run_usage,
                "usage_by_node": usage_by_node,
                "pricing_status": composition.pricing_status,
                "pricing_snapshots": composition.pricing_snapshots,
                "provider_calls": tuple(provider_calls),
                "node_executions": tuple(node_executions),
                "parsed_artifacts": tuple(parsed_records),
                "evidence_hashes": tuple(evidence_records),
                "source_snapshot_ids": source_snapshot_ids,
                "artifact_ids": artifacts,
                "run_event_count": len(events),
                "run_events_sha256": hashlib.sha256(
                    _canonical_bytes([item.model_dump(mode="json") for item in events])
                ).hexdigest(),
                "seed": context.config.seed,
                "seed_supported": composition.seed_supported,
                "cache_hit_count": sum(item.cache_hit for item in provider_calls),
                "stop_reason": state["stop_reason"],
                "is_partial": state["is_partial"],
                "failure_codes": tuple(
                    dict.fromkeys(
                        item.error_code
                        for item in node_executions
                        if item.error_code is not None
                    )
                ),
                "replay_parent": composition.replay_parent,
                "started_at": started_at,
                "finished_at": finished_at,
            }
        )
        manifest_bytes = _canonical_bytes(manifest.model_dump(mode="json"))
        manifest_ref = self.artifact_store.put_bytes(
            manifest_bytes,
            media_type="application/vnd.deepresearch.run-manifest+json",
        )
        try:
            stored_manifest_bytes = self.artifact_store.get_bytes(
                manifest_ref.artifact_id
            )
            stored_manifest = RunManifest.model_validate_json(
                stored_manifest_bytes,
                strict=True,
            )
        except (FileNotFoundError, TypeError, ValueError):
            raise ArtifactIntegrityError("persisted manifest is corrupt") from None
        if stored_manifest_bytes != manifest_bytes or stored_manifest != manifest:
            raise ArtifactIntegrityError("persisted manifest bytes conflict")
        return {
            "evidence_graph_artifact_id": graph_artifact_id,
            "manifest_artifact_id": manifest_ref.artifact_id,
        }


def baseline_is_sufficient(
    plan: ResearchPlan,
    ledger: Sequence[CoverageLedgerEntry],
    *,
    high_priority_unresolved_conflict_ids: Sequence[str] = (),
) -> bool:
    if high_priority_unresolved_conflict_ids:
        return False
    by_id: dict[str, CoverageLedgerEntry] = {}
    for entry in ledger:
        if entry.subquestion_id in by_id:
            return False
        by_id[entry.subquestion_id] = entry
    plan_ids = {item.id for item in plan.subquestions}
    if set(by_id) != plan_ids:
        return False
    total_importance = sum(item.importance for item in plan.subquestions)
    if total_importance <= 0.0:
        return False
    weighted_coverage = 0.0
    for subquestion in plan.subquestions:
        entry = by_id[subquestion.id]
        if entry.coverage_score < 0.85:
            return False
        if (
            entry.independent_source_count
            < subquestion.evidence_requirements.min_independent_sources
        ):
            return False
        weighted_coverage += subquestion.importance * entry.coverage_score
    return weighted_coverage / total_importance >= 0.80


def rank_baseline_coverage(
    plan: ResearchPlan,
    evidence_spans: Sequence[EvidenceSpan],
    sources: Sequence[SourceDocument],
    scores_by_need: Mapping[str, Sequence[RerankScore]],
    *,
    previous_ledger: Sequence[CoverageLedgerEntry],
) -> tuple[tuple[str, ...], tuple[CoverageLedgerEntry, ...]]:
    evidence_by_id = {item.evidence_id: item for item in evidence_spans}
    family_by_source = {item.source_id: item.source_family_id for item in sources}
    previous_by_subquestion = {item.subquestion_id: item for item in previous_ledger}
    globally_selected: set[str] = set()
    ledger: list[CoverageLedgerEntry] = []
    for subquestion in plan.subquestions:
        selected_for_subquestion: set[str] = set()
        need_coverages: list[tuple[float, float]] = []
        for need in subquestion.information_needs:
            candidates = sorted(
                (
                    score
                    for score in scores_by_need.get(need.need_id, ())
                    if score.evidence_id in evidence_by_id
                    and need.need_id in evidence_by_id[score.evidence_id].information_need_ids
                ),
                key=lambda score: (-score.total, score.evidence_id),
            )
            selected_for_need: list[RerankScore] = []
            seen_families: set[str] = set()
            for score in candidates:
                evidence = evidence_by_id[score.evidence_id]
                family = family_by_source.get(evidence.source_id)
                if family is None or family in seen_families:
                    continue
                seen_families.add(family)
                selected_for_need.append(score)
                if (
                    len(selected_for_need)
                    >= subquestion.evidence_requirements.min_independent_sources
                ):
                    break
            selected_for_subquestion.update(item.evidence_id for item in selected_for_need)
            coverage = max((item.total for item in selected_for_need), default=0.0)
            need_coverages.append((need.importance, coverage))
        total_need_importance = sum(weight for weight, _ in need_coverages)
        coverage_score = (
            0.0
            if total_need_importance <= 0.0
            else sum(weight * score for weight, score in need_coverages) / total_need_importance
        )
        families = {
            family_by_source[evidence_by_id[evidence_id].source_id]
            for evidence_id in selected_for_subquestion
            if evidence_by_id[evidence_id].source_id in family_by_source
        }
        previous = previous_by_subquestion.get(subquestion.id)
        previous_score = 0.0 if previous is None else previous.coverage_score
        selected_ids = tuple(sorted(selected_for_subquestion))
        globally_selected.update(selected_ids)
        ledger.append(
            CoverageLedgerEntry(
                subquestion_id=subquestion.id,
                coverage_score=coverage_score,
                independent_source_count=len(families),
                unresolved_conflict_ids=(
                    () if previous is None else previous.unresolved_conflict_ids
                ),
                uncertainty_score=1.0 - coverage_score,
                last_marginal_gain=max(0.0, coverage_score - previous_score),
                evidence_ids=selected_ids,
                attempt_count=1 if previous is None else previous.attempt_count + 1,
                last_decision_code="R1_DISTINCT_SOURCE_FAMILIES",
            )
        )
    return tuple(sorted(globally_selected)), tuple(ledger)


def decide_baseline_stop(
    state: BaselineState,
    plan: ResearchPlan,
) -> StopReason | None:
    restored = validate_baseline_state(state)
    if baseline_is_sufficient(
        plan,
        restored["coverage_ledger"],
        high_priority_unresolved_conflict_ids=restored["high_priority_unresolved_conflict_ids"],
    ):
        return "SUFFICIENT"
    if restored["budget_snapshot"].exhausted:
        return "BUDGET_EXHAUSTED"
    gains = restored["recent_marginal_gains"]
    if len(gains) >= 2 and gains[-2] < 0.05 and gains[-1] < 0.05:
        return "PLATEAU"
    blocked = restored["blocked_needs"]
    if blocked and all(
        item["required_source_unavailable"]
        and item["alternative_strategies_exhausted"]
        and item["retry_count"] >= item["max_retries"]
        for item in blocked
    ):
        return "BLOCKED"
    if restored["pending_subquestion_ids"]:
        return None
    raise WorkflowInvariantError(code="NO_LEGAL_CONTINUATION")


def route_after_decide(state: BaselineState) -> BaselineRoute:
    restored = validate_baseline_state(state)
    if restored["error_code"] is not None:
        return "PersistResults"
    if restored["stop_reason"] is not None:
        return "DraftReport"
    if restored["pending_subquestion_ids"]:
        return "Search"
    raise WorkflowInvariantError(code="NO_LEGAL_CONTINUATION")


def _usage_delta(before: BudgetSnapshot, after: BudgetSnapshot) -> ResourceUsage:
    before_input = sum(item.input_tokens for item in before.used_by_node.values())
    after_input = sum(item.input_tokens for item in after.used_by_node.values())
    before_output = sum(item.output_tokens for item in before.used_by_node.values())
    after_output = sum(item.output_tokens for item in after.used_by_node.values())
    before_reasoning = sum(item.reasoning_tokens for item in before.used_by_node.values())
    after_reasoning = sum(item.reasoning_tokens for item in after.used_by_node.values())
    before_cached = sum(item.cached_tokens for item in before.used_by_node.values())
    after_cached = sum(item.cached_tokens for item in after.used_by_node.values())
    cost: Decimal | None
    if before.used_cost_usd is None or after.used_cost_usd is None:
        cost = None
    else:
        cost = after.used_cost_usd - before.used_cost_usd
    return ResourceUsage(
        input_tokens=max(0, after_input - before_input),
        output_tokens=max(0, after_output - before_output),
        reasoning_tokens=max(0, after_reasoning - before_reasoning),
        cached_tokens=max(0, after_cached - before_cached),
        total_tokens=max(0, after.used_tokens - before.used_tokens),
        search_calls=max(0, after.used_search_calls - before.used_search_calls),
        pages=max(0, after.used_pages - before.used_pages),
        retries=max(0, after.used_retries - before.used_retries),
        wall_seconds=max(0.0, after.used_wall_seconds - before.used_wall_seconds),
        cost_usd=cost,
    )


def _artifact_ids_in(value: object) -> tuple[str, ...]:
    found: list[str] = []

    def visit(item: object) -> None:
        if type(item) is str and item.startswith("sha256:"):
            found.append(item)
        elif type(item) is dict:
            for nested in cast("dict[object, object]", item).values():
                visit(nested)
        elif type(item) in (list, tuple):
            for nested in cast("list[object] | tuple[object, ...]", item):
                visit(nested)

    visit(value)
    return tuple(dict.fromkeys(found))


def _audit_usage(
    calls: Sequence[ProviderCallRecord],
    *,
    wall_seconds: float,
    pricing_status: Literal["estimated", "unknown"],
) -> ResourceUsage:
    cost: Decimal | None
    if pricing_status == "unknown":
        cost = None
    else:
        cost = sum(
            (
                call.estimated_cost_usd or Decimal(0)
                for call in calls
                if call.operation == "model" and not call.cache_hit
            ),
            Decimal(0),
        )
    return ResourceUsage(
        input_tokens=sum(item.usage.input_tokens for item in calls),
        output_tokens=sum(item.usage.output_tokens for item in calls),
        reasoning_tokens=sum(item.usage.reasoning_tokens for item in calls),
        cached_tokens=sum(item.usage.cached_tokens for item in calls),
        total_tokens=sum(item.usage.total_tokens for item in calls),
        search_calls=sum(item.usage.search_calls for item in calls),
        pages=sum(item.usage.pages for item in calls),
        retries=sum(item.usage.retries for item in calls),
        wall_seconds=wall_seconds,
        cost_usd=cost,
    )


def _manifest_usage(
    usages: Sequence[ResourceUsage],
    *,
    wall_seconds: float,
    cost_usd: Decimal | None,
) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        reasoning_tokens=sum(item.reasoning_tokens for item in usages),
        cached_tokens=sum(item.cached_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        search_calls=sum(item.search_calls for item in usages),
        pages=sum(item.pages for item in usages),
        retries=sum(item.retries for item in usages),
        wall_seconds=wall_seconds,
        cost_usd=cost_usd,
    )


def _validate_typed_receipt_closure(
    *,
    artifact_store: LocalArtifactStore,
    evidence_store: LocalEvidenceStore,
    receipts: Sequence[_AuditReceiptEnvelope],
    state: BaselineState,
    normalization_version: str,
) -> tuple[tuple[ParsedArtifactRecord, ...], tuple[EvidenceHashRecord, ...]]:
    associations: dict[str, tuple[RawDocument, ParsedDocument]] = {}
    for artifact_id in state["baseline_work_artifact_ids"]:
        if _load_audit_receipt(artifact_store, artifact_id) is not None:
            continue
        try:
            raw_work = artifact_store.get_bytes(artifact_id)
            work: object = json.loads(
                raw_work,
                object_pairs_hook=_unique_json_object,
                parse_constant=_reject_json_constant,
            )
            if _canonical_bytes(work) != raw_work:
                raise ValueError("non-canonical baseline work")
        except (FileNotFoundError, TypeError, ValueError):
            raise ArtifactIntegrityError("typed receipt work association is corrupt") from None
        if type(work) is not dict:
            continue
        work_mapping = cast("dict[str, object]", work)
        if work_mapping.get("kind") != "parsed":
            continue
        if set(work_mapping) != {
            "documents",
            "kind",
            "parsed_artifact_ids",
            "raw_documents",
        }:
            raise ArtifactIntegrityError("typed receipt work association is corrupt")
        documents = work_mapping["documents"]
        parsed_artifact_ids = work_mapping["parsed_artifact_ids"]
        raw_documents = work_mapping["raw_documents"]
        if (
            type(documents) is not list
            or type(parsed_artifact_ids) is not list
            or type(raw_documents) is not list
        ):
            raise ArtifactIntegrityError("typed receipt work association is corrupt")
        document_items = cast("list[object]", documents)
        parsed_id_items = cast("list[object]", parsed_artifact_ids)
        raw_document_items = cast("list[object]", raw_documents)
        if not (
            len(document_items) == len(parsed_id_items) == len(raw_document_items)
        ):
            raise ArtifactIntegrityError("typed receipt work association is corrupt")
        for raw_value, parsed_value, parsed_artifact_id in zip(
            raw_document_items,
            document_items,
            parsed_id_items,
            strict=True,
        ):
            try:
                if type(parsed_artifact_id) is not str:
                    raise ValueError("parsed artifact ID is invalid")
                raw_document = RawDocument.model_validate_json(
                    _canonical_bytes(raw_value),
                    strict=True,
                )
                parsed_document = ParsedDocument.model_validate_json(
                    _canonical_bytes(parsed_value),
                    strict=True,
                )
                stored_bytes = artifact_store.get_bytes(parsed_artifact_id)
                stored_document = ParsedDocument.model_validate_json(
                    stored_bytes,
                    strict=True,
                )
                if (
                    stored_bytes
                    != _canonical_bytes(stored_document.model_dump(mode="json"))
                    or stored_document != parsed_document
                    or parsed_artifact_id in associations
                ):
                    raise ValueError("parsed artifact association is inconsistent")
            except (FileNotFoundError, TypeError, ValueError):
                raise ArtifactIntegrityError(
                    "parsed artifact association is corrupt"
                ) from None
            associations[parsed_artifact_id] = (raw_document, parsed_document)

    parsed_records: list[ParsedArtifactRecord] = []
    evidence_records: list[EvidenceHashRecord] = []
    seen_sources: set[str] = set()
    seen_evidence: set[str] = set()
    parser_versions: dict[str, str] = {}
    for receipt in receipts:
        try:
            if receipt.kind == "parsed-artifact":
                record = ParsedArtifactRecord.model_validate_json(
                    _canonical_bytes(receipt.payload.get("record")),
                    strict=True,
                )
                association = associations.get(record.artifact_id)
                if association is None:
                    raise ValueError("parsed artifact has no work association")
                raw_document, parsed_document = association
                source = evidence_store.get_source(record.source_id)
                expected_source_id = (
                    f"S-{_hash_json({'url': str(parsed_document.canonical_url)})}"
                )
                existing_version = parser_versions.get(record.parser_id)
                if (
                    record.source_id in seen_sources
                    or record.source_id not in state["source_ids"]
                    or record.source_id != expected_source_id
                    or record.raw_content_hash
                    != hashlib.sha256(raw_document.body_bytes).hexdigest()
                    or record.parsed_content_hash
                    != parsed_document.parsed_content_hash
                    or record.parser_id != parsed_document.parser_id
                    or record.parser_version != parsed_document.parser_version
                    or record.normalization_version != normalization_version
                    or (existing_version is not None and existing_version != record.parser_version)
                    or source.source_id != record.source_id
                    or str(source.canonical_url) != str(parsed_document.canonical_url)
                    or source.parsed_content_hash != record.parsed_content_hash
                    or source.content_hash != record.raw_content_hash
                    or source.parser_version != record.parser_version
                ):
                    raise ValueError("parsed artifact record is inconsistent")
                seen_sources.add(record.source_id)
                parser_versions[record.parser_id] = record.parser_version
                parsed_records.append(record)
            elif receipt.kind == "evidence-hash":
                record = EvidenceHashRecord.model_validate_json(
                    _canonical_bytes(receipt.payload.get("record")),
                    strict=True,
                )
                stored_bytes = artifact_store.get_bytes(record.artifact_id)
                evidence = EvidenceSpan.model_validate_json(
                    stored_bytes,
                    strict=True,
                )
                if stored_bytes != _canonical_bytes(evidence.model_dump(mode="json")):
                    raise ValueError("evidence artifact is non-canonical")
                stored_evidence = evidence_store.get_evidence(record.evidence_id)
                if (
                    record.evidence_id in seen_evidence
                    or record.evidence_id not in state["evidence_ids"]
                    or record.source_id not in state["source_ids"]
                    or evidence != stored_evidence
                    or evidence.evidence_id != record.evidence_id
                    or evidence.source_id != record.source_id
                    or _hash_json(evidence.locator.model_dump(mode="json"))
                    != record.locator_sha256
                    or evidence.excerpt_hash != record.excerpt_hash
                ):
                    raise ValueError("evidence artifact record is inconsistent")
                seen_evidence.add(record.evidence_id)
                evidence_records.append(record)
        except (
            EvidenceIntegrityError,
            FileNotFoundError,
            TypeError,
            ValueError,
        ):
            raise ArtifactIntegrityError("typed audit record is corrupt") from None
    if seen_sources != set(state["source_ids"]) or seen_evidence != set(
        state["evidence_ids"]
    ):
        raise ArtifactIntegrityError("typed audit record coverage is incomplete")
    return tuple(parsed_records), tuple(evidence_records)


def _validate_child_receipt_owner(
    *,
    child_id: str,
    child: _AuditReceiptEnvelope,
    execution: NodeExecutionRecord,
    base_event: RunEvent,
    durable_event: RunEvent,
    state: BaselineState,
    terminal_receipt_id: str,
    terminal_event_seq: int,
    terminal_finished_at: datetime,
    terminal_error_code: str | None,
) -> None:
    if child.kind == "provider-call":
        return
    if child.kind == "parsed-artifact":
        try:
            record = ParsedArtifactRecord.model_validate_json(
                _canonical_bytes(child.payload.get("record")),
                strict=True,
            )
        except (TypeError, ValueError):
            raise ArtifactIntegrityError("parsed receipt owner is corrupt") from None
        if (
            execution.node != "StoreEvidence"
            or base_event.node != "StoreEvidence"
            or durable_event.node != "StoreEvidence"
            or child_id not in execution.output_artifact_ids
            or child_id not in base_event.artifact_ids
            or child.receipt_key
            != f"parsed:{record.source_id}:{record.artifact_id}"
        ):
            raise ArtifactIntegrityError("parsed receipt has an illegal owner")
        return
    if child.kind == "evidence-hash":
        try:
            record = EvidenceHashRecord.model_validate_json(
                _canonical_bytes(child.payload.get("record")),
                strict=True,
            )
        except (TypeError, ValueError):
            raise ArtifactIntegrityError("evidence receipt owner is corrupt") from None
        if (
            execution.node != "StoreEvidence"
            or base_event.node != "StoreEvidence"
            or durable_event.node != "StoreEvidence"
            or child_id not in execution.output_artifact_ids
            or child_id not in base_event.artifact_ids
            or child.receipt_key != f"evidence:{record.evidence_id}"
        ):
            raise ArtifactIntegrityError("evidence receipt has an illegal owner")
        return
    if child.kind != "terminal":
        raise ArtifactIntegrityError("node child receipt kind is not supported")
    if (
        child_id != terminal_receipt_id
        or child.receipt_key != "terminal"
        or durable_event.seq != terminal_event_seq
        or terminal_finished_at != execution.finished_at
        or terminal_finished_at != durable_event.timestamp
        or terminal_error_code != execution.error_code
        or terminal_error_code != durable_event.error_code
        or terminal_error_code != state["error_code"]
    ):
        raise ArtifactIntegrityError("terminal receipt has an illegal owner")
    if terminal_error_code is None:
        if (
            execution.node != "FinalizeCitations"
            or execution.status != "completed"
            or durable_event.kind != "node_completed"
            or durable_event.status != "running"
        ):
            raise ArtifactIntegrityError("successful terminal receipt owner is invalid")
        return
    if (
        execution.node != state["failed_node"]
        or execution.status not in {"failed", "cancelled"}
        or durable_event.kind != "node_failed"
        or durable_event.status not in {"failed", "cancelled"}
    ):
        raise ArtifactIntegrityError("failed terminal receipt owner is invalid")


def _validate_complete_receipt_ownership(
    *,
    indexed_receipts: Mapping[str, _AuditReceiptEnvelope],
    consumed_node_receipts: set[str],
    consumed_child_receipts: set[str],
) -> None:
    expected_node_ids = {
        artifact_id
        for artifact_id, receipt in indexed_receipts.items()
        if receipt.kind == "node-execution"
    }
    expected_child_ids = {
        artifact_id
        for artifact_id, receipt in indexed_receipts.items()
        if receipt.kind not in {"run-header", "node-execution"}
    }
    if (
        consumed_node_receipts != expected_node_ids
        or consumed_child_receipts != expected_child_ids
    ):
        raise ArtifactIntegrityError("audit receipt ownership is incomplete")


def _next_node_attempt(
    composition: _AuditComposition,
    state: BaselineState,
    node: str,
) -> int:
    attempts: list[int] = []
    for artifact_id in state["baseline_work_artifact_ids"]:
        receipt = _load_audit_receipt(composition.artifact_store, artifact_id)
        if receipt is None or receipt.kind != "node-execution":
            continue
        if receipt.run_id != state["run_id"] or receipt.thread_id != state["thread_id"]:
            raise ArtifactIntegrityError("node audit receipt belongs to another run")
        record = _node_execution_from_receipt_payload(receipt.payload.get("record"))
        if record.node == node:
            attempts.append(record.attempt)
    if attempts != list(range(1, len(attempts) + 1)):
        raise ArtifactIntegrityError("node audit attempts are corrupt")
    return len(attempts) + 1


def _budget_node_for_call(call: ProviderCallRecord) -> str:
    if call.operation == "model":
        return "Writer" if call.node == "DraftReport" else "Planner"
    if call.operation == "embed":
        return "Ranker"
    return "Tool"


def _restore_receipted_budget(
    context: BaselineRuntimeContext,
    facts: Sequence[_BudgetReplayFact],
) -> None:
    for fact in facts:
        reservation = context.budget_accountant.reserve(
            fact.estimate,
            node=fact.budget_node,
            idempotency_key=fact.operation_id,
        )
        if fact.decision == "release":
            context.budget_accountant.release(reservation)
        else:
            if fact.actual is None:
                raise ArtifactIntegrityError("settled budget receipt has no actual usage")
            context.budget_accountant.settle(
                reservation,
                actual=fact.actual,
                charge=fact.decision == "charge",
            )


def _provider_calls_from_receipt(
    composition: _AuditComposition,
    *,
    run_id: str,
    thread_id: str,
    owning_node: str,
    owning_output_artifact_ids: Sequence[str],
    receipt_ids: object,
) -> tuple[tuple[ProviderCallRecord, ...], tuple[_BudgetReplayFact, ...]]:
    if type(receipt_ids) is not list:
        raise ArtifactIntegrityError("node audit provider receipts are corrupt")
    calls: list[ProviderCallRecord] = []
    facts: list[_BudgetReplayFact] = []
    owning_outputs = set(owning_output_artifact_ids)
    for raw_receipt_id in cast("list[object]", receipt_ids):
        if type(raw_receipt_id) is not str:
            raise ArtifactIntegrityError("node audit provider receipts are corrupt")
        receipt_id = raw_receipt_id
        receipt = _load_audit_receipt(composition.artifact_store, receipt_id)
        if (
            receipt is None
            or receipt.kind != "provider-call"
            or receipt.run_id != run_id
            or receipt.thread_id != thread_id
        ):
            raise ArtifactIntegrityError("node audit provider receipt is corrupt")
        try:
            call = ProviderCallRecord.model_validate_json(
                _canonical_bytes(receipt.payload.get("record")),
                strict=True,
            )
            fact = _BudgetReplayFact.model_validate_json(
                _canonical_bytes(receipt.payload.get("budget_replay")),
                strict=True,
            )
            result_ids_value = receipt.payload.get("result_artifact_ids")
            if type(result_ids_value) is not list:
                raise ValueError("provider result artifact list is invalid")
            result_id_items = cast("list[object]", result_ids_value)
            if any(type(item) is not str for item in result_id_items):
                raise ValueError("provider result artifact list is invalid")
            result_ids = cast("list[str]", result_id_items)
            if len(set(result_ids)) != len(result_ids):
                raise ValueError("provider result artifact list is invalid")
        except (TypeError, ValueError):
            raise ArtifactIntegrityError(
                "node audit provider receipt is corrupt"
            ) from None
        if call.outcome_code == "SUCCESS":
            expected_decision = "charge"
            outcome_consistent = (
                not call.cache_hit and fact.actual is not None and len(result_ids) == 1
            )
        elif call.outcome_code == "CACHE_HIT":
            expected_decision = "observe"
            outcome_consistent = (
                call.cache_hit and fact.actual is not None and len(result_ids) == 1
            )
        else:
            expected_decision = "release" if fact.actual is None else "charge"
            outcome_consistent = not call.cache_hit and len(result_ids) <= 1
        if (
            call.node != owning_node
            or fact.budget_node != _budget_node_for_call(call)
            or receipt.receipt_key
            != f"call:{fact.operation_id}:{call.attempt}"
            or fact.decision != expected_decision
            or not outcome_consistent
            or (fact.actual is not None and fact.actual != call.usage)
            or any(result_id not in owning_outputs for result_id in result_ids)
        ):
            raise ArtifactIntegrityError("provider receipt closure is inconsistent")
        try:
            for result_id in result_ids:
                composition.artifact_store.get_bytes(result_id)
        except FileNotFoundError:
            raise ArtifactIntegrityError(
                "provider receipt result artifact is missing"
            ) from None
        calls.append(call)
        facts.append(fact)
    return tuple(calls), tuple(facts)


async def _recover_durable_node_event(
    *,
    composition: _AuditComposition,
    context: BaselineRuntimeContext,
    sink: DurableRunEventSink,
    restored: BaselineState,
    node: str,
    event: RunEvent,
) -> StateUpdate:
    exact_event = _strict_run_event(event)
    if (
        type(event) is not RunEvent
        or exact_event != event
        or event.run_id != restored["run_id"]
        or event.seq != restored["next_event_seq"]
        or event.node != node
    ):
        raise ArtifactIntegrityError("durable run event conflicts with graph state")
    input_sha256 = _hash_json(_state_payload(restored))
    matches: list[tuple[str, dict[str, object]]] = []
    for artifact_id in event.artifact_ids:
        receipt = _load_audit_receipt(composition.artifact_store, artifact_id)
        if (
            receipt is not None
            and receipt.kind == "node-execution"
            and receipt.run_id == restored["run_id"]
            and receipt.thread_id == restored["thread_id"]
            and receipt.payload.get("input_state_sha256") == input_sha256
        ):
            matches.append((artifact_id, receipt.payload))
    if len(matches) != 1:
        raise ArtifactIntegrityError("durable event node receipt is missing or ambiguous")
    node_receipt_id, payload = matches[0]
    try:
        execution = _node_execution_from_receipt_payload(payload.get("record"))
    except ArtifactIntegrityError:
        raise ArtifactIntegrityError(
            "durable event node receipt is corrupt"
        ) from None
    base_event = _run_event_from_receipt_payload(payload.get("event"))
    if execution.node != node or base_event.node != node:
        raise ArtifactIntegrityError("durable event node receipt conflicts with graph state")
    expected_event = base_event.model_copy(
        update={
            "artifact_ids": tuple(
                dict.fromkeys((*base_event.artifact_ids, node_receipt_id))
            )
        }
    )
    if not _run_events_match_exact(event, expected_event):
        raise ArtifactIntegrityError("durable event bytes conflict with node receipt")
    recovered = _state_from_payload(payload.get("state"))
    if (
        recovered["run_id"] != restored["run_id"]
        or recovered["thread_id"] != restored["thread_id"]
        or recovered["next_event_seq"] != restored["next_event_seq"]
    ):
        raise ArtifactIntegrityError("durable event state receipt is corrupt")
    calls, budget_facts = _provider_calls_from_receipt(
        composition,
        run_id=restored["run_id"],
        thread_id=restored["thread_id"],
        owning_node=execution.node,
        owning_output_artifact_ids=execution.output_artifact_ids,
        receipt_ids=payload.get("provider_receipt_ids"),
    )
    if any(call.node != node for call in calls):
        raise ArtifactIntegrityError("provider receipt belongs to another graph node")
    audit = _audit_buffer(context)
    if tuple(audit.provider_calls) or tuple(audit.provider_receipt_ids):
        raise ArtifactIntegrityError("event recovery began after provider work")
    _restore_receipted_budget(context, budget_facts)
    replayed_snapshot = context.budget_accountant.snapshot()
    if replayed_snapshot != recovered["budget_snapshot"]:
        raise ArtifactIntegrityError("durable event budget receipt is inconsistent")
    context.elapsed_tracker.recover(
        elapsed_wall_seconds=recovered["elapsed_wall_seconds"],
        elapsed_base_seconds=context.elapsed_base_seconds,
    )
    await sink(event)
    verified = await sink.get_event(run_id=event.run_id, seq=event.seq)
    if not _run_events_match_exact(verified, event):
        raise ArtifactIntegrityError("durable event sink failed exact verification")
    work_ids = tuple(
        dict.fromkeys((*recovered["baseline_work_artifact_ids"], node_receipt_id))
    )
    return {
        **recovered,
        "baseline_work_artifact_ids": work_ids,
        "next_event_seq": restored["next_event_seq"] + 1,
    }


def _verified_persist_result_pair(
    *,
    composition: _AuditComposition,
    state: BaselineState,
    evidence_graph_artifact_id: object,
    manifest_artifact_id: object,
) -> tuple[str | None, str | None]:
    """Return a result pair only after strict content-addressed readback."""
    if (
        type(evidence_graph_artifact_id) is not str
        or type(manifest_artifact_id) is not str
    ):
        return None, None
    graph_id = evidence_graph_artifact_id
    manifest_id = manifest_artifact_id
    if (
        not graph_id.startswith("sha256:")
        or not manifest_id.startswith("sha256:")
    ):
        return None, None
    try:
        graph_bytes = composition.artifact_store.get_bytes(graph_id)
        manifest_bytes = composition.artifact_store.get_bytes(manifest_id)
        if (
            hashlib.sha256(graph_bytes).hexdigest() != graph_id.removeprefix("sha256:")
            or hashlib.sha256(manifest_bytes).hexdigest()
            != manifest_id.removeprefix("sha256:")
        ):
            return None, None
        graph_value: object = json.loads(
            graph_bytes,
            object_pairs_hook=_unique_json_object,
            parse_constant=_reject_json_constant,
        )
        if type(graph_value) is not dict:
            return None, None
        graph_payload = cast("dict[str, object]", graph_value)
        if set(graph_payload) != {"evidence", "sources"}:
            return None, None
        evidence_values = graph_payload["evidence"]
        source_values = graph_payload["sources"]
        if type(evidence_values) is not list or type(source_values) is not list:
            return None, None
        evidence_items = cast("list[object]", evidence_values)
        source_items = cast("list[object]", source_values)
        evidence = tuple(
            EvidenceSpan.model_validate_json(_canonical_bytes(item), strict=True)
            for item in evidence_items
        )
        sources = tuple(
            SourceDocument.model_validate_json(_canonical_bytes(item), strict=True)
            for item in source_items
        )
        canonical_graph = _canonical_bytes(
            {
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "sources": [item.model_dump(mode="json") for item in sources],
            }
        )
        if (
            canonical_graph != graph_bytes
            or tuple(item.evidence_id for item in evidence) != state["evidence_ids"]
            or tuple(item.source_id for item in sources) != state["source_ids"]
        ):
            return None, None
        manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
        if (
            _canonical_bytes(manifest.model_dump(mode="json")) != manifest_bytes
            or manifest.run_id != state["run_id"]
            or manifest.thread_id != state["thread_id"]
            or graph_id not in manifest.artifact_ids
        ):
            return None, None
    except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except (ArtifactIntegrityError, FileNotFoundError, OSError, TypeError, ValueError):
        return None, None
    return graph_id, manifest_id


def _node_error_code(error: Exception, *, node: str) -> str:
    if node == "PersistResults":
        return "PERSIST_RESULTS_FAILED"
    if isinstance(error, OperationCancelled):
        return "CANCELLED"
    if isinstance(error, PlanGenerationError):
        return "PLAN_INVALID"
    if isinstance(error, ProviderError):
        return error.code
    if isinstance(error, WorkflowInvariantError):
        return error.code
    if isinstance(error, UsageIntegrityError):
        return error.code
    if isinstance(
        error,
        (
            ArtifactIntegrityError,
            CacheIntegrityError,
            EvidenceIntegrityError,
            StateValidationError,
        ),
    ):
        return "DATA_CORRUPTION"
    return "INTERNAL_ERROR"


type _EventPublicationStatus = Literal[
    "clean_exact",
    "recovered_exact",
    "absent",
    "unreadable",
]


async def _publish_event_with_bounded_recovery(
    *,
    sink: DurableRunEventSink,
    event: RunEvent,
) -> _EventPublicationStatus:
    try:
        await sink(event)
        immediate = await sink.get_event(run_id=event.run_id, seq=event.seq)
    except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:  # noqa: BLE001 - public-safe durable sink boundary
        immediate = None
    else:
        if immediate is not None:
            if not _run_events_match_exact(immediate, event):
                raise ArtifactIntegrityError(
                    "durable event verification failed"
                )
            return "clean_exact"
    try:
        recovered = await sink.get_event(run_id=event.run_id, seq=event.seq)
    except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
        raise
    except Exception:  # noqa: BLE001 - public-safe durable sink boundary
        return "unreadable"
    if recovered is None:
        return "absent"
    if not _run_events_match_exact(recovered, event):
        raise ArtifactIntegrityError("durable event verification failed")
    return "recovered_exact"


async def _publish_missing_event_failure_transition(
    *,
    composition: _AuditComposition,
    context: BaselineRuntimeContext,
    sink: DurableRunEventSink,
    restored: BaselineState,
    node: str,
    update: Mapping[str, object],
    work_ids: Sequence[str],
    audit: _AuditBuffer,
    execution: NodeExecutionRecord,
    base_event: RunEvent,
) -> StateUpdate:
    failure_monotonic = context.monotonic()
    failure_finished_at = context.utc_now()
    latency_seconds = (failure_finished_at - execution.started_at).total_seconds()
    if latency_seconds < 0.0:
        raise ArtifactIntegrityError("node audit clock moved backwards")
    failure_elapsed = max(
        cast("float", update["elapsed_wall_seconds"]),
        context.elapsed_base_seconds
        + context.elapsed_tracker.recovered_offset_seconds
        + max(0.0, failure_monotonic - context.run_started_monotonic),
    )
    terminal_receipt_id = _put_audit_receipt(
        composition.artifact_store,
        kind="terminal",
        run_id=restored["run_id"],
        thread_id=restored["thread_id"],
        receipt_key="terminal",
        payload={
            "error_code": "DATA_CORRUPTION",
            "elapsed_wall_seconds": failure_elapsed,
            "finished_at": failure_finished_at.isoformat(),
            "terminal_event_seq": restored["next_event_seq"],
        },
    )
    child_receipt_ids = list(
        dict.fromkeys(
            (
                *audit.provider_receipt_ids,
                *audit.child_receipt_ids,
                terminal_receipt_id,
            )
        )
    )
    failure_update = {
        **update,
        "baseline_work_artifact_ids": tuple(
            dict.fromkeys((*work_ids, *child_receipt_ids))
        ),
        "elapsed_wall_seconds": failure_elapsed,
        "error_code": "DATA_CORRUPTION",
        "failed_node": node,
        "next_event_seq": restored["next_event_seq"],
    }
    failure_state = validate_baseline_state({**restored, **failure_update})
    failure_execution = NodeExecutionRecord(
        node=node,
        attempt=execution.attempt,
        started_at=execution.started_at,
        finished_at=failure_finished_at,
        latency_ms=round(latency_seconds * 1000),
        status="failed",
        input_artifact_ids=execution.input_artifact_ids,
        output_artifact_ids=execution.output_artifact_ids,
        usage=_audit_usage(
            tuple(audit.provider_calls),
            wall_seconds=latency_seconds,
            pricing_status=composition.pricing_status,
        ),
        error_code="DATA_CORRUPTION",
    )
    failure_base_event = base_event.model_copy(
        update={
            "timestamp": failure_finished_at,
            "kind": "node_failed",
            "status": "failed",
            "public_payload": {
                "is_partial": failure_state["is_partial"],
                "stop_reason": failure_state["stop_reason"],
            },
            "error_code": "DATA_CORRUPTION",
        }
    )
    failure_receipt_id = _put_audit_receipt(
        composition.artifact_store,
        kind="node-execution",
        run_id=restored["run_id"],
        thread_id=restored["thread_id"],
        receipt_key=f"node:{node}:{failure_execution.attempt}",
        payload={
            "event": failure_base_event.model_dump(mode="json"),
            "input_state_sha256": _hash_json(_state_payload(restored)),
            "provider_receipt_ids": list(audit.provider_receipt_ids),
            "child_receipt_ids": child_receipt_ids,
            "record": failure_execution.model_dump(mode="json"),
            "state": _state_payload(failure_state),
        },
    )
    failure_event = failure_base_event.model_copy(
        update={
            "artifact_ids": tuple(
                dict.fromkeys(
                    (*failure_base_event.artifact_ids, failure_receipt_id)
                )
            )
        }
    )
    publication = await _publish_event_with_bounded_recovery(
        sink=sink,
        event=failure_event,
    )
    if publication in {"absent", "unreadable"}:
        raise ArtifactIntegrityError("durable event failure transition failed")
    return {
        **failure_update,
        "baseline_work_artifact_ids": tuple(
            dict.fromkeys(
                (*failure_state["baseline_work_artifact_ids"], failure_receipt_id)
            )
        ),
        "next_event_seq": failure_event.seq + 1,
    }


def _safe_node(
    node: str,
    handler: BaselineNode,
    *,
    audit_composition: _AuditComposition | None,
) -> BaselineNode:
    async def invoke(state: BaselineState) -> StateUpdate:
        restored = validate_baseline_state(state)
        runtime = get_runtime(BaselineRuntimeContext)
        context = cast("BaselineRuntimeContext | None", runtime.context)
        if context is None:
            return await handler(restored)
        audit = context.audit
        audit.reset()
        if not isinstance(context.emit, DurableRunEventSink):
            raise WorkflowInvariantError(code="INVALID_EVENT_SINK")
        sink = context.emit
        if audit_composition is not None:
            try:
                existing_event = await sink.get_event(
                    run_id=restored["run_id"],
                    seq=restored["next_event_seq"],
                )
            except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
                raise
            except Exception:  # noqa: BLE001 - public-safe durable sink boundary
                raise ArtifactIntegrityError("durable event lookup failed") from None
            if existing_event is not None:
                try:
                    return await _recover_durable_node_event(
                        composition=audit_composition,
                        context=context,
                        sink=sink,
                        restored=restored,
                        node=node,
                        event=existing_event,
                    )
                except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
                    raise
                except Exception:  # noqa: BLE001 - public-safe durable sink boundary
                    raise ArtifactIntegrityError("durable event recovery failed") from None
        node_started_at = context.utc_now()
        audit.begin(
            graph_node=node,
            node_attempt=(
                _next_node_attempt(audit_composition, restored, node)
                if audit_composition is not None
                else 1
            ),
        )
        before = context.budget_accountant.snapshot()
        update: dict[str, object]
        try:
            if node != "PersistResults":
                _ensure_operation_active(context, operation=node)
            update = dict(await handler(restored))
        except BudgetExceeded as error:
            update = {
                "budget_snapshot": error.snapshot,
                "is_partial": True,
                "stop_reason": "BUDGET_EXHAUSTED",
            }
        except MemoryError:
            raise
        except Exception as error:  # noqa: BLE001 - typed node failure boundary
            update = {
                "error_code": _node_error_code(error, node=node),
                "failed_node": node,
            }
        if audit.pending_provider_call is not None:
            raise ArtifactIntegrityError(
                "provider result audit publication is incomplete"
            )
        after = context.budget_accountant.snapshot()
        update["budget_snapshot"] = update.get("budget_snapshot", after)
        elapsed_wall_seconds = (
            restored["elapsed_wall_seconds"]
            if node == "PersistResults"
            else max(
                restored["elapsed_wall_seconds"],
                context.elapsed_base_seconds
                + context.elapsed_tracker.recovered_offset_seconds
                + max(0.0, context.monotonic() - context.run_started_monotonic),
            )
        )
        update["elapsed_wall_seconds"] = elapsed_wall_seconds
        try:
            merged = validate_baseline_state({**restored, **update})
        except MemoryError:
            raise
        except Exception:  # noqa: BLE001 - malformed dependency update is internal
            update = {
                "budget_snapshot": after,
                "elapsed_wall_seconds": elapsed_wall_seconds,
                "error_code": "INTERNAL_ERROR",
                "failed_node": node,
            }
            merged = validate_baseline_state({**restored, **update})
        prior_artifact_ids = set(_artifact_ids_in(restored))
        result_artifact_ids = tuple(
            dict.fromkeys(
                (
                    *(
                        item
                        for item in _artifact_ids_in(update)
                        if item not in prior_artifact_ids
                    ),
                    *audit.result_artifact_ids,
                )
            )
        )
        node_finished_at = context.utc_now()
        node_receipt_id: str | None = None
        execution: NodeExecutionRecord | None = None
        base_event: RunEvent
        if audit_composition is not None:
            calls = tuple(audit.provider_calls)
            if calls:
                node_started_at = min(
                    node_started_at,
                    *(call.started_at for call in calls),
                )
                node_finished_at = max(
                    node_finished_at,
                    *(call.finished_at for call in calls),
                )
            latency_ms = round(
                (node_finished_at - node_started_at).total_seconds() * 1000
            )
            if latency_ms < 0:
                raise ArtifactIntegrityError("node audit clock moved backwards")
            error_code = merged["error_code"]
            execution = NodeExecutionRecord(
                node=node,
                attempt=audit.node_attempt,
                started_at=node_started_at,
                finished_at=node_finished_at,
                latency_ms=latency_ms,
                status=(
                    "cancelled"
                    if error_code == "CANCELLED"
                    else "failed"
                    if error_code is not None
                    else "completed"
                ),
                input_artifact_ids=_artifact_ids_in(restored),
                output_artifact_ids=result_artifact_ids,
                usage=_audit_usage(
                    calls,
                    wall_seconds=(
                        node_finished_at - node_started_at
                    ).total_seconds(),
                    pricing_status=audit_composition.pricing_status,
                ),
                error_code=error_code,
            )
        error_code = merged["error_code"]
        base_event = RunEvent(
            seq=restored["next_event_seq"],
            run_id=restored["run_id"],
            timestamp=node_finished_at,
            node=node,
            kind="node_completed" if error_code is None else "node_failed",
            status=(
                "cancelled"
                if error_code == "CANCELLED"
                else "failed"
                if error_code is not None
                else "running"
            ),
            public_payload={
                "is_partial": merged["is_partial"],
                "stop_reason": merged["stop_reason"],
            },
            usage_delta=_usage_delta(before, after),
            artifact_ids=result_artifact_ids,
            error_code=error_code,
        )
        event = base_event
        work_ids = tuple(
            cast(
                "tuple[str, ...]",
                update.get(
                    "baseline_work_artifact_ids",
                    restored["baseline_work_artifact_ids"],
                ),
            )
        )
        if audit_composition is not None:
            child_receipt_ids = [
                *audit.provider_receipt_ids,
                *audit.child_receipt_ids,
            ]
            if node != "PersistResults" and (
                merged["error_code"] is not None or node == "FinalizeCitations"
            ):
                terminal_receipt_id = _put_audit_receipt(
                    audit_composition.artifact_store,
                    kind="terminal",
                    run_id=restored["run_id"],
                    thread_id=restored["thread_id"],
                    receipt_key="terminal",
                    payload={
                        "error_code": merged["error_code"],
                        "elapsed_wall_seconds": merged["elapsed_wall_seconds"],
                        "finished_at": node_finished_at.isoformat(),
                        "terminal_event_seq": restored["next_event_seq"],
                    },
                )
                child_receipt_ids.append(terminal_receipt_id)
            update["baseline_work_artifact_ids"] = tuple(
                dict.fromkeys((*work_ids, *child_receipt_ids))
            )
            merged = validate_baseline_state({**restored, **update})
            if execution is None:
                raise ArtifactIntegrityError("node audit execution was not recorded")
            node_receipt_id = _put_audit_receipt(
                audit_composition.artifact_store,
                kind="node-execution",
                run_id=restored["run_id"],
                thread_id=restored["thread_id"],
                receipt_key=f"node:{node}:{execution.attempt}",
                payload={
                    "event": base_event.model_dump(mode="json"),
                    "input_state_sha256": _hash_json(_state_payload(restored)),
                    "provider_receipt_ids": list(audit.provider_receipt_ids),
                    "child_receipt_ids": child_receipt_ids,
                    "record": execution.model_dump(mode="json"),
                    "state": _state_payload(merged),
                },
            )
            event = base_event.model_copy(
                update={
                    "artifact_ids": tuple(
                        dict.fromkeys((*result_artifact_ids, node_receipt_id))
                    )
                }
            )
        publication = await _publish_event_with_bounded_recovery(
            sink=sink,
            event=event,
        )
        if node == "PersistResults" and publication in {"absent", "unreadable"}:
            if (
                audit_composition is not None
                and type(update.get("evidence_graph_artifact_id")) is str
                and type(update.get("manifest_artifact_id")) is str
            ):
                graph_id, manifest_id = _verified_persist_result_pair(
                    composition=audit_composition,
                    state=merged,
                    evidence_graph_artifact_id=merged["evidence_graph_artifact_id"],
                    manifest_artifact_id=merged["manifest_artifact_id"],
                )
            else:
                graph_id, manifest_id = None, None
            update["evidence_graph_artifact_id"] = graph_id
            update["manifest_artifact_id"] = manifest_id
            update["error_code"] = "PERSIST_RESULTS_FAILED"
            update["failed_node"] = "PersistResults"
            update["baseline_work_artifact_ids"] = work_ids
            update["next_event_seq"] = restored["next_event_seq"]
            return validate_baseline_state({**restored, **update})
        if publication == "unreadable":
            raise ArtifactIntegrityError("durable event publication failed")
        if publication == "absent":
            if (
                audit_composition is None
                or execution is None
                or node_receipt_id is None
                or not merged["evidence_ids"]
                or node == "PersistResults"
            ):
                raise ArtifactIntegrityError("durable event publication failed")
            if merged["error_code"] is None:
                return await _publish_missing_event_failure_transition(
                    composition=audit_composition,
                    context=context,
                    sink=sink,
                    restored=restored,
                    node=node,
                    update=update,
                    work_ids=work_ids,
                    audit=audit,
                    execution=execution,
                    base_event=base_event,
                )
            retry_publication = await _publish_event_with_bounded_recovery(
                sink=sink,
                event=event,
            )
            if retry_publication in {"absent", "unreadable"}:
                raise ArtifactIntegrityError("durable event publication failed")
        elif (
            publication == "recovered_exact"
            and merged["error_code"] is None
            and node != "PersistResults"
        ):
            if (
                audit_composition is None
                or execution is None
                or node_receipt_id is None
                or not restored["evidence_ids"]
                or merged["error_code"] is not None
                or node == "FinalizeCitations"
            ):
                raise ArtifactIntegrityError("durable event publication failed")

            work_ids = cast("tuple[str, ...]", update["baseline_work_artifact_ids"])
            update["baseline_work_artifact_ids"] = tuple(
                dict.fromkeys((*work_ids, node_receipt_id))
            )
            update["next_event_seq"] = restored["next_event_seq"] + 1
            successful_state = validate_baseline_state({**restored, **update})
            failure_finished_at = context.utc_now()
            failure_elapsed = max(
                successful_state["elapsed_wall_seconds"],
                context.elapsed_base_seconds
                + context.elapsed_tracker.recovered_offset_seconds
                + max(0.0, context.monotonic() - context.run_started_monotonic),
            )
            terminal_receipt_id = _put_audit_receipt(
                audit_composition.artifact_store,
                kind="terminal",
                run_id=restored["run_id"],
                thread_id=restored["thread_id"],
                receipt_key="terminal",
                payload={
                    "error_code": "DATA_CORRUPTION",
                    "elapsed_wall_seconds": failure_elapsed,
                    "finished_at": failure_finished_at.isoformat(),
                    "terminal_event_seq": successful_state["next_event_seq"],
                },
            )
            update.update(
                {
                    "baseline_work_artifact_ids": tuple(
                        dict.fromkeys(
                            (
                                *successful_state["baseline_work_artifact_ids"],
                                terminal_receipt_id,
                            )
                        )
                    ),
                    "elapsed_wall_seconds": failure_elapsed,
                    "error_code": "DATA_CORRUPTION",
                    "failed_node": node,
                }
            )
            failure_state = validate_baseline_state({**restored, **update})
            failure_execution = NodeExecutionRecord(
                node=node,
                attempt=execution.attempt + 1,
                started_at=failure_finished_at,
                finished_at=failure_finished_at,
                latency_ms=0,
                status="failed",
                input_artifact_ids=_artifact_ids_in(successful_state),
                output_artifact_ids=(),
                usage=_audit_usage(
                    (),
                    wall_seconds=0.0,
                    pricing_status=audit_composition.pricing_status,
                ),
                error_code="DATA_CORRUPTION",
            )
            failure_base_event = RunEvent(
                seq=successful_state["next_event_seq"],
                run_id=restored["run_id"],
                timestamp=failure_finished_at,
                node=node,
                kind="node_failed",
                status="failed",
                public_payload={
                    "is_partial": failure_state["is_partial"],
                    "stop_reason": failure_state["stop_reason"],
                },
                usage_delta=ResourceUsage.zero(
                    cost_known=audit_composition.pricing_status != "unknown"
                ),
                artifact_ids=(),
                error_code="DATA_CORRUPTION",
            )
            failure_receipt_id = _put_audit_receipt(
                audit_composition.artifact_store,
                kind="node-execution",
                run_id=restored["run_id"],
                thread_id=restored["thread_id"],
                receipt_key=f"node:{node}:{failure_execution.attempt}",
                payload={
                    "event": failure_base_event.model_dump(mode="json"),
                    "input_state_sha256": _hash_json(_state_payload(successful_state)),
                    "provider_receipt_ids": [],
                    "child_receipt_ids": [terminal_receipt_id],
                    "record": failure_execution.model_dump(mode="json"),
                    "state": _state_payload(failure_state),
                },
            )
            failure_event = failure_base_event.model_copy(
                update={"artifact_ids": (failure_receipt_id,)}
            )
            try:
                await sink(failure_event)
                verified_failure = await sink.get_event(
                    run_id=failure_event.run_id,
                    seq=failure_event.seq,
                )
            except (asyncio.CancelledError, KeyboardInterrupt, MemoryError, SystemExit):
                raise
            except Exception:  # noqa: BLE001 - public-safe durable sink boundary
                raise ArtifactIntegrityError(
                    "durable event failure transition failed"
                ) from None
            if not _run_events_match_exact(verified_failure, failure_event):
                raise ArtifactIntegrityError(
                    "durable event failure transition is corrupt"
                )
            update["baseline_work_artifact_ids"] = tuple(
                dict.fromkeys(
                    (*failure_state["baseline_work_artifact_ids"], failure_receipt_id)
                )
            )
            update["next_event_seq"] = failure_event.seq + 1
            return update
        if node_receipt_id is not None:
            work_ids = cast("tuple[str, ...]", update["baseline_work_artifact_ids"])
            update["baseline_work_artifact_ids"] = tuple(
                dict.fromkeys((*work_ids, node_receipt_id))
            )
        update["next_event_seq"] = restored["next_event_seq"] + 1
        return update

    return invoke


def _route_on_success(
    success: str,
    *,
    honor_stop: bool = True,
) -> Callable[[BaselineState], str]:
    def route(state: BaselineState) -> str:
        restored = validate_baseline_state(state)
        if restored["error_code"] is not None:
            return "PersistResults"
        if honor_stop and restored["stop_reason"] is not None:
            return "DraftReport"
        return success

    return route


def build_baseline_graph(
    dependencies: BaselineDependencies,
) -> CompiledStateGraph[
    BaselineState,
    BaselineRuntimeContext,
    BaselineState,
    BaselineState,
]:
    graph = cast(
        "Any",
        StateGraph(BaselineState, context_schema=BaselineRuntimeContext),
    )
    composition = dependencies._audit_composition  # pyright: ignore[reportPrivateUsage]

    def safe(node: str, handler: BaselineNode) -> BaselineNode:
        return _safe_node(node, handler, audit_composition=composition)
    graph.add_node("ValidateRequest", safe("ValidateRequest", dependencies.validate_request))
    graph.add_node("Plan", safe("Plan", dependencies.plan))
    graph.add_node("DecideNext", safe("DecideNext", dependencies.decide_next))
    graph.add_node("Search", safe("Search", dependencies.search))
    graph.add_node("Fetch", safe("Fetch", dependencies.fetch))
    graph.add_node(
        "ParseAndNormalize",
        safe("ParseAndNormalize", dependencies.parse_and_normalize),
    )
    graph.add_node("StoreEvidence", safe("StoreEvidence", dependencies.store_evidence))
    graph.add_node("RankEvidence", safe("RankEvidence", dependencies.rank_evidence))
    graph.add_node("DraftReport", safe("DraftReport", dependencies.draft_report))
    graph.add_node(
        "FinalizeCitations",
        safe("FinalizeCitations", dependencies.finalize_citations),
    )
    graph.add_node("PersistResults", safe("PersistResults", dependencies.persist_results))
    graph.set_entry_point("ValidateRequest")
    graph.add_conditional_edges(
        "ValidateRequest",
        _route_on_success("Plan"),
        {"Plan": "Plan", "PersistResults": "PersistResults"},
    )
    graph.add_conditional_edges(
        "Plan",
        _route_on_success("DecideNext"),
        {
            "DecideNext": "DecideNext",
            "DraftReport": "DraftReport",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "DecideNext",
        route_after_decide,
        {
            "Search": "Search",
            "DraftReport": "DraftReport",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "Search",
        _route_on_success("Fetch"),
        {
            "DraftReport": "DraftReport",
            "Fetch": "Fetch",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "Fetch",
        _route_on_success("ParseAndNormalize"),
        {
            "DraftReport": "DraftReport",
            "ParseAndNormalize": "ParseAndNormalize",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "ParseAndNormalize",
        _route_on_success("StoreEvidence"),
        {
            "DraftReport": "DraftReport",
            "StoreEvidence": "StoreEvidence",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "StoreEvidence",
        _route_on_success("RankEvidence"),
        {
            "DraftReport": "DraftReport",
            "RankEvidence": "RankEvidence",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "RankEvidence",
        _route_on_success("DecideNext"),
        {
            "DecideNext": "DecideNext",
            "DraftReport": "DraftReport",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_conditional_edges(
        "DraftReport",
        _route_on_success("FinalizeCitations", honor_stop=False),
        {
            "FinalizeCitations": "FinalizeCitations",
            "PersistResults": "PersistResults",
        },
    )
    graph.add_edge("FinalizeCitations", "PersistResults")
    graph.add_edge("PersistResults", END)
    compiled = cast(
        "CompiledStateGraph[BaselineState, BaselineRuntimeContext, BaselineState, BaselineState]",
        graph.compile(checkpointer=dependencies.checkpointer),
    )
    cast("Any", compiled)._baseline_audit_composition = composition
    return compiled


__all__ = [
    "BaselineDependencies",
    "BaselineNode",
    "BaselineNodeHandlers",
    "BaselineRuntimeContext",
    "DurableRunEventSink",
    "InvocationUsageObserver",
    "StateUpdate",
    "UsageCostResolver",
    "UsageIntegrityError",
    "WorkflowInvariantError",
    "baseline_is_sufficient",
    "build_baseline_graph",
    "decide_baseline_stop",
    "rank_baseline_coverage",
    "route_after_decide",
    "stable_operation_id",
]
