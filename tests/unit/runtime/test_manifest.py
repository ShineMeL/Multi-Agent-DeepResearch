from datetime import UTC, datetime, timedelta
from decimal import Decimal

import pytest
from pydantic import ValidationError

from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.runtime.manifest import (
    CostCalculator,
    NodeExecutionRecord,
    PricingSnapshot,
    ProviderCallRecord,
    ProviderProfileRecord,
    RunManifest,
)


def _usage(*, cost: Decimal | None = Decimal("0.114")) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=1_000_000,
        cached_tokens=200_000,
        output_tokens=100_000,
        reasoning_tokens=50_000,
        total_tokens=1_150_000,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=1.0,
        cost_usd=cost,
    )


@pytest.fixture
def pricing_snapshot() -> PricingSnapshot:
    return PricingSnapshot(
        snapshot_id="pricing-v1",
        provider_id="model-provider",
        endpoint_type="responses",
        model_id="model-v1",
        effective_at=datetime(2026, 8, 29, tzinfo=UTC),
        currency="USD",
        input_tokens_per_million_usd=Decimal("0.10"),
        output_tokens_per_million_usd=Decimal("0.20"),
        cached_tokens_per_million_usd=Decimal("0.02"),
        reasoning_tokens_per_million_usd=Decimal("0.20"),
    )


def _manifest(pricing_snapshot: PricingSnapshot, **updates: object) -> RunManifest:
    started = datetime(2026, 8, 29, tzinfo=UTC)
    call = ProviderCallRecord(
        operation="model",
        node="Planner",
        provider_id="model-provider",
        endpoint_type="responses",
        model_id="model-v1",
        model_revision="revision-1",
        request_sha256="1" * 64,
        complete_parameters={"seed_supported": True},
        prompt_version="planner-v1",
        system_prompt_hash="2" * 64,
        tool_schema_hash="3" * 64,
        output_schema_hash="4" * 64,
        temperature=Decimal(0),
        seed=7,
        started_at=started,
        finished_at=started + timedelta(milliseconds=100),
        latency_ms=100,
        attempt=1,
        cache_hit=False,
        outcome_code="SUCCESS",
        usage=_usage(),
        pricing_snapshot_id=pricing_snapshot.snapshot_id,
        estimated_cost_usd=Decimal("0.114"),
    )
    budget = RunBudget(
        max_search_calls=8,
        max_pages=12,
        max_total_tokens=2_000_000,
        max_wall_time_seconds=300,
        max_cost_usd=Decimal(1),
        max_retries=2,
        used_by_node=RunBudget.preset("medium").used_by_node,
    )
    payload: dict[str, object] = {
        "schema_version": "run-manifest-v1",
        "run_id": "run-1",
        "thread_id": "thread-1",
        "code_commit": "a" * 40,
        "dependency_lock_sha256": "5" * 64,
        "request_sha256": "6" * 64,
        "config_sha256": "7" * 64,
        "workflow_id": "baseline-v1",
        "graph_version": "graph-v1",
        "planner_id": "P1",
        "provider_profiles": (ProviderProfileRecord(profile_id="default", execution_mode="replay", provider_ids=("model-provider",), configuration_sha256="8" * 64),),
        "model_ids": ("model-v1",),
        "prompt_versions": {"planner": "v1"},
        "parser_versions": {"html": "v1"},
        "ranker_id": "R1",
        "ranker_weights_version": "weights-v1",
        "budget": budget,
        "usage": _usage(),
        "usage_by_node": {"Planner": _usage()},
        "pricing_status": "estimated",
        "pricing_snapshots": (pricing_snapshot,),
        "provider_calls": (call,),
        "node_executions": (NodeExecutionRecord(node="Planner", attempt=1, started_at=started, finished_at=started + timedelta(milliseconds=100), latency_ms=100, status="completed", input_artifact_ids=(), output_artifact_ids=(), usage=_usage()),),
        "parsed_artifacts": (),
        "evidence_hashes": (),
        "source_snapshot_ids": (),
        "artifact_ids": (),
        "run_event_count": 1,
        "run_events_sha256": "9" * 64,
        "seed": 7,
        "seed_supported": True,
        "cache_hit_count": 0,
        "stop_reason": "SUFFICIENT",
        "is_partial": False,
        "failure_codes": (),
        "replay_parent": None,
        "started_at": started,
        "finished_at": started + timedelta(seconds=1),
        "manifest_sha256": "",
    }
    payload.update(updates)
    return RunManifest.create(payload)


def test_manifest_hash_changes_when_prompt_version_changes(pricing_snapshot: PricingSnapshot) -> None:
    left = _manifest(pricing_snapshot, prompt_versions={"planner": "v1"})
    right = _manifest(pricing_snapshot, prompt_versions={"planner": "v2"})

    assert left.canonical_sha256() != right.canonical_sha256()
    with pytest.raises(TypeError):
        left.prompt_versions["planner"] = "changed"


def test_estimated_cost_requires_a_complete_pricing_snapshot(pricing_snapshot: PricingSnapshot) -> None:
    with pytest.raises(ValidationError):
        _manifest(pricing_snapshot, pricing_snapshots=())


def test_cost_calculator_uses_decimal_unit_rates(pricing_snapshot: PricingSnapshot) -> None:
    breakdown = CostCalculator.estimate(_usage(), pricing_snapshot)

    assert breakdown.total_usd == Decimal("0.114000000")


def test_manifest_preserves_cache_hit_usage_without_double_charging(pricing_snapshot: PricingSnapshot) -> None:
    base = _manifest(pricing_snapshot)
    cached_call = base.provider_calls[0].model_copy(
        update={"cache_hit": True, "estimated_cost_usd": Decimal(0)}
    )
    cached_usage = base.usage.model_copy(update={"cost_usd": Decimal(0)})
    cached_execution = base.node_executions[0].model_copy(update={"usage": cached_usage})
    cached = RunManifest.create(
        {
            **base.model_dump(),
            "provider_calls": (cached_call,),
            "cache_hit_count": 1,
            "usage": cached_usage,
            "usage_by_node": {"Planner": cached_usage},
            "node_executions": (cached_execution,),
        }
    )

    assert cached.provider_calls[0].usage.total_tokens > 0
    assert cached.usage.cost_usd == 0
    with pytest.raises(ValidationError, match="cache hit.*zero"):
        cached.model_copy(
            update={
                "provider_calls": (
                    cached_call.model_copy(update={"estimated_cost_usd": Decimal("0.000000001")}),
                )
            }
        )


def test_unknown_pricing_must_remain_unestimated(pricing_snapshot: PricingSnapshot) -> None:
    base = _manifest(pricing_snapshot)
    unknown_call = base.provider_calls[0].model_copy(
        update={"pricing_snapshot_id": None, "estimated_cost_usd": None}
    )

    unknown_usage = base.usage.model_copy(update={"cost_usd": None})
    unknown_execution = base.node_executions[0].model_copy(update={"usage": unknown_usage})
    unknown = RunManifest.create(
        {
            **base.model_dump(),
            "provider_calls": (unknown_call,),
            "pricing_status": "unknown",
            "pricing_snapshots": (),
            "usage": unknown_usage,
            "usage_by_node": {"Planner": unknown_usage},
            "node_executions": (unknown_execution,),
        }
    )

    assert unknown.usage.cost_usd is None


def test_provider_call_revalidates_nonfinite_usage(pricing_snapshot: PricingSnapshot) -> None:
    base = _manifest(pricing_snapshot)
    invalid_usage = base.provider_calls[0].usage.model_copy(
        update={"wall_seconds": float("inf")}
    )

    with pytest.raises(ValidationError, match="finite|wall_seconds"):
        base.provider_calls[0].model_copy(update={"usage": invalid_usage})


def test_manifest_rejects_provider_usage_hidden_by_smaller_aggregates(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    provider_usage = ResourceUsage(
        input_tokens=3_000_000,
        output_tokens=0,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=3_000_000,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=1.0,
        cost_usd=Decimal("0.3"),
    )
    call = base.provider_calls[0].model_copy(
        update={"usage": provider_usage, "estimated_cost_usd": Decimal("0.3")}
    )
    understated = ResourceUsage.zero(cost_known=True).model_copy(
        update={"cost_usd": Decimal("0.3"), "wall_seconds": 1.0}
    )
    execution = base.node_executions[0].model_copy(update={"usage": understated})

    with pytest.raises(ValidationError, match="usage|budget|provider"):
        RunManifest.create(
            {
                **base.model_dump(),
                "provider_calls": (call,),
                "node_executions": (execution,),
                "usage_by_node": {"Planner": understated},
                "usage": understated,
            }
        )


def test_manifest_rejects_call_without_matching_node_attempt(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    orphan = base.provider_calls[0].model_copy(update={"node": "Writer"})

    with pytest.raises(ValidationError, match="node|attempt"):
        RunManifest.create(
            {**base.model_dump(), "provider_calls": (orphan,)}
        )


def test_manifest_rejects_usage_by_node_that_disagrees_with_executions(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    understated = ResourceUsage.zero(cost_known=True).model_copy(
        update={"cost_usd": Decimal("0.114"), "wall_seconds": 1.0}
    )

    with pytest.raises(ValidationError, match="usage_by_node|execution"):
        RunManifest.create(
            {**base.model_dump(), "usage_by_node": {"Planner": understated}}
        )


def test_manifest_wall_time_uses_run_envelope_not_concurrent_sum(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    zero = ResourceUsage.zero(cost_known=True).model_copy(update={"wall_seconds": 0.1})
    writer = base.node_executions[0].model_copy(
        update={"node": "Writer", "usage": zero}
    )

    concurrent = base.model_copy(
        update={
            "node_executions": (*base.node_executions, writer),
            "usage_by_node": {**base.usage_by_node, "Writer": zero},
        }
    )

    assert concurrent.usage.wall_seconds == 1.0
    assert sum(item.wall_seconds for item in concurrent.usage_by_node.values()) == 1.1


def _search_call(
    started: datetime,
    *,
    request_sha256: str,
    attempt: int = 1,
    offset_ms: int = 10,
    usage_retries: int | None = None,
) -> ProviderCallRecord:
    usage = ResourceUsage.zero().model_copy(
        update={
            "search_calls": 1,
            "retries": int(attempt > 1) if usage_retries is None else usage_retries,
            "wall_seconds": 0.01,
        }
    )
    call_started = started + timedelta(milliseconds=offset_ms)
    return ProviderCallRecord(
        operation="search",
        node="Tool",
        provider_id="search-provider",
        endpoint_type="search",
        request_sha256=request_sha256,
        snapshot_id="search-snapshot",
        normalized_query=f"query-{request_sha256[0]}",
        locale="en-US",
        complete_parameters={"filters": None, "limit": 5},
        time_policy="recorded",
        started_at=call_started,
        finished_at=call_started + timedelta(milliseconds=10),
        latency_ms=10,
        attempt=attempt,
        cache_hit=False,
        outcome_code="SUCCESS",
        usage=usage,
    )


def _fetch_call(
    started: datetime,
    *,
    request_sha256: str = "f" * 64,
    path: str = "document",
    offset_ms: int = 30,
) -> ProviderCallRecord:
    usage = ResourceUsage.zero().model_copy(
        update={"pages": 1, "wall_seconds": 0.01}
    )
    call_started = started + timedelta(milliseconds=offset_ms)
    return ProviderCallRecord(
        operation="fetch",
        node="Tool",
        provider_id="fetch-provider",
        endpoint_type="fetch",
        request_sha256=request_sha256,
        snapshot_id="fetch-snapshot",
        complete_parameters={
            "canonical_url": f"https://example.com/{path}",
            "fetch_policy": "recorded",
            "accepted_content_types": ["text/html"],
        },
        started_at=call_started,
        finished_at=call_started + timedelta(milliseconds=10),
        latency_ms=10,
        attempt=1,
        cache_hit=False,
        outcome_code="SUCCESS",
        usage=usage,
    )


@pytest.mark.parametrize("cache_hit", (False, True))
def test_estimated_manifest_supports_priced_model_with_unpriced_search_and_fetch(
    pricing_snapshot: PricingSnapshot, cache_hit: bool
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    charged = Decimal(0) if cache_hit else Decimal("0.114")
    model_call = base.provider_calls[0].model_copy(
        update={"cache_hit": cache_hit, "estimated_cost_usd": charged}
    )
    planner_usage = base.usage.model_copy(update={"cost_usd": charged})
    planner_execution = base.node_executions[0].model_copy(
        update={"usage": planner_usage}
    )
    search = _search_call(started, request_sha256="a" * 64)
    fetch = _fetch_call(started)
    tool_observed = ResourceUsage.zero().model_copy(
        update={"search_calls": 1, "pages": 1, "wall_seconds": 0.1}
    )
    tool_attributed = tool_observed.model_copy(update={"cost_usd": Decimal(0)})
    tool_execution = NodeExecutionRecord(
        node="Tool",
        attempt=1,
        started_at=started,
        finished_at=started + timedelta(milliseconds=100),
        latency_ms=100,
        status="completed",
        input_artifact_ids=(),
        output_artifact_ids=(),
        usage=tool_observed,
    )
    run_usage = planner_usage.model_copy(
        update={"search_calls": 1, "pages": 1, "wall_seconds": 1.0}
    )

    mixed = RunManifest.create(
        {
            **base.model_dump(),
            "provider_calls": (model_call, search, fetch),
            "node_executions": (planner_execution, tool_execution),
            "usage_by_node": {
                "Planner": planner_usage,
                "Tool": tool_attributed,
            },
            "usage": run_usage,
            "cache_hit_count": int(cache_hit),
        }
    )

    assert mixed.usage.cost_usd == charged
    assert mixed.usage_by_node["Tool"].cost_usd == 0
    assert mixed.usage.search_calls == 1
    assert mixed.usage.pages == 1


def test_provider_attempts_are_per_request_and_link_by_containing_execution(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    calls = (
        _search_call(started, request_sha256="a" * 64, offset_ms=10),
        _search_call(started, request_sha256="b" * 64, offset_ms=30),
        _search_call(started, request_sha256="c" * 64, attempt=1, offset_ms=50),
        _search_call(started, request_sha256="c" * 64, attempt=2, offset_ms=70),
    )
    fetches = (
        _fetch_call(
            started, request_sha256="d" * 64, path="first", offset_ms=20
        ),
        _fetch_call(
            started, request_sha256="e" * 64, path="second", offset_ms=40
        ),
    )
    tool_observed = ResourceUsage.zero().model_copy(
        update={"search_calls": 4, "pages": 2, "retries": 1, "wall_seconds": 0.1}
    )
    tool_attributed = tool_observed.model_copy(update={"cost_usd": Decimal(0)})
    tool_execution = NodeExecutionRecord(
        node="Tool", attempt=1, started_at=started,
        finished_at=started + timedelta(milliseconds=100), latency_ms=100,
        status="completed", input_artifact_ids=(), output_artifact_ids=(),
        usage=tool_observed,
    )
    run_usage = base.usage.model_copy(
        update={"search_calls": 4, "pages": 2, "retries": 1, "wall_seconds": 1.0}
    )

    manifest = RunManifest.create(
        {
            **base.model_dump(),
            "provider_calls": (*base.provider_calls, *calls, *fetches),
            "node_executions": (*base.node_executions, tool_execution),
            "usage_by_node": {**base.usage_by_node, "Tool": tool_attributed},
            "usage": run_usage,
        }
    )

    assert [call.attempt for call in manifest.provider_calls[1:]] == [1, 1, 1, 2, 1, 1]


def _manifest_with_tool_history(
    base: RunManifest,
    *,
    calls: tuple[ProviderCallRecord, ...],
    executions: tuple[NodeExecutionRecord, ...],
) -> RunManifest:
    search_calls = sum(item.usage.search_calls for item in executions)
    pages = sum(item.usage.pages for item in executions)
    retries = sum(item.usage.retries for item in executions)
    tool_usage = ResourceUsage.zero(cost_known=True).model_copy(
        update={
            "search_calls": search_calls,
            "pages": pages,
            "retries": retries,
            "wall_seconds": max(item.usage.wall_seconds for item in executions),
        }
    )
    run_usage = base.usage.model_copy(
        update={
            "search_calls": search_calls,
            "pages": pages,
            "retries": retries,
            "wall_seconds": 1.0,
        }
    )
    return RunManifest.create(
        {
            **base.model_dump(),
            "provider_calls": (*base.provider_calls, *calls),
            "node_executions": (*base.node_executions, *executions),
            "usage_by_node": {**base.usage_by_node, "Tool": tool_usage},
            "usage": run_usage,
            "cache_hit_count": base.cache_hit_count + sum(call.cache_hit for call in calls),
        }
    )


def _tool_execution(
    started: datetime,
    *,
    attempt: int,
    start_ms: int,
    finish_ms: int,
    search_calls: int = 1,
    retries: int = 0,
) -> NodeExecutionRecord:
    return NodeExecutionRecord(
        node="Tool",
        attempt=attempt,
        started_at=started + timedelta(milliseconds=start_ms),
        finished_at=started + timedelta(milliseconds=finish_ms),
        latency_ms=finish_ms - start_ms,
        status="completed",
        input_artifact_ids=(),
        output_artifact_ids=(),
        usage=ResourceUsage.zero().model_copy(
            update={
                "search_calls": search_calls,
                "retries": retries,
                "wall_seconds": 0.05,
            }
        ),
    )


def test_provider_attempts_reset_for_each_containing_node_execution(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    executions = (
        _tool_execution(started, attempt=1, start_ms=0, finish_ms=50),
        _tool_execution(started, attempt=2, start_ms=50, finish_ms=100),
    )
    first = _search_call(started, request_sha256="a" * 64, offset_ms=10)
    second = _search_call(started, request_sha256="a" * 64, offset_ms=60)

    manifest = _manifest_with_tool_history(
        base, calls=(first, second), executions=executions
    )

    assert [call.attempt for call in manifest.provider_calls[-2:]] == [1, 1]


def test_distinct_snapshot_identities_each_start_at_provider_attempt_one(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    execution = _tool_execution(
        started, attempt=1, start_ms=0, finish_ms=100, search_calls=2
    )
    first = _search_call(started, request_sha256="a" * 64, offset_ms=10).model_copy(
        update={"snapshot_id": "snapshot-one"}
    )
    second = _search_call(started, request_sha256="a" * 64, offset_ms=30).model_copy(
        update={"snapshot_id": "snapshot-two"}
    )

    manifest = _manifest_with_tool_history(
        base, calls=(first, second), executions=(execution,)
    )

    assert [call.attempt for call in manifest.provider_calls[-2:]] == [1, 1]


def _manifest_with_one_tool_execution(
    base: RunManifest, calls: tuple[ProviderCallRecord, ...]
) -> RunManifest:
    execution = _tool_execution(
        base.started_at,
        attempt=1,
        start_ms=0,
        finish_ms=100,
        search_calls=len(calls),
        retries=sum(call.usage.retries for call in calls),
    )
    return _manifest_with_tool_history(
        base, calls=calls, executions=(execution,)
    )


@pytest.mark.parametrize("history", ("reversed", "overlapping"))
def test_provider_retry_attempts_are_chronological_and_non_overlapping(
    pricing_snapshot: PricingSnapshot, history: str
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    if history == "reversed":
        calls = (
            _search_call(
                started, request_sha256="a" * 64, attempt=1, offset_ms=70
            ),
            _search_call(
                started, request_sha256="a" * 64, attempt=2, offset_ms=10
            ),
        )
    else:
        calls = (
            _search_call(
                started, request_sha256="a" * 64, attempt=1, offset_ms=10
            ).model_copy(
                update={
                    "finished_at": started + timedelta(milliseconds=80),
                    "latency_ms": 70,
                }
            ),
            _search_call(
                started, request_sha256="a" * 64, attempt=2, offset_ms=50
            ),
        )

    with pytest.raises(ValidationError, match="chronological|overlap|attempt"):
        _manifest_with_one_tool_execution(base, calls)


def test_identical_fresh_provider_invocations_can_each_start_at_attempt_one(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    calls = (
        _search_call(started, request_sha256="a" * 64, attempt=1, offset_ms=10),
        _search_call(started, request_sha256="a" * 64, attempt=1, offset_ms=30),
    )

    manifest = _manifest_with_one_tool_execution(base, calls)

    assert [call.attempt for call in manifest.provider_calls[-2:]] == [1, 1]
    assert manifest.usage.retries == 0


def test_retry_history_rejects_hidden_retry_under_zero_budget(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    calls = (
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=1,
            offset_ms=10,
            usage_retries=0,
        ),
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=2,
            offset_ms=30,
            usage_retries=0,
        ),
    )
    zero_retry_budget = base.budget.model_copy(update={"max_retries": 0})

    with pytest.raises(ValidationError, match="retry"):
        _manifest_with_one_tool_execution(
            base.model_copy(update={"budget": zero_retry_budget}), calls
        )


def test_retry_history_rejects_provider_call_double_count(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    calls = (
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=1,
            offset_ms=10,
            usage_retries=1,
        ),
    )

    with pytest.raises(ValidationError, match="retry"):
        _manifest_with_one_tool_execution(base, calls)


def test_retry_history_reconciles_multiple_retries_and_cache_outcomes(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    calls = (
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=1,
            offset_ms=10,
        ).model_copy(update={"outcome_code": "NETWORK"}),
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=2,
            offset_ms=30,
        ).model_copy(update={"outcome_code": "RATE_LIMITED"}),
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=3,
            offset_ms=50,
        ).model_copy(update={"cache_hit": True}),
    )
    execution = _tool_execution(
        base.started_at,
        attempt=1,
        start_ms=0,
        finish_ms=100,
        search_calls=3,
        retries=2,
    )

    manifest = _manifest_with_tool_history(
        base,
        calls=calls,
        executions=(execution,),
    )

    assert manifest.usage.retries == 2


def test_retry_history_is_revalidated_on_manifest_copy(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    calls = (
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=1,
            offset_ms=10,
        ),
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=2,
            offset_ms=30,
        ),
    )
    manifest = _manifest_with_one_tool_execution(base, calls)
    hidden_calls = tuple(
        call.model_copy(update={"usage": call.usage.model_copy(update={"retries": 0})})
        for call in manifest.provider_calls
    )
    hidden_executions = tuple(
        execution.model_copy(
            update={"usage": execution.usage.model_copy(update={"retries": 0})}
        )
        for execution in manifest.node_executions
    )
    hidden_by_node = {
        node: usage.model_copy(update={"retries": 0})
        for node, usage in manifest.usage_by_node.items()
    }

    with pytest.raises(ValidationError, match="retry"):
        manifest.model_copy(
            update={
                "provider_calls": hidden_calls,
                "node_executions": hidden_executions,
                "usage_by_node": hidden_by_node,
                "usage": manifest.usage.model_copy(update={"retries": 0}),
            }
        )


@pytest.mark.parametrize(
    "attempts",
    (
        (2,),
        (1, 3),
        (1, 2, 2),
        (1, 1, 2, 2),
    ),
)
def test_provider_invocation_sequences_reject_gaps_and_ambiguous_interleaving(
    pricing_snapshot: PricingSnapshot, attempts: tuple[int, ...]
) -> None:
    base = _manifest(pricing_snapshot)
    calls = tuple(
        _search_call(
            base.started_at,
            request_sha256="a" * 64,
            attempt=attempt,
            offset_ms=10 + index * 20,
        )
        for index, attempt in enumerate(attempts)
    )

    with pytest.raises(ValidationError, match="attempt|invocation|ambiguous"):
        _manifest_with_one_tool_execution(base, calls)


def test_node_executions_must_be_inside_the_run_envelope(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    shifted_start = base.started_at - timedelta(days=1)
    shifted_call = base.provider_calls[0].model_copy(
        update={
            "started_at": shifted_start,
            "finished_at": shifted_start + timedelta(milliseconds=100),
        }
    )
    shifted_execution = base.node_executions[0].model_copy(
        update={
            "started_at": shifted_start,
            "finished_at": shifted_start + timedelta(milliseconds=100),
        }
    )

    with pytest.raises(ValidationError, match="run envelope|execution"):
        RunManifest.create(
            {
                **base.model_dump(),
                "provider_calls": (shifted_call,),
                "node_executions": (shifted_execution,),
            }
        )


def test_same_node_attempt_intervals_must_not_overlap(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    executions = (
        _tool_execution(
            started, attempt=1, start_ms=0, finish_ms=100, search_calls=0
        ),
        _tool_execution(
            started, attempt=2, start_ms=50, finish_ms=150, search_calls=0
        ),
    )

    with pytest.raises(ValidationError, match="overlap|chronological"):
        _manifest_with_tool_history(base, calls=(), executions=executions)


def test_touching_node_attempts_use_half_open_call_containment(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    executions = (
        _tool_execution(
            started, attempt=1, start_ms=0, finish_ms=50, search_calls=0
        ),
        _tool_execution(
            started, attempt=2, start_ms=50, finish_ms=100, search_calls=0
        ),
    )
    boundary = started + timedelta(milliseconds=50)
    call = _search_call(started, request_sha256="a" * 64, offset_ms=50).model_copy(
        update={
            "started_at": boundary,
            "finished_at": boundary,
            "latency_ms": 0,
            "usage": ResourceUsage.zero(cost_known=True),
        }
    )

    manifest = _manifest_with_tool_history(
        base, calls=(call,), executions=executions
    )

    assert manifest.provider_calls[-1].started_at == executions[1].started_at


@pytest.mark.parametrize("ambiguous", (False, True))
def test_provider_call_requires_exactly_one_containing_node_execution(
    pricing_snapshot: PricingSnapshot, ambiguous: bool
) -> None:
    base = _manifest(pricing_snapshot)
    started = base.started_at
    call = _search_call(
        started, request_sha256="a" * 64, offset_ms=10
    ).model_copy(update={"usage": ResourceUsage.zero(cost_known=True).model_copy(
        update={"search_calls": 1, "wall_seconds": 0.01}
    )})
    observed = call.usage
    first_start = started if ambiguous else started + timedelta(milliseconds=30)
    first = NodeExecutionRecord(
        node="Tool", attempt=1, started_at=first_start,
        finished_at=started + timedelta(milliseconds=100),
        latency_ms=round((started + timedelta(milliseconds=100) - first_start).total_seconds() * 1000),
        status="completed", input_artifact_ids=(), output_artifact_ids=(), usage=observed,
    )
    executions = [first]
    if ambiguous:
        executions.append(first.model_copy(update={"attempt": 2}))
    node_usage = ResourceUsage.zero().model_copy(
        update={"search_calls": len(executions), "wall_seconds": 0.01, "cost_usd": Decimal(0)}
    )
    run_usage = base.usage.model_copy(
        update={"search_calls": len(executions), "wall_seconds": 1.0}
    )

    with pytest.raises(ValidationError, match="contain|execution"):
        RunManifest.create(
            {
                **base.model_dump(),
                "provider_calls": (*base.provider_calls, call),
                "node_executions": (*base.node_executions, *executions),
                "usage_by_node": {**base.usage_by_node, "Tool": node_usage},
                "usage": run_usage,
            }
        )


def test_node_artifact_references_are_content_addressed_and_linked(
    pricing_snapshot: PricingSnapshot,
) -> None:
    base = _manifest(pricing_snapshot)
    artifact_id = f"sha256:{'a' * 64}"
    with pytest.raises(ValidationError, match="artifact"):
        base.node_executions[0].model_copy(
            update={"input_artifact_ids": ("not-a-hash",)}
        )
    with pytest.raises(ValidationError, match="artifact|unique"):
        base.node_executions[0].model_copy(
            update={"output_artifact_ids": (artifact_id, artifact_id)}
        )

    linked_execution = base.node_executions[0].model_copy(
        update={"input_artifact_ids": (artifact_id,)}
    )
    with pytest.raises(ValidationError, match="artifact"):
        base.model_copy(update={"node_executions": (linked_execution,)})

    linked = base.model_copy(
        update={
            "node_executions": (linked_execution,),
            "artifact_ids": (artifact_id,),
        }
    )
    assert linked.node_executions[0].input_artifact_ids == (artifact_id,)


def _call_payload(base: ProviderCallRecord, **updates: object) -> dict[str, object]:
    payload: dict[str, object] = base.model_dump(round_trip=True)
    payload.update(updates)
    return payload


@pytest.mark.parametrize(
    "updates",
    [
        {
            "operation": "search",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "normalized_query": "query",
            "locale": "en-US",
            "time_policy": "recorded",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {"limit": 5, "api_key": "TOP-SECRET"},
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "search",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "normalized_query": "query",
            "locale": "en-US",
            "time_policy": "recorded",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {"filters": {}, "limit": "five"},
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "search",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "normalized_query": "query",
            "locale": "en-US",
            "time_policy": "recorded",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {"filters": None, "limit": True},
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "search",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "normalized_query": "query",
            "locale": "en-US",
            "time_policy": "recorded",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {"filters": None, "limit": 0},
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "search",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "normalized_query": "query",
            "locale": "en-US",
            "time_policy": "recorded",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {"filters": ["not", "mapping"], "limit": 5},
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "search",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "normalized_query": "query",
            "locale": "en-US",
            "time_policy": "recorded",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {"filters": None, "limit": 5},
            "pricing_snapshot_id": "pricing-v1",
            "estimated_cost_usd": None,
        },
        {
            "operation": "fetch",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {
                "canonical_url": "HTTPS://Example.COM:443/path?b=2&a=1",
                "fetch_policy": "recorded",
                "accepted_content_types": ["not a media type"],
            },
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "parse",
            "model_id": None,
            "model_revision": None,
            "snapshot_id": "snapshot-1",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {
                "raw_content_hash": "a" * 64,
                "parser_id": "html",
                "parser_version": "",
                "normalization_version": "v1",
            },
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {
            "operation": "embed",
            "model_id": "embed-v1",
            "model_revision": "revision-1",
            "prompt_version": None,
            "system_prompt_hash": None,
            "tool_schema_hash": None,
            "output_schema_hash": None,
            "temperature": None,
            "seed": None,
            "complete_parameters": {
                "snapshot_sha256": "a" * 64,
                "normalize_embeddings": "yes",
                "canonical_texts_hash": "b" * 64,
            },
            "pricing_snapshot_id": None,
            "estimated_cost_usd": None,
        },
        {"complete_parameters": {"seed_supported": True, "password": "secret"}},
    ],
)
def test_provider_call_parameters_are_exact_typed_and_credential_safe(
    pricing_snapshot: PricingSnapshot, updates: dict[str, object]
) -> None:
    base = _manifest(pricing_snapshot).provider_calls[0]

    with pytest.raises(ValidationError):
        ProviderCallRecord.model_validate(_call_payload(base, **updates))


def test_manifest_rejects_stale_nonblank_hash(pricing_snapshot: PricingSnapshot) -> None:
    base = _manifest(pricing_snapshot)
    payload = base.model_dump()
    payload["prompt_versions"] = {"planner": "changed"}

    with pytest.raises(ValidationError, match="manifest_sha256|hash"):
        RunManifest.model_validate(payload)
