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
    return RunManifest.model_validate(payload)


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
    cached = RunManifest.model_validate(
        {
            **base.model_dump(),
            "provider_calls": (cached_call,),
            "cache_hit_count": 1,
            "usage": base.usage.model_copy(update={"cost_usd": Decimal(0)}),
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

    unknown = RunManifest.model_validate(
        {
            **base.model_dump(),
            "provider_calls": (unknown_call,),
            "pricing_status": "unknown",
            "pricing_snapshots": (),
            "usage": base.usage.model_copy(update={"cost_usd": None}),
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
