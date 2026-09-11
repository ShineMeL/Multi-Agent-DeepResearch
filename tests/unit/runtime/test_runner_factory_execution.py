from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Any

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from pydantic import ValidationError

from deepresearch.domain import RunConfig
from deepresearch.providers.httpx_fetcher import no_op_host_slot
from deepresearch.runtime import CancellationToken
from deepresearch.runtime.checkpoints import checkpoint_ref_from_tuple, checkpoint_serializer
from deepresearch.runtime.manifest import CostCalculator, PricingSnapshot, RunManifest
from deepresearch.runtime.runner_factory import (
    DefaultCoreRunnerBuilder,
    EnvCredentialResolver,
    FrozenProviderRoute,
    FrozenProviderRoutes,
    ProviderProfileDrift,
    _SnapshotCostResolver,
    default_provider_constructors,
)
from deepresearch.storage import LocalArtifactStore, LocalEvidenceStore
from deepresearch.workflow.runner import BaselineRuntimeHooks
from tests.integration.replay.test_baseline_graph import (
    ControlledSegmentClock,
    CountingOfflineEmbedder,
    CountingOfflineFetcher,
    CountingOfflineModel,
    CountingOfflineParser,
    CountingOfflineSearch,
    CrashAfterDurableEventSink,
    MemoryEventSink,
    _offline_plan,
    config,
)


def route(operation: str, provider: Any, **changes: Any) -> dict[str, Any]:
    value = {
        "operation": operation,
        "provider_id": getattr(provider, "provider_id", getattr(provider, "parser_id", "")),
        "endpoint_type": "chat.completions" if operation == "model" else operation,
        "model_id": getattr(provider, "model_id", None),
        "model_revision": getattr(provider, "model_revision", None),
        "base_url": None,
        "credential_ref": None,
        "fallback_rank": 0,
        "parameters": {},
    }
    value.update(changes)
    return value


def freeze(conf: Any, rows: list[dict[str, Any]]) -> FrozenProviderRoutes:
    value = {
        "profile_id": conf.request.provider_profile_id,
        "execution_mode": conf.request.execution_mode,
        "routes": rows,
    }
    value["configuration_sha256"] = hashlib.sha256(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return FrozenProviderRoutes.model_validate(value)


def pricing(provider: str, endpoint: str, model: str, rate: str = "0") -> PricingSnapshot:
    return PricingSnapshot(
        snapshot_id=f"{provider}-{endpoint}",
        provider_id=provider,
        endpoint_type=endpoint,
        model_id=model,
        effective_at="2026-08-29T00:00:00Z",
        currency="USD",
        input_tokens_per_million_usd=Decimal(rate),
        output_tokens_per_million_usd=Decimal(rate),
        cached_tokens_per_million_usd=Decimal(rate),
        reasoning_tokens_per_million_usd=Decimal(rate),
    )


def composition(
    tmp_path: Path,
) -> tuple[
    DefaultCoreRunnerBuilder,
    RunConfig,
    FrozenProviderRoutes,
    tuple[PricingSnapshot, ...],
    Counter[str],
    LocalArtifactStore,
]:
    calls: Counter[str] = Counter()
    providers = {
        "model": CountingOfflineModel(_offline_plan(), calls),
        "search": CountingOfflineSearch(calls),
        "fetch": CountingOfflineFetcher(calls),
        "parse": CountingOfflineParser(calls),
        "embed": CountingOfflineEmbedder(calls),
    }
    conf = config(
        prompt_versions={
            "planner": "fixed-planner-v1",
            "writer": "baseline-writer-v1",
            "planner_queries": "fixed-planner-v1-queries",
        }
    )
    conf = conf.model_copy(
        update={"request": conf.request.model_copy(update={"execution_mode": "live"})}
    )
    rows = [route(operation, provider) for operation, provider in providers.items()]
    routes = freeze(conf, rows)
    snapshots = tuple(
        pricing(
            row["provider_id"],
            endpoint,
            row["model_id"] or row["operation"],
            "1" if row["operation"] == "model" else "0",
        )
        for row in rows
        for endpoint in (
            ("complete", "structured") if row["operation"] == "model" else (row["operation"],)
        )
    )
    registry = {
        row["provider_id"]: (
            lambda route, secret, slot, provider=providers[row["operation"]]: provider
        )
        for row in rows
    }
    artifacts = LocalArtifactStore(tmp_path)
    builder = DefaultCoreRunnerBuilder(
        provider_constructors=registry,
        credential_resolver=EnvCredentialResolver(frozenset()),
        artifact_store=artifacts,
        evidence_store=LocalEvidenceStore(tmp_path),
    )
    return builder, conf, routes, snapshots, calls, artifacts


async def test_priced_baseline_executes_both_model_endpoints_and_parse_with_audited_cost(tmp_path):
    builder, conf, routes, snapshots, calls, artifacts = composition(tmp_path)
    required = builder.required_pricing_keys(conf, routes)
    assert ("offline-model", "complete", "offline-model-v1") in required
    assert ("offline-model", "structured", "offline-model-v1") in required
    assert ("offline-parser", "parse", "parse") in required
    runner = builder.build(
        config=conf,
        provider_routes=routes,
        pricing_snapshots=snapshots,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        cost_calculator=CostCalculator,
    )
    clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
    runner._runtime_hooks = BaselineRuntimeHooks(monotonic=clock.monotonic, utc_now=clock.utc_now)
    result = await runner.run(
        run_id="priced-run",
        thread_id="priced-thread",
        config=conf,
        checkpoint=None,
        emit=MemoryEventSink(),
        cancellation_token=CancellationToken(),
    )
    assert result.status == "completed", result.error_code
    assert result.report_artifact_id is not None
    assert result.manifest_artifact_id is not None
    manifest = RunManifest.model_validate_json(artifacts.get_bytes(result.manifest_artifact_id))
    model_calls = [call for call in manifest.provider_calls if call.operation == "model"]
    assert {call.endpoint_type for call in model_calls} == {"complete", "structured"}
    assert all(call.estimated_cost_usd == Decimal("0.000020") for call in model_calls)
    assert all(
        call.pricing_snapshot_id == f"offline-model-{call.endpoint_type}" for call in model_calls
    )
    parsed = [call for call in manifest.provider_calls if call.operation == "parse"]
    assert parsed and all(call.usage.cost_usd == Decimal(0) for call in parsed)
    assert result.final_usage.cost_usd == Decimal("0.000020") * len(model_calls)
    assert any(key.startswith("parse:") for key in calls)


async def test_research_v1_p1_r1_replay_composes_and_completes(tmp_path):
    """The supported research showcase must execute the real graph, not baseline fallback."""
    builder, conf, routes, snapshots, calls, artifacts = composition(tmp_path)
    research_config = conf.model_copy(update={"workflow_id": "research-v1"})
    runner = builder.build(
        config=research_config,
        provider_routes=routes,
        pricing_snapshots=snapshots,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        cost_calculator=CostCalculator,
    )
    assert getattr(runner, "_research_graph", None) is not None

    from deepresearch.runtime import CancellationToken
    from tests.integration.replay.test_baseline_graph import MemoryEventSink

    result = await runner.run(
        run_id="research-v1-run",
        thread_id="research-v1-thread",
        config=research_config,
        checkpoint=None,
        emit=MemoryEventSink(),
        cancellation_token=CancellationToken(),
    )

    assert result.status == "completed", result.error_code
    assert result.report_artifact_id is not None
    assert result.evidence_graph_artifact_id is not None
    assert result.manifest_artifact_id is not None
    report = artifacts.get_bytes(result.report_artifact_id).decode("utf-8")
    assert report.strip()
    # Deterministic claim verification is an internal research stage; no
    # additional model/provider calls are allowed beyond the shared baseline.
    assert not any(key.startswith(("claims:", "judge:")) for key in calls)


async def test_research_v1_durable_event_recovery_preserves_research_state(tmp_path):
    builder, conf, routes, snapshots, _calls, _artifacts = composition(tmp_path)
    research_config = conf.model_copy(update={"workflow_id": "research-v1"})
    saver = InMemorySaver(serde=checkpoint_serializer())
    runner = builder.build(
        config=research_config,
        provider_routes=routes,
        pricing_snapshots=snapshots,
        checkpointer=saver,
        cost_calculator=CostCalculator,
    )
    clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
    runner._runtime_hooks = BaselineRuntimeHooks(monotonic=clock.monotonic, utc_now=clock.utc_now)
    sink = CrashAfterDurableEventSink(crash_seq=11)
    with pytest.raises(MemoryError):
        await runner.run(
            run_id="research-recovery-run",
            thread_id="research-recovery-thread",
            config=research_config,
            checkpoint=None,
            emit=sink,
            cancellation_token=CancellationToken(),
        )
    saved = await saver.aget_tuple(
        {"configurable": {"thread_id": "research-recovery-thread", "checkpoint_ns": ""}}
    )
    assert saved is not None
    result = await runner.run(
        run_id="research-recovery-run",
        thread_id="research-recovery-thread",
        config=research_config,
        checkpoint=checkpoint_ref_from_tuple(saved),
        emit=sink,
        cancellation_token=CancellationToken(),
    )
    assert result.status == "completed", result.error_code
    assert [event.node for event in sink.calls].count("ExtractClaims") == 2
    assert {event.node for event in sink.calls[-4:]} >= {
        "VerifyClaims",
        "FinalizeCitations",
        "PersistResults",
    }


@pytest.mark.parametrize("missing", ["complete", "structured", "parse"])
def test_missing_audited_pricing_is_rejected_before_any_calls(tmp_path, missing):
    builder, conf, routes, snapshots, calls, _ = composition(tmp_path)
    with pytest.raises(ProviderProfileDrift):
        builder.build(
            config=conf,
            provider_routes=routes,
            pricing_snapshots=tuple(item for item in snapshots if item.endpoint_type != missing),
            checkpointer=InMemorySaver(),
            cost_calculator=CostCalculator,
        )
    assert not calls


def test_conflicting_model_endpoint_rates_and_nonzero_tool_rates_fail_preflight(tmp_path):
    builder, conf, routes, snapshots, calls, _ = composition(tmp_path)
    for endpoint in ("structured", "parse", "embed"):
        changed = tuple(
            item.model_copy(update={"input_tokens_per_million_usd": Decimal(2)})
            if item.endpoint_type == endpoint
            else item
            for item in snapshots
        )
        with pytest.raises(ProviderProfileDrift):
            builder.build(
                config=conf,
                provider_routes=routes,
                pricing_snapshots=changed,
                checkpointer=InMemorySaver(),
                cost_calculator=CostCalculator,
            )
    assert not calls


@pytest.mark.parametrize(
    "parameter,value",
    [("temperature", 0.4), ("seed", 7), ("max_output_tokens", 256), ("timeout_seconds", 5)],
)
def test_unsupported_model_overrides_rejected_before_cache_or_provider(parameter, value):
    with pytest.raises(ValidationError):
        FrozenProviderRoute.model_validate(
            route(
                "model",
                CountingOfflineModel(_offline_plan(), Counter()),
                parameters={parameter: value},
            )
        )


async def test_dimension_changes_actual_embedding_and_snapshot_identity():
    from deepresearch.providers.embeddings import DeterministicHashTextEmbedder

    constructor = default_provider_constructors()["deterministic-hash"]
    configured = FrozenProviderRoute.model_validate(
        route("embed", DeterministicHashTextEmbedder(), parameters={"dimension": 8})
    )
    adapter = constructor(configured, None, no_op_host_slot)
    vectors = await adapter.embed(
        ("same text",), deadline=time.monotonic() + 10, cancellation_token=CancellationToken()
    )
    assert len(vectors[0]) == 8
    assert adapter.snapshot_sha256 == hashlib.sha256(b"deterministic-hash-v1:1:8").hexdigest()


async def test_real_replay_adapter_identity_and_pricing_match_and_mismatch_fails(tmp_path):
    from deepresearch.providers import ModelMessage, ModelRequest
    from deepresearch.providers.recording import RecordingModelProvider, ReplayBundleWriter

    delegate = CountingOfflineModel(_offline_plan(), Counter())
    writer = ReplayBundleWriter.create(tmp_path / "bundle", run_id="recorded")
    writer.register_provider(
        "model",
        provider_id=delegate.provider_id,
        model_id=delegate.model_id,
        model_revision=delegate.model_revision,
    )
    recorded = RecordingModelProvider(delegate, writer)
    request = ModelRequest(
        model_id=delegate.model_id,
        messages=(ModelMessage(role="user", content="test"),),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=128,
        prompt_version="fixed-planner-v1",
        system_prompt_hash="a" * 64,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )
    await recorded.complete(
        request, deadline=time.monotonic() + 10, cancellation_token=CancellationToken()
    )
    await writer.finalize()
    constructor = default_provider_constructors()["replay"]
    frozen = FrozenProviderRoute.model_validate(
        route("model", delegate, parameters={"bundle_path": str(tmp_path / "bundle")})
    )
    replay = constructor(frozen, None, no_op_host_slot)
    assert replay.provider_id == frozen.provider_id
    result = await replay.complete(
        request, deadline=time.monotonic() + 10, cancellation_token=CancellationToken()
    )
    conf = config()
    routes = freeze(conf, [frozen.model_dump(mode="json")])
    snapshots = tuple(
        pricing(delegate.provider_id, endpoint, delegate.model_id, "1")
        for endpoint in ("complete", "structured")
    )
    resolver = _SnapshotCostResolver(routes, snapshots, CostCalculator)
    assert resolver.resolve_cost(
        operation="model",
        provider_id=replay.provider_id,
        model_id=delegate.model_id,
        outcome="success",
        usage=result.usage,
    ) == Decimal("0.000020")
    mismatched = FrozenProviderRoute.model_validate(
        {**frozen.model_dump(mode="json"), "provider_id": "replay"}
    )
    with pytest.raises(ProviderProfileDrift):
        constructor(mismatched, None, no_op_host_slot)


async def test_priced_baseline_reexecutes_recorded_model_through_real_replay_adapter(tmp_path):
    from deepresearch.providers.recording import RecordingModelProvider, ReplayBundleWriter
    from deepresearch.workflow.runner import BaselineRuntimeHooks

    writer = ReplayBundleWriter.create(tmp_path / "recorded-model", run_id="recorded-baseline")
    totals = []
    for replaying in (False, True):
        builder, conf, routes, snapshots, calls, artifacts = composition(tmp_path / str(replaying))
        model_route = next(item for item in routes.routes if item.operation == "model")
        if not replaying:
            delegate = builder.provider_constructors[model_route.provider_id](
                model_route, None, no_op_host_slot
            )
            writer.register_provider(
                "model",
                provider_id=model_route.provider_id,
                model_id=model_route.model_id,
                model_revision=model_route.model_revision,
            )
            recorded = RecordingModelProvider(delegate, writer)
            builder.provider_constructors[model_route.provider_id] = (
                lambda route, secret, slot, provider=recorded: provider
            )
        else:
            rows = [item.model_dump(mode="json") for item in routes.routes]
            for row in rows:
                if row["operation"] == "model":
                    row["parameters"] = {"bundle_path": str(tmp_path / "recorded-model")}
            routes = freeze(conf, rows)
            builder.provider_constructors[model_route.provider_id] = (
                default_provider_constructors()["replay"]
            )
        runner = builder.build(
            config=conf,
            provider_routes=routes,
            pricing_snapshots=snapshots,
            checkpointer=InMemorySaver(serde=checkpoint_serializer()),
            cost_calculator=CostCalculator,
        )
        clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
        runner._runtime_hooks = BaselineRuntimeHooks(
            monotonic=clock.monotonic, utc_now=clock.utc_now
        )
        result = await runner.run(
            run_id=f"run-{replaying}",
            thread_id=f"thread-{replaying}",
            config=conf,
            checkpoint=None,
            emit=MemoryEventSink(),
            cancellation_token=CancellationToken(),
        )
        assert result.status == "completed", result.error_code
        manifest = RunManifest.model_validate_json(artifacts.get_bytes(result.manifest_artifact_id))
        model_calls = [call for call in manifest.provider_calls if call.operation == "model"]
        assert {call.endpoint_type for call in model_calls} == {"complete", "structured"}
        assert all(call.provider_id == model_route.provider_id for call in model_calls)
        totals.append(result.final_usage.cost_usd)
        if not replaying:
            await writer.finalize()
        else:
            assert not any(key.startswith("model:") for key in calls)
    assert totals == [Decimal("0.000060"), Decimal("0.000060")]
