import hashlib
import json
from pathlib import Path

import pytest


def test_paired_runtime_hooks_keep_manifest_wall_clock_ahead_of_monotonic(
    monkeypatch: pytest.MonkeyPatch,
):
    """Service runners must use one elapsed-time source for audit envelopes."""
    import deepresearch.workflow.runner as runner_module

    ticks = iter((100.0, 100.0, 100.0, 100.5, 100.5))
    monkeypatch.setattr(runner_module.time, "monotonic", lambda: next(ticks))
    hooks = runner_module.paired_runtime_hooks()

    start = hooks.monotonic()
    started_at = hooks.utc_now()
    finish = hooks.monotonic()
    finished_at = hooks.utc_now()

    assert finish - start == 0.5
    assert (finished_at - started_at).total_seconds() >= finish - start


def test_paired_runtime_hooks_advance_equal_monotonic_samples(
    monkeypatch: pytest.MonkeyPatch,
):
    import deepresearch.workflow.runner as runner_module

    monkeypatch.setattr(runner_module.time, "monotonic", lambda: 100.0)
    hooks = runner_module.paired_runtime_hooks()

    first, second, third = hooks.utc_now(), hooks.utc_now(), hooks.utc_now()
    assert first < second < third


def test_tool_routes_have_nonempty_pricing_keys(tmp_path):
    from deepresearch.runtime.runner_factory import (
        DefaultCoreRunnerBuilder,
        EnvCredentialResolver,
        FrozenProviderRoutes,
    )
    from deepresearch.storage import LocalArtifactStore, LocalEvidenceStore
    from tests.integration.replay.test_baseline_graph import config

    route = {
        "operation": "search",
        "provider_id": "search",
        "endpoint_type": "search",
        "model_id": None,
        "model_revision": None,
        "base_url": None,
        "credential_ref": None,
        "fallback_rank": 0,
        "parameters": {},
    }
    routes = FrozenProviderRoutes.model_validate(frozen_payload(routes=[route]))
    builder = DefaultCoreRunnerBuilder(
        provider_constructors={},
        credential_resolver=EnvCredentialResolver(frozenset()),
        artifact_store=LocalArtifactStore(tmp_path),
        evidence_store=LocalEvidenceStore(tmp_path),
    )
    assert builder.required_pricing_keys(config(), routes) == {("search", "search", "search")}


async def test_search_slot_preserves_usage_reporting():
    from types import SimpleNamespace

    from deepresearch.domain import ResourceUsage
    from deepresearch.providers import ProviderUsageResult
    from deepresearch.runtime.runner_factory import _SlottedUsageSearch, no_op_search_slot

    usage = ResourceUsage.zero().model_copy(update={"search_calls": 1, "retries": 2})

    async def search_with_usage(query, limit, filters, **kwargs):
        return ProviderUsageResult([], usage)

    wrapper = _SlottedUsageSearch(
        SimpleNamespace(provider_id="search", search_with_usage=search_with_usage),
        no_op_search_slot,
    )
    result = await wrapper.search_with_usage("query", 1, None)
    assert result.usage == usage


def frozen_payload(**updates):
    payload = {"profile_id": "replay", "execution_mode": "replay", "routes": []}
    payload.update(updates)
    payload["configuration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    ).hexdigest()
    return payload


def test_route_hash_rejects_tampering():
    from pydantic import ValidationError

    from deepresearch.runtime.runner_factory import FrozenProviderRoutes

    payload = frozen_payload()
    payload["profile_id"] = "changed"
    with pytest.raises(ValidationError):
        FrozenProviderRoutes.model_validate(payload)


def test_frozen_parameters_cannot_be_mutated_after_hashing():
    from deepresearch.runtime.runner_factory import FrozenProviderRoute

    route = FrozenProviderRoute(
        operation="search",
        provider_id="test",
        endpoint_type="model",
        model_id="m",
        model_revision="v",
        base_url=None,
        credential_ref=None,
        fallback_rank=0,
        parameters={"snapshot_id": "v1"},
    )
    with pytest.raises(TypeError):
        route.parameters["snapshot_id"] = "v2"


def test_snapshot_cost_resolver_prices_tool_identity_and_excludes_cached_input():
    from decimal import Decimal

    from deepresearch.domain import ResourceUsage
    from deepresearch.runtime.manifest import CostCalculator, PricingSnapshot
    from deepresearch.runtime.runner_factory import FrozenProviderRoutes, _SnapshotCostResolver

    route = {
        "operation": "model",
        "provider_id": "provider",
        "endpoint_type": "chat.completions",
        "model_id": "model",
        "model_revision": "v",
        "base_url": None,
        "credential_ref": None,
        "fallback_rank": 0,
        "parameters": {},
    }
    routes = FrozenProviderRoutes.model_validate(frozen_payload(routes=[route]))
    snapshot = PricingSnapshot(
        snapshot_id="p",
        provider_id="provider",
        endpoint_type="complete",
        model_id="model",
        effective_at="2026-08-29T00:00:00Z",
        currency="USD",
        input_tokens_per_million_usd=Decimal(1),
        output_tokens_per_million_usd=Decimal(4),
        cached_tokens_per_million_usd=Decimal("0.25"),
        reasoning_tokens_per_million_usd=Decimal(4),
    )
    usage = ResourceUsage(
        input_tokens=100,
        cached_tokens=40,
        output_tokens=20,
        reasoning_tokens=10,
        total_tokens=130,
        search_calls=0,
        pages=0,
        retries=0,
        wall_seconds=0,
        cost_usd=None,
    )
    resolver = _SnapshotCostResolver(routes, (snapshot,), CostCalculator)
    assert resolver.resolve_cost(
        operation="model", provider_id="provider", model_id="model", outcome="success", usage=usage
    ) == Decimal("0.000190")


def test_research_capability_is_rejected_before_adapter_construction(tmp_path):
    from langgraph.checkpoint.memory import InMemorySaver

    from deepresearch.runtime.manifest import CostCalculator
    from deepresearch.runtime.runner_factory import (
        DefaultCoreRunnerBuilder,
        EnvCredentialResolver,
        FrozenProviderRoutes,
        ResearchGraphUnavailable,
    )
    from deepresearch.storage import LocalArtifactStore, LocalEvidenceStore
    from tests.integration.replay.test_baseline_graph import config

    conf = config(workflow_id="research-v1")
    routes = FrozenProviderRoutes.model_validate(
        frozen_payload(profile_id=conf.request.provider_profile_id)
    )
    builder = DefaultCoreRunnerBuilder(
        provider_constructors={},
        credential_resolver=EnvCredentialResolver(frozenset()),
        artifact_store=LocalArtifactStore(tmp_path),
        evidence_store=LocalEvidenceStore(tmp_path),
    )
    with pytest.raises(ResearchGraphUnavailable) as error:
        builder.build(
            config=conf,
            provider_routes=routes,
            pricing_snapshots=(),
            checkpointer=InMemorySaver(),
            cost_calculator=CostCalculator,
        )
    assert error.value.code == "RESEARCH_GRAPH_UNAVAILABLE"


def test_catalog_strips_secrets_and_frozen_routes_reject_credentials_in_urls(tmp_path):
    from pydantic import ValidationError

    from deepresearch.runtime.runner_factory import FileProviderRouteCatalog, FrozenProviderRoute

    route = {
        "operation": "model",
        "provider_id": "openai-compatible",
        "endpoint_type": "chat.completions",
        "model_id": "model",
        "model_revision": "v1",
        "base_url": "https://example.com/v1",
        "credential_ref": "TEST_SERVICE_TOKEN",
        "fallback_rank": 0,
        "parameters": {
            "api_key": "actual-api-key",
            "headers": {"Authorization": "Bearer actual-api-key"},
        },
        "api_key": "actual-api-key",
    }
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps({"profiles": {"live": {"execution_mode": "live", "routes": [route]}}})
    )
    frozen = FileProviderRouteCatalog.load(path).resolve("live")
    assert "actual-api-key" not in frozen.model_dump_json()
    assert frozen.routes[0].parameters == {}
    unsafe = frozen.routes[0].model_dump(mode="json")
    unsafe["base_url"] = "https://user:password@example.com"
    with pytest.raises(ValidationError):
        FrozenProviderRoute.model_validate(unsafe)


def test_pricing_catalog_rejects_duplicates_and_unknown_structure(tmp_path):
    from deepresearch.runtime.runner_factory import FilePricingCatalog

    assert FilePricingCatalog.load(None).resolve("missing") == ()
    path = tmp_path / "pricing.json"
    row = {
        "snapshot_id": "p",
        "provider_id": "p",
        "endpoint_type": "chat.completions",
        "model_id": "m",
        "effective_at": "2026-08-29T00:00:00Z",
        "currency": "USD",
        "input_tokens_per_million_usd": "1",
        "output_tokens_per_million_usd": "4",
        "cached_tokens_per_million_usd": "0.25",
        "reasoning_tokens_per_million_usd": "4",
    }
    path.write_text(json.dumps({"profiles": {"live": [row]}}))
    assert FilePricingCatalog.load(path).resolve("live")[0].model_id == "m"
    path.write_text(json.dumps({"profiles": {"live": [row, row]}}))
    with pytest.raises(ValueError):
        FilePricingCatalog.load(path)
    path.write_text('{"other": {}}')
    with pytest.raises(ValueError):
        FilePricingCatalog.load(path)


def test_factory_rejects_mode_and_profile_drift_before_builder():
    from unittest.mock import Mock

    from deepresearch.runtime.runner_factory import (
        FileProviderRouteCatalog,
        FrozenProviderRoutes,
        LangGraphServiceRunnerFactory,
        ProviderProfileDrift,
    )
    from tests.integration.replay.test_baseline_graph import config

    builder = Mock()
    factory = LangGraphServiceRunnerFactory(builder, FileProviderRouteCatalog.load(None))
    conf = config()
    for profile, mode in ((conf.request.provider_profile_id, "live"), ("wrong", "replay")):
        routes = FrozenProviderRoutes.model_validate(
            frozen_payload(profile_id=profile, execution_mode=mode)
        )
        with pytest.raises(ProviderProfileDrift):
            factory.create(
                config=conf, provider_routes=routes, pricing_snapshots=(), checkpointer=Mock()
            )
    assert builder.build.call_count == 0


def test_concrete_builder_compiles_real_core_baseline_with_same_saver(tmp_path: Path):
    from langgraph.checkpoint.memory import InMemorySaver

    from deepresearch.runtime.manifest import CostCalculator
    from deepresearch.runtime.runner_factory import (
        DefaultCoreRunnerBuilder,
        EnvCredentialResolver,
        FileProviderRouteCatalog,
        default_provider_constructors,
    )
    from deepresearch.storage import LocalArtifactStore, LocalEvidenceStore
    from deepresearch.workflow.runner import LangGraphResearchRunner
    from tests.integration.replay.test_baseline_graph import config

    source = Path("tests/fixtures/replay/baseline").resolve()
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    for file in source.iterdir():
        (bundle / file.name).write_bytes(file.read_bytes().replace(b"\r\n", b"\n"))
    snapshot = json.loads((bundle / "snapshot.json").read_text())
    conf = config()
    conf = conf.model_copy(update={"budget": conf.budget.model_copy(update={"max_cost_usd": None})})
    rows = []
    for operation, provider in snapshot["providers"].items():
        rows.append(
            {
                "operation": operation,
                "provider_id": provider["provider_id"],
                "endpoint_type": operation,
                "model_id": provider["model_id"],
                "model_revision": provider["model_revision"],
                "base_url": None,
                "credential_ref": None,
                "fallback_rank": 0,
                "parameters": {"bundle_path": str(bundle)},
            }
        )
    rows.append(
        {
            "operation": "parse",
            "provider_id": "trafilatura-html",
            "endpoint_type": "parse",
            "model_id": None,
            "model_revision": None,
            "base_url": None,
            "credential_ref": None,
            "fallback_rank": 0,
            "parameters": {},
        }
    )
    path = tmp_path / "routes.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    conf.request.provider_profile_id: {"execution_mode": "replay", "routes": rows}
                }
            }
        )
    )
    routes = FileProviderRouteCatalog.load(path).resolve(conf.request.provider_profile_id)
    registry = default_provider_constructors()
    registry.update(
        {provider["provider_id"]: registry["replay"] for provider in snapshot["providers"].values()}
    )
    builder = DefaultCoreRunnerBuilder(
        provider_constructors=registry,
        credential_resolver=EnvCredentialResolver(frozenset()),
        artifact_store=LocalArtifactStore(tmp_path),
        evidence_store=LocalEvidenceStore(tmp_path),
    )
    saver = InMemorySaver()
    runner = builder.build(
        config=conf,
        provider_routes=routes,
        pricing_snapshots=(),
        checkpointer=saver,
        cost_calculator=CostCalculator,
    )
    assert isinstance(runner, LangGraphResearchRunner)
    assert runner._baseline_graph.checkpointer is saver
    audit_composition = runner._baseline_graph._baseline_audit_composition
    assert audit_composition.replay_parent == snapshot["run_id"]


def test_replay_bound_model_stabilizes_runtime_profile_in_planner_request():
    from decimal import Decimal
    from types import SimpleNamespace

    from deepresearch.providers import ModelMessage, ModelRequest
    from deepresearch.runtime.runner_factory import FrozenProviderRoute, _BoundModel

    route = FrozenProviderRoute(
        operation="model",
        provider_id="baseline-model",
        endpoint_type="chat.completions",
        model_id="baseline-model-v1",
        model_revision="c" * 40,
        base_url=None,
        credential_ref=None,
        fallback_rank=0,
        parameters={"bundle_path": "bundle"},
    )
    request = ModelRequest(
        model_id="baseline-model-v1",
        messages=(
            ModelMessage(
                role="user",
                content=(
                    '{"access_profile":"local","provider_profile_id":"replay-default",'
                    '"question":"Compare planner strategies"}'
                ),
            ),
        ),
        temperature=Decimal(0),
        seed=0,
        max_output_tokens=128,
        prompt_version="fixed-planner-v1",
        system_prompt_hash="a" * 64,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )
    delegate = SimpleNamespace(provider_id="baseline-model")

    bound = _BoundModel(delegate, route, bind_runtime_profile=True)
    rebound = bound._request(request)

    payload = json.loads(rebound.messages[-1].content)
    assert payload["provider_profile_id"] == "runtime-profile-bound-v1"
    assert payload["access_profile"] == "local"


def test_baseline_parser_router_accepts_html_and_pdf_documents():
    from deepresearch.providers.parsers import HtmlParser, PdfParser
    from deepresearch.runtime.runner_factory import _ParserRouter

    parser = _ParserRouter((HtmlParser(), PdfParser()))

    assert parser.parser_id == "baseline-parser-router"
    assert parser.parser_version == "baseline-parser-v1"
    assert parser.supports("text/html")
    assert parser.supports("application/pdf")


def test_default_catalog_is_strict_replay_and_credentials_are_allowlisted(monkeypatch):
    from deepresearch.runtime.runner_factory import (
        EnvCredentialResolver,
        FileProviderRouteCatalog,
        ProviderProfileDrift,
    )

    frozen = FileProviderRouteCatalog.load(None).resolve("replay")
    assert frozen.execution_mode == "replay"
    assert all(route.credential_ref is None for route in frozen.routes)
    monkeypatch.setenv("TEST_SERVICE_TOKEN", "secret-value")
    assert (
        EnvCredentialResolver(frozenset({"TEST_SERVICE_TOKEN"})).resolve("TEST_SERVICE_TOKEN")
        == "secret-value"
    )
    with pytest.raises(ProviderProfileDrift):
        EnvCredentialResolver(frozenset()).resolve("TEST_SERVICE_TOKEN")
