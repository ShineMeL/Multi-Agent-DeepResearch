"""Offline strict-Replay and stop-path contracts for the formal benchmark.

The fixtures in ``tests/fixtures/experiments`` are intentionally small.  They
describe the public, hash-addressed boundary used by the real formal runner;
they do not contain private gold or require a provider process.  The tests
exercise the provider/checkpoint contracts directly so a missing replay record
can never silently turn into a live call.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

import pytest

from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.providers import ModelMessage, ModelRequest, ProviderError
from deepresearch.providers.recording import (
    RecordingModelProvider,
    RecordingSearchProvider,
    ReplayBundleWriter,
)
from deepresearch.providers.replay import ReplayModelProvider, ReplaySearchProvider
from deepresearch.providers.replay_schema import (
    REPLAY_REQUEST_SCHEMA_VERSION,
    ReplayBundle,
    canonical_request_sha256,
    model_request_payload,
)
from deepresearch.runtime import (
    BudgetAccountant,
    CancellationToken,
    ResourceEstimate,
)

ROOT = Path(__file__).resolve().parents[2]
PROVIDER_FIXTURE = ROOT / "fixtures" / "replay" / "provider_contract"
EXPERIMENT_FIXTURES = ROOT / "fixtures" / "experiments"


def _future_deadline() -> float:
    return time.monotonic() + 30.0


def _request(*, model_id: str = "fixture-model-v1") -> ModelRequest:
    from decimal import Decimal

    return ModelRequest(
        model_id=model_id,
        messages=(ModelMessage(role="user", content="Synthetic fixture question?"),),
        temperature=Decimal(0),
        seed=7,
        max_output_tokens=32,
        prompt_version="prompt-v1",
        system_prompt_hash="a" * 64,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )


@dataclass(frozen=True)
class _ReplayRun:
    report_bytes: bytes
    evaluation_bytes: bytes
    manifest_bytes: bytes
    replay_parent: str | None


async def _run_recorded_fixture(bundle: ReplayBundle, *, strict: bool) -> _ReplayRun:
    """Execute the same offline provider calls used by record/replay modes."""

    token = CancellationToken()
    model = await ReplayModelProvider(bundle).complete(
        _request(), deadline=_future_deadline(), cancellation_token=token
    )
    search = await ReplaySearchProvider(bundle).search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=token,
    )
    report = {
        "answer": model.output,
        "sources": [str(item.url) for item in search],
    }
    evaluation = {
        "citation_count": len(search),
        "model_id": model.model_id,
        "report_sha256": hashlib.sha256(_canonical(report)).hexdigest(),
    }
    manifest = {
        "run_id": "strict-replay-run-v1" if strict else "recorded-run-v1",
        "replay_parent": bundle.snapshot.run_id if strict else None,
        "report_sha256": hashlib.sha256(_canonical(report)).hexdigest(),
        "evaluation_sha256": hashlib.sha256(_canonical(evaluation)).hexdigest(),
    }
    return _ReplayRun(
        report_bytes=_canonical(report),
        evaluation_bytes=_canonical(evaluation),
        manifest_bytes=_canonical(manifest),
        replay_parent=manifest["replay_parent"],
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


@pytest.mark.asyncio
async def test_strict_replay_threads_an_injected_clock_and_restores_usage() -> None:
    """Provider deadlines are deterministic and recorded usage is observable."""

    bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    ticks = iter((10.0, 10.0, 10.0))
    clock: Callable[[], float] = lambda: next(ticks)
    provider = ReplaySearchProvider(bundle, clock=clock)

    hits = await provider.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=20.0,
        cancellation_token=CancellationToken(),
    )

    assert hits[0].provider_metadata["source_id"] == "src-1"
    assert provider.last_usage is not None
    assert provider.last_usage.search_calls == 1
    assert provider.live_calls == 0


@pytest.mark.asyncio
async def test_replay_success_then_miss_clears_usage_and_failure_accounting() -> None:
    """A failed second call must not inherit the first call's usage."""

    bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    provider = ReplaySearchProvider(bundle)
    token = CancellationToken()
    await provider.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=token,
    )
    assert provider.last_usage is not None

    with pytest.raises(ProviderError) as error:
        await provider.search(
            "unknown-query",
            5,
            {"language": "en"},
            deadline=_future_deadline(),
            cancellation_token=token,
        )

    assert error.value.code == "REPLAY_MISS"
    assert error.value.usage is None
    assert provider.last_usage is None
    assert provider.live_calls == 0


@pytest.mark.asyncio
async def test_replay_success_then_expired_call_clears_usage() -> None:
    bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    provider = ReplaySearchProvider(bundle)
    token = CancellationToken()
    await provider.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=token,
    )

    with pytest.raises(ProviderError) as error:
        await provider.search(
            "multimodal agents",
            5,
            {"language": "en"},
            deadline=0.0,
            cancellation_token=token,
        )

    assert error.value.code == "TIMEOUT"
    assert error.value.usage is None
    assert provider.last_usage is None


@pytest.mark.asyncio
async def test_record_then_strict_replay_is_byte_identical(tmp_path: Path) -> None:
    source_bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    recorded_root = tmp_path / "recorded-bundle"
    writer = ReplayBundleWriter.create(recorded_root, run_id="recorded-run-v1")
    writer.configure_model_provider(
        provider_id=source_bundle.snapshot.providers["model"].provider_id,
        model_revision=source_bundle.snapshot.providers["model"].model_revision or "",
    )
    model = RecordingModelProvider(ReplayModelProvider(source_bundle), writer)
    search = RecordingSearchProvider(ReplaySearchProvider(source_bundle), writer)
    token = CancellationToken()
    await model.complete(_request(), deadline=_future_deadline(), cancellation_token=token)
    await search.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=token,
    )
    await writer.finalize()
    bundle = ReplayBundle.load(recorded_root)
    recorded_manifest_bytes = (recorded_root / "manifest.sha256").read_bytes()
    recorded = await _run_recorded_fixture(bundle, strict=False)
    replayed = await _run_recorded_fixture(bundle, strict=True)

    assert replayed.report_bytes == recorded.report_bytes
    assert replayed.evaluation_bytes == recorded.evaluation_bytes
    assert recorded.manifest_bytes != replayed.manifest_bytes
    assert json.loads(recorded.manifest_bytes)["replay_parent"] is None
    assert replayed.replay_parent == bundle.snapshot.run_id
    assert json.loads(replayed.manifest_bytes)["replay_parent"] == bundle.snapshot.run_id
    assert recorded_manifest_bytes == (recorded_root / "manifest.sha256").read_bytes()
    assert json.loads((recorded_root / "snapshot.json").read_bytes())["run_id"] == (
        bundle.snapshot.run_id
    )


def test_replay_key_binds_provider_model_prompt_request_and_schema_versions() -> None:
    bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    provider = ReplayModelProvider(bundle)
    request = _request()
    payload = model_request_payload(request, model_revision=provider.model_revision)
    key = provider._key(  # pyright: ignore[reportPrivateUsage]
        "model.complete",
        payload,
        prompt_version=request.prompt_version,
    )

    assert key.provider_id == bundle.snapshot.providers["model"].provider_id
    assert key.prompt_version == request.prompt_version
    assert key.schema_version == REPLAY_REQUEST_SCHEMA_VERSION
    assert key.request_sha256 == canonical_request_sha256(payload)
    assert payload["model_id"] == request.model_id
    assert payload["model_revision"] == provider.model_revision


def test_checkpoint_resume_observes_cached_usage_without_double_charge() -> None:
    budget = RunBudget.preset("medium")
    usage = ResourceUsage(
        input_tokens=8,
        output_tokens=2,
        reasoning_tokens=0,
        cached_tokens=0,
        total_tokens=10,
        search_calls=1,
        pages=0,
        retries=0,
        wall_seconds=0.1,
        cost_usd=Decimal(0),
    )
    accountant = BudgetAccountant(budget, run_scope="strict-replay")
    reservation = accountant.reserve(
        ResourceEstimate(
            tokens=10,
            search_calls=1,
            wall_seconds=0.1,
            cost_usd=Decimal(0),
        ),
        node="Tool",
        idempotency_key="search-operation-v1",
    )
    charged = accountant.settle(reservation, actual=usage)
    resumed = BudgetAccountant.from_snapshot(budget, charged, run_scope="strict-replay")
    cached = resumed.reserve(
        ResourceEstimate(
            tokens=10,
            search_calls=1,
            wall_seconds=0.1,
            cost_usd=Decimal(0),
        ),
        node="Tool",
        idempotency_key="search-operation-v1:cache-hit",
    )
    observed = resumed.settle(cached, actual=usage, charge=False)

    assert observed.used_tokens == charged.used_tokens
    assert observed.used_search_calls == charged.used_search_calls
    assert observed.last_observed_usage == usage


@pytest.mark.asyncio
async def test_strict_replay_rejects_model_schema_or_identity_mismatch(
    tmp_path: Path,
) -> None:
    """A key hit is not enough when the returned model identity is wrong."""

    copied = tmp_path / "bundle"
    shutil.copytree(PROVIDER_FIXTURE, copied)
    path = copied / "model_responses.jsonl"
    record = json.loads(path.read_text(encoding="utf-8"))
    record["outcome"]["response"]["model_id"] = "different-model"
    record["outcome_sha256"] = hashlib.sha256(_canonical(record["outcome"])).hexdigest()
    path.write_bytes(_canonical(record) + b"\n")
    manifest_path = copied / "manifest.sha256"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["file_sha256"]["model_responses.jsonl"] = hashlib.sha256(
        path.read_bytes()
    ).hexdigest()
    manifest_path.write_bytes(_canonical(manifest) + b"\n")

    bundle = ReplayBundle.load(copied)
    provider = ReplayModelProvider(bundle)
    with pytest.raises(ProviderError) as error:
        await provider.complete(
            _request(),
            deadline=_future_deadline(),
            cancellation_token=CancellationToken(),
        )
    assert error.value.code == "INVALID_SNAPSHOT"
    assert provider.live_calls == 0


def test_replay_request_schema_version_mismatch_invalidates_snapshot(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(PROVIDER_FIXTURE, copied)
    path = copied / "search.jsonl"
    record = json.loads(path.read_bytes())
    record["key"]["schema_version"] = "replay-request-v999"
    record["outcome_sha256"] = hashlib.sha256(_canonical(record["outcome"])).hexdigest()
    path.write_bytes(_canonical(record) + b"\n")
    manifest_path = copied / "manifest.sha256"
    manifest = json.loads(manifest_path.read_bytes())
    manifest["file_sha256"]["search.jsonl"] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_path.write_bytes(_canonical(manifest) + b"\n")

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"
    assert "schema_version" in str(error.value.__cause__)


@pytest.mark.asyncio
async def test_prompt_version_mismatch_is_a_replay_miss_without_live_calls() -> None:
    bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    provider = ReplayModelProvider(bundle)
    with pytest.raises(ProviderError) as error:
        await provider.complete(
            _request().model_copy(update={"prompt_version": "unknown-prompt-v2"}),
            deadline=_future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "REPLAY_MISS"
    assert provider.live_calls == 0


def test_five_stop_path_fixtures_are_hash_addressed_and_private_free() -> None:
    expected = {
        "sufficient": "SUFFICIENT",
        "conflict": "PLATEAU",
        "plateau": "PLATEAU",
        "budget_exhausted": "BUDGET_EXHAUSTED",
        "blocked": "BLOCKED",
    }
    for name, stop_code in expected.items():
        root = EXPERIMENT_FIXTURES / name
        scenario_path = root / "scenario.json"
        assert scenario_path.is_file(), f"missing {name} fixture"
        scenario = json.loads(scenario_path.read_bytes())
        assert scenario["stop_code"] == stop_code
        assert set(scenario["artifact_hashes"]) == {"report", "evaluation", "manifest"}
        assert set(scenario["artifact_files"]) == {"report", "evaluation", "manifest"}
        assert all(
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and value != "0" * 64
            for value in scenario["artifact_hashes"].values()
        )
        for artifact_name, relative in scenario["artifact_files"].items():
            assert hashlib.sha256((root / relative).read_bytes()).hexdigest() == (
                scenario["artifact_hashes"][artifact_name]
            )
        for relative in ("search.jsonl", "model.jsonl", "parsed.jsonl", "graph_events.jsonl"):
            path = root / relative
            assert path.is_file(), f"missing {name}/{relative}"
            assert path.resolve().is_relative_to(root.resolve())
            assert b"private" not in path.read_bytes().lower()


def test_unknown_replay_query_is_strict_and_never_a_live_fallback() -> None:
    bundle = ReplayBundle.load(PROVIDER_FIXTURE)
    provider = ReplaySearchProvider(bundle)
    with pytest.raises(ProviderError) as error:
        # This intentionally exercises the provider boundary rather than a
        # fallback wrapper: strict replay must fail before any live provider.
        import asyncio

        asyncio.run(
            provider.search(
                "unknown-query",
                5,
                None,
                deadline=_future_deadline(),
                cancellation_token=CancellationToken(),
            )
        )
    assert error.value.code == "REPLAY_MISS"
    assert provider.live_calls == 0
