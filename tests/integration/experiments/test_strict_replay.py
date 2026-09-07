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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import cast

import pytest
from pydantic import AnyHttpUrl, JsonValue

from deepresearch.domain import (
    CoverageLedgerEntry,
    EvidenceRequirements,
    FreshnessRequirement,
    InformationNeed,
    ResearchPlan,
    ResearchScope,
    ResourceUsage,
    RunBudget,
    SubQuestion,
)
from deepresearch.planning.contracts import PlannerState
from deepresearch.planning.ledger import CoverageLedger
from deepresearch.planning.stop import BlockedNeed, evaluate_stop
from deepresearch.providers import (
    ModelMessage,
    ModelProvider,
    ModelRequest,
    ModelResult,
    ProviderError,
    ProviderUsageResult,
    SearchHit,
)
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
from deepresearch.runtime.manifest import RunManifest
from deepresearch.workflow.research_graph import result_status_for

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
    manifest: RunManifest
    manifest_path: Path


def _build_run_manifest(
    bundle: ReplayBundle,
    *,
    report_bytes: bytes,
    evaluation_bytes: bytes,
    strict: bool,
    recorded_manifest: RunManifest | None,
) -> RunManifest:
    report_artifact_id = f"sha256:{hashlib.sha256(report_bytes).hexdigest()}"
    evaluation_artifact_id = f"sha256:{hashlib.sha256(evaluation_bytes).hexdigest()}"
    if strict:
        if recorded_manifest is None:
            raise AssertionError("strict replay requires a recorded RunManifest")
        if recorded_manifest.run_id != bundle.snapshot.run_id:
            raise AssertionError("recorded manifest must identify the replay bundle")
        return recorded_manifest.model_copy(
            update={
                "run_id": "strict-replay-run-v1",
                "replay_parent": recorded_manifest.run_id,
            }
        )

    started_at = datetime(2026, 9, 7, tzinfo=UTC)
    zero_usage = ResourceUsage.zero(cost_known=True)
    return RunManifest.create(
        {
            "schema_version": "run-manifest-v1",
            "run_id": bundle.snapshot.run_id,
            "thread_id": "recorded-thread-v1",
            "code_commit": "a" * 40,
            "dependency_lock_sha256": "b" * 64,
            "request_sha256": "c" * 64,
            "config_sha256": "d" * 64,
            "workflow_id": "research-v1",
            "graph_version": "strict-replay-graph-v1",
            "planner_id": "P1",
            "provider_profiles": (),
            "model_ids": (),
            "prompt_versions": {"planner": "prompt-v1"},
            "parser_versions": {"json": "parser-v1"},
            "ranker_id": "R1",
            "ranker_weights_version": "ranker-v1",
            "budget": RunBudget.preset("medium"),
            "usage": zero_usage,
            "usage_by_node": {},
            "pricing_status": "estimated",
            "pricing_snapshots": (),
            "provider_calls": (),
            "node_executions": (),
            "parsed_artifacts": (),
            "evidence_hashes": (),
            "source_snapshot_ids": (),
            "artifact_ids": (report_artifact_id, evaluation_artifact_id),
            "run_event_count": 0,
            "run_events_sha256": hashlib.sha256(_canonical([])).hexdigest(),
            "seed": 7,
            "seed_supported": True,
            "cache_hit_count": 0,
            "stop_reason": "SUFFICIENT",
            "is_partial": False,
            "failure_codes": (),
            "replay_parent": None,
            "started_at": started_at,
            "finished_at": started_at + timedelta(seconds=1),
        }
    )


async def _run_recorded_fixture(
    bundle: ReplayBundle,
    *,
    strict: bool,
    output_root: Path,
    recorded_manifest: RunManifest | None = None,
) -> _ReplayRun:
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
    report_bytes = _canonical(report)
    evaluation_bytes = _canonical(evaluation)
    manifest = _build_run_manifest(
        bundle,
        report_bytes=report_bytes,
        evaluation_bytes=evaluation_bytes,
        strict=strict,
        recorded_manifest=recorded_manifest,
    )
    output_root.mkdir(parents=True, exist_ok=False)
    (output_root / "report.md").write_bytes(report_bytes)
    (output_root / "evaluation.json").write_bytes(evaluation_bytes)
    manifest_path = output_root / "run-manifest.json"
    manifest_path.write_bytes(_canonical(manifest.model_dump(mode="json")))
    persisted_manifest = RunManifest.model_validate_json(
        manifest_path.read_bytes(), strict=True
    )
    return _ReplayRun(
        report_bytes=report_bytes,
        evaluation_bytes=evaluation_bytes,
        manifest=persisted_manifest,
        manifest_path=manifest_path,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _read_normalized_manifest(
    path: Path,
) -> tuple[RunManifest, dict[str, object], bytes]:
    manifest_bytes = path.read_bytes()
    manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
    normalized = json.loads(_canonical(manifest.model_dump(mode="json")))
    assert manifest_bytes == _canonical(normalized)
    assert manifest.manifest_sha256 == manifest.canonical_sha256()
    return manifest, normalized, manifest_bytes


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
    recorded = await _run_recorded_fixture(
        bundle,
        strict=False,
        output_root=tmp_path / "recorded-run",
    )
    recorded_manifest, recorded_payload, recorded_manifest_bytes = (
        _read_normalized_manifest(recorded.manifest_path)
    )
    assert recorded_manifest == recorded.manifest
    assert recorded_manifest.run_id == bundle.snapshot.run_id
    replayed = await _run_recorded_fixture(
        bundle,
        strict=True,
        output_root=tmp_path / "replayed-run",
        recorded_manifest=recorded_manifest,
    )
    replayed_manifest, replayed_payload, replayed_manifest_bytes = (
        _read_normalized_manifest(replayed.manifest_path)
    )
    assert replayed_manifest == replayed.manifest

    assert replayed.report_bytes == recorded.report_bytes
    assert replayed.evaluation_bytes == recorded.evaluation_bytes
    assert recorded_manifest_bytes != replayed_manifest_bytes
    assert set(recorded_payload) == set(replayed_payload)
    differing_fields = {
        field
        for field in recorded_payload
        if recorded_payload[field] != replayed_payload[field]
    }
    assert differing_fields == {"run_id", "replay_parent", "manifest_sha256"}
    for field in recorded_payload:
        if field not in differing_fields:
            assert recorded_payload[field] == replayed_payload[field]

    assert recorded.manifest.replay_parent is None
    assert replayed.manifest.replay_parent == recorded.manifest.run_id
    assert replayed_payload["replay_parent"] == recorded_payload["run_id"]
    expected_artifacts = {
        f"sha256:{hashlib.sha256(recorded.report_bytes).hexdigest()}",
        f"sha256:{hashlib.sha256(recorded.evaluation_bytes).hexdigest()}",
    }
    assert set(recorded.manifest.artifact_ids) == expected_artifacts
    assert set(replayed.manifest.artifact_ids) == expected_artifacts
    assert json.loads((recorded_root / "snapshot.json").read_bytes())["run_id"] == (
        recorded.manifest.run_id
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


_OFFLINE_FIXTURE_NAMES = (
    "sufficient",
    "conflict",
    "plateau",
    "budget_exhausted",
    "blocked",
)

_MODEL_USAGE = ResourceUsage(
    input_tokens=2,
    output_tokens=1,
    reasoning_tokens=0,
    cached_tokens=0,
    total_tokens=3,
    search_calls=0,
    pages=0,
    retries=0,
    wall_seconds=0.011,
    cost_usd=None,
)
_SEARCH_USAGE = ResourceUsage(
    input_tokens=0,
    output_tokens=0,
    reasoning_tokens=0,
    cached_tokens=0,
    total_tokens=0,
    search_calls=1,
    pages=0,
    retries=0,
    wall_seconds=0.007,
    cost_usd=None,
)


@dataclass(frozen=True)
class _OfflineFixtureRun:
    report_bytes: bytes
    evaluation_bytes: bytes
    manifest_bytes: bytes
    stop_code: str
    is_partial: bool
    query_trace: tuple[str, ...]
    usage: ResourceUsage
    live_calls: int


class _FixtureModelProvider:
    provider_id = "fixture-model"
    model_revision = "fixture-revision-v1"

    def __init__(self, *, model_id: str, prompt_version: str, response: str) -> None:
        self._model_id = model_id
        self._prompt_version = prompt_version
        self._response = response

    async def complete(
        self,
        request: ModelRequest,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ModelResult[str]:
        del deadline
        cancellation_token.raise_if_cancelled()
        if (
            request.model_id != self._model_id
            or request.prompt_version != self._prompt_version
        ):
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="model.complete",
                public_message="fixture model request identity mismatch",
                retryable=False,
            )
        return ModelResult[str](
            output=self._response,
            usage=_MODEL_USAGE,
            provider_id=self.provider_id,
            model_id=self._model_id,
            raw_response_artifact_id="fixture-model-artifact",
        )


class _FixtureSearchProvider:
    provider_id = "fixture-search"

    def __init__(
        self,
        *,
        fixture_name: str,
        query: str,
        records: list[JsonValue],
        blocked: bool,
    ) -> None:
        self._fixture_name = fixture_name
        self._query = query
        self._records = records
        self._blocked = blocked

    def _hits(self) -> list[SearchHit]:
        hits: list[SearchHit] = []
        for rank, record in enumerate(self._records, start=1):
            if isinstance(record, str):
                evidence_id = record
                metadata: dict[str, JsonValue] = {"evidence_id": evidence_id}
            elif isinstance(record, dict):
                metadata = dict(record)
                evidence_id = metadata.get("evidence_id")
                if not isinstance(evidence_id, str) or not evidence_id:
                    raise AssertionError("fixture search record needs evidence_id")
                metadata["evidence_id"] = evidence_id
            else:
                raise TypeError("fixture search record must be a string or object")
            hits.append(
                SearchHit(
                    url=cast(
                        "AnyHttpUrl",
                        f"https://fixture.example/{self._fixture_name}/{rank}",
                    ),
                    title=f"{self._fixture_name} evidence {evidence_id}",
                    snippet=f"offline evidence {evidence_id}",
                    rank=rank,
                    provider_metadata=metadata,
                )
            )
        return hits

    async def search(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> list[SearchHit]:
        result = await self.search_with_usage(
            query,
            limit,
            filters,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return result.value

    async def search_with_usage(
        self,
        query: str,
        limit: int,
        filters: Mapping[str, JsonValue] | None,
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> ProviderUsageResult[list[SearchHit]]:
        del deadline, limit, filters
        cancellation_token.raise_if_cancelled()
        if query != self._query:
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.provider_id,
                operation="search",
                public_message="fixture search request identity mismatch",
                retryable=False,
            )
        if self._blocked:
            raise ProviderError(
                code="NETWORK",
                provider=self.provider_id,
                operation="search",
                public_message="all fixture source strategies failed",
                retryable=False,
                usage=_SEARCH_USAGE,
            )
        return ProviderUsageResult(value=self._hits(), usage=_SEARCH_USAGE)


def _fixture_jsonl(path: Path) -> list[dict[str, JsonValue]]:
    records: list[dict[str, JsonValue]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            payload = json.loads(line)
            if not isinstance(payload, dict):
                raise AssertionError(f"{path.name} record must be an object")
            records.append(cast("dict[str, JsonValue]", payload))
    return records


def _fixture_plan() -> ResearchPlan:
    return ResearchPlan(
        plan_id="offline-fixture-plan",
        scope=ResearchScope(
            included_topics=("offline stop contract",),
            excluded_topics=(),
            answer_shape="brief",
        ),
        subquestions=(
            SubQuestion(
                id="sq-1",
                question="What does the frozen fixture establish?",
                rationale_code="fixture",
                importance=0.9,
                dependencies=(),
                information_needs=(
                    InformationNeed(
                        need_id="need-1",
                        text="Frozen fixture evidence",
                        importance=0.9,
                    ),
                ),
                evidence_requirements=EvidenceRequirements(
                    min_independent_sources=2,
                    allowed_source_types=frozenset({"paper"}),
                    must_include_primary=False,
                    freshness=FreshnessRequirement(kind="none"),
                ),
                status="active",
            ),
        ),
        created_by_model="fixture-model-v1",
        prompt_version="fixture-prompt-v1",
    )


def _fixture_stop_state(
    fixture_name: str,
    *,
    query: str,
    evidence_ids: tuple[str, ...],
) -> tuple[PlannerState, tuple[BlockedNeed, ...]]:
    plan = _fixture_plan()
    if fixture_name == "sufficient":
        coverage, source_count, conflicts, gains = 0.90, 2, (), ()
    elif fixture_name == "conflict":
        coverage, source_count, conflicts, gains = 0.90, 2, ("claim-1",), (0.04, 0.03)
    elif fixture_name in {"plateau", "budget_exhausted"}:
        coverage, source_count, conflicts, gains = 0.20, 1, (), (0.04, 0.03)
    elif fixture_name == "blocked":
        coverage, source_count, conflicts, gains = 0.20, 0, (), ()
    else:
        raise AssertionError(f"unknown offline fixture {fixture_name}")

    budget = BudgetAccountant(RunBudget.preset("low"), run_scope=fixture_name).snapshot()
    if fixture_name == "budget_exhausted":
        budget = budget.model_copy(update={"exhausted": frozenset({"search_calls"})})
    ledger = CoverageLedger(
        plan,
        {
            "sq-1": CoverageLedgerEntry(
                subquestion_id="sq-1",
                coverage_score=coverage,
                independent_source_count=source_count,
                unresolved_conflict_ids=conflicts,
                uncertainty_score=1.0 - coverage,
                last_marginal_gain=gains[-1] if gains else 0.0,
                evidence_ids=evidence_ids,
                attempt_count=len(gains),
                last_decision_code="RANKED",
            ),
        },
    )
    blocked_needs = (
        (
            BlockedNeed(
                need_id="need-1",
                required_source_unavailable=True,
                alternative_strategies_exhausted=True,
                retries_used=2,
                max_retries=2,
            ),
        )
        if fixture_name == "blocked"
        else ()
    )
    return (
        PlannerState(
            plan=plan,
            ledger=ledger,
            budget_snapshot=budget,
            blocked_needs=blocked_needs,
            round_index=len(gains),
            recent_marginal_gains=gains,
            query_history=(query,),
        ),
        blocked_needs,
    )


def _usage_payload(usage: ResourceUsage) -> dict[str, object]:
    return {
        "cached_tokens": usage.cached_tokens,
        "cost_usd": None if usage.cost_usd is None else str(usage.cost_usd),
        "input_tokens": usage.input_tokens,
        "output_tokens": usage.output_tokens,
        "pages": usage.pages,
        "reasoning_tokens": usage.reasoning_tokens,
        "retries": usage.retries,
        "search_calls": usage.search_calls,
        "total_tokens": usage.total_tokens,
        "wall_seconds": usage.wall_seconds,
    }


def _sum_fixture_usage(*usages: ResourceUsage) -> ResourceUsage:
    return ResourceUsage(
        input_tokens=sum(item.input_tokens for item in usages),
        output_tokens=sum(item.output_tokens for item in usages),
        reasoning_tokens=sum(item.reasoning_tokens for item in usages),
        cached_tokens=sum(item.cached_tokens for item in usages),
        total_tokens=sum(item.total_tokens for item in usages),
        search_calls=sum(item.search_calls for item in usages),
        pages=sum(item.pages for item in usages),
        retries=sum(item.retries for item in usages),
        wall_seconds=round(sum(item.wall_seconds for item in usages), 3),
        cost_usd=None,
    )


def _search_hit_payload(hit: SearchHit) -> dict[str, JsonValue]:
    return cast("dict[str, JsonValue]", hit.model_dump(mode="json"))


def _render_fixture_report(
    fixture_name: str,
    *,
    model_output: str | None,
    hits: Sequence[SearchHit],
) -> bytes:
    title = fixture_name.replace("_", " ").title()
    answer = model_output or "No model report was published."
    evidence_lines = [
        f"- {hit.provider_metadata.get('evidence_id', 'unknown')}: "
        f"{hit.title} | {hit.snippet}"
        for hit in hits
    ]
    evidence = "\n".join(evidence_lines) or "- none"
    report = f"# {title} fixture report\n\n{answer}\n\nEvidence:\n{evidence}\n"
    return report.encode("utf-8")


async def _record_offline_fixture_bundle(
    fixture_name: str,
    *,
    model_id: str,
    prompt_version: str,
    model_output: str,
    query: str,
    search_records: list[JsonValue],
    tmp_path: Path,
) -> ReplayBundle:
    bundle_root = tmp_path / "recorded-bundle"
    writer = ReplayBundleWriter.create(bundle_root, run_id="recorded-run-v1")
    writer.configure_model_provider(
        provider_id="fixture-model",
        model_revision=_FixtureModelProvider.model_revision,
    )
    model = RecordingModelProvider(
        cast(
            "ModelProvider",
            _FixtureModelProvider(
                model_id=model_id,
                prompt_version=prompt_version,
                response=model_output,
            ),
        ),
        writer,
    )
    search = RecordingSearchProvider(
        _FixtureSearchProvider(
            fixture_name=fixture_name,
            query=query,
            records=search_records,
            blocked=fixture_name == "blocked",
        ),
        writer,
    )
    token = CancellationToken()
    if fixture_name != "blocked":
        await model.complete(
            _request(model_id=model_id).model_copy(
                update={"prompt_version": prompt_version}
            ),
            deadline=_future_deadline(),
            cancellation_token=token,
        )
    if fixture_name == "blocked":
        with pytest.raises(ProviderError):
            await search.search(
                query,
                5,
                None,
                deadline=_future_deadline(),
                cancellation_token=token,
            )
    else:
        await search.search(
            query,
            5,
            None,
            deadline=_future_deadline(),
            cancellation_token=token,
        )
    await writer.finalize()
    return ReplayBundle.load(bundle_root)


async def _execute_offline_fixture(
    fixture_name: str,
    tmp_path: Path,
    *,
    fixture_root: Path | None = None,
) -> _OfflineFixtureRun:
    if fixture_name not in _OFFLINE_FIXTURE_NAMES:
        raise AssertionError(f"unknown offline fixture {fixture_name}")
    root = EXPERIMENT_FIXTURES / fixture_name if fixture_root is None else fixture_root
    scenario = json.loads((root / "scenario.json").read_bytes())
    if scenario["name"] != fixture_name or scenario["mode"] != "strict_replay":
        raise AssertionError("fixture scenario identity is invalid")
    if scenario.get("fixture_version") != "strict-replay-fixture-v1":
        raise AssertionError("offline fixture schema version is invalid")
    expected_status = "failed" if fixture_name == "blocked" else "completed"
    if scenario.get("status") != expected_status:
        raise AssertionError("offline fixture status is invalid")
    if fixture_name == "conflict" and scenario.get("targeted_research_rounds") != 1:
        raise AssertionError("conflict fixture must use one targeted round")
    if fixture_name == "plateau" and scenario.get("marginal_gains") != [0.04, 0.03]:
        raise AssertionError("plateau fixture must contain two sub-threshold gains")
    if fixture_name == "blocked" and scenario.get("public_steps") != [
        "PROVIDER_FAILURE",
        "ALTERNATIVE_STRATEGY",
        "BLOCKED",
    ]:
        raise AssertionError("blocked fixture alternatives are not exhausted")
    model_records = _fixture_jsonl(root / "model.jsonl")
    search_records = _fixture_jsonl(root / "search.jsonl")
    parsed_records = _fixture_jsonl(root / "parsed.jsonl")
    event_records = _fixture_jsonl(root / "graph_events.jsonl")
    if len(model_records) != 1 or len(search_records) != 1 or len(parsed_records) != 1:
        raise AssertionError("offline fixture must contain one model/search/parsed record")
    model_record = model_records[0]
    search_record = search_records[0]
    query = search_record.get("query")
    model_id = model_record.get("model_id")
    prompt_version = model_record.get("prompt_version")
    model_output = model_record.get("response")
    search_items = search_record.get("records")
    if not isinstance(query, str) or not isinstance(model_id, str):
        raise TypeError("offline fixture request metadata is invalid")
    if not isinstance(prompt_version, str) or not isinstance(model_output, str):
        raise TypeError("offline fixture model metadata is invalid")
    if not isinstance(search_items, list):
        raise TypeError("offline fixture search records must be a list")

    bundle = await _record_offline_fixture_bundle(
        fixture_name,
        model_id=model_id,
        prompt_version=prompt_version,
        model_output=model_output,
        query=query,
        search_records=search_items,
        tmp_path=tmp_path,
    )
    replay_search = ReplaySearchProvider(bundle, clock=lambda: 0.0)
    replay_model = ReplayModelProvider(bundle, clock=lambda: 0.0)
    query_trace = (query,)
    token = CancellationToken()
    try:
        hits = await replay_search.search(
            query,
            5,
            None,
            deadline=30.0,
            cancellation_token=token,
        )
    except ProviderError as error:
        if fixture_name != "blocked" or error.code != "NETWORK":
            raise
        hits = []
    model_result: ModelResult[str] | None = None
    if fixture_name != "blocked":
        model_result = await replay_model.complete(
            _request(model_id=model_id).model_copy(
                update={"prompt_version": prompt_version}
            ),
            deadline=30.0,
            cancellation_token=token,
        )
    evidence_ids_list: list[str] = []
    for record in search_items:
        if isinstance(record, str):
            evidence_ids_list.append(record)
        elif isinstance(record, dict):
            evidence_id = record.get("evidence_id")
            if isinstance(evidence_id, str):
                evidence_ids_list.append(evidence_id)
    evidence_ids = tuple(evidence_ids_list)
    state, blocked_needs = _fixture_stop_state(
        fixture_name,
        query=query,
        evidence_ids=evidence_ids,
    )
    decision = evaluate_stop(state, state.budget_snapshot, blocked_needs=blocked_needs)
    if decision is None:
        raise AssertionError(f"fixture {fixture_name} did not reach a terminal stop")
    model_output = None if model_result is None else model_result.output
    hit_payloads = [_search_hit_payload(hit) for hit in hits]
    replay_evidence_ids = [
        evidence_id
        for hit in hits
        if isinstance(evidence_id := hit.provider_metadata.get("evidence_id"), str)
    ]
    report_bytes = _render_fixture_report(
        fixture_name,
        model_output=model_output,
        hits=hits,
    )
    report_artifact_id = (
        None
        if fixture_name == "blocked"
        else f"sha256:{hashlib.sha256(report_bytes).hexdigest()}"
    )
    status, is_partial = result_status_for(decision.code, report_artifact_id)
    search_usage = replay_search.last_usage
    if search_usage is None:
        raise AssertionError("replay search usage was not restored")
    model_usage = replay_model.last_usage if model_result is not None else None
    total_usage = _sum_fixture_usage(
        search_usage,
        *(item for item in (model_usage,) if item is not None),
    )
    event_trace = event_records
    evaluation = {
        "event_trace": event_trace,
        "model_output": model_output,
        "query_trace": list(query_trace),
        "report_sha256": hashlib.sha256(report_bytes).hexdigest(),
        "search_hits": hit_payloads,
        "search_trace": {
            "evidence_ids": replay_evidence_ids,
            "query": query,
            "result_count": len(hit_payloads),
        },
        "status": status,
        "stop_code": decision.code.value,
        "usage": {
            "model": None if model_usage is None else _usage_payload(model_usage),
            "search": _usage_payload(search_usage),
            "total": _usage_payload(total_usage),
        },
    }
    evaluation_bytes = _canonical(evaluation) + b"\n"
    manifest = {
        "evaluation_sha256": hashlib.sha256(evaluation_bytes).hexdigest(),
        "event_trace": event_trace,
        "is_partial": is_partial,
        "model_output": model_output,
        "query_trace": list(query_trace),
        "replay_parent": bundle.snapshot.run_id,
        "report_sha256": (
            None if report_artifact_id is None else hashlib.sha256(report_bytes).hexdigest()
        ),
        "search_evidence_ids": replay_evidence_ids,
        "stop_code": decision.code.value,
        "usage": _usage_payload(total_usage),
    }
    manifest_bytes = _canonical(manifest) + b"\n"
    if replay_model.live_calls or replay_search.live_calls:
        raise AssertionError("strict offline fixture invoked a live provider")
    return _OfflineFixtureRun(
        report_bytes=report_bytes,
        evaluation_bytes=evaluation_bytes,
        manifest_bytes=manifest_bytes,
        stop_code=decision.code.value,
        is_partial=is_partial,
        query_trace=query_trace,
        usage=total_usage,
        live_calls=replay_model.live_calls + replay_search.live_calls,
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "fixture_name", _OFFLINE_FIXTURE_NAMES,
)
async def test_offline_stop_fixtures_execute_replay_and_match_expected(
    fixture_name: str, tmp_path: Path
) -> None:
    result = await _execute_offline_fixture(fixture_name, tmp_path)
    fixture_root = EXPERIMENT_FIXTURES / fixture_name
    scenario = json.loads((fixture_root / "scenario.json").read_bytes())

    assert result.report_bytes == (fixture_root / "expected_report.md").read_bytes()
    assert result.evaluation_bytes == (
        fixture_root / "expected_evaluation.json"
    ).read_bytes()
    assert result.manifest_bytes == (fixture_root / "expected_manifest.json").read_bytes()
    assert result.stop_code == scenario["stop_code"]
    assert result.is_partial is scenario["is_partial"]
    evaluation = json.loads(result.evaluation_bytes)
    manifest = json.loads(result.manifest_bytes)
    assert evaluation["stop_code"] == result.stop_code
    assert manifest["stop_code"] == result.stop_code
    assert evaluation["query_trace"] == list(result.query_trace)
    assert manifest["query_trace"] == list(result.query_trace)
    assert evaluation["usage"]["total"] == _usage_payload(result.usage)
    assert manifest["usage"] == _usage_payload(result.usage)
    assert manifest["is_partial"] is result.is_partial
    assert manifest["replay_parent"] == "recorded-run-v1"
    assert manifest["evaluation_sha256"] == hashlib.sha256(
        result.evaluation_bytes
    ).hexdigest()
    expected_report_sha = hashlib.sha256(result.report_bytes).hexdigest()
    assert evaluation["report_sha256"] == expected_report_sha
    assert manifest["report_sha256"] == (
        None if fixture_name == "blocked" else expected_report_sha
    )
    assert evaluation["status"] == scenario["status"]
    if "public_steps" in scenario:
        assert [item["kind"] for item in evaluation["event_trace"]] == scenario[
            "public_steps"
        ]
    assert result.live_calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("tamper_kind", ("model", "search"))
async def test_legal_fixture_content_tamper_changes_replay_artifacts(
    tamper_kind: str, tmp_path: Path
) -> None:
    """Replay artifacts must depend on returned model output and search hits."""

    fixture_name = "sufficient"
    baseline_root = tmp_path / "baseline"
    baseline_root.mkdir()
    baseline = await _execute_offline_fixture(fixture_name, baseline_root)
    tampered_root = tmp_path / "tampered-fixtures"
    tampered_fixture = tampered_root / fixture_name
    shutil.copytree(EXPERIMENT_FIXTURES / fixture_name, tampered_fixture)
    if tamper_kind == "model":
        model_path = tampered_fixture / "model.jsonl"
        model_record = json.loads(model_path.read_text(encoding="utf-8").splitlines()[0])
        model_record["response"] = "a different valid fixture answer"
        model_path.write_text(
            json.dumps(model_record, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
    else:
        search_path = tampered_fixture / "search.jsonl"
        search_record = json.loads(
            search_path.read_text(encoding="utf-8").splitlines()[0]
        )
        search_record["records"] = ["evidence-alt-1", "evidence-alt-2"]
        search_path.write_text(
            json.dumps(search_record, ensure_ascii=False, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )

    tampered_root_path = tmp_path / "tampered-run"
    tampered_root_path.mkdir()
    tampered = await _execute_offline_fixture(
        fixture_name,
        tampered_root_path,
        fixture_root=tampered_fixture,
    )

    assert tampered.report_bytes != baseline.report_bytes
    assert tampered.evaluation_bytes != baseline.evaluation_bytes
    assert tampered.manifest_bytes != baseline.manifest_bytes
    fixture_root = EXPERIMENT_FIXTURES / fixture_name
    assert tampered.report_bytes != (fixture_root / "expected_report.md").read_bytes()
    assert tampered.evaluation_bytes != (
        fixture_root / "expected_evaluation.json"
    ).read_bytes()
    assert tampered.manifest_bytes != (fixture_root / "expected_manifest.json").read_bytes()


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
