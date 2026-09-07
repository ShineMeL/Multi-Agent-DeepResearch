from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from math import isclose
from typing import Literal

import pytest
from pydantic import ValidationError

from benchmarks.datasets.models import AnnotatedQuestion, GoldInformationNeed
from benchmarks.evaluators.metrics import (
    EvaluatedClaim,
    EvaluatedInformationNeed,
    MetricNote,
    MetricValue,
    backtracking_gain,
    citation_coverage,
    citation_support_precision,
    claim_coverage_at_k,
    evaluate_quality_metrics,
    execution_adherence,
    independent_source_coverage,
    information_completeness,
    marginal_utility_per_search,
    mean_reciprocal_rank,
    ndcg_at_k,
    query_redundancy,
    ratio_metric,
    recall_at_k,
    stop_calibration,
    summarize_efficiency,
    unsupported_claim_rate,
)
from deepresearch.domain import Claim, ClaimEvidenceLink, ResourceUsage, RunBudget, RunEvent
from deepresearch.runtime.manifest import (
    NodeExecutionRecord,
    PricingSnapshot,
    ProviderCallRecord,
    RunManifest,
)
from tests.unit.benchmarks.test_models import _question  # pyright: ignore[reportPrivateUsage]
from tests.unit.runtime.test_manifest import _manifest  # pyright: ignore[reportPrivateUsage]


def evaluated_claim(
    claim_id: str,
    *,
    factual: bool = True,
    requires_evidence: bool = True,
    relations: tuple[Literal["support", "contradict", "context", "insufficient"] | None, ...] = (),
) -> EvaluatedClaim:
    return EvaluatedClaim(
        claim=Claim(
            claim_id=claim_id,
            text="Atomic claim",
            claim_type="fact",
            entities=(),
            numbers=(),
            qualifiers=(),
            report_section="findings",
            verification_status="supported",
        ),
        is_factual=factual,
        requires_evidence=requires_evidence,
        citations=tuple(
            None
            if relation is None
            else ClaimEvidenceLink(
                claim_id=claim_id,
                evidence_id=f"e{i}",
                relation=relation,
                entailment_score=1.0,
                relevance_score=1.0,
                judge_model="judge-v1",
                prompt_version="v1",
                decision_code="JUDGED",
            )
            for i, relation in enumerate(relations)
        ),
    )


def evaluated_need(
    need_id: str,
    *,
    importance: float,
    claims: tuple[EvaluatedClaim, ...] = (),
    families: dict[str, str] | None = None,
    required: int = 1,
) -> EvaluatedInformationNeed:
    return EvaluatedInformationNeed(
        need=GoldInformationNeed(
            need_id=need_id,
            text="Private gold need",
            importance=importance,
            acceptable_claim_ids=["c1"],
        ),
        claims=claims,
        source_family_by_evidence_id=families or {},
        required_independent_sources=required,
    )


def test_rank_metrics_use_grade_two_and_exponential_dcg() -> None:
    grades = {"e1": 3, "e2": 1, "e3": 2}
    ranked = ["e2", "e1", "e3"]
    assert recall_at_k(ranked, grades, k=2) == 0.5
    assert mean_reciprocal_rank(ranked, grades) == 0.5
    assert isclose(ndcg_at_k(ranked, grades, k=3), 0.7363636171343382)
    assert recall_at_k(["e1", "e1"], grades, k=2) == 0.5
    assert recall_at_k([], {}, 0) == 0.0
    assert mean_reciprocal_rank(["unknown"], grades) == 0.0
    assert ndcg_at_k(["unknown"], {}, 2) == 0.0


def test_claim_coverage_weights_gold_support_evidence_only() -> None:
    question = AnnotatedQuestion.model_validate(_question())
    assert claim_coverage_at_k(["ev-1"], question.gold_claim_links, {"claim-1": 0.8}, 1) == 1
    assert claim_coverage_at_k(["unknown"], question.gold_claim_links, {"claim-1": 0.8}, 1) == 0
    assert claim_coverage_at_k([], [], {}, 1) == 0


def test_citation_metrics_filter_nonfactual_and_count_unknown_as_unsupported() -> None:
    claims = [
        evaluated_claim("c1", relations=("support", "context", "contradict", "insufficient", None)),
        evaluated_claim("c2"),
        evaluated_claim("c3", factual=False, relations=("support",)),
        evaluated_claim("c4", requires_evidence=False),
    ]
    assert isclose(citation_support_precision(claims), 0.2)
    assert citation_coverage(claims) == 0.5
    assert isclose(unsupported_claim_rate(claims), 2 / 3)
    assert citation_support_precision([]) == 0
    assert citation_coverage([]) == 0
    assert unsupported_claim_rate([]) == 0


def test_completeness_requires_acceptable_claim_and_verified_evidence() -> None:
    needs = [
        evaluated_need(
            "n1", importance=0.8, claims=(evaluated_claim("c1", relations=("support",)),)
        ),
        evaluated_need(
            "n2", importance=0.2, claims=(evaluated_claim("wrong", relations=("support",)),)
        ),
    ]
    assert information_completeness(needs) == 0.8
    assert (
        information_completeness(
            [evaluated_need("n", importance=1, claims=(evaluated_claim("c1"),))]
        )
        == 0
    )
    assert information_completeness([]) == 0


def test_independent_sources_deduplicates_families_and_requires_satisfaction() -> None:
    claim = evaluated_claim("c1", relations=("support", "support", "context"))
    needs = [
        evaluated_need(
            "n1",
            importance=0.8,
            claims=(claim,),
            required=2,
            families={"e0": "family1", "e1": "family1", "e2": "family2"},
        ),
        evaluated_need(
            "n2",
            importance=0.2,
            claims=(claim,),
            required=2,
            families={"e0": "family1", "e1": "family2"},
        ),
        evaluated_need("n3", importance=0),
    ]
    assert isclose(independent_source_coverage(needs), 1 / 3)
    assert independent_source_coverage([]) == 0


def test_efficiency_scalar_formulas_and_separate_stop_labels() -> None:
    assert (
        query_redundancy(removed_queries=2, duplicate_executed_queries=1, proposed_queries=6) == 0.5
    )
    assert execution_adherence(completed_needs=2, scheduled_nonblocked_needs=4) == 0.5
    assert isclose(marginal_utility_per_search(0.2, 0.8, executed_searches=3), 0.2)
    assert marginal_utility_per_search(0.8, 0.2, executed_searches=3) == 0
    assert backtracking_gain([0.1, 0.3, 0.5, 0.8], first_directed_replan_index=2) == 0.5
    assert backtracking_gain([0.1, 0.3], first_directed_replan_index=None) == 0
    question = AnnotatedQuestion.model_validate(_question())
    assert stop_calibration("SUFFICIENT", True, question) == (1.0, 0.0)
    assert stop_calibration("BLOCKED", False, question) == (0.0, 1.0)
    assert query_redundancy(0, 0, 0) == 0
    assert execution_adherence(0, 0) == 0
    assert marginal_utility_per_search(0, 1, 0) == 0


@pytest.mark.parametrize(
    "field,value",
    [
        ("value", float("nan")),
        ("numerator", float("inf")),
        ("denominator", float("-inf")),
        ("denominator", -1),
        ("denominator", 0),
    ],
)
def test_metric_value_rejects_invalid_numbers_and_undocumented_zero(
    field: str, value: float
) -> None:
    payload: dict[str, object] = {
        "name": "metric",
        "value": 0.5,
        "numerator": 1,
        "denominator": 2,
        "version": "v1",
    }
    payload[field] = value
    with pytest.raises(ValidationError):
        MetricValue.model_validate(payload)


def test_ratio_metric_preserves_raw_counts_and_structured_zero_note() -> None:
    metric = ratio_metric("quality_per_search", 0.8, 4)
    assert (metric.value, metric.numerator, metric.denominator) == (0.2, 0.8, 4)
    zero = ratio_metric("quality_per_search", 0.8, 0)
    assert zero.value == 0
    assert zero.notes == (MetricNote(code="ZERO_DENOMINATOR", fields={"denominator": 0.0}),)
    assert "Private gold" not in zero.model_dump_json()


def test_quality_entrypoint_preserves_denominators_without_serializing_gold() -> None:
    claim = evaluated_claim("c1", relations=("support", None))
    needs = [
        evaluated_need("n1", importance=0.8, claims=(claim,), families={"e0": "family1"}),
        evaluated_need("n2", importance=0.2),
    ]
    metrics = evaluate_quality_metrics(
        ranked_ids=["e1", "e0"],
        graded_relevance={"e0": 3, "e1": 1},
        k=2,
        claims=[claim, evaluated_claim("c2")],
        needs=needs,
        gold_claim_links=[],
        importance_by_claim_id={},
    )
    by_name = {m.name: m for m in metrics}
    precision = by_name["citation_support_precision"]
    assert (precision.value, precision.numerator, precision.denominator) == (0.5, 1, 2)
    assert by_name["information_completeness"].value == 0.8
    assert by_name["mean_reciprocal_rank"].value == 0.5
    assert by_name["claim_coverage_at_k"].notes[0].code == "ZERO_DENOMINATOR"
    assert all("Private gold" not in m.model_dump_json() for m in metrics)


def _efficiency_fixture() -> tuple[RunManifest, tuple[RunEvent, ...]]:
    start = datetime(2026, 8, 29, tzinfo=UTC)
    zero = ResourceUsage.zero()
    calls = tuple(
        ProviderCallRecord(
            operation="search",
            node="Tool",
            provider_id="search",
            endpoint_type="search",
            request_sha256="1" * 64,
            snapshot_id="snapshot",
            normalized_query="query",
            locale="en",
            time_policy="frozen",
            complete_parameters={"filters": None, "limit": 5},
            started_at=start + timedelta(milliseconds=offset),
            finished_at=start + timedelta(milliseconds=offset + latency),
            latency_ms=latency,
            attempt=i + 1,
            cache_hit=False,
            outcome_code="TIMEOUT" if i == 0 else "SUCCESS",
            usage=zero.model_copy(update={"search_calls": 1, "retries": int(i > 0)}),
        )
        for i, (offset, latency) in enumerate([(0, 100), (100, 300)])
    )
    tool_usage = zero.model_copy(update={"search_calls": 2, "retries": 1})
    writer_usage = zero.model_copy(
        update={
            "input_tokens": 100,
            "output_tokens": 20,
            "reasoning_tokens": 10,
            "total_tokens": 130,
        }
    )
    events = tuple(
        RunEvent(
            seq=i,
            run_id="run-1",
            timestamp=start + timedelta(seconds=i),
            node="Tool",
            kind="run_started" if i == 0 else "run_completed",
            status="running" if i == 0 else "completed",
            public_payload={},
            usage_delta=zero,
            artifact_ids=(),
        )
        for i in range(2)
    )
    event_hash = hashlib.sha256(
        json.dumps(
            [e.model_dump(mode="json") for e in events],
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    manifest = RunManifest.create(
        {
            "schema_version": "run-manifest-v1",
            "run_id": "run-1",
            "thread_id": "thread-1",
            "code_commit": "a" * 40,
            "dependency_lock_sha256": "2" * 64,
            "request_sha256": "3" * 64,
            "config_sha256": "4" * 64,
            "workflow_id": "baseline-v1",
            "graph_version": "v1",
            "planner_id": "P1",
            "provider_profiles": (),
            "model_ids": (),
            "prompt_versions": {},
            "parser_versions": {},
            "ranker_id": "R0",
            "ranker_weights_version": None,
            "budget": RunBudget.preset("medium"),
            "usage": writer_usage.model_copy(
                update={"search_calls": 2, "retries": 1, "wall_seconds": 0.5}
            ),
            "usage_by_node": {"Tool": tool_usage, "Writer": writer_usage},
            "pricing_status": "unknown",
            "pricing_snapshots": (),
            "provider_calls": calls,
            "node_executions": tuple(
                NodeExecutionRecord(
                    node=node,
                    attempt=1,
                    started_at=start,
                    finished_at=start + timedelta(seconds=1),
                    latency_ms=1000,
                    status="completed",
                    input_artifact_ids=(),
                    output_artifact_ids=(),
                    usage=usage,
                )
                for node, usage in [("Tool", tool_usage), ("Writer", writer_usage)]
            ),
            "parsed_artifacts": (),
            "evidence_hashes": (),
            "source_snapshot_ids": (),
            "artifact_ids": (),
            "run_event_count": len(events),
            "run_events_sha256": event_hash,
            "seed_supported": False,
            "cache_hit_count": 0,
            "stop_reason": "SUFFICIENT",
            "is_partial": False,
            "failure_codes": (),
            "started_at": start,
            "finished_at": start + timedelta(seconds=1),
        }
    )
    return manifest, events


def test_efficiency_uses_manifest_usage_without_double_counting_events() -> None:
    manifest, events = _efficiency_fixture()
    before = manifest.model_dump_json()
    result = summarize_efficiency(manifest, events)
    assert result.search_calls == 2
    assert result.fetch_calls == 0
    assert result.prompt_tokens_by_node == {"Tool": 0, "Writer": 100}
    assert result.completion_tokens_by_node == {"Tool": 0, "Writer": 20}
    assert result.total_tokens == 130
    assert result.p50_tool_latency_ms == 200
    assert result.p95_tool_latency_ms == 290
    assert result.wall_time_ms == 1000
    assert result.retries == 1
    assert result.failures == 1
    assert result.cost_usd is None
    assert manifest.model_dump_json() == before


def test_efficiency_rejects_incomplete_wrong_run_reordered_or_tampered_events() -> None:
    manifest, events = _efficiency_fixture()
    bad_streams = [
        events[:1],
        tuple(reversed(events)),
        (events[0].model_copy(update={"run_id": "other"}), events[1]),
        (events[0], events[1].model_copy(update={"public_payload": {"tampered": True}})),
    ]
    for stream in bad_streams:
        with pytest.raises(ValueError):
            summarize_efficiency(manifest, stream)


def test_efficiency_preserves_known_zero_cost_and_empty_tool_latency() -> None:
    manifest, events = _efficiency_fixture()
    zero = ResourceUsage.zero(cost_known=True)
    empty = RunManifest.create(
        {
            **manifest.model_dump(),
            "provider_calls": (),
            "node_executions": (),
            "usage_by_node": {},
            "usage": zero,
            "pricing_status": "estimated",
        }
    )
    result = summarize_efficiency(empty, events)
    assert result.cost_usd == Decimal(0)
    assert result.p50_tool_latency_ms == result.p95_tool_latency_ms == 0
    assert result.search_calls == result.fetch_calls == result.retries == result.failures == 0


def test_efficiency_counts_cached_fetch_and_does_not_duplicate_failed_node() -> None:
    manifest, events = _efficiency_fixture()
    zero = ResourceUsage.zero()
    fetch = ProviderCallRecord(
        operation="fetch",
        node="Tool",
        provider_id="fetch",
        endpoint_type="fetch",
        request_sha256="9" * 64,
        snapshot_id="snapshot",
        complete_parameters={
            "canonical_url": "https://example.org/",
            "fetch_policy": "default",
            "accepted_content_types": ["text/html"],
        },
        started_at=manifest.started_at + timedelta(milliseconds=400),
        finished_at=manifest.started_at + timedelta(milliseconds=500),
        latency_ms=100,
        attempt=1,
        cache_hit=True,
        outcome_code="CACHE_HIT",
        usage=zero,
    )
    changed = RunManifest.create(
        {
            **manifest.model_dump(),
            "provider_calls": (*manifest.provider_calls, fetch),
            "cache_hit_count": 1,
            "node_executions": (
                manifest.node_executions[0].model_copy(
                    update={"status": "failed", "error_code": "TIMEOUT"}
                ),
                manifest.node_executions[1].model_copy(
                    update={"status": "failed", "error_code": "SCHEMA_ERROR"}
                ),
            ),
        }
    )
    result = summarize_efficiency(changed, events)
    assert result.fetch_calls == 1
    assert result.search_calls == 2
    assert result.failures == 2
    assert result.p50_tool_latency_ms == 100
    assert result.p95_tool_latency_ms == 280


def test_efficiency_preserves_exact_decimal_charge() -> None:
    pricing = PricingSnapshot(
        snapshot_id="pricing-v1",
        provider_id="model-provider",
        endpoint_type="responses",
        model_id="model-v1",
        effective_at=datetime(2026, 8, 29, tzinfo=UTC),
        currency="USD",
        input_tokens_per_million_usd=Decimal("0.1"),
        output_tokens_per_million_usd=Decimal("0.2"),
        cached_tokens_per_million_usd=Decimal("0.02"),
        reasoning_tokens_per_million_usd=Decimal("0.2"),
    )
    _, events = _efficiency_fixture()
    manifest = _manifest(
        pricing,
        run_event_count=2,
        run_events_sha256=hashlib.sha256(
            json.dumps(
                [e.model_dump(mode="json") for e in events],
                sort_keys=True,
                ensure_ascii=False,
                separators=(",", ":"),
                allow_nan=False,
            ).encode()
        ).hexdigest(),
    )
    assert summarize_efficiency(manifest, events).cost_usd == Decimal("0.114000000")


def test_formulas_reject_invalid_inputs_instead_of_producing_nan() -> None:
    with pytest.raises(ValueError):
        recall_at_k([], {}, -1)
    with pytest.raises(ValueError):
        ndcg_at_k(["e"], {"e": 4}, 1)
    with pytest.raises(ValueError):
        claim_coverage_at_k([], [], {"c": float("nan")}, 1)
    with pytest.raises(ValueError):
        marginal_utility_per_search(float("nan"), 1, 1)
    with pytest.raises(ValueError):
        query_redundancy(2, 2, 3)
    with pytest.raises(ValueError):
        execution_adherence(2, 1)
    with pytest.raises(ValueError):
        backtracking_gain([0.1, 0.2], 0)
    with pytest.raises(ValueError):
        ratio_metric("overflow", 1e308, 1e-308)


def test_evaluated_claim_rejects_support_link_belonging_to_another_claim() -> None:
    claim = evaluated_claim("c1", relations=("support",))
    other = evaluated_claim("c2")
    with pytest.raises(ValidationError):
        EvaluatedClaim(claim=other.claim, citations=claim.citations)
