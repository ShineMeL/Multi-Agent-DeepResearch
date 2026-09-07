"""Versioned, deterministic benchmark formulas over canonical Core artifacts.

Evaluated wrappers contain evaluator judgments, not alternative domain schemas.
Keep these inputs private; only scalar MetricValue/EfficiencySummary outputs are
intended for result artifacts. No gold text or labels are copied into outputs.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from decimal import Decimal
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from benchmarks.datasets.models import AnnotatedQuestion, GoldClaimLink, GoldInformationNeed
from deepresearch.domain import Claim, ClaimEvidenceLink, RunEvent, StopReason
from deepresearch.runtime.manifest import RunManifest

METRIC_VERSION = "benchmark-metrics-v1"


class MetricNote(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    code: str
    fields: dict[str, JsonValue] = Field(default_factory=dict)


class MetricValue(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    name: str
    value: float
    numerator: float
    denominator: float
    version: str
    notes: tuple[MetricNote, ...] = ()

    @model_validator(mode="after")
    def validate_numbers(self) -> Self:
        if not all(math.isfinite(x) for x in (self.value, self.numerator, self.denominator)):
            raise ValueError("metric fields must be finite")
        if self.denominator < 0:
            raise ValueError("denominator must be nonnegative")
        if self.denominator == 0 and not any(n.code == "ZERO_DENOMINATOR" for n in self.notes):
            raise ValueError("zero denominator requires a structured ZERO_DENOMINATOR note")
        return self


def _ratio(numerator: float, denominator: float) -> float:
    if not math.isfinite(numerator) or not math.isfinite(denominator):
        raise ValueError("ratio inputs must be finite")
    if denominator < 0:
        raise ValueError("denominator must be nonnegative")
    value = numerator / denominator if denominator else 0.0
    if not math.isfinite(value):
        raise ValueError("ratio result must be finite")
    return value


def ratio_metric(
    name: str,
    numerator: float,
    denominator: float,
    *,
    version: str = METRIC_VERSION,
) -> MetricValue:
    """Wrap a derived ratio without discarding its measured inputs."""
    return MetricValue(
        name=name,
        value=_ratio(numerator, denominator),
        numerator=numerator,
        denominator=denominator,
        version=version,
        notes=(MetricNote(code="ZERO_DENOMINATOR", fields={"denominator": 0.0}),)
        if denominator == 0
        else (),
    )


def _cutoff(k: int) -> None:
    if type(k) is not int or k < 0:
        raise ValueError("k must be a nonnegative integer")


def _grades(grades: Mapping[str, int]) -> None:
    if any(type(g) is not int or not 0 <= g <= 3 for g in grades.values()):
        raise ValueError("relevance grades must be integers from 0 to 3")


def recall_at_k(ranked_ids: Sequence[str], graded_relevance: Mapping[str, int], k: int) -> float:
    _cutoff(k)
    _grades(graded_relevance)
    relevant = {item for item, grade in graded_relevance.items() if grade >= 2}
    return _ratio(len(set(ranked_ids[:k]) & relevant), len(relevant))


def mean_reciprocal_rank(ranked_ids: Sequence[str], graded_relevance: Mapping[str, int]) -> float:
    _grades(graded_relevance)
    for rank, item in enumerate(ranked_ids, start=1):
        if graded_relevance.get(item, 0) >= 2:
            return 1.0 / rank
    return 0.0


def dcg(grades: Sequence[int]) -> float:
    if any(type(g) is not int or not 0 <= g <= 3 for g in grades):
        raise ValueError("relevance grades must be integers from 0 to 3")
    return math.fsum(
        (2**grade - 1) / math.log2(rank + 1) for rank, grade in enumerate(grades, start=1)
    )


def ndcg_at_k(ranked_ids: Sequence[str], graded_relevance: Mapping[str, int], k: int) -> float:
    _cutoff(k)
    _grades(graded_relevance)
    observed = [graded_relevance.get(item, 0) for item in ranked_ids[:k]]
    ideal = sorted(graded_relevance.values(), reverse=True)[:k]
    return _ratio(dcg(observed), dcg(ideal))


def claim_coverage_at_k(
    ranked_ids: Sequence[str],
    gold_claim_links: Sequence[GoldClaimLink],
    importance_by_claim_id: Mapping[str, float],
    k: int,
) -> float:
    """Weighted gold claims with at least one acceptable support span in top k."""
    _cutoff(k)
    for weight in importance_by_claim_id.values():
        _unit_interval(weight)
    top = set(ranked_ids[:k])
    covered = {
        claim.claim_id
        for claim in gold_claim_links
        if any(
            link.relation == "support" and link.evidence_id in top for link in claim.evidence_links
        )
    }
    return _ratio(
        math.fsum(weight for cid, weight in importance_by_claim_id.items() if cid in covered),
        math.fsum(importance_by_claim_id.values()),
    )


class EvaluatedClaim(BaseModel):
    """Private judgments attached to a canonical atomic claim.

    Each citation is one cited claim/evidence link. None represents an unknown
    or unjudged citation and remains in the precision denominator. A gold match
    is evaluator supplied; absent one, matching uses the canonical claim ID.
    """

    model_config = ConfigDict(extra="forbid", frozen=True)
    claim: Claim
    is_factual: bool = True
    requires_evidence: bool = True
    citations: tuple[ClaimEvidenceLink | None, ...] = ()
    matched_gold_claim_id: str | None = None

    @model_validator(mode="after")
    def validate_link_ownership(self) -> Self:
        if any(
            link is not None and link.claim_id != self.claim.claim_id for link in self.citations
        ):
            raise ValueError("citation claim_id must match the evaluated claim")
        return self

    @property
    def has_verified_support(self) -> bool:
        return any(link is not None and link.relation == "support" for link in self.citations)


class EvaluatedInformationNeed(BaseModel):
    """Private gold need plus canonical claim judgments and source-family IDs."""

    model_config = ConfigDict(extra="forbid", frozen=True)
    need: GoldInformationNeed
    claims: tuple[EvaluatedClaim, ...] = ()
    source_family_by_evidence_id: dict[str, str] = Field(default_factory=dict)
    required_independent_sources: Annotated[int, Field(ge=1)] = 1

    @property
    def acceptable_supported_claims(self) -> tuple[EvaluatedClaim, ...]:
        return tuple(
            c
            for c in self.claims
            if c.is_factual
            and c.has_verified_support
            and (c.matched_gold_claim_id or c.claim.claim_id) in self.need.acceptable_claim_ids
        )


def citation_support_precision(claims: Sequence[EvaluatedClaim]) -> float:
    citations = [link for c in claims if c.is_factual for link in c.citations]
    return _ratio(
        sum(link is not None and link.relation == "support" for link in citations), len(citations)
    )


def citation_coverage(claims: Sequence[EvaluatedClaim]) -> float:
    eligible = [c for c in claims if c.is_factual and c.requires_evidence]
    return _ratio(sum(c.has_verified_support for c in eligible), len(eligible))


def unsupported_claim_rate(claims: Sequence[EvaluatedClaim]) -> float:
    factual = [c for c in claims if c.is_factual]
    return _ratio(sum(not c.has_verified_support for c in factual), len(factual))


def information_completeness(needs: Sequence[EvaluatedInformationNeed]) -> float:
    return _ratio(
        math.fsum(n.need.importance for n in needs if n.acceptable_supported_claims),
        math.fsum(n.need.importance for n in needs),
    )


def independent_source_coverage(needs: Sequence[EvaluatedInformationNeed]) -> float:
    """Unweighted share of all needs meeting their distinct-family requirement."""
    return _ratio(_independent_source_count(needs), len(needs))


def _independent_source_count(needs: Sequence[EvaluatedInformationNeed]) -> int:
    count = 0
    for need in needs:
        families = {
            need.source_family_by_evidence_id[link.evidence_id]
            for claim in need.acceptable_supported_claims
            for link in claim.citations
            if link is not None
            and link.relation == "support"
            and need.source_family_by_evidence_id.get(link.evidence_id)
        }
        count += len(families) >= need.required_independent_sources
    return count


def evaluate_quality_metrics(
    *,
    ranked_ids: Sequence[str],
    graded_relevance: Mapping[str, int],
    k: int,
    claims: Sequence[EvaluatedClaim],
    needs: Sequence[EvaluatedInformationNeed],
    gold_claim_links: Sequence[GoldClaimLink],
    importance_by_claim_id: Mapping[str, float],
) -> tuple[MetricValue, ...]:
    """Emit scalar-only quality results with their actual counts/weight sums.

    One ranking yields one reciprocal-rank observation, so its denominator is
    one. Cross-run mean reciprocal rank is a later aggregation operation.
    """
    _cutoff(k)
    _grades(graded_relevance)
    for weight in importance_by_claim_id.values():
        _unit_interval(weight)
    top = set(ranked_ids[:k])
    relevant = {item for item, grade in graded_relevance.items() if grade >= 2}
    covered_claim_ids = {
        claim.claim_id
        for claim in gold_claim_links
        if any(
            link.relation == "support" and link.evidence_id in top for link in claim.evidence_links
        )
    }
    factual = [c for c in claims if c.is_factual]
    cited = [link for claim in factual for link in claim.citations]
    requiring = [c for c in factual if c.requires_evidence]
    components: dict[str, tuple[float, float]] = {
        "recall_at_k": (len(top & relevant), len(relevant)),
        "mean_reciprocal_rank": (mean_reciprocal_rank(ranked_ids, graded_relevance), 1),
        "ndcg_at_k": (
            dcg([graded_relevance.get(item, 0) for item in ranked_ids[:k]]),
            dcg(sorted(graded_relevance.values(), reverse=True)[:k]),
        ),
        "claim_coverage_at_k": (
            math.fsum(w for cid, w in importance_by_claim_id.items() if cid in covered_claim_ids),
            math.fsum(importance_by_claim_id.values()),
        ),
        "citation_support_precision": (
            sum(link is not None and link.relation == "support" for link in cited),
            len(cited),
        ),
        "citation_coverage": (sum(c.has_verified_support for c in requiring), len(requiring)),
        "unsupported_claim_rate": (sum(not c.has_verified_support for c in factual), len(factual)),
        "information_completeness": (
            math.fsum(n.need.importance for n in needs if n.acceptable_supported_claims),
            math.fsum(n.need.importance for n in needs),
        ),
        "independent_source_coverage": (_independent_source_count(needs), len(needs)),
    }
    return tuple(
        ratio_metric(name, numerator, denominator)
        for name, (numerator, denominator) in components.items()
    )


def _count(value: int) -> None:
    if type(value) is not int or value < 0:
        raise ValueError("counts must be nonnegative integers")


def query_redundancy(
    removed_queries: int, duplicate_executed_queries: int, proposed_queries: int
) -> float:
    """Removed and executed-duplicate counts are disjoint evaluator judgments."""
    for count in (removed_queries, duplicate_executed_queries, proposed_queries):
        _count(count)
    numerator = removed_queries + duplicate_executed_queries
    if numerator > proposed_queries:
        raise ValueError("redundant query count exceeds proposed queries")
    return _ratio(numerator, proposed_queries)


def execution_adherence(completed_needs: int, scheduled_nonblocked_needs: int) -> float:
    _count(completed_needs)
    _count(scheduled_nonblocked_needs)
    if completed_needs > scheduled_nonblocked_needs:
        raise ValueError("completed needs exceed scheduled nonblocked needs")
    return _ratio(completed_needs, scheduled_nonblocked_needs)


def stop_calibration(
    actual_stop_reason: StopReason | None,
    actual_is_partial: bool,
    question: AnnotatedQuestion,
) -> tuple[float, float]:
    """Return stop exact match and partial-flag accuracy, as separate scalars."""
    return (
        float(actual_stop_reason == question.expected_stop_reason),
        float(actual_is_partial == question.expected_is_partial),
    )


def _unit_interval(value: float) -> None:
    if not math.isfinite(value) or not 0 <= value <= 1:
        raise ValueError("score must be finite and in [0, 1]")


def marginal_utility_per_search(before: float, after: float, executed_searches: int) -> float:
    _unit_interval(before)
    _unit_interval(after)
    _count(executed_searches)
    return _ratio(max(after - before, 0.0), executed_searches)


def backtracking_gain(
    completeness_history: Sequence[float],
    first_directed_replan_index: int | None,
) -> float:
    """Final minus pre-replan completeness; index marks first post-replan sample.

    Unlike positive marginal utility, a regression remains a negative gain.
    No directed replan means zero. A pre-replan sample must be available.
    """
    for value in completeness_history:
        _unit_interval(value)
    if first_directed_replan_index is None:
        return 0.0
    if not 1 <= first_directed_replan_index < len(completeness_history):
        raise ValueError("replan index must have a preceding and a following sample")
    return completeness_history[-1] - completeness_history[first_directed_replan_index - 1]


class EfficiencySummary(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)
    search_calls: Annotated[int, Field(ge=0)]
    fetch_calls: Annotated[int, Field(ge=0)]
    prompt_tokens_by_node: dict[str, Annotated[int, Field(ge=0)]]
    completion_tokens_by_node: dict[str, Annotated[int, Field(ge=0)]]
    total_tokens: Annotated[int, Field(ge=0)]
    p50_tool_latency_ms: Annotated[float, Field(ge=0)]
    p95_tool_latency_ms: Annotated[float, Field(ge=0)]
    wall_time_ms: Annotated[float, Field(ge=0)]
    retries: Annotated[int, Field(ge=0)]
    failures: Annotated[int, Field(ge=0)]
    cost_usd: Annotated[Decimal | None, Field(ge=0)]


def _percentile(values: Sequence[int], quantile: float) -> float:
    """Linear interpolation at (n - 1) * q; empty populations report zero."""
    if not values:
        return 0.0
    ordered = sorted(values)
    position = (len(ordered) - 1) * quantile
    low, high = math.floor(position), math.ceil(position)
    return ordered[low] + (ordered[high] - ordered[low]) * (position - low)


def summarize_efficiency(manifest: RunManifest, events: Sequence[RunEvent]) -> EfficiencySummary:
    """Summarize a complete, hash-bound Core run without summing usage twice.

    Tool calls are search/fetch attempts, including failed attempts and cache
    hits. Tool latency covers those same records. Failures count unsuccessful
    provider attempts plus failed nodes with no failed provider attempt (so a
    propagated failure is counted once). Retries are reconciled manifest usage.
    Wall time is the elapsed run envelope, including interruptions; it is not
    the budget ledger's active time. Unknown pricing remains None.
    """
    if len(events) != manifest.run_event_count:
        raise ValueError("event count does not match manifest")
    for seq, event in enumerate(events):
        if event.seq != seq or event.run_id != manifest.run_id:
            raise ValueError("events must be contiguous from zero and belong to the manifest run")
        if not manifest.started_at <= event.timestamp <= manifest.finished_at:
            raise ValueError("event timestamp is outside the run envelope")
        if seq and events[seq - 1].timestamp > event.timestamp:
            raise ValueError("event timestamps must be chronological")
    event_bytes = json.dumps(
        [e.model_dump(mode="json") for e in events],
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    if hashlib.sha256(event_bytes).hexdigest() != manifest.run_events_sha256:
        raise ValueError("event stream hash does not match manifest")
    calls = manifest.provider_calls
    tools = [call for call in calls if call.operation in {"search", "fetch"}]
    failed_calls = [call for call in calls if call.outcome_code not in {"SUCCESS", "CACHE_HIT"}]
    node_only_failures = sum(
        node.status == "failed"
        and not any(
            call.node == node.node
            and node.started_at <= call.started_at < node.finished_at
            and call.finished_at <= node.finished_at
            for call in failed_calls
        )
        for node in manifest.node_executions
    )
    return EfficiencySummary(
        search_calls=sum(c.operation == "search" for c in calls),
        fetch_calls=sum(c.operation == "fetch" for c in calls),
        prompt_tokens_by_node={
            node: usage.input_tokens for node, usage in manifest.usage_by_node.items()
        },
        completion_tokens_by_node={
            node: usage.output_tokens for node, usage in manifest.usage_by_node.items()
        },
        total_tokens=manifest.usage.total_tokens,
        p50_tool_latency_ms=_percentile([c.latency_ms for c in tools], 0.5),
        p95_tool_latency_ms=_percentile([c.latency_ms for c in tools], 0.95),
        wall_time_ms=(manifest.finished_at - manifest.started_at).total_seconds() * 1000,
        retries=manifest.usage.retries,
        failures=len(failed_calls) + node_only_failures,
        cost_usd=manifest.usage.cost_usd,
    )
