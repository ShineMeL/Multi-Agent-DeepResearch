"""Task-level paired inference; repetitions are averaged before resampling.

Differences are right minus left. Each bootstrap draw preserves the observed
sizes of the six canonical strata; empty strata contribute no tasks. Presets
are never pooled. Stable task/seed/category ordering makes input order irrelevant.
"""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Annotated, Literal, Self

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

from benchmarks.datasets.models import TaskCategory
from benchmarks.evaluators.metrics import MetricValue

BudgetPreset = Literal["low", "medium", "high"]
DEFAULT_SEED = 20260829


class SeedRunRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    category: TaskCategory
    variant: str
    budget_preset: BudgetPreset
    seed: Annotated[int, Field(strict=True, ge=0)] | None = None
    repeat_id: Annotated[int, Field(strict=True, ge=0)] | None = None
    metrics: dict[str, MetricValue]

    @model_validator(mode="after")
    def validate_record(self) -> Self:
        if (self.seed is None) == (self.repeat_id is None):
            raise ValueError("exactly one of seed/repeat_id must be set")
        if not self.task_id.strip() or not self.variant.strip():
            raise ValueError("task_id and variant must be nonblank")
        if any(name != metric.name for name, metric in self.metrics.items()):
            raise ValueError("metric name must match its dictionary key")
        return self


class TaskVariantAggregate(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    task_id: str
    category: TaskCategory
    variant: str
    budget_preset: BudgetPreset
    metric_name: str
    mean: float
    observed_values: tuple[float, ...]
    run_count: Annotated[int, Field(strict=True, ge=1)]

    @model_validator(mode="after")
    def validate_aggregate(self) -> Self:
        if not all(s.strip() for s in (self.task_id, self.variant, self.metric_name)):
            raise ValueError("aggregate identifiers must be nonblank")
        if len(self.observed_values) != self.run_count:
            raise ValueError("run_count must match observed_values")
        if not all(math.isfinite(v) for v in (self.mean, *self.observed_values)):
            raise ValueError("aggregate values must be finite")
        if not math.isclose(
            self.mean,
            math.fsum(v / self.run_count for v in self.observed_values),
            rel_tol=1e-12,
            abs_tol=1e-15,
        ):
            raise ValueError("mean must equal the average of observed_values")
        return self


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float
    confidence: float
    n_tasks: int
    n_resamples: int
    seed: int


def _require_record(value: object) -> None:
    if not isinstance(value, SeedRunRecord):
        raise TypeError("expected SeedRunRecord")


def _require_aggregate(value: object) -> None:
    if not isinstance(value, TaskVariantAggregate):
        raise TypeError("expected TaskVariantAggregate")


def aggregate_seeds(
    records: Sequence[SeedRunRecord], *, metric_name: str
) -> tuple[TaskVariantAggregate, ...]:
    if not records:
        raise ValueError("empty seed records")
    groups: dict[tuple[str, str, BudgetPreset], list[SeedRunRecord]] = defaultdict(list)
    seen: set[tuple[str, str, BudgetPreset, str, int | None]] = set()
    categories: dict[str, TaskCategory] = {}
    versions: set[str] = set()
    for record in records:
        _require_record(record)
        if metric_name not in record.metrics:
            raise ValueError(f"missing metric {metric_name}")
        metric = record.metrics[metric_name]
        if metric.name != metric_name:
            raise ValueError("metric name must match its dictionary key")
        key = (record.task_id, record.variant, record.budget_preset)
        identity = (
            *key,
            "seed" if record.seed is not None else "repeat",
            record.seed if record.seed is not None else record.repeat_id,
        )
        if identity in seen:
            raise ValueError("duplicate task/variant/budget/seed or repeat")
        seen.add(identity)
        if categories.setdefault(record.task_id, record.category) != record.category:
            raise ValueError("inconsistent task category")
        versions.add(metric.version)
        groups[key].append(record)
    if len(versions) != 1:
        raise ValueError("inconsistent metric version")
    result: list[TaskVariantAggregate] = []
    for (task_id, variant, budget), runs in sorted(groups.items()):
        runs.sort(
            key=lambda r: (
                "seed" if r.seed is not None else "repeat",
                r.seed if r.seed is not None else r.repeat_id or 0,
            )
        )
        values = tuple(r.metrics[metric_name].value for r in runs)
        result.append(
            TaskVariantAggregate(
                task_id=task_id,
                category=categories[task_id],
                variant=variant,
                budget_preset=budget,
                metric_name=metric_name,
                mean=math.fsum(v / len(values) for v in values),
                observed_values=values,
                run_count=len(values),
            )
        )
    return tuple(result)


def align_task_pairs(
    left: Sequence[TaskVariantAggregate],
    right: Sequence[TaskVariantAggregate],
    *,
    categories: Mapping[str, str],
    budget_preset: BudgetPreset | None = None,
) -> tuple[tuple[TaskVariantAggregate, ...], tuple[TaskVariantAggregate, ...]]:
    """Validate and order paired observations for both CI and joint Pareto draws."""
    if budget_preset is not None and budget_preset not in ("low", "medium", "high"):
        raise ValueError("invalid budget preset")
    sides: list[dict[tuple[str, BudgetPreset], TaskVariantAggregate]] = []
    for rows in (left, right):
        indexed: dict[tuple[str, BudgetPreset], TaskVariantAggregate] = {}
        for row in rows:
            _require_aggregate(row)
            if budget_preset is not None and row.budget_preset != budget_preset:
                continue
            key = (row.task_id, row.budget_preset)
            if key in indexed:
                raise ValueError("duplicate paired task IDs")
            indexed[key] = row
        if not indexed:
            raise ValueError("empty paired inputs after budget filtering")
        if len({r.variant for r in indexed.values()}) != 1:
            raise ValueError("each side must contain one variant")
        sides.append(indexed)
    first, second = sides
    if first.keys() != second.keys():
        raise ValueError("paired task IDs and budget presets must be identical")
    if len({budget for _, budget in first}) != 1:
        raise ValueError("select one budget preset; pooling is forbidden")
    if len({r.metric_name for side in sides for r in side.values()}) != 1:
        raise ValueError("paired metric names must match")
    for task, budget in first:
        try:
            category = TaskCategory(categories[task])
        except (KeyError, ValueError) as exc:
            raise ValueError("missing or invalid task category") from exc
        if (
            first[(task, budget)].category != category
            or second[(task, budget)].category != category
        ):
            raise ValueError("paired task category mismatch")
    keys = sorted(first)
    return tuple(first[key] for key in keys), tuple(second[key] for key in keys)


def stratified_resample_indices(
    rows: Sequence[TaskVariantAggregate],
    *,
    n_resamples: int,
    seed: int,
) -> NDArray[np.int64]:
    """Reuse each task draw across paired variants and both Pareto coordinates."""
    if type(n_resamples) is not int or n_resamples < 1:
        raise ValueError("n_resamples must be a positive integer")
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    if not rows:
        raise ValueError("empty task rows")
    rng = np.random.default_rng(seed)
    strata: list[NDArray[np.int64]] = []
    for category in TaskCategory:
        positions = np.array(
            [i for i, row in enumerate(rows) if row.category == category], dtype=np.int64
        )
        if positions.size:
            draws = rng.integers(
                0, len(positions), size=(n_resamples, len(positions)), dtype=np.int64
            )
            strata.append(positions[draws])
    return np.concatenate(strata, axis=1)


def paired_stratified_bootstrap(
    left: Sequence[TaskVariantAggregate],
    right: Sequence[TaskVariantAggregate],
    *,
    categories: Mapping[str, str],
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
    budget_preset: BudgetPreset | None = None,
) -> ConfidenceInterval:
    left, right = align_task_pairs(left, right, categories=categories, budget_preset=budget_preset)
    differences = np.array(
        [r.mean - l.mean for l, r in zip(left, right, strict=True)], dtype=np.float64
    )
    indices = stratified_resample_indices(left, n_resamples=n_resamples, seed=seed)
    means = np.mean(differences[indices], axis=1)
    lower, upper = np.quantile(means, [0.025, 0.975], method="linear")
    return ConfidenceInterval(
        float(np.mean(differences)), float(lower), float(upper), 0.95, len(left), n_resamples, seed
    )


def planner_non_inferiority(interval: ConfidenceInterval) -> bool:
    """The pre-registered margin is strict and fixed, in completeness units."""
    return interval.lower > -0.03


@dataclass(frozen=True)
class PlannerComparison:
    completeness: ConfidenceInterval
    non_inferior: bool
    search_calls_reduction: ConfidenceInterval | None
    query_redundancy_reduction: ConfidenceInterval | None


def _comparison(
    records: Sequence[SeedRunRecord],
    metric_name: str,
    *,
    baseline: str,
    candidate: str,
    budget_preset: BudgetPreset,
    n_resamples: int,
    seed: int,
    reduction: bool = False,
) -> ConfidenceInterval:
    rows = aggregate_seeds(
        [
            r
            for r in records
            if r.variant in (baseline, candidate) and r.budget_preset == budget_preset
        ],
        metric_name=metric_name,
    )
    left = [r for r in rows if r.variant == baseline]
    right = [r for r in rows if r.variant == candidate]
    return paired_stratified_bootstrap(
        right if reduction else left,
        left if reduction else right,
        categories={r.task_id: r.category for r in rows},
        n_resamples=n_resamples,
        seed=seed,
        budget_preset=budget_preset,
    )


def compare_rankers(
    records: Sequence[SeedRunRecord],
    *,
    budget_preset: BudgetPreset,
    baseline: str = "R1",
    candidate: str = "R2",
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """R2 minus R1 citation precision; caller supplies the sealed fixed-pool runs."""
    return _comparison(
        records,
        "citation_support_precision",
        baseline=baseline,
        candidate=candidate,
        budget_preset=budget_preset,
        n_resamples=n_resamples,
        seed=seed,
    )


def compare_planners(
    records: Sequence[SeedRunRecord],
    *,
    budget_preset: BudgetPreset,
    baseline: str = "P1",
    candidate: str = "P2",
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> PlannerComparison:
    """Only a passing completeness CI unlocks confirmatory resource reductions."""
    completeness = _comparison(
        records,
        "information_completeness",
        baseline=baseline,
        candidate=candidate,
        budget_preset=budget_preset,
        n_resamples=n_resamples,
        seed=seed,
    )
    passes = planner_non_inferiority(completeness)
    reductions = [
        _comparison(
            records,
            metric,
            baseline=baseline,
            candidate=candidate,
            budget_preset=budget_preset,
            n_resamples=n_resamples,
            seed=seed,
            reduction=True,
        )
        if passes
        else None
        for metric in ("search_calls", "query_redundancy")
    ]
    return PlannerComparison(completeness, passes, reductions[0], reductions[1])
