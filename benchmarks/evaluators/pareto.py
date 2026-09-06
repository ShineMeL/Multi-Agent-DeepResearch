"""Separate quality/cost and completeness/search planes with joint task draws."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from benchmarks.evaluators.statistics import (
    DEFAULT_SEED,
    BudgetPreset,
    SeedRunRecord,
    aggregate_seeds,
    align_task_pairs,
    stratified_resample_indices,
)


@dataclass(frozen=True)
class QualityCost:
    quality: float
    cost: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.quality) or not math.isfinite(self.cost) or self.cost < 0:
            raise ValueError("quality and cost must be finite; cost must be nonnegative")


@dataclass(frozen=True)
class ParetoDecision:
    dominates: bool


def pareto_dominance(*, baseline: QualityCost, candidate: QualityCost) -> ParetoDecision:
    return ParetoDecision(
        candidate.quality >= baseline.quality
        and candidate.cost <= baseline.cost
        and (candidate.quality > baseline.quality or candidate.cost < baseline.cost)
    )


@dataclass(frozen=True)
class ParetoPlaneResult:
    quality_metric: str
    cost_metric: str
    baseline: QualityCost
    candidate: QualityCost
    dominates: bool
    bootstrap_dominance_proportion: float
    n_tasks: int
    n_resamples: int
    seed: int


def analyze_pareto(
    records: Sequence[SeedRunRecord],
    *,
    budget_preset: BudgetPreset,
    baseline: str = "A",
    candidate: str = "D",
    n_resamples: int = 10_000,
    seed: int = DEFAULT_SEED,
) -> tuple[ParetoPlaneResult, ...]:
    selected = [
        r
        for r in records
        if r.variant in (baseline, candidate) and r.budget_preset == budget_preset
    ]
    results: list[ParetoPlaneResult] = []
    for quality_metric, cost_metric in (
        ("citation_support_precision", "cost_usd"),
        ("information_completeness", "search_calls"),
    ):
        quality = aggregate_seeds(selected, metric_name=quality_metric)
        cost = aggregate_seeds(selected, metric_name=cost_metric)
        categories = {r.task_id: r.category for r in quality}
        qleft, qright = align_task_pairs(
            [r for r in quality if r.variant == baseline],
            [r for r in quality if r.variant == candidate],
            categories=categories,
            budget_preset=budget_preset,
        )
        cleft, cright = align_task_pairs(
            [r for r in cost if r.variant == baseline],
            [r for r in cost if r.variant == candidate],
            categories=categories,
            budget_preset=budget_preset,
        )
        # Both coordinates originate from the same validated, complete seed records.
        if any(r.mean < 0 for r in (*cleft, *cright)):
            raise ValueError("cost must be nonnegative")
        qvalues: NDArray[np.float64] = np.array(
            [[r.mean for r in qleft], [r.mean for r in qright]], dtype=np.float64
        )
        cvalues: NDArray[np.float64] = np.array(
            [[r.mean for r in cleft], [r.mean for r in cright]], dtype=np.float64
        )
        quality_means: NDArray[np.float64] = np.mean(qvalues, axis=1)
        cost_means: NDArray[np.float64] = np.mean(cvalues, axis=1)
        base = QualityCost(float(quality_means[0]), float(cost_means[0]))
        cand = QualityCost(float(quality_means[1]), float(cost_means[1]))
        indices = stratified_resample_indices(qleft, n_resamples=n_resamples, seed=seed)
        qmeans: NDArray[np.float64] = np.mean(qvalues[:, indices], axis=2)
        cmeans: NDArray[np.float64] = np.mean(cvalues[:, indices], axis=2)
        dominates: NDArray[np.bool_] = (
            (qmeans[1] >= qmeans[0])
            & (cmeans[1] <= cmeans[0])
            & ((qmeans[1] > qmeans[0]) | (cmeans[1] < cmeans[0]))
        )
        results.append(
            ParetoPlaneResult(
                quality_metric,
                cost_metric,
                base,
                cand,
                pareto_dominance(baseline=base, candidate=cand).dominates,
                float(np.mean(dominates)),
                len(qleft),
                n_resamples,
                seed,
            )
        )
    return tuple(results)
