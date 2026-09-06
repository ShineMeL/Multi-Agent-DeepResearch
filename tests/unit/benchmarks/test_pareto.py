from math import isclose

import pytest

from benchmarks.evaluators.pareto import QualityCost, analyze_pareto, pareto_dominance
from tests.unit.benchmarks.test_statistics import record


@pytest.mark.parametrize(
    "quality,cost,want",
    [
        (0.8, 1, False),
        (0.9, 1, True),
        (0.8, 0.9, True),
        (0.9, 0.9, True),
        (0.7, 0.9, False),
        (0.9, 1.1, False),
    ],
)
def test_strict_pareto_requires_both_weak_inequalities_and_one_strict(
    quality: float,
    cost: float,
    want: bool,
):
    assert (
        pareto_dominance(
            baseline=QualityCost(0.8, 1), candidate=QualityCost(quality, cost)
        ).dominates
        is want
    )


@pytest.mark.parametrize("quality,cost", [(float("nan"), 1), (0.5, float("inf")), (0.5, -1)])
def test_pareto_rejects_invalid_points(quality: float, cost: float):
    with pytest.raises(ValueError):
        QualityCost(quality, cost)


def test_two_planes_stay_separate_and_bootstrap_preserves_joint_task_pairing():
    from benchmarks.evaluators.metrics import ratio_metric

    records = [
        record("a", "A", 0.5),
        record("a", "D", 0.9),
        record("b", "A", 0.5),
        record("b", "D", 0.3),
    ]
    # Precision increases by .4/-.2, while cost increases by .4/-.2.
    # No resample can dominate on the quality/cost plane, despite a quality gain.
    for r in records:
        r.metrics["cost_usd"] = ratio_metric(
            "cost_usd", r.metrics["citation_support_precision"].value, 1
        )
    results = analyze_pareto(records, budget_preset="medium", n_resamples=10000, seed=0)
    assert len(results) == 2
    quality, completeness = results
    assert (quality.quality_metric, quality.cost_metric) == (
        "citation_support_precision",
        "cost_usd",
    )
    assert not quality.dominates
    assert quality.bootstrap_dominance_proportion == 0
    assert (completeness.quality_metric, completeness.cost_metric) == (
        "information_completeness",
        "search_calls",
    )
    assert completeness.dominates
    assert isclose(completeness.candidate.quality, 0.6)
    assert 0.73 < completeness.bootstrap_dominance_proportion < 0.77
    assert results == analyze_pareto(
        records[::-1], budget_preset="medium", n_resamples=10000, seed=0
    )


def test_equal_points_never_bootstrap_dominate_and_missing_cost_is_rejected():
    from benchmarks.evaluators.metrics import ratio_metric

    records = [record("t", "A", 0.5), record("t", "D", 0.5)]
    with pytest.raises(ValueError, match="missing metric"):
        analyze_pareto(records, budget_preset="medium")
    for r in records:
        r.metrics["cost_usd"] = ratio_metric("cost_usd", 1, 1)
    for result in analyze_pareto(records, budget_preset="medium"):
        assert not result.dominates
        assert result.bootstrap_dominance_proportion == 0
