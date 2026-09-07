from __future__ import annotations

import json
from dataclasses import asdict
from math import isclose

import pytest
from pydantic import ValidationError

from benchmarks.datasets.models import TaskCategory
from benchmarks.evaluators.metrics import ratio_metric
from benchmarks.evaluators.statistics import (
    ConfidenceInterval,
    SeedRunRecord,
    TaskVariantAggregate,
    aggregate_seeds,
    compare_planners,
    compare_rankers,
    paired_stratified_bootstrap,
    planner_non_inferiority,
)


def record(task: str, variant: str, value: float, seed: int = 1) -> SeedRunRecord:
    return SeedRunRecord(
        task_id=task,
        category=TaskCategory.TECHNICAL_SURVEY,
        variant=variant,
        budget_preset="medium",
        seed=seed,
        metrics={
            name: ratio_metric(name, val, 1)
            for name, val in {
                "information_completeness": value,
                "citation_support_precision": value,
                "search_calls": 10 - value,
                "query_redundancy": 1 - value,
            }.items()
        },
    )


def aggregate(
    task: str, variant: str, value: float, category: TaskCategory = TaskCategory.TECHNICAL_SURVEY
) -> TaskVariantAggregate:
    return TaskVariantAggregate(
        task_id=task,
        category=category,
        variant=variant,
        budget_preset="medium",
        metric_name="information_completeness",
        mean=value,
        observed_values=(value,),
        run_count=1,
    )


def test_seed_averages_do_not_weight_tasks_by_repetition_count():
    values = aggregate_seeds(
        [
            record("t1", "A", 0.4),
            record("t1", "A", 0.8, 2),
            record("t1", "D", 0.7),
            record("t1", "D", 0.9, 2),
            record("t2", "A", 0.2),
            record("t2", "D", 0.3),
        ],
        metric_name="information_completeness",
    )
    assert len(values) == 4
    assert isclose(values[0].mean, 0.6)
    assert values[0].observed_values == (0.4, 0.8)
    ci = paired_stratified_bootstrap(
        [v for v in values if v.variant == "A"],
        [v for v in values if v.variant == "D"],
        categories={"t1": "technical_survey", "t2": "technical_survey"},
    )
    assert isclose(ci.estimate, 0.15)
    assert ci.n_tasks == 2
    assert ci.n_resamples == 10_000


@pytest.mark.parametrize("seed,repeat", [(None, None), (1, 1)])
def test_exactly_one_repeat_identifier(seed: int | None, repeat: int | None):
    fields = record("t", "A", 0.5).model_dump()
    fields.update(seed=seed, repeat_id=repeat)
    with pytest.raises(ValidationError, match="exactly one"):
        SeedRunRecord.model_validate(fields)


def test_repeat_ids_work_and_do_not_collide_with_seed_namespace():
    first = record("t", "A", 0.5)
    repeated = SeedRunRecord.model_validate(first.model_dump() | {"seed": None, "repeat_id": 1})
    assert (
        aggregate_seeds([first, repeated], metric_name="information_completeness")[0].run_count == 2
    )
    with pytest.raises(ValidationError):
        first.variant = "changed"


def test_aggregation_rejects_duplicate_and_missing_metrics():
    first = record("t", "A", 0.5)
    with pytest.raises(ValueError, match="duplicate"):
        aggregate_seeds([first, first], metric_name="information_completeness")
    with pytest.raises(ValueError, match="missing metric"):
        aggregate_seeds([first], metric_name="absent")
    with pytest.raises(ValueError, match="empty"):
        aggregate_seeds([], metric_name="information_completeness")


def test_aggregation_rejects_inconsistent_category_and_metric_version():
    first, second = record("t", "A", 0.5), record("t", "A", 0.7, 2)
    other = second.model_copy(update={"category": TaskCategory.BILINGUAL})
    with pytest.raises(ValueError, match="category"):
        aggregate_seeds([first, other], metric_name="information_completeness")
    metric = second.metrics["information_completeness"].model_copy(update={"version": "v2"})
    other = second.model_copy(update={"metrics": {"information_completeness": metric}})
    with pytest.raises(ValueError, match="version"):
        aggregate_seeds([first, other], metric_name="information_completeness")


def test_metric_dictionary_key_must_match_metric_name():
    with pytest.raises(ValidationError, match="metric name"):
        SeedRunRecord.model_validate(
            record("t", "A", 0.5).model_dump()
            | {
                "metrics": {"wrong": ratio_metric("actual", 1, 1)},
            }
        )


@pytest.mark.parametrize(
    "update",
    [
        {"mean": float("nan")},
        {"observed_values": (float("inf"),)},
        {"run_count": 2},
        {"mean": 0.9},
    ],
)
def test_aggregates_reject_invalid_observations(update: dict[str, object]):
    with pytest.raises(ValidationError):
        TaskVariantAggregate.model_validate(aggregate("t", "A", 0.5).model_dump() | update)


def test_pairing_rejects_different_tasks_duplicates_and_empty_inputs():
    a, d = aggregate("t", "A", 0.5), aggregate("u", "D", 0.7)
    categories = {"t": "technical_survey", "u": "technical_survey"}
    for left, right, message in [
        ([a], [d], "paired task IDs"),
        ([a, a], [a], "duplicate"),
        ([], [], "empty"),
    ]:
        with pytest.raises(ValueError, match=message):
            paired_stratified_bootstrap(left, right, categories=categories)


def test_budgets_are_filtered_explicitly_and_never_pooled():
    a, d = aggregate("t", "A", 0.5), aggregate("t", "D", 0.7)
    ah = a.model_copy(update={"budget_preset": "high"})
    dh = aggregate("t", "D", 0.1).model_copy(update={"budget_preset": "high"})
    with pytest.raises(ValueError, match="budget"):
        paired_stratified_bootstrap([a, ah], [d, dh], categories={"t": "technical_survey"})
    ci = paired_stratified_bootstrap(
        [a, ah], [d, dh], categories={"t": "technical_survey"}, budget_preset="medium"
    )
    assert isclose(ci.estimate, 0.2)
    with pytest.raises(ValueError, match="paired task IDs"):
        paired_stratified_bootstrap([a], [dh], categories={"t": "technical_survey"})


def test_pairing_rejects_category_metric_and_variant_mismatches():
    a, d = aggregate("t", "A", 0.5), aggregate("t", "D", 0.7)
    for categories in [{}, {"t": "unknown"}, {"t": "bilingual"}]:
        with pytest.raises(ValueError, match="categor"):
            paired_stratified_bootstrap([a], [d], categories=categories)
    with pytest.raises(ValueError, match="metric"):
        paired_stratified_bootstrap(
            [a],
            [d.model_copy(update={"metric_name": "other"})],
            categories={"t": "technical_survey"},
        )
    with pytest.raises(ValueError, match="variant"):
        paired_stratified_bootstrap(
            [a, aggregate("u", "B", 0.5)],
            [d, aggregate("u", "D", 0.7)],
            categories={"t": "technical_survey", "u": "technical_survey"},
        )


@pytest.mark.parametrize("count", [0, -1, True, 1.5])
def test_invalid_resample_counts(count: int):
    a = aggregate("t", "A", 0.5)
    with pytest.raises(ValueError, match="n_resamples"):
        paired_stratified_bootstrap(
            [a], [a], categories={"t": "technical_survey"}, n_resamples=count
        )


def test_six_strata_preserve_sizes_instead_of_pooling_categories():
    left = [aggregate(str(i), "A", 0, c) for i, c in enumerate(TaskCategory)]
    right = [aggregate(str(i), "D", float(i), c) for i, c in enumerate(TaskCategory)]
    # A second task in stratum zero makes this 7 tasks, not 6 equally weighted strata.
    left.append(aggregate("extra", "A", 0))
    right.append(aggregate("extra", "D", 0))
    ci = paired_stratified_bootstrap(left, right, categories={x.task_id: x.category for x in left})
    assert ci.estimate == ci.lower == ci.upper == 15 / 7


def test_linear_quantiles_and_serialized_ci_are_deterministic(capsys: pytest.CaptureFixture[str]):
    left = [aggregate("a", "A", 0), aggregate("b", "A", 0)]
    right = [aggregate("a", "D", 0), aggregate("b", "D", 1)]
    categories = {"a": "technical_survey", "b": "technical_survey"}
    ci = paired_stratified_bootstrap(left, right, categories=categories, n_resamples=4, seed=0)
    # PCG64 draws (b,b), (b,a), (a,a), (a,a): means 1, .5, 0, 0.
    assert ci == ConfidenceInterval(0.5, 0.0, 0.9624999999999999, 0.95, 2, 4, 0)
    reverse = paired_stratified_bootstrap(
        left[::-1], right[::-1], categories=categories, n_resamples=4, seed=0
    )
    serialized = json.dumps(asdict(ci), sort_keys=True)
    assert serialized == json.dumps(asdict(reverse), sort_keys=True)
    with capsys.disabled():
        print("CI_FIXTURE=" + serialized)


@pytest.mark.parametrize("lower,passes", [(-0.031, False), (-0.03, False), (-0.029, True)])
def test_planner_margin_is_strict(lower: float, passes: bool):
    assert planner_non_inferiority(ConfidenceInterval(0, lower, 0.1, 0.95, 2, 10000, 0)) is passes


def test_registered_comparisons_use_candidate_minus_baseline_and_gate_secondary_results():
    records = [
        record("t", "P1", 0.5),
        record("t", "P2", 0.6),
        record("t", "R1", 0.7),
        record("t", "R2", 0.9),
    ]
    planner = compare_planners(records, budget_preset="medium")
    assert planner.non_inferior
    assert isclose(planner.completeness.estimate, 0.1)
    assert planner.search_calls_reduction is not None
    assert isclose(planner.search_calls_reduction.estimate, 0.1)
    assert planner.query_redundancy_reduction is not None
    assert isclose(planner.query_redundancy_reduction.estimate, 0.1)
    assert isclose(compare_rankers(records, budget_preset="medium").estimate, 0.2)
    failed = compare_planners(
        [record("t", "P1", 0.5), record("t", "P2", 0.4)], budget_preset="medium"
    )
    assert not failed.non_inferior
    assert failed.search_calls_reduction is None
    assert failed.query_redundancy_reduction is None
