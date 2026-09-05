from __future__ import annotations

from math import isclose

from deepresearch.planning.priority import PRIORITY_EPSILON, compute_priority


def test_priority_uses_spec_weights_and_epsilon() -> None:
    result = compute_priority(
        importance=0.8,
        coverage_score=0.2,
        recent_gain=1.0,
        new_source_need=0.5,
        conflict_resolution_need=0.5,
        historical_success=0.5,
        token_fraction=0.5,
        search_fraction=0.5,
        time_fraction=0.5,
    )

    expected_gain = 0.40 * 1.0 + 0.25 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5
    estimated_cost = 0.50 * 0.5 + 0.30 * 0.5 + 0.20 * 0.5
    assert isclose(result, 0.8 * 0.8 * expected_gain / (estimated_cost + PRIORITY_EPSILON))


def test_priority_uses_neutral_defaults_without_history() -> None:
    result = compute_priority(
        importance=0.8,
        coverage_score=0.2,
        new_source_need=0.5,
        conflict_resolution_need=0.5,
        token_fraction=0.5,
        search_fraction=0.5,
        time_fraction=0.5,
    )
    neutral_gain = 0.40 * 0.5 + 0.25 * 0.5 + 0.20 * 0.5 + 0.15 * 0.5
    estimated_cost = 0.50 * 0.5 + 0.30 * 0.5 + 0.20 * 0.5
    assert isclose(result, 0.8 * 0.8 * neutral_gain / (estimated_cost + PRIORITY_EPSILON))


def test_priority_clips_feature_inputs_before_calculation() -> None:
    result = compute_priority(
        importance=2.0,
        coverage_score=-1.0,
        new_source_need=2.0,
        conflict_resolution_need=-1.0,
        token_fraction=2.0,
        search_fraction=-1.0,
        time_fraction=2.0,
        recent_gain=2.0,
        historical_success=-1.0,
    )

    expected_gain = 0.40 * 1.0 + 0.25 * 1.0 + 0.20 * 0.0 + 0.15 * 0.0
    estimated_cost = 0.50 * 1.0 + 0.30 * 0.0 + 0.20 * 1.0
    assert isclose(result, expected_gain / (estimated_cost + PRIORITY_EPSILON))
