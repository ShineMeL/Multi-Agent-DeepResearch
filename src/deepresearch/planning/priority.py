from __future__ import annotations

from math import isfinite

PRIORITY_EPSILON = 0.05


def _clip(value: object, *, field: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{field} must be numeric")
    if not isfinite(value):
        raise ValueError(f"{field} must be finite")
    return float(min(1.0, max(0.0, value)))


def compute_priority(
    *,
    importance: float,
    coverage_score: float,
    new_source_need: float,
    conflict_resolution_need: float,
    token_fraction: float,
    search_fraction: float,
    time_fraction: float,
    recent_gain: float = 0.5,
    historical_success: float = 0.5,
) -> float:
    expected_gain = (
        0.40 * _clip(recent_gain, field="recent_gain")
        + 0.25 * _clip(new_source_need, field="new_source_need")
        + 0.20 * _clip(conflict_resolution_need, field="conflict_resolution_need")
        + 0.15 * _clip(historical_success, field="historical_success")
    )
    estimated_cost = _clip(
        0.50 * _clip(token_fraction, field="token_fraction")
        + 0.30 * _clip(search_fraction, field="search_fraction")
        + 0.20 * _clip(time_fraction, field="time_fraction"),
        field="estimated_cost",
    )
    return (
        _clip(importance, field="importance")
        * (1.0 - _clip(coverage_score, field="coverage_score"))
        * expected_gain
        / (estimated_cost + PRIORITY_EPSILON)
    )


__all__ = ["PRIORITY_EPSILON", "compute_priority"]
