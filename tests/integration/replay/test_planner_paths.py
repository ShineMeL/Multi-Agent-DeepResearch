from __future__ import annotations

from deepresearch.planning.planners import P0ReActPlanner, P1FixedPlanner, P2AdaptivePlanner


def test_planner_variants_have_stable_public_variant_codes() -> None:
    assert P0ReActPlanner.variant == "P0"
    assert P1FixedPlanner.variant == "P1"
    assert P2AdaptivePlanner.variant == "P2"
