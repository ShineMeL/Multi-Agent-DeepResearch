from __future__ import annotations

import pytest

from deepresearch.evidence.features import EvidenceFeatures


def test_feature_scores_reject_non_finite_or_out_of_range_values() -> None:
    with pytest.raises(ValueError):
        EvidenceFeatures(
            relevance=1.1,
            support_strength=0.0,
            source_quality=0.0,
            coverage_gain=0.0,
            independence=0.0,
            freshness=0.0,
            redundancy=0.0,
            risk=0.0,
        )

