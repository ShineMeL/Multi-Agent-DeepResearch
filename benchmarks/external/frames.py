"""FRAMES adapter for deterministic multi-document retrieval evaluation."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import cast

from benchmarks.datasets.models import TaskCategory

from .base import BaseExternalAdapter, BenchmarkName


class FramesAdapter(BaseExternalAdapter):
    benchmark: BenchmarkName = "frames"
    expected_count = 20
    supported_metric_names = ("evidence_ndcg", "evidence_recall")

    @property
    def runtime_category(self) -> TaskCategory:
        return TaskCategory.MULTI_HOP_HISTORY

    def _eligible(self, item: Mapping[str, object]) -> bool:
        # FRAMES is only meaningful when more than one context item is
        # locally available.  ``evidence_mapping`` is optional in simple raw
        # fixtures; the presence of two documents is the non-gold invariant.
        for key in ("documents", "context", "evidence", "sources", "passages"):
            value = item.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                return len(cast(Sequence[object], value)) >= 2
        mapping = item.get("evidence_mapping")
        if isinstance(mapping, Mapping):
            return len(cast(Mapping[object, object], mapping)) >= 2
        # A canonical evidence row can be duplicated by the fixture builder;
        # leave that decision to the verified snapshot materializer.
        return item.get("allow_single_document", False) is True


Adapter = FramesAdapter

__all__ = ["Adapter", "FramesAdapter"]
