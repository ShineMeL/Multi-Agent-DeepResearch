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
        # FRAMES requires both multi-document context and an explicit mapping
        # from evidence to the question.  ``allow_single_document`` is never a
        # substitute for either invariant.
        documents_count = 0
        for key in ("documents", "context", "evidence", "sources", "passages"):
            value = item.get(key)
            if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
                documents_count = len(cast(Sequence[object], value))
                break
        mapping = item.get("evidence_mapping")
        has_mapping = isinstance(mapping, Mapping) and bool(cast(Mapping[object, object], mapping))
        return documents_count >= 2 and has_mapping


Adapter = FramesAdapter

__all__ = ["Adapter", "FramesAdapter"]
