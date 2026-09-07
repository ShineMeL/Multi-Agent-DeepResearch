"""LiveDRBench adapter.

Only the hash-verified local representation is consumed.  The adapter does
not contain a network client and therefore cannot silently turn a missing
dataset into a live request.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from benchmarks.datasets.models import TaskCategory

from .base import BaseExternalAdapter, BenchmarkName


class LiveDRBenchAdapter(BaseExternalAdapter):
    benchmark: BenchmarkName = "livedrbench"
    expected_count = 10
    supported_metric_names = ("citation_precision", "evidence_recall")

    @property
    def runtime_category(self) -> TaskCategory:
        return TaskCategory.TECHNICAL_SURVEY

    def _eligible(self, item: Mapping[str, object]) -> bool:
        """Select CS prior-art/dataset-discovery records when labels exist.

        Third-party exports use several names for this field.  A fixture that
        omits the optional label is still eligible: the lock and frozen
        snapshot, rather than an invented local label, remain authoritative.
        Explicitly unrelated records are rejected.
        """

        value: Any = item.get("task_type", item.get("category", item.get("domain")))
        if value is None:
            return True
        text = str(value).casefold()
        return not any(token in text for token in ("unrelated", "non-cs", "non_cs"))


Adapter = LiveDRBenchAdapter

__all__ = ["Adapter", "LiveDRBenchAdapter"]
