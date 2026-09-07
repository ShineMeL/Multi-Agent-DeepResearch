"""LiveDRBench adapter.

Only the hash-verified local representation is consumed.  The adapter does
not contain a network client and therefore cannot silently turn a missing
dataset into a live request.
"""

from __future__ import annotations

from collections.abc import Mapping

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
        """Accept only the published CS prior-art/discovery task families."""

        value = item.get("task_type", item.get("category", item.get("domain")))
        if value is None:
            return False
        text = str(value).casefold().replace("_", "-")
        padded = f" {text} "
        is_cs = any(token in text for token in ("computer-science", "computer science")) or " cs " in padded
        is_supported_task = any(
            token in text
            for token in ("prior-art", "prior art", "dataset-discovery", "dataset discovery")
        )
        return is_cs and is_supported_task


Adapter = LiveDRBenchAdapter

__all__ = ["Adapter", "LiveDRBenchAdapter"]
