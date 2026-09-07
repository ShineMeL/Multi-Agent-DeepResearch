"""DeepResearch Bench adapter for report/citation comparison."""

from __future__ import annotations

from collections.abc import Mapping

from benchmarks.datasets.models import TaskCategory

from .base import BaseExternalAdapter, BenchmarkName


class DeepResearchBenchAdapter(BaseExternalAdapter):
    benchmark: BenchmarkName = "deepresearchbench"
    expected_count = 10
    supported_metric_names = ("citation_precision", "citation_recall", "report_quality")

    @property
    def runtime_category(self) -> TaskCategory:
        return TaskCategory.METHOD_COMPARISON

    def _eligible(self, item: Mapping[str, object]) -> bool:
        # Do not copy or score rubric fields.  Their presence merely indicates
        # that an evaluator-side report/citation reference may exist.  Missing
        # optional metadata is accepted for offline fixtures and is rejected
        # later by the formal lock/materialization path if it is required.
        value = item.get("task_type", item.get("category", item.get("kind")))
        if value is None:
            return True
        return "short-answer" not in str(value).casefold() and "short_answer" not in str(value).casefold()


Adapter = DeepResearchBenchAdapter

__all__ = ["Adapter", "DeepResearchBenchAdapter"]
