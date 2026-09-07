"""DeepResearch Bench adapter for report/citation comparison."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

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
        # Rubric/citation metadata is evaluator-only, but both must be present
        # before a record can enter the deterministic portfolio.  Neither
        # field is projected into RuntimeTask or FrozenEvidenceRecord.
        value = item.get("task_type", item.get("category", item.get("kind")))
        if value is None or "research-report" not in str(value).casefold().replace("_", "-"):
            return False
        citation = next(
            (item.get(key) for key in ("citations", "citation_metadata", "references")),
            None,
        )
        rubric = item.get("rubric")
        if isinstance(citation, Mapping):
            has_citation = bool(cast(Mapping[object, object], citation))
        elif isinstance(citation, (list, tuple)):
            has_citation = bool(cast(list[object] | tuple[object, ...], citation))
        else:
            has_citation = False
        has_rubric = isinstance(rubric, Mapping) and bool(cast(Mapping[object, object], rubric))
        return has_citation and has_rubric


Adapter = DeepResearchBenchAdapter

__all__ = ["Adapter", "DeepResearchBenchAdapter"]
