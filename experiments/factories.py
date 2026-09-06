"""Exact formal components and the frozen source composition boundary."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from benchmarks.datasets.models import RuntimeTask
from deepresearch.domain import RunBudget, RunConfig
from deepresearch.evidence.rankers import EvidenceRanker
from deepresearch.planning.planners import Planner
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from deepresearch.providers.frozen_search import (
    FrozenCorpusFetcher,
    FrozenCorpusMaterializer,
    FrozenCorpusSearchProvider,
    FrozenMaterialization,
)
from deepresearch.providers.parsers.html import HtmlParser
from deepresearch.providers.parsers.pdf import PdfParser
from deepresearch.providers.types import ParsedDocument
from experiments.config import FormalExperimentConfig, authorized_staged_task
from experiments.models import COMPONENT_IDS, ExperimentVariant, canonical_sha256


class ComponentContainer(Protocol):
    def planner_for(self, identity: Literal["P0", "P1", "P2"]) -> Planner: ...
    def ranker_for(self, identity: Literal["R0", "R1", "R2"]) -> EvidenceRanker: ...


@dataclass(frozen=True)
class ExperimentComponents:
    planner: Planner
    ranker: EvidenceRanker

    @property
    def ids(self) -> tuple[str, str]:
        return self.planner.variant, self.ranker.ranker_id


def components_for(
    variant: ExperimentVariant, container: ComponentContainer
) -> ExperimentComponents:
    if variant == ExperimentVariant.ORACLE:
        raise ValueError("ORACLE is evaluator-only and cannot receive an agent RunConfig")
    planner_id, ranker_id = COMPONENT_IDS[variant]
    result = ExperimentComponents(
        container.planner_for(planner_id), container.ranker_for(ranker_id)
    )
    if result.ids != (planner_id, ranker_id):
        raise ValueError("factory returned components with incorrect identities")
    return result


def validate_frozen_bindings(
    *, search: object, fetcher: object, materializer: object, store_materializer: object
) -> None:
    if (
        type(search) is not FrozenCorpusSearchProvider
        or type(fetcher) is not FrozenCorpusFetcher
        or type(materializer) is not FrozenCorpusMaterializer
    ):
        raise ValueError("formal factories require frozen search/fetch/materializer")
    if (
        search.snapshot is not fetcher.snapshot
        or search.snapshot is not materializer.snapshot
        or store_materializer is not materializer
    ):
        raise ValueError("one frozen snapshot and identity-preserving StoreEvidence are required")


@dataclass(frozen=True)
class FrozenStoreEvidence:
    materializer: FrozenCorpusMaterializer

    def __call__(
        self,
        *,
        selected_evidence_ids: Sequence[str],
        parsed_documents: Mapping[str, ParsedDocument],
        information_need_ids: tuple[str, ...],
    ) -> FrozenMaterialization:
        return self.materializer.materialize(
            selected_evidence_ids=selected_evidence_ids,
            parsed_documents=parsed_documents,
            information_need_ids=information_need_ids,
        )


@dataclass(frozen=True)
class FrozenComposition:
    search: FrozenCorpusSearchProvider
    fetcher: FrozenCorpusFetcher
    materializer: FrozenCorpusMaterializer
    store_evidence: FrozenStoreEvidence
    parsers: tuple[HtmlParser, PdfParser]

    def __post_init__(self) -> None:
        if type(self.store_evidence) is not FrozenStoreEvidence:
            raise ValueError("StoreEvidence must preserve frozen evidence IDs")
        validate_frozen_bindings(
            search=self.search,
            fetcher=self.fetcher,
            materializer=self.materializer,
            store_materializer=self.store_evidence.materializer,
        )
        if tuple(type(parser) for parser in self.parsers) != (HtmlParser, PdfParser):
            raise ValueError("formal parsing requires locked Core HTML/PDF parsers")


def frozen_composition(*, snapshot_dir: Path, task_id: str) -> FrozenComposition:
    snapshot = FrozenCorpusSnapshot.load(snapshot_dir, task_id=task_id)
    materializer = FrozenCorpusMaterializer(snapshot)
    return FrozenComposition(
        search=FrozenCorpusSearchProvider(snapshot),
        fetcher=FrozenCorpusFetcher(snapshot),
        materializer=materializer,
        store_evidence=FrozenStoreEvidence(materializer),
        parsers=(HtmlParser(), PdfParser()),
    )


def formal_run_config(
    *,
    config: FormalExperimentConfig,
    task: RuntimeTask,
    variant: ExperimentVariant,
    budget: RunBudget,
    seed: int | None,
) -> RunConfig:
    if variant == ExperimentVariant.ORACLE:
        raise ValueError("ORACLE is evaluator-only")
    planner_id, ranker_id = COMPONENT_IDS[variant]
    task = RuntimeTask.model_validate_json(task.model_dump_json(), strict=True)
    authorized_staged_task(
        config,
        task,
        staged_sha256=canonical_sha256(task.model_dump(mode="json")),
        budget_preset=task.request.budget_preset,
    )
    if budget != RunBudget.preset(task.request.budget_preset):
        raise ValueError("formal run budget must match the request preset with unused counters")
    return RunConfig(
        request=task.request,
        workflow_id="research-v1",
        planner_id=planner_id,
        ranker_id=ranker_id,
        budget=budget,
        seed=seed,
        prompt_versions={
            "planner": config.prompt_version,
            "writer": config.writer_prompt_version,
            "judge": config.judge_prompt_version,
        },
        ranker_weights_version=config.ranker_weights_version,
    )
