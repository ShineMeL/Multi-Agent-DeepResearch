from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepresearch.domain import ResourceUsage
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from deepresearch.providers.frozen_search import (
    FrozenCorpusFetcher,
    FrozenCorpusMaterializer,
    FrozenCorpusSearchProvider,
)
from experiments.factories import components_for, validate_frozen_bindings
from experiments.models import ExperimentTaskRun, ExperimentVariant


class Container:
    def planner_for(self, identity):
        return SimpleNamespace(variant=identity)

    def ranker_for(self, identity):
        return SimpleNamespace(ranker_id=identity)


@pytest.mark.parametrize(
    ("variant", "expected"),
    [
        ("A", ("P1", "R1")),
        ("B", ("P1", "R2")),
        ("C", ("P2", "R1")),
        ("D", ("P2", "R2")),
        ("P0", ("P0", "R0")),
    ],
)
def test_exact_component_mapping(variant, expected):
    assert components_for(ExperimentVariant(variant), Container()).ids == expected


def test_oracle_never_receives_agent_components():
    with pytest.raises(ValueError, match="evaluator-only"):
        components_for(ExperimentVariant.ORACLE, Container())


def test_frozen_bindings_reject_live_fetch_and_foreign_materializer():
    snapshot = FrozenCorpusSnapshot.load(
        Path("tests/fixtures/frozen_corpus/task-fixture"), task_id="task-fixture"
    )
    search = FrozenCorpusSearchProvider(snapshot)
    fetcher = FrozenCorpusFetcher(snapshot)
    materializer = FrozenCorpusMaterializer(snapshot)
    validate_frozen_bindings(
        search=search, fetcher=fetcher, materializer=materializer, store_materializer=materializer
    )
    with pytest.raises(ValueError):
        validate_frozen_bindings(
            search=search,
            fetcher=object(),
            materializer=materializer,
            store_materializer=materializer,
        )
    with pytest.raises(ValueError):
        validate_frozen_bindings(
            search=search, fetcher=fetcher, materializer=materializer, store_materializer=None
        )
    other = FrozenCorpusSnapshot.load(
        Path("tests/fixtures/frozen_corpus/task-fixture"), task_id="task-fixture"
    )
    with pytest.raises(ValueError):
        validate_frozen_bindings(
            search=search,
            fetcher=fetcher,
            materializer=FrozenCorpusMaterializer(other),
            store_materializer=materializer,
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"variant": "ORACLE"},
        {"seed": None},
        {"repeat_id": 1},
        {"protocol": "ranker_component"},
        {"variant": "P0"},
        {"validity": "invalid"},
        {"error_code": "REPLAY_MISS"},
    ],
)
def test_agent_record_protocol_and_replication_boundaries(patch):
    payload = {
        "task_id": "test-t1",
        "protocol": "end_to_end",
        "variant": "A",
        "planner_id": "P1",
        "ranker_id": "R1",
        "budget_preset": "medium",
        "seed": 7,
        "status": "completed",
        "manifest_path": "manifest.json",
        "artifact_ids": (),
        "usage": ResourceUsage.zero(cost_known=True),
        "pricing_snapshot_ids": ("sealed",),
        "pricing_status": "estimated",
        "cost_label": "estimated_from_normalized_schedule",
    }
    ExperimentTaskRun.model_validate(payload)
    with pytest.raises(ValidationError):
        ExperimentTaskRun.model_validate({**payload, **patch})


def test_frozen_factory_loads_one_snapshot_and_core_parsers():
    from deepresearch.providers.parsers.html import HtmlParser
    from deepresearch.providers.parsers.pdf import PdfParser
    from experiments.factories import frozen_composition

    composition = frozen_composition(
        snapshot_dir=Path("tests/fixtures/frozen_corpus/task-fixture"), task_id="task-fixture"
    )
    assert composition.search.snapshot is composition.fetcher.snapshot
    assert composition.materializer.snapshot is composition.search.snapshot
    assert composition.store_evidence.materializer is composition.materializer
    assert tuple(type(parser) for parser in composition.parsers) == (HtmlParser, PdfParser)


def test_formal_record_copies_verified_core_cost_and_rejects_unsealed_pricing():
    from experiments.models import task_run_from_manifest
    from tests.unit.runtime.test_manifest import _manifest, pricing_snapshot

    pricing = pricing_snapshot.__wrapped__()
    manifest = _manifest(pricing, workflow_id="research-v1")
    record = task_run_from_manifest(
        manifest,
        sealed_pricing=pricing,
        task_id="test-t1",
        protocol="end_to_end",
        variant=ExperimentVariant.A,
        budget_preset="medium",
        seed=7,
        manifest_path="manifest.json",
        status="completed",
    )
    assert record.usage.cost_usd == manifest.usage.cost_usd
    assert record.pricing_snapshot_ids == (pricing.snapshot_id,)
    with pytest.raises(ValueError, match="pricing"):
        task_run_from_manifest(
            manifest,
            sealed_pricing=pricing.model_copy(update={"snapshot_id": "unsealed"}),
            task_id="test-t1",
            protocol="end_to_end",
            variant=ExperimentVariant.A,
            budget_preset="medium",
            seed=7,
            manifest_path="manifest.json",
            status="completed",
        )
