from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from deepresearch.providers.frozen_search import (
    FrozenCorpusFetcher,
    FrozenCorpusMaterializer,
    FrozenCorpusSearchProvider,
)
from experiments.factories import components_for, validate_frozen_bindings
from experiments.models import ExperimentTaskRun, ExperimentVariant


@pytest.fixture
def formal_task_config():
    from benchmarks.datasets.isolation import GoldIsolationGuard
    from benchmarks.datasets.models import AnnotatedQuestion
    from experiments.config import FormalExperimentConfig, canonical_sha256
    from tests.unit.experiments.test_config import valid_payload

    payload = valid_payload.__wrapped__()
    question = AnnotatedQuestion.model_validate_json(
        Path("benchmarks/datasets/templates/question.example.json").read_bytes()
    )
    task = GoldIsolationGuard.runtime_view(question).model_copy(
        update={
            "task_id": "test-t1",
            "corpus_version": payload["corpus_version"],
            "index_version": payload["index_version"],
            "request": question.request.model_copy(
                update={
                    "execution_mode": "hybrid",
                    "access_profile": "local",
                    "run_purpose": "benchmark",
                    "provider_profile_id": payload["provider_profile_id"],
                    "budget_preset": "medium",
                }
            ),
        }
    )
    payload["internal_runtime_task_hashes"] = {
        task.task_id: canonical_sha256(task.model_dump(mode="json"))
    }
    return task, FormalExperimentConfig.model_validate(payload)


@pytest.mark.parametrize(
    "patch",
    [
        {"execution_mode": "live"},
        {"access_profile": "public_live"},
        {"run_purpose": "demo"},
        {"provider_profile_id": "unsealed-provider"},
    ],
)
@pytest.mark.parametrize("reseal_hash", [False, True])
def test_formal_factory_rejects_nonformal_request_identity(formal_task_config, patch, reseal_hash):
    from benchmarks.datasets.isolation import GoldAccessViolation
    from experiments.config import canonical_sha256
    from experiments.factories import formal_run_config

    task, config = formal_task_config
    task = task.model_copy(update={"request": task.request.model_copy(update=patch)})
    if reseal_hash:
        config = config.model_copy(
            update={
                "internal_runtime_task_hashes": {
                    task.task_id: canonical_sha256(task.model_dump(mode="json"))
                }
            }
        )
    with pytest.raises((GoldAccessViolation, ValueError)):
        formal_run_config(
            config=config,
            task=task,
            variant=ExperimentVariant.A,
            budget=RunBudget.preset("medium"),
            seed=7,
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"task_id": "test-unsealed"},
        {"corpus_version": "other"},
        {"index_version": "other"},
    ],
)
def test_formal_factory_rejects_unsealed_task_or_corpus(formal_task_config, patch):
    from benchmarks.datasets.isolation import GoldAccessViolation
    from experiments.factories import formal_run_config

    task, config = formal_task_config
    with pytest.raises((GoldAccessViolation, ValueError)):
        formal_run_config(
            config=config,
            task=task.model_copy(update=patch),
            variant=ExperimentVariant.A,
            budget=RunBudget.preset("medium"),
            seed=7,
        )


def test_formal_factory_rejects_budget_limits_mismatching_request(formal_task_config):
    from experiments.factories import formal_run_config

    task, config = formal_task_config
    with pytest.raises(ValueError, match="budget"):
        formal_run_config(
            config=config,
            task=task,
            variant=ExperimentVariant.A,
            budget=RunBudget.preset("low"),
            seed=7,
        )


@pytest.mark.parametrize("field", ["corpus_version", "index_version"])
def test_formal_factory_checks_corpus_identity_even_for_authorized_hash(formal_task_config, field):
    from experiments.config import canonical_sha256
    from experiments.factories import formal_run_config

    task, config = formal_task_config
    task = task.model_copy(update={field: "other"})
    config = config.model_copy(
        update={
            "internal_runtime_task_hashes": {
                task.task_id: canonical_sha256(task.model_dump(mode="json"))
            }
        }
    )
    with pytest.raises(ValueError, match="corpus/index"):
        formal_run_config(
            config=config,
            task=task,
            variant=ExperimentVariant.A,
            budget=RunBudget.preset("medium"),
            seed=7,
        )


def test_formal_factory_accepts_exact_external_authorization(formal_task_config):
    from experiments.config import canonical_sha256
    from experiments.factories import formal_run_config

    task, config = formal_task_config
    task = task.model_copy(update={"task_id": "ext-frames-row1"})
    config = config.model_copy(
        update={
            "external_config_sha256": "b" * 64,
            "external_lock_sha256": "c" * 64,
            "external_runtime_task_hashes": {
                task.task_id: canonical_sha256(task.model_dump(mode="json"))
            },
        }
    )
    run_config = formal_run_config(
        config=config,
        task=task,
        variant=ExperimentVariant.A,
        budget=RunBudget.preset("medium"),
        seed=7,
    )
    assert run_config.request == task.request


@pytest.mark.parametrize("preset", ["low", "medium", "high"])
def test_formal_factory_accepts_authorized_budget_arms(formal_task_config, preset):
    from experiments.factories import formal_run_config

    base, config = formal_task_config
    task = base.model_copy(
        update={"request": base.request.model_copy(update={"budget_preset": preset})}
    )
    run_config = formal_run_config(
        config=config,
        task=task,
        variant=ExperimentVariant.A,
        budget=RunBudget.preset(preset),
        seed=7,
    )
    assert run_config.request == task.request
    assert run_config.workflow_id == "research-v1"
    assert run_config.budget == RunBudget.preset(preset)


def _formal_manifest_for(task, config):
    import hashlib
    import json

    from deepresearch.runtime.manifest import CostCalculator
    from tests.unit.runtime.test_manifest import _manifest, pricing_snapshot

    base = _manifest(pricing_snapshot.__wrapped__(), workflow_id="research-v1")
    usage = ResourceUsage.zero().model_copy(
        update={"input_tokens": 100, "output_tokens": 50, "total_tokens": 150}
    )
    usage = usage.model_copy(
        update={"cost_usd": CostCalculator.estimate(usage, config.pricing_snapshot).total_usd}
    )
    call = base.provider_calls[0].model_copy(
        update={
            "provider_id": config.provider_id,
            "endpoint_type": config.endpoint_type,
            "model_id": config.model_id,
            "usage": usage,
            "pricing_snapshot_id": config.pricing_snapshot.snapshot_id,
            "estimated_cost_usd": usage.cost_usd,
        }
    )
    return base.model_copy(
        update={
            "request_sha256": hashlib.sha256(
                json.dumps(
                    task.request.model_dump(mode="json"),
                    sort_keys=True,
                    ensure_ascii=False,
                    separators=(",", ":"),
                ).encode()
            ).hexdigest(),
            "budget": RunBudget.preset(task.request.budget_preset),
            "usage": usage,
            "usage_by_node": {"Planner": usage},
            "pricing_snapshots": (config.pricing_snapshot,),
            "provider_calls": (call,),
            "model_ids": (config.model_id,),
            "provider_profiles": (
                base.provider_profiles[0].model_copy(
                    update={
                        "profile_id": config.provider_profile_id,
                        "execution_mode": "hybrid",
                        "provider_ids": (config.provider_id,),
                    }
                ),
            ),
            "node_executions": (base.node_executions[0].model_copy(update={"usage": usage}),),
        }
    )


def test_manifest_conversion_rejects_unbound_budget_relabeling(formal_task_config):
    from experiments.models import task_run_from_manifest

    task, config = formal_task_config
    manifest = _formal_manifest_for(task, config)
    low = task.model_copy(
        update={"request": task.request.model_copy(update={"budget_preset": "low"})}
    )
    with pytest.raises(ValueError, match="request|budget"):
        task_run_from_manifest(
            manifest,
            config=config,
            task=low,
            protocol="end_to_end",
            variant=ExperimentVariant.A,
            seed=7,
            manifest_path="manifest.json",
            status="completed",
        )


@pytest.mark.parametrize(
    "patch",
    [
        {"request_sha256": "a" * 64},
        {"budget": RunBudget.preset("low")},
    ],
)
def test_manifest_conversion_verifies_request_hash_and_budget_limits(formal_task_config, patch):
    from experiments.models import task_run_from_manifest

    task, config = formal_task_config
    manifest = _formal_manifest_for(task, config).model_copy(update=patch)
    with pytest.raises(ValueError, match="request|budget"):
        task_run_from_manifest(
            manifest,
            config=config,
            task=task,
            protocol="end_to_end",
            variant=ExperimentVariant.A,
            seed=7,
            manifest_path="manifest.json",
            status="completed",
        )


def test_manifest_conversion_rejects_unsealed_task_with_matching_request_hash(formal_task_config):
    from benchmarks.datasets.isolation import GoldAccessViolation
    from experiments.models import task_run_from_manifest

    base, config = formal_task_config
    task = base.model_copy(
        update={"request": base.request.model_copy(update={"question": "Unsealed question?"})}
    )
    manifest = _formal_manifest_for(task, config)
    with pytest.raises(GoldAccessViolation):
        task_run_from_manifest(
            manifest,
            config=config,
            task=task,
            protocol="end_to_end",
            variant=ExperimentVariant.A,
            seed=7,
            manifest_path="manifest.json",
            status="completed",
        )


@pytest.mark.parametrize("preset", ["low", "medium", "high"])
def test_manifest_record_derives_selected_budget_from_authorized_task(formal_task_config, preset):
    from experiments.models import task_run_from_manifest

    base, config = formal_task_config
    task = base.model_copy(
        update={"request": base.request.model_copy(update={"budget_preset": preset})}
    )
    manifest = _formal_manifest_for(task, config)
    record = task_run_from_manifest(
        manifest,
        config=config,
        task=task,
        protocol="end_to_end",
        variant=ExperimentVariant.A,
        seed=7,
        manifest_path="manifest.json",
        status="completed",
    )
    assert record.budget_preset == preset
    assert record.task_id == task.task_id
    assert record.usage == manifest.usage


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


def test_formal_record_copies_verified_core_cost_and_rejects_unsealed_pricing(formal_task_config):
    from experiments.models import task_run_from_manifest

    task, config = formal_task_config
    pricing = config.pricing_snapshot
    manifest = _formal_manifest_for(task, config)
    record = task_run_from_manifest(
        manifest,
        config=config,
        task=task,
        protocol="end_to_end",
        variant=ExperimentVariant.A,
        seed=7,
        manifest_path="manifest.json",
        status="completed",
    )
    assert record.usage.cost_usd == manifest.usage.cost_usd
    assert record.pricing_snapshot_ids == (pricing.snapshot_id,)
    with pytest.raises(ValueError, match="pricing"):
        task_run_from_manifest(
            manifest,
            config=config.model_copy(
                update={"pricing_snapshot": pricing.model_copy(update={"snapshot_id": "unsealed"})}
            ),
            task=task,
            protocol="end_to_end",
            variant=ExperimentVariant.A,
            seed=7,
            manifest_path="manifest.json",
            status="completed",
        )
