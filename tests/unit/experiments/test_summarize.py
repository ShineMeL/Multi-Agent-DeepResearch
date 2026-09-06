from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest
import yaml

from benchmarks.datasets.models import TaskCategory
from benchmarks.datasets.validator import canonical_json_bytes
from benchmarks.evaluators.metrics import MetricValue
from benchmarks.evaluators.statistics import SeedRunRecord, compare_rankers
from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.runtime.manifest import PricingSnapshot, RunManifest
from experiments.models import (
    EvaluatorReferenceManifest,
    ExperimentTaskRun,
    ExperimentVariant,
    OracleReferenceResult,
    canonical_sha256,
)
from experiments.runner import ExperimentRunner
from experiments.summarize import _load_replication_bindings, summarize_experiment
from tests.integration.experiments.test_abcd_runner import _config_and_task


def _write_full_group(root: Path) -> Path:
    (root / "raw").mkdir(parents=True)
    config, task = _config_and_task()
    task = task.model_copy(update={"task_id": "test-a"})
    pricing = config.pricing_snapshot.model_copy(update={"snapshot_id": "pricing-v1"})
    config = config.model_copy(
        update={
            "pricing_snapshot": pricing,
            "main_test_task_ids": ("test-a",),
            "stability_task_ids": ("test-a",),
            "cost_subset_task_ids": ("test-a",),
            "p0_task_ids": ("test-a",),
            "oracle_task_ids": ("test-a",),
            "internal_runtime_task_hashes": {
                "test-a": canonical_sha256(task.model_dump(mode="json"))
            },
        }
    )
    group_id = config.experiment_group_id()
    config_bytes = yaml.safe_dump(
        config.model_dump(mode="json"), sort_keys=True
    ).encode("utf-8")
    (root / "config").mkdir()
    (root / "config" / "formal.yaml").write_bytes(config_bytes)
    group = {
        "group_id": group_id,
        "dataset_version": "dataset-v1",
        "budget_preset": "medium",
        "candidate_pool_seed": 7,
        "pricing_snapshot_id": "pricing-v1",
        "code_commit": "a" * 40,
        "private_manifest_sha256": "b" * 64,
        "config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "evaluator_version": "evaluator-v1",
        "protocols": ["ranker_component", "planner_policy", "end_to_end", "reference"],
        "protocol_task_ids": {
            "ranker_component": ["test-a"],
            "planner_policy": ["test-a"],
            "end_to_end": ["test-a"],
            "reference": ["test-a"],
        },
        "oracle_task_ids": ["test-a"],
        "cost_subset_task_ids": ["test-a"],
        "expected_variants": {
            "ranker_component": ["R0", "R1", "R2"],
            "planner_policy": ["A", "B", "C", "D"],
            "end_to_end": ["A", "B", "C", "D"],
            "reference": ["P0"],
        },
        "budgets": ["medium"],
        "required_budgets": {
            "ranker_component": ["medium"],
            "planner_policy": ["medium"],
            "end_to_end": ["medium"],
            "reference": ["medium"],
        },
        "replication": {
            "mode": "seeds",
            "seed_supported": True,
            "seed_values": [1],
            "repeat_ids": [],
        },
        "task_categories": {"test-a": "technical_survey"},
    }
    (root / "group.json").write_text(json.dumps(group), encoding="utf-8")
    variants = {
        "ranker_component": (("R0", "P1", "R0"), ("R1", "P1", "R1"), ("R2", "P1", "R2")),
        "planner_policy": (("A", "P1", "R1"), ("B", "P1", "R2"), ("C", "P2", "R1"), ("D", "P2", "R2")),
        "end_to_end": (("A", "P1", "R1"), ("B", "P1", "R2"), ("C", "P2", "R1"), ("D", "P2", "R2")),
        "reference": (("P0", "P0", "R0"),),
    }
    for protocol, protocol_variants in variants.items():
        for variant, planner_id, ranker_id in protocol_variants:
            run = ExperimentTaskRun(
                task_id="test-a",
                protocol=protocol,
                variant=variant,
                planner_id=planner_id,
                ranker_id=ranker_id,
                budget_preset="medium",
                seed=1,
                status="completed",
                candidate_pool_hash="a" * 64 if protocol == "ranker_component" else None,
                manifest_path="manifest.json",
                artifact_ids=(),
                usage=ResourceUsage.zero(cost_known=True),
                pricing_snapshot_ids=("pricing-v1",),
                pricing_status="estimated",
                cost_label="estimated_from_normalized_schedule",
                category="technical_survey",
                metrics={
                    **(
                        {
                            "citation_support_precision": MetricValue(
                                name="citation_support_precision",
                                value={"R0": 0.4, "R1": 0.6, "R2": 0.8}[variant],
                                numerator={"R0": 0.4, "R1": 0.6, "R2": 0.8}[variant],
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                            "information_completeness": MetricValue(
                                name="information_completeness",
                                value={"R0": 0.5, "R1": 0.6, "R2": 0.7}[variant],
                                numerator={"R0": 0.5, "R1": 0.6, "R2": 0.7}[variant],
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                        }
                        if protocol == "ranker_component"
                        else {}
                    ),
                    **(
                        {
                            "citation_support_precision": MetricValue(
                                name="citation_support_precision",
                                value={"A": 0.6, "B": 0.65, "C": 0.7, "D": 0.8, "P0": 0.65}[variant],
                                numerator={"A": 0.6, "B": 0.65, "C": 0.7, "D": 0.8, "P0": 0.65}[variant],
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                            "information_completeness": MetricValue(
                                name="information_completeness",
                                value={"A": 0.6, "B": 0.55, "C": 0.7, "D": 0.75, "P0": 0.65}[variant],
                                numerator={"A": 0.6, "B": 0.55, "C": 0.7, "D": 0.75, "P0": 0.65}[variant],
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                            "query_redundancy": MetricValue(
                                name="query_redundancy",
                                value=0.2,
                                numerator=0.2,
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                        }
                        if protocol in {"planner_policy", "reference"}
                        else {}
                    ),
                    **(
                        {
                            "citation_support_precision": MetricValue(
                                name="citation_support_precision",
                                value={"A": 0.6, "B": 0.65, "C": 0.7, "D": 0.8}[variant],
                                numerator={"A": 0.6, "B": 0.65, "C": 0.7, "D": 0.8}[variant],
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                            "information_completeness": MetricValue(
                                name="information_completeness",
                                value={"A": 0.6, "B": 0.55, "C": 0.7, "D": 0.75}[variant],
                                numerator={"A": 0.6, "B": 0.55, "C": 0.7, "D": 0.75}[variant],
                                denominator=1.0,
                                version="benchmark-metrics-v1",
                            ),
                        }
                        if protocol == "end_to_end"
                        else {}
                    ),
                },
            )
            key = ExperimentRunner.idempotency_key(
                group_id, protocol, variant, "test-a", 1, None, "medium"
            )
            (root / "raw" / f"{key}.json").write_bytes(
                json.dumps(run.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            )
    candidate_key = ExperimentRunner.idempotency_key(
        group_id, "ranker_component", "POOL", "test-a", 7, None, "medium"
    )
    candidate_payload = {"candidate_pool_version": "formal-v1", "evidence_ids": [], "task_id": "test-a"}
    candidate_bytes = canonical_json_bytes(candidate_payload)
    candidate_root = root / "candidate-pools"
    setup_root = root / "candidate-pool-setup"
    artifact_root = root / "artifacts"
    candidate_root.mkdir()
    setup_root.mkdir()
    artifact_root.mkdir()
    candidate_path = candidate_root / f"{candidate_key}.json"
    candidate_path.write_bytes(candidate_bytes)
    candidate_hash = hashlib.sha256(candidate_bytes).hexdigest()
    for raw_path in (root / "raw").glob("*.json"):
        raw_payload = json.loads(raw_path.read_bytes())
        if raw_payload["protocol"] == "ranker_component":
            raw_payload["candidate_pool_hash"] = candidate_hash
            raw_path.write_bytes(canonical_json_bytes(raw_payload))
    usage = ResourceUsage.zero(cost_known=True)
    (root / "agent-inputs").mkdir()
    (root / "agent-inputs" / f"{candidate_key[:32]}.json").write_bytes(
        canonical_json_bytes(task.model_dump(mode="json"))
    )
    request_hash = hashlib.sha256(
        json.dumps(
            task.request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    manifest_config_hash = ExperimentRunner._core_config_sha256(
        config=config,
        task=task,
        planner_id="P1",
        ranker_id="R1",
        seed=7,
    )
    manifest = RunManifest.create(
        {
            "schema_version": "run-manifest-v1",
            "run_id": "pool-run-1",
            "thread_id": "pool-thread-1",
            "code_commit": "a" * 40,
            "dependency_lock_sha256": "b" * 64,
            "request_sha256": request_hash,
            "config_sha256": manifest_config_hash,
            "workflow_id": "research-v1",
            "graph_version": "graph-v1",
            "planner_id": "P1",
            "provider_profiles": (),
            "model_ids": (),
            "prompt_versions": {},
            "parser_versions": {},
            "ranker_id": "R1",
            "ranker_weights_version": "weights-v1",
            "budget": RunBudget.preset("medium"),
            "usage": usage,
            "usage_by_node": {},
            "pricing_status": "estimated",
            "pricing_snapshots": (config.pricing_snapshot,),
            "provider_calls": (),
            "node_executions": (),
            "parsed_artifacts": (),
            "evidence_hashes": (),
            "source_snapshot_ids": (),
            "artifact_ids": (),
            "run_event_count": 0,
            "run_events_sha256": "e" * 64,
            "seed": 7,
            "seed_supported": True,
            "cache_hit_count": 0,
            "stop_reason": "SUFFICIENT",
            "is_partial": False,
            "failure_codes": (),
            "started_at": datetime(2026, 9, 6, tzinfo=UTC),
            "finished_at": datetime(2026, 9, 6, 0, 0, 1, tzinfo=UTC),
        }
    )
    manifest_path = artifact_root / f"{candidate_key}.json"
    manifest_bytes = manifest.model_dump_json().encode("utf-8")
    manifest_path.write_bytes(manifest_bytes)
    setup = {
        "schema_version": "candidate-pool-setup-v1",
        "group_id": group_id,
        "task_id": "test-a",
        "pool_key": candidate_key,
        "candidate_pool_sha256": candidate_hash,
        "evidence_ids_sha256": hashlib.sha256(canonical_json_bytes([])).hexdigest(),
        "manifest_path": str(manifest_path),
        "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "usage": usage.model_dump(mode="json"),
        "pricing_snapshot_ids": ["pricing-v1"],
        "pricing_status": "estimated",
    }
    (setup_root / f"{candidate_key}.json").write_bytes(canonical_json_bytes(setup))
    created_at = datetime(2026, 9, 6, tzinfo=UTC)
    oracle_result = OracleReferenceResult(
        task_id="test-a",
        dataset_version="dataset-v1",
        private_manifest_sha256="b" * 64,
        frozen_snapshot_id="snapshot-test-a",
        approved_id_set_sha256="c" * 64,
        metric_values={
            "approved_evidence_recall": MetricValue(
                name="approved_evidence_recall",
                value=1.0,
                numerator=1.0,
                denominator=1.0,
                version="benchmark-metrics-v1",
            )
        },
        evaluator_version="evaluator-v1",
        created_at=created_at,
    )
    oracle_manifest = EvaluatorReferenceManifest(
        group_id=group_id,
        private_manifest_sha256=oracle_result.private_manifest_sha256,
        evaluator_version=oracle_result.evaluator_version,
        task_ids_sha256=canonical_sha256(("test-a",)),
        oracle_results_sha256=canonical_sha256([oracle_result.model_dump(mode="json")]),
        created_at=created_at,
    )
    (root / "oracle-reference.jsonl").write_bytes(
        json.dumps(oracle_result.model_dump(mode="json"), sort_keys=True).encode("utf-8") + b"\n"
    )
    (root / "evaluator-reference-manifest.json").write_bytes(
        json.dumps(oracle_manifest.model_dump(mode="json"), sort_keys=True).encode("utf-8")
    )
    return root


def test_summary_contains_real_sections_and_verify_only_returns_them(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")

    summary = summarize_experiment(root, bootstrap_resamples=4)
    assert summary["sections"]["end_to_end"]["metrics"]
    confidence = json.loads((root / "confidence_intervals.json").read_bytes())
    assert confidence["sections"]["end_to_end"]
    planner = confidence["sections"]["planner_policy"]["comparisons"]
    assert (planner["R1"]["baseline"], planner["R1"]["candidate"]) == ("A", "C")
    assert (planner["R2"]["baseline"], planner["R2"]["candidate"]) == ("B", "D")
    assert confidence["sections"]["ranker_component"]["metric"] == (
        "citation_support_precision"
    )
    assert "total_tokens" not in json.dumps(confidence)
    verified = summarize_experiment(root, verify_only=True)
    assert verified["sections"]["end_to_end"]


def test_summary_fails_closed_when_raw_records_have_no_quality_metrics(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    for path in (root / "raw").glob("*.json"):
        payload = json.loads(path.read_bytes())
        payload["metrics"] = {}
        path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="quality metric"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_does_not_trust_raw_category_over_sealed_group_category(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    raw_path = next((root / "raw").glob("*.json"))
    payload = json.loads(raw_path.read_bytes())
    payload["category"] = "fact_checking"
    raw_path.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="category"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_rejects_raw_filename_that_is_not_the_sealed_identity(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    raw_paths = sorted((root / "raw").glob("*.json"))
    renamed = root / "raw" / "not-the-sealed-id.json"
    raw_paths[0].rename(renamed)

    with pytest.raises(ValueError, match="idempotency key"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_verify_only_detects_tampered_oracle_reference_artifact(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    summarize_experiment(root, bootstrap_resamples=4)
    (root / "oracle-reference.jsonl").write_bytes(
        (root / "oracle-reference.jsonl").read_bytes() + b"tamper\n"
    )

    with pytest.raises(ValueError, match="artifact hash"):
        summarize_experiment(root, verify_only=True)


def test_summary_rejects_oracle_dataset_version_tamper(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    oracle_path = root / "oracle-reference.jsonl"
    payload = json.loads(oracle_path.read_bytes())
    payload["dataset_version"] = "wrong-dataset"
    oracle_path.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")
    manifest_path = root / "evaluator-reference-manifest.json"
    manifest_payload = json.loads(manifest_path.read_bytes())
    manifest_payload["oracle_results_sha256"] = canonical_sha256([payload])
    manifest_path.write_text(json.dumps(manifest_payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(ValueError, match="dataset|identity|artifact hash"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_rejects_extra_candidate_setup_artifact(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    setup_root = root / "candidate-pool-setup"
    (setup_root / "extra.json").write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="candidate pool setup"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_rejects_completed_ranker_without_verified_pool_setup(
    tmp_path: Path,
) -> None:
    root = _write_full_group(tmp_path / "group")
    for path in (root / "candidate-pool-setup").glob("*.json"):
        path.unlink()
    for path in (root / "candidate-pools").glob("*.json"):
        path.unlink()
    (root / "candidate-pool-setup").rmdir()
    (root / "candidate-pools").rmdir()

    with pytest.raises(ValueError, match="candidate pool|setup"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_rejects_setup_manifest_with_forged_request_provenance(
    tmp_path: Path,
) -> None:
    root = _write_full_group(tmp_path / "group")
    setup_payload = json.loads(
        next((root / "candidate-pool-setup").glob("*.json")).read_bytes()
    )
    manifest_path = Path(setup_payload["manifest_path"])
    forged_manifest = RunManifest.model_validate_json(manifest_path.read_bytes(), strict=True).model_copy(
        update={"request_sha256": "f" * 64}
    )
    forged_bytes = forged_manifest.model_dump_json().encode("utf-8")
    manifest_path.write_bytes(forged_bytes)
    setup_payload["manifest_sha256"] = hashlib.sha256(forged_bytes).hexdigest()
    setup_path = next((root / "candidate-pool-setup").glob("*.json"))
    setup_path.write_bytes(canonical_json_bytes(setup_payload))

    with pytest.raises(ValueError, match="manifest|provenance|request"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_rejects_receipt_identity_reused_across_unseeded_repeats(
    tmp_path: Path,
) -> None:
    root = tmp_path / "group"
    artifacts = root / "artifacts"
    requests = root / "requests"
    bindings = artifacts / "replication-bindings"
    artifacts.mkdir(parents=True)
    requests.mkdir()
    bindings.mkdir()
    group = {
        "group_id": "group",
        "replication": {
            "mode": "independent_repeats",
            "seed_supported": False,
            "repeat_ids": [1, 2],
        },
    }
    pricing = PricingSnapshot(
        snapshot_id="pricing-v1",
        provider_id="model-provider",
        endpoint_type="responses",
        model_id="model-v1",
        effective_at=datetime(2026, 9, 6, tzinfo=UTC),
        currency="USD",
        input_tokens_per_million_usd=Decimal(0),
        output_tokens_per_million_usd=Decimal(0),
        cached_tokens_per_million_usd=Decimal(0),
        reasoning_tokens_per_million_usd=Decimal(0),
    )
    usage = ResourceUsage.zero(cost_known=True)
    runs: list[ExperimentTaskRun] = []
    binding_paths: dict[int, Path] = {}
    for repeat_id in (1, 2):
        key = ExperimentRunner.idempotency_key(
            "group", "end_to_end", "D", "task-a", None, repeat_id, "medium"
        )
        manifest = RunManifest.create(
            {
                "schema_version": "run-manifest-v1",
                "run_id": f"run-{repeat_id}",
                "thread_id": f"thread-{repeat_id}",
                "code_commit": "a" * 40,
                "dependency_lock_sha256": "b" * 64,
                "request_sha256": "c" * 64,
                "config_sha256": "d" * 64,
                "workflow_id": "research-v1",
                "graph_version": "graph-v1",
                "planner_id": "P2",
                "provider_profiles": (),
                "model_ids": (),
                "prompt_versions": {},
                "parser_versions": {},
                "ranker_id": "R2",
                "ranker_weights_version": None,
                "budget": RunBudget.preset("medium"),
                "usage": usage,
                "usage_by_node": {},
                "pricing_status": "estimated",
                "pricing_snapshots": (pricing,),
                "provider_calls": (),
                "node_executions": (),
                "parsed_artifacts": (),
                "evidence_hashes": (),
                "source_snapshot_ids": (),
                "artifact_ids": (),
                "run_event_count": 0,
                "run_events_sha256": "e" * 64,
                "seed": None,
                "seed_supported": False,
                "cache_hit_count": 0,
                "stop_reason": "SUFFICIENT",
                "is_partial": False,
                "failure_codes": (),
                "started_at": datetime(2026, 9, 6, tzinfo=UTC),
                "finished_at": datetime(2026, 9, 6, 0, 0, 1, tzinfo=UTC),
            }
        )
        manifest_path = artifacts / f"manifest-{repeat_id}.json"
        manifest_bytes = manifest.model_dump_json().encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        request_name = f"{key}.resume.json" if repeat_id == 1 else f"{key}.json"
        request_path = requests / request_name
        request_bytes = f"request-{repeat_id}".encode()
        request_path.write_bytes(request_bytes)
        runs.append(
            ExperimentTaskRun(
                task_id="task-a",
                protocol="end_to_end",
                variant=ExperimentVariant.D,
                planner_id="P2",
                ranker_id="R2",
                budget_preset="medium",
                repeat_id=repeat_id,
                status="completed",
                manifest_path=str(manifest_path),
                artifact_ids=(),
                usage=usage,
                pricing_snapshot_ids=("pricing-v1",),
                pricing_status="estimated",
                cost_label="estimated_from_normalized_schedule",
                metrics={},
            )
        )
        binding = {
            "schema_version": "unseeded-replication-binding-v1",
            "group_id": "group",
            "task_id": "task-a",
            "protocol": "end_to_end",
            "variant": "D",
            "budget_preset": "medium",
            "repeat_id": repeat_id,
            "request_sha256": hashlib.sha256(request_bytes).hexdigest(),
            "receipt_identity": "d" * 64,
            "manifest_provenance": {
                "manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
                "run_id": manifest.run_id,
                "thread_id": manifest.thread_id,
            },
        }
        binding_path = bindings / f"{binding['request_sha256']}.json"
        binding_path.write_bytes(canonical_json_bytes(binding))
        binding_paths[repeat_id] = binding_path

    # A resume request has the same canonical raw idempotency key but a
    # distinct request filename.  Summary must accept that binding when the
    # other repeat is not part of the selected raw set.
    second_binding = binding_paths[2].read_bytes()
    binding_paths[2].unlink()
    try:
        assert len(_load_replication_bindings(root, group=group, runs=[runs[0]])) == 1
    finally:
        binding_paths[2].write_bytes(second_binding)

    with pytest.raises(ValueError, match="receipt identity|duplicate"):
        _load_replication_bindings(root, group=group, runs=runs)


def test_summary_rejects_unseeded_completed_records_without_replication_binding(
    tmp_path: Path,
) -> None:
    root = _write_full_group(tmp_path / "group")
    group_path = root / "group.json"
    group = json.loads(group_path.read_bytes())
    group["replication"] = {
        "mode": "independent_repeats",
        "seed_supported": False,
        "seed_values": [],
        "repeat_ids": [1],
    }
    group_path.write_text(json.dumps(group), encoding="utf-8")
    for path in tuple((root / "raw").glob("*.json")):
        payload = json.loads(path.read_bytes())
        payload["seed"] = None
        payload["repeat_id"] = 1
        renamed = root / "raw" / (
            ExperimentRunner.idempotency_key(
                group["group_id"],
                payload["protocol"],
                payload["variant"],
                payload["task_id"],
                None,
                1,
                payload["budget_preset"],
            )
            + ".json"
        )
        path.unlink()
        renamed.write_bytes(json.dumps(payload, sort_keys=True).encode("utf-8"))

    with pytest.raises(ValueError, match="replication binding|sidecar"):
        summarize_experiment(root, bootstrap_resamples=4)


def test_summary_verify_only_rejects_output_artifact_symlink(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    summarize_experiment(root, bootstrap_resamples=4)
    target = tmp_path / "summary-target.json"
    target.write_bytes((root / "summary.json").read_bytes())
    (root / "summary.json").unlink()
    (root / "summary.json").symlink_to(target)

    with pytest.raises(ValueError, match="symlink|reparse|artifact"):
        summarize_experiment(root, verify_only=True)


def test_ranker_confidence_uses_paired_bootstrap_not_observed_extrema() -> None:
    records: list[SeedRunRecord] = []
    differences = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5)
    for index, difference in enumerate(differences):
        task_id = f"task-{index}"
        for variant, value in (("R1", 0.4), ("R2", 0.4 + difference)):
            metric = MetricValue(
                name="citation_support_precision",
                value=value,
                numerator=value,
                denominator=1.0,
                version="benchmark-metrics-v1",
            )
            records.append(
                SeedRunRecord(
                    task_id=task_id,
                    category=TaskCategory.TECHNICAL_SURVEY,
                    variant=variant,
                    budget_preset="medium",
                    seed=1,
                    metrics={"citation_support_precision": metric},
                )
            )

    interval = compare_rankers(records, budget_preset="medium", n_resamples=2_000, seed=7)
    assert interval.estimate == pytest.approx(sum(differences) / len(differences))
    assert interval.lower > min(differences)
    assert interval.upper < max(differences)
