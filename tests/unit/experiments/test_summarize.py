from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.evaluators.metrics import MetricValue
from deepresearch.domain import ResourceUsage
from experiments.models import (
    EvaluatorReferenceManifest,
    ExperimentTaskRun,
    OracleReferenceResult,
    canonical_sha256,
)
from experiments.runner import ExperimentRunner
from experiments.summarize import summarize_experiment


def _write_full_group(root: Path) -> Path:
    (root / "raw").mkdir(parents=True)
    group = {
        "group_id": "group",
        "private_manifest_sha256": "b" * 64,
        "evaluator_version": "evaluator-v1",
        "protocols": ["ranker_component", "planner_policy", "end_to_end", "reference"],
        "protocol_task_ids": {
            "ranker_component": ["task-a"],
            "planner_policy": ["task-a"],
            "end_to_end": ["task-a"],
            "reference": ["task-a"],
        },
        "oracle_task_ids": ["task-a"],
        "cost_subset_task_ids": ["task-a"],
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
        "replication": {"mode": "seeds", "seed_values": [1], "repeat_ids": []},
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
                task_id="task-a",
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
            )
            key = ExperimentRunner.idempotency_key(
                "group", protocol, variant, "task-a", 1, None, "medium"
            )
            (root / "raw" / f"{key}.json").write_bytes(
                json.dumps(run.model_dump(mode="json"), sort_keys=True).encode("utf-8")
            )
    created_at = datetime(2026, 9, 6, tzinfo=UTC)
    oracle_result = OracleReferenceResult(
        task_id="task-a",
        dataset_version="dataset-v1",
        private_manifest_sha256="b" * 64,
        frozen_snapshot_id="snapshot-task-a",
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
        group_id="group",
        private_manifest_sha256=oracle_result.private_manifest_sha256,
        evaluator_version=oracle_result.evaluator_version,
        task_ids_sha256=canonical_sha256(("task-a",)),
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
    verified = summarize_experiment(root, verify_only=True)
    assert verified["sections"]["end_to_end"]


def test_summary_rejects_raw_filename_that_is_not_the_sealed_identity(tmp_path: Path) -> None:
    root = _write_full_group(tmp_path / "group")
    raw_paths = sorted((root / "raw").glob("*.json"))
    renamed = root / "raw" / "not-the-sealed-id.json"
    raw_paths[0].rename(renamed)

    with pytest.raises(ValueError, match="idempotency key"):
        summarize_experiment(root, bootstrap_resamples=4)
