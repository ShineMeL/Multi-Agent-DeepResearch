from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.datasets.models import TaskCategory
from benchmarks.evaluators.metrics import MetricValue
from benchmarks.evaluators.statistics import SeedRunRecord, compare_rankers
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
        "replication": {
            "mode": "seeds",
            "seed_supported": True,
            "seed_values": [1],
            "repeat_ids": [],
        },
        "task_categories": {"task-a": "technical_survey"},
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
