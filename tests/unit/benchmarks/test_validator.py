from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from typer.testing import CliRunner

from benchmarks.datasets.models import AnnotatedQuestion, PrivateDatasetManifest, TaskCategory
from benchmarks.datasets.validator import DatasetValidator, app

TEMPLATE = (
    Path(__file__).parents[3]
    / "benchmarks"
    / "datasets"
    / "templates"
    / "question.example.json"
)


def _question_payload() -> dict[str, object]:
    return json.loads(TEMPLATE.read_bytes())


def _question() -> AnnotatedQuestion:
    return AnnotatedQuestion.model_validate_json(TEMPLATE.read_bytes(), strict=True)


def test_batch_requires_exactly_ten_records(tmp_path: Path) -> None:
    batch_path = tmp_path / "technical_survey.jsonl"
    batch_path.write_bytes(
        json.dumps(_question_payload(), ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
    )
    report = DatasetValidator(snapshot_root=tmp_path / "snapshots").validate_batch(
        batch_path,
        expected_category=TaskCategory.TECHNICAL_SURVEY,
        expected_count=10,
    )
    assert report.valid is False
    assert "expected 10 records" in " ".join(report.errors)


def test_dataset_requires_six_categories_and_thirty_thirty_split(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    incomplete = PrivateDatasetManifest(
        dataset_id="frozen-ai-cs-60",
        version="1.0.0",
        record_count=1,
        split_counts={"dev": 1, "test": 0},
        category_counts={TaskCategory.TECHNICAL_SURVEY: 1},
        batch_sha256={TaskCategory.TECHNICAL_SURVEY: "a" * 64},
        snapshot_manifest_sha256={},
        public_runtime_files=[],
        private_test_runtime_files=[],
        main_test_task_ids=(),
        stability_task_ids=(),
        cost_subset_task_ids=(),
        p0_task_ids=(),
        oracle_task_ids=(),
        subset_seed=20260829,
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    manifest_path = private_root / "private_manifest.json"
    manifest_path.write_bytes(
        json.dumps(incomplete.model_dump(mode="json"), sort_keys=True).encode("utf-8") + b"\n"
    )
    report = DatasetValidator(snapshot_root=tmp_path / "snapshots").validate_private_preflight(
        manifest_path
    )
    assert report.valid is False
    assert report.category_counts != {category: 10 for category in TaskCategory}
    assert "expected split counts dev=30 test=30" in " ".join(report.errors)


def test_dataset_rejects_snapshot_locks_for_unreferenced_tasks(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    manifest = PrivateDatasetManifest(
        dataset_id="frozen-ai-cs-60",
        version="1.0.0",
        record_count=0,
        split_counts={"dev": 0, "test": 0},
        category_counts={},
        batch_sha256={TaskCategory.TECHNICAL_SURVEY: "a" * 64},
        snapshot_manifest_sha256={"unreferenced-task": "b" * 64},
        public_runtime_files=[],
        private_test_runtime_files=[],
        main_test_task_ids=(),
        stability_task_ids=(),
        cost_subset_task_ids=(),
        p0_task_ids=(),
        oracle_task_ids=(),
        subset_seed=20260829,
        created_at=datetime(2026, 8, 29, tzinfo=UTC),
    )
    manifest_path = private_root / "private_manifest.json"
    manifest_path.write_bytes(manifest.model_dump_json().encode("utf-8"))

    report = DatasetValidator(snapshot_root=tmp_path / "snapshots").validate_private_preflight(
        manifest_path
    )

    assert "batch hash categories must match the six dataset categories" in report.errors
    assert "snapshot lock task IDs must match dataset task IDs" in report.errors


def test_validate_records_revalidates_mutated_frozen_models(tmp_path: Path) -> None:
    record = _question()
    record.information_needs.clear()

    report = DatasetValidator(snapshot_root=tmp_path / "snapshots").validate_records(
        (record,),
        batch_id="mutated",
        expected_category=TaskCategory.TECHNICAL_SURVEY,
        expected_count=1,
    )

    assert "invalid AnnotatedQuestion" in " ".join(report.errors)


def test_duplicate_question_normalization_handles_unicode_and_punctuation(tmp_path: Path) -> None:
    first = _question()
    second = first.model_copy(
        update={
            "task_id": "dev-ts-unicode-duplicate",
            "request": first.request.model_copy(update={"question": "Re\u0301sume\u0301 -- methods"}),
        }
    )
    first = first.model_copy(
        update={"request": first.request.model_copy(update={"question": "R\u00e9sum\u00e9: methods!"})}
    )

    report = DatasetValidator(snapshot_root=tmp_path / "snapshots").validate_records(
        (first, second),
        batch_id="unicode",
        expected_category=TaskCategory.TECHNICAL_SURVEY,
        expected_count=2,
    )

    assert "duplicate semantic question" in report.errors


def test_distinct_questions_emit_auditable_manual_semantic_review_warning(tmp_path: Path) -> None:
    first = _question()
    second = first.model_copy(
        update={
            "task_id": "dev-ts-synonym-review",
            "request": first.request.model_copy(
                update={"question": "Contrast the primary planner routes for research agents."}
            ),
        }
    )

    report = DatasetValidator(snapshot_root=tmp_path / "snapshots").validate_records(
        (first, second),
        batch_id="synonym-review",
        expected_category=TaskCategory.TECHNICAL_SURVEY,
        expected_count=2,
    )

    assert any("manual semantic review" in warning for warning in report.warnings)


def test_validate_dataset_cli_accepts_manifest_option(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        ["validate-dataset", "--manifest", str(tmp_path / "missing-private-manifest.json")],
    )

    assert result.exit_code == 2
    assert json.loads(result.output)["record_count"] == 0
