from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot

from .models import AnnotatedQuestion, PrivateDatasetManifest, TaskCategory

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256(value: str, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase SHA-256")
    return value


class _ReportModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    errors: tuple[str, ...]

    @field_validator("sha256", check_fields=False)
    @classmethod
    def validate_sha256(cls, value: str) -> str:
        return _sha256(value, field="sha256")

    @property
    def valid(self) -> bool:
        return not self.errors


class BatchValidationReport(_ReportModel):
    batch_id: str
    category: TaskCategory
    record_count: int = Field(ge=0)
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sha256: str


class DatasetValidationReport(_ReportModel):
    dataset_id: str
    version: str
    record_count: int = Field(ge=0)
    split_counts: dict[Literal["dev", "test"], int]
    category_counts: dict[TaskCategory, int]
    batch_reports: tuple[BatchValidationReport, ...]
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    sha256: str


def _stable_messages(messages: Sequence[str]) -> tuple[str, ...]:
    return tuple(sorted(set(messages)))


def _question_key(question: AnnotatedQuestion) -> str:
    return " ".join(question.request.question.casefold().split())


def read_annotated_questions(path: Path) -> tuple[AnnotatedQuestion, ...]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"batch file does not exist: {source}")
    questions: list[AnnotatedQuestion] = []
    for line_number, line in enumerate(source.read_bytes().splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {source}:{line_number}")
        try:
            questions.append(AnnotatedQuestion.model_validate_json(line, strict=True))
        except (TypeError, ValueError, ValidationError) as error:
            raise ValueError(f"invalid AnnotatedQuestion at {source}:{line_number}") from error
    return tuple(questions)


def _batch_report(
    *,
    batch_id: str,
    category: TaskCategory,
    record_count: int,
    errors: Sequence[str],
    warnings: Sequence[str] = (),
) -> BatchValidationReport:
    stable_errors = _stable_messages(errors)
    stable_warnings = _stable_messages(warnings)
    payload: dict[str, object] = {
        "batch_id": batch_id,
        "category": category,
        "record_count": record_count,
        "errors": stable_errors,
        "warnings": stable_warnings,
    }
    return BatchValidationReport(
        batch_id=batch_id,
        category=category,
        record_count=record_count,
        errors=stable_errors,
        warnings=stable_warnings,
        sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


def _dataset_report(
    *,
    dataset_id: str,
    version: str,
    record_count: int,
    split_counts: dict[Literal["dev", "test"], int],
    category_counts: dict[TaskCategory, int],
    batch_reports: Sequence[BatchValidationReport],
    errors: Sequence[str],
    warnings: Sequence[str] = (),
) -> DatasetValidationReport:
    stable_errors = _stable_messages(errors)
    stable_warnings = _stable_messages(warnings)
    payload: dict[str, object] = {
        "dataset_id": dataset_id,
        "version": version,
        "record_count": record_count,
        "split_counts": split_counts,
        "category_counts": category_counts,
        "batch_reports": [report.model_dump(mode="json") for report in batch_reports],
        "errors": stable_errors,
        "warnings": stable_warnings,
    }
    return DatasetValidationReport(
        dataset_id=dataset_id,
        version=version,
        record_count=record_count,
        split_counts=split_counts,
        category_counts=category_counts,
        batch_reports=tuple(batch_reports),
        errors=stable_errors,
        warnings=stable_warnings,
        sha256=sha256_bytes(canonical_json_bytes(payload)),
    )


class DatasetValidator:
    def __init__(self, *, snapshot_root: Path) -> None:
        self.snapshot_root = Path(snapshot_root)

    def _validate_snapshot(self, question: AnnotatedQuestion) -> tuple[str, ...]:
        snapshot_dir = self.snapshot_root / question.task_id
        try:
            snapshot = FrozenCorpusSnapshot.load(snapshot_dir, task_id=question.task_id)
        except ProviderError as error:
            return (f"task {question.task_id}: snapshot unavailable ({error.code})",)
        manifest = snapshot.manifest
        errors: list[str] = []
        if manifest.snapshot_id != question.snapshot_id:
            errors.append(f"task {question.task_id}: snapshot_id does not match")
        if manifest.corpus_version != question.corpus_version:
            errors.append(f"task {question.task_id}: corpus_version does not match")
        if manifest.index_version != question.index_version:
            errors.append(f"task {question.task_id}: index_version does not match")
        source_ids = {record.source_id for record in snapshot.records}
        source_families = {record.source_family_id for record in snapshot.records}
        if not set(question.candidate_source_ids) <= source_ids:
            errors.append(f"task {question.task_id}: candidate source is absent from snapshot")
        if not set(question.gold_source_family_ids) <= source_families:
            errors.append(f"task {question.task_id}: gold source family is absent from snapshot")
        for span in question.gold_evidence_spans:
            try:
                record = snapshot.record(span.evidence_id)
            except ProviderError:
                errors.append(f"task {question.task_id}: gold evidence is absent from snapshot")
                continue
            if (
                record.source_id != span.source_id
                or record.locator != span.locator
                or record.excerpt_hash != span.excerpt_hash
            ):
                errors.append(f"task {question.task_id}: gold evidence disagrees with snapshot")
        return _stable_messages(errors)

    def validate_records(
        self,
        records: Sequence[AnnotatedQuestion],
        *,
        batch_id: str,
        expected_category: TaskCategory,
        expected_count: int = 10,
    ) -> BatchValidationReport:
        if type(expected_count) is not int or expected_count <= 0:
            raise ValueError("expected_count must be a positive integer")
        errors: list[str] = []
        if len(records) != expected_count:
            errors.append(f"expected {expected_count} records, got {len(records)}")
        task_ids = [record.task_id for record in records]
        if len(task_ids) != len(set(task_ids)):
            errors.append("duplicate immutable task_id")
        question_keys = [_question_key(record) for record in records]
        if len(question_keys) != len(set(question_keys)):
            errors.append("duplicate semantic question")
        for record in records:
            if record.category != expected_category:
                errors.append(
                    f"task {record.task_id}: category is {record.category}, expected {expected_category}"
                )
            errors.extend(self._validate_snapshot(record))
        return _batch_report(
            batch_id=batch_id,
            category=expected_category,
            record_count=len(records),
            errors=errors,
        )

    def validate_batch(
        self,
        path: Path,
        *,
        expected_category: TaskCategory,
        expected_count: int = 10,
    ) -> BatchValidationReport:
        batch_path = Path(path)
        try:
            records = read_annotated_questions(batch_path)
        except (OSError, ValueError, TypeError) as error:
            return _batch_report(
                batch_id=batch_path.stem or "invalid",
                category=expected_category,
                record_count=0,
                errors=(str(error),),
            )
        return self.validate_records(
            records,
            batch_id=batch_path.stem,
            expected_category=expected_category,
            expected_count=expected_count,
        )

    def validate_dataset(self, private_manifest_path: Path) -> DatasetValidationReport:
        manifest_path = Path(private_manifest_path)
        empty_counts: dict[Literal["dev", "test"], int] = {"dev": 0, "test": 0}
        empty_categories = {category: 0 for category in TaskCategory}
        try:
            private_manifest = PrivateDatasetManifest.model_validate_json(
                manifest_path.read_bytes(), strict=True
            )
        except (OSError, ValueError, TypeError, ValidationError) as error:
            return _dataset_report(
                dataset_id="invalid",
                version="invalid",
                record_count=0,
                split_counts=empty_counts,
                category_counts=empty_categories,
                batch_reports=(),
                errors=(str(error),),
            )

        private_root = manifest_path.parent
        reports: list[BatchValidationReport] = []
        records: list[AnnotatedQuestion] = []
        errors: list[str] = []
        expected_files = {f"{category.value}.jsonl" for category in TaskCategory}
        batch_root = private_root / "batches"
        actual_files: set[str] = (
            {path.name for path in batch_root.glob("*.jsonl")} if batch_root.is_dir() else set()
        )
        if actual_files != expected_files:
            errors.append("dataset must contain exactly six category batch files")
        for category in TaskCategory:
            batch_path = batch_root / f"{category.value}.jsonl"
            report = self.validate_batch(
                batch_path,
                expected_category=category,
                expected_count=10,
            )
            reports.append(report)
            if not report.valid:
                errors.extend(f"{category.value}: {error}" for error in report.errors)
            try:
                batch_records = read_annotated_questions(batch_path)
            except (OSError, ValueError, TypeError):
                continue
            records.extend(batch_records)
            expected_hash = private_manifest.batch_sha256.get(category)
            actual_hash = sha256_bytes(batch_path.read_bytes())
            if expected_hash != actual_hash:
                errors.append(f"{category.value}: batch hash does not match private manifest")

        split_counts: dict[Literal["dev", "test"], int] = {
            "dev": sum(record.split == "dev" for record in records),
            "test": sum(record.split == "test" for record in records),
        }
        category_counts = {
            category: sum(record.category == category for record in records)
            for category in TaskCategory
        }
        if len(records) != 60:
            errors.append(f"expected 60 records, got {len(records)}")
        if split_counts != {"dev": 30, "test": 30}:
            errors.append("expected split counts dev=30 test=30")
        expected_categories = {category: 10 for category in TaskCategory}
        if category_counts != expected_categories:
            errors.append("expected ten records for each category")
        if private_manifest.record_count != len(records):
            errors.append("private manifest record_count does not match batches")
        if private_manifest.split_counts != split_counts:
            errors.append("private manifest split_counts do not match batches")
        if private_manifest.category_counts != category_counts:
            errors.append("private manifest category_counts do not match batches")
        task_ids = [record.task_id for record in records]
        if len(task_ids) != len(set(task_ids)):
            errors.append("duplicate immutable task_id across dataset")
        question_keys = [_question_key(record) for record in records]
        if len(question_keys) != len(set(question_keys)):
            errors.append("duplicate semantic question across dataset")

        if set(private_manifest.batch_sha256) != set(TaskCategory):
            errors.append("batch hash categories must match the six dataset categories")

        snapshot_locks = private_manifest.snapshot_manifest_sha256
        if set(snapshot_locks) != set(task_ids):
            errors.append("snapshot lock task IDs must match dataset task IDs")
        for record in records:
            manifest_file = self.snapshot_root / record.task_id / "manifest.sha256"
            try:
                digest = sha256_bytes(manifest_file.read_bytes())
            except OSError:
                errors.append(f"task {record.task_id}: snapshot manifest is missing")
                continue
            if snapshot_locks.get(record.task_id) != digest:
                errors.append(f"task {record.task_id}: snapshot manifest hash does not match")

        test_ids = {record.task_id for record in records if record.split == "test"}
        if private_manifest.main_test_task_ids != tuple(sorted(test_ids)):
            errors.append("main_test_task_ids do not match the frozen test split")
        if private_manifest.stability_task_ids != private_manifest.cost_subset_task_ids:
            errors.append("stability_task_ids must equal cost_subset_task_ids")
        if len(private_manifest.cost_subset_task_ids) != 20:
            errors.append("expected exactly 20 test-only cost tasks")
        for label, ids, expected_size in (
            ("stability_task_ids", private_manifest.stability_task_ids, 20),
            ("cost_subset_task_ids", private_manifest.cost_subset_task_ids, 20),
            ("p0_task_ids", private_manifest.p0_task_ids, 10),
            ("oracle_task_ids", private_manifest.oracle_task_ids, 10),
        ):
            if len(ids) != expected_size or not set(ids) <= test_ids:
                errors.append(f"{label} must contain exactly {expected_size} test task IDs")

        return _dataset_report(
            dataset_id=private_manifest.dataset_id,
            version=private_manifest.version,
            record_count=len(records),
            split_counts=split_counts,
            category_counts=category_counts,
            batch_reports=reports,
            errors=errors,
        )


app = typer.Typer(add_completion=False, no_args_is_help=True)
_DEFAULT_SNAPSHOT_ROOT = Path("benchmarks/snapshots/frozen_ai_cs_60")


@app.command("validate-batch")
def validate_batch_command(
    batch: Annotated[Path, typer.Option(...)],
    category: Annotated[TaskCategory, typer.Option(...)],
    expected_count: Annotated[int, typer.Option()] = 10,
    snapshot_root: Annotated[Path, typer.Option()] = _DEFAULT_SNAPSHOT_ROOT,
) -> None:
    report = DatasetValidator(snapshot_root=snapshot_root).validate_batch(
        batch,
        expected_category=category,
        expected_count=expected_count,
    )
    typer.echo(report.model_dump_json())
    if not report.valid:
        raise typer.Exit(code=2)


@app.command("validate-dataset")
def validate_dataset_command(
    private_manifest: Annotated[Path, typer.Option(...)],
    snapshot_root: Annotated[Path, typer.Option()] = _DEFAULT_SNAPSHOT_ROOT,
) -> None:
    report = DatasetValidator(snapshot_root=snapshot_root).validate_dataset(private_manifest)
    typer.echo(report.model_dump_json())
    if not report.valid:
        raise typer.Exit(code=2)


if __name__ == "__main__":
    app()


__all__ = [
    "BatchValidationReport",
    "DatasetValidationReport",
    "DatasetValidator",
    "app",
    "canonical_json_bytes",
    "read_annotated_questions",
    "sha256_bytes",
]
