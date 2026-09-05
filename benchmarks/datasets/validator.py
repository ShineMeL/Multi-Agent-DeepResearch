from __future__ import annotations

import hashlib
import json
import random
import re
import unicodedata
from collections import Counter
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot

from .isolation import GoldIsolationGuard
from .models import (
    AnnotatedQuestion,
    DatasetManifest,
    PrivateDatasetManifest,
    RuntimeTask,
    TaskCategory,
)

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(
            value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False
        )
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
    normalized = unicodedata.normalize("NFKC", question.request.question).casefold()
    return " ".join(re.findall(r"[^\W_]+", normalized))


def _parse_annotated_questions(source: Path, payload: bytes) -> tuple[AnnotatedQuestion, ...]:
    questions: list[AnnotatedQuestion] = []
    for line_number, line in enumerate(payload.splitlines(), start=1):
        if not line.strip():
            raise ValueError(f"blank JSONL line at {source}:{line_number}")
        try:
            questions.append(AnnotatedQuestion.model_validate_json(line, strict=True))
        except (TypeError, ValueError, ValidationError) as error:
            raise ValueError(f"invalid AnnotatedQuestion at {source}:{line_number}") from error
    return tuple(questions)


def read_annotated_questions_with_bytes(path: Path) -> tuple[bytes, tuple[AnnotatedQuestion, ...]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"batch file does not exist: {source}")
    payload = source.read_bytes()
    return payload, _parse_annotated_questions(source, payload)


def read_annotated_questions(path: Path) -> tuple[AnnotatedQuestion, ...]:
    return read_annotated_questions_with_bytes(path)[1]


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


def _expected_subset(
    records: Sequence[AnnotatedQuestion],
    *,
    quotas: tuple[int, int, int, int, int, int],
    seed: int,
    label: str,
) -> tuple[str, ...]:
    by_category: dict[TaskCategory, list[str]] = {category: [] for category in TaskCategory}
    for record in records:
        if record.split == "test":
            by_category[record.category].append(record.task_id)
    selected: list[str] = []
    for category, quota in zip(TaskCategory, quotas, strict=True):
        candidates = sorted(by_category[category])
        if len(candidates) < quota:
            return ()
        randomizer = random.Random(f"{seed}:{label}:{category.value}")
        randomizer.shuffle(candidates)
        selected.extend(candidates[:quota])
    return tuple(sorted(selected))


def _runtime_payload(records: Sequence[AnnotatedQuestion]) -> bytes:
    return b"".join(
        canonical_json_bytes(GoldIsolationGuard.runtime_view(record).model_dump(mode="json"))
        for record in sorted(records, key=lambda item: item.task_id)
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
        warnings: list[str] = []
        if len(records) != expected_count:
            errors.append(f"expected {expected_count} records, got {len(records)}")
        normalized_records: list[AnnotatedQuestion] = []
        for index, record in enumerate(records, start=1):
            if type(record) is not AnnotatedQuestion:
                errors.append(f"record {index}: must be an AnnotatedQuestion")
                continue
            try:
                normalized_records.append(
                    AnnotatedQuestion.model_validate(record.model_dump(mode="python"), strict=True)
                )
            except (TypeError, ValueError, ValidationError):
                errors.append(f"record {index}: invalid AnnotatedQuestion after revalidation")
        task_ids = [record.task_id for record in normalized_records]
        if len(task_ids) != len(set(task_ids)):
            errors.append("duplicate immutable task_id")
        question_keys = [_question_key(record) for record in normalized_records]
        if len(question_keys) != len(set(question_keys)):
            errors.append("duplicate semantic question")
        for first_index, first in enumerate(normalized_records):
            for second in normalized_records[first_index + 1 :]:
                if _question_key(first) != _question_key(second):
                    warnings.append(
                        "manual semantic review required for distinct prompts: "
                        f"{first.task_id}, {second.task_id}"
                    )
        for record in normalized_records:
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
            warnings=warnings,
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
            _, records = read_annotated_questions_with_bytes(batch_path)
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

    def validate_dataset(
        self,
        private_manifest_path: Path,
        *,
        public_manifest_path: Path | None = None,
    ) -> DatasetValidationReport:
        manifest_path = Path(private_manifest_path)
        empty_counts: dict[Literal["dev", "test"], int] = {"dev": 0, "test": 0}
        empty_categories = {category: 0 for category in TaskCategory}
        try:
            private_bytes = manifest_path.read_bytes()
            private_manifest = PrivateDatasetManifest.model_validate_json(private_bytes, strict=True)
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
            try:
                batch_bytes, batch_records = read_annotated_questions_with_bytes(batch_path)
            except (OSError, ValueError, TypeError) as error:
                report = _batch_report(
                    batch_id=batch_path.stem or "invalid",
                    category=category,
                    record_count=0,
                    errors=(str(error),),
                )
                batch_bytes = None
                batch_records = ()
            else:
                report = self.validate_records(
                    batch_records,
                    batch_id=batch_path.stem,
                    expected_category=category,
                    expected_count=10,
                )
            reports.append(report)
            if not report.valid:
                errors.extend(f"{category.value}: {error}" for error in report.errors)
            records.extend(batch_records)
            expected_hash = private_manifest.batch_sha256.get(category)
            if batch_bytes is None or expected_hash != sha256_bytes(batch_bytes):
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

        expected_subsets = (
            ("stability_task_ids", private_manifest.stability_task_ids, (4, 4, 3, 3, 3, 3), "stability-cost"),
            ("cost_subset_task_ids", private_manifest.cost_subset_task_ids, (4, 4, 3, 3, 3, 3), "stability-cost"),
            ("p0_task_ids", private_manifest.p0_task_ids, (2, 2, 2, 2, 1, 1), "p0"),
            ("oracle_task_ids", private_manifest.oracle_task_ids, (2, 2, 2, 2, 1, 1), "oracle"),
        )
        for label, ids, quotas, selection_label in expected_subsets:
            expected_ids = _expected_subset(
                records,
                quotas=quotas,
                seed=private_manifest.subset_seed,
                label=selection_label,
            )
            if expected_ids and ids != expected_ids:
                errors.append(f"{label} does not match its deterministic seed selection")
            actual_quotas: Counter[TaskCategory] = Counter(
                record.category for record in records if record.task_id in set(ids)
            )
            expected_quotas: Counter[TaskCategory] = Counter()
            for category, quota in zip(TaskCategory, quotas, strict=True):
                expected_quotas[category] = quota
            if actual_quotas != expected_quotas:
                errors.append(f"{label} does not satisfy the category quotas")

        if public_manifest_path is not None:
            public_path = Path(public_manifest_path)
            try:
                public_bytes = public_path.read_bytes()
                public_manifest = DatasetManifest.model_validate_json(public_bytes, strict=True)
            except (OSError, ValueError, TypeError, ValidationError) as error:
                errors.append(f"public manifest is invalid: {error}")
            else:
                if public_manifest.dataset_id != private_manifest.dataset_id:
                    errors.append("public dataset_id does not match private manifest")
                if public_manifest.version != private_manifest.version:
                    errors.append("public version does not match private manifest")
                if public_manifest.record_count != private_manifest.record_count:
                    errors.append("public record_count does not match private manifest")
                if public_manifest.split_counts != private_manifest.split_counts:
                    errors.append("public split_counts do not match private manifest")
                if public_manifest.category_counts != private_manifest.category_counts:
                    errors.append("public category_counts do not match private manifest")
                if public_manifest.private_manifest_sha256 != sha256_bytes(private_bytes):
                    errors.append("public private_manifest_sha256 does not match")
                if public_manifest.snapshot_collection_sha256 != sha256_bytes(
                    canonical_json_bytes(dict(sorted(snapshot_locks.items())))
                ):
                    errors.append("public snapshot_collection_sha256 does not match")
                if public_manifest.cost_subset_sha256 != sha256_bytes(
                    canonical_json_bytes(list(private_manifest.cost_subset_task_ids))
                ):
                    errors.append("public cost_subset_sha256 does not match")
                if public_manifest.public_runtime_files != private_manifest.public_runtime_files:
                    errors.append("public runtime file list does not match private manifest")
                errors.extend(
                    self._validate_runtime_files(
                        root=public_path.parent,
                        paths=public_manifest.public_runtime_files,
                        records=records,
                        split="dev",
                    )
                )
                errors.extend(
                    self._validate_runtime_files(
                        root=private_root,
                        paths=private_manifest.private_test_runtime_files,
                        records=records,
                        split="test",
                    )
                )

        return _dataset_report(
            dataset_id=private_manifest.dataset_id,
            version=private_manifest.version,
            record_count=len(records),
            split_counts=split_counts,
            category_counts=category_counts,
            batch_reports=reports,
            errors=errors,
        )

    def _validate_runtime_files(
        self,
        *,
        root: Path,
        paths: Sequence[str],
        records: Sequence[AnnotatedQuestion],
        split: Literal["dev", "test"],
    ) -> tuple[str, ...]:
        expected_paths = [f"runtime/{split}/{category.value}.jsonl" for category in TaskCategory]
        errors: list[str] = []
        if list(paths) != expected_paths:
            errors.append(f"{split} runtime files must name the six category files")
            return tuple(errors)
        for category, relative_path in zip(TaskCategory, paths, strict=True):
            relative = Path(relative_path)
            if relative.is_absolute() or ".." in relative.parts:
                errors.append(f"{split} runtime path is not relative: {relative_path}")
                continue
            target = root / relative
            expected_records = [
                record
                for record in records
                if record.category == category and record.split == split
            ]
            try:
                actual = target.read_bytes()
            except OSError:
                errors.append(f"{split} runtime file is missing: {relative_path}")
                continue
            if actual != _runtime_payload(expected_records):
                errors.append(f"{split} runtime file content does not match frozen records: {relative_path}")
                continue
            for line_number, line in enumerate(actual.splitlines(), start=1):
                try:
                    raw = json.loads(line)
                    raw_mapping = cast("dict[str, object]", raw)
                    if not isinstance(raw, dict) or set(raw_mapping) != set(RuntimeTask.model_fields):
                        raise ValueError("RuntimeTask fields are not exact")
                    RuntimeTask.model_validate_json(line, strict=True)
                except (TypeError, ValueError, ValidationError, json.JSONDecodeError):
                    errors.append(f"{split} runtime file has invalid RuntimeTask: {relative_path}:{line_number}")
        return _stable_messages(errors)


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
    manifest: Annotated[Path, typer.Option("--manifest", "--private-manifest")],
    snapshot_root: Annotated[Path, typer.Option()] = _DEFAULT_SNAPSHOT_ROOT,
    public_manifest: Annotated[Path | None, typer.Option()] = None,
) -> None:
    report = DatasetValidator(snapshot_root=snapshot_root).validate_dataset(
        manifest,
        public_manifest_path=public_manifest,
    )
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
