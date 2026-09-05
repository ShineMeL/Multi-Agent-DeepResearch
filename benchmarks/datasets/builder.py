from __future__ import annotations

import json
import os
import random
import re
import shutil
import unicodedata
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, time
from pathlib import Path
from typing import Annotated, Literal, cast

import typer
from pydantic import BaseModel, ConfigDict

from benchmarks.scripts.build_snapshot import (
    _publish_no_replace,  # pyright: ignore[reportPrivateUsage]
)

from .isolation import GoldIsolationGuard
from .models import AnnotatedQuestion, DatasetManifest, PrivateDatasetManifest, TaskCategory
from .validator import (
    BatchValidationReport,
    DatasetValidator,
    canonical_json_bytes,
    read_annotated_questions,
    read_annotated_questions_with_bytes,
    sha256_bytes,
)


class DatasetFrozenError(RuntimeError):
    """Raised when an immutable dataset version would be replaced."""


class DatasetFinalizeResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    public_manifest_path: Path
    private_manifest_path: Path
    public_manifest: DatasetManifest
    private_manifest: PrivateDatasetManifest


def _path_exists(path: Path) -> bool:
    return path.exists() or path.is_symlink()


def _write_staging_file(path: Path, payload: bytes) -> None:
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_file(staging: Path, output: Path) -> None:
    if _path_exists(output):
        if output.is_file() and output.read_bytes() == staging.read_bytes():
            staging.unlink()
            return
        raise DatasetFrozenError("new semantic version is required for existing dataset output")
    try:
        _publish_no_replace(staging, output)
    except FileExistsError as error:
        raise DatasetFrozenError(
            "new semantic version is required for existing dataset output"
        ) from error


def _runtime_payload(
    records: Sequence[AnnotatedQuestion], *, include_split: Literal["dev", "test"]
) -> bytes:
    runtime_records = [
        GoldIsolationGuard.runtime_view(record).model_dump(mode="json")
        for record in sorted(records, key=lambda item: item.task_id)
        if record.split == include_split
    ]
    return b"".join(canonical_json_bytes(record) for record in runtime_records)


def _select_test_subset(
    records_by_category: Mapping[TaskCategory, Sequence[AnnotatedQuestion]],
    *,
    quotas: tuple[int, int, int, int, int, int],
    seed: int,
    label: str,
) -> tuple[str, ...]:
    selected: list[str] = []
    for category, quota in zip(TaskCategory, quotas, strict=True):
        candidates = sorted(
            record.task_id for record in records_by_category[category] if record.split == "test"
        )
        if len(candidates) < quota:
            raise ValueError(f"not enough test records for {category}")
        randomizer = random.Random(f"{seed}:{label}:{category.value}")
        randomizer.shuffle(candidates)
        selected.extend(candidates[:quota])
    return tuple(sorted(selected))


def _question_content_hash(question: AnnotatedQuestion) -> str:
    text = unicodedata.normalize("NFKC", question.request.question)
    return sha256_bytes(text.encode("utf-8"))


def _review_tokens(question: AnnotatedQuestion) -> set[str]:
    stop_words = {
        "a",
        "an",
        "the",
        "for",
        "of",
        "in",
        "on",
        "to",
        "what",
        "question",
        "dev",
        "test",
        *(part for category in TaskCategory for part in category.value.split("_")),
    }
    tokens = re.findall(
        r"[^\W_]+", unicodedata.normalize("NFKC", question.request.question).casefold()
    )
    return {token for token in tokens if token not in stop_words and not token.isdecimal()}


def _semantic_review_candidates(
    records: Sequence[AnnotatedQuestion],
) -> tuple[tuple[AnnotatedQuestion, AnnotatedQuestion], ...]:
    candidates: list[tuple[AnnotatedQuestion, AnnotatedQuestion]] = []
    for index, first in enumerate(records):
        first_tokens = _review_tokens(first)
        for second in records[index + 1 :]:
            second_tokens = _review_tokens(second)
            union = first_tokens | second_tokens
            if union and len(first_tokens & second_tokens) / len(union) >= 0.5:
                candidates.append((first, second))
    return tuple(candidates)


def _require_semantic_reviews(private_root: Path, records: Sequence[AnnotatedQuestion]) -> None:
    review_path = private_root / "semantic_reviews.json"
    try:
        raw: object = json.loads(review_path.read_bytes())
    except OSError as error:
        if _semantic_review_candidates(records):
            raise ValueError(
                "semantic review artifact is required for candidate prompts"
            ) from error
        return
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ValueError("semantic review artifact is invalid") from error
    if not isinstance(raw, list):
        raise TypeError("semantic review artifact must be a list")
    reviewed: set[tuple[frozenset[str], frozenset[str]]] = set()
    for item in cast("list[object]", raw):
        if not isinstance(item, dict):
            raise TypeError("semantic review entry is invalid")
        entry = cast("dict[str, object]", item)
        if set(entry) != {"disposition", "question_sha256", "task_ids"}:
            raise ValueError("semantic review entry is invalid")
        task_ids = entry.get("task_ids")
        hashes = entry.get("question_sha256")
        if not isinstance(task_ids, list) or not isinstance(hashes, list):
            raise TypeError("semantic review entry is invalid")
        task_id_values = cast("list[object]", task_ids)
        hash_values = cast("list[object]", hashes)
        if (
            entry.get("disposition") != "distinct"
            or len(task_id_values) != 2
            or len(hash_values) != 2
            or any(not isinstance(value, str) for value in [*task_id_values, *hash_values])
        ):
            raise ValueError("semantic review entry is invalid")
        reviewed.add(
            (
                frozenset(cast("str", value) for value in task_id_values),
                frozenset(cast("str", value) for value in hash_values),
            )
        )
    for first, second in _semantic_review_candidates(records):
        expected = (
            frozenset((first.task_id, second.task_id)),
            frozenset((_question_content_hash(first), _question_content_hash(second))),
        )
        if expected not in reviewed:
            raise ValueError(f"unreviewed semantic candidate: {first.task_id}, {second.task_id}")


class DatasetBuilder:
    def __init__(self, *, snapshot_root: Path) -> None:
        self.snapshot_root = Path(snapshot_root)
        self.validator = DatasetValidator(snapshot_root=self.snapshot_root)

    def add_batch(
        self,
        batch_id: str,
        records: Sequence[AnnotatedQuestion],
        *,
        expected_category: TaskCategory,
        expected_count: int = 10,
    ) -> BatchValidationReport:
        if type(batch_id) is not str or not batch_id.strip():
            raise ValueError("batch_id must be non-empty")
        if any(type(record) is not AnnotatedQuestion for record in records):
            raise TypeError("records must contain AnnotatedQuestion values")
        return self.validator.validate_records(
            records,
            batch_id=batch_id,
            expected_category=expected_category,
            expected_count=expected_count,
        )

    def export_runtime(
        self,
        records: Sequence[AnnotatedQuestion],
        *,
        output_path: Path,
        include_split: Literal["dev", "test"],
    ) -> Path:
        if include_split not in {"dev", "test"}:
            raise ValueError("include_split must be dev or test")
        if any(type(record) is not AnnotatedQuestion for record in records):
            raise TypeError("records must contain AnnotatedQuestion values")
        target = Path(output_path).absolute()
        if not target.name:
            raise ValueError("output_path must name a file")
        staging: Path | None = None
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            staging = target.with_name(f".{target.name}.{os.getpid()}.{uuid.uuid4().hex}.staging")
            if _path_exists(staging):
                raise FileExistsError("runtime staging path already exists")
            _write_staging_file(
                staging,
                _runtime_payload(records, include_split=include_split),
            )
            _publish_file(staging, target)
        except Exception:
            if staging is not None and _path_exists(staging):
                staging.unlink()
            raise
        return target

    def finalize(
        self,
        *,
        dataset_id: str,
        version: str,
        private_root: Path,
        public_root: Path,
        snapshot_root: Path,
        subset_seed: int,
    ) -> DatasetFinalizeResult:
        if type(dataset_id) is not str or not dataset_id.strip():
            raise ValueError("dataset_id must be non-empty")
        if type(version) is not str or not version.strip():
            raise ValueError("version must be non-empty")
        if type(subset_seed) is not int:
            raise TypeError("subset_seed must be an integer")
        private_path = Path(private_root).absolute()
        public_path = Path(public_root).absolute()
        snapshot_path = Path(snapshot_root).absolute()
        private_manifest_path = private_path / "private_manifest.json"
        public_manifest_path = public_path / "public_manifest.json"
        commit_marker_path = private_path / ".dataset_commit.json"
        existing_private = _path_exists(private_manifest_path)
        existing_public = _path_exists(public_manifest_path)
        existing_commit = _path_exists(commit_marker_path)
        if existing_private:
            try:
                PrivateDatasetManifest.model_validate_json(
                    private_manifest_path.read_bytes(), strict=True
                )
            except (OSError, TypeError, ValueError):
                raise DatasetFrozenError(
                    "new semantic version is required for frozen dataset"
                ) from None
        if existing_public:
            try:
                DatasetManifest.model_validate_json(public_manifest_path.read_bytes(), strict=True)
            except (OSError, TypeError, ValueError):
                raise DatasetFrozenError(
                    "new semantic version is required for frozen dataset"
                ) from None

        batch_root = private_path / "batches"
        expected_batch_files = {f"{category.value}.jsonl" for category in TaskCategory}
        actual_batch_files: set[str] = (
            {path.name for path in batch_root.glob("*.jsonl")} if batch_root.is_dir() else set()
        )
        if actual_batch_files != expected_batch_files:
            raise ValueError("finalize requires exactly six category batch files")

        validator = DatasetValidator(snapshot_root=snapshot_path)
        records_by_category: dict[TaskCategory, tuple[AnnotatedQuestion, ...]] = {}
        batch_sha256: dict[TaskCategory, str] = {}
        for category in TaskCategory:
            batch_path = private_path / "batches" / f"{category.value}.jsonl"
            batch_bytes, records = read_annotated_questions_with_bytes(batch_path)
            report = validator.validate_records(
                records,
                batch_id=batch_path.stem,
                expected_category=category,
                expected_count=10,
            )
            if not report.valid:
                raise ValueError("batch is invalid: " + "; ".join(report.errors))
            records_by_category[category] = records
            batch_sha256[category] = sha256_bytes(batch_bytes)

        records = tuple(
            record for category in TaskCategory for record in records_by_category[category]
        )
        if len(records) != 60:
            raise ValueError("finalize requires exactly 60 records")
        if (
            sum(record.split == "dev" for record in records) != 30
            or sum(record.split == "test" for record in records) != 30
        ):
            raise ValueError("finalize requires a 30/30 dev/test split")
        if len({record.task_id for record in records}) != len(records):
            raise ValueError("finalize requires unique task IDs")
        _require_semantic_reviews(private_path, records)

        snapshot_hashes: dict[str, str] = {}
        for record in records:
            manifest = snapshot_path / record.task_id / "manifest.sha256"
            snapshot_hashes[record.task_id] = sha256_bytes(manifest.read_bytes())

        main_test_task_ids = tuple(
            sorted(record.task_id for record in records if record.split == "test")
        )
        stability_task_ids = _select_test_subset(
            records_by_category,
            quotas=(4, 4, 3, 3, 3, 3),
            seed=subset_seed,
            label="stability-cost",
        )
        p0_task_ids = _select_test_subset(
            records_by_category,
            quotas=(2, 2, 2, 2, 1, 1),
            seed=subset_seed,
            label="p0",
        )
        oracle_task_ids = _select_test_subset(
            records_by_category,
            quotas=(2, 2, 2, 2, 1, 1),
            seed=subset_seed,
            label="oracle",
        )
        created_at = datetime.combine(
            max(record.created_at for record in records), time.min, tzinfo=UTC
        )
        category_counts = {
            category: len(records_by_category[category]) for category in TaskCategory
        }
        public_runtime_files = [f"runtime/dev/{category.value}.jsonl" for category in TaskCategory]
        private_test_runtime_files = [
            f"runtime/test/{category.value}.jsonl" for category in TaskCategory
        ]
        private_manifest = PrivateDatasetManifest(
            dataset_id=dataset_id,
            version=version,
            record_count=len(records),
            split_counts={"dev": 30, "test": 30},
            category_counts=category_counts,
            batch_sha256=batch_sha256,
            snapshot_manifest_sha256=snapshot_hashes,
            public_runtime_files=public_runtime_files,
            private_test_runtime_files=private_test_runtime_files,
            main_test_task_ids=main_test_task_ids,
            stability_task_ids=stability_task_ids,
            cost_subset_task_ids=stability_task_ids,
            p0_task_ids=p0_task_ids,
            oracle_task_ids=oracle_task_ids,
            subset_seed=subset_seed,
            created_at=created_at,
        )
        private_bytes = canonical_json_bytes(private_manifest.model_dump(mode="json"))
        public_manifest = DatasetManifest(
            dataset_id=dataset_id,
            version=version,
            record_count=len(records),
            split_counts={"dev": 30, "test": 30},
            category_counts=category_counts,
            public_runtime_files=public_runtime_files,
            private_manifest_sha256=sha256_bytes(private_bytes),
            snapshot_collection_sha256=sha256_bytes(
                canonical_json_bytes(dict(sorted(snapshot_hashes.items())))
            ),
            cost_subset_sha256=sha256_bytes(canonical_json_bytes(list(stability_task_ids))),
            created_at=created_at,
        )
        public_bytes = canonical_json_bytes(public_manifest.model_dump(mode="json"))
        commit_bytes = canonical_json_bytes(
            {
                "dataset_id": dataset_id,
                "private_manifest_sha256": sha256_bytes(private_bytes),
                "public_manifest_sha256": sha256_bytes(public_bytes),
                "version": version,
            }
        )

        if existing_private and private_manifest_path.read_bytes() != private_bytes:
            raise DatasetFrozenError("new semantic version is required for existing dataset output")
        if existing_public and public_manifest_path.read_bytes() != public_bytes:
            raise DatasetFrozenError("new semantic version is required for existing dataset output")
        if existing_commit and commit_marker_path.read_bytes() != commit_bytes:
            raise DatasetFrozenError("existing dataset commit marker does not match output")
        if existing_commit and not (existing_private and existing_public):
            raise DatasetFrozenError("existing dataset commit marker has incomplete manifests")
        if existing_private and existing_public:
            report = validator.validate_dataset(
                private_manifest_path,
                public_manifest_path=public_manifest_path,
            )
            if report.valid:
                if existing_commit:
                    raise DatasetFrozenError("new semantic version is required for frozen dataset")
            else:
                raise DatasetFrozenError(
                    "existing finalized dataset is invalid and cannot be replaced"
                )

        private_stage_root = (
            private_path.parent / f".{private_path.name}.{uuid.uuid4().hex}.staging"
        )
        public_stage_root = public_path.parent / f".{public_path.name}.{uuid.uuid4().hex}.staging"
        commit_staging = private_stage_root / ".dataset_commit.json"
        staging_paths: list[tuple[Path, Path]] = []
        staging_files: list[Path] = [commit_staging]
        try:
            for category in TaskCategory:
                dev_target = public_path / "runtime" / "dev" / f"{category.value}.jsonl"
                test_target = private_path / "runtime" / "test" / f"{category.value}.jsonl"
                dev_staging = public_stage_root / "runtime" / "dev" / f"{category.value}.jsonl"
                test_staging = private_stage_root / "runtime" / "test" / f"{category.value}.jsonl"
                staging_paths.extend(((dev_staging, dev_target), (test_staging, test_target)))
                staging_files.extend((dev_staging, test_staging))
                dev_staging.parent.mkdir(parents=True, exist_ok=True)
                test_staging.parent.mkdir(parents=True, exist_ok=True)
                _write_staging_file(
                    dev_staging,
                    _runtime_payload(records_by_category[category], include_split="dev"),
                )
                _write_staging_file(
                    test_staging,
                    _runtime_payload(records_by_category[category], include_split="test"),
                )
            private_staging = private_stage_root / "private_manifest.json"
            public_staging = public_stage_root / "public_manifest.json"
            staging_paths.extend(
                (
                    (private_staging, private_manifest_path),
                    (public_staging, public_manifest_path),
                )
            )
            staging_files.extend((private_staging, public_staging))
            private_staging.parent.mkdir(parents=True, exist_ok=True)
            public_staging.parent.mkdir(parents=True, exist_ok=True)
            _write_staging_file(private_staging, private_bytes)
            _write_staging_file(public_staging, public_bytes)

            validation_staging = validator.validate_dataset(
                private_staging,
                public_manifest_path=public_staging,
                batch_root=batch_root,
                private_runtime_root=private_stage_root,
                public_runtime_root=public_stage_root,
            )
            if not validation_staging.valid:
                raise ValueError(
                    "finalized dataset is invalid: " + "; ".join(validation_staging.errors)
                )
            for _, target in staging_paths:
                target.parent.mkdir(parents=True, exist_ok=True)
            for staging, target in staging_paths:
                _publish_file(staging, target)
            report = validator.validate_dataset(
                private_manifest_path,
                public_manifest_path=public_manifest_path,
            )
            if not report.valid:
                raise ValueError("published dataset is invalid: " + "; ".join(report.errors))
            _write_staging_file(commit_staging, commit_bytes)
            _publish_file(commit_staging, commit_marker_path)
            if commit_marker_path.read_bytes() != commit_bytes:
                raise ValueError("published dataset commit marker does not match")
        except Exception:
            for staging in staging_files:
                if _path_exists(staging):
                    staging.unlink()
            raise
        finally:
            for stage_root in (private_stage_root, public_stage_root):
                if _path_exists(stage_root):
                    shutil.rmtree(stage_root)
        return DatasetFinalizeResult(
            public_manifest_path=public_manifest_path,
            private_manifest_path=private_manifest_path,
            public_manifest=public_manifest,
            private_manifest=private_manifest,
        )


app = typer.Typer(add_completion=False, no_args_is_help=True)
_DEFAULT_SNAPSHOT_ROOT = Path("benchmarks/snapshots/frozen_ai_cs_60")


@app.command("export-runtime")
def export_runtime_command(
    batch: Annotated[Path, typer.Option(...)],
    split: Annotated[Literal["dev", "test"], typer.Option(...)],
    output: Annotated[Path, typer.Option(...)],
    snapshot_root: Annotated[Path, typer.Option()] = _DEFAULT_SNAPSHOT_ROOT,
) -> None:
    records = read_annotated_questions(batch)
    DatasetBuilder(snapshot_root=snapshot_root).export_runtime(
        records,
        output_path=output,
        include_split=split,
    )


@app.command("finalize")
def finalize_command(
    dataset_id: Annotated[str, typer.Option(...)],
    version: Annotated[str, typer.Option(...)],
    private_root: Annotated[Path, typer.Option(...)],
    public_root: Annotated[Path, typer.Option(...)],
    snapshot_root: Annotated[Path, typer.Option()] = _DEFAULT_SNAPSHOT_ROOT,
    subset_seed: Annotated[int, typer.Option()] = 20260829,
) -> None:
    result = DatasetBuilder(snapshot_root=snapshot_root).finalize(
        dataset_id=dataset_id,
        version=version,
        private_root=private_root,
        public_root=public_root,
        snapshot_root=snapshot_root,
        subset_seed=subset_seed,
    )
    typer.echo(result.public_manifest.model_dump_json())


if __name__ == "__main__":
    app()


__all__ = ["DatasetBuilder", "DatasetFinalizeResult", "DatasetFrozenError", "app"]
