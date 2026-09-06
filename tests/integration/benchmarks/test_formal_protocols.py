from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path

import pytest

from benchmarks.datasets.builder import DatasetBuilder, DatasetFrozenError
from benchmarks.datasets.models import (
    DatasetManifest,
    PrivateDatasetManifest,
    RuntimeTask,
    TaskCategory,
)
from benchmarks.datasets.validator import DatasetValidator, canonical_json_bytes

ROOT = Path(__file__).resolve().parents[3]
PUBLIC = ROOT / "benchmarks/datasets/frozen_ai_cs_60"
PRIVATE = ROOT / "benchmarks/private/frozen_ai_cs_60"
SNAPSHOTS = ROOT / "benchmarks/snapshots/frozen_ai_cs_60"


def _public_manifest() -> DatasetManifest:
    path = PUBLIC / "public_manifest.json"
    assert path.is_file(), "Task 11 requires a finalized public manifest"
    return DatasetManifest.model_validate_json(path.read_bytes(), strict=True)


def _private_manifest() -> PrivateDatasetManifest:
    public_manifest = _public_manifest()
    if not (PRIVATE / "batches").is_dir():
        pytest.skip("Evaluator-only dataset is not installed")
    path = PRIVATE / "private_manifest.json"
    assert path.is_file(), "Installed private batches must have a finalized manifest"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == public_manifest.private_manifest_sha256
    return PrivateDatasetManifest.model_validate_json(path.read_bytes(), strict=True)


def test_formal_dataset_has_six_balanced_categories() -> None:
    public_manifest = _public_manifest()
    assert public_manifest.dataset_id == "frozen_ai_cs_60"
    assert public_manifest.version == "1.0.0"
    assert public_manifest.record_count == 60
    assert public_manifest.split_counts == {"dev": 30, "test": 30}
    assert public_manifest.category_counts == {category: 10 for category in TaskCategory}


def test_public_manifest_contains_only_safe_metadata() -> None:
    public_manifest = _public_manifest()
    assert set(json.loads((PUBLIC / "public_manifest.json").read_bytes())) == {
        "dataset_id",
        "version",
        "record_count",
        "split_counts",
        "category_counts",
        "public_runtime_files",
        "private_manifest_sha256",
        "snapshot_collection_sha256",
        "cost_subset_sha256",
        "created_at",
    }
    assert set(public_manifest.public_runtime_files) == {
        f"runtime/dev/{category.value}.jsonl" for category in TaskCategory
    }
    assert not (PUBLIC / "runtime/test").exists()


def test_locked_subsets_are_test_only_and_stratified() -> None:
    private_manifest = _private_manifest()
    assert private_manifest.subset_seed == 20260829
    assert private_manifest.stability_task_ids == private_manifest.cost_subset_task_ids
    assert len(private_manifest.main_test_task_ids) == 30
    categories = {
        task.task_id: task.category
        for relative in private_manifest.private_test_runtime_files
        for line in (PRIVATE / relative).read_bytes().splitlines()
        if (task := RuntimeTask.model_validate_json(line, strict=True))
    }
    assert set(categories) == set(private_manifest.main_test_task_ids)
    assert all(task_id.startswith("test-") for task_id in categories)
    for subset, quotas in (
        (private_manifest.stability_task_ids, (4, 4, 3, 3, 3, 3)),
        (private_manifest.p0_task_ids, (2, 2, 2, 2, 1, 1)),
        (private_manifest.oracle_task_ids, (2, 2, 2, 2, 1, 1)),
    ):
        assert len(subset) == len(set(subset)) == sum(quotas)
        assert set(subset) <= set(categories)
        assert Counter(categories[task_id] for task_id in subset) == dict(
            zip(TaskCategory, quotas, strict=True)
        )


def test_formal_dataset_validates_all_batches_snapshots_and_runtime() -> None:
    private_manifest = _private_manifest()
    report = DatasetValidator(snapshot_root=SNAPSHOTS).validate_dataset(
        PRIVATE / "private_manifest.json", public_manifest_path=PUBLIC / "public_manifest.json"
    )
    assert report.valid, report.errors
    assert report.record_count == 60
    assert report.split_counts == {"dev": 30, "test": 30}
    assert len(report.batch_reports) == 6
    assert len(private_manifest.snapshot_manifest_sha256) == 60
    assert {path.name for path in (PRIVATE / "batches").glob("*.jsonl")} == {
        f"{category.value}.jsonl" for category in TaskCategory
    }


def test_runtime_split_boundary_and_public_hash_chain() -> None:
    public_manifest = _public_manifest()
    private_manifest = _private_manifest()
    for root, files, prefix in (
        (PUBLIC, public_manifest.public_runtime_files, "dev-"),
        (PRIVATE, private_manifest.private_test_runtime_files, "test-"),
    ):
        assert len(files) == 6
        all_ids: set[str] = set()
        for relative in files:
            path = root / relative
            assert path.resolve().is_relative_to(root.resolve())
            lines = path.read_bytes().splitlines()
            assert len(lines) == 5
            for line in lines:
                task = RuntimeTask.model_validate_json(line, strict=True)
                assert task.task_id.startswith(prefix)
                assert task.task_id not in all_ids
                all_ids.add(task.task_id)
        assert len(all_ids) == 30
    assert (
        public_manifest.snapshot_collection_sha256
        == hashlib.sha256(
            canonical_json_bytes(dict(sorted(private_manifest.snapshot_manifest_sha256.items())))
        ).hexdigest()
    )
    assert (
        public_manifest.cost_subset_sha256
        == hashlib.sha256(
            canonical_json_bytes(list(private_manifest.cost_subset_task_ids))
        ).hexdigest()
    )
    marker = json.loads((PRIVATE / ".dataset_commit.json").read_bytes())
    assert marker == {
        "dataset_id": "frozen_ai_cs_60",
        "version": "1.0.0",
        "private_manifest_sha256": public_manifest.private_manifest_sha256,
        "public_manifest_sha256": hashlib.sha256(
            (PUBLIC / "public_manifest.json").read_bytes()
        ).hexdigest(),
    }


def test_frozen_version_refuses_replacement() -> None:
    private_manifest = _private_manifest()
    before = {
        path: hashlib.sha256(path.read_bytes()).hexdigest()
        for root in (PUBLIC, PRIVATE / "runtime/test")
        for path in root.rglob("*")
        if path.is_file()
    }
    with pytest.raises(DatasetFrozenError, match="new semantic version"):
        DatasetBuilder(snapshot_root=SNAPSHOTS).finalize(
            dataset_id=private_manifest.dataset_id,
            version=private_manifest.version,
            private_root=PRIVATE,
            public_root=PUBLIC,
            snapshot_root=SNAPSHOTS,
            subset_seed=20260829,
        )
    assert all(
        hashlib.sha256(path.read_bytes()).hexdigest() == digest for path, digest in before.items()
    )
