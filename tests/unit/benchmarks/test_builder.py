from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

import benchmarks.datasets.builder as builder_module
from benchmarks.datasets.builder import DatasetBuilder, DatasetFrozenError
from benchmarks.datasets.models import AnnotatedQuestion, FrozenEvidenceRecord, TaskCategory
from benchmarks.datasets.validator import DatasetValidator
from benchmarks.scripts.build_snapshot import main as build_snapshot

TEMPLATE = (
    Path(__file__).parents[3] / "benchmarks" / "datasets" / "templates" / "question.example.json"
)


def _question() -> AnnotatedQuestion:
    return AnnotatedQuestion.model_validate_json(TEMPLATE.read_bytes(), strict=True)


def _write_snapshot(question: AnnotatedQuestion, snapshot_root: Path) -> AnnotatedQuestion:
    updated_spans = []
    for span in question.gold_evidence_spans:
        text = "x" * span.locator.end_char
        updated_spans.append(
            span.model_copy(
                update={"excerpt_hash": hashlib.sha256(text.encode("utf-8")).hexdigest()}
            )
        )
    question = question.model_copy(update={"gold_evidence_spans": updated_spans})
    records: list[FrozenEvidenceRecord] = []
    for span, source_family_id in zip(
        question.gold_evidence_spans,
        question.gold_source_family_ids,
        strict=True,
    ):
        text = "x" * span.locator.end_char
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        records.append(
            FrozenEvidenceRecord(
                task_id=question.task_id,
                evidence_id=span.evidence_id,
                source_id=span.source_id,
                source_family_id=source_family_id,
                canonical_url=f"https://example.test/{question.task_id}/{span.source_id}",
                title=f"Source {span.source_id}",
                authors=(),
                media_type="text/html",
                raw_body_b64=base64.b64encode(text.encode("utf-8")).decode("ascii"),
                content_hash=digest,
                normalized_text=text,
                parsed_content_hash=digest,
                locator_text=text,
                locator=span.locator,
                excerpt=text,
                excerpt_hash=digest,
                published_at=datetime(2026, 8, 29, tzinfo=UTC),
                retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
                language="en",
                source_type="paper",
            )
        )
    documents = snapshot_root / f"{question.task_id}.jsonl"
    documents.parent.mkdir(parents=True, exist_ok=True)
    documents.write_bytes(
        b"".join(
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
            for record in records
        )
    )
    snapshot = snapshot_root / question.task_id
    assert (
        build_snapshot(
            [
                "one",
                "--task-id",
                question.task_id,
                "--documents",
                str(documents),
                "--output",
                str(snapshot),
                "--corpus-version",
                question.corpus_version,
                "--index-version",
                question.index_version,
            ]
        )
        == 0
    )
    snapshot_id = json.loads((snapshot / "snapshot.json").read_bytes())["snapshot_id"]
    return question.model_copy(update={"snapshot_id": snapshot_id})


def _write_complete_private_dataset(private_root: Path, snapshot_root: Path) -> None:
    batches = private_root / "batches"
    batches.mkdir(parents=True)
    for category in TaskCategory:
        records = []
        for index in range(10):
            split = "dev" if index < 5 else "test"
            task_id = f"{split}-{category.value}-{index}"
            question = _question().model_copy(
                update={
                    "task_id": task_id,
                    "split": split,
                    "category": category,
                    "request": _question().request.model_copy(
                        update={"question": f"Question for {task_id}?"}
                    ),
                }
            )
            if category == TaskCategory.SOURCE_CONFLICT:
                payload = question.model_dump(mode="json")
                payload["gold_claim_links"][0]["evidence_links"].append(
                    {"evidence_id": "ev-pdf", "relation": "contradict"}
                )
                payload["gold_claim_links"][1]["evidence_links"][0]["relation"] = "support"
                question = AnnotatedQuestion.model_validate_json(json.dumps(payload), strict=True)
            records.append(_write_snapshot(question, snapshot_root))
        (batches / f"{category.value}.jsonl").write_bytes(
            b"".join(record.model_dump_json().encode("utf-8") + b"\n" for record in records)
        )


def test_finalize_refuses_to_replace_frozen_version(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    private_root.mkdir()
    public_root.mkdir()
    (public_root / "public_manifest.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DatasetFrozenError, match="new semantic version"):
        DatasetBuilder(snapshot_root=tmp_path / "snapshots").finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=tmp_path / "snapshots",
            subset_seed=20260829,
        )


def test_export_runtime_uses_the_redacted_runtime_task_schema(tmp_path: Path) -> None:
    output = tmp_path / "runtime.jsonl"
    DatasetBuilder(snapshot_root=tmp_path / "snapshots").export_runtime(
        (_question(),),
        output_path=output,
        include_split="dev",
    )
    payload = json.loads(output.read_bytes())
    assert set(payload) == {
        "task_id",
        "category",
        "request",
        "evaluation_cutoff",
        "snapshot_id",
        "corpus_version",
        "index_version",
    }


def test_finalize_rejects_batch_files_outside_the_six_categories(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    batches = private_root / "batches"
    batches.mkdir(parents=True)
    for category in TaskCategory:
        (batches / f"{category.value}.jsonl").write_bytes(b"")
    (batches / "unexpected.jsonl").write_bytes(b"")

    with pytest.raises(ValueError, match="exactly six category batch files"):
        DatasetBuilder(snapshot_root=tmp_path / "snapshots").finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=tmp_path / "public",
            snapshot_root=tmp_path / "snapshots",
            subset_seed=20260829,
        )


def test_finalize_exports_disjoint_dev_and_test_runtime_files(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)

    DatasetBuilder(snapshot_root=snapshot_root).finalize(
        dataset_id="frozen-ai-cs-60",
        version="1.0.0",
        private_root=private_root,
        public_root=public_root,
        snapshot_root=snapshot_root,
        subset_seed=20260829,
    )

    for category in TaskCategory:
        dev_records = [
            json.loads(line)
            for line in (public_root / "runtime" / "dev" / f"{category.value}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        test_records = [
            json.loads(line)
            for line in (private_root / "runtime" / "test" / f"{category.value}.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert len(dev_records) == len(test_records) == 5
        assert {record["task_id"] for record in dev_records}.isdisjoint(
            record["task_id"] for record in test_records
        )
        assert all(
            set(record)
            == {
                "task_id",
                "category",
                "request",
                "evaluation_cutoff",
                "snapshot_id",
                "corpus_version",
                "index_version",
            }
            for record in [*dev_records, *test_records]
        )


def test_finalize_validates_public_manifest_hash_chain_and_runtime_content(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    result = DatasetBuilder(snapshot_root=snapshot_root).finalize(
        dataset_id="frozen-ai-cs-60",
        version="1.0.0",
        private_root=private_root,
        public_root=public_root,
        snapshot_root=snapshot_root,
        subset_seed=20260829,
    )

    validator = DatasetValidator(snapshot_root=snapshot_root)
    assert validator.validate_dataset(
        result.private_manifest_path,
        public_manifest_path=result.public_manifest_path,
    ).valid
    commit = json.loads((private_root / ".dataset_commit.json").read_bytes())
    assert commit["dataset_id"] == "frozen-ai-cs-60"
    assert commit["private_manifest_sha256"]
    assert commit["public_manifest_sha256"]

    public_bytes = result.public_manifest_path.read_bytes()
    public_payload = json.loads(public_bytes)
    public_payload["private_manifest_sha256"] = "0" * 64
    result.public_manifest_path.write_text(json.dumps(public_payload), encoding="utf-8")
    assert validator.validate_private_preflight(result.private_manifest_path).valid
    report = validator.validate_dataset(
        result.private_manifest_path,
        public_manifest_path=result.public_manifest_path,
    )
    assert report.valid is False
    assert "public private_manifest_sha256 does not match" in report.errors

    result.public_manifest_path.write_bytes(public_bytes)
    runtime_path = public_root / "runtime" / "dev" / f"{TaskCategory.TECHNICAL_SURVEY.value}.jsonl"
    runtime_path.write_bytes(runtime_path.read_bytes() + b" ")
    report = validator.validate_dataset(
        result.private_manifest_path,
        public_manifest_path=result.public_manifest_path,
    )
    assert report.valid is False
    assert any("runtime file content does not match" in error for error in report.errors)


def test_finalize_recovers_after_public_manifest_publish_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    original_publish = builder_module._publish_file
    failed = False

    def fail_public_manifest(staging: Path, output: Path) -> None:
        nonlocal failed
        if output == public_root / "public_manifest.json" and not failed:
            failed = True
            raise OSError("simulated public manifest publish failure")
        original_publish(staging, output)

    monkeypatch.setattr(builder_module, "_publish_file", fail_public_manifest)
    with pytest.raises(OSError, match="simulated public manifest"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )
    assert not (private_root / ".dataset_commit.json").exists()
    monkeypatch.setattr(builder_module, "_publish_file", original_publish)

    result = DatasetBuilder(snapshot_root=snapshot_root).finalize(
        dataset_id="frozen-ai-cs-60",
        version="1.0.0",
        private_root=private_root,
        public_root=public_root,
        snapshot_root=snapshot_root,
        subset_seed=20260829,
    )
    assert result.public_manifest_path.is_file()
    assert not list(tmp_path.rglob("*.staging"))


def test_export_runtime_removes_staging_when_write_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "runtime.jsonl"
    original_write = builder_module._write_staging_file

    def write_then_fail(path: Path, payload: bytes) -> None:
        original_write(path, payload)
        raise OSError("simulated runtime write failure")

    monkeypatch.setattr(builder_module, "_write_staging_file", write_then_fail)
    with pytest.raises(OSError, match="simulated runtime write failure"):
        DatasetBuilder(snapshot_root=tmp_path / "snapshots").export_runtime(
            (_question(),), output_path=output, include_split="dev"
        )

    assert not output.exists()
    assert not list(tmp_path.rglob("*.staging"))


def test_finalize_validates_staged_joint_dataset_before_any_final_publish(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    original_runtime_payload = builder_module._runtime_payload

    def corrupt_staged_runtime(*args: object, **kwargs: object) -> bytes:
        return original_runtime_payload(*args, **kwargs) + b" "

    monkeypatch.setattr(builder_module, "_runtime_payload", corrupt_staged_runtime)
    with pytest.raises(ValueError, match="finalized dataset is invalid"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )

    assert not (private_root / "private_manifest.json").exists()
    assert not (public_root / "public_manifest.json").exists()
    assert not (private_root / ".dataset_commit.json").exists()
    assert not list(private_root.rglob("runtime"))
    assert not list(public_root.rglob("runtime"))


def test_finalize_requires_content_bound_distinct_review_across_categories(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    first_path = private_root / "batches" / "technical_survey.jsonl"
    second_path = private_root / "batches" / "method_comparison.jsonl"

    def replace_question(path: Path, question: str) -> AnnotatedQuestion:
        records = [
            AnnotatedQuestion.model_validate_json(line, strict=True)
            for line in path.read_bytes().splitlines()
        ]
        replacement = records[0].model_copy(
            update={"request": records[0].request.model_copy(update={"question": question})}
        )
        records[0] = replacement
        path.write_bytes(
            b"".join(record.model_dump_json().encode("utf-8") + b"\n" for record in records)
        )
        return replacement

    first = replace_question(first_path, "Compare research agent planner routes")
    second = replace_question(second_path, "Contrast research agent planner routes")
    with pytest.raises(ValueError, match="semantic review artifact is required"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )

    (private_root / "semantic_reviews.json").write_text(
        json.dumps(
            [
                {
                    "disposition": "distinct",
                    "question_sha256": [
                        builder_module._question_content_hash(second),
                        builder_module._question_content_hash(first),
                    ],
                    "task_ids": [first.task_id, second.task_id],
                }
            ]
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="semantic review entry is invalid"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )

    (private_root / "semantic_reviews.json").write_text(
        json.dumps(
            [
                {
                    "disposition": "distinct",
                    "question_sha256": [
                        builder_module._question_content_hash(first),
                        builder_module._question_content_hash(second),
                    ],
                    "task_ids": [first.task_id, second.task_id],
                }
            ]
        ),
        encoding="utf-8",
    )
    DatasetBuilder(snapshot_root=snapshot_root).finalize(
        dataset_id="frozen-ai-cs-60",
        version="1.0.0",
        private_root=private_root,
        public_root=public_root,
        snapshot_root=snapshot_root,
        subset_seed=20260829,
    )
    assert not (public_root / "semantic_reviews.json").exists()
    assert all(
        "semantic_reviews" not in path.read_text(encoding="utf-8")
        for path in (public_root / "runtime" / "dev").glob("*.jsonl")
    )


def test_finalize_rejects_review_when_prompt_content_changes(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    public_root = tmp_path / "public"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    first_path = private_root / "batches" / "technical_survey.jsonl"
    second_path = private_root / "batches" / "method_comparison.jsonl"

    def replace_question(path: Path, question: str) -> AnnotatedQuestion:
        records = [
            AnnotatedQuestion.model_validate_json(line, strict=True)
            for line in path.read_bytes().splitlines()
        ]
        replacement = records[0].model_copy(
            update={"request": records[0].request.model_copy(update={"question": question})}
        )
        records[0] = replacement
        path.write_bytes(
            b"".join(record.model_dump_json().encode("utf-8") + b"\n" for record in records)
        )
        return replacement

    first = replace_question(first_path, "Compare research agent planner routes")
    second = replace_question(second_path, "Contrast research agent planner routes")
    (private_root / "semantic_reviews.json").write_text(
        json.dumps(
            [
                {
                    "disposition": "distinct",
                    "question_sha256": [
                        builder_module._question_content_hash(first),
                        builder_module._question_content_hash(second),
                    ],
                    "task_ids": [first.task_id, second.task_id],
                }
            ]
        ),
        encoding="utf-8",
    )
    replace_question(first_path, "Compare research agent planner pathways")

    with pytest.raises(ValueError, match="semantic review entry is invalid"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )


def test_finalize_private_staging_is_contained_in_private_root(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private-dataset"
    public_root = tmp_path / "public-dataset"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    staging_paths: list[Path] = []

    def record_private_staging_then_fail(path: Path, payload: bytes) -> None:
        if "test" in path.parts and "runtime" in path.parts:
            staging_paths.append(path.resolve())
            raise OSError("stop after private staging path allocation")

    monkeypatch.setattr(builder_module, "_write_staging_file", record_private_staging_then_fail)
    with pytest.raises(OSError, match="stop after private staging"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=public_root,
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )

    assert staging_paths
    assert all(path.is_relative_to(private_root.resolve()) for path in staging_paths)
    assert not list(private_root.rglob("*.staging"))


def test_finalize_preserves_existing_private_staging_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    staging_root = private_root / ".dataset-staging-sentinel.staging"
    staging_root.mkdir()
    sentinel = staging_root / "keep.txt"
    sentinel.write_text("must survive", encoding="utf-8")

    class FixedUUID:
        hex = "sentinel"

    monkeypatch.setattr(builder_module.uuid, "uuid4", lambda: FixedUUID())
    with pytest.raises(FileExistsError, match="private staging path already exists"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=tmp_path / "public",
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )

    assert sentinel.read_text(encoding="utf-8") == "must survive"


def test_finalize_preserves_existing_private_staging_symlink(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    private_root = tmp_path / "private"
    snapshot_root = tmp_path / "snapshots"
    _write_complete_private_dataset(private_root, snapshot_root)
    sentinel_root = tmp_path / "symlink-target"
    sentinel_root.mkdir()
    sentinel = sentinel_root / "keep.txt"
    sentinel.write_text("must survive", encoding="utf-8")
    staging_root = private_root / ".dataset-staging-sentinel.staging"
    try:
        staging_root.symlink_to(sentinel_root, target_is_directory=True)
    except OSError:
        pytest.skip("directory symlinks are not permitted on this platform")

    class FixedUUID:
        hex = "sentinel"

    monkeypatch.setattr(builder_module.uuid, "uuid4", lambda: FixedUUID())
    with pytest.raises(FileExistsError, match="private staging path already exists"):
        DatasetBuilder(snapshot_root=snapshot_root).finalize(
            dataset_id="frozen-ai-cs-60",
            version="1.0.0",
            private_root=private_root,
            public_root=tmp_path / "public",
            snapshot_root=snapshot_root,
            subset_seed=20260829,
        )

    assert staging_root.is_symlink()
    assert sentinel.read_text(encoding="utf-8") == "must survive"
