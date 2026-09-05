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
    Path(__file__).parents[3]
    / "benchmarks"
    / "datasets"
    / "templates"
    / "question.example.json"
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
        assert all(set(record) == {
            "task_id",
            "category",
            "request",
            "evaluation_cutoff",
            "snapshot_id",
            "corpus_version",
            "index_version",
        } for record in [*dev_records, *test_records])


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

    public_payload = json.loads(result.public_manifest_path.read_bytes())
    public_payload["private_manifest_sha256"] = "0" * 64
    result.public_manifest_path.write_text(json.dumps(public_payload), encoding="utf-8")
    report = validator.validate_dataset(
        result.private_manifest_path,
        public_manifest_path=result.public_manifest_path,
    )
    assert "public private_manifest_sha256 does not match" in report.errors


def test_finalize_recovers_after_public_manifest_publish_failure(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
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
