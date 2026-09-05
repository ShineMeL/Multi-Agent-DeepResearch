from __future__ import annotations

import base64
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from benchmarks.datasets.models import FrozenEvidenceRecord
from benchmarks.scripts.build_snapshot import _publish_no_replace, main
from deepresearch.domain import HtmlLocator
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot


def _record(evidence_id: str, source_id: str, text: str) -> FrozenEvidenceRecord:
    raw = text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return FrozenEvidenceRecord(
        task_id="task-builder",
        evidence_id=evidence_id,
        source_id=source_id,
        source_family_id=source_id,
        canonical_url=f"https://example.test/{source_id}",
        title=f"Source {source_id}",
        authors=(),
        media_type="text/html",
        raw_body_b64=base64.b64encode(raw).decode("ascii"),
        content_hash=digest,
        normalized_text=text,
        parsed_content_hash=digest,
        locator_text=text,
        locator=HtmlLocator(paragraph_id="main-0", start_char=0, end_char=len(text)),
        excerpt=text,
        excerpt_hash=digest,
        published_at=datetime(2026, 8, 29, tzinfo=UTC),
        retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
        language="en",
        source_type="paper",
    )


def _write_records(path: Path) -> None:
    records = (_record("ev-2", "src-2", "two"), _record("ev-1", "src-1", "one"))
    payload = b"".join(
        (
            json.dumps(
                record.model_dump(mode="json"),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
            + b"\n"
        )
        for record in records
    )
    path.write_bytes(payload)


def test_build_one_is_deterministic_and_refuses_overwrite(tmp_path: Path) -> None:
    documents = tmp_path / "documents.jsonl"
    _write_records(documents)
    first = tmp_path / "snapshot-one"
    second = tmp_path / "snapshot-two"
    args = [
        "one",
        "--task-id",
        "task-builder",
        "--documents",
        str(documents),
        "--output",
        str(first),
        "--corpus-version",
        "corpus-v1",
        "--index-version",
        "bm25-v1",
    ]
    assert main(args) == 0
    assert FrozenCorpusSnapshot.load(first, task_id="task-builder").manifest.document_count == 2
    assert main(args) == 4

    second_args = [*args]
    second_args[second_args.index(str(first))] = str(second)
    assert main(second_args) == 0
    assert {
        filename: (first / filename).read_bytes()
        for filename in ("documents.jsonl", "index.json", "snapshot.json", "manifest.sha256")
    } == {
        filename: (second / filename).read_bytes()
        for filename in ("documents.jsonl", "index.json", "snapshot.json", "manifest.sha256")
    }


def test_build_one_reports_missing_documents_as_snapshot_failure(tmp_path: Path) -> None:
    result = main(
        [
            "one",
            "--task-id",
            "task-builder",
            "--documents",
            str(tmp_path / "missing.jsonl"),
            "--output",
            str(tmp_path / "snapshot"),
            "--corpus-version",
            "corpus-v1",
            "--index-version",
            "bm25-v1",
        ]
    )
    assert result == 4


def test_publish_no_replace_refuses_an_existing_snapshot_directory(tmp_path: Path) -> None:
    staging = tmp_path / "staging"
    output = tmp_path / "output"
    staging.mkdir()
    output.mkdir()
    with pytest.raises(FileExistsError):
        _publish_no_replace(staging, output)
    assert staging.is_dir()
    assert output.is_dir()
