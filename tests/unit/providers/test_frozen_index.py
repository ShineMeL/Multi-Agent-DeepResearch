from __future__ import annotations

import base64
import hashlib
from datetime import UTC, datetime

import pytest

from benchmarks.datasets.models import FrozenEvidenceRecord
from deepresearch.domain import HtmlLocator
from deepresearch.providers.frozen_index import FrozenBm25Index, deterministic_tokens


def _record(evidence_id: str, source_id: str, text: str) -> FrozenEvidenceRecord:
    raw = text.encode("utf-8")
    digest = hashlib.sha256(raw).hexdigest()
    return FrozenEvidenceRecord(
        task_id="task-fixture",
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


def test_tokenizer_handles_english_and_cjk_deterministically() -> None:
    assert deterministic_tokens("Multi-Agent 多模态 Agent") == (
        "multi",
        "agent",
        "多",
        "模",
        "态",
        "agent",
    )


def test_index_search_ties_use_evidence_then_source_id() -> None:
    records = (
        _record("ev-2", "src-1", "equal score"),
        _record("ev-1", "src-2", "equal score"),
    )
    index = FrozenBm25Index.build(records, index_version="bm25-fixture-v1")
    hits = index.search("equal score", limit=10)
    assert [(hit.record.evidence_id, hit.record.source_id) for hit in hits] == [
        ("ev-1", "src-2"),
        ("ev-2", "src-1"),
    ]


def test_index_rejects_invalid_limit() -> None:
    index = FrozenBm25Index.build(
        (_record("ev-1", "src-1", "one"),), index_version="bm25-fixture-v1"
    )
    with pytest.raises(ValueError, match="limit"):
        index.search("one", limit=0)
