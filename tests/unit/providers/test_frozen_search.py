from __future__ import annotations

import base64
import hashlib
import time
from datetime import UTC, datetime
from pathlib import Path

import pytest

import deepresearch.providers.frozen_search as frozen_search_module
from benchmarks.datasets.models import FrozenEvidenceRecord
from deepresearch.domain import HtmlLocator
from deepresearch.providers import ParsedBlock, ParsedDocument, SearchProvider
from deepresearch.providers.errors import ProviderError
from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from deepresearch.providers.frozen_search import (
    FrozenCorpusFetcher,
    FrozenCorpusMaterializer,
    FrozenCorpusSearchProvider,
)
from deepresearch.runtime import CancellationToken, OperationCancelled

SNAPSHOT_ROOT = Path(__file__).parents[2] / "fixtures" / "frozen_corpus" / "task-fixture"


def _deadline() -> float:
    return time.monotonic() + 10.0


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


@pytest.fixture
def snapshot() -> FrozenCorpusSnapshot:
    return FrozenCorpusSnapshot.load(SNAPSHOT_ROOT, task_id="task-fixture")


@pytest.mark.asyncio
async def test_ties_use_evidence_then_source_id(snapshot: FrozenCorpusSnapshot) -> None:
    provider = FrozenCorpusSearchProvider(snapshot)
    hits = await provider.search(
        "equal score",
        limit=10,
        filters=None,
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert [
        (hit.provider_metadata["evidence_id"], hit.provider_metadata["source_id"])
        for hit in hits
    ] == [("ev-1", "src-2"), ("ev-2", "src-1")]


@pytest.mark.asyncio
async def test_published_at_filter_accepts_canonical_iso_timestamp(
    snapshot: FrozenCorpusSnapshot,
) -> None:
    provider = FrozenCorpusSearchProvider(snapshot)
    hits = await provider.search(
        "equal score",
        limit=10,
        filters={"published_at": "2026-08-29T00:00:00Z"},
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert len(hits) == 2


def test_unknown_snapshot_never_calls_live(tmp_path: Path) -> None:
    with pytest.raises(ProviderError) as error:
        FrozenCorpusSearchProvider.from_snapshot(tmp_path / "missing", task_id="task-fixture")
    assert error.value.code in {"REPLAY_MISS", "INVALID_SNAPSHOT"}


@pytest.mark.asyncio
async def test_frozen_fetcher_returns_locked_raw_body(snapshot: FrozenCorpusSnapshot) -> None:
    record = snapshot.record("ev-1")
    raw = await FrozenCorpusFetcher(snapshot).fetch(
        str(record.canonical_url),
        deadline=_deadline(),
        cancellation_token=CancellationToken(),
    )
    assert hashlib.sha256(raw.body_bytes).hexdigest() == record.content_hash


@pytest.mark.asyncio
@pytest.mark.parametrize("operation", ["search", "fetch"])
async def test_frozen_adapters_recheck_cancellation_after_their_final_yield(
    snapshot: FrozenCorpusSnapshot,
    monkeypatch: pytest.MonkeyPatch,
    operation: str,
) -> None:
    token = CancellationToken()

    async def cancel_during_yield(_: float) -> None:
        token.cancel()

    monkeypatch.setattr(frozen_search_module.asyncio, "sleep", cancel_during_yield)
    if operation == "search":
        with pytest.raises(OperationCancelled):
            await FrozenCorpusSearchProvider(snapshot).search(
                "equal score",
                limit=10,
                filters=None,
                deadline=_deadline(),
                cancellation_token=token,
            )
    else:
        with pytest.raises(OperationCancelled):
            await FrozenCorpusFetcher(snapshot).fetch(
                str(snapshot.record("ev-1").canonical_url),
                deadline=_deadline(),
                cancellation_token=token,
            )


def test_materializer_preserves_snapshot_evidence_id_and_verifies_parse(
    snapshot: FrozenCorpusSnapshot,
) -> None:
    record = snapshot.record("ev-1")
    parsed = ParsedDocument(
        canonical_url=record.canonical_url,
        title=record.title,
        authors=(),
        published_at=record.published_at,
        normalized_text=record.normalized_text,
        blocks=(
            ParsedBlock(
                block_id="html-main-0",
                text=record.normalized_text,
                locator=record.locator,
                text_hash=record.parsed_content_hash,
            ),
        ),
        parser_id="fixture-parser",
        parser_version="fixture-parser-v1",
        parsed_content_hash=record.parsed_content_hash,
    )
    result = FrozenCorpusMaterializer(snapshot).materialize(
        selected_evidence_ids=("ev-1",),
        parsed_documents={record.source_id: parsed},
        information_need_ids=("need-runtime-1",),
    )
    assert result.evidence_spans[0].evidence_id == "ev-1"
    assert result.evidence_spans[0].information_need_ids == ("need-runtime-1",)
    assert result.source_documents[0].parser_version == "fixture-parser-v1"


def test_frozen_provider_implements_shared_search_protocol(snapshot: FrozenCorpusSnapshot) -> None:
    assert isinstance(FrozenCorpusSearchProvider(snapshot), SearchProvider)
