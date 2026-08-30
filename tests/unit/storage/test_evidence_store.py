import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deepresearch.domain import EvidenceSpan, HtmlLocator, SourceDocument
from deepresearch.storage import EvidenceConflictError, EvidenceIntegrityError, LocalEvidenceStore


def _source(*, source_id: str = "source-1", text: str = "short text") -> SourceDocument:
    from deepresearch.retrieval import sha256_text

    return SourceDocument(
        source_id=source_id,
        canonical_url="https://example.com/source",
        title="Example",
        authors=("Author",),
        published_at=None,
        retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
        content_hash="a" * 64,
        parsed_content_hash=sha256_text(text),
        source_type="news",
        source_family_id="family-1",
        parser_version="parser-1",
    )


def _evidence(
    *,
    source_id: str = "source-1",
    evidence_id: str = "evidence-1",
    start_char: int = 0,
    end_char: int = 5,
    excerpt: str = "short",
    excerpt_hash: str | None = None,
) -> EvidenceSpan:
    from deepresearch.retrieval import sha256_text

    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=HtmlLocator(
            paragraph_id="paragraph-1",
            start_char=start_char,
            end_char=end_char,
        ),
        excerpt=excerpt,
        excerpt_hash=excerpt_hash or sha256_text(excerpt),
        language="en",
        information_need_ids=("need-1",),
    )


def test_evidence_store_rejects_locator_hash_mismatch(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    store.put_source(source, normalized_text="short text")

    with pytest.raises(EvidenceIntegrityError, match="excerpt_hash"):
        store.put_evidence(_evidence(excerpt_hash="0" * 64))


def test_evidence_store_validates_source_parsed_hash_and_identity(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)

    with pytest.raises(EvidenceIntegrityError, match="parsed_content_hash"):
        store.put_source(_source(text="different"), normalized_text="short text")

    source = _source(source_id="../source")
    assert store.put_source(source, normalized_text="short text") == source
    assert store.get_source("../source") == source
    assert not (tmp_path.parent / "source").exists()


def test_evidence_store_validates_source_existence_locator_excerpt_and_bounds(
    tmp_path: Path,
) -> None:
    store = LocalEvidenceStore(tmp_path)
    store.put_source(_source(), normalized_text="short text")

    with pytest.raises(EvidenceIntegrityError, match="source"):
        store.put_evidence(_evidence(source_id="missing"))
    with pytest.raises(EvidenceIntegrityError, match="bounds"):
        store.put_evidence(_evidence(end_char=20))
    with pytest.raises(EvidenceIntegrityError, match="excerpt"):
        store.put_evidence(_evidence(excerpt="wrong"))


def test_evidence_store_round_trips_full_source_payload_and_evidence(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    evidence = _evidence()

    assert store.put_source(source, normalized_text="short text") == source
    assert store.put_evidence(evidence) == evidence
    assert store.get_source(source.source_id) == source
    assert store.get_evidence(evidence.evidence_id) == evidence
    assert store.has_evidence(evidence.evidence_id)
    assert not store.has_evidence("absent")


def test_evidence_store_duplicate_ids_are_idempotent_but_conflicts_are_rejected(
    tmp_path: Path,
) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    store.put_source(source, normalized_text="short text")
    evidence = _evidence()
    store.put_evidence(evidence)

    assert store.put_source(source, normalized_text="short text") == source
    assert store.put_evidence(evidence) == evidence
    conflicting_source = source.model_copy(update={"title": "Conflicting title"})
    conflicting_evidence = evidence.model_copy(update={"language": "fr"})
    with pytest.raises(EvidenceConflictError):
        store.put_source(conflicting_source, normalized_text="short text")
    with pytest.raises(EvidenceConflictError):
        store.put_evidence(conflicting_evidence)


def test_evidence_store_concurrent_writers_are_idempotent(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    evidence = _evidence()

    with ThreadPoolExecutor(max_workers=8) as executor:
        sources = tuple(executor.map(lambda _: store.put_source(source, normalized_text="short text"), range(16)))
    with ThreadPoolExecutor(max_workers=8) as executor:
        spans = tuple(executor.map(lambda _: store.put_evidence(evidence), range(16)))

    assert sources == (source,) * 16
    assert spans == (evidence,) * 16
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


def test_evidence_store_detects_persisted_record_corruption(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    store.put_source(source, normalized_text="short text")
    source_path = next((tmp_path / "sources").rglob("*.json"))
    record = json.loads(source_path.read_text(encoding="utf-8"))
    payload = record.get("payload", record)
    payload["source"]["title"] = "tampered-but-valid"
    source_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(EvidenceIntegrityError, match="corrupt"):
        store.get_source(source.source_id)
