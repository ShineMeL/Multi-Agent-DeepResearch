import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from deepresearch.domain import EvidenceSpan, HtmlLocator, PdfLocator, SourceDocument
from deepresearch.providers import ParsedBlock, ParsedDocument
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


def _parsed_document(
    *,
    text: str = "short text",
    blocks: tuple[ParsedBlock, ...] | None = None,
    canonical_url: str = "https://example.com/source",
) -> ParsedDocument:
    from deepresearch.retrieval import sha256_text

    default_blocks = (
        ParsedBlock(
            block_id="block-1",
            text=text,
            locator=HtmlLocator(
                paragraph_id="paragraph-1",
                start_char=0,
                end_char=len(text),
            ),
            text_hash=sha256_text(text),
        ),
    )
    return ParsedDocument(
        canonical_url=canonical_url,
        title="Example",
        authors=("Author",),
        normalized_text=text,
        blocks=blocks if blocks is not None else default_blocks,
        parser_id="parser",
        parser_version="parser-1",
        parsed_content_hash=sha256_text(text),
    )


def _html_block(block_id: str, paragraph_id: str, text: str) -> ParsedBlock:
    from deepresearch.retrieval import sha256_text

    return ParsedBlock(
        block_id=block_id,
        text=text,
        locator=HtmlLocator(
            paragraph_id=paragraph_id,
            start_char=0,
            end_char=len(text),
        ),
        text_hash=sha256_text(text),
    )


def _pdf_evidence(
    *,
    source_id: str,
    evidence_id: str,
    page_index: int,
    block_index: int,
    excerpt: str,
) -> EvidenceSpan:
    from deepresearch.retrieval import sha256_text

    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=source_id,
        locator=PdfLocator(
            page_index=page_index,
            block_index=block_index,
            start_char=0,
            end_char=len(excerpt),
        ),
        excerpt=excerpt,
        excerpt_hash=sha256_text(excerpt),
        language="en",
        information_need_ids=("need-1",),
    )


def _symlink_directory(link: Path, target: Path) -> None:
    target.mkdir(parents=True, exist_ok=True)
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError as error:
        pytest.skip(f"directory symlinks are unavailable: {error}")


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

    with pytest.raises(EvidenceIntegrityError, match="normalized"):
        store.put_source(_source(text=" not normalized "), normalized_text=" not normalized ")


def test_evidence_store_registers_only_matching_unique_parsed_structure(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    parsed = _parsed_document()

    with pytest.raises(EvidenceIntegrityError, match="source"):
        store.put_parsed_document("missing", parsed)

    source = _source()
    store.put_source(source, normalized_text="short text")
    with pytest.raises(EvidenceIntegrityError, match="canonical_url"):
        store.put_parsed_document(
            source.source_id,
            _parsed_document(canonical_url="https://other.example/source"),
        )
    with pytest.raises(EvidenceIntegrityError, match="parsed_content_hash"):
        store.put_parsed_document(source.source_id, _parsed_document(text="different text"))

    duplicate_html = _parsed_document(
        blocks=(
            parsed.blocks[0],
            parsed.blocks[0].model_copy(update={"block_id": "block-2"}),
        )
    )
    with pytest.raises(EvidenceIntegrityError, match="paragraph"):
        store.put_parsed_document(source.source_id, duplicate_html)

    assert store.put_parsed_document(source.source_id, parsed) == parsed
    assert store.put_parsed_document(source.source_id, parsed) == parsed
    with pytest.raises(EvidenceConflictError):
        store.put_parsed_document(
            source.source_id,
            parsed.model_copy(update={"title": "Conflicting parsed title"}),
        )


def test_evidence_store_rejects_unknown_html_container(tmp_path: Path) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    store.put_source(source, normalized_text="short text")
    with pytest.raises(EvidenceIntegrityError, match="parsed"):
        store.put_evidence(_evidence())
    store.put_parsed_document(source.source_id, _parsed_document())
    evidence = _evidence().model_copy(
        update={
            "locator": HtmlLocator(
                paragraph_id="unknown-paragraph",
                start_char=0,
                end_char=5,
            )
        }
    )

    with pytest.raises(EvidenceIntegrityError, match="paragraph|container"):
        store.put_evidence(evidence)


def test_evidence_store_validates_non_first_pdf_block_local_offsets(tmp_path: Path) -> None:
    from deepresearch.retrieval import sha256_text

    text = "alpha omega"
    source = _source(source_id="pdf-source", text=text)
    blocks = (
        ParsedBlock(
            block_id="pdf-0",
            text="alpha",
            locator=PdfLocator(page_index=0, block_index=0, start_char=0, end_char=5),
            text_hash=sha256_text("alpha"),
        ),
        ParsedBlock(
            block_id="pdf-1",
            text="omega",
            locator=PdfLocator(page_index=0, block_index=1, start_char=0, end_char=5),
            text_hash=sha256_text("omega"),
        ),
    )
    parsed = _parsed_document(text=text, blocks=blocks)
    store = LocalEvidenceStore(tmp_path)
    store.put_source(source, normalized_text=text)
    store.put_parsed_document(source.source_id, parsed)

    valid = _pdf_evidence(
        source_id=source.source_id,
        evidence_id="pdf-evidence",
        page_index=0,
        block_index=1,
        excerpt="omega",
    )
    assert store.put_evidence(valid) == valid

    unknown = _pdf_evidence(
        source_id=source.source_id,
        evidence_id="unknown-pdf-evidence",
        page_index=9,
        block_index=9,
        excerpt="alpha",
    )
    with pytest.raises(EvidenceIntegrityError, match="PDF|container|block"):
        store.put_evidence(unknown)


def test_evidence_store_rejects_duplicate_pdf_container_keys(tmp_path: Path) -> None:
    from deepresearch.retrieval import sha256_text

    source = _source(source_id="pdf-source", text="alpha omega")
    first = ParsedBlock(
        block_id="pdf-0",
        text="alpha",
        locator=PdfLocator(page_index=0, block_index=0, start_char=0, end_char=5),
        text_hash=sha256_text("alpha"),
    )
    duplicate = ParsedBlock(
        block_id="pdf-1",
        text="omega",
        locator=PdfLocator(page_index=0, block_index=0, start_char=0, end_char=5),
        text_hash=sha256_text("omega"),
    )
    store = LocalEvidenceStore(tmp_path)
    store.put_source(source, normalized_text="alpha omega")

    with pytest.raises(EvidenceIntegrityError, match="PDF|container"):
        store.put_parsed_document(
            source.source_id,
            _parsed_document(text="alpha omega", blocks=(first, duplicate)),
        )


def test_evidence_store_binds_ordered_html_blocks_to_exact_source_text(tmp_path: Path) -> None:
    text = "alpha\n\nomega"
    source = _source(text=text)
    parsed = _parsed_document(
        text=text,
        blocks=(
            _html_block("block-1", "paragraph-1", "alpha"),
            _html_block("block-2", "paragraph-2", "omega"),
        ),
    )
    store = LocalEvidenceStore(tmp_path)
    store.put_source(source, normalized_text=text)

    assert store.put_parsed_document(source.source_id, parsed) == parsed


@pytest.mark.parametrize(
    "blocks",
    [
        (_html_block("fabricated", "paragraph-1", "fabricated"),),
        (
            _html_block("block-1", "paragraph-1", "alpha"),
            _html_block("block-2", "paragraph-2", "omega"),
        ),
        (
            _html_block("block-1", "paragraph-1", "omega"),
            _html_block("block-2", "paragraph-2", "alpha"),
        ),
    ],
    ids=("fabricated", "omitted-non-whitespace", "reordered"),
)
def test_evidence_store_rejects_blocks_not_covering_source_in_order(
    tmp_path: Path, blocks: tuple[ParsedBlock, ...]
) -> None:
    text = "alpha hidden omega"
    source = _source(text=text)
    parsed = _parsed_document(text=text, blocks=blocks)
    store = LocalEvidenceStore(tmp_path)
    store.put_source(source, normalized_text=text)

    with pytest.raises(EvidenceIntegrityError, match="block|source|content|order"):
        store.put_parsed_document(source.source_id, parsed)


def test_evidence_store_rejects_mixed_html_pdf_blocks(tmp_path: Path) -> None:
    from deepresearch.retrieval import sha256_text

    text = "alpha omega"
    blocks = (
        _html_block("html", "paragraph-1", "alpha"),
        ParsedBlock(
            block_id="pdf",
            text="omega",
            locator=PdfLocator(page_index=0, block_index=0, start_char=0, end_char=5),
            text_hash=sha256_text("omega"),
        ),
    )
    store = LocalEvidenceStore(tmp_path)
    source = _source(text=text)
    store.put_source(source, normalized_text=text)

    with pytest.raises(EvidenceIntegrityError, match="format|HTML|PDF"):
        store.put_parsed_document(
            source.source_id, _parsed_document(text=text, blocks=blocks)
        )


def test_evidence_store_rejects_pdf_container_keys_out_of_tuple_order(tmp_path: Path) -> None:
    from deepresearch.retrieval import sha256_text

    text = "alpha omega"
    blocks = (
        ParsedBlock(
            block_id="pdf-1",
            text="alpha",
            locator=PdfLocator(page_index=0, block_index=1, start_char=0, end_char=5),
            text_hash=sha256_text("alpha"),
        ),
        ParsedBlock(
            block_id="pdf-0",
            text="omega",
            locator=PdfLocator(page_index=0, block_index=0, start_char=0, end_char=5),
            text_hash=sha256_text("omega"),
        ),
    )
    store = LocalEvidenceStore(tmp_path)
    source = _source(text=text)
    store.put_source(source, normalized_text=text)

    with pytest.raises(EvidenceIntegrityError, match="PDF|order|increasing"):
        store.put_parsed_document(
            source.source_id, _parsed_document(text=text, blocks=blocks)
        )


@pytest.mark.parametrize(
    "field,replacement",
    [
        ("title", "Different title"),
        ("authors", ("Different author",)),
        ("published_at", datetime(2025, 1, 1, tzinfo=UTC)),
        ("parser_version", "different-parser"),
    ],
)
def test_evidence_store_rejects_mismatched_overlapping_source_metadata(
    tmp_path: Path, field: str, replacement: object
) -> None:
    source = _source()
    parsed = _parsed_document().model_copy(update={field: replacement})
    store = LocalEvidenceStore(tmp_path)
    store.put_source(source, normalized_text="short text")

    with pytest.raises(EvidenceIntegrityError, match=field):
        store.put_parsed_document(source.source_id, parsed)


@pytest.mark.parametrize("subdirectory", ["sources", "evidence", "parsed"])
def test_evidence_store_rejects_symlinked_store_directory(
    tmp_path: Path, subdirectory: str
) -> None:
    outside = tmp_path / "outside"
    _symlink_directory(tmp_path / "root" / subdirectory, outside)

    with pytest.raises(EvidenceIntegrityError, match="symlink|reparse|contain"):
        LocalEvidenceStore(tmp_path / "root").put_source(
            _source(), normalized_text="short text"
        )
    assert not list(outside.rglob("*.json"))


def test_evidence_store_rejects_symlinked_source_shard_directory(tmp_path: Path) -> None:
    source = _source()
    digest = hashlib.sha256(source.source_id.encode("utf-8")).hexdigest()
    store = LocalEvidenceStore(tmp_path / "root")
    outside = tmp_path / "outside"
    _symlink_directory(tmp_path / "root" / "sources" / digest[:2], outside)

    with pytest.raises(EvidenceIntegrityError, match="symlink|reparse|contain"):
        store.put_source(source, normalized_text="short text")
    assert not list(outside.rglob("*.json"))


def test_evidence_store_validates_source_existence_locator_excerpt_and_bounds(
    tmp_path: Path,
) -> None:
    store = LocalEvidenceStore(tmp_path)
    source = _source()
    store.put_source(source, normalized_text="short text")
    store.put_parsed_document(source.source_id, _parsed_document())

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
    assert store.put_parsed_document(source.source_id, _parsed_document()) == _parsed_document()
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
    store.put_parsed_document(source.source_id, _parsed_document())
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
    store.put_parsed_document(source.source_id, _parsed_document())
    with ThreadPoolExecutor(max_workers=8) as executor:
        spans = tuple(executor.map(lambda _: store.put_evidence(evidence), range(16)))

    assert sources == (source,) * 16
    assert spans == (evidence,) * 16
    assert not list(tmp_path.rglob("*.tmp"))


def test_evidence_store_ignores_preexisting_unlocked_lock_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _source()
    digest = hashlib.sha256(source.source_id.encode("utf-8")).hexdigest()
    store = LocalEvidenceStore(tmp_path)
    lock_path = tmp_path / "sources" / digest[:2] / f"{digest}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    lock_path.write_bytes(b"stale owner")
    ticks = iter((0.0, 11.0))
    monkeypatch.setattr("deepresearch.storage.evidence_store.time.monotonic", lambda: next(ticks))

    assert store.put_source(source, normalized_text="short text") == source


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


def test_evidence_store_rejects_symlinked_parsed_and_evidence_shards(tmp_path: Path) -> None:
    source = _source()
    digest = hashlib.sha256(source.source_id.encode("utf-8")).hexdigest()
    store = LocalEvidenceStore(tmp_path / "root")
    store.put_source(source, normalized_text="short text")
    outside_parsed = tmp_path / "outside-parsed"
    _symlink_directory(tmp_path / "root" / "parsed" / digest[:2], outside_parsed)

    with pytest.raises(EvidenceIntegrityError, match="symlink|reparse|contain"):
        store.put_parsed_document(source.source_id, _parsed_document())
    assert not list(outside_parsed.rglob("*.json"))

    evidence_root = tmp_path / "evidence-root"
    evidence_store = LocalEvidenceStore(evidence_root)
    evidence_store.put_source(source, normalized_text="short text")
    evidence_store.put_parsed_document(source.source_id, _parsed_document())
    evidence = _evidence()
    evidence_digest = hashlib.sha256(evidence.evidence_id.encode("utf-8")).hexdigest()
    outside_evidence = tmp_path / "outside-evidence"
    _symlink_directory(
        evidence_root / "evidence" / evidence_digest[:2], outside_evidence
    )

    with pytest.raises(EvidenceIntegrityError, match="symlink|reparse|contain"):
        evidence_store.put_evidence(evidence)
    assert not list(outside_evidence.rglob("*.json"))
