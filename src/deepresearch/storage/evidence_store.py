from __future__ import annotations

import hashlib
import json
import os
import tempfile
import time
from collections.abc import Generator
from contextlib import contextmanager
from pathlib import Path
from typing import cast

from deepresearch.domain import EvidenceSpan, HtmlLocator, PdfLocator, SourceDocument
from deepresearch.providers import ParsedBlock, ParsedDocument
from deepresearch.retrieval import normalize_text

from .artifacts import (
    _ensure_safe_directory,  # pyright: ignore[reportPrivateUsage]
    _ensure_safe_file_path,  # pyright: ignore[reportPrivateUsage]
    _is_link_or_reparse,  # pyright: ignore[reportPrivateUsage]
    _release_advisory_lock,  # pyright: ignore[reportPrivateUsage]
    _try_advisory_lock,  # pyright: ignore[reportPrivateUsage]
    _UnsafeStoragePathError,  # pyright: ignore[reportPrivateUsage]
)


class EvidenceIntegrityError(ValueError):
    pass


class EvidenceConflictError(EvidenceIntegrityError):
    pass


@contextmanager
def _key_lock(lock_path: Path, *, timeout_seconds: float = 10.0) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    if _is_link_or_reparse(lock_path):
        raise _UnsafeStoragePathError("evidence lock path is a symlink or reparse point")
    descriptor = os.open(
        lock_path,
        os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0),
        0o600,
    )
    acquired = False
    try:
        if os.fstat(descriptor).st_size == 0:
            os.write(descriptor, b"\0")
            os.fsync(descriptor)
        while not acquired:
            acquired = _try_advisory_lock(descriptor)
            if not acquired:
                if time.monotonic() >= deadline:
                    raise TimeoutError(f"timed out waiting for evidence lock {lock_path.name}")
                time.sleep(0.005)
        yield
    finally:
        if acquired:
            try:
                _release_advisory_lock(descriptor)
            except OSError:
                pass
        os.close(descriptor)


def _atomic_write_bytes(destination: Path, payload: bytes) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            temporary.write(payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_path, destination)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)


def _canonical_payload(value: object) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _envelope(payload: object) -> dict[str, object]:
    return {
        "payload": payload,
        "payload_sha256": hashlib.sha256(_canonical_payload(payload)).hexdigest(),
    }


def _validated_envelope(loaded: object) -> dict[str, object]:
    if not isinstance(loaded, dict):
        raise TypeError("record envelope is not an object")
    envelope = cast("dict[str, object]", loaded)
    if set(envelope) != {"payload", "payload_sha256"}:
        raise TypeError("record envelope fields are invalid")
    raw_payload = envelope["payload"]
    if not isinstance(raw_payload, dict):
        raise TypeError("record payload is not an object")
    payload = cast("dict[str, object]", raw_payload)
    checksum = envelope["payload_sha256"]
    if not isinstance(checksum, str) or hashlib.sha256(_canonical_payload(payload)).hexdigest() != checksum:
        raise TypeError("record payload checksum does not match")
    return payload


def _record_digest(record_id: str) -> str:
    if not record_id:
        raise EvidenceIntegrityError("record ID must not be empty")
    return hashlib.sha256(record_id.encode("utf-8")).hexdigest()


class LocalEvidenceStore:
    def __init__(self, root: str | Path) -> None:
        self._root = Path(root).resolve()
        self._source_root = self._root / "sources"
        self._evidence_root = self._root / "evidence"
        self._parsed_root = self._root / "parsed"
        try:
            self._root.mkdir(parents=True, exist_ok=True)
            _ensure_safe_directory(self._root, self._source_root)
            _ensure_safe_directory(self._root, self._evidence_root)
            _ensure_safe_directory(self._root, self._parsed_root)
        except _UnsafeStoragePathError as error:
            raise EvidenceIntegrityError(str(error)) from error

    def _path(self, root: Path, record_id: str) -> Path:
        digest = _record_digest(record_id)
        return root / digest[:2] / f"{digest}.json"

    def _read_source_record(self, source_id: str) -> tuple[SourceDocument, str]:
        path = self._path(self._source_root, source_id)
        try:
            _ensure_safe_file_path(self._root, path)
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
            raw = _validated_envelope(loaded)
            source = SourceDocument.model_validate(raw.get("source"))
            normalized_text = raw.get("normalized_text")
            if not isinstance(normalized_text, str):
                raise TypeError("normalized_text is not text")
        except FileNotFoundError:
            raise
        except Exception as error:
            raise EvidenceIntegrityError("persisted source record is corrupt") from error
        if source.source_id != source_id:
            raise EvidenceIntegrityError("persisted source identity does not match its key")
        parsed_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        if source.parsed_content_hash != parsed_hash:
            raise EvidenceIntegrityError(
                "source parsed_content_hash does not match exact normalized_text"
            )
        return source, normalized_text

    def _read_evidence_record(self, evidence_id: str) -> EvidenceSpan:
        path = self._path(self._evidence_root, evidence_id)
        try:
            _ensure_safe_file_path(self._root, path)
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
            evidence = EvidenceSpan.model_validate(_validated_envelope(loaded))
        except FileNotFoundError:
            raise
        except Exception as error:
            raise EvidenceIntegrityError("persisted evidence record is corrupt") from error
        if evidence.evidence_id != evidence_id:
            raise EvidenceIntegrityError("persisted evidence identity does not match its key")
        self._validate_evidence(evidence)
        return evidence

    def _validate_parsed_structure(self, parsed_document: ParsedDocument) -> None:
        html_keys: set[str] = set()
        pdf_keys: set[tuple[int, int]] = set()
        for block in parsed_document.blocks:
            locator = block.locator
            if isinstance(locator, HtmlLocator):
                if locator.paragraph_id in html_keys:
                    raise EvidenceIntegrityError(
                        "parsed document HTML paragraph IDs must be unique"
                    )
                html_keys.add(locator.paragraph_id)
            else:
                key = (locator.page_index, locator.block_index)
                if key in pdf_keys:
                    raise EvidenceIntegrityError(
                        "parsed document PDF container keys must be unique"
                    )
                pdf_keys.add(key)

    def _validate_parsed_identity(
        self,
        source_id: str,
        parsed_document: ParsedDocument,
    ) -> SourceDocument:
        try:
            source, normalized_text = self._read_source_record(source_id)
        except FileNotFoundError as error:
            raise EvidenceIntegrityError("parsed document source does not exist") from error
        if parsed_document.canonical_url != source.canonical_url:
            raise EvidenceIntegrityError("parsed document canonical_url does not match source")
        if parsed_document.parsed_content_hash != source.parsed_content_hash:
            raise EvidenceIntegrityError(
                "parsed document parsed_content_hash does not match source"
            )
        if parsed_document.normalized_text != normalized_text:
            raise EvidenceIntegrityError(
                "parsed document normalized_text does not match stored source text"
            )
        self._validate_parsed_structure(parsed_document)
        return source

    def _read_parsed_document(self, source_id: str) -> ParsedDocument:
        path = self._path(self._parsed_root, source_id)
        try:
            _ensure_safe_file_path(self._root, path)
            loaded: object = json.loads(path.read_text(encoding="utf-8"))
            payload = _validated_envelope(loaded)
            persisted_source_id = payload.get("source_id")
            parsed_document = ParsedDocument.model_validate(payload.get("parsed_document"))
        except FileNotFoundError:
            raise
        except Exception as error:
            raise EvidenceIntegrityError("persisted parsed document is corrupt") from error
        if persisted_source_id != source_id:
            raise EvidenceIntegrityError("persisted parsed document identity does not match its key")
        self._validate_parsed_identity(source_id, parsed_document)
        return parsed_document

    @staticmethod
    def _locator_block(
        locator: HtmlLocator | PdfLocator,
        blocks: tuple[ParsedBlock, ...],
    ) -> ParsedBlock:
        for block in blocks:
            block_locator = block.locator
            if (
                isinstance(locator, HtmlLocator)
                and isinstance(block_locator, HtmlLocator)
                and block_locator.paragraph_id == locator.paragraph_id
            ):
                return block
            if (
                isinstance(locator, PdfLocator)
                and isinstance(block_locator, PdfLocator)
                and block_locator.page_index == locator.page_index
                and block_locator.block_index == locator.block_index
            ):
                return block
        if isinstance(locator, HtmlLocator):
            raise EvidenceIntegrityError("evidence HTML paragraph container does not exist")
        raise EvidenceIntegrityError("evidence PDF page/block container does not exist")

    def _validate_evidence(self, evidence: EvidenceSpan) -> None:
        expected_hash = hashlib.sha256(evidence.excerpt.encode("utf-8")).hexdigest()
        if evidence.excerpt_hash != expected_hash:
            raise EvidenceIntegrityError("evidence excerpt_hash does not match exact excerpt")
        try:
            parsed_document = self._read_parsed_document(evidence.source_id)
        except FileNotFoundError as error:
            try:
                self._read_source_record(evidence.source_id)
            except FileNotFoundError:
                message = "evidence source does not exist"
            else:
                message = "evidence source has no registered parsed document"
            raise EvidenceIntegrityError(message) from error
        locator = evidence.locator
        block = self._locator_block(locator, parsed_document.blocks)
        start_char = locator.start_char
        end_char = locator.end_char
        if start_char < 0 or end_char <= start_char or end_char > len(block.text):
            raise EvidenceIntegrityError("evidence locator bounds exceed parsed container text")
        exact_excerpt = block.text[start_char:end_char]
        if evidence.excerpt != exact_excerpt:
            raise EvidenceIntegrityError("evidence excerpt does not match locator source slice")

    def put_source(
        self,
        source: SourceDocument,
        *,
        normalized_text: str,
    ) -> SourceDocument:
        if normalize_text(normalized_text) != normalized_text:
            raise EvidenceIntegrityError("normalized_text must already be normalized")
        parsed_hash = hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
        if source.parsed_content_hash != parsed_hash:
            raise EvidenceIntegrityError(
                "source parsed_content_hash does not match exact normalized_text"
            )
        path = self._path(self._source_root, source.source_id)
        lock_path = path.with_suffix(".lock")
        record = {
            "normalized_text": normalized_text,
            "source": source.model_dump(mode="json"),
        }
        payload = _canonical_payload(_envelope(record))
        try:
            _ensure_safe_file_path(self._root, path)
            _ensure_safe_file_path(self._root, lock_path)
            with _key_lock(lock_path):
                _ensure_safe_file_path(self._root, path)
                if path.exists():
                    existing_source, existing_text = self._read_source_record(source.source_id)
                    if existing_source != source or existing_text != normalized_text:
                        raise EvidenceConflictError("source ID already contains a different record")
                    return existing_source
                _ensure_safe_file_path(self._root, path)
                _atomic_write_bytes(path, payload)
        except _UnsafeStoragePathError as error:
            raise EvidenceIntegrityError(str(error)) from error
        return source

    def put_parsed_document(
        self,
        source_id: str,
        parsed_document: ParsedDocument,
    ) -> ParsedDocument:
        self._validate_parsed_identity(source_id, parsed_document)
        path = self._path(self._parsed_root, source_id)
        lock_path = path.with_suffix(".lock")
        record = {
            "parsed_document": parsed_document.model_dump(mode="json"),
            "source_id": source_id,
        }
        payload = _canonical_payload(_envelope(record))
        try:
            _ensure_safe_file_path(self._root, path)
            _ensure_safe_file_path(self._root, lock_path)
            with _key_lock(lock_path):
                _ensure_safe_file_path(self._root, path)
                if path.exists():
                    existing = self._read_parsed_document(source_id)
                    if existing != parsed_document:
                        raise EvidenceConflictError(
                            "source ID already contains a different parsed document"
                        )
                    return existing
                _ensure_safe_file_path(self._root, path)
                _atomic_write_bytes(path, payload)
        except _UnsafeStoragePathError as error:
            raise EvidenceIntegrityError(str(error)) from error
        return parsed_document

    def put_evidence(self, evidence: EvidenceSpan) -> EvidenceSpan:
        self._validate_evidence(evidence)
        path = self._path(self._evidence_root, evidence.evidence_id)
        lock_path = path.with_suffix(".lock")
        payload = _canonical_payload(_envelope(evidence.model_dump(mode="json")))
        try:
            _ensure_safe_file_path(self._root, path)
            _ensure_safe_file_path(self._root, lock_path)
            with _key_lock(lock_path):
                _ensure_safe_file_path(self._root, path)
                if path.exists():
                    existing = self._read_evidence_record(evidence.evidence_id)
                    if existing != evidence:
                        raise EvidenceConflictError(
                            "evidence ID already contains a different record"
                        )
                    return existing
                _ensure_safe_file_path(self._root, path)
                _atomic_write_bytes(path, payload)
        except _UnsafeStoragePathError as error:
            raise EvidenceIntegrityError(str(error)) from error
        return evidence

    def get_source(self, source_id: str) -> SourceDocument:
        return self._read_source_record(source_id)[0]

    def get_evidence(self, evidence_id: str) -> EvidenceSpan:
        return self._read_evidence_record(evidence_id)

    def has_evidence(self, evidence_id: str) -> bool:
        path = self._path(self._evidence_root, evidence_id)
        try:
            _ensure_safe_file_path(self._root, path)
        except _UnsafeStoragePathError as error:
            raise EvidenceIntegrityError(str(error)) from error
        if not path.is_file():
            return False
        self._read_evidence_record(evidence_id)
        return True


__all__ = [
    "EvidenceConflictError",
    "EvidenceIntegrityError",
    "LocalEvidenceStore",
]
