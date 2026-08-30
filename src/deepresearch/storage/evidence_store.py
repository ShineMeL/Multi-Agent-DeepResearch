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

from deepresearch.domain import EvidenceSpan, SourceDocument


class EvidenceIntegrityError(ValueError):
    pass


class EvidenceConflictError(EvidenceIntegrityError):
    pass


@contextmanager
def _key_lock(lock_path: Path, *, timeout_seconds: float = 10.0) -> Generator[None, None, None]:
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + timeout_seconds
    descriptor: int | None = None
    while descriptor is None:
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            if time.monotonic() >= deadline:
                raise TimeoutError(f"timed out waiting for evidence lock {lock_path.name}")
            time.sleep(0.005)
    try:
        os.write(descriptor, str(os.getpid()).encode("ascii"))
        os.fsync(descriptor)
        yield
    finally:
        os.close(descriptor)
        lock_path.unlink(missing_ok=True)


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
        self._source_root.mkdir(parents=True, exist_ok=True)
        self._evidence_root.mkdir(parents=True, exist_ok=True)

    def _path(self, root: Path, record_id: str) -> Path:
        digest = _record_digest(record_id)
        return root / digest[:2] / f"{digest}.json"

    def _read_source_record(self, source_id: str) -> tuple[SourceDocument, str]:
        path = self._path(self._source_root, source_id)
        try:
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

    def _validate_evidence(self, evidence: EvidenceSpan) -> None:
        expected_hash = hashlib.sha256(evidence.excerpt.encode("utf-8")).hexdigest()
        if evidence.excerpt_hash != expected_hash:
            raise EvidenceIntegrityError("evidence excerpt_hash does not match exact excerpt")
        try:
            source, normalized_text = self._read_source_record(evidence.source_id)
        except FileNotFoundError as error:
            raise EvidenceIntegrityError("evidence source does not exist") from error
        if source.source_id != evidence.source_id:
            raise EvidenceIntegrityError("evidence source identity does not match")
        locator = evidence.locator
        start_char = locator.start_char
        end_char = locator.end_char
        if start_char < 0 or end_char <= start_char or end_char > len(normalized_text):
            raise EvidenceIntegrityError("evidence locator bounds exceed normalized source text")
        exact_excerpt = normalized_text[start_char:end_char]
        if evidence.excerpt != exact_excerpt:
            raise EvidenceIntegrityError("evidence excerpt does not match locator source slice")

    def put_source(
        self,
        source: SourceDocument,
        *,
        normalized_text: str,
    ) -> SourceDocument:
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
        with _key_lock(lock_path):
            if path.exists():
                existing_source, existing_text = self._read_source_record(source.source_id)
                if existing_source != source or existing_text != normalized_text:
                    raise EvidenceConflictError("source ID already contains a different record")
                return existing_source
            _atomic_write_bytes(path, payload)
        return source

    def put_evidence(self, evidence: EvidenceSpan) -> EvidenceSpan:
        self._validate_evidence(evidence)
        path = self._path(self._evidence_root, evidence.evidence_id)
        lock_path = path.with_suffix(".lock")
        payload = _canonical_payload(_envelope(evidence.model_dump(mode="json")))
        with _key_lock(lock_path):
            if path.exists():
                existing = self._read_evidence_record(evidence.evidence_id)
                if existing != evidence:
                    raise EvidenceConflictError("evidence ID already contains a different record")
                return existing
            _atomic_write_bytes(path, payload)
        return evidence

    def get_source(self, source_id: str) -> SourceDocument:
        return self._read_source_record(source_id)[0]

    def get_evidence(self, evidence_id: str) -> EvidenceSpan:
        return self._read_evidence_record(evidence_id)

    def has_evidence(self, evidence_id: str) -> bool:
        path = self._path(self._evidence_root, evidence_id)
        if not path.is_file():
            return False
        self._read_evidence_record(evidence_id)
        return True


__all__ = [
    "EvidenceConflictError",
    "EvidenceIntegrityError",
    "LocalEvidenceStore",
]
