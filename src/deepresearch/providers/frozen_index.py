from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol, cast

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator
from rank_bm25 import BM25Okapi  # pyright: ignore[reportMissingTypeStubs]

from benchmarks.datasets.models import FrozenEvidenceRecord
from deepresearch.providers.errors import ProviderError
from deepresearch.retrieval import canonicalize_url

_SHA256 = re.compile(r"[0-9a-f]{64}\Z")


class _Bm25Index(Protocol):
    def get_scores(self, query: list[str]) -> Sequence[float]: ...


class _Bm25Factory(Protocol):
    def __call__(self, corpus: list[list[str]]) -> _Bm25Index: ...


def _canonical_json_bytes(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _sha256(value: str, *, field: str) -> str:
    if type(value) is not str or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{field} must be a lowercase 64-character SHA-256")
    return value


def _invalid_snapshot(message: str) -> ProviderError:
    return ProviderError(
        code="INVALID_SNAPSHOT",
        provider="frozen-corpus",
        operation="snapshot",
        public_message=message,
        retryable=False,
    )


def deterministic_tokens(text: str) -> tuple[str, ...]:
    if type(text) is not str:
        raise TypeError("text must be a string")
    normalized = unicodedata.normalize("NFC", text).casefold()
    pieces = re.findall(r"[a-z0-9]+|[\u3400-\u4dbf\u4e00-\u9fff]", normalized)
    return tuple(pieces)


class FrozenCorpusManifest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str
    task_id: str
    corpus_version: str
    index_version: str
    document_count: int = Field(ge=0)
    documents_sha256: str
    index_sha256: str

    @field_validator("snapshot_id", "task_id", "corpus_version", "index_version")
    @classmethod
    def require_identity(cls, value: str) -> str:
        if type(value) is not str or not value.strip():
            raise ValueError("snapshot identity must be non-empty")
        return value

    @field_validator("documents_sha256", "index_sha256")
    @classmethod
    def require_hash(cls, value: str, info: object) -> str:
        field = getattr(info, "field_name", "hash")
        return _sha256(value, field=str(field))


@dataclass(frozen=True, slots=True)
class FrozenHit:
    score: float
    record: FrozenEvidenceRecord


class FrozenBm25Index:
    def __init__(
        self,
        records: tuple[FrozenEvidenceRecord, ...],
        *,
        index_version: str,
        token_rows: tuple[tuple[str, ...], ...],
    ) -> None:
        self.records = records
        self.index_version = index_version
        self._token_rows = token_rows
        # rank_bm25 divides by the document count while constructing its
        # statistics, so keep an explicit empty-index representation for a
        # valid zero-document snapshot.
        bm25_factory = cast("_Bm25Factory", BM25Okapi)
        self._bm25: _Bm25Index | None = (
            bm25_factory([list(tokens) for tokens in token_rows]) if records else None
        )

    @classmethod
    def build(
        cls,
        records: Sequence[FrozenEvidenceRecord],
        *,
        index_version: str,
    ) -> FrozenBm25Index:
        if type(index_version) is not str or not index_version.strip():
            raise ValueError("index_version must be non-empty")
        ordered = tuple(sorted(records, key=lambda item: (item.evidence_id, item.source_id)))
        evidence_ids = [item.evidence_id for item in ordered]
        if len(evidence_ids) != len(set(evidence_ids)):
            raise ValueError("duplicate evidence_id")
        token_rows = tuple(deterministic_tokens(item.normalized_text) for item in ordered)
        return cls(ordered, index_version=index_version, token_rows=token_rows)

    @classmethod
    def from_payload(
        cls,
        payload: object,
        records: Sequence[FrozenEvidenceRecord],
        *,
        expected_index_version: str,
    ) -> FrozenBm25Index:
        if not isinstance(payload, Mapping):
            raise TypeError("index payload must be an object")
        raw = cast("Mapping[object, object]", payload)
        version = raw.get("index_version")
        raw_records = raw.get("records")
        if version != expected_index_version or not isinstance(raw_records, list):
            raise ValueError("index metadata does not match snapshot")
        expected = cls.build(records, index_version=expected_index_version)
        canonical_rows: list[dict[str, object]] = []
        for item in cast("list[object]", raw_records):
            if not isinstance(item, Mapping):
                raise TypeError("index record must be an object")
            mapping = cast("Mapping[object, object]", item)
            evidence_id = mapping.get("evidence_id")
            source_id = mapping.get("source_id")
            tokens = mapping.get("tokens")
            raw_tokens = cast("list[object]", tokens)
            if (
                not isinstance(evidence_id, str)
                or not isinstance(source_id, str)
                or not isinstance(tokens, list)
                or any(type(token) is not str for token in raw_tokens)
            ):
                raise ValueError("index record is invalid")
            canonical_rows.append(
                {
                    "evidence_id": evidence_id,
                    "source_id": source_id,
                    "tokens": list(cast("list[str]", raw_tokens)),
                }
            )
        if canonical_rows != expected.to_payload()["records"]:
            raise ValueError("serialized index does not match normalized records")
        if _canonical_json_bytes(cast("object", payload)) != _canonical_json_bytes(
            expected.to_payload()
        ):
            raise ValueError("serialized index is not canonical")
        return expected

    def to_payload(self) -> dict[str, object]:
        return {
            "index_version": self.index_version,
            "records": [
                {
                    "evidence_id": record.evidence_id,
                    "source_id": record.source_id,
                    "tokens": list(tokens),
                }
                for record, tokens in zip(self.records, self._token_rows, strict=True)
            ],
        }

    def search(self, query: str, limit: int) -> tuple[FrozenHit, ...]:
        if type(limit) is not int or limit <= 0:
            raise ValueError("limit must be a positive integer")
        tokens = deterministic_tokens(query)
        if not self.records:
            return ()
        if self._bm25 is None:
            return ()
        scores = self._bm25.get_scores(list(tokens))
        hits = [
            FrozenHit(float(score), record)
            for score, record in zip(scores, self.records, strict=True)
        ]
        hits.sort(key=lambda item: (-item.score, item.record.evidence_id, item.record.source_id))
        return tuple(hits[:limit])


class FrozenCorpusSnapshot:
    def __init__(
        self,
        *,
        root: Path,
        manifest: FrozenCorpusManifest,
        records: tuple[FrozenEvidenceRecord, ...],
        index: FrozenBm25Index,
    ) -> None:
        self.root = root
        self._manifest = manifest
        self._records = records
        self._record_by_id = {record.evidence_id: record for record in records}
        self._records_by_url: dict[str, tuple[FrozenEvidenceRecord, ...]] = {}
        grouped: defaultdict[str, list[FrozenEvidenceRecord]] = defaultdict(list)
        for record in records:
            grouped[canonicalize_url(str(record.canonical_url))].append(record)
        self._records_by_url = {
            url: tuple(sorted(items, key=lambda item: (item.evidence_id, item.source_id)))
            for url, items in grouped.items()
        }
        self._index = index

    @classmethod
    def load(cls, snapshot_dir: Path, *, task_id: str) -> FrozenCorpusSnapshot:
        requested = Path(snapshot_dir).absolute()
        try:
            root = requested.resolve(strict=True)
            if not root.is_dir():
                raise ValueError("snapshot root must be a directory")
            files = {
                name: root / name
                for name in ("documents.jsonl", "index.json", "snapshot.json", "manifest.sha256")
            }
            if any(not path.is_file() for path in files.values()):
                raise ValueError("snapshot file is missing")
            manifest_payload = json.loads(files["manifest.sha256"].read_bytes())
            if not isinstance(manifest_payload, Mapping):
                raise TypeError("snapshot file manifest is invalid")
            raw_manifest = cast("Mapping[object, object]", manifest_payload)
            file_hashes = raw_manifest.get("file_sha256")
            if not isinstance(file_hashes, Mapping):
                raise TypeError("snapshot file manifest has no hashes")
            for name in ("documents.jsonl", "index.json", "snapshot.json"):
                expected_hash = cast("Mapping[object, object]", file_hashes).get(name)
                if (
                    not isinstance(expected_hash, str)
                    or hashlib.sha256(files[name].read_bytes()).hexdigest() != expected_hash
                ):
                    raise ValueError(f"snapshot file hash mismatch: {name}")

            manifest = FrozenCorpusManifest.model_validate_json(
                files["snapshot.json"].read_bytes(), strict=True
            )
            if manifest.task_id != task_id:
                raise ValueError("snapshot task_id does not match requested task")
            raw_documents = files["documents.jsonl"].read_bytes()
            if hashlib.sha256(raw_documents).hexdigest() != manifest.documents_sha256:
                raise ValueError("documents hash does not match snapshot manifest")
            lines = () if not raw_documents else raw_documents.splitlines()
            if raw_documents and not raw_documents.endswith(b"\n"):
                raise ValueError("documents JSONL must end with a newline")
            records = tuple(
                FrozenEvidenceRecord.model_validate_json(line, strict=True) for line in lines
            )
            if len(records) != manifest.document_count:
                raise ValueError("document_count does not match documents")
            if any(record.task_id != task_id for record in records):
                raise ValueError("document task_id does not match snapshot task")
            if len({record.evidence_id for record in records}) != len(records):
                raise ValueError("duplicate evidence_id")

            by_source: defaultdict[str, list[FrozenEvidenceRecord]] = defaultdict(list)
            by_url: defaultdict[str, list[FrozenEvidenceRecord]] = defaultdict(list)
            for record in records:
                by_source[record.source_id].append(record)
                by_url[canonicalize_url(str(record.canonical_url))].append(record)
            for group in (*by_source.values(), *by_url.values()):
                first = group[0]
                identity = (
                    first.raw_body_b64,
                    first.content_hash,
                    first.media_type,
                    first.retrieved_at,
                    first.parsed_content_hash,
                )
                if any(
                    (
                        item.raw_body_b64,
                        item.content_hash,
                        item.media_type,
                        item.retrieved_at,
                        item.parsed_content_hash,
                    )
                    != identity
                    for item in group[1:]
                ):
                    raise ValueError("records sharing a source or URL disagree")

            index_payload = json.loads(files["index.json"].read_bytes())
            if hashlib.sha256(files["index.json"].read_bytes()).hexdigest() != manifest.index_sha256:
                raise ValueError("index hash does not match snapshot manifest")
            index = FrozenBm25Index.from_payload(
                index_payload,
                records,
                expected_index_version=manifest.index_version,
            )
            return cls(root=root, manifest=manifest, records=records, index=index)
        except ProviderError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError, ValidationError) as error:
            if isinstance(error, FileNotFoundError):
                raise ProviderError(
                    code="REPLAY_MISS",
                    provider="frozen-corpus",
                    operation="snapshot",
                    public_message="frozen snapshot is unavailable",
                    retryable=False,
                ) from error
            raise _invalid_snapshot("frozen snapshot is invalid") from error

    @property
    def manifest(self) -> FrozenCorpusManifest:
        return self._manifest

    @property
    def index(self) -> FrozenBm25Index:
        return self._index

    @property
    def records(self) -> tuple[FrozenEvidenceRecord, ...]:
        return self._records

    def record(self, evidence_id: str) -> FrozenEvidenceRecord:
        try:
            return self._record_by_id[evidence_id]
        except KeyError as error:
            raise ProviderError(
                code="REPLAY_MISS",
                provider="frozen-corpus",
                operation="record",
                public_message="frozen evidence is not present in the snapshot",
                retryable=False,
            ) from error

    def records_for_url(self, url: str) -> tuple[FrozenEvidenceRecord, ...]:
        try:
            canonical = canonicalize_url(url)
        except (TypeError, ValueError):
            return ()
        return self._records_by_url.get(canonical, ())


__all__ = [
    "FrozenBm25Index",
    "FrozenCorpusManifest",
    "FrozenCorpusSnapshot",
    "FrozenHit",
    "deterministic_tokens",
]
