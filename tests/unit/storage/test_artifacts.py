import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from pydantic import ValidationError

from deepresearch.storage import (
    ArtifactIntegrityError,
    ArtifactRef,
    LocalArtifactStore,
)


def test_artifact_put_is_content_addressed_and_atomic(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    first = store.put_bytes(b"same", media_type="text/plain")
    second = store.put_bytes(b"same", media_type="text/plain")

    assert first.artifact_id == second.artifact_id
    assert first.artifact_id == f"sha256:{first.sha256}"
    assert store.get_bytes(first.artifact_id) == b"same"
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


def test_artifact_ref_rejects_inconsistent_identity() -> None:
    with pytest.raises(ValidationError, match="artifact_id"):
        ArtifactRef(
            artifact_id=f"sha256:{'0' * 64}",
            sha256="1" * 64,
            media_type="text/plain",
            size_bytes=1,
        )


def test_artifact_store_rejects_path_traversal_ids(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with pytest.raises(ArtifactIntegrityError):
        store.get_bytes("../../outside")
    assert not store.exists("../../outside")


def test_artifact_read_detects_payload_and_metadata_corruption(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes(b"original", media_type="text/plain")
    artifact_path = next(tmp_path.rglob("*.json"))
    envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload = envelope.get("payload", envelope)
    payload["data_base64"] = "dGFtcGVyZWQ="
    artifact_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="hash|size"):
        store.get_bytes(ref.artifact_id)


def test_artifact_read_detects_schema_valid_media_metadata_tampering(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)
    ref = store.put_bytes(b"original", media_type="text/plain")
    artifact_path = next(tmp_path.rglob("*.json"))
    envelope = json.loads(artifact_path.read_text(encoding="utf-8"))
    payload = envelope.get("payload", envelope)
    payload["ref"]["media_type"] = "application/json"
    artifact_path.write_text(json.dumps(envelope), encoding="utf-8")

    with pytest.raises(ArtifactIntegrityError, match="corrupt"):
        store.get_bytes(ref.artifact_id)


def test_artifact_concurrent_writers_are_idempotent(tmp_path: Path) -> None:
    store = LocalArtifactStore(tmp_path)

    with ThreadPoolExecutor(max_workers=8) as executor:
        refs = tuple(
            executor.map(
                lambda _: store.put_bytes(b"concurrent", media_type="application/octet-stream"),
                range(32),
            )
        )

    assert len({ref.artifact_id for ref in refs}) == 1
    assert store.get_bytes(refs[0].artifact_id) == b"concurrent"
    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))


def test_failed_atomic_artifact_replace_cleans_temporary_and_lock_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalArtifactStore(tmp_path)

    def fail_replace(source: str | Path, destination: str | Path) -> None:
        raise OSError("replace failed")

    monkeypatch.setattr("deepresearch.storage.artifacts.os.replace", fail_replace)

    with pytest.raises(OSError, match="replace failed"):
        store.put_bytes(b"will fail", media_type="text/plain")

    assert not list(tmp_path.rglob("*.tmp"))
    assert not list(tmp_path.rglob("*.lock"))
