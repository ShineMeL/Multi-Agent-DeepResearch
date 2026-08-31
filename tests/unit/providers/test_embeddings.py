import json
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from deepresearch.providers.embeddings import (
    DeterministicHashTextEmbedder,
    EmbeddingModelFile,
    EmbeddingModelLock,
    ModelSnapshotUnavailable,
    SentenceTransformerTextEmbedder,
    fetch_and_lock,
)
from deepresearch.runtime import CancellationToken, OperationCancelled

FIXTURE_LOCK = Path(__file__).parents[2] / "fixtures" / "models" / "embedding.lock.json"
PRODUCTION_LOCK = Path(__file__).parents[3] / "models" / "embedding.lock.json"
SYNTHETIC_CONFIG = b'{"hidden_size":3}\n'


def _deadline() -> float:
    return time.monotonic() + 10.0


def _locked_root(tmp_path: Path) -> tuple[EmbeddingModelLock, Path]:
    root = tmp_path / "embedding"
    root.mkdir()
    (root / "config.json").write_bytes(SYNTHETIC_CONFIG)
    return EmbeddingModelLock.load(FIXTURE_LOCK), root


def test_sentence_embedder_refuses_unlocked_missing_or_tampered_snapshot(
    tmp_path: Path,
) -> None:
    lock = EmbeddingModelLock.load(FIXTURE_LOCK)

    with pytest.raises(ModelSnapshotUnavailable):
        SentenceTransformerTextEmbedder.from_lock(lock, model_root=tmp_path / "missing")

    _, root = _locked_root(tmp_path)
    (root / "config.json").write_bytes(b"tampered")
    with pytest.raises(ModelSnapshotUnavailable):
        SentenceTransformerTextEmbedder.from_lock(lock, model_root=root)


def test_embedding_lock_rejects_traversal_and_noncanonical_manifest() -> None:
    lock = EmbeddingModelLock.load(FIXTURE_LOCK)

    with pytest.raises(ValidationError):
        EmbeddingModelFile(
            path="../escape",
            sha256="a" * 64,
            size_bytes=1,
        )
    with pytest.raises(ValidationError, match="manifest|snapshot|hash"):
        lock.model_copy(update={"snapshot_sha256": "0" * 64})


@pytest.mark.asyncio
async def test_deterministic_embedder_is_byte_stable_offline_and_cancellable() -> None:
    embedder = DeterministicHashTextEmbedder(dimension=8)
    token = CancellationToken()

    first = await embedder.embed(
        ["Agent", "多模态"], deadline=_deadline(), cancellation_token=token
    )
    second = await embedder.embed(
        ["Agent", "多模态"], deadline=_deadline(), cancellation_token=token
    )

    assert first == second
    assert len(first) == 2
    assert all(len(vector) == 8 for vector in first)
    assert embedder.network_calls == 0
    assert "sentence_transformers" not in sys.modules

    token.cancel()
    with pytest.raises(OperationCancelled):
        await embedder.embed(["cancelled"], deadline=_deadline(), cancellation_token=token)


@pytest.mark.asyncio
async def test_sentence_transformer_is_lazy_local_only_and_constructed_off_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lock, root = _locked_root(tmp_path)
    calls: dict[str, object] = {}
    event_loop_thread = threading.get_ident()

    class FakeSentenceTransformer:
        def __init__(self, model_name_or_path: str, **kwargs: object) -> None:
            calls["path"] = model_name_or_path
            calls["kwargs"] = kwargs
            calls["thread"] = threading.get_ident()

        def encode(self, texts: list[str], **kwargs: object) -> list[list[float]]:
            calls["encode_kwargs"] = kwargs
            return [[1.0, 0.0, 0.0] for _ in texts]

    monkeypatch.setitem(
        sys.modules,
        "sentence_transformers",
        SimpleNamespace(SentenceTransformer=FakeSentenceTransformer),
    )
    embedder = SentenceTransformerTextEmbedder.from_lock(lock, model_root=root)
    assert calls == {}

    vectors = await embedder.embed(
        ["one", "two"], deadline=_deadline(), cancellation_token=CancellationToken()
    )

    assert vectors == ((1.0, 0.0, 0.0), (1.0, 0.0, 0.0))
    assert calls["path"] == str(root.resolve())
    assert calls["kwargs"] == {
        "local_files_only": True,
        "trust_remote_code": False,
    }
    assert calls["thread"] != event_loop_thread
    assert calls["encode_kwargs"] == {
        "convert_to_numpy": True,
        "normalize_embeddings": True,
        "show_progress_bar": False,
    }


def test_fetch_and_lock_is_revision_pinned_atomic_and_requires_replace(
    tmp_path: Path,
) -> None:
    revision = "2" * 40
    model_root = tmp_path / "models" / "embedding"
    lock_path = tmp_path / "models" / "embedding.lock.json"
    calls: list[tuple[str, str, Path]] = []

    def resolve_revision(model_id: str, requested_revision: str) -> str:
        assert model_id == "synthetic/model"
        return requested_revision

    def download(model_id: str, requested_revision: str, destination: Path) -> None:
        calls.append((model_id, requested_revision, destination))
        (destination / "config.json").write_bytes(SYNTHETIC_CONFIG)

    created = fetch_and_lock(
        model_id="synthetic/model",
        revision=revision,
        model_root=model_root,
        lock_path=lock_path,
        revision_resolver=resolve_revision,
        snapshot_downloader=download,
    )

    assert created.revision == revision
    assert created.verify(model_root) is None
    assert len(calls) == 1
    assert calls[0][2].parent == model_root.parent.resolve()
    assert not list(model_root.parent.glob(".embedding.staging.*"))

    with pytest.raises(FileExistsError):
        fetch_and_lock(
            model_id="synthetic/model",
            revision=revision,
            model_root=model_root,
            lock_path=lock_path,
            revision_resolver=resolve_revision,
            snapshot_downloader=download,
        )

    replaced = fetch_and_lock(
        model_id="synthetic/model",
        revision=revision,
        model_root=model_root,
        lock_path=lock_path,
        replace=True,
        revision_resolver=resolve_revision,
        snapshot_downloader=download,
    )
    assert EmbeddingModelLock.load(lock_path) == replaced
    assert replaced.verify(model_root) is None


def test_fetch_and_lock_rejects_resolved_revision_mismatch_before_publication(
    tmp_path: Path,
) -> None:
    downloaded = False

    def download(model_id: str, revision: str, destination: Path) -> None:
        nonlocal downloaded
        del model_id, revision, destination
        downloaded = True

    with pytest.raises(ModelSnapshotUnavailable, match="revision"):
        fetch_and_lock(
            model_id="synthetic/model",
            revision="3" * 40,
            model_root=tmp_path / "embedding",
            lock_path=tmp_path / "embedding.lock.json",
            revision_resolver=lambda model_id, revision: "4" * 40,
            snapshot_downloader=download,
        )

    assert downloaded is False
    assert not (tmp_path / "embedding").exists()
    assert not (tmp_path / "embedding.lock.json").exists()


def test_fetch_and_lock_replace_recovers_corrupt_existing_lock(tmp_path: Path) -> None:
    revision = "5" * 40
    model_root = tmp_path / "embedding"
    lock_path = tmp_path / "embedding.lock.json"
    lock_path.write_text("not-json", encoding="utf-8")

    def download(model_id: str, requested_revision: str, destination: Path) -> None:
        del model_id, requested_revision
        (destination / "config.json").write_bytes(SYNTHETIC_CONFIG)

    lock = fetch_and_lock(
        model_id="synthetic/model",
        revision=revision,
        model_root=model_root,
        lock_path=lock_path,
        replace=True,
        revision_resolver=lambda model_id, requested: requested,
        snapshot_downloader=download,
    )

    assert EmbeddingModelLock.load(lock_path) == lock
    assert lock.verify(model_root) is None


def test_production_embedding_lock_pins_official_snapshot_without_model_bytes() -> None:
    lock = EmbeddingModelLock.load(PRODUCTION_LOCK)

    assert lock.model_id == "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
    assert lock.revision == "e62509716f15c5fd03a6fd3156a4bc5e43f83f26"
    assert lock.vector_dimension == 384
    assert lock.normalize_embeddings is True
    assert lock.files
    assert json.loads(PRODUCTION_LOCK.read_text(encoding="utf-8"))["files"]
