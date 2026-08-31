import hashlib
import json
import shutil
import time
from decimal import Decimal
from pathlib import Path

import pytest

from deepresearch.providers import ModelMessage, ModelRequest, ProviderError
from deepresearch.providers.replay import (
    ReplayFetcher,
    ReplayModelProvider,
    ReplaySearchProvider,
    ReplayTextEmbedder,
)
from deepresearch.providers.replay_schema import REPLAY_FILES, ReplayBundle
from deepresearch.runtime import CancellationToken, OperationCancelled

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "replay" / "provider_contract"


@pytest.fixture
def bundle() -> ReplayBundle:
    return ReplayBundle.load(FIXTURE_ROOT)


def _future_deadline() -> float:
    return time.monotonic() + 10.0


def _fixture_model_request() -> ModelRequest:
    return ModelRequest(
        model_id="fixture-model-v1",
        messages=(ModelMessage(role="user", content="Synthetic fixture question?"),),
        temperature=Decimal(0),
        seed=7,
        max_output_tokens=32,
        prompt_version="prompt-v1",
        system_prompt_hash="a" * 64,
        tool_schema_hash="b" * 64,
        output_schema_hash="c" * 64,
    )


def _canonical(value: object) -> bytes:
    return json.dumps(
        value, ensure_ascii=False, allow_nan=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")


def _update_file_hash(root: Path, filename: str) -> None:
    manifest_path = root / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"][filename] = hashlib.sha256(
        (root / filename).read_bytes()
    ).hexdigest()
    manifest_path.write_bytes(_canonical(manifest) + b"\n")


def _write_record(root: Path, filename: str, record: dict[str, object]) -> None:
    outcome = record["outcome"]
    record["outcome_sha256"] = hashlib.sha256(_canonical(outcome)).hexdigest()
    (root / filename).write_bytes(_canonical(record) + b"\n")
    _update_file_hash(root, filename)


@pytest.mark.asyncio
async def test_replay_search_exact_request_returns_recorded_hit(bundle: ReplayBundle) -> None:
    provider = ReplaySearchProvider(bundle)

    hits = await provider.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert [hit.provider_metadata["source_id"] for hit in hits] == ["src-1"]
    assert provider.live_calls == 0


@pytest.mark.asyncio
async def test_replay_miss_never_falls_back(bundle: ReplayBundle) -> None:
    provider = ReplaySearchProvider(bundle)

    with pytest.raises(ProviderError) as error:
        await provider.search(
            "unknown query",
            5,
            None,
            deadline=_future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "REPLAY_MISS"
    assert provider.live_calls == 0


def test_bundle_verification_rejects_tampering_and_duplicate_keys(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    search_path = copied / "search.jsonl"
    record = json.loads(search_path.read_text(encoding="utf-8"))
    search_path.write_bytes(_canonical(record) + b"\n" + _canonical(record) + b"\n")
    _update_file_hash(copied, "search.jsonl")

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"
    assert "duplicate replay request key" in str(error.value.__cause__)


def test_bundle_load_rejects_symlinked_record_file(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    target = tmp_path / "outside.jsonl"
    target.write_text("", encoding="utf-8")
    (copied / "embeddings.jsonl").unlink()
    try:
        (copied / "embeddings.jsonl").symlink_to(target)
    except OSError as error:
        pytest.skip(f"file symlinks are unavailable: {error}")

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


def test_bundle_verify_detects_tampering_after_load(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    bundle = ReplayBundle.load(copied)
    (copied / "search.jsonl").write_bytes(b"tampered\n")

    verification = bundle.verify()

    assert verification.valid is False
    assert verification.errors


def test_bundle_rejects_snapshot_provider_mismatch_even_with_updated_file_hash(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    snapshot_path = copied / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["providers"]["search"]["provider_id"] = "different-search"
    snapshot_path.write_text(
        json.dumps(snapshot, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    manifest_path = copied / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["file_sha256"]["snapshot.json"] = hashlib.sha256(
        snapshot_path.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


@pytest.mark.asyncio
async def test_bundle_hashes_and_parses_each_exact_file_read_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    search_path = copied / "search.jsonl"
    record = json.loads(search_path.read_text(encoding="utf-8"))
    record["outcome"]["response"][0]["title"] = "swapped title"
    record["outcome_sha256"] = hashlib.sha256(
        _canonical(record["outcome"])
    ).hexdigest()
    swapped = _canonical(record) + b"\n"
    original_read_bytes = Path.read_bytes
    reads: dict[str, int] = {}

    def swapping_read(path: Path) -> bytes:
        if path.parent == copied:
            reads[path.name] = reads.get(path.name, 0) + 1
            if path == search_path and reads[path.name] > 1:
                return swapped
        return original_read_bytes(path)

    monkeypatch.setattr(Path, "read_bytes", swapping_read)
    loaded = ReplayBundle.load(copied)
    hits = await ReplaySearchProvider(loaded).search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert hits[0].title == "Synthetic result"
    assert reads == {"manifest.sha256": 1, **dict.fromkeys(REPLAY_FILES, 1)}


@pytest.mark.asyncio
async def test_fixture_exactly_replays_every_represented_operation(
    bundle: ReplayBundle,
) -> None:
    token = CancellationToken()
    model = await ReplayModelProvider(bundle).complete(
        _fixture_model_request(), deadline=_future_deadline(), cancellation_token=token
    )
    search = await ReplaySearchProvider(bundle).search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=_future_deadline(),
        cancellation_token=token,
    )
    document = await ReplayFetcher(bundle).fetch(
        "https://example.com/document",
        deadline=_future_deadline(),
        cancellation_token=token,
    )
    embeddings = await ReplayTextEmbedder(bundle).embed(
        ("synthetic one", "synthetic two"),
        deadline=_future_deadline(),
        cancellation_token=token,
    )

    assert model.output == "Synthetic answer."
    assert search[0].provider_metadata["source_id"] == "src-1"
    assert document.body_bytes == b"synthetic document"
    assert embeddings == ((0.1, 0.2), (0.3, 0.4))


@pytest.mark.asyncio
async def test_all_replay_operations_reject_expired_absolute_deadline(
    bundle: ReplayBundle,
) -> None:
    token = CancellationToken()
    expired = time.monotonic() - 1
    calls = (
        ReplayModelProvider(bundle).complete(
            _fixture_model_request(), deadline=expired, cancellation_token=token
        ),
        ReplayModelProvider(bundle).structured(
            _fixture_model_request(), dict[str, str], deadline=expired, cancellation_token=token
        ),
        ReplaySearchProvider(bundle).search(
            "multimodal agents",
            5,
            {"language": "en"},
            deadline=expired,
            cancellation_token=token,
        ),
        ReplayFetcher(bundle).fetch(
            "https://example.com/document", deadline=expired, cancellation_token=token
        ),
        ReplayTextEmbedder(bundle).embed(
            ("synthetic one", "synthetic two"),
            deadline=expired,
            cancellation_token=token,
        ),
    )
    for call in calls:
        with pytest.raises(ProviderError) as error:
            await call
        assert error.value.code == "TIMEOUT"

    stream = ReplayModelProvider(bundle).stream(
        _fixture_model_request(), deadline=expired, cancellation_token=token
    )
    with pytest.raises(ProviderError) as stream_error:
        await anext(stream)
    assert stream_error.value.code == "TIMEOUT"


@pytest.mark.asyncio
@pytest.mark.parametrize("deadline", (float("nan"), float("inf"), float("-inf")))
async def test_replay_rejects_nonfinite_deadlines_for_unary_and_stream(
    bundle: ReplayBundle, deadline: float
) -> None:
    cancelled = CancellationToken()
    cancelled.cancel()
    with pytest.raises(ValueError, match="finite"):
        await ReplayModelProvider(bundle).complete(
            _fixture_model_request(),
            deadline=deadline,
            cancellation_token=cancelled,
        )
    stream = ReplayModelProvider(bundle).stream(
        _fixture_model_request(),
        deadline=deadline,
        cancellation_token=cancelled,
    )
    with pytest.raises(ValueError, match="finite"):
        await anext(stream)


@pytest.mark.asyncio
async def test_replay_finite_deadline_keeps_cancellation_first(bundle: ReplayBundle) -> None:
    token = CancellationToken()
    token.cancel()
    with pytest.raises(OperationCancelled):
        await ReplayModelProvider(bundle).complete(
            _fixture_model_request(),
            deadline=time.monotonic() - 1,
            cancellation_token=token,
        )


def test_bundle_rejects_typed_invalid_success_with_consistent_hashes(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    record = json.loads((copied / "search.jsonl").read_text(encoding="utf-8"))
    record["outcome"]["response"] = "not-a-search-hit-list"
    _write_record(copied, "search.jsonl", record)

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


def test_bundle_rejects_non_string_model_complete_output_with_consistent_hashes(
    tmp_path: Path,
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    record = json.loads(
        (copied / "model_responses.jsonl").read_text(encoding="utf-8")
    )
    record["outcome"]["response"]["output"] = {"wrong": "type"}
    _write_record(copied, "model_responses.jsonl", record)

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


def test_bundle_rejects_incomplete_metadata_for_nonzero_operation(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    snapshot_path = copied / "snapshot.json"
    snapshot = json.loads(snapshot_path.read_text(encoding="utf-8"))
    snapshot["providers"]["embed"].pop("model_revision")
    snapshot_path.write_bytes(_canonical(snapshot) + b"\n")
    _update_file_hash(copied, "snapshot.json")

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


@pytest.mark.parametrize("framing", ["crlf", "missing-final-lf"])
def test_bundle_rejects_noncanonical_jsonl_framing(
    tmp_path: Path, framing: str
) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    search_path = copied / "search.jsonl"
    payload = search_path.read_bytes()
    if framing == "crlf":
        payload = payload.replace(b"\n", b"\r\n")
    else:
        payload = payload.removesuffix(b"\n")
    search_path.write_bytes(payload)
    _update_file_hash(copied, "search.jsonl")

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


def test_bundle_rejects_cross_file_record_with_consistent_hash(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    search_record = json.loads((copied / "search.jsonl").read_text(encoding="utf-8"))
    model_path = copied / "model_responses.jsonl"
    model_path.write_bytes(model_path.read_bytes() + _canonical(search_record) + b"\n")
    _update_file_hash(copied, "model_responses.jsonl")
    manifest_path = copied / "manifest.sha256"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["record_count_by_operation"]["search"] = 2
    manifest_path.write_bytes(_canonical(manifest) + b"\n")

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"
    assert "operation/file mismatch" in str(error.value.__cause__)
