import hashlib
import json
import shutil
from pathlib import Path

import pytest

from deepresearch.providers import ProviderError
from deepresearch.providers.replay import ReplaySearchProvider
from deepresearch.providers.replay_schema import ReplayBundle
from deepresearch.runtime import CancellationToken

FIXTURE_ROOT = Path(__file__).parents[2] / "fixtures" / "replay" / "provider_contract"


@pytest.fixture
def bundle() -> ReplayBundle:
    return ReplayBundle.load(FIXTURE_ROOT)


@pytest.mark.asyncio
async def test_replay_search_exact_request_returns_recorded_hit(bundle: ReplayBundle) -> None:
    provider = ReplaySearchProvider(bundle)

    hits = await provider.search(
        "multimodal agents",
        5,
        {"language": "en"},
        deadline=100.0,
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
            deadline=100.0,
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "REPLAY_MISS"
    assert provider.live_calls == 0


def test_bundle_verification_rejects_tampering_and_duplicate_keys(tmp_path: Path) -> None:
    copied = tmp_path / "bundle"
    shutil.copytree(FIXTURE_ROOT, copied)
    search_path = copied / "search.jsonl"
    record = json.loads(search_path.read_text(encoding="utf-8"))
    search_path.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
        + json.dumps(record, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ProviderError) as error:
        ReplayBundle.load(copied)

    assert error.value.code == "INVALID_SNAPSHOT"


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
