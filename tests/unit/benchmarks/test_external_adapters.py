from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.external import (
    BENCHMARK_COUNTS,
    DeepResearchBenchAdapter,
    ExternalBenchmarkLock,
    ExternalSnapshotLock,
    ExternalSourceLock,
    FramesAdapter,
    fixed_external_root,
    load_external_config,
    write_external_json,
)
from benchmarks.external.base import ExternalSnapshotMaterializer, _as_frozen_records
from benchmarks.external.livedrbench import LiveDRBenchAdapter
from benchmarks.scripts.build_snapshot import build_one
from benchmarks.scripts.fetch_external import _parse_counts, restore
from deepresearch.providers.errors import ProviderError


def _item(benchmark: str, index: int) -> dict[str, object]:
    documents = [
        {
            "id": f"{benchmark}-doc-{index}-{part}",
            "url": f"https://example.org/{benchmark}/{index}/{part}",
            "title": f"{benchmark} context {index} {part}",
            "text": f"Evidence text for {benchmark} task {index}, document {part}.",
        }
        for part in range(2 if benchmark == "frames" else 1)
    ]
    item: dict[str, object] = {
        "external_id": f"{benchmark}-{index:02d}",
        "question": f"What does {benchmark} task {index} ask?",
        "documents": documents,
        "rubric": {"private": "ignored"},
        "acceptable_claims": ["ignored"],
    }
    if benchmark == "livedrbench":
        item["task_type"] = "computer-science prior-art"
    elif benchmark == "frames":
        item["evidence_mapping"] = {documents[0]["id"]: ["claim-1"]}
    else:
        item["task_type"] = "research-report"
        item["citations"] = [{"id": documents[0]["id"], "url": documents[0]["url"]}]
    return item


@pytest.fixture
def external_fixture(tmp_path: Path):
    repo = tmp_path / "repo"
    raw_root = repo / "benchmarks/private/external/raw"
    staging_root = repo / "benchmarks/private/external/staging"
    snapshot_root = repo / "benchmarks/snapshots/external"
    lock_path = repo / "benchmarks/external/external.lock.json"
    for path in (raw_root, staging_root, snapshot_root, lock_path.parent, repo / "benchmarks/configs"):
        path.mkdir(parents=True, exist_ok=True)
    config = load_external_config(Path("benchmarks/configs/external.yaml"))
    entries: dict[str, ExternalSourceLock] = {}
    for benchmark in ("livedrbench", "frames", "deepresearchbench"):
        count = BENCHMARK_COUNTS[benchmark]
        items = [_item(benchmark, index) for index in range(count)]
        payload = canonical_json_bytes(items)
        raw_path = raw_root / f"{benchmark}.json"
        raw_path.write_bytes(payload)
        snapshots: list[ExternalSnapshotLock] = []
        for item in items:
            external_id = str(item["external_id"])
            task_id = f"ext-{benchmark}-{external_id}"
            records = _as_frozen_records(
                item,
                benchmark=benchmark,
                external_id=external_id,
                task_id=task_id,
                retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
            )
            documents = staging_root / f"{task_id}.jsonl"
            documents.write_bytes(
                b"".join(canonical_json_bytes(record.model_dump(mode="json")) for record in records)
            )
            output = snapshot_root / benchmark / task_id
            manifest = build_one(
                task_id=task_id,
                documents=documents,
                output=output,
                corpus_version=config.spec(benchmark).corpus_version,
                index_version=config.index_version,
            )
            snapshots.append(
                ExternalSnapshotLock(
                    benchmark=benchmark,
                    external_id=external_id,
                    task_id=task_id,
                    snapshot_id=manifest.snapshot_id,
                    corpus_version=manifest.corpus_version,
                    index_version=manifest.index_version,
                    snapshot_relative_path=f"{benchmark}/{task_id}",
                    records_sha256=manifest.documents_sha256,
                    manifest_sha256=sha256_bytes((output / "manifest.sha256").read_bytes()),
                )
            )
        entries[benchmark] = ExternalSourceLock(
            benchmark=benchmark,
            upstream_revision="1" * 40,
            source_url=f"https://example.org/{benchmark}",
            raw_relative_path=f"{benchmark}.json",
            sha256=sha256_bytes(payload),
            license_id="fixture-license",
            adapter_version=config.spec(benchmark).adapter_version,
            fetched_at=datetime(2026, 8, 29, tzinfo=UTC),
            corpus_version=config.spec(benchmark).corpus_version,
            index_version=config.index_version,
            snapshot_locks=tuple(sorted(snapshots, key=lambda row: row.task_id)),
        )
    lock = ExternalBenchmarkLock(
        livedrbench=entries["livedrbench"],
        frames=entries["frames"],
        deepresearchbench=entries["deepresearchbench"],
    )
    lock_path.write_bytes(canonical_json_bytes(lock.model_dump(mode="json")))
    config_path = repo / "benchmarks/configs/external.yaml"
    config_path.write_text(Path("benchmarks/configs/external.yaml").read_text(), encoding="utf-8")
    return repo, config, config_path, lock_path, raw_root, snapshot_root


def test_external_config_uses_fixed_roots():
    config = load_external_config(Path("benchmarks/configs/external.yaml"))
    assert config.raw_root == "benchmarks/private/external/raw"
    assert config.documents_staging_root == "benchmarks/private/external/staging"
    assert config.snapshot_root == "benchmarks/snapshots/external"


def test_lock_requires_immutable_revision_hash_and_license():
    source = {
        "benchmark": "frames",
        "upstream_revision": "1" * 40,
        "source_url": "https://example.org/frames",
        "raw_relative_path": "frames.json",
        "sha256": "a" * 64,
        "license_id": "fixture",
        "adapter_version": "frames-adapter-v1",
        "fetched_at": "2026-08-29T00:00:00Z",
        "corpus_version": "frames-v1",
        "index_version": "bm25-mixed-v1",
    }
    with pytest.raises(ValidationError):
        ExternalSourceLock.model_validate({**source, "sha256": ""})
    with pytest.raises(ValidationError):
        ExternalSourceLock.model_validate({**source, "upstream_revision": "main"})


def test_external_selection_is_pinned_and_deterministic(external_fixture):
    _, config, _, lock_path, raw_root, snapshot_root = external_fixture
    adapter = FramesAdapter(
        lock_file=lock_path,
        raw_root=raw_root,
        snapshot_root=snapshot_root,
        external_config=config,
    )
    first = adapter.select(provider_profile_id="formal-local-vllm", budget_preset="medium")
    second = adapter.select(provider_profile_id="formal-local-vllm", budget_preset="medium")
    assert first == second
    assert len(first) == 20
    assert all(isinstance(item.runtime_task, RuntimeTask) for item in first)
    assert all(
        set(item.runtime_task.model_dump())
        == {
            "task_id",
            "category",
            "request",
            "evaluation_cutoff",
            "snapshot_id",
            "corpus_version",
            "index_version",
        }
        for item in first
    )
    assert all(item.evaluation_plan.benchmark == "frames" for item in first)


@pytest.mark.parametrize(
    ("adapter_type", "benchmark"),
    [
        (LiveDRBenchAdapter, "livedrbench"),
        (FramesAdapter, "frames"),
        (DeepResearchBenchAdapter, "deepresearchbench"),
    ],
)
def test_external_adapters_apply_semantic_positive_and_negative_filters(
    external_fixture, adapter_type, benchmark
):
    _, config, _, lock_path, raw_root, snapshot_root = external_fixture
    adapter = adapter_type(
        lock_file=lock_path,
        raw_root=raw_root,
        snapshot_root=snapshot_root,
        external_config=config,
    )
    positive = _item(benchmark, 0)
    assert adapter._eligible(positive)  # pyright: ignore[reportPrivateUsage]
    if benchmark == "livedrbench":
        negative = {**positive, "task_type": "physics prior-art"}
    elif benchmark == "frames":
        negative = {**positive, "evidence_mapping": None}
        single_document = {**positive, "documents": positive["documents"][:1], "allow_single_document": True}
        assert not adapter._eligible(single_document)  # pyright: ignore[reportPrivateUsage]
    else:
        negative = {**positive, "citations": []}
    assert not adapter._eligible(negative)  # pyright: ignore[reportPrivateUsage]


def test_external_adapter_rechecks_cached_raw_bytes_after_mutation(external_fixture):
    _, config, _, lock_path, raw_root, snapshot_root = external_fixture
    adapter = FramesAdapter(
        lock_file=lock_path,
        raw_root=raw_root,
        snapshot_root=snapshot_root,
        external_config=config,
    )
    adapter.select(provider_profile_id="formal-local-vllm", budget_preset="medium")
    (raw_root / "frames.json").write_bytes(b"[]\n")
    with pytest.raises(ProviderError, match="hash mismatch"):
        adapter.select(provider_profile_id="formal-local-vllm", budget_preset="medium")


def test_restore_rejects_raw_payload_tampering(external_fixture):
    repo, config, _, lock_path, raw_root, snapshot_root = external_fixture
    raw_path = raw_root / "frames.json"
    raw_path.write_bytes(raw_path.read_bytes() + b"\n")
    with pytest.raises(ValueError, match="raw payload hash mismatch"):
        restore(
            config=config,
            raw_root=raw_root,
            documents_staging_root=repo / "benchmarks/private/external/staging",
            snapshot_root=snapshot_root,
            lock_path=lock_path,
            repo_root=repo,
        )


def test_restore_rejects_staging_document_tampering(external_fixture):
    repo, config, _, lock_path, raw_root, snapshot_root = external_fixture
    lock = ExternalBenchmarkLock.model_validate_json(lock_path.read_bytes())
    task_id = lock.frames.snapshot_locks[0].task_id
    documents = repo / "benchmarks/private/external/staging" / f"{task_id}.jsonl"
    documents.write_bytes(documents.read_bytes() + b"tampered")
    with pytest.raises(ValueError, match="staging document hash mismatch"):
        restore(
            config=config,
            raw_root=raw_root,
            documents_staging_root=repo / "benchmarks/private/external/staging",
            snapshot_root=snapshot_root,
            lock_path=lock_path,
            repo_root=repo,
        )


def test_cli_expected_counts_rejects_noncanonical_portfolio():
    with pytest.raises(ValueError, match="canonical 10/20/10"):
        _parse_counts("frames=1")


def test_external_raw_and_snapshot_roots_reject_wrong_suffix(tmp_path: Path):
    with pytest.raises(ValueError):
        fixed_external_root(tmp_path / "not-raw", kind="raw")
    with pytest.raises(ValueError):
        fixed_external_root(tmp_path / "not-snapshot", kind="snapshot")


def test_external_root_is_bound_to_explicit_repository(tmp_path: Path):
    repo = tmp_path / "repo"
    outside = tmp_path / "other" / "benchmarks" / "private" / "external" / "raw"
    outside.mkdir(parents=True)
    with pytest.raises(ValueError, match="this repository"):
        fixed_external_root(outside, kind="raw", repo_root=repo)


def test_external_json_publication_is_atomic_and_no_replace(tmp_path: Path):
    target = tmp_path / "nested" / "artifact.json"
    write_external_json(target, {"version": 1})
    with pytest.raises(FileExistsError):
        write_external_json(target, {"version": 1})
    assert target.read_bytes() == canonical_json_bytes({"version": 1})
    assert not tuple(target.parent.glob("*.staging"))


def test_external_snapshot_lock_path_is_namespaced():
    payload = {
        "benchmark": "frames",
        "external_id": "frames-01",
        "task_id": "ext-frames-frames-01",
        "snapshot_id": "snapshot-frames-01",
        "corpus_version": "frames-v1",
        "index_version": "bm25-mixed-v1",
        "snapshot_relative_path": "livedrbench/ext-frames-frames-01",
        "records_sha256": "a" * 64,
        "manifest_sha256": "b" * 64,
    }
    with pytest.raises(ValidationError):
        ExternalSnapshotLock.model_validate(payload)


def test_external_materializer_rejects_manifest_outside_staging(tmp_path: Path):
    staging = tmp_path / "repo/benchmarks/private/external/staging"
    snapshots = tmp_path / "repo/benchmarks/snapshots/external"
    staging.mkdir(parents=True)
    snapshots.mkdir(parents=True)
    outside = tmp_path / "selection.json"
    outside.write_text('{"selections": []}', encoding="utf-8")
    with pytest.raises(ValueError, match="inside external staging root"):
        ExternalSnapshotMaterializer().build(
            selection_manifest_path=outside,
            documents_staging_root=staging,
            snapshot_root=snapshots,
        )


def test_external_record_materialization_rejects_missing_evidence_fields():
    with pytest.raises(ValueError, match="question"):
        _as_frozen_records(
            {"documents": [{"text": "evidence without a question or URL"}]},
            benchmark="frames",
            external_id="frames-01",
            task_id="ext-frames-frames-01",
            retrieved_at=datetime(2026, 8, 29, tzinfo=UTC),
        )


def test_restore_rebuilds_only_an_absent_snapshot(external_fixture):
    _, config, _, lock_path, raw_root, snapshot_root = external_fixture
    lock = ExternalBenchmarkLock.model_validate_json(lock_path.read_bytes())
    lock = ExternalBenchmarkLock(
        **{
            benchmark: lock.entry(benchmark).model_copy(
                update={"snapshot_locks": lock.entry(benchmark).snapshot_locks[: BENCHMARK_COUNTS[benchmark]]}
            )
            for benchmark in ("livedrbench", "frames", "deepresearchbench")
        }
    )
    lock_path.write_bytes(canonical_json_bytes(lock.model_dump(mode="json")))
    missing = lock.frames.snapshot_locks[0]
    missing_path = snapshot_root / missing.snapshot_relative_path
    for path in sorted(missing_path.rglob("*"), reverse=True):
        if path.is_file():
            path.unlink()
        else:
            path.rmdir()
    missing_path.rmdir()
    counts = restore(
        config=config,
        raw_root=raw_root,
        documents_staging_root=raw_root.parent / "staging",
        snapshot_root=snapshot_root,
        lock_path=lock_path,
    )
    assert counts == {"livedrbench": 10, "frames": 20, "deepresearchbench": 10}
    assert missing_path.is_dir()
