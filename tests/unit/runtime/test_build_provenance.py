"""Packaged execution retains validated audit identifiers without Git installed."""

import subprocess
import time
from pathlib import Path

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from deepresearch.runtime import CancellationToken
from deepresearch.runtime.checkpoints import checkpoint_serializer
from deepresearch.runtime.manifest import CostCalculator, RunManifest
from deepresearch.runtime.provenance import BuildProvenance, resolve_build_provenance
from deepresearch.workflow.runner import BaselineRuntimeHooks
from tests.integration.replay.test_baseline_graph import ControlledSegmentClock, MemoryEventSink
from tests.unit.runtime.test_runner_factory_execution import composition


@pytest.mark.parametrize("revision", [None, "a" * 40])
async def test_service_executes_and_audits_without_git_executable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    revision: str | None,
) -> None:
    def missing_git(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("git is absent from runtime image")

    monkeypatch.setattr(subprocess, "run", missing_git)
    monkeypatch.delenv("DEEPRESEARCH_CODE_COMMIT", raising=False)
    monkeypatch.delenv("DEEPRESEARCH_DEPENDENCY_LOCK_SHA256", raising=False)
    if revision is not None:
        monkeypatch.setenv("DEEPRESEARCH_CODE_COMMIT", revision)
    clock = ControlledSegmentClock(monotonic_start=time.monotonic(), utc_offset_seconds=0)
    monkeypatch.setattr(
        "deepresearch.runtime.runner_factory.paired_runtime_hooks",
        lambda: BaselineRuntimeHooks(monotonic=clock.monotonic, utc_now=clock.utc_now),
    )
    builder, conf, routes, snapshots, _, artifacts = composition(tmp_path)
    runner = builder.build(
        config=conf,
        provider_routes=routes,
        pricing_snapshots=snapshots,
        checkpointer=InMemorySaver(serde=checkpoint_serializer()),
        cost_calculator=CostCalculator,
    )
    result = await runner.run(
        run_id="packaged",
        thread_id="packaged",
        config=conf,
        checkpoint=None,
        emit=MemoryEventSink(),
        cancellation_token=CancellationToken(),
    )
    assert result.status == "completed", result.error_code
    assert result.manifest_artifact_id is not None
    manifest = RunManifest.model_validate_json(artifacts.get_bytes(result.manifest_artifact_id))
    assert manifest.code_commit == (revision or "0" * 40)
    assert manifest.dependency_lock_sha256 != "0" * 64


@pytest.mark.parametrize(
    "name,value",
    [
        ("DEEPRESEARCH_CODE_COMMIT", "unknown"),
        ("DEEPRESEARCH_DEPENDENCY_LOCK_SHA256", "not-a-digest"),
    ],
)
def test_service_rejects_malformed_runtime_provenance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    name: str,
    value: str,
) -> None:
    monkeypatch.setenv(name, value)
    builder, conf, routes, snapshots, _, _ = composition(tmp_path)
    with pytest.raises(ValueError):
        builder.build(
            config=conf,
            provider_routes=routes,
            pricing_snapshots=snapshots,
            checkpointer=InMemorySaver(),
            cost_calculator=CostCalculator,
        )


def test_packaged_provenance_uses_explicit_hashes_or_stable_unknown_sentinels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DEEPRESEARCH_CODE_COMMIT", raising=False)
    monkeypatch.delenv("DEEPRESEARCH_DEPENDENCY_LOCK_SHA256", raising=False)
    assert resolve_build_provenance(tmp_path) == BuildProvenance("0" * 40, "0" * 64)
    monkeypatch.setenv("DEEPRESEARCH_CODE_COMMIT", "a" * 40)
    monkeypatch.setenv("DEEPRESEARCH_DEPENDENCY_LOCK_SHA256", "b" * 64)
    assert resolve_build_provenance(tmp_path) == BuildProvenance("a" * 40, "b" * 64)


def test_runtime_lock_cannot_override_different_packaged_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    (tmp_path / "uv.lock").write_bytes(b"version = 1\n")
    monkeypatch.setenv("DEEPRESEARCH_DEPENDENCY_LOCK_SHA256", "b" * 64)
    with pytest.raises(ValueError, match="does not match"):
        resolve_build_provenance(tmp_path)
