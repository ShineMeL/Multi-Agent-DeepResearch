from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.external import (
    BENCHMARK_NAMES,
    FramesAdapter,
)
from benchmarks.external.deepresearchbench import DeepResearchBenchAdapter
from benchmarks.external.livedrbench import LiveDRBenchAdapter
from experiments.config import FormalExperimentConfig
from experiments.external_runner import ExternalExperimentRunner
from experiments.models import canonical_sha256

pytest_plugins = ("tests.unit.benchmarks.test_external_adapters",)


class SpyLauncher:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def __call__(self, request: object) -> None:
        self.requests.append(request)


@pytest.mark.asyncio
async def test_missing_external_authorization_fails_before_agent_launch(tmp_path: Path):
    payload = yaml.safe_load(Path("benchmarks/configs/formal.template.yaml").read_text())
    payload.update(
        {
            name: "a" * 64
            for name in (
                "private_manifest_sha256",
                "model_snapshot_sha256",
                "judge_model_snapshot_sha256",
                "r1_model_snapshot_sha256",
                "serving_environment_sha256",
                "code_tree_sha256",
            )
        }
    )
    payload.update(
        {
            name: ["test-a"]
            for name in (
                "main_test_task_ids",
                "stability_task_ids",
                "cost_subset_task_ids",
                "p0_task_ids",
                "oracle_task_ids",
            )
        }
    )
    payload["internal_runtime_task_hashes"] = {"test-a": "b" * 64}
    config = FormalExperimentConfig.model_validate(payload)
    launcher = SpyLauncher()
    runner = ExternalExperimentRunner(launch_agent=launcher, repo_root=tmp_path, preflight=False)
    with pytest.raises(ValueError, match="authorization"):
        await runner.run(
            config=config,
            external_config_path=tmp_path / "external.yaml",
            external_lock_path=tmp_path / "external.lock.json",
            benchmarks=("frames",),
        )
    assert launcher.requests == []


@pytest.mark.asyncio
async def test_external_runner_stages_source_config_and_redacts_plan(external_fixture):
    repo, external, external_config_path, external_lock_path, raw_root, snapshot_root = external_fixture
    hashes: dict[str, str] = {}
    adapters = {
        "livedrbench": LiveDRBenchAdapter,
        "frames": FramesAdapter,
        "deepresearchbench": DeepResearchBenchAdapter,
    }
    for benchmark in BENCHMARK_NAMES:
        adapter = adapters[benchmark](
            lock_file=external_lock_path,
            raw_root=raw_root,
            snapshot_root=snapshot_root,
            external_config=external,
        )
        for selection in adapter.select(
            provider_profile_id="formal-local-vllm", budget_preset="medium"
        ):
            hashes[selection.runtime_task.task_id] = canonical_sha256(
                selection.runtime_task.model_dump(mode="json")
            )

    payload = yaml.safe_load(Path("benchmarks/configs/formal.template.yaml").read_text())
    payload.update(
        {
            name: "a" * 64
            for name in (
                "private_manifest_sha256",
                "model_snapshot_sha256",
                "judge_model_snapshot_sha256",
                "r1_model_snapshot_sha256",
                "serving_environment_sha256",
                "code_tree_sha256",
            )
        }
    )
    payload.update(
        {
            name: ["test-a"]
            for name in (
                "main_test_task_ids",
                "stability_task_ids",
                "cost_subset_task_ids",
                "p0_task_ids",
                "oracle_task_ids",
            )
        }
    )
    payload.update(
        {
            "internal_runtime_task_hashes": {"test-a": "b" * 64},
            "external_config_sha256": sha256_bytes(external_config_path.read_bytes()),
            "external_lock_sha256": sha256_bytes(external_lock_path.read_bytes()),
            "external_runtime_task_hashes": dict(sorted(hashes.items())),
        }
    )
    config = FormalExperimentConfig.model_validate(payload)
    formal_path = repo / "benchmarks/configs/formal-portfolio.yaml"
    formal_bytes = b"# sealed portfolio source\n" + canonical_json_bytes(
        config.model_dump(mode="json")
    )
    formal_path.write_bytes(formal_bytes)
    launcher = SpyLauncher()
    runner = ExternalExperimentRunner(
        launch_agent=launcher,
        repo_root=repo,
        experiment_root=repo / "experiments",
        config_source=formal_path,
        preflight=False,
    )
    result = await runner.run(
        config=config,
        external_config_path=external_config_path,
        external_lock_path=external_lock_path,
        benchmarks=("frames",),
    )
    assert result.formal_config_sha256 == sha256_bytes(formal_bytes)
    assert len(launcher.requests) == 20
    request = launcher.requests[0]
    assert "evaluation_plan" not in request.model_dump_json()
    staged = RuntimeTask.model_validate_json(
        Path(request.runtime_task_path).read_text(encoding="utf-8")
    )
    assert set(staged.model_dump()) == {
        "task_id",
        "category",
        "request",
        "evaluation_cutoff",
        "snapshot_id",
        "corpus_version",
        "index_version",
    }
    default_runner = ExternalExperimentRunner(
        repo_root=repo,
        experiment_root=repo / "experiments",
        config_source=formal_path,
        preflight=False,
    )
    receipt = await default_runner._default_launch_agent(request)  # pyright: ignore[reportPrivateUsage]
    assert receipt.task_id == staged.task_id
