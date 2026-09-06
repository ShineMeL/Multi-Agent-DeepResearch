from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmarks.datasets.isolation import GoldIsolationGuard
from benchmarks.datasets.models import AnnotatedQuestion
from experiments.config import FormalExperimentConfig, canonical_sha256
from experiments.runner import ExperimentRunner


def _config_and_task() -> tuple[FormalExperimentConfig, Any]:
    payload = yaml.safe_load(Path("benchmarks/configs/formal.template.yaml").read_text())
    question = AnnotatedQuestion.model_validate_json(
        Path("benchmarks/datasets/templates/question.example.json").read_bytes()
    )
    task = GoldIsolationGuard.runtime_view(question).model_copy(
        update={
            "task_id": "test-t1",
            "corpus_version": payload["corpus_version"],
            "index_version": payload["index_version"],
            "request": question.request.model_copy(
                update={
                    "execution_mode": "hybrid",
                    "access_profile": "local",
                    "run_purpose": "benchmark",
                    "provider_profile_id": payload["provider_profile_id"],
                    "budget_preset": "medium",
                }
            ),
        }
    )
    payload.update({name: "a" * 64 for name in (
        "private_manifest_sha256", "model_snapshot_sha256",
        "judge_model_snapshot_sha256", "r1_model_snapshot_sha256",
        "serving_environment_sha256", "code_tree_sha256",
    )})
    for name in ("main_test_task_ids", "stability_task_ids", "cost_subset_task_ids", "p0_task_ids", "oracle_task_ids"):
        payload[name] = [task.task_id]
    payload["internal_runtime_task_hashes"] = {
        task.task_id: canonical_sha256(task.model_dump(mode="json"))
    }
    return FormalExperimentConfig.model_validate(payload), task


class SpyLauncher:
    def __init__(self) -> None:
        self.requests: list[Any] = []

    async def __call__(self, request: Any) -> None:
        self.requests.append(request)


@pytest.mark.asyncio
async def test_abcd_runner_uses_exact_component_pairs(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
    )
    result = await runner.run_abcd(config=config)
    assert result.variant_components == {
        "A": ("P1", "R1"),
        "B": ("P1", "R2"),
        "C": ("P2", "R1"),
        "D": ("P2", "R2"),
    }


@pytest.mark.asyncio
async def test_ranker_protocol_reuses_identical_candidate_pool(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
    )
    result = await runner.run_ranker_component(config=config, task_ids=[task.task_id])
    hashes = {run.candidate_pool_hash for run in result.runs}
    assert len(hashes) == 1
    assert {run.ranker_id for run in result.runs} == {"R0", "R1", "R2"}
    assert len([item for item in launcher.requests if item.kind == "candidate_pool"]) == 1
