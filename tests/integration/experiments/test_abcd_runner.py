from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmarks.datasets.isolation import AgentRuntimeGuard, GoldAccessViolation, GoldIsolationGuard
from benchmarks.datasets.models import AnnotatedQuestion
from benchmarks.datasets.validator import sha256_bytes
from benchmarks.processes.agent import AgentVariantRunRequest, load_authorized_agent_inputs
from benchmarks.processes.evaluator import stage_authorized_runtime_task, stage_sealed_config
from deepresearch.runtime import CheckpointRef
from experiments.config import FormalExperimentConfig, canonical_sha256
from experiments.models import ExperimentVariant
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
async def test_missing_agent_receipt_is_recorded_as_failed(tmp_path: Path) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    run = await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
    )
    assert run.status == "failed"
    assert run.validity == "invalid"
    assert run.error_code == "AGENT_RECEIPT_INVALID"
    assert run.usage.cost_usd is None


@pytest.mark.asyncio
async def test_formal_runner_requires_private_manifest_preflight(tmp_path: Path) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
    )
    with pytest.raises(RuntimeError, match="private manifest"):
        await runner.run_one(
            config=config,
            protocol="end_to_end",
            task_id=task.task_id,
            variant=ExperimentVariant.D,
            budget_preset=config.budget_preset,
            seed=config.replication.seed_values[0],
        )


@pytest.mark.asyncio
async def test_missing_candidate_pool_does_not_fallback_to_empty_pool(tmp_path: Path) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    result = await runner.run_ranker_component(config=config, task_ids=[task.task_id])
    group_root = tmp_path / config.experiment_group_id()
    assert not tuple((group_root / "candidate-pools").glob("*.json"))
    assert result.runs
    assert all(run.status == "failed" for run in result.runs)


@pytest.mark.asyncio
async def test_group_binds_staged_config_bytes(tmp_path: Path) -> None:
    config, task = _config_and_task()
    source = tmp_path / "sealed-formal.yaml"
    source.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path / "experiments",
        private_root=tmp_path / "private",
        config_source=source,
        preflight=False,
    )
    await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
    )
    group_root = tmp_path / "experiments" / config.experiment_group_id()
    group = yaml.safe_load((group_root / "group.json").read_text(encoding="utf-8"))
    staged = (group_root / "config" / "formal.yaml").read_bytes()
    assert group["config_sha256"] == sha256_bytes(staged)
    assert staged == source.read_bytes()


def test_agent_loader_rejects_tampered_snapshot_manifest(tmp_path: Path) -> None:
    config, task = _config_and_task()
    root = tmp_path / "group"
    runtime_root = root / "agent-inputs"
    snapshots = tmp_path / "snapshots"
    snapshot = snapshots / task.task_id
    (root / "config").mkdir(parents=True)
    for directory in (runtime_root, root / "requests", root / "candidate-pools", root / "resume-checkpoints"):
        directory.mkdir(parents=True)
    private = tmp_path / "private"
    private.mkdir()
    config_source = tmp_path / "formal.yaml"
    config_source.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    config_path = stage_sealed_config(
        config_source,
        expected_sha256=sha256_bytes(config_source.read_bytes()),
        group_run_root=root,
    )
    staged = stage_authorized_runtime_task(
        task,
        config=config,
        budget_preset=config.budget_preset,
        agent_input_root=runtime_root,
        request_id="request-1",
        forbidden_private_root=private,
    )
    snapshot.mkdir(parents=True)
    files = {"documents.jsonl": b"", "index.json": b"{}\n"}
    snapshot_payload = {
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "corpus_version": task.corpus_version,
        "index_version": task.index_version,
        "documents_sha256": sha256_bytes(files["documents.jsonl"]),
        "index_sha256": sha256_bytes(files["index.json"]),
    }
    files["snapshot.json"] = json.dumps(snapshot_payload, sort_keys=True).encode("utf-8")
    for name, payload in files.items():
        (snapshot / name).write_bytes(payload)
    manifest = {"file_sha256": {name: sha256_bytes(payload) for name, payload in files.items()}}
    (snapshot / "manifest.sha256").write_text(json.dumps(manifest), encoding="utf-8")
    (snapshot / "documents.jsonl").write_bytes(b"tampered")
    request = AgentVariantRunRequest(
        task_id=task.task_id,
        runtime_task_path=staged.runtime_task_path,
        runtime_task_sha256=staged.runtime_task_sha256,
        base_runtime_task_sha256=staged.base_runtime_task_sha256,
        snapshot_dir=str(snapshot.resolve()),
        run_dir=str(root.resolve()),
        config_path=str(config_path.resolve()),
        config_sha256=sha256_bytes(config_source.read_bytes()),
        seed=config.replication.seed_values[0],
        protocol="end_to_end",
        variant="D",
        budget_preset=config.budget_preset,
    )
    guard = AgentRuntimeGuard(
        runtime_root=runtime_root,
        snapshot_root=snapshots,
        run_root=root,
    )
    with pytest.raises((GoldAccessViolation, ValueError), match="snapshot|hash"):
        load_authorized_agent_inputs(request, guard=guard)


@pytest.mark.asyncio
async def test_resume_stages_checkpoint_and_binds_identity(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    kwargs = {
        "config": config,
        "protocol": "end_to_end",
        "task_id": task.task_id,
        "variant": ExperimentVariant.D,
        "budget_preset": config.budget_preset,
        "seed": config.replication.seed_values[0],
    }
    first = await runner.run_one(**kwargs)
    assert first.status == "failed"
    group_root = tmp_path / config.experiment_group_id()
    key = runner._idempotency_key(
        config.experiment_group_id(),
        "end_to_end",
        "D",
        task.task_id,
        config.replication.seed_values[0],
        None,
        config.budget_preset,
    )
    checkpoint = group_root / "artifacts" / f"{key}.sqlite3"
    with sqlite3.connect(checkpoint) as connection:
        connection.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)"
        )
        connection.execute("INSERT INTO checkpoints VALUES (?, ?, ?)", ("thread-1", "", "cp-1"))
    await runner.run_one(
        **kwargs,
        resume=True,
        resume_checkpoint_path=checkpoint,
        resume_checkpoint_ref=CheckpointRef(
            checkpoint_id="cp-1",
            thread_id="thread-1",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        ),
    )
    resumed = launcher.requests[-1]
    assert resumed.resume_checkpoint_path is not None
    assert Path(resumed.resume_checkpoint_path).parent == group_root / "resume-checkpoints"
    assert resumed.resume_checkpoint_sha256 == sha256_bytes(
        Path(resumed.resume_checkpoint_path).read_bytes()
    )
    assert resumed.resume_checkpoint_ref is not None


@pytest.mark.asyncio
async def test_resume_requires_prior_failed_attempt_for_same_idempotency_key(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    group_root = tmp_path / config.experiment_group_id()
    await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
    )
    key = runner._idempotency_key(
        config.experiment_group_id(),
        "end_to_end",
        "D",
        task.task_id,
        config.replication.seed_values[0],
        None,
        config.budget_preset,
    )
    (group_root / "raw" / f"{key}.json").unlink()
    checkpoint = group_root / "artifacts" / f"{key}.sqlite3"
    with sqlite3.connect(checkpoint) as connection:
        connection.execute(
            "CREATE TABLE checkpoints (thread_id TEXT, checkpoint_ns TEXT, checkpoint_id TEXT)"
        )
        connection.execute("INSERT INTO checkpoints VALUES (?, ?, ?)", ("thread-1", "", "cp-1"))
    result = await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
        resume=True,
        resume_checkpoint_path=checkpoint,
        resume_checkpoint_ref=CheckpointRef(
            checkpoint_id="cp-1",
            thread_id="thread-1",
            created_at=datetime(2026, 9, 1, tzinfo=UTC),
        ),
    )
    assert result.status == "failed"
    assert result.error_code == "RESUME_CHECKPOINT_INVALID"
    assert len(launcher.requests) == 1


@pytest.mark.asyncio
async def test_abcd_runner_uses_exact_component_pairs(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
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
        preflight=False,
    )
    result = await runner.run_ranker_component(config=config, task_ids=[task.task_id])
    hashes = {run.candidate_pool_hash for run in result.runs}
    assert len(hashes) == 1
    assert {run.ranker_id for run in result.runs} == {"R0", "R1", "R2"}
    assert len([item for item in launcher.requests if item.kind == "candidate_pool"]) == 1
