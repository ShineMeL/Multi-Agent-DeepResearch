from __future__ import annotations

import hashlib
import json
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest
import yaml

from benchmarks.datasets.isolation import AgentRuntimeGuard, GoldAccessViolation, GoldIsolationGuard
from benchmarks.datasets.models import AnnotatedQuestion
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.processes.agent import (
    AgentRunReceipt,
    AgentVariantRunRequest,
    load_authorized_agent_inputs,
)
from benchmarks.processes.evaluator import stage_authorized_runtime_task, stage_sealed_config
from deepresearch.domain import ResourceUsage, RunBudget
from deepresearch.runtime import CheckpointRef
from deepresearch.runtime.manifest import RunManifest
from experiments.config import FormalExperimentConfig, canonical_sha256
from experiments.models import ExperimentTaskRun, ExperimentVariant
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
async def test_invalid_receipt_is_rejected_before_evaluator_callback(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    evaluator_calls: list[object] = []

    async def evaluator(request: object, receipt: object) -> None:
        del request
        evaluator_calls.append(receipt)

    runner = ExperimentRunner(
        launch_agent=launcher,
        evaluator=evaluator,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    result = await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
    )
    assert result.status == "failed"
    assert result.error_code == "AGENT_RECEIPT_INVALID"
    assert evaluator_calls == []


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
async def test_invalid_candidate_pool_does_not_reuse_completed_raw_result(
    tmp_path: Path,
) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    group_root = runner._group_root(config)
    key = runner._idempotency_key(
        config.experiment_group_id(),
        "ranker_component",
        "R1",
        task.task_id,
        config.replication.seed_values[0],
        None,
        config.budget_preset,
    )
    existing = ExperimentTaskRun(
        task_id=task.task_id,
        protocol="ranker_component",
        variant="R1",
        planner_id="P1",
        ranker_id="R1",
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
        status="completed",
        candidate_pool_hash="c" * 64,
        manifest_path="completed-manifest.json",
        artifact_ids=(),
        usage=ResourceUsage.zero(cost_known=True),
        pricing_snapshot_ids=(config.pricing_snapshot.snapshot_id,),
        pricing_status="estimated",
        cost_label="estimated_from_normalized_schedule",
        category=task.category,
    )
    raw_path = group_root / "raw" / f"{key}.json"
    raw_bytes = canonical_json_bytes(existing.model_dump(mode="json"))
    raw_path.write_bytes(raw_bytes)

    result = await runner.run_ranker_component(config=config, task_ids=[task.task_id])

    r1 = next(run for run in result.runs if run.variant.value == "R1")
    assert r1.status == "failed"
    assert r1.error_code == "CANDIDATE_POOL_INVALID"
    assert raw_path.read_bytes() == raw_bytes
    assert launcher.requests == []


@pytest.mark.asyncio
async def test_candidate_pool_setup_rejects_missing_manifest_artifact(
    tmp_path: Path,
) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    group_root = runner._group_root(config)
    request_key = runner._idempotency_key(
        config.experiment_group_id(),
        "ranker_component",
        "POOL",
        task.task_id,
        config.replication.candidate_pool_seed,
        None,
        config.budget_preset,
    )
    pool = group_root / "candidate-pools" / f"{request_key}.json"
    setup = group_root / "candidate-pool-setup" / f"{request_key}.json"
    pool_payload = canonical_json_bytes(
        {"candidate_pool_version": "formal-v1", "evidence_ids": [], "task_id": task.task_id}
    )
    pool.write_bytes(pool_payload)
    setup.write_bytes(
        canonical_json_bytes(
            {
                "schema_version": "candidate-pool-setup-v1",
                "group_id": config.experiment_group_id(),
                "task_id": task.task_id,
                "pool_key": request_key,
                "candidate_pool_sha256": sha256_bytes(pool_payload),
                "evidence_ids_sha256": sha256_bytes(canonical_json_bytes([])),
                "manifest_path": str(group_root / "artifacts" / "missing-manifest.json"),
                "manifest_sha256": "a" * 64,
                "usage": ResourceUsage.zero(cost_known=True).model_dump(mode="json"),
                "pricing_snapshot_ids": [config.pricing_snapshot.snapshot_id],
                "pricing_status": "estimated",
            }
        )
    )

    assert await runner._candidate_pool(config=config, task=task) is None


def test_runner_rejects_private_or_snapshot_root_symlink(tmp_path: Path) -> None:
    _, task = _config_and_task()
    private_target = tmp_path / "private-target"
    private_target.mkdir()
    private_link = tmp_path / "private-link"
    private_link.symlink_to(private_target, target_is_directory=True)
    with pytest.raises(RuntimeError, match="symlink|reparse|private root"):
        ExperimentRunner(
            launch_agent=SpyLauncher(),
            task_loader={task.task_id: task},
            experiment_root=tmp_path / "experiments",
            private_root=private_link,
            preflight=False,
        )


def test_runner_rejects_lexical_parent_traversal_before_root_resolution(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="parent traversal|lexical"):
        ExperimentRunner(
            launch_agent=SpyLauncher(),
            experiment_root=tmp_path,
            private_root=tmp_path / "private" / ".." / "outside",
            preflight=False,
        )


@pytest.mark.asyncio
async def test_request_staging_failure_is_recorded_without_evaluator(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, task = _config_and_task()
    evaluator_calls: list[object] = []

    async def evaluator(request: object, receipt: object) -> None:
        del request, receipt
        evaluator_calls.append(True)

    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        evaluator=evaluator,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )

    async def fail_stage(*args: object, **kwargs: object) -> object:
        del args, kwargs
        raise ValueError("staging path rejected")

    monkeypatch.setattr(runner, "_stage_request", fail_stage)
    result = await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        seed=config.replication.seed_values[0],
    )
    assert result.status == "failed"
    assert result.error_code == "AGENT_REQUEST_STAGE_FAILED"
    assert evaluator_calls == []


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


def test_runner_rejects_lexical_symlink_config_source(tmp_path: Path) -> None:
    config, task = _config_and_task()
    source = tmp_path / "sealed-formal.yaml"
    source.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    linked = tmp_path / "linked-formal.yaml"
    linked.symlink_to(source)
    with pytest.raises(RuntimeError, match="symlink|reparse|config source"):
        ExperimentRunner(
            launch_agent=SpyLauncher(),
            task_loader={task.task_id: task},
            experiment_root=tmp_path / "experiments",
            private_root=tmp_path / "private",
            config_source=linked,
            preflight=False,
        )


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
        seed_supported=True,
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


def test_agent_loader_requires_request_run_dir_to_match_guard_root(tmp_path: Path) -> None:
    from benchmarks.datasets.isolation import AgentRuntimeGuard

    config, task = _config_and_task()
    root = tmp_path / "group"
    runtime_root = root / "agent-inputs"
    snapshots = tmp_path / "snapshots"
    snapshot = snapshots / task.task_id
    for directory in (
        runtime_root,
        root / "requests",
        root / "config",
        root / "candidate-pools",
        root / "resume-checkpoints",
        snapshot,
    ):
        directory.mkdir(parents=True)
    private = tmp_path / "private"
    private.mkdir()
    source = tmp_path / "formal.yaml"
    source.write_text(yaml.safe_dump(config.model_dump(mode="json")), encoding="utf-8")
    config_path = stage_sealed_config(
        source,
        expected_sha256=sha256_bytes(source.read_bytes()),
        group_run_root=root,
    )
    staged = stage_authorized_runtime_task(
        task,
        config=config,
        budget_preset=config.budget_preset,
        agent_input_root=runtime_root,
        request_id="request-run-dir",
        forbidden_private_root=private,
    )
    files = {"documents.jsonl": b"", "index.json": b"{}\n"}
    snapshot_payload = {
        "task_id": task.task_id,
        "snapshot_id": task.snapshot_id,
        "corpus_version": task.corpus_version,
        "index_version": task.index_version,
        "documents_sha256": sha256_bytes(files["documents.jsonl"]),
        "index_sha256": sha256_bytes(files["index.json"]),
    }
    files["snapshot.json"] = json.dumps(snapshot_payload, sort_keys=True).encode()
    for name, payload in files.items():
        (snapshot / name).write_bytes(payload)
    (snapshot / "manifest.sha256").write_text(
        json.dumps({"file_sha256": {name: sha256_bytes(payload) for name, payload in files.items()}}),
        encoding="utf-8",
    )
    request = AgentVariantRunRequest(
        task_id=task.task_id,
        runtime_task_path=staged.runtime_task_path,
        runtime_task_sha256=staged.runtime_task_sha256,
        base_runtime_task_sha256=staged.base_runtime_task_sha256,
        snapshot_dir=str(snapshot.resolve()),
        run_dir=str((root / "other-run").resolve()),
        config_path=str(config_path.resolve()),
        config_sha256=sha256_bytes(source.read_bytes()),
        seed_supported=True,
        seed=config.replication.seed_values[0],
        protocol="end_to_end",
        variant="D",
        budget_preset=config.budget_preset,
    )
    guard = AgentRuntimeGuard(runtime_root=runtime_root, snapshot_root=snapshots, run_root=root)
    with pytest.raises(GoldAccessViolation, match="run_dir|run root"):
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
async def test_unseeded_runner_uses_sealed_repeat_ids_and_request_identity(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        seed_supported=False,
        preflight=False,
    )
    result = await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        repeat_id=1,
    )
    assert result.repeat_id == 1
    assert result.seed is None
    assert launcher.requests[0].seed_supported is False
    assert launcher.requests[0].seed is None
    assert launcher.requests[0].repeat_id == 1


@pytest.mark.asyncio
async def test_unseeded_completed_fast_path_requires_replication_binding(
    tmp_path: Path,
) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        seed_supported=False,
        preflight=False,
    )
    group_root = runner._group_root(config)
    key = runner._idempotency_key(
        config.experiment_group_id(),
        "end_to_end",
        "D",
        task.task_id,
        None,
        1,
        config.budget_preset,
    )
    existing = ExperimentTaskRun(
        task_id=task.task_id,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        planner_id="P2",
        ranker_id="R2",
        budget_preset=config.budget_preset,
        repeat_id=1,
        status="completed",
        manifest_path=str(group_root / "artifacts" / "missing.json"),
        artifact_ids=(),
        usage=ResourceUsage.zero(cost_known=True),
        pricing_snapshot_ids=(config.pricing_snapshot.snapshot_id,),
        pricing_status="estimated",
        cost_label="estimated_from_normalized_schedule",
        category=task.category,
        metrics={},
    )
    (group_root / "raw" / f"{key}.json").write_bytes(
        canonical_json_bytes(existing.model_dump(mode="json"))
    )

    with pytest.raises(RuntimeError, match="replication binding|sidecar"):
        await runner.run_one(
            config=config,
            protocol="end_to_end",
            task_id=task.task_id,
            variant=ExperimentVariant.D,
            budget_preset=config.budget_preset,
            repeat_id=1,
        )


@pytest.mark.asyncio
async def test_unseeded_resume_sidecar_uses_resume_request_for_fast_path(
    tmp_path: Path,
) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        seed_supported=False,
        preflight=False,
    )
    group_root = runner._group_root(config)
    key = runner._idempotency_key(
        config.experiment_group_id(),
        "end_to_end",
        "D",
        task.task_id,
        None,
        1,
        config.budget_preset,
    )
    usage = ResourceUsage.zero(cost_known=True)
    request_hash = hashlib.sha256(
        json.dumps(
            task.request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    config_hash = runner._core_config_sha256(
        config=config,
        task=task,
        planner_id="P2",
        ranker_id="R2",
        seed=None,
    )
    group = json.loads((group_root / "group.json").read_bytes())
    manifest = RunManifest.create(
        {
            "schema_version": "run-manifest-v1",
            "run_id": "resume-run-1",
            "thread_id": "resume-thread-1",
            "code_commit": group["code_commit"],
            "dependency_lock_sha256": "b" * 64,
            "request_sha256": request_hash,
            "config_sha256": config_hash,
            "workflow_id": "research-v1",
            "graph_version": "graph-v1",
            "planner_id": "P2",
            "provider_profiles": (),
            "model_ids": (),
            "prompt_versions": {},
            "parser_versions": {},
            "ranker_id": "R2",
            "ranker_weights_version": None,
            "budget": RunBudget.preset(config.budget_preset),
            "usage": usage,
            "usage_by_node": {},
            "pricing_status": "estimated",
            "pricing_snapshots": (config.pricing_snapshot,),
            "provider_calls": (),
            "node_executions": (),
            "parsed_artifacts": (),
            "evidence_hashes": (),
            "source_snapshot_ids": (),
            "artifact_ids": (),
            "run_event_count": 0,
            "run_events_sha256": "c" * 64,
            "seed": None,
            "seed_supported": False,
            "cache_hit_count": 0,
            "stop_reason": "SUFFICIENT",
            "is_partial": False,
            "failure_codes": (),
            "started_at": datetime(2026, 9, 6, tzinfo=UTC),
            "finished_at": datetime(2026, 9, 6, 0, 0, 1, tzinfo=UTC),
        }
    )
    manifest_path = group_root / "artifacts" / f"{key}.run-manifest.json"
    manifest_path.write_bytes(manifest.model_dump_json().encode())
    existing = ExperimentTaskRun(
        task_id=task.task_id,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        planner_id="P2",
        ranker_id="R2",
        budget_preset=config.budget_preset,
        repeat_id=1,
        status="completed",
        manifest_path=str(manifest_path),
        artifact_ids=(),
        usage=usage,
        pricing_snapshot_ids=(config.pricing_snapshot.snapshot_id,),
        pricing_status="estimated",
        cost_label="estimated_from_normalized_schedule",
        category=task.category,
        metrics={},
    )
    (group_root / "raw" / f"{key}.json").write_bytes(
        canonical_json_bytes(existing.model_dump(mode="json"))
    )
    resume_request = group_root / "requests" / f"{key}.resume.json"
    resume_request.write_bytes(b"resume-request\n")
    runner._bind_unseeded_receipt(
        group_root=group_root,
        config=config,
        task=task,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        budget=config.budget_preset,
        repeat_id=1,
        request_sha256=sha256_bytes(resume_request.read_bytes()),
        receipt_identity="d" * 64,
        manifest_provenance={
            "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
            "run_id": manifest.run_id,
            "thread_id": manifest.thread_id,
        },
    )

    result = await runner.run_one(
        config=config,
        protocol="end_to_end",
        task_id=task.task_id,
        variant=ExperimentVariant.D,
        budget_preset=config.budget_preset,
        repeat_id=1,
    )

    assert result == existing


@pytest.mark.asyncio
async def test_evaluator_failure_removes_unseeded_receipt_sidecar(
    tmp_path: Path,
) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        seed_supported=False,
        preflight=False,
    )
    group_root = runner._group_root(config)
    request, _, key = await runner._stage_request(
        config=config,
        group_root=group_root,
        task=task,
        task_id=task.task_id,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        budget=config.budget_preset,
        seed=None,
        repeat_id=1,
    )
    receipt = AgentRunReceipt(
        task_id=task.task_id,
        status="completed",
        run_result_path=str((group_root / "artifacts" / "result.json").absolute()),
        manifest_path=str((group_root / "artifacts" / "manifest.json").absolute()),
        run_result_sha256="a" * 64,
        manifest_sha256="b" * 64,
        artifact_ids=(),
    )
    runner._bind_unseeded_receipt(
        group_root=group_root,
        config=config,
        task=task,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        budget=config.budget_preset,
        repeat_id=1,
        request_sha256=sha256_bytes(canonical_json_bytes(request.model_dump(mode="json"))),
        receipt_identity=sha256_bytes(canonical_json_bytes(receipt.model_dump(mode="json"))),
        manifest_provenance={
            "manifest_sha256": "d" * 64,
            "run_id": "failed-run",
            "thread_id": "failed-thread",
        },
    )

    failed = runner._record(
        config=config,
        group_root=group_root,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        task=task,
        budget=config.budget_preset,
        seed=None,
        repeat_id=1,
        candidate_pool_hash=None,
        receipt=receipt,
        request=request,
        key=key,
        forced_error_code="EVALUATOR_VALIDATION_FAILED",
    )

    assert failed.status == "failed"
    assert list((group_root / "artifacts" / "replication-bindings").glob("*.json")) == []


def test_unseeded_receipt_binding_rejects_cross_repeat_reuse(tmp_path: Path) -> None:
    config, task = _config_and_task()
    runner = ExperimentRunner(
        launch_agent=SpyLauncher(),
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        seed_supported=False,
        preflight=False,
    )
    group_root = runner._group_root(config)
    provenance = {
        "manifest_sha256": "a" * 64,
        "run_id": "run-1",
        "thread_id": "thread-1",
    }
    runner._bind_unseeded_receipt(
        group_root=group_root,
        config=config,
        task=task,
        protocol="end_to_end",
        variant=ExperimentVariant.D,
        budget=config.budget_preset,
        repeat_id=1,
        request_sha256="b" * 64,
        receipt_identity="c" * 64,
        manifest_provenance=provenance,
    )
    with pytest.raises(ValueError, match="receipt.*repeat|replication"):
        runner._bind_unseeded_receipt(
            group_root=group_root,
            config=config,
            task=task,
            protocol="end_to_end",
            variant=ExperimentVariant.D,
            budget=config.budget_preset,
            repeat_id=2,
            request_sha256="d" * 64,
            receipt_identity="c" * 64,
            manifest_provenance=provenance,
        )


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


@pytest.mark.asyncio
async def test_promoted_candidate_pool_is_setup_once_and_reused(tmp_path: Path) -> None:
    config, task = _config_and_task()
    launcher = SpyLauncher()
    runner = ExperimentRunner(
        launch_agent=launcher,
        task_loader={task.task_id: task},
        experiment_root=tmp_path,
        private_root=tmp_path / "private",
        preflight=False,
    )
    group_root = runner._group_root(config)
    request_key = runner._idempotency_key(
        config.experiment_group_id(),
        "ranker_component",
        "POOL",
        task.task_id,
        config.replication.candidate_pool_seed,
        None,
        config.budget_preset,
    )
    pool = group_root / "candidate-pools" / f"{request_key}.json"
    setup = group_root / "candidate-pool-setup" / f"{request_key}.json"
    pool_payload = json.dumps(
        {"candidate_pool_version": "formal-v1", "evidence_ids": [], "task_id": task.task_id},
        sort_keys=True,
        separators=(",", ":"),
    ).encode() + b"\n"
    pool.write_bytes(pool_payload)
    request_sha256 = hashlib.sha256(
        json.dumps(
            task.request.model_dump(mode="json"),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()
    manifest = RunManifest.create(
        {
            "schema_version": "run-manifest-v1",
            "run_id": "pool-run-1",
            "thread_id": "pool-thread-1",
            "code_commit": "a" * 40,
            "dependency_lock_sha256": "b" * 64,
            "request_sha256": request_sha256,
            "config_sha256": runner._core_config_sha256(
                config=config,
                task=task,
                planner_id="P1",
                ranker_id="R1",
                seed=config.replication.candidate_pool_seed,
            ),
            "workflow_id": "research-v1",
            "graph_version": "graph-v1",
            "planner_id": "P1",
            "provider_profiles": (),
            "model_ids": (),
            "prompt_versions": {},
            "parser_versions": {},
            "ranker_id": "R1",
            "ranker_weights_version": None,
            "budget": RunBudget.preset(config.budget_preset),
            "usage": ResourceUsage.zero(cost_known=True),
            "usage_by_node": {},
            "pricing_status": "estimated",
            "pricing_snapshots": (config.pricing_snapshot,),
            "provider_calls": (),
            "node_executions": (),
            "parsed_artifacts": (),
            "evidence_hashes": (),
            "source_snapshot_ids": (),
            "artifact_ids": (),
            "run_event_count": 0,
            "run_events_sha256": "c" * 64,
            "seed": config.replication.candidate_pool_seed,
            "seed_supported": True,
            "cache_hit_count": 0,
            "stop_reason": "SUFFICIENT",
            "is_partial": False,
            "failure_codes": (),
            "started_at": datetime(2026, 9, 6, tzinfo=UTC),
            "finished_at": datetime(2026, 9, 6, 0, 0, 1, tzinfo=UTC),
        }
    )
    manifest_path = group_root / "artifacts" / "pool-manifest.json"
    manifest_path.write_bytes(manifest.model_dump_json().encode())
    manifest_sha256 = sha256_bytes(manifest_path.read_bytes())
    setup.write_text(
        json.dumps(
            {
                "schema_version": "candidate-pool-setup-v1",
                "group_id": config.experiment_group_id(),
                "task_id": task.task_id,
                "pool_key": request_key,
                "candidate_pool_sha256": sha256_bytes(pool_payload),
                "evidence_ids_sha256": sha256_bytes(canonical_json_bytes([])),
                "manifest_path": str(manifest_path),
                "manifest_sha256": manifest_sha256,
                "usage": {"input_tokens": 0, "cached_tokens": 0, "output_tokens": 0, "reasoning_tokens": 0, "total_tokens": 0, "search_calls": 0, "pages": 0, "retries": 0, "wall_seconds": 0.0, "cost_usd": "0"},
                "pricing_snapshot_ids": [config.pricing_snapshot.snapshot_id],
                "pricing_status": "estimated",
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    forged_manifest = manifest.model_copy(
        update={
            "ranker_id": "R2",
            "config_sha256": runner._core_config_sha256(
                config=config,
                task=task,
                planner_id="P1",
                ranker_id="R2",
                seed=config.replication.candidate_pool_seed,
            ),
        }
    )
    forged_bytes = forged_manifest.model_dump_json().encode()
    manifest_path.write_bytes(forged_bytes)
    forged_setup = json.loads(setup.read_bytes())
    forged_setup["manifest_sha256"] = sha256_bytes(forged_bytes)
    setup.write_bytes(canonical_json_bytes(forged_setup))
    assert await runner._candidate_pool(config=config, task=task) is None
    manifest_path.write_bytes(manifest.model_dump_json().encode())
    setup_payload = json.loads(setup.read_bytes())
    setup_payload["manifest_sha256"] = manifest_sha256
    setup.write_bytes(canonical_json_bytes(setup_payload))

    first = await runner._candidate_pool(config=config, task=task)
    second = await runner._candidate_pool(config=config, task=task)
    assert first == second == (pool, sha256_bytes(pool_payload))
    assert launcher.requests == []
