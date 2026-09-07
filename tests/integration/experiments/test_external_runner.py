from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
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
from benchmarks.processes.agent import AgentRunReceipt
from deepresearch.domain import ResourceUsage, RunBudget, RunResult
from deepresearch.runtime.manifest import RunManifest
from experiments.config import FormalExperimentConfig
from experiments.external_runner import ExternalExperimentRunner
from experiments.models import canonical_sha256
from experiments.runner import ExperimentRunner

pytest_plugins = ("tests.unit.benchmarks.test_external_adapters",)


class SpyLauncher:
    def __init__(self) -> None:
        self.requests: list[object] = []

    async def __call__(self, request: object) -> None:
        self.requests.append(request)


def _formal_config_with_external(
    external_config_path: Path,
    external_lock_path: Path,
    hashes: dict[str, str],
) -> FormalExperimentConfig:
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
    return FormalExperimentConfig.model_validate(payload)


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
async def test_external_preflight_rejects_partial_portfolio_map(
    external_fixture, monkeypatch: pytest.MonkeyPatch
):
    repo, external, external_config_path, external_lock_path, raw_root, snapshot_root = external_fixture
    adapter = FramesAdapter(
        lock_file=external_lock_path,
        raw_root=raw_root,
        snapshot_root=snapshot_root,
        external_config=external,
    )
    partial_hashes = {
        selection.runtime_task.task_id: canonical_sha256(
            selection.runtime_task.model_dump(mode="json")
        )
        for selection in adapter.select(
            provider_profile_id="formal-local-vllm", budget_preset="medium"
        )
    }
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
            "external_runtime_task_hashes": dict(sorted(partial_hashes.items())),
        }
    )
    config = FormalExperimentConfig.model_validate(payload)
    # Isolate the portfolio-map assertion from the unavailable internal seal
    # files in this synthetic repository. A real preflight still runs the
    # complete validator before launching any agent.
    monkeypatch.setattr("experiments.external_runner.preflight_config", lambda *args, **kwargs: None)
    monkeypatch.setattr("experiments.external_runner.code_tree_sha256", lambda _: config.code_tree_sha256)
    monkeypatch.setattr(ExternalExperimentRunner, "_code_commit", lambda self: "a" * 40)
    launcher = SpyLauncher()
    runner = ExternalExperimentRunner(
        launch_agent=launcher,
        repo_root=repo,
        experiment_root=repo / "experiments",
        preflight=True,
    )
    with pytest.raises(ValueError, match="40|canonical|Portfolio"):
        await runner.run(
            config=config,
            external_config_path=external_config_path,
            external_lock_path=external_lock_path,
            benchmarks=("frames",),
        )
    assert launcher.requests == []


@pytest.mark.asyncio
async def test_external_runner_rechecks_raw_after_selection_before_launch(
    external_fixture, monkeypatch: pytest.MonkeyPatch
):
    repo, external, external_config_path, external_lock_path, raw_root, snapshot_root = external_fixture
    adapter = FramesAdapter(
        lock_file=external_lock_path,
        raw_root=raw_root,
        snapshot_root=snapshot_root,
        external_config=external,
    )
    hashes = {
        selection.runtime_task.task_id: canonical_sha256(
            selection.runtime_task.model_dump(mode="json")
        )
        for selection in adapter.select(
            provider_profile_id="formal-local-vllm", budget_preset="medium"
        )
    }
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
    original_select = FramesAdapter.select

    def select_then_mutate(self: FramesAdapter, **kwargs: object):
        result = original_select(self, **kwargs)  # type: ignore[arg-type]
        raw_path = self.raw_root / self._entry.raw_relative_path  # pyright: ignore[reportPrivateUsage]
        raw_path.write_bytes(b"[]\n")
        return result

    monkeypatch.setattr(FramesAdapter, "select", select_then_mutate)
    launcher = SpyLauncher()
    runner = ExternalExperimentRunner(
        launch_agent=launcher,
        repo_root=repo,
        experiment_root=repo / "experiments",
        preflight=False,
    )
    with pytest.raises(Exception, match="hash mismatch|changed") as error:
        await runner.run(
            config=config,
            external_config_path=external_config_path,
            external_lock_path=external_lock_path,
            benchmarks=("frames",),
        )
    assert getattr(error.value, "code", None) == "INVALID_SNAPSHOT"
    assert launcher.requests == []


@pytest.mark.asyncio
async def test_external_runner_records_all_unseeded_repeats(external_fixture):
    repo, external, external_config_path, external_lock_path, raw_root, snapshot_root = external_fixture
    adapter = FramesAdapter(
        lock_file=external_lock_path,
        raw_root=raw_root,
        snapshot_root=snapshot_root,
        external_config=external,
    )
    hashes = {
        selection.runtime_task.task_id: canonical_sha256(
            selection.runtime_task.model_dump(mode="json")
        )
        for selection in adapter.select(
            provider_profile_id="formal-local-vllm", budget_preset="medium"
        )
    }
    config = _formal_config_with_external(
        external_config_path, external_lock_path, hashes
    )
    launcher = SpyLauncher()
    runner = ExternalExperimentRunner(
        launch_agent=launcher,
        repo_root=repo,
        experiment_root=repo / "experiments",
        preflight=False,
        seed_supported=False,
    )
    result = await runner.run(
        config=config,
        external_config_path=external_config_path,
        external_lock_path=external_lock_path,
        benchmarks=("frames",),
    )
    expected_repeats = range(1, config.replication.unseeded_repeat_count + 1)
    assert len(result.runs) == 20 * config.replication.unseeded_repeat_count
    assert {run.repeat_id for run in result.runs} == set(expected_repeats)
    assert all(run.seed is None and run.status == "failed" for run in result.runs)
    assert len(launcher.requests) == len(result.runs)
    assert {
        request.repeat_id for request in launcher.requests  # type: ignore[union-attr]
    } == set(expected_repeats)
    assert all(request.seed is None for request in launcher.requests)  # type: ignore[union-attr]


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
    assert len(launcher.requests) == 20 * len(config.replication.seed_values)
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


@pytest.mark.asyncio
async def test_external_runner_validates_receipts_calls_evaluator_and_replicates_seeds(
    external_fixture,
):
    repo, external, external_config_path, external_lock_path, raw_root, snapshot_root = external_fixture
    hashes: dict[str, str] = {}
    adapter = FramesAdapter(
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
    launched: list[object] = []
    plans: list[object] = []

    async def launch(request: object) -> AgentRunReceipt:
        launched.append(request)
        typed = request
        task = RuntimeTask.model_validate_json(
            Path(typed.runtime_task_path).read_bytes(), strict=True  # type: ignore[attr-defined]
        )
        run_root = Path(typed.run_dir)  # type: ignore[attr-defined]
        key = hashlib.sha256(
            f"{task.task_id}:{typed.seed}".encode()  # type: ignore[attr-defined]
        ).hexdigest()
        artifact_root = run_root / "external" / "frames" / "receipts"
        artifact_root.mkdir(parents=True, exist_ok=True)
        started = datetime(2026, 9, 7, tzinfo=UTC)
        usage = ResourceUsage.zero(cost_known=True)
        manifest = RunManifest.create(
            {
                "schema_version": "run-manifest-v1",
                "run_id": f"run-{key[:16]}",
                "thread_id": f"thread-{key[:16]}",
                "code_commit": json.loads((run_root / "group.json").read_bytes())["code_commit"],
                "dependency_lock_sha256": "c" * 64,
                "request_sha256": hashlib.sha256(
                    json.dumps(
                        task.request.model_dump(mode="json"),
                        ensure_ascii=False,
                        sort_keys=True,
                        separators=(",", ":"),
                    ).encode("utf-8")
                ).hexdigest(),
                "config_sha256": ExperimentRunner._core_config_sha256(
                    config=config,
                    task=task,
                    planner_id="P1",
                    ranker_id="R2",
                    seed=typed.seed,  # type: ignore[attr-defined]
                ),
                "workflow_id": "research-v1",
                "graph_version": "external-graph-v1",
                "planner_id": "P1",
                "provider_profiles": (),
                "model_ids": (),
                "prompt_versions": {},
                "parser_versions": {},
                "ranker_id": "R2",
                "ranker_weights_version": None,
                "budget": RunBudget.preset(task.request.budget_preset),
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
                "run_events_sha256": "d" * 64,
                "seed": typed.seed,  # type: ignore[attr-defined]
                "seed_supported": True,
                "cache_hit_count": 0,
                "stop_reason": "SUFFICIENT",
                "is_partial": False,
                "failure_codes": (),
                "started_at": started,
                "finished_at": started,
            }
        )
        result = RunResult(
            run_id=manifest.run_id,
            thread_id=manifest.thread_id,
            status="completed",
            stop_reason="SUFFICIENT",
            is_partial=False,
            final_usage=usage,
        )
        manifest_path = artifact_root / f"{key}.manifest.json"
        result_path = artifact_root / f"{key}.result.json"
        manifest_bytes = manifest.model_dump_json().encode("utf-8")
        result_bytes = result.model_dump_json().encode("utf-8")
        manifest_path.write_bytes(manifest_bytes)
        result_path.write_bytes(result_bytes)
        return AgentRunReceipt(
            task_id=task.task_id,
            status="completed",
            run_result_path=str(result_path.resolve()),
            manifest_path=str(manifest_path.resolve()),
            run_result_sha256=sha256_bytes(result_bytes),
            manifest_sha256=sha256_bytes(manifest_bytes),
            artifact_ids=(),
        )

    async def evaluate(plan: object) -> None:
        plans.append(plan)
        return {}

    runner = ExternalExperimentRunner(
        launch_agent=launch,
        evaluator=evaluate,
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
    assert len(launched) == 20 * len(config.replication.seed_values)
    assert len(plans) == len(launched)
    assert all(run.status == "completed" for run in result.runs)
    assert {run.seed for run in result.runs} == set(config.replication.seed_values)
    assert all("evaluation_plan" not in json.dumps(request, default=str) for request in launched)
    metrics = repo / "experiments" / result.portfolio_group_id / "external" / "metrics.json"
    metrics_payload = metrics.read_text(encoding="utf-8")
    assert "private_scoring_reference" not in metrics_payload
