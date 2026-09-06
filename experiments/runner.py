"""Evaluator-side orchestration for reproducible formal component experiments.

This module deliberately contains no Core agent implementation import.  The
only code that can construct a model/provider graph is the child entrypoint in
``benchmarks.processes.agent`` after it has verified its request and roots.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import json
import os
import stat
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import JsonValue

from benchmarks.datasets.isolation import GoldIsolationGuard
from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.evaluators.metrics import MetricValue
from benchmarks.processes.agent import (
    AgentCandidatePoolReceipt,
    AgentCandidatePoolRequest,
    AgentRunReceipt,
    AgentRunRequest,
    AgentVariantRunRequest,
)
from benchmarks.processes.evaluator import (
    StagedRuntimeTask,
    stage_authorized_runtime_task,
    stage_sealed_config,
)
from deepresearch.domain import ResourceUsage, RunBudget, RunConfig, RunResult
from deepresearch.runtime import CheckpointRef
from deepresearch.runtime.manifest import (
    RunManifest,
    _canonical_bytes,  # pyright: ignore[reportPrivateUsage]
)
from experiments.config import FormalExperimentConfig, preflight_config
from experiments.models import (
    COMPONENT_IDS,
    ExperimentRunResult,
    ExperimentTaskRun,
    ExperimentVariant,
    RankerComponentVariant,
    task_run_from_manifest,
)

Protocol = Literal["ranker_component", "planner_policy", "end_to_end", "reference"]
BudgetPreset = Literal["low", "medium", "high"]


@dataclass(frozen=True)
class _ValidatedAgentRun:
    receipt: AgentRunReceipt
    manifest: RunManifest
    result: RunResult
    manifest_path: Path
    result_path: Path


async def _run_subprocess(
    command: Sequence[str], *, cwd: Path, env: Mapping[str, str]
) -> subprocess.CompletedProcess[str]:
    return await asyncio.to_thread(
        subprocess.run,
        list(command),
        cwd=cwd,
        env=dict(env),
        capture_output=True,
        text=True,
        check=False,
    )


def _canonical(value: object) -> bytes:
    return canonical_json_bytes(value)


def _write_immutable(path: Path, payload: bytes) -> Path:
    path = Path(path)
    if path.is_symlink():
        raise FileExistsError(path)
    _assert_lexically_safe(path.parent, label="immutable artifact destination")
    if path.exists():
        if path.is_file() and path.read_bytes() == payload:
            return path
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    # Keep the temporary sibling short: Windows formal-run paths can already
    # contain a 64-character idempotency key and must remain below MAX_PATH.
    staging = path.with_name(f".{path.name}.staging")
    try:
        with staging.open("xb") as stream:
            stream.write(payload)
            stream.flush()
        # ``os.replace`` is intentionally not used: a completed formal record
        # is never overwritten.  The exclusive create above protects the
        # normal evaluator path; a race is surfaced as a failed publication.
        staging.rename(path)
    finally:
        staging.unlink(missing_ok=True)
    return path


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0) & reparse_flag
    )


def _assert_lexically_safe(path: Path, *, label: str) -> Path:
    absolute = Path(path).absolute()
    current = absolute
    while current != Path(current.anchor):
        if _is_link_or_reparse(current):
            raise RuntimeError(f"{label} contains a symlink or reparse point")
        current = current.parent
    if _is_link_or_reparse(current):
        raise RuntimeError(f"{label} contains a symlink or reparse point")
    return absolute


def _git_commit(root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        raise RuntimeError("git commit identity is unavailable") from None
    if len(commit) != 40 or any(character not in "0123456789abcdef" for character in commit):
        raise RuntimeError("git commit identity is unavailable")
    return commit


class ExperimentRunner:
    """Coordinate sealed evaluator work and isolated agent requests.

    ``launch_agent`` is intentionally dependency-injected.  Production uses a
    subprocess launcher; tests use a replay spy.  The evaluator never receives
    private gold in the request object and never instantiates a Core runner.
    """

    def __init__(
        self,
        launch_agent: Callable[[AgentRunRequest], object | Awaitable[object]] | None = None,
        evaluator: Callable[..., object | Awaitable[object]] | None = None,
        *,
        task_loader: Mapping[str, RuntimeTask]
        | Callable[[str], RuntimeTask | Awaitable[RuntimeTask]]
        | None = None,
        experiment_root: Path = Path("experiments"),
        repo_root: Path | None = None,
        private_root: Path | None = None,
        snapshot_root: Path | None = None,
        config_source: Path | None = None,
        seed_supported: bool = True,
        oracle_provider: object | None = None,
        oracle_records_loader: Mapping[str, Mapping[str, object]] | None = None,
        preflight: bool = True,
    ) -> None:
        self._launch_agent = launch_agent or self._default_launch_agent
        self._evaluator = evaluator
        self._task_loader = task_loader
        self.experiment_root = Path(experiment_root)
        self.repo_root = (repo_root or Path.cwd()).resolve()
        private_root_path = _assert_lexically_safe(
            Path(private_root)
            if private_root is not None
            else self.repo_root / "benchmarks" / "private",
            label="private root",
        )
        snapshot_root_path = _assert_lexically_safe(
            Path(snapshot_root)
            if snapshot_root is not None
            else self.repo_root / "benchmarks" / "snapshots",
            label="snapshot root",
        )
        self.private_root = (
            private_root_path.resolve()
        )
        self.snapshot_root = (
            snapshot_root_path.resolve()
        )
        self.config_source = (
            _assert_lexically_safe(Path(config_source), label="sealed config source")
            if config_source
            else None
        )
        self.seed_supported = seed_supported
        self.oracle_provider = oracle_provider
        self.oracle_records_loader = oracle_records_loader or {}
        self.preflight = preflight

    async def _default_launch_agent(self, request: AgentRunRequest) -> object:
        """Launch the typed child entrypoint with an explicit environment.

        Real deployments may inject a hardened launcher.  This default keeps
        the process boundary usable offline and does not pass private roots.
        """
        run_root = Path(request.run_dir).resolve()
        request_payload = _canonical(request.model_dump(mode="json"))
        request_path: Path | None = None
        for candidate in sorted((run_root / "requests").glob("*.json")):
            if candidate.is_file() and candidate.read_bytes() == request_payload:
                request_path = candidate
                break
        if request_path is None:
            raise RuntimeError("staged agent request is unavailable")
        receipt_path = run_root / "receipts" / f"{request_path.stem}.json"
        environment = {
            key: os.environ[key]
            for key in ("PATH", "SystemRoot", "TEMP", "TMP", "PYTHONIOENCODING")
            if key in os.environ
        }
        environment["PYTHONPATH"] = os.pathsep.join(
            [str(self.repo_root), str(self.repo_root / "src")]
        )
        environment["DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT"] = str(
            (run_root / "agent-inputs").resolve()
        )
        environment["DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT"] = str(self.snapshot_root)
        environment["DEEPRESEARCH_BENCHMARK_RUN_ROOT"] = str(run_root)
        for name, value in os.environ.items():
            if name.startswith("DEEPRESEARCH_PROVIDER_"):
                environment[name] = value
        completed = await _run_subprocess(
            [
                os.environ.get("PYTHON", "python"),
                "-m",
                "benchmarks.processes.agent",
                "--request",
                str(request_path),
                "--receipt",
                str(receipt_path),
            ],
            cwd=self.repo_root,
            env=environment,
        )
        if completed.returncode != 0 or not receipt_path.is_file():
            raise RuntimeError("agent process failed")
        return AgentRunReceipt.model_validate_json(receipt_path.read_bytes(), strict=True)

    async def _call_launcher(self, request: AgentRunRequest) -> object:
        result = self._launch_agent(request)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    async def _call_evaluator(
        self, request: AgentRunRequest, receipt: AgentRunReceipt
    ) -> object | None:
        if self._evaluator is None:
            return None
        result = self._evaluator(request, receipt)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    def _group_root(self, config: FormalExperimentConfig) -> Path:
        experiment_base = _assert_lexically_safe(
            self.experiment_root, label="experiment root"
        )
        private_root = self.private_root
        dataset_private_root = private_root / config.dataset_id
        if not (private_root / "private_manifest.json").is_file() and (
            dataset_private_root / "private_manifest.json"
        ).is_file():
            private_root = dataset_private_root
        if self.preflight:
            if not (private_root / "private_manifest.json").is_file():
                raise RuntimeError("sealed private manifest is required before an experiment")
            self._assert_clean_result_tree(config.experiment_group_id())
            preflight_config(config, repo_root=self.repo_root, private_root=private_root)
        group = config.experiment_group_id()
        root = (experiment_base / group).absolute()
        if not root.is_relative_to(experiment_base):
            raise RuntimeError("experiment group is outside experiment root")
        if _is_link_or_reparse(root):
            raise RuntimeError("experiment group root cannot be a symlink")
        experiment_base.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        config_path = root / "config" / "formal.yaml"
        if self.config_source is not None:
            source = _assert_lexically_safe(self.config_source, label="sealed config source")
            if not source.is_file() or source.is_symlink():
                raise RuntimeError("sealed config source is not a regular file")
            source_bytes = source.read_bytes()
            try:
                staged_config = FormalExperimentConfig.model_validate(yaml.safe_load(source_bytes))
            except (TypeError, ValueError, yaml.YAMLError) as error:
                raise RuntimeError("sealed config source is invalid") from error
            if staged_config != config:
                raise RuntimeError("sealed config source disagrees with requested config")
            config_hash = sha256_bytes(source_bytes)
            stage_sealed_config(
                source,
                expected_sha256=config_hash,
                group_run_root=root,
            )
        else:
            config_bytes = _canonical(config.model_dump(mode="json"))
            _write_immutable(config_path, config_bytes)
            config_hash = sha256_bytes(config_bytes)
        task_categories: dict[str, str] = {}
        loader_for_categories = self._task_loader
        if isinstance(loader_for_categories, Mapping):
            typed_loader = cast(Mapping[str, RuntimeTask], loader_for_categories)
            task_categories = {
                task_id: task.category.value for task_id, task in typed_loader.items()
            }
        payload: dict[str, object] = {
            "schema_version": "formal-experiment-group-v1",
            "group_id": group,
            "config_sha256": config_hash,
            "code_commit": _git_commit(self.repo_root),
            "dataset_version": config.dataset_version,
            "private_manifest_sha256": config.private_manifest_sha256,
            "evaluator_version": config.evaluator_version,
            "protocols": ["ranker_component", "planner_policy", "end_to_end", "reference"],
            "protocol_task_ids": {
                "ranker_component": list(config.main_test_task_ids),
                "planner_policy": list(config.main_test_task_ids),
                "end_to_end": list(config.main_test_task_ids),
                "reference": list(config.p0_task_ids),
            },
            "oracle_task_ids": list(config.oracle_task_ids),
            "expected_variants": {
                "ranker_component": ["R0", "R1", "R2"],
                "planner_policy": ["A", "B", "C", "D"],
                "end_to_end": ["A", "B", "C", "D"],
                "reference": ["P0"],
            },
            "budgets": list(config.budget_sensitivity_presets),
            "required_budgets": {
                "ranker_component": [config.budget_preset],
                "planner_policy": [config.budget_preset],
                "end_to_end": [config.budget_preset],
                "reference": [config.budget_preset],
            },
            "cost_subset_task_ids": list(config.cost_subset_task_ids),
            "budget_preset": config.budget_preset,
            "candidate_pool_seed": config.replication.candidate_pool_seed,
            "pricing_snapshot_id": config.pricing_snapshot.snapshot_id,
            "replication": {
                "mode": "seeds" if self.seed_supported else "independent_repeats",
                "seed_supported": self.seed_supported,
                "seed_values": list(config.replication.seed_values)
                if self.seed_supported
                else [],
                "repeat_ids": []
                if self.seed_supported
                else list(range(1, config.replication.unseeded_repeat_count + 1)),
            },
            "task_categories": task_categories,
        }
        _write_immutable(root / "group.json", _canonical(payload))
        for directory in (
            "agent-inputs",
            "requests",
            "candidate-pools",
            "candidate-pool-setup",
            "resume-checkpoints",
            "raw",
            "artifacts",
        ):
            child = root / directory
            _assert_lexically_safe(child, label=f"experiment {directory} subtree")
            if _is_link_or_reparse(child):
                raise RuntimeError(f"experiment {directory} subtree cannot be a symlink")
            child.mkdir(parents=True, exist_ok=True)
        return root

    def _assert_clean_result_tree(self, group_id: str) -> None:
        try:
            output = subprocess.run(
                ["git", "-C", str(self.repo_root), "status", "--porcelain", "--untracked-files=all"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError("unable to verify clean result-affecting tree") from error
        allowed_prefixes = {
            f"experiments/{group_id}/",
            f"experiments\\{group_id}\\",
        }
        for line in output:
            relative = line[3:].replace("\\", "/") if len(line) >= 3 else line
            if not any(relative.startswith(prefix.replace("\\", "/")) for prefix in allowed_prefixes):
                raise RuntimeError("result-affecting worktree is not clean")

    def _task_sync(self, task_id: str) -> RuntimeTask:
        loader = self._task_loader
        if isinstance(loader, Mapping):
            typed_loader = cast(Mapping[str, RuntimeTask], loader)
            task = typed_loader.get(task_id)
            if task is None:
                raise ValueError(f"unknown task: {task_id}")
            return task
        if loader is not None:
            result = loader(task_id)
            if inspect.isawaitable(result):
                raise RuntimeError("async task loader requires async task resolution")
            return result
        roots = (
            self.private_root / "runtime" / "test",
            self.private_root / "frozen_ai_cs_60" / "runtime" / "test",
            self.repo_root / "benchmarks" / "private" / "frozen_ai_cs_60" / "runtime" / "test",
            self.repo_root / "benchmarks" / "datasets" / "frozen_ai_cs_60" / "runtime" / "dev",
        )
        for root in roots:
            for path in sorted(root.glob("*.jsonl")):
                for line in path.read_bytes().splitlines():
                    task = RuntimeTask.model_validate_json(line, strict=True)
                    if task.task_id == task_id:
                        return task
        raise ValueError(f"task input is unavailable: {task_id}")

    async def _task(self, task_id: str) -> RuntimeTask:
        loader = self._task_loader
        if loader is not None and not isinstance(loader, Mapping):
            result = loader(task_id)
            if inspect.isawaitable(result):
                return await result
            return result
        return self._task_sync(task_id)

    def _snapshot_dir(self, task: RuntimeTask) -> Path:
        direct = self.snapshot_root / task.task_id
        if direct.is_dir():
            return direct
        candidate = self.snapshot_root / "frozen_ai_cs_60" / task.task_id
        return candidate

    def _config_path_and_hash(self, group_root: Path) -> tuple[Path, str]:
        path = group_root / "config" / "formal.yaml"
        payload = path.read_bytes()
        return path, sha256_bytes(payload)

    @staticmethod
    def _is_sha256(value: object) -> bool:
        return (
            isinstance(value, str)
            and len(value) == 64
            and value == value.lower()
            and value != "0" * 64
            and all(character in "0123456789abcdef" for character in value)
        )

    @staticmethod
    def _validate_setup_manifest(
        manifest: RunManifest,
        *,
        config: FormalExperimentConfig,
        task: RuntimeTask,
        seed: int,
        planner_id: str,
    ) -> None:
        request_hash = sha256_bytes(_canonical_bytes(task.request.model_dump(mode="json")))
        if manifest.request_sha256 != request_hash:
            raise ValueError("candidate pool manifest request identity mismatch")
        if manifest.workflow_id != "research-v1" or manifest.planner_id != planner_id:
            raise ValueError("candidate pool manifest workflow identity mismatch")
        if manifest.seed != seed or not manifest.seed_supported:
            raise ValueError("candidate pool manifest seed identity mismatch")
        if manifest.config_sha256 != ExperimentRunner._core_config_sha256(
            config=config,
            task=task,
            planner_id=manifest.planner_id,
            ranker_id=manifest.ranker_id,
            seed=seed,
        ):
            raise ValueError("candidate pool manifest config identity mismatch")
        if manifest.budget.model_dump(mode="json", exclude={"used_by_node"}) != RunBudget.preset(
            task.request.budget_preset
        ).model_dump(mode="json", exclude={"used_by_node"}):
            raise ValueError("candidate pool manifest budget mismatch")
        if manifest.pricing_status != "estimated" or manifest.pricing_snapshots != (
            config.pricing_snapshot,
        ):
            raise ValueError("candidate pool manifest pricing mismatch")
        if manifest.usage.cost_usd is None:
            raise ValueError("candidate pool manifest usage is not cost-verified")

    @staticmethod
    def _core_config_sha256(
        *,
        config: FormalExperimentConfig,
        task: RuntimeTask,
        planner_id: str,
        ranker_id: str,
        seed: int | None,
    ) -> str:
        run_config = RunConfig(
            request=task.request,
            workflow_id="research-v1",
            planner_id=cast(Any, planner_id),
            ranker_id=cast(Any, ranker_id),
            budget=RunBudget.preset(cast(Any, task.request.budget_preset)),
            prompt_versions={
                "planner": config.prompt_version,
                "writer": config.writer_prompt_version,
                "judge": config.judge_prompt_version,
            },
            ranker_weights_version=config.ranker_weights_version,
            seed=seed,
        )
        payload = json.dumps(
            run_config.model_dump(mode="json"),
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    def _seed_or_repeat(
        self,
        config: FormalExperimentConfig,
        *,
        seed: int | None,
        repeat_id: int | None,
    ) -> tuple[int | None, int | None]:
        if self.seed_supported:
            if seed is None or repeat_id is not None or seed not in config.replication.seed_values:
                raise ValueError("seed is not part of the sealed replication policy")
            return seed, None
        if seed is not None or repeat_id is None:
            raise ValueError("unseeded provider requires a sealed repeat_id")
        if repeat_id < 1 or repeat_id > config.replication.unseeded_repeat_count:
            raise ValueError("repeat_id is not part of the sealed replication policy")
        return None, repeat_id

    @staticmethod
    def idempotency_key(
        group_id: str,
        protocol: str,
        variant: str,
        task_id: str,
        seed: int | None,
        repeat_id: int | None,
        budget: str,
    ) -> str:
        token = {
            "group": group_id,
            "protocol": protocol,
            "variant": variant,
            "task_id": task_id,
            "seed": seed,
            "repeat_id": repeat_id,
            "budget_preset": budget,
        }
        return hashlib.sha256(_canonical(token)).hexdigest()

    @staticmethod
    def _idempotency_key(
        group_id: str,
        protocol: str,
        variant: str,
        task_id: str,
        seed: int | None,
        repeat_id: int | None,
        budget: str,
    ) -> str:
        """Backward-compatible private alias for existing callers/tests."""
        return ExperimentRunner.idempotency_key(
            group_id, protocol, variant, task_id, seed, repeat_id, budget
        )

    def _raw_path(self, group_root: Path, key: str) -> Path:
        return group_root / "raw" / f"{key}.json"

    def _load_existing(self, path: Path) -> ExperimentTaskRun | None:
        if not path.is_file():
            return None
        return ExperimentTaskRun.model_validate_json(path.read_bytes(), strict=True)

    def _parse_receipt(self, receipt: object) -> AgentRunReceipt | None:
        if isinstance(receipt, AgentRunReceipt):
            return receipt
        if isinstance(receipt, Mapping):
            try:
                return AgentRunReceipt.model_validate(receipt, strict=True)
            except (TypeError, ValueError):
                return None
        return None

    @staticmethod
    def _parse_evaluator_metrics(value: object) -> dict[str, MetricValue]:
        """Parse evaluator output without permitting arbitrary public payloads."""
        if value is None:
            return {}
        if not isinstance(value, Mapping):
            raise TypeError("evaluator output must be a metric mapping")
        parsed: dict[str, MetricValue] = {}
        items = cast(Mapping[object, object], value)
        for name, metric in items.items():
            if not isinstance(name, str) or not name.strip():
                raise ValueError("evaluator metric names must be non-empty strings")
            if isinstance(metric, MetricValue):
                typed = metric
            else:
                typed = MetricValue.model_validate(metric, strict=True)
            if typed.name != name:
                raise ValueError("evaluator metric name does not match its key")
            parsed[name] = typed
        return parsed

    def _bind_unseeded_receipt(
        self,
        *,
        group_root: Path,
        config: FormalExperimentConfig,
        task: RuntimeTask,
        protocol: Protocol,
        variant: ExperimentVariant | RankerComponentVariant,
        budget: BudgetPreset,
        repeat_id: int,
        request_sha256: str,
        receipt_identity: str,
        manifest_provenance: Mapping[str, object],
    ) -> None:
        """Persist the missing repeat identity beside the Core receipt.

        Core's public receipt/manifest intentionally has no ``repeat_id``.
        This evaluator-owned, content-addressed sidecar binds that receipt to
        the sealed request identity and prevents one manifest from being
        accepted for two unseeded repeats.
        """
        if self.seed_supported:
            raise ValueError("unseeded receipt binding is not valid for seeded runs")
        if repeat_id < 1 or repeat_id > config.replication.unseeded_repeat_count:
            raise ValueError("repeat_id is outside the sealed replication policy")
        if not self._is_sha256(request_sha256) or not self._is_sha256(receipt_identity):
            raise ValueError("replication binding hashes are invalid")
        required_provenance = {"manifest_sha256", "run_id", "thread_id"}
        if set(manifest_provenance) != required_provenance:
            raise ValueError("manifest provenance is incomplete")
        manifest_sha256 = manifest_provenance["manifest_sha256"]
        if not self._is_sha256(manifest_sha256):
            raise ValueError("manifest provenance hash is invalid")
        run_id = manifest_provenance["run_id"]
        thread_id = manifest_provenance["thread_id"]
        if (
            not isinstance(run_id, str)
            or not run_id
            or not isinstance(thread_id, str)
            or not thread_id
        ):
            raise ValueError("manifest provenance identity is invalid")
        payload: dict[str, object] = {
            "schema_version": "unseeded-replication-binding-v1",
            "group_id": config.experiment_group_id(),
            "task_id": task.task_id,
            "protocol": protocol,
            "variant": variant.value,
            "budget_preset": budget,
            "repeat_id": repeat_id,
            "request_sha256": request_sha256,
            "receipt_identity": receipt_identity,
            "manifest_provenance": {
                "manifest_sha256": manifest_sha256,
                "run_id": run_id,
                "thread_id": thread_id,
            },
        }
        binding_root = group_root / "artifacts" / "replication-bindings"
        _assert_lexically_safe(binding_root, label="replication binding directory")
        for path in sorted(binding_root.glob("*.json")) if binding_root.is_dir() else ():
            if path.is_symlink() or not path.is_file():
                raise ValueError("replication binding artifact is unsafe")
            existing = json.loads(path.read_bytes())
            if not isinstance(existing, dict):
                raise TypeError("replication binding artifact is invalid")
            typed_existing = cast(dict[str, object], existing)
            if set(typed_existing) != set(payload):
                raise ValueError("replication binding schema is invalid")
            existing_request_sha256 = typed_existing.get("request_sha256")
            existing_receipt_identity = typed_existing.get("receipt_identity")
            existing_repeat_id = typed_existing.get("repeat_id")
            existing_provenance = typed_existing.get("manifest_provenance")
            if (
                not isinstance(existing_request_sha256, str)
                or not self._is_sha256(existing_request_sha256)
                or path.stem != existing_request_sha256
                or not isinstance(existing_receipt_identity, str)
                or not self._is_sha256(existing_receipt_identity)
                or type(existing_repeat_id) is not int
                or existing_repeat_id < 1
                or existing_repeat_id > config.replication.unseeded_repeat_count
                or not isinstance(existing_provenance, dict)
            ):
                raise ValueError("replication binding identity is invalid")
            typed_provenance = cast(dict[object, object], existing_provenance)
            if set(typed_provenance) != {"manifest_sha256", "run_id", "thread_id"}:
                raise ValueError("replication binding provenance is invalid")
            if (
                not self._is_sha256(typed_provenance["manifest_sha256"])
                or not isinstance(typed_provenance["run_id"], str)
                or not typed_provenance["run_id"]
                or not isinstance(typed_provenance["thread_id"], str)
                or not typed_provenance["thread_id"]
            ):
                raise ValueError("replication binding provenance is invalid")
            if (
                typed_existing.get("group_id") != config.experiment_group_id()
                or typed_existing.get("task_id") != task.task_id
                or typed_existing.get("protocol") != protocol
                or typed_existing.get("variant") != variant.value
                or typed_existing.get("budget_preset") != budget
            ):
                raise ValueError("replication binding experiment identity mismatch")
            if (
                existing_receipt_identity == receipt_identity
                and existing_repeat_id != repeat_id
            ):
                raise ValueError("receipt identity is bound to another repeat")
        binding_path = binding_root / f"{request_sha256}.json"
        _write_immutable(binding_path, _canonical(payload))

    def _validate_agent_receipt(
        self,
        *,
        request: AgentRunRequest,
        config: FormalExperimentConfig,
        group_root: Path,
        protocol: Protocol,
        variant: ExperimentVariant | RankerComponentVariant,
        task: RuntimeTask,
        budget: BudgetPreset,
        seed: int | None,
        repeat_id: int | None,
        candidate_pool_hash: str | None,
        receipt: object,
    ) -> _ValidatedAgentRun:
        """Validate every public child artifact before evaluator callbacks.

        This is intentionally separate from ``_record``.  A callback can receive
        only the typed receipt returned by this method, never a raw launcher
        object or a path that has not passed Core manifest/result validation.
        """
        parsed = self._parse_receipt(receipt)
        if parsed is None or parsed.task_id != task.task_id:
            raise ValueError("receipt identity is invalid")
        manifest_path, manifest_bytes = self._read_verified_artifact(
            parsed.manifest_path,
            expected_sha256=parsed.manifest_sha256,
            group_root=group_root,
        )
        result_path, result_bytes = self._read_verified_artifact(
            parsed.run_result_path,
            expected_sha256=parsed.run_result_sha256,
            group_root=group_root,
        )
        result_payload = RunResult.model_validate_json(result_bytes, strict=True)
        manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
        if isinstance(variant, RankerComponentVariant):
            expected_components = ("P1", variant.value)
        else:
            expected_components = COMPONENT_IDS[variant.value]
        if (manifest.planner_id, manifest.ranker_id) != expected_components:
            raise ValueError("run manifest component identity does not match variant")
        if manifest.config_sha256 != self._core_config_sha256(
            config=config,
            task=task,
            planner_id=manifest.planner_id,
            ranker_id=manifest.ranker_id,
            seed=seed,
        ):
            raise ValueError("run manifest config identity mismatch")
        group_metadata = json.loads((group_root / "group.json").read_bytes())
        if not isinstance(group_metadata, dict):
            raise TypeError("experiment group metadata is invalid")
        typed_group_metadata = cast(dict[str, object], group_metadata)
        if typed_group_metadata.get("code_commit") != manifest.code_commit:
            raise ValueError("run manifest code identity mismatch")
        if manifest.seed_supported != self.seed_supported:
            raise ValueError("run manifest seed support identity mismatch")
        if self.seed_supported:
            if seed is None or repeat_id is not None or manifest.seed != seed:
                raise ValueError("run manifest seed identity mismatch")
        elif seed is not None or repeat_id is None or manifest.seed is not None:
            raise ValueError("run manifest repeat identity mismatch")
        if (
            result_payload.status != parsed.status
            or result_payload.run_id != manifest.run_id
            or result_payload.thread_id != manifest.thread_id
            or result_payload.final_usage != manifest.usage
            or result_payload.stop_reason != manifest.stop_reason
            or result_payload.is_partial != manifest.is_partial
        ):
            raise ValueError("run result identity/accounting does not match manifest")
        result_artifacts = tuple(
            artifact_id
            for artifact_id in (
                result_payload.report_artifact_id,
                result_payload.evidence_graph_artifact_id,
                result_payload.manifest_artifact_id,
            )
            if artifact_id is not None
        )
        if not set(result_artifacts).issubset(set(manifest.artifact_ids)):
            raise ValueError("run result artifact identity does not match manifest")
        if tuple(parsed.artifact_ids) != tuple(manifest.artifact_ids):
            raise ValueError("receipt artifact identity does not match manifest")
        # This call revalidates task/request/budget/pricing/usage bindings using
        # the canonical Core converter; it must succeed before evaluator access.
        task_run_from_manifest(
            manifest,
            config=config,
            task=task,
            protocol=protocol,
            variant=variant,
            manifest_path=str(manifest_path),
            status=parsed.status,
            seed=seed,
            repeat_id=repeat_id,
            candidate_pool_hash=candidate_pool_hash,
        )
        if not self.seed_supported:
            if request.seed is not None or request.repeat_id != repeat_id:
                raise ValueError("request repeat identity does not match receipt binding")
            self._bind_unseeded_receipt(
                group_root=group_root,
                config=config,
                task=task,
                protocol=protocol,
                variant=variant,
                budget=budget,
                repeat_id=cast(int, repeat_id),
                request_sha256=sha256_bytes(_canonical(request.model_dump(mode="json"))),
                receipt_identity=sha256_bytes(_canonical(parsed.model_dump(mode="json"))),
                manifest_provenance={
                    "manifest_sha256": parsed.manifest_sha256,
                    "run_id": manifest.run_id,
                    "thread_id": manifest.thread_id,
                },
            )
        return _ValidatedAgentRun(
            receipt=parsed,
            manifest=manifest,
            result=result_payload,
            manifest_path=manifest_path,
            result_path=result_path,
        )

    @staticmethod
    def _failure_record(
        *,
        config: FormalExperimentConfig,
        group_root: Path,
        protocol: Protocol,
        variant: ExperimentVariant | RankerComponentVariant,
        task: RuntimeTask,
        budget: BudgetPreset,
        seed: int | None,
        repeat_id: int | None,
        candidate_pool_hash: str | None,
        key: str,
        error_code: str,
    ) -> ExperimentTaskRun:
        if isinstance(variant, RankerComponentVariant):
            planner_id, ranker_id = "P1", variant.value
        else:
            planner_id, ranker_id = COMPONENT_IDS[variant.value]
        return ExperimentTaskRun(
            task_id=task.task_id,
            protocol=protocol,
            variant=variant,
            planner_id=planner_id,
            ranker_id=ranker_id,
            budget_preset=budget,
            seed=seed,
            repeat_id=repeat_id,
            status="failed",
            validity="invalid",
            error_code=error_code,
            candidate_pool_hash=candidate_pool_hash,
            manifest_path=str((group_root / "artifacts" / f"{key}.run-manifest.json").resolve()),
            artifact_ids=(),
            usage=ResourceUsage.zero(cost_known=False),
            pricing_snapshot_ids=(config.pricing_snapshot.snapshot_id,),
            pricing_status="estimated",
            cost_label="estimated_from_normalized_schedule",
            category=task.category,
            metrics={},
        )

    @staticmethod
    def _read_verified_artifact(
        path: str,
        *,
        expected_sha256: str,
        group_root: Path,
    ) -> tuple[Path, bytes]:
        candidate = Path(path)
        if ".." in candidate.parts or candidate.is_symlink():
            raise ValueError("agent artifact path is unsafe")
        _assert_lexically_safe(candidate.parent, label="agent artifact path")
        resolved = candidate.resolve(strict=True)
        if not resolved.is_file() or not resolved.is_relative_to(group_root.resolve()):
            raise ValueError("agent artifact path is outside the experiment group")
        payload = resolved.read_bytes()
        if sha256_bytes(payload) != expected_sha256:
            raise ValueError("agent artifact hash mismatch")
        return resolved, payload

    def _record(
        self,
        *,
        config: FormalExperimentConfig,
        group_root: Path,
        protocol: Protocol,
        variant: ExperimentVariant | RankerComponentVariant,
        task: RuntimeTask,
        budget: BudgetPreset,
        seed: int | None,
        repeat_id: int | None,
        candidate_pool_hash: str | None,
        receipt: object,
        key: str,
        forced_error_code: str | None = None,
        metrics: Mapping[str, MetricValue] | None = None,
    ) -> ExperimentTaskRun:
        parsed = self._parse_receipt(receipt)
        error_code = forced_error_code or "AGENT_RECEIPT_INVALID"
        run: ExperimentTaskRun
        try:
            # A launcher/evaluator failure is authoritative.  Even if a child
            # happened to leave a syntactically valid receipt behind, that
            # receipt was not accepted by the failed boundary and must never
            # be promoted to a completed experiment record.
            if forced_error_code is not None:
                raise ValueError("forced runner failure")
            if parsed is None or parsed.task_id != task.task_id:
                raise ValueError("receipt identity is invalid")
            manifest_path, manifest_bytes = self._read_verified_artifact(
                parsed.manifest_path,
                expected_sha256=parsed.manifest_sha256,
                group_root=group_root,
            )
            _, result_bytes = self._read_verified_artifact(
                parsed.run_result_path,
                expected_sha256=parsed.run_result_sha256,
                group_root=group_root,
            )
            result_payload = RunResult.model_validate_json(result_bytes, strict=True)
            manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
            if isinstance(variant, RankerComponentVariant):
                expected_components = ("P1", variant.value)
            else:
                expected_components = COMPONENT_IDS[variant.value]
            if (manifest.planner_id, manifest.ranker_id) != expected_components:
                raise ValueError("run manifest component identity does not match variant")
            if manifest.config_sha256 != self._core_config_sha256(
                config=config,
                task=task,
                planner_id=manifest.planner_id,
                ranker_id=manifest.ranker_id,
                seed=seed,
            ):
                raise ValueError("run manifest config identity mismatch")
            group_metadata = json.loads((group_root / "group.json").read_bytes())
            typed_group_metadata = cast(dict[str, object], group_metadata)
            if typed_group_metadata.get("code_commit") != manifest.code_commit:
                raise ValueError("run manifest code identity mismatch")
            if manifest.seed_supported != self.seed_supported:
                raise ValueError("run manifest seed support identity mismatch")
            if self.seed_supported:
                if seed is None or repeat_id is not None or manifest.seed != seed:
                    raise ValueError("run manifest seed identity mismatch")
            elif seed is not None or repeat_id is None or manifest.seed is not None:
                raise ValueError("run manifest repeat identity mismatch")
            if (
                result_payload.status != parsed.status
                or result_payload.run_id != manifest.run_id
                or result_payload.thread_id != manifest.thread_id
                or result_payload.final_usage != manifest.usage
            ):
                raise ValueError("run result identity/accounting does not match manifest")
            result_artifacts = tuple(
                artifact_id
                for artifact_id in (
                    result_payload.report_artifact_id,
                    result_payload.evidence_graph_artifact_id,
                    result_payload.manifest_artifact_id,
                )
                if artifact_id is not None
            )
            if not set(result_artifacts).issubset(set(manifest.artifact_ids)):
                raise ValueError("run result artifact identity does not match manifest")
            if tuple(parsed.artifact_ids) != tuple(manifest.artifact_ids):
                raise ValueError("receipt artifact identity does not match manifest")
            run = task_run_from_manifest(
                manifest,
                config=config,
                task=task,
                protocol=protocol,
                variant=variant,
                manifest_path=str(manifest_path),
                status=parsed.status,
                seed=seed,
                repeat_id=repeat_id,
                candidate_pool_hash=candidate_pool_hash,
                metrics=metrics,
            )
            if parsed.status != "completed":
                run = ExperimentTaskRun.model_validate(
                    {
                        **run.model_dump(mode="json"),
                        "validity": "invalid",
                        "error_code": parsed.error_code or run.error_code or "AGENT_FAILED",
                    },
                    strict=True,
                )
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            if forced_error_code is None:
                if parsed is not None and parsed.status != "completed":
                    error_code = parsed.error_code or "AGENT_FAILED"
                elif parsed is not None:
                    error_code = "AGENT_MANIFEST_INVALID"
            run = self._failure_record(
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget,
                seed=seed,
                repeat_id=repeat_id,
                candidate_pool_hash=candidate_pool_hash,
                key=key,
                error_code=error_code,
            )
        raw_path = self._raw_path(group_root, key)
        existing = self._load_existing(raw_path)
        if existing is not None:
            if existing.status == "completed":
                raise RuntimeError("completed experiment record cannot be overwritten")
            raw_path.unlink()
        _write_immutable(raw_path, _canonical(run.model_dump(mode="json")))
        return run

    async def _stage_request(
        self,
        *,
        config: FormalExperimentConfig,
        group_root: Path,
        task: RuntimeTask,
        task_id: str,
        protocol: Protocol,
        variant: ExperimentVariant | RankerComponentVariant,
        budget: BudgetPreset,
        seed: int | None,
        repeat_id: int | None,
        candidate_pool_path: Path | None = None,
        candidate_pool_hash: str | None = None,
        resume_checkpoint_path: Path | None = None,
        resume_checkpoint_sha256: str | None = None,
        resume_checkpoint_ref: CheckpointRef | None = None,
    ) -> tuple[AgentRunRequest, StagedRuntimeTask, str]:
        staged = stage_authorized_runtime_task(
            task,
            config=config,
            budget_preset=budget,
            agent_input_root=group_root / "agent-inputs",
            request_id=self._idempotency_key(
                config.experiment_group_id(), protocol, variant.value, task_id, seed, repeat_id, budget
            )[:32],
            forbidden_private_root=self.private_root,
        )
        config_path, config_hash = self._config_path_and_hash(group_root)
        request_key = self._idempotency_key(
            config.experiment_group_id(), protocol, variant.value, task_id, seed, repeat_id, budget
        )
        common: dict[str, object] = {
            "task_id": task_id,
            "runtime_task_path": staged.runtime_task_path,
            "runtime_task_sha256": staged.runtime_task_sha256,
            "base_runtime_task_sha256": staged.base_runtime_task_sha256,
            "snapshot_dir": str(self._snapshot_dir(task).resolve()),
            "run_dir": str(group_root.resolve()),
            "config_path": str(config_path.resolve()),
            "config_sha256": config_hash,
            "seed_supported": self.seed_supported,
            "seed": seed,
            "repeat_id": repeat_id,
            "budget_preset": budget,
        }
        if isinstance(variant, RankerComponentVariant):
            request: AgentRunRequest = AgentVariantRunRequest.model_validate(
                {
                    **common,
                    "kind": "variant_run",
                    "protocol": protocol,
                    "variant": variant.value,
                    "candidate_pool_path": str(candidate_pool_path.resolve()) if candidate_pool_path else None,
                    "candidate_pool_sha256": candidate_pool_hash,
                    "resume_checkpoint_path": str(resume_checkpoint_path.resolve())
                    if resume_checkpoint_path
                    else None,
                    "resume_checkpoint_sha256": resume_checkpoint_sha256,
                    "resume_checkpoint_ref": resume_checkpoint_ref,
                }
            )
        else:
            request = AgentVariantRunRequest.model_validate(
                {
                    **common,
                    "kind": "variant_run",
                    "protocol": protocol,
                    "variant": variant.value,
                    "candidate_pool_path": str(candidate_pool_path.resolve()) if candidate_pool_path else None,
                    "candidate_pool_sha256": candidate_pool_hash,
                    "resume_checkpoint_path": str(resume_checkpoint_path.resolve())
                    if resume_checkpoint_path
                    else None,
                    "resume_checkpoint_sha256": resume_checkpoint_sha256,
                    "resume_checkpoint_ref": resume_checkpoint_ref,
                }
            )
        request_path = group_root / "requests" / f"{request_key}.json"
        if resume_checkpoint_path is not None:
            request_path = group_root / "requests" / f"{request_key}.resume.json"
        _write_immutable(request_path, _canonical(request.model_dump(mode="json")))
        return request, staged, request_key

    def _stage_resume_checkpoint(
        self,
        *,
        group_root: Path,
        key: str,
        source: Path,
        ref: CheckpointRef,
    ) -> tuple[Path, str]:
        from benchmarks.processes.agent import verify_checkpoint_identity

        source = Path(source)
        if ".." in source.parts or source.is_symlink():
            raise ValueError("resume checkpoint source is unsafe")
        if source.name != f"{key}.sqlite3":
            raise ValueError("resume checkpoint is not bound to the experiment key")
        source = source.resolve(strict=True)
        if not source.is_file() or not source.is_relative_to((group_root / "artifacts").resolve()):
            raise ValueError("resume checkpoint must be a prior group artifact")
        verify_checkpoint_identity(source, ref)
        payload = source.read_bytes()
        destination = group_root / "resume-checkpoints" / f"{key}.sqlite3"
        _write_immutable(destination, payload)
        digest = sha256_bytes(destination.read_bytes())
        return destination, digest

    async def run_one(
        self,
        *,
        config: FormalExperimentConfig,
        protocol: Protocol,
        task_id: str,
        variant: ExperimentVariant | RankerComponentVariant,
        budget_preset: BudgetPreset,
        seed: int | None = None,
        repeat_id: int | None = None,
        resume: bool = False,
        candidate_pool_path: Path | None = None,
        candidate_pool_hash: str | None = None,
        resume_checkpoint_path: Path | None = None,
        resume_checkpoint_ref: CheckpointRef | None = None,
    ) -> ExperimentTaskRun:
        if variant == ExperimentVariant.ORACLE:
            raise ValueError("ORACLE is evaluator-only")
        if protocol == "ranker_component" and not isinstance(variant, RankerComponentVariant):
            raise ValueError("ranker protocol requires R0/R1/R2")
        if protocol != "ranker_component" and isinstance(variant, RankerComponentVariant):
            raise ValueError("R variants require ranker_component protocol")
        if budget_preset not in config.budget_sensitivity_presets:
            raise ValueError("budget is not sealed")
        selected_seed, selected_repeat = self._seed_or_repeat(
            config, seed=seed, repeat_id=repeat_id
        )
        task = await self._task(task_id)
        group_root = self._group_root(config)
        key = self._idempotency_key(
            config.experiment_group_id(), protocol, variant.value, task_id,
            selected_seed, selected_repeat, budget_preset,
        )
        raw_path = self._raw_path(group_root, key)
        existing = self._load_existing(raw_path)
        if existing is not None:
            if existing.status == "completed":
                return existing
            if not resume:
                raise RuntimeError("failed experiment key requires explicit resume")
        elif resume:
            return self._record(
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_hash=candidate_pool_hash,
                receipt=None,
                key=key,
                forced_error_code="RESUME_CHECKPOINT_INVALID",
            )
        staged_checkpoint: Path | None = None
        staged_checkpoint_sha256: str | None = None
        if resume:
            if resume_checkpoint_path is None or resume_checkpoint_ref is None:
                return self._record(
                    config=config,
                    group_root=group_root,
                    protocol=protocol,
                    variant=variant,
                    task=task,
                    budget=budget_preset,
                    seed=selected_seed,
                    repeat_id=selected_repeat,
                    candidate_pool_hash=candidate_pool_hash,
                    receipt=None,
                    key=key,
                    forced_error_code="RESUME_CHECKPOINT_INVALID",
                )
            try:
                staged_checkpoint, staged_checkpoint_sha256 = self._stage_resume_checkpoint(
                    group_root=group_root,
                    key=key,
                    source=resume_checkpoint_path,
                    ref=resume_checkpoint_ref,
                )
            except (OSError, ValueError, TypeError):
                return self._record(
                    config=config,
                    group_root=group_root,
                    protocol=protocol,
                    variant=variant,
                    task=task,
                    budget=budget_preset,
                    seed=selected_seed,
                    repeat_id=selected_repeat,
                    candidate_pool_hash=candidate_pool_hash,
                    receipt=None,
                    key=key,
                    forced_error_code="RESUME_CHECKPOINT_INVALID",
                )
        try:
            request, _, _ = await self._stage_request(
                config=config,
                group_root=group_root,
                task=task,
                task_id=task_id,
                protocol=protocol,
                variant=variant,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_path=candidate_pool_path,
                candidate_pool_hash=candidate_pool_hash,
                resume_checkpoint_path=staged_checkpoint,
                resume_checkpoint_sha256=staged_checkpoint_sha256,
                resume_checkpoint_ref=resume_checkpoint_ref,
            )
        except Exception:  # noqa: BLE001 - staging failures are auditable
            return self._record(
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_hash=candidate_pool_hash,
                receipt=None,
                key=key,
                forced_error_code="AGENT_REQUEST_STAGE_FAILED",
            )
        try:
            receipt = await self._call_launcher(request)
        except Exception:  # noqa: BLE001 - child failures become auditable records
            return self._record(
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_hash=candidate_pool_hash,
                receipt=None,
                key=key,
                forced_error_code="AGENT_LAUNCH_FAILED",
            )
        try:
            validated = self._validate_agent_receipt(
                request=request,
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_hash=candidate_pool_hash,
                receipt=receipt,
            )
        except Exception:  # noqa: BLE001 - invalid child output is an auditable record
            return self._record(
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_hash=candidate_pool_hash,
                receipt=receipt,
                key=key,
                forced_error_code="AGENT_RECEIPT_INVALID",
            )
        try:
            evaluator_output = await self._call_evaluator(request, validated.receipt)
            metrics = self._parse_evaluator_metrics(evaluator_output)
        except Exception:  # noqa: BLE001 - evaluator validation becomes an auditable record
            return self._record(
                config=config,
                group_root=group_root,
                protocol=protocol,
                variant=variant,
                task=task,
                budget=budget_preset,
                seed=selected_seed,
                repeat_id=selected_repeat,
                candidate_pool_hash=candidate_pool_hash,
                receipt=validated.receipt,
                key=key,
                forced_error_code="EVALUATOR_VALIDATION_FAILED",
            )
        return self._record(
            config=config,
            group_root=group_root,
            protocol=protocol,
            variant=variant,
            task=task,
            budget=budget_preset,
            seed=selected_seed,
            repeat_id=selected_repeat,
            candidate_pool_hash=candidate_pool_hash,
            receipt=validated.receipt,
            key=key,
            metrics=metrics,
        )

    async def _replications(self, config: FormalExperimentConfig) -> tuple[tuple[int | None, int | None], ...]:
        if self.seed_supported:
            return tuple((seed, None) for seed in config.replication.seed_values)
        return tuple((None, repeat) for repeat in range(1, config.replication.unseeded_repeat_count + 1))

    async def _candidate_pool(
        self, *, config: FormalExperimentConfig, task: RuntimeTask
    ) -> tuple[Path, str] | None:
        group_root = self._group_root(config)
        request_key = self._idempotency_key(
            config.experiment_group_id(), "ranker_component", "POOL", task.task_id,
            config.replication.candidate_pool_seed, None, config.budget_preset,
        )
        path = group_root / "candidate-pools" / f"{request_key}.json"
        setup_path = group_root / "candidate-pool-setup" / f"{request_key}.json"
        request_path = group_root / "requests" / f"{request_key}.json"
        try:
            _assert_lexically_safe(path.parent, label="candidate pool input subtree")
            _assert_lexically_safe(setup_path.parent, label="candidate pool setup subtree")
            # Setup is immutable and paid once per (group, task, pool seed).
            # Reusing a promoted pool without its matching setup accounting is
            # ambiguous, so a partial or inconsistent pair fails closed.
            if path.exists() or setup_path.exists():
                if (
                    not path.is_file()
                    or path.is_symlink()
                    or not setup_path.is_file()
                    or setup_path.is_symlink()
                ):
                    raise ValueError("candidate pool setup pair is incomplete")
                setup_payload = json.loads(setup_path.read_bytes())
                if not isinstance(setup_payload, dict):
                    raise ValueError("candidate pool setup metadata is invalid")
                typed_setup = cast(dict[str, object], setup_payload)
                if set(typed_setup) != {
                    "schema_version",
                    "group_id",
                    "task_id",
                    "pool_key",
                    "candidate_pool_sha256",
                    "evidence_ids_sha256",
                    "manifest_path",
                    "manifest_sha256",
                    "usage",
                    "pricing_snapshot_ids",
                    "pricing_status",
                }:
                    raise ValueError("candidate pool setup metadata schema is invalid")
                if (
                    typed_setup["schema_version"] != "candidate-pool-setup-v1"
                    or typed_setup["group_id"] != config.experiment_group_id()
                    or typed_setup["task_id"] != task.task_id
                    or typed_setup["pool_key"] != request_key
                    or typed_setup["pricing_status"] != "estimated"
                ):
                    raise ValueError("candidate pool setup identity mismatch")
                if setup_path.name != f"{request_key}.json":
                    raise ValueError("candidate pool setup filename is not pool-bound")
                if not self._is_sha256(typed_setup["candidate_pool_sha256"]):
                    raise ValueError("candidate pool setup candidate hash is invalid")
                if not self._is_sha256(typed_setup["evidence_ids_sha256"]):
                    raise ValueError("candidate pool setup evidence hash is invalid")
                if not self._is_sha256(typed_setup["manifest_sha256"]):
                    raise ValueError("candidate pool setup manifest hash is invalid")
                candidate_bytes = path.read_bytes()
                candidate_hash = sha256_bytes(candidate_bytes)
                if typed_setup["candidate_pool_sha256"] != candidate_hash:
                    raise ValueError("candidate pool setup hash mismatch")
                setup_usage = ResourceUsage.model_validate_json(
                    _canonical(typed_setup["usage"]), strict=True
                )
                setup_pricing = typed_setup["pricing_snapshot_ids"]
                if (
                    setup_usage.cost_usd is None
                    or setup_pricing != [config.pricing_snapshot.snapshot_id]
                    or not isinstance(typed_setup["manifest_path"], str)
                ):
                    raise ValueError("candidate pool setup accounting is inconsistent")
                manifest_path, manifest_bytes = self._read_verified_artifact(
                    typed_setup["manifest_path"],
                    expected_sha256=cast(str, typed_setup["manifest_sha256"]),
                    group_root=group_root,
                )
                if manifest_path.parent != (group_root / "artifacts").resolve():
                    raise ValueError("candidate pool setup manifest is not an artifact")
                manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
                self._validate_setup_manifest(
                    manifest,
                    config=config,
                    task=task,
                    seed=config.replication.candidate_pool_seed,
                    planner_id="P1",
                )
                if manifest.usage != setup_usage or tuple(
                    item.snapshot_id for item in manifest.pricing_snapshots
                ) != tuple(cast(list[str], setup_pricing)):
                    raise ValueError("candidate pool setup manifest accounting mismatch")
                candidate_payload = json.loads(candidate_bytes)
                if not isinstance(candidate_payload, dict):
                    raise ValueError("candidate pool payload is invalid")
                typed_candidate = cast(dict[str, object], candidate_payload)
                if typed_candidate.get("task_id") != task.task_id:
                    raise ValueError("candidate pool task identity mismatch")
                evidence_ids = typed_candidate.get("evidence_ids")
                if (
                    not isinstance(evidence_ids, list)
                    or any(
                        not isinstance(item, str) or not item
                        for item in cast(list[object], evidence_ids)
                    )
                    or cast(list[object], evidence_ids)
                    != sorted(set(cast(list[str], evidence_ids)))
                ):
                    raise ValueError("candidate pool evidence IDs are not canonical")
                if sha256_bytes(_canonical(cast(list[str], evidence_ids))) != typed_setup[
                    "evidence_ids_sha256"
                ]:
                    raise ValueError("candidate pool setup evidence hash mismatch")
                GoldIsolationGuard(
                    runtime_root=group_root / "agent-inputs",
                    snapshot_root=self.snapshot_root,
                    private_root=self.private_root,
                ).validate_run_payload(cast(JsonValue, typed_candidate))
                return path, candidate_hash
            staged = stage_authorized_runtime_task(
                task,
                config=config,
                budget_preset=config.budget_preset,
                agent_input_root=group_root / "agent-inputs",
                request_id=request_key[:32],
                forbidden_private_root=self.private_root,
            )
            cfg_path, cfg_hash = self._config_path_and_hash(group_root)
            request = AgentCandidatePoolRequest(
                task_id=task.task_id,
                runtime_task_path=staged.runtime_task_path,
                runtime_task_sha256=staged.runtime_task_sha256,
                base_runtime_task_sha256=staged.base_runtime_task_sha256,
                snapshot_dir=str(self._snapshot_dir(task).absolute()),
                run_dir=str(group_root.absolute()),
                config_path=str(cfg_path.absolute()),
                config_sha256=cfg_hash,
                seed_supported=True,
                seed=config.replication.candidate_pool_seed,
                budget_preset=config.budget_preset,
            )
            _write_immutable(request_path, _canonical(request.model_dump(mode="json")))
            receipt = await self._call_launcher(request)
            if not isinstance(receipt, AgentCandidatePoolReceipt):
                receipt = AgentCandidatePoolReceipt.model_validate(receipt, strict=True)
            if receipt.task_id != task.task_id:
                raise ValueError("candidate pool receipt task identity mismatch")
            received, candidate_bytes = self._read_verified_artifact(
                receipt.candidate_pool_path,
                expected_sha256=receipt.candidate_pool_sha256,
                group_root=group_root,
            )
            if received.parent != (group_root / "staging").resolve():
                raise ValueError("candidate pool must be staged outside protected inputs")
            candidate_payload = json.loads(candidate_bytes)
            if not isinstance(candidate_payload, dict):
                raise TypeError("candidate pool payload is invalid")
            candidate_payload = cast(dict[str, object], candidate_payload)
            GoldIsolationGuard(
                runtime_root=group_root / "agent-inputs",
                snapshot_root=self.snapshot_root,
                private_root=self.private_root,
            ).validate_run_payload(cast(JsonValue, candidate_payload))
            if candidate_payload.get("task_id") != task.task_id:
                raise ValueError("candidate pool task identity mismatch")
            evidence_ids = candidate_payload.get("evidence_ids")
            if (
                not isinstance(evidence_ids, list)
                or any(not isinstance(item, str) or not item for item in cast(list[object], evidence_ids))
                or cast(list[object], evidence_ids)
                != sorted(set(cast(list[str], evidence_ids)))
            ):
                raise ValueError("candidate pool evidence IDs are not canonical")
            typed_evidence_ids = cast(list[str], evidence_ids)
            if sha256_bytes(_canonical(typed_evidence_ids)) != receipt.evidence_ids_sha256:
                raise ValueError("candidate pool evidence hash mismatch")
            _, manifest_bytes = self._read_verified_artifact(
                receipt.manifest_path,
                expected_sha256=receipt.manifest_sha256,
                group_root=group_root,
            )
            manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
            self._validate_setup_manifest(
                manifest,
                config=config,
                task=task,
                seed=config.replication.candidate_pool_seed,
                planner_id="P1",
            )
            if receipt.usage != manifest.usage or receipt.pricing_snapshot_ids != tuple(
                item.snapshot_id for item in manifest.pricing_snapshots
            ):
                raise ValueError("candidate pool receipt accounting mismatch")
            # Promotion is the only write to the protected shared pool tree.
            _write_immutable(path, candidate_bytes)
            _write_immutable(
                setup_path,
                _canonical(
                    {
                        "schema_version": "candidate-pool-setup-v1",
                        "group_id": config.experiment_group_id(),
                        "task_id": task.task_id,
                        "pool_key": request_key,
                        "candidate_pool_sha256": sha256_bytes(candidate_bytes),
                        "evidence_ids_sha256": receipt.evidence_ids_sha256,
                        "manifest_path": str(receipt.manifest_path),
                        "manifest_sha256": receipt.manifest_sha256,
                        "usage": manifest.usage.model_dump(mode="json"),
                        "pricing_snapshot_ids": list(receipt.pricing_snapshot_ids),
                        "pricing_status": manifest.pricing_status,
                    }
                ),
            )
            return path, sha256_bytes(candidate_bytes)
        except (OSError, RuntimeError, TypeError, ValueError, KeyError):
            # A missing/invalid setup receipt is a failed protocol arm.  An
            # empty or evaluator-authored fallback pool would turn failure into
            # an apparently comparable ranker result, so fail closed here.
            return None

    def _record_candidate_pool_failure(
        self,
        *,
        config: FormalExperimentConfig,
        group_root: Path,
        task: RuntimeTask,
        variant: RankerComponentVariant,
        budget: BudgetPreset,
        seed: int | None,
        repeat_id: int | None,
        key: str,
    ) -> ExperimentTaskRun:
        failed = self._failure_record(
            config=config,
            group_root=group_root,
            protocol="ranker_component",
            variant=variant,
            task=task,
            budget=budget,
            seed=seed,
            repeat_id=repeat_id,
            candidate_pool_hash=None,
            key=key,
            error_code="CANDIDATE_POOL_INVALID",
        )
        raw_path = self._raw_path(group_root, key)
        existing = self._load_existing(raw_path)
        if existing is not None:
            if existing.status == "completed":
                # A completed record is immutable and must not be reused when
                # its required pool has become invalid. Keep an additional
                # auditable failure outside ``raw/`` rather than replacing the
                # completed idempotency record or deleting a prior failure.
                audit_path = (
                    group_root
                    / "artifacts"
                    / "candidate-pool-failures"
                    / f"{key}.json"
                )
                _write_immutable(audit_path, _canonical(failed.model_dump(mode="json")))
                return failed
            return existing
        _write_immutable(raw_path, _canonical(failed.model_dump(mode="json")))
        return failed

    async def _run_variants(
        self,
        *,
        config: FormalExperimentConfig,
        protocol: Protocol,
        task_ids: Sequence[str],
        variants: Sequence[ExperimentVariant | RankerComponentVariant],
        budget: BudgetPreset | None = None,
    ) -> ExperimentRunResult:
        runs: list[ExperimentTaskRun] = []
        pools: dict[str, tuple[Path, str]] = {}
        for task_id in task_ids:
            task = await self._task(task_id)
            if protocol == "ranker_component":
                group_root = self._group_root(config)
                pool_key = self._idempotency_key(
                    config.experiment_group_id(),
                    "ranker_component",
                    "POOL",
                    task.task_id,
                    config.replication.candidate_pool_seed,
                    None,
                    config.budget_preset,
                )
                pool_path = group_root / "candidate-pools" / f"{pool_key}.json"
                setup_path = group_root / "candidate-pool-setup" / f"{pool_key}.json"
                replications = await self._replications(config)
                prior_arm_attempt = any(
                    self._load_existing(
                        self._raw_path(
                            group_root,
                            self._idempotency_key(
                                config.experiment_group_id(),
                                protocol,
                                variant.value,
                                task_id,
                                seed,
                                repeat,
                                budget or config.budget_preset,
                            ),
                        )
                    )
                    is not None
                    for variant in variants
                    for seed, repeat in replications
                )
                # Once a consumer attempt exists, a missing promoted pool is
                # not permission to launch setup again. That would let a
                # deleted pool silently resurrect completed/failed arms without
                # an explicit resume decision.
                if prior_arm_attempt and not (pool_path.exists() or setup_path.exists()):
                    pool = None
                else:
                    pool = await self._candidate_pool(config=config, task=task)
                if pool is None:
                    for variant in variants:
                        for seed, repeat in await self._replications(config):
                            key = self._idempotency_key(
                                config.experiment_group_id(),
                                protocol,
                                variant.value,
                                task_id,
                                seed,
                                repeat,
                                budget or config.budget_preset,
                            )
                            if not isinstance(variant, RankerComponentVariant):
                                raise TypeError("ranker protocol variant is invalid")
                            failed = self._record_candidate_pool_failure(
                                config=config,
                                group_root=self._group_root(config),
                                variant=variant,
                                task=task,
                                budget=budget or config.budget_preset,
                                seed=seed,
                                repeat_id=repeat,
                                key=key,
                            )
                            runs.append(failed)
                    continue
                pools[task_id] = pool
            for variant in variants:
                for seed, repeat in await self._replications(config):
                    path, digest = pools.get(task_id, (None, None))  # type: ignore[assignment]
                    run = await self.run_one(
                        config=config,
                        protocol=protocol,
                        task_id=task_id,
                        variant=variant,
                        budget_preset=budget or config.budget_preset,
                        seed=seed,
                        repeat_id=repeat,
                        candidate_pool_path=path,
                        candidate_pool_hash=digest,
                    )
                    runs.append(run)
        components = {
            variant.value: (("P1", variant.value) if isinstance(variant, RankerComponentVariant) else COMPONENT_IDS[variant.value])
            for variant in variants
        }
        return ExperimentRunResult(
            group_id=config.experiment_group_id(),
            protocol=protocol,
            variant_components=components,
            runs=tuple(runs),
        )

    async def run_ranker_component(
        self, *, config: FormalExperimentConfig, task_ids: Sequence[str]
    ) -> ExperimentRunResult:
        return await self._run_variants(
            config=config,
            protocol="ranker_component",
            task_ids=task_ids,
            variants=(RankerComponentVariant.R0, RankerComponentVariant.R1, RankerComponentVariant.R2),
        )

    async def run_planner_policy(
        self, *, config: FormalExperimentConfig, task_ids: Sequence[str], ranker_id: Literal["R1", "R2"]
    ) -> ExperimentRunResult:
        variants = (
            (ExperimentVariant.A, ExperimentVariant.C)
            if ranker_id == "R1"
            else (ExperimentVariant.B, ExperimentVariant.D)
        )
        return await self._run_variants(
            config=config, protocol="planner_policy", task_ids=task_ids, variants=variants
        )

    async def run_variant(
        self,
        variant: ExperimentVariant,
        *,
        config: FormalExperimentConfig,
        task_ids: Sequence[str],
    ) -> ExperimentRunResult:
        if variant == ExperimentVariant.ORACLE:
            raise ValueError("ORACLE is evaluator-only")
        return await self._run_variants(
            config=config, protocol="reference" if variant == ExperimentVariant.P0 else "end_to_end",
            task_ids=task_ids, variants=(variant,)
        )

    async def run_abcd(self, *, config: FormalExperimentConfig) -> ExperimentRunResult:
        return await self._run_variants(
            config=config,
            protocol="end_to_end",
            task_ids=config.main_test_task_ids,
            variants=(ExperimentVariant.A, ExperimentVariant.B, ExperimentVariant.C, ExperimentVariant.D),
        )

    async def run_cost_subset(
        self, *, config: FormalExperimentConfig
    ) -> tuple[ExperimentRunResult, ...]:
        results: list[ExperimentRunResult] = []
        for budget in config.budget_sensitivity_presets:
            results.append(
                await self._run_variants(
                    config=config,
                    protocol="end_to_end",
                    task_ids=config.cost_subset_task_ids,
                    variants=(ExperimentVariant.D,),
                    budget=budget,
                )
            )
        return tuple(results)

    async def run_reference(
        self,
        *,
        config: FormalExperimentConfig,
        task_ids: Sequence[str] | None = None,
    ) -> ExperimentRunResult:
        """Run P0 through the agent and, when supplied, ORACLE in evaluator only."""
        result = await self.run_variant(
            ExperimentVariant.P0,
            config=config,
            task_ids=tuple(task_ids or config.p0_task_ids),
        )
        if self.oracle_provider is None:
            raise RuntimeError("evaluator ORACLE provider is required for reference runs")
        provider = cast(Any, self.oracle_provider)
        selected = tuple(config.oracle_task_ids)
        oracle_results: list[object] = []
        for task_id in selected:
            records = self.oracle_records_loader.get(task_id)
            if records is None:
                raise ValueError("ORACLE frozen records are unavailable")
            score = provider.score_reference
            oracle_results.append(score(task_id, frozen_records=records))
        manifest = provider.reference_manifest(
            group_id=config.experiment_group_id(), results=tuple(oracle_results)
        )
        group_root = self._group_root(config)
        oracle_bytes = b"".join(
            _canonical(cast(Any, item).model_dump(mode="json")) for item in oracle_results
        )
        _write_immutable(group_root / "oracle-reference.jsonl", oracle_bytes)
        _write_immutable(
            group_root / "evaluator-reference-manifest.json",
            _canonical(manifest.model_dump(mode="json")),
        )
        return result


__all__ = ["ExperimentRunner"]
