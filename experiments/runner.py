"""Evaluator-side orchestration for reproducible formal component experiments.

This module deliberately contains no Core agent implementation import.  The
only code that can construct a model/provider graph is the child entrypoint in
``benchmarks.processes.agent`` after it has verified its request and roots.
"""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import subprocess
from collections.abc import Awaitable, Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Literal, cast

from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
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
from deepresearch.domain import ResourceUsage
from experiments.config import FormalExperimentConfig, preflight_config
from experiments.models import (
    COMPONENT_IDS,
    ExperimentRunResult,
    ExperimentTaskRun,
    ExperimentVariant,
    RankerComponentVariant,
    canonical_sha256,
)

Protocol = Literal["ranker_component", "planner_policy", "end_to_end", "reference"]
BudgetPreset = Literal["low", "medium", "high"]


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


def _git_commit(root: Path) -> str:
    try:
        commit = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = "unknown"
    return commit if len(commit) == 40 else "unknown"


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
    ) -> None:
        self._launch_agent = launch_agent or self._default_launch_agent
        self._evaluator = evaluator
        self._task_loader = task_loader
        self.experiment_root = Path(experiment_root)
        self.repo_root = (repo_root or Path.cwd()).resolve()
        self.private_root = (
            Path(private_root).resolve()
            if private_root is not None
            else self.repo_root / "benchmarks" / "private"
        )
        self.snapshot_root = (
            Path(snapshot_root).resolve()
            if snapshot_root is not None
            else self.repo_root / "benchmarks" / "snapshots"
        )
        self.config_source = Path(config_source).resolve() if config_source else None
        self.seed_supported = seed_supported
        self.oracle_provider = oracle_provider
        self.oracle_records_loader = oracle_records_loader or {}

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

    async def _call_evaluator(self, request: AgentRunRequest, receipt: object) -> object | None:
        if self._evaluator is None:
            return None
        result = self._evaluator(request, receipt)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    def _group_root(self, config: FormalExperimentConfig) -> Path:
        private_root = self.private_root
        dataset_private_root = private_root / config.dataset_id
        if not (private_root / "private_manifest.json").is_file() and (
            dataset_private_root / "private_manifest.json"
        ).is_file():
            private_root = dataset_private_root
        if (private_root / "private_manifest.json").is_file():
            preflight_config(config, repo_root=self.repo_root, private_root=private_root)
        group = config.experiment_group_id()
        root = (self.experiment_root / group).resolve()
        self.experiment_root.mkdir(parents=True, exist_ok=True)
        root.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "formal-experiment-group-v1",
            "group_id": group,
            "config_sha256": canonical_sha256(config.model_dump(mode="json")),
            "code_commit": _git_commit(self.repo_root),
            "dataset_version": config.dataset_version,
            "protocols": ["ranker_component", "planner_policy", "end_to_end", "reference"],
        }
        _write_immutable(root / "group.json", _canonical(payload))
        config_path = root / "config" / "formal.yaml"
        config_bytes = _canonical(config.model_dump(mode="json"))
        if self.config_source is not None and self.config_source.exists():
            expected = sha256_bytes(self.config_source.read_bytes())
            staged = stage_sealed_config(
                self.config_source,
                expected_sha256=expected,
                group_run_root=root,
            )
            config_bytes = staged.read_bytes()
        else:
            _write_immutable(config_path, config_bytes)
        for directory in ("agent-inputs", "requests", "candidate-pools", "resume-checkpoints", "raw", "artifacts"):
            (root / directory).mkdir(parents=True, exist_ok=True)
        return root

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
    def _idempotency_key(
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
                return AgentRunReceipt.model_validate(receipt)
            except (TypeError, ValueError):
                return None
        return None

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
    ) -> ExperimentTaskRun:
        parsed = self._parse_receipt(receipt)
        status: Literal["completed", "failed"] = "completed"
        error_code: str | None = None
        if parsed is not None and parsed.status != "completed":
            status = "failed"
            error_code = parsed.error_code or "AGENT_FAILED"
        if parsed is not None:
            manifest_path = parsed.manifest_path
            artifact_ids = parsed.artifact_ids
        else:
            manifest_path = str((group_root / "artifacts" / f"{key}.run-manifest.json").resolve())
            artifact_ids = ()
        usage = ResourceUsage.zero(cost_known=True)
        if isinstance(variant, RankerComponentVariant):
            planner_id, ranker_id = "P1", variant.value
        else:
            planner_id, ranker_id = COMPONENT_IDS[variant.value]
        run = ExperimentTaskRun(
            task_id=task.task_id,
            protocol=protocol,
            variant=variant,
            planner_id=planner_id,
            ranker_id=ranker_id,
            budget_preset=budget,
            seed=seed,
            repeat_id=repeat_id,
            status=status,
            validity="valid" if status == "completed" else "invalid",
            error_code=error_code,
            candidate_pool_hash=candidate_pool_hash,
            manifest_path=manifest_path,
            artifact_ids=artifact_ids,
            usage=usage,
            pricing_snapshot_ids=(config.pricing_snapshot.snapshot_id,),
            pricing_status="estimated",
            cost_label="estimated_from_normalized_schedule",
        )
        raw_path = self._raw_path(group_root, key)
        existing = self._load_existing(raw_path)
        if existing is not None and existing.status != "completed":
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
                }
            )
        request_path = group_root / "requests" / f"{request_key}.json"
        _write_immutable(request_path, _canonical(request.model_dump(mode="json")))
        return request, staged, request_key

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
        )
        receipt = await self._call_launcher(request)
        await self._call_evaluator(request, receipt)
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
        )

    async def _replications(self, config: FormalExperimentConfig) -> tuple[tuple[int | None, int | None], ...]:
        if self.seed_supported:
            return tuple((seed, None) for seed in config.replication.seed_values)
        return tuple((None, repeat) for repeat in range(1, config.replication.unseeded_repeat_count + 1))

    async def _candidate_pool(
        self, *, config: FormalExperimentConfig, task: RuntimeTask
    ) -> tuple[Path, str]:
        group_root = self._group_root(config)
        path = group_root / "candidate-pools" / f"{task.task_id}-{config.replication.candidate_pool_seed}.json"
        payload = _canonical(
            {"task_id": task.task_id, "seed": config.replication.candidate_pool_seed, "version": "formal-v1"}
        )
        request_key = self._idempotency_key(
            config.experiment_group_id(), "ranker_component", "POOL", task.task_id,
            config.replication.candidate_pool_seed, None, config.budget_preset,
        )
        request_path = group_root / "requests" / f"{request_key}.json"
        if not request_path.exists():
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
                snapshot_dir=str(self._snapshot_dir(task).resolve()),
                run_dir=str(group_root.resolve()),
                config_path=str(cfg_path.resolve()),
                config_sha256=cfg_hash,
                seed=config.replication.candidate_pool_seed,
                budget_preset=config.budget_preset,
            )
            _write_immutable(request_path, _canonical(request.model_dump(mode="json")))
            receipt = await self._call_launcher(request)
            if isinstance(receipt, AgentCandidatePoolReceipt):
                received = Path(receipt.candidate_pool_path)
                if received.is_file() and sha256_bytes(received.read_bytes()) == receipt.candidate_pool_sha256:
                    _write_immutable(path, received.read_bytes())
        if not path.is_file():
            _write_immutable(path, payload)
        digest = sha256_bytes(path.read_bytes())
        return path, digest

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
                pools[task_id] = await self._candidate_pool(config=config, task=task)
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
            return result
        provider = cast(Any, self.oracle_provider)
        selected = tuple(task_ids or config.oracle_task_ids)
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
