"""Evaluator-only runner for the optional external Portfolio benchmarks."""

from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import subprocess
import sys
from collections.abc import Awaitable, Callable, Sequence
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import ConfigDict

from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from benchmarks.external import (
    BENCHMARK_NAMES,
    ExternalConfig,
    ExternalTaskSelection,
    fixed_external_root,
    load_external_config,
    load_external_lock,
    verify_external_snapshot,
)
from benchmarks.external.deepresearchbench import DeepResearchBenchAdapter
from benchmarks.external.frames import FramesAdapter
from benchmarks.external.livedrbench import LiveDRBenchAdapter
from benchmarks.processes.agent import AgentRunReceipt, AgentVariantRunRequest
from benchmarks.processes.evaluator import materialize_agent_runtime_task
from deepresearch.domain import ResourceUsage
from experiments.config import FormalExperimentConfig, code_tree_sha256
from experiments.models import (
    ExperimentTaskRun,
    ExperimentVariant,
    RankerComponentVariant,
    SealedModel,
    canonical_sha256,
)

ExternalBenchmark = Literal["livedrbench", "frames", "deepresearchbench"]


class ExternalExperimentResult(SealedModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    portfolio_group_id: str
    formal_config_sha256: str
    external_lock_sha256: str
    benchmark_counts: dict[str, int]
    runs: tuple[ExperimentTaskRun, ...]
    metrics_artifact_sha256: str

    @staticmethod
    def _hash(value: str) -> str:
        if len(value) != 64 or value != value.lower() or value == "0" * 64:
            raise ValueError("external result hash must be a non-zero SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError("external result hash must be a non-zero SHA-256") from error
        return value

    def model_post_init(self, __context: object, /) -> None:
        del __context
        self._hash(self.formal_config_sha256)
        self._hash(self.external_lock_sha256)
        self._hash(self.metrics_artifact_sha256)
        if any(value < 0 for value in self.benchmark_counts.values()):
            raise ValueError("external benchmark counts must be non-negative")


class ExternalExperimentRunner:
    """Run external tasks without widening the canonical agent request.

    ``launch_agent`` is dependency-injected for offline tests.  A default
    launcher uses the existing Core child process; it receives only the same
    sanitized ``AgentVariantRunRequest`` as internal experiments.
    """

    def __init__(
        self,
        launch_agent: Callable[[object], object | Awaitable[object]] | None = None,
        evaluator: Callable[..., object | Awaitable[object]] | None = None,
        *,
        repo_root: Path | None = None,
        experiment_root: Path = Path("experiments"),
        config_source: Path | None = None,
        preflight: bool = True,
    ) -> None:
        self._launch_agent = launch_agent or self._default_launch_agent
        self._evaluator = evaluator
        self.repo_root = (repo_root or Path.cwd()).resolve()
        experiment_base = Path(experiment_root)
        self.experiment_root = (
            experiment_base
            if experiment_base.is_absolute()
            else self.repo_root / experiment_base
        )
        source = Path(config_source) if config_source else None
        self.config_source = (
            source if source is None or source.is_absolute() else self.repo_root / source
        )
        self.preflight = preflight

    @staticmethod
    def _assert_hash(value: str, *, label: str) -> str:
        if len(value) != 64 or value != value.lower() or value == "0" * 64:
            raise ValueError(f"{label} is not a valid SHA-256")
        try:
            int(value, 16)
        except ValueError as error:
            raise ValueError(f"{label} is not a valid SHA-256") from error
        return value

    def _group_root(self, config: FormalExperimentConfig) -> Path:
        group = config.experiment_group_id()
        configured_lexical = Path(self.experiment_root).absolute()
        current = configured_lexical
        while current != Path(current.anchor):
            if current.is_symlink():
                raise ValueError("external experiment root contains a symlink")
            current = current.parent
        expected_root = (self.repo_root / "experiments").resolve()
        configured_root = configured_lexical.resolve()
        if configured_root != expected_root:
            raise ValueError("external experiment root must be the repository experiments root")
        root = configured_root / group
        if group in {"", ".", ".."} or Path(group).name != group:
            raise ValueError("external experiment group contains traversal")
        root.mkdir(parents=True, exist_ok=True)
        for name in (
            "config",
            "agent-inputs",
            "requests",
            "candidate-pools",
            "artifacts",
            "external",
        ):
            child = root / name
            if child.is_symlink():
                raise ValueError("external experiment root contains a symlink")
            child.mkdir(parents=True, exist_ok=True)
        return root

    def _repo_file(self, path: Path, *, label: str) -> Path:
        lexical = Path(path)
        if ".." in lexical.parts:
            raise ValueError(f"{label} contains traversal")
        candidate = lexical.absolute()
        try:
            relative = candidate.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError(f"{label} must be inside the repository") from error
        if not relative.parts:
            raise ValueError(f"{label} must name a file")
        current = self.repo_root
        for part in relative.parts:
            current = current / part
            if current.is_symlink():
                raise ValueError(f"{label} contains a symlink")
        if not candidate.is_file():
            raise FileNotFoundError(candidate)
        return candidate

    @staticmethod
    def _write_no_replace(path: Path, payload: bytes) -> Path:
        if path.exists() or path.is_symlink():
            if path.is_file() and path.read_bytes() == payload:
                return path
            raise FileExistsError(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        return path

    async def _call_launcher(self, request: AgentVariantRunRequest) -> object:
        if self._launch_agent is None:
            raise RuntimeError("external agent launcher is unavailable")
        result = self._launch_agent(request)
        if inspect.isawaitable(result):
            return await cast(Awaitable[object], result)
        return result

    def _formal_config_payload(self, config: FormalExperimentConfig) -> bytes:
        """Return the sealed source bytes while binding them to ``config``.

        A caller that only has a model (as in offline tests) receives the
        canonical JSON representation.  The CLI supplies ``config_source``;
        in that case preserve its exact YAML bytes so the result hash names
        the same artifact that was reviewed and sealed.
        """

        if self.config_source is None:
            return canonical_json_bytes(config.model_dump(mode="json"))
        source = Path(self.config_source)
        if ".." in source.parts:
            raise ValueError("sealed formal config source contains traversal")
        raw_source = source.absolute()
        try:
            raw_source.relative_to(self.repo_root)
        except ValueError as error:
            raise ValueError("sealed formal config source must be inside the repository") from error
        current = raw_source
        while current != Path(current.anchor):
            if current.is_symlink():
                raise ValueError("sealed formal config source contains a symlink")
            current = current.parent
        source = raw_source
        if source.is_symlink() or not source.is_file():
            raise ValueError("sealed formal config source is unavailable")
        try:
            payload = source.read_bytes()
            parsed = FormalExperimentConfig.model_validate(yaml.safe_load(payload))
        except (OSError, TypeError, ValueError, yaml.YAMLError) as error:
            raise ValueError("sealed formal config source is invalid") from error
        if parsed.model_dump(mode="json") != config.model_dump(mode="json"):
            raise ValueError("sealed formal config disagrees with runner config")
        return payload

    async def _default_launch_agent(self, request: AgentVariantRunRequest) -> object:
        """Launch the existing sanitized agent entrypoint.

        The child receives the same canonical request used by internal runs;
        notably, no external raw-root path is placed in its environment.
        """

        run_root = Path(request.run_dir).resolve()
        request_payload = canonical_json_bytes(request.model_dump(mode="json"))
        request_path: Path | None = None
        request_root = run_root / "requests"
        for candidate in sorted(request_root.glob("*.json")):
            if candidate.is_file() and not candidate.is_symlink() and candidate.read_bytes() == request_payload:
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
        environment["DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT"] = str(
            (self.repo_root / "benchmarks" / "snapshots" / "external").resolve()
        )
        environment["DEEPRESEARCH_BENCHMARK_RUN_ROOT"] = str(run_root)
        for name, value in os.environ.items():
            if name.startswith("DEEPRESEARCH_PROVIDER_"):
                environment[name] = value
        command = [
            sys.executable,
            "-m",
            "benchmarks.processes.agent",
            "--request",
            str(request_path),
            "--receipt",
            str(receipt_path),
        ]
        completed = await asyncio.to_thread(
            subprocess.run,
            command,
            cwd=self.repo_root,
            env=environment,
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0 or not receipt_path.is_file():
            raise RuntimeError("agent process failed")
        return AgentRunReceipt.model_validate_json(receipt_path.read_bytes(), strict=True)

    @staticmethod
    def _adapter(
        benchmark: ExternalBenchmark,
        *,
        lock_path: Path,
        raw_root: Path,
        snapshot_root: Path,
        external_config: ExternalConfig,
    ) -> object:
        adapter_type = {
            "livedrbench": LiveDRBenchAdapter,
            "frames": FramesAdapter,
            "deepresearchbench": DeepResearchBenchAdapter,
        }[benchmark]
        return adapter_type(
            lock_file=lock_path,
            raw_root=raw_root,
            snapshot_root=snapshot_root,
            external_config=external_config,
        )

    @staticmethod
    def _candidate_pool(
        *,
        task: RuntimeTask,
        snapshot: object,
        root: Path,
        key: str,
    ) -> tuple[Path, str]:
        records = getattr(snapshot, "records", ())
        evidence_ids = sorted(str(record.evidence_id) for record in records)
        payload = canonical_json_bytes(
            {
                "candidate_pool_version": "external-frames-v1",
                "evidence_ids": evidence_ids,
                "task_id": task.task_id,
            }
        )
        path = root / "candidate-pools" / f"{key}.json"
        ExternalExperimentRunner._write_no_replace(path, payload)
        return path, sha256_bytes(payload)

    @staticmethod
    def _failed_run(
        *,
        config: FormalExperimentConfig,
        group_root: Path,
        benchmark: ExternalBenchmark,
        task: RuntimeTask,
        key: str,
        seed: int,
        error_code: str,
    ) -> ExperimentTaskRun:
        if benchmark == "frames":
            variant: ExperimentVariant | RankerComponentVariant = RankerComponentVariant.R2
            protocol = "ranker_component"
            planner_id, ranker_id = "P1", "R2"
        else:
            variant = ExperimentVariant.D
            protocol = "end_to_end"
            planner_id, ranker_id = "P2", "R2"
        return ExperimentTaskRun(
            task_id=task.task_id,
            protocol=protocol,
            variant=variant,
            planner_id=planner_id,
            ranker_id=ranker_id,
            budget_preset=task.request.budget_preset,
            seed=seed,
            status="failed",
            validity="invalid",
            error_code=error_code,
            manifest_path=str((group_root / "external" / benchmark / f"{key}.run-manifest.json").resolve()),
            artifact_ids=(),
            usage=ResourceUsage.zero(cost_known=False),
            pricing_snapshot_ids=(config.pricing_snapshot.snapshot_id,),
            pricing_status="estimated",
            cost_label="estimated_from_normalized_schedule",
            category=task.category,
            metrics={},
        )

    async def run(
        self,
        *,
        config: FormalExperimentConfig,
        external_config_path: Path,
        external_lock_path: Path,
        benchmarks: Sequence[ExternalBenchmark],
    ) -> ExternalExperimentResult:
        requested = tuple(benchmarks)
        if not requested or len(set(requested)) != len(requested) or not set(requested) <= set(BENCHMARK_NAMES):
            raise ValueError("INVALID_REQUEST: external benchmark selection is invalid")
        if config.external_config_sha256 is None or config.external_lock_sha256 is None or not config.external_runtime_task_hashes:
            raise ValueError("INVALID_REQUEST: Portfolio external authorization is absent")
        try:
            config_path = self._repo_file(external_config_path, label="external config")
            lock_path = self._repo_file(external_lock_path, label="external lock")
            expected_config_path = self.repo_root / "benchmarks" / "configs" / "external.yaml"
            expected_lock_path = self.repo_root / "benchmarks" / "external" / "external.lock.json"
            if (config_path, lock_path) != (expected_config_path, expected_lock_path):
                raise ValueError("external config/lock must use the fixed repository paths")
        except (OSError, ValueError) as error:
            raise ValueError("INVALID_REQUEST: external config/lock is unavailable") from error
        self._assert_hash(config.external_config_sha256, label="external config hash")
        self._assert_hash(config.external_lock_sha256, label="external lock hash")
        if sha256_bytes(config_path.read_bytes()) != config.external_config_sha256:
            raise ValueError("INVALID_REQUEST: external config hash mismatch")
        if sha256_bytes(lock_path.read_bytes()) != config.external_lock_sha256:
            raise ValueError("INVALID_REQUEST: external lock hash mismatch")
        external = load_external_config(config_path)
        load_external_lock(lock_path)
        if (
            external.raw_root,
            external.documents_staging_root,
            external.snapshot_root,
        ) != (
            "benchmarks/private/external/raw",
            "benchmarks/private/external/staging",
            "benchmarks/snapshots/external",
        ):
            raise ValueError("INVALID_REQUEST: external roots are not fixed")
        raw_root = fixed_external_root(self.repo_root / external.raw_root, kind="raw")
        snapshot_root = fixed_external_root(self.repo_root / external.snapshot_root, kind="snapshot")
        if self.preflight:
            try:
                if code_tree_sha256(self.repo_root) != config.code_tree_sha256:
                    raise ValueError("Portfolio code tree hash mismatch")
            except (OSError, RuntimeError, ValueError) as error:
                raise ValueError("INVALID_REQUEST: Portfolio code tree is not sealed") from error
        group_root = self._group_root(config)
        config_payload = self._formal_config_payload(config)
        staged_config = group_root / "config" / "formal.yaml"
        self._write_no_replace(staged_config, config_payload)
        config_hash = sha256_bytes(config_payload)
        adapters = {
            benchmark: self._adapter(
                benchmark,
                lock_path=lock_path,
                raw_root=raw_root,
                snapshot_root=snapshot_root,
                external_config=external,
            )
            for benchmark in requested
        }
        all_runs: list[ExperimentTaskRun] = []
        counts: dict[str, int] = {}
        seed = config.replication.seed_values[0]
        for benchmark in requested:
            adapter = adapters[benchmark]
            selections = cast(Any, adapter).select(
                provider_profile_id=config.provider_profile_id,
                budget_preset=config.budget_preset,
            )
            counts[benchmark] = len(selections)
            benchmark_root = group_root / "external" / benchmark
            benchmark_root.mkdir(parents=True, exist_ok=True)
            for selection in selections:
                if not isinstance(selection, ExternalTaskSelection):
                    raise TypeError("INVALID_REQUEST: adapter returned an invalid selection")
                task = selection.runtime_task
                expected_hash = config.external_runtime_task_hashes.get(task.task_id)
                if expected_hash != canonical_sha256(task.model_dump(mode="json")):
                    raise ValueError("INVALID_REQUEST: external RuntimeTask authorization mismatch")
                snapshot_lock = cast(Any, adapter).snapshot_lock_for(task.task_id)
                snapshot_dir = snapshot_root / snapshot_lock.snapshot_relative_path
                # ``select`` has already loaded this snapshot.  Reload at the
                # runner boundary to close the TOCTOU gap before any launch.
                snapshot = verify_external_snapshot(snapshot_lock, snapshot_root=snapshot_root)
                if task.snapshot_id != snapshot.manifest.snapshot_id:
                    raise ValueError("INVALID_SNAPSHOT: RuntimeTask snapshot identity mismatch")
                key = hashlib.sha256(
                    canonical_json_bytes(
                        {
                            "benchmark": benchmark,
                            "group": config.experiment_group_id(),
                            "task_id": task.task_id,
                            "seed": seed,
                        }
                    )
                ).hexdigest()
                staged = materialize_agent_runtime_task(
                    task,
                    agent_input_root=group_root / "agent-inputs",
                    request_id=key[:32],
                    forbidden_private_root=raw_root,
                )
                candidate_path: Path | None = None
                candidate_hash: str | None = None
                if benchmark == "frames":
                    candidate_path, candidate_hash = self._candidate_pool(
                        task=task, snapshot=snapshot, root=group_root, key=key
                    )
                request_payload: dict[str, object] = {
                    "kind": "variant_run",
                    "protocol": "ranker_component" if benchmark == "frames" else "end_to_end",
                    "variant": "R2" if benchmark == "frames" else "D",
                    "task_id": task.task_id,
                    "runtime_task_path": str(staged.resolve()),
                    "runtime_task_sha256": sha256_bytes(staged.read_bytes()),
                    "base_runtime_task_sha256": expected_hash,
                    "snapshot_dir": str(snapshot_dir.resolve()),
                    "run_dir": str(group_root.resolve()),
                    "config_path": str(staged_config.resolve()),
                    "config_sha256": config_hash,
                    "seed_supported": True,
                    "seed": seed,
                    "budget_preset": task.request.budget_preset,
                }
                if candidate_path is not None:
                    request_payload.update(
                        {
                            "candidate_pool_path": str(candidate_path.resolve()),
                            "candidate_pool_sha256": candidate_hash,
                        }
                    )
                request = AgentVariantRunRequest.model_validate(request_payload, strict=True)
                request_path = group_root / "requests" / f"{key}.json"
                self._write_no_replace(request_path, canonical_json_bytes(request.model_dump(mode="json")))
                try:
                    receipt = await self._call_launcher(request)
                    if not isinstance(receipt, AgentRunReceipt):
                        raise TypeError("agent receipt is not canonical")
                    # The Core runner remains the authority for RunManifest
                    # accounting.  External scoring never turns an opaque
                    # receipt into a successful formal record here.
                    error_code = "EXTERNAL_RECEIPT_DEFERRED"
                except Exception:  # noqa: BLE001 - external failures are auditable
                    error_code = "AGENT_LAUNCH_FAILED"
                all_runs.append(
                    self._failed_run(
                        config=config,
                        group_root=group_root,
                        benchmark=benchmark,
                        task=task,
                        key=key,
                        seed=seed,
                        error_code=error_code,
                    )
                )
        metrics_payload = {
            "schema_version": "external-metrics-v1",
            "portfolio_group_id": config.experiment_group_id(),
            "formal_config_sha256": sha256_bytes(config_payload),
            "external_lock_sha256": config.external_lock_sha256,
            "benchmark_counts": counts,
            "runs": [run.model_dump(mode="json") for run in all_runs],
        }
        metrics_bytes = canonical_json_bytes(metrics_payload)
        metrics_path = group_root / "external" / "metrics.json"
        self._write_no_replace(metrics_path, metrics_bytes)
        return ExternalExperimentResult(
            portfolio_group_id=config.experiment_group_id(),
            formal_config_sha256=sha256_bytes(config_payload),
            external_lock_sha256=config.external_lock_sha256,
            benchmark_counts=counts,
            runs=tuple(all_runs),
            metrics_artifact_sha256=sha256_bytes(metrics_bytes),
        )


__all__ = ["ExternalBenchmark", "ExternalExperimentResult", "ExternalExperimentRunner"]
