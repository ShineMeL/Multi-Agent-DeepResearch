from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sqlite3
import stat
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, field_validator, model_validator

from benchmarks.datasets.models import RuntimeTask
from benchmarks.datasets.validator import sha256_bytes
from deepresearch.domain import ResourceUsage, RunStatus
from deepresearch.runtime import CheckpointRef


def _require_hash(value: str | None) -> str | None:
    if value is None:
        return None
    if re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
        raise ValueError("request hashes must be non-zero lowercase SHA-256")
    return value


def _require_absolute_path(value: str) -> str:
    if not value or not Path(value).is_absolute() or ".." in Path(value).parts:
        raise ValueError("paths must be absolute and non-traversing")
    return value


class AgentRequestBase(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str
    runtime_task_path: str
    runtime_task_sha256: str
    base_runtime_task_sha256: str
    snapshot_dir: str
    run_dir: str
    config_path: str
    config_sha256: str
    seed_supported: bool
    seed: int | None = None
    repeat_id: Annotated[int | None, Field(ge=1)] = None
    resume_checkpoint_path: str | None = None
    resume_checkpoint_sha256: str | None = None
    resume_checkpoint_ref: CheckpointRef | None = None

    @field_validator(
        "runtime_task_sha256",
        "base_runtime_task_sha256",
        "config_sha256",
        "resume_checkpoint_sha256",
    )
    @classmethod
    def validate_hash(cls, value: str | None) -> str | None:
        return _require_hash(value)

    @model_validator(mode="after")
    def validate_resume_fields(self) -> AgentRequestBase:
        if not self.task_id.strip():
            raise ValueError("request task_id must not be empty")
        for name, path in (
            ("runtime_task_path", self.runtime_task_path),
            ("snapshot_dir", self.snapshot_dir),
            ("run_dir", self.run_dir),
            ("config_path", self.config_path),
        ):
            if not path or not Path(path).is_absolute() or ".." in Path(path).parts:
                raise ValueError(f"request {name} must be an absolute non-traversing path")
        if self.resume_checkpoint_path is not None:
            _require_absolute_path(self.resume_checkpoint_path)
        values = (
            self.resume_checkpoint_path,
            self.resume_checkpoint_sha256,
            self.resume_checkpoint_ref,
        )
        if any(item is not None for item in values) and not all(item is not None for item in values):
            raise ValueError("resume checkpoint fields must be all present or all absent")
        if self.seed is not None and self.repeat_id is not None:
            raise ValueError("seed and repeat_id are mutually exclusive")
        if self.seed is None and self.repeat_id is None:
            raise ValueError("exactly one seed or repeat_id is required")
        if type(self.seed_supported) is not bool:
            raise ValueError("seed_supported must be a boolean")
        if self.seed_supported and self.seed is None:
            raise ValueError("seed-supported request requires a seed")
        if not self.seed_supported and self.repeat_id is None:
            raise ValueError("unseeded request requires a repeat_id")
        return self


class AgentCandidatePoolRequest(AgentRequestBase):
    kind: Literal["candidate_pool"] = "candidate_pool"
    protocol: Literal["ranker_component"] = "ranker_component"
    planner_id: Literal["P1"] = "P1"
    budget_preset: Literal["low", "medium", "high"]


class AgentVariantRunRequest(AgentRequestBase):
    kind: Literal["variant_run"] = "variant_run"
    protocol: Literal["ranker_component", "planner_policy", "end_to_end", "reference"]
    variant: Literal["A", "B", "C", "D", "P0", "R0", "R1", "R2"]
    budget_preset: Literal["low", "medium", "high"]
    candidate_pool_path: str | None = None
    candidate_pool_sha256: str | None = None

    @field_validator("candidate_pool_path")
    @classmethod
    def validate_candidate_pool_path(cls, value: str | None) -> str | None:
        return None if value is None else _require_absolute_path(value)

    @field_validator("candidate_pool_sha256")
    @classmethod
    def validate_candidate_pool_hash(cls, value: str | None) -> str | None:
        return _require_hash(value)

    @model_validator(mode="after")
    def validate_pool_contract(self) -> AgentVariantRunRequest:
        ranker_variant = self.variant in {"R0", "R1", "R2"}
        if (self.protocol == "ranker_component") != ranker_variant:
            raise ValueError("ranker protocol and R variant must agree")
        if self.protocol == "reference" and self.variant != "P0":
            raise ValueError("reference protocol accepts only P0")
        if self.protocol in {"planner_policy", "end_to_end"} and self.variant not in {
            "A",
            "B",
            "C",
            "D",
        }:
            raise ValueError("planner/end-to-end protocols accept only A/B/C/D")
        requires = ranker_variant
        if requires != (self.candidate_pool_path is not None and self.candidate_pool_sha256 is not None):
            raise ValueError("ranker variants require exactly one candidate pool binding")
        return self


type AgentRunRequest = Annotated[
    AgentCandidatePoolRequest | AgentVariantRunRequest,
    Field(discriminator="kind"),
]
AgentRunRequestAdapter: TypeAdapter[AgentCandidatePoolRequest | AgentVariantRunRequest] = TypeAdapter(
    AgentRunRequest
)


class AgentCandidatePoolReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    candidate_pool_path: str = Field(min_length=1)
    candidate_pool_sha256: str
    evidence_ids_sha256: str
    manifest_path: str = Field(min_length=1)
    manifest_sha256: str
    usage: ResourceUsage
    pricing_snapshot_ids: tuple[str, ...] = Field(min_length=1)

    @field_validator("candidate_pool_path", "manifest_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _require_absolute_path(value)

    @field_validator(
        "candidate_pool_sha256", "evidence_ids_sha256", "manifest_sha256"
    )
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if re.fullmatch(r"[0-9a-f]{64}", value) is None or value == "0" * 64:
            raise ValueError("receipt hashes must be non-zero lowercase SHA-256")
        return value

    @field_validator("pricing_snapshot_ids")
    @classmethod
    def validate_pricing_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if any(not item.strip() for item in value):
            raise ValueError("pricing snapshot IDs must be non-empty")
        return value


class AgentRunReceipt(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    task_id: str = Field(min_length=1)
    status: RunStatus
    error_code: str | None = None
    run_result_path: str = Field(min_length=1)
    manifest_path: str = Field(min_length=1)
    run_result_sha256: str
    manifest_sha256: str
    artifact_ids: tuple[str, ...]

    @field_validator("run_result_path", "manifest_path")
    @classmethod
    def validate_paths(cls, value: str) -> str:
        return _require_absolute_path(value)

    @field_validator("run_result_sha256", "manifest_sha256")
    @classmethod
    def validate_hash(cls, value: str) -> str:
        if _require_hash(value) is None:
            raise ValueError("receipt hashes must be non-zero lowercase SHA-256")
        return value

    @field_validator("artifact_ids")
    @classmethod
    def validate_artifact_ids(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if len(set(value)) != len(value):
            raise ValueError("receipt artifact IDs must be unique")
        if any(_require_hash(item) is None for item in value):
            raise ValueError("receipt artifact IDs must be content-addressed hashes")
        return value


def _canonical_json(value: object) -> bytes:
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        .encode("utf-8")
        + b"\n"
    )


def _is_link_or_reparse(path: Path) -> bool:
    try:
        details = path.lstat()
    except FileNotFoundError:
        return False
    reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
    return stat.S_ISLNK(details.st_mode) or bool(
        getattr(details, "st_file_attributes", 0) & reparse_flag
    )


def _assert_snapshot_file(path: Path, *, label: str, regular_file: bool = True) -> None:
    absolute = path.absolute()
    current = absolute
    while current != Path(current.anchor):
        if _is_link_or_reparse(current):
            raise ValueError(f"snapshot {label} contains a symlink or reparse point")
        current = current.parent
    if _is_link_or_reparse(current):
        raise ValueError(f"snapshot {label} contains a symlink or reparse point")
    if regular_file and (not absolute.exists() or not absolute.is_file()):
        raise ValueError(f"snapshot {label} is missing")


def _verify_snapshot(snapshot_dir: Path, *, task: RuntimeTask | None = None) -> None:
    _assert_snapshot_file(snapshot_dir, label="root", regular_file=False)
    manifest_path = snapshot_dir / "manifest.sha256"
    _assert_snapshot_file(manifest_path, label="manifest")
    manifest = json.loads(manifest_path.read_bytes())
    if not isinstance(manifest, dict):
        raise TypeError("snapshot manifest is invalid")
    raw_manifest = cast("dict[object, object]", manifest)
    if set(raw_manifest) != {"file_sha256"}:
        raise ValueError("snapshot manifest schema is invalid")
    file_hashes = raw_manifest.get("file_sha256")
    if not isinstance(file_hashes, dict) or not file_hashes:
        raise ValueError("snapshot manifest has no file hashes")
    typed_hashes = cast("dict[object, object]", file_hashes)
    required = {"documents.jsonl", "index.json", "snapshot.json"}
    if task is not None and set(typed_hashes) != required:
        raise ValueError("snapshot manifest file set is invalid")
    for filename, expected in typed_hashes.items():
        if not isinstance(filename, str) or Path(filename).name != filename:
            raise ValueError("snapshot manifest contains an unsafe filename")
        if (
            not isinstance(expected, str)
            or len(expected) != 64
            or expected != expected.lower()
            or any(character not in "0123456789abcdef" for character in expected)
        ):
            raise ValueError("snapshot manifest contains an invalid hash")
        path = snapshot_dir / filename
        _assert_snapshot_file(path, label=filename)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise ValueError(f"snapshot hash mismatch: {filename}")
    _assert_snapshot_file(snapshot_dir / "snapshot.json", label="snapshot.json")
    snapshot_payload = json.loads((snapshot_dir / "snapshot.json").read_bytes())
    if not isinstance(snapshot_payload, dict):
        raise TypeError("snapshot metadata is invalid")
    typed_snapshot = cast("dict[str, object]", snapshot_payload)
    if task is not None:
        for name in ("task_id", "snapshot_id", "corpus_version", "index_version"):
            if typed_snapshot.get(name) != getattr(task, name):
                raise ValueError(f"snapshot {name} does not match RuntimeTask")
    for filename, field in (("documents.jsonl", "documents_sha256"), ("index.json", "index_sha256")):
        expected = typed_snapshot.get(field)
        if expected is not None and expected != hashlib.sha256((snapshot_dir / filename).read_bytes()).hexdigest():
            raise ValueError(f"snapshot {field} does not match file")


def verify_checkpoint_identity(path: Path, ref: CheckpointRef) -> None:
    """Prove the requested checkpoint tuple exists before opening Core state."""
    from benchmarks.datasets.isolation import GoldAccessViolation

    try:
        uri = f"file:{path.as_posix()}?mode=ro"
        with sqlite3.connect(uri, uri=True) as connection:
            columns = {
                str(row[1])
                for row in connection.execute("PRAGMA table_info(checkpoints)")
            }
            if not {"thread_id", "checkpoint_id"}.issubset(columns):
                raise GoldAccessViolation("checkpoint identity source is invalid")
            if "checkpoint_ns" in columns:
                cursor = connection.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ? "
                    "AND checkpoint_ns = '' LIMIT 1",
                    (ref.thread_id, ref.checkpoint_id),
                )
            else:
                cursor = connection.execute(
                    "SELECT 1 FROM checkpoints WHERE thread_id = ? AND checkpoint_id = ? LIMIT 1",
                    (ref.thread_id, ref.checkpoint_id),
                )
            if cursor.fetchone() is None:
                raise GoldAccessViolation("checkpoint identity is not present in verified source")
    except GoldAccessViolation:
        raise
    except (OSError, sqlite3.Error) as error:
        raise GoldAccessViolation("checkpoint identity source is invalid") from error


def _verify_checkpoint_identity(path: Path, ref: CheckpointRef) -> None:  # pyright: ignore[reportUnusedFunction]
    """Compatibility alias for the focused isolation tests."""
    verify_checkpoint_identity(path, ref)


def _write_probe(path: Path, payload: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{os.getpid()}.staging")
    data = _canonical_json(payload)
    try:
        with staging.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(staging, path)
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def load_authorized_agent_inputs(
    request: AgentRunRequest,
    *,
    guard: object,
) -> tuple[Any, Any]:
    """Validate every evaluator-supplied input before importing agent code.

    The imports of ``FormalExperimentConfig`` and the component factory are
    deliberately local: malformed or untrusted requests must not reach a
    provider/client construction path.
    """
    from benchmarks.datasets.isolation import AgentRuntimeGuard, GoldAccessViolation
    from benchmarks.datasets.models import RuntimeTask

    if not isinstance(guard, AgentRuntimeGuard):
        raise TypeError("guard must be an AgentRuntimeGuard")
    if Path(request.run_dir).absolute() != guard.run_root:
        raise GoldAccessViolation("request run_dir must equal the guarded run root")
    if request.seed_supported:
        if request.seed is None or request.repeat_id is not None:
            raise GoldAccessViolation("seed-supported request has invalid replication identity")
    elif request.seed is not None or request.repeat_id is None:
        raise GoldAccessViolation("unseeded request has invalid replication identity")
    task_path = guard.resolve_runtime_task(Path(request.runtime_task_path))
    snapshot_dir = guard.resolve_snapshot(Path(request.snapshot_dir))
    config_path = guard.resolve_staged_config(
        Path(request.config_path), expected_sha256=request.config_sha256
    )
    task_bytes = task_path.read_bytes()
    if sha256_bytes(task_bytes) != request.runtime_task_sha256:
        raise GoldAccessViolation("runtime task hash mismatch")
    task = RuntimeTask.model_validate_json(task_bytes, strict=True)
    guard.validate_payload(task.model_dump(mode="json"))
    if task.task_id != request.task_id:
        raise GoldAccessViolation("runtime task identity mismatch")
    _verify_snapshot(snapshot_dir, task=task)

    from experiments.config import FormalExperimentConfig, authorized_staged_task

    config_payload = yaml.safe_load(config_path.read_bytes())
    config = FormalExperimentConfig.model_validate(config_payload)
    if request.task_id != task.task_id:
        raise GoldAccessViolation("request task identity mismatch")
    if request.budget_preset != task.request.budget_preset:
        raise GoldAccessViolation("request budget does not match staged RuntimeTask")
    authorized_staged_task(
        config,
        task,
        staged_sha256=request.runtime_task_sha256,
        budget_preset=request.budget_preset,
    )
    if (
        request.base_runtime_task_sha256 != config.internal_runtime_task_hashes.get(task.task_id)
        and request.base_runtime_task_sha256
        != config.external_runtime_task_hashes.get(task.task_id)
    ):
        raise GoldAccessViolation("base runtime task authorization mismatch")

    if (
        isinstance(request, AgentVariantRunRequest)
        and request.candidate_pool_path is not None
        and request.candidate_pool_sha256 is not None
    ):
        guard.resolve_candidate_pool(
            Path(request.candidate_pool_path), expected_sha256=request.candidate_pool_sha256
        )
    if request.resume_checkpoint_path is not None and request.resume_checkpoint_sha256 is not None:
        checkpoint = guard.resolve_resume_checkpoint(
            Path(request.resume_checkpoint_path), expected_sha256=request.resume_checkpoint_sha256
        )
        if not checkpoint.is_file() or request.resume_checkpoint_ref is None:
            raise GoldAccessViolation("checkpoint identity is incomplete")
        verify_checkpoint_identity(checkpoint, request.resume_checkpoint_ref)
    return task, config


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m benchmarks.processes.agent")
    parser.add_argument("--probe-runtime-task", type=Path)
    parser.add_argument("--request", type=Path)
    parser.add_argument("--receipt", type=Path)
    parser.add_argument("--runtime-root", type=Path)
    parser.add_argument("--snapshot-dir", type=Path)
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--output", type=Path)
    return parser


def _write_json_no_replace(path: Path, payload: object) -> None:
    data = _canonical_json(payload)
    absolute = path.absolute()
    current = absolute.parent
    while current != Path(current.anchor):
        if _is_link_or_reparse(current):
            raise ValueError("agent artifact parent contains a symlink or reparse point")
        current = current.parent
    if _is_link_or_reparse(current):
        raise ValueError("agent artifact parent contains a symlink or reparse point")
    if _is_link_or_reparse(absolute):
        raise ValueError("agent artifact path is a symlink or reparse point")
    if absolute.exists() or absolute.is_symlink():
        if absolute.is_file() and absolute.read_bytes() == data:
            return
        raise FileExistsError(absolute)
    absolute.parent.mkdir(parents=True, exist_ok=True)
    staging = absolute.with_name(f".{absolute.name}.{os.getpid()}.staging")
    try:
        with staging.open("xb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        from benchmarks.scripts.build_snapshot import (  # pyright: ignore[reportPrivateUsage]
            _publish_no_replace,  # pyright: ignore[reportPrivateUsage]
        )

        _publish_no_replace(staging, absolute)  # pyright: ignore[reportPrivateUsage]
    finally:
        try:
            staging.unlink()
        except FileNotFoundError:
            pass


def _run_request(args: argparse.Namespace) -> int:
    from benchmarks.datasets.isolation import AgentRuntimeGuard, GoldAccessViolation

    if args.receipt is None:
        raise ValueError("--receipt is required for a typed request")
    def environment_root(name: str) -> Path:
        raw = os.environ.get(name)
        if raw is None or not raw.strip():
            raise GoldAccessViolation("agent root environment is incomplete")
        path = Path(raw)
        if not path.is_absolute() or ".." in path.parts:
            raise GoldAccessViolation("agent root environment is invalid")
        return path

    runtime_root = environment_root("DEEPRESEARCH_BENCHMARK_RUNTIME_ROOT")
    snapshot_root = environment_root("DEEPRESEARCH_BENCHMARK_SNAPSHOT_ROOT")
    run_root = environment_root("DEEPRESEARCH_BENCHMARK_RUN_ROOT")
    guard = AgentRuntimeGuard(
        runtime_root=runtime_root,
        snapshot_root=snapshot_root,
        run_root=run_root,
    )
    request_path = guard.resolve_request(args.request)
    raw = request_path.read_bytes()
    request = AgentRunRequestAdapter.validate_json(raw, strict=True)
    guard.validate_payload(request.model_dump(mode="json"))
    task, config = load_authorized_agent_inputs(request, guard=guard)
    # Only after all path/hash/budget authorization does the agent import the
    # formal component boundary.  The lightweight implementation writes a
    # deterministic public receipt; Task 16 supplies strict replay execution.
    from deepresearch.runtime.ports import ResearchRunner as _ResearchRunner
    from experiments import factories as _component_factories

    del _ResearchRunner, _component_factories
    del config
    output_root = guard.resolve_output(Path(args.receipt))
    usage = ResourceUsage.zero(cost_known=True)
    if isinstance(request, AgentCandidatePoolRequest):
        # Candidate pools are an evaluator-promoted input.  The child may
        # publish only into its writable staging subtree; it must never write
        # directly into the shared protected pool directory.
        candidate_root = guard.resolve_output(guard.run_root / "staging")
        candidate_root.mkdir(parents=True, exist_ok=True)
        candidate = guard.resolve_output(
            candidate_root / f"{request.task_id}-{request.budget_preset}-candidate-pool.json"
        )
        candidate_payload: dict[str, object] = {
            "task_id": request.task_id,
            "evidence_ids": [],
            "candidate_pool_version": "formal-v1",
        }
        _write_json_no_replace(candidate, candidate_payload)
        candidate_sha = sha256_bytes(candidate.read_bytes())
        evidence_sha = hashlib.sha256(_canonical_json([])).hexdigest()
        manifest = guard.resolve_output(
            guard.run_root / "artifacts" / f"{request.task_id}-candidate-manifest.json"
        )
        _write_json_no_replace(manifest, {"kind": "candidate_pool", "task_id": request.task_id})
        manifest_sha = sha256_bytes(manifest.read_bytes())
        receipt = AgentCandidatePoolReceipt(
            task_id=request.task_id,
            candidate_pool_path=str(candidate.resolve()),
            candidate_pool_sha256=candidate_sha,
            evidence_ids_sha256=evidence_sha,
            manifest_path=str(manifest.resolve()),
            manifest_sha256=manifest_sha,
            usage=usage,
            pricing_snapshot_ids=("formal-local-accounting",),
        )
        _write_json_no_replace(output_root, receipt.model_dump(mode="json"))
        return 0

    variant_request = request
    result_path = guard.resolve_output(
        guard.run_root
        / "artifacts"
        / f"{variant_request.task_id}-{variant_request.variant}-result.json"
    )
    manifest_path = guard.resolve_output(
        guard.run_root
        / "artifacts"
        / f"{variant_request.task_id}-{variant_request.variant}-manifest.json"
    )
    _write_json_no_replace(result_path, {"task_id": task.task_id, "status": "completed"})
    _write_json_no_replace(manifest_path, {"task_id": task.task_id, "status": "completed"})
    receipt = AgentRunReceipt(
        task_id=variant_request.task_id,
        status="completed",
        run_result_path=str(result_path.resolve()),
        manifest_path=str(manifest_path.resolve()),
        run_result_sha256=sha256_bytes(result_path.read_bytes()),
        manifest_sha256=sha256_bytes(manifest_path.read_bytes()),
        artifact_ids=(),
    )
    _write_json_no_replace(output_root, receipt.model_dump(mode="json"))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # This is intentionally the first non-stdlib operation: an agent process
    # must reject a gold environment before importing benchmark/domain code.
    from benchmarks.datasets.isolation import (
        AgentRuntimeGuard,
        GoldAccessViolation,
        assert_agent_environment,
    )
    from benchmarks.datasets.models import RuntimeTask

    try:
        assert_agent_environment(os.environ)
    except GoldAccessViolation as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 3

    try:
        args = _parser().parse_args(argv)
        if args.request is not None:
            return _run_request(args)
        if args.probe_runtime_task is None or args.runtime_root is None or args.snapshot_dir is None or args.run_root is None or args.output is None:
            raise ValueError("probe arguments are incomplete")
        guard = AgentRuntimeGuard(
            runtime_root=args.runtime_root,
            snapshot_root=args.snapshot_dir,
            run_root=args.run_root,
        )
        task_path = guard.resolve_runtime_task(args.probe_runtime_task)
        snapshot_dir = guard.resolve_snapshot(args.snapshot_dir)
        output_path = guard.resolve_output(args.output)
        task = RuntimeTask.model_validate_json(task_path.read_bytes(), strict=True)
        guard.validate_payload(task.model_dump(mode="json"))
        _verify_snapshot(snapshot_dir)
        _write_probe(
            output_path,
            {
                "probe_status": "ok",
                "snapshot_id": task.snapshot_id,
                "task_id": task.task_id,
            },
        )
        return 0
    except GoldAccessViolation as error:
        print(f"{error.code}: {error}", file=sys.stderr)
        return 3
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        print("SNAPSHOT_OR_RUNTIME_INVALID", file=sys.stderr)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "AgentCandidatePoolReceipt",
    "AgentCandidatePoolRequest",
    "AgentRequestBase",
    "AgentRunReceipt",
    "AgentRunRequest",
    "AgentVariantRunRequest",
    "load_authorized_agent_inputs",
    "main",
    "verify_checkpoint_identity",
]
