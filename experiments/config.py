"""Evaluator-side sealing and preflight of result-affecting experiment inputs."""

from __future__ import annotations

import json
import os
import re
import subprocess
import uuid
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Annotated, Literal, Self

import yaml
from pydantic import AnyHttpUrl, Field, field_serializer, field_validator, model_validator

from benchmarks.datasets.models import (
    DatasetManifest,
    PrivateDatasetManifest,
    RuntimeTask,
    TaskCategory,
)
from benchmarks.datasets.validator import DatasetValidator, canonical_json_bytes, sha256_bytes
from benchmarks.scripts.build_snapshot import (
    _publish_no_replace,  # pyright: ignore[reportPrivateUsage]
)
from deepresearch.providers.embeddings import EmbeddingModelLock
from deepresearch.runtime.manifest import PricingSnapshot
from experiments.models import (
    BudgetPreset,
    DecodingConfig,
    InferenceEnvironmentLock,
    ModelSnapshotLock,
    RelativePath,
    ReplicationConfig,
    Revision,
    SealedModel,
    Sha256,
    canonical_sha256,
    immutable_hash_map,
)

TASK_LIST_FIELDS = (
    "main_test_task_ids",
    "stability_task_ids",
    "cost_subset_task_ids",
    "p0_task_ids",
    "oracle_task_ids",
)


class FormalExperimentTemplate(SealedModel):
    dataset_id: str
    dataset_version: str
    evaluation_timestamp: datetime
    model_id: str
    model_revision: Revision
    model_lock_path: RelativePath
    base_url: AnyHttpUrl
    provider_id: str
    provider_profile_id: str
    endpoint_type: Literal["openai_compatible_chat_completions"]
    decoding: DecodingConfig
    replication: ReplicationConfig
    prompt_version: str
    writer_prompt_version: str
    judge_model_id: str
    judge_model_revision: Revision
    judge_model_lock_path: RelativePath
    judge_prompt_version: str
    r1_model_id: str
    r1_model_revision: Revision
    r1_model_lock_path: RelativePath
    ranker_weights_version: str
    serving_runtime: Literal["vllm"]
    serving_runtime_version: str
    serving_runtime_platform: str
    serving_runtime_artifact_sha256: Sha256
    serving_environment_lock_path: RelativePath
    budget_preset: BudgetPreset
    budget_sensitivity_presets: Annotated[tuple[BudgetPreset, ...], Field(min_length=1)]
    snapshot_collection_id: str
    corpus_version: str
    index_version: str
    evaluator_version: str
    pricing_status: Literal["estimated"]
    pricing_snapshot: PricingSnapshot

    @model_validator(mode="after")
    def validate_template(self) -> Self:
        if (
            self.base_url.username
            or self.base_url.password
            or self.base_url.query
            or self.base_url.fragment
        ):
            raise ValueError("base_url must not contain credentials, query or fragment")
        if self.evaluation_timestamp.tzinfo is None:
            raise ValueError("evaluation_timestamp must be timezone aware")
        pricing = self.pricing_snapshot
        if (pricing.provider_id, pricing.endpoint_type, pricing.model_id) != (
            self.provider_id,
            self.endpoint_type,
            self.model_id,
        ):
            raise ValueError("pricing identity must equal provider/endpoint/model identity")
        if (
            len(set(self.budget_sensitivity_presets)) != len(self.budget_sensitivity_presets)
            or self.budget_preset not in self.budget_sensitivity_presets
        ):
            raise ValueError("budget presets must be unique and contain the primary budget")
        if not self.dataset_id or any(
            c not in "abcdefghijklmnopqrstuvwxyz0123456789_-" for c in self.dataset_id
        ):
            raise ValueError("dataset_id must be a canonical directory identifier")
        return self


class FormalExperimentConfig(FormalExperimentTemplate):
    private_manifest_sha256: Sha256
    model_snapshot_sha256: Sha256
    judge_model_snapshot_sha256: Sha256
    r1_model_snapshot_sha256: Sha256
    serving_environment_sha256: Sha256
    code_tree_sha256: Sha256
    internal_runtime_task_hashes: dict[str, str]
    main_test_task_ids: tuple[str, ...]
    stability_task_ids: tuple[str, ...]
    cost_subset_task_ids: tuple[str, ...]
    p0_task_ids: tuple[str, ...]
    oracle_task_ids: tuple[str, ...]
    external_config_sha256: Sha256 | None = None
    external_lock_sha256: Sha256 | None = None
    external_runtime_task_hashes: dict[str, str] = Field(default_factory=dict)

    _maps = field_validator("internal_runtime_task_hashes", "external_runtime_task_hashes")(
        immutable_hash_map
    )

    @field_serializer("internal_runtime_task_hashes", "external_runtime_task_hashes")
    def serialize_hashes(self, value: dict[str, str]) -> dict[str, str]:
        return dict(value)

    @model_validator(mode="after")
    def validate_seal(self) -> Self:
        main = set(self.main_test_task_ids)
        for name in TASK_LIST_FIELDS:
            values: tuple[str, ...] = getattr(self, name)
            if (
                not values
                or len(set(values)) != len(values)
                or not set(values) <= main
                or any(re.fullmatch(r"test-[a-z0-9][a-z0-9_-]*", v) is None for v in values)
            ):
                raise ValueError("internal task lists must be non-empty, unique and test-only")
        if self.stability_task_ids != self.cost_subset_task_ids:
            raise ValueError("stability_task_ids must equal cost_subset_task_ids")
        if set(self.internal_runtime_task_hashes) != main:
            raise ValueError("internal runtime hash map must exactly cover all test tasks")
        external = (
            self.external_config_sha256 is not None,
            self.external_lock_sha256 is not None,
            bool(self.external_runtime_task_hashes),
        )
        if any(external) and not all(external):
            raise ValueError("external authorization fields are all-or-none")
        if set(self.external_runtime_task_hashes) & main:
            raise ValueError("task may occur in exactly one authorization map")
        if any(
            re.fullmatch(
                r"ext-(?:livedrbench|frames|deepresearchbench)-[a-zA-Z0-9][a-zA-Z0-9_.-]*",
                task,
            )
            is None
            for task in self.external_runtime_task_hashes
        ):
            raise ValueError(
                "external task IDs must use canonical ext-benchmark-stable-id namespaces"
            )
        return self

    def experiment_group_id(self) -> str:
        return canonical_sha256(self.model_dump(mode="json"))[:16]


def write_immutable(path: Path, payload: bytes) -> None:
    """Publish a complete file atomically; never replace even byte-identical output."""
    path = Path(path)
    if path.exists() or path.is_symlink():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.staging")
    try:
        with staging.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        _publish_no_replace(staging, path)
    finally:
        if staging.exists():
            staging.unlink()


def load_template(path: Path) -> FormalExperimentTemplate:
    return FormalExperimentTemplate.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_config(path: Path) -> FormalExperimentConfig:
    return FormalExperimentConfig.model_validate(yaml.safe_load(path.read_text(encoding="utf-8")))


def _within(root: Path, relative: str) -> Path:
    from experiments.models import relative_path

    relative_path(relative)
    path = root / relative
    if not path.resolve(strict=True).is_relative_to(root.resolve(strict=True)):
        raise ValueError("input escapes its fixed root")
    # Reject redirects even if they lead back inside the allowed tree.
    if any(part.is_symlink() for part in (path, *path.parents) if part != root.parent):
        raise ValueError("symlink input is forbidden")
    return path


def code_tree_sha256(repo_root: Path) -> str:
    output = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=repo_root,
        check=True,
        capture_output=True,
    ).stdout
    paths = sorted(set(output.decode("utf-8").split("\0")) - {""})
    entries: list[tuple[str, str]] = []
    for relative in paths:
        path = Path(relative)
        include = relative in {"pyproject.toml", "uv.lock", "models/embedding.lock.json"}
        include |= relative.startswith(("src/", "apps/"))
        include |= (
            relative.startswith("experiments/") and len(path.parts) == 2 and path.suffix == ".py"
        )
        include |= relative.startswith("benchmarks/") and (
            path.suffix == ".py" or relative.startswith("benchmarks/configs/")
        )
        # A Portfolio seal is bound to the exact external source/snapshot
        # lock.  The lock is evaluator metadata (not raw payload), so include
        # it in the result-affecting tree hash.
        include |= relative == "benchmarks/external/external.lock.json"
        if relative.startswith(
            ("benchmarks/private/", "benchmarks/snapshots/", "benchmarks/results/")
        ):
            include = False
        if (
            relative.startswith("benchmarks/configs/formal")
            and path.suffix == ".yaml"
            and relative != "benchmarks/configs/formal.template.yaml"
        ):
            include = False
        if include:
            source = _within(repo_root, relative)
            entries.append((relative, sha256_bytes(source.read_bytes())))
    if not entries:
        raise ValueError("result-affecting code tree is empty")
    return canonical_sha256(entries)


def _model_hash(repo_root: Path, path: str, model_id: str, revision: str) -> str:
    payload = _within(repo_root, path).read_bytes()
    if json.loads(payload).get("schema_version") == "embedding-model-lock-v1":
        embedding = EmbeddingModelLock.model_validate_json(payload)
        identity = embedding.model_id, embedding.revision
        digest = embedding.snapshot_sha256
    else:
        lock = ModelSnapshotLock.model_validate_json(payload)
        identity = lock.repository_id, lock.requested_revision
        digest = lock.snapshot_sha256
    if identity != (model_id, revision):
        raise ValueError("model lock identity mismatch")
    return digest


def validate_base_task(task: RuntimeTask, config: FormalExperimentTemplate) -> None:
    request = task.request
    if (
        request.execution_mode,
        request.access_profile,
        request.run_purpose,
        request.provider_profile_id,
        request.budget_preset,
    ) != ("hybrid", "local", "benchmark", config.provider_profile_id, config.budget_preset):
        raise ValueError("base RuntimeTask request must match sealed primary budget/profile")
    external_hashes = getattr(config, "external_runtime_task_hashes", {})
    # External Portfolio snapshots intentionally have their own corpus
    # version. Their canonical task hash is sealed in the external map and
    # the runner has already verified the matching frozen snapshot. Internal
    # tasks continue to require the primary corpus/index identity.
    if task.task_id in external_hashes:
        if not task.task_id.startswith("ext-"):
            raise ValueError("external RuntimeTask namespace is invalid")
        return
    if (task.corpus_version, task.index_version) != (config.corpus_version, config.index_version):
        raise ValueError("RuntimeTask corpus/index identity mismatch")


def freeze_config(
    template: FormalExperimentTemplate,
    *,
    repo_root: Path,
    private_root: Path,
    output_path: Path | None = None,
    external_config_path: Path | None = None,
    external_lock_path: Path | None = None,
) -> FormalExperimentConfig:
    """Evaluator-only: seal from verified fixed dataset roots, never caller-selected task files."""
    template = FormalExperimentTemplate.model_validate(template.model_dump(mode="json"))
    public_root = repo_root / "benchmarks" / "datasets" / template.dataset_id
    snapshots = repo_root / "benchmarks" / "snapshots" / template.dataset_id
    private_path = _within(private_root, "private_manifest.json")
    public_path = _within(public_root, "public_manifest.json")
    public = DatasetManifest.model_validate_json(public_path.read_bytes())
    private_bytes = private_path.read_bytes()
    private = PrivateDatasetManifest.model_validate_json(private_bytes)
    if sha256_bytes(private_bytes) != public.private_manifest_sha256:
        raise ValueError("private manifest hash mismatch")
    if (public.dataset_id, public.version) != (template.dataset_id, template.dataset_version) or (
        private.dataset_id,
        private.version,
    ) != (public.dataset_id, public.version):
        raise ValueError("dataset identity mismatch")
    # Resolve all fixed inputs before the shared validator can open them.
    expected_test = [f"runtime/test/{category.value}.jsonl" for category in TaskCategory]
    expected_dev = [f"runtime/dev/{category.value}.jsonl" for category in TaskCategory]
    if (
        private.private_test_runtime_files != expected_test
        or private.public_runtime_files != expected_dev
        or public.public_runtime_files != expected_dev
    ):
        raise ValueError("runtime files must name the fixed category roots")
    for category in TaskCategory:
        _within(private_root, f"batches/{category.value}.jsonl")
    for relative in expected_test:
        _within(private_root, relative)
    for relative in expected_dev:
        _within(public_root, relative)
    for task_id in private.snapshot_manifest_sha256:
        if re.fullmatch(r"(dev|test)-[a-z0-9][a-z0-9_-]*", task_id) is None:
            raise ValueError("noncanonical internal task ID")
        _within(snapshots, f"{task_id}/manifest.sha256")
    # Validate runtime projections against private annotations and snapshot hashes using Task 1.
    report = DatasetValidator(snapshot_root=snapshots).validate_dataset(
        private_path, public_manifest_path=public_path
    )
    if not report.valid:
        raise ValueError("frozen dataset verification failed: " + "; ".join(report.errors))
    tasks: dict[str, str] = {}
    for relative in private.private_test_runtime_files:
        if not relative.startswith("runtime/test/") or Path(relative).suffix != ".jsonl":
            raise ValueError("test RuntimeTask files must use the fixed runtime/test root")
        for line in _within(private_root, relative).read_bytes().splitlines():
            task = RuntimeTask.model_validate_json(line, strict=True)
            validate_base_task(task, template)
            if task.task_id in tasks or task.task_id not in private.main_test_task_ids:
                raise ValueError("duplicate or unauthorized internal RuntimeTask")
            tasks[task.task_id] = canonical_sha256(task.model_dump(mode="json"))
    model_hash = _model_hash(
        repo_root, template.model_lock_path, template.model_id, template.model_revision
    )
    judge_hash = _model_hash(
        repo_root,
        template.judge_model_lock_path,
        template.judge_model_id,
        template.judge_model_revision,
    )
    r1_hash = _model_hash(
        repo_root, template.r1_model_lock_path, template.r1_model_id, template.r1_model_revision
    )
    environment = InferenceEnvironmentLock.model_validate_json(
        _within(repo_root, template.serving_environment_lock_path).read_bytes()
    )
    runtime = next((d for d in environment.distributions if d.name == "vllm"), None)
    if (
        environment.model_snapshot_sha256 != model_hash
        or runtime is None
        or (runtime.version, runtime.artifact_sha256)
        != (template.serving_runtime_version, template.serving_runtime_artifact_sha256)
    ):
        raise ValueError("serving environment/model/runtime identity mismatch")
    external_values: dict[str, object] = {
        "external_config_sha256": None,
        "external_lock_sha256": None,
        "external_runtime_task_hashes": {},
    }
    external_pair = (external_config_path is not None, external_lock_path is not None)
    if any(external_pair) and not all(external_pair):
        raise ValueError("external config and lock must be supplied together")
    if all(external_pair):
        # Keep optional Portfolio inputs out of the primary import path.  The
        # adapters only read the fixed, hash-checked local roots and expose a
        # canonical RuntimeTask; the evaluator plan is intentionally ignored
        # when producing the authorization hash map.
        from benchmarks.external import BENCHMARK_NAMES, load_external_config, load_external_lock
        from benchmarks.external.deepresearchbench import DeepResearchBenchAdapter
        from benchmarks.external.frames import FramesAdapter
        from benchmarks.external.livedrbench import LiveDRBenchAdapter

        try:
            if external_config_path is None or external_lock_path is None:
                raise ValueError("external config and lock must be supplied together")
            external_source_raw = Path(external_config_path)
            external_lock_raw = Path(external_lock_path)
            if ".." in external_source_raw.parts or ".." in external_lock_raw.parts:
                raise ValueError("external config/lock contains traversal")
            external_source = external_source_raw.absolute()
            external_lock_source = external_lock_raw.absolute()
            external_source.relative_to(Path(repo_root).absolute())
            external_lock_source.relative_to(Path(repo_root).absolute())
            expected_external_source = Path(repo_root).absolute() / "benchmarks" / "configs" / "external.yaml"
            expected_external_lock = Path(repo_root).absolute() / "benchmarks" / "external" / "external.lock.json"
            if (external_source, external_lock_source) != (
                expected_external_source,
                expected_external_lock,
            ):
                raise ValueError("external config/lock must use the fixed repository paths")
        except ValueError as error:
            raise ValueError("external config/lock must be inside the repository") from error
        external = load_external_config(external_source)
        external_lock = load_external_lock(external_lock_source)
        if (
            external.raw_root,
            external.documents_staging_root,
            external.snapshot_root,
        ) != (
            "benchmarks/private/external/raw",
            "benchmarks/private/external/staging",
            "benchmarks/snapshots/external",
        ):
            raise ValueError("external roots are not fixed")
        # Parse once at the seal boundary so a malformed aggregate cannot be
        # hidden by an adapter's benchmark-specific lookup.
        _ = external_lock
        adapters = {
            "livedrbench": LiveDRBenchAdapter,
            "frames": FramesAdapter,
            "deepresearchbench": DeepResearchBenchAdapter,
        }
        external_hashes: dict[str, str] = {}
        for benchmark in BENCHMARK_NAMES:
            adapter = adapters[benchmark](
                lock_file=external_lock_source,
                raw_root=Path(repo_root) / external.raw_root,
                snapshot_root=Path(repo_root) / external.snapshot_root,
                external_config=external,
                repo_root=Path(repo_root),
            )
            selections = adapter.select(
                provider_profile_id=template.provider_profile_id,
                budget_preset=template.budget_preset,
            )
            if len(selections) != external.spec(benchmark).expected_count:
                raise ValueError("external selection count is not sealed")
            for selection in selections:
                task = selection.runtime_task
                request = task.request
                if (
                    request.execution_mode,
                    request.access_profile,
                    request.run_purpose,
                    request.provider_profile_id,
                    request.budget_preset,
                ) != (
                    "hybrid",
                    "local",
                    "benchmark",
                    template.provider_profile_id,
                    template.budget_preset,
                ):
                    raise ValueError("external RuntimeTask request identity mismatch")
                snapshot_lock = adapter.snapshot_lock_for(task.task_id)
                if (
                    task.snapshot_id,
                    task.corpus_version,
                    task.index_version,
                ) != (
                    snapshot_lock.snapshot_id,
                    snapshot_lock.corpus_version,
                    snapshot_lock.index_version,
                ):
                    raise ValueError("external RuntimeTask snapshot identity mismatch")
                if task.task_id in external_hashes:
                    raise ValueError("duplicate external RuntimeTask ID")
                external_hashes[task.task_id] = canonical_sha256(task.model_dump(mode="json"))
        expected_external_count = sum(
            external.spec(name).expected_count for name in BENCHMARK_NAMES
        )
        if len(external_hashes) != expected_external_count:
            raise ValueError("external RuntimeTask authorization must cover the full Portfolio")
        external_values = {
            "external_config_sha256": sha256_bytes(external_source.read_bytes()),
            "external_lock_sha256": sha256_bytes(external_lock_source.read_bytes()),
            "external_runtime_task_hashes": dict(sorted(external_hashes.items())),
        }
    config = FormalExperimentConfig.model_validate(
        {
            **template.model_dump(mode="json"),
            **{name: getattr(private, name) for name in TASK_LIST_FIELDS},
            "private_manifest_sha256": public.private_manifest_sha256,
            "model_snapshot_sha256": model_hash,
            "judge_model_snapshot_sha256": judge_hash,
            "r1_model_snapshot_sha256": r1_hash,
            "serving_environment_sha256": environment.environment_sha256,
            "code_tree_sha256": code_tree_sha256(repo_root),
            "internal_runtime_task_hashes": dict(sorted(tasks.items())),
            **external_values,
        }
    )
    if output_path is not None:
        write_immutable(output_path, canonical_json_bytes(config.model_dump(mode="json")))
    return config


def preflight_config(
    config: FormalExperimentConfig,
    *,
    repo_root: Path,
    private_root: Path,
    external_config_path: Path | None = None,
    external_lock_path: Path | None = None,
) -> None:
    payload = config.model_dump(mode="json")
    template = FormalExperimentTemplate.model_validate(
        {
            key: value
            for key, value in payload.items()
            if key in FormalExperimentTemplate.model_fields
        }
    )
    if config.external_config_sha256 is not None:
        external_config_path = external_config_path or (
            repo_root / "benchmarks" / "configs" / "external.yaml"
        )
        external_lock_path = external_lock_path or (
            repo_root / "benchmarks" / "external" / "external.lock.json"
        )
    expected = freeze_config(
        template,
        repo_root=repo_root,
        private_root=private_root,
        external_config_path=external_config_path,
        external_lock_path=external_lock_path,
    )
    external_fields = {
        "external_config_sha256",
        "external_lock_sha256",
        "external_runtime_task_hashes",
    }
    if config.model_dump(mode="json", exclude=external_fields) != expected.model_dump(
        mode="json", exclude=external_fields
    ):
        raise ValueError("sealed config no longer matches verified inputs")
    # The external fields are part of the formal seal too.  Comparing only the
    # common/internal template above would allow a caller to provide a valid
    # internal seal with a partial map, a different lock, or an evaluator-plan
    # hash.  ``freeze_config`` recomputes all three values from the full
    # 10/20/10 adapter portfolio, so exact equality is required here.
    if (
        config.external_config_sha256,
        config.external_lock_sha256,
        config.external_runtime_task_hashes,
    ) != (
        expected.external_config_sha256,
        expected.external_lock_sha256,
        expected.external_runtime_task_hashes,
    ):
        raise ValueError("sealed external Portfolio authorization does not match verified inputs")


def authorized_staged_task(
    config: FormalExperimentConfig,
    task: RuntimeTask,
    *,
    staged_sha256: str,
    budget_preset: BudgetPreset,
) -> RuntimeTask:
    """Validate a staged budget arm by reconstructing exactly its authorized base object."""
    from benchmarks.datasets.isolation import GoldAccessViolation

    maps: tuple[Mapping[str, str], ...] = (
        config.internal_runtime_task_hashes,
        config.external_runtime_task_hashes,
    )
    matches = [mapping[task.task_id] for mapping in maps if task.task_id in mapping]
    if (
        len(matches) != 1
        or canonical_sha256(task.model_dump(mode="json")) != staged_sha256
        or task.request.budget_preset != budget_preset
        or budget_preset not in config.budget_sensitivity_presets
    ):
        raise GoldAccessViolation("staged RuntimeTask hash/budget/authorization mismatch")
    base = task.model_copy(
        update={"request": task.request.model_copy(update={"budget_preset": config.budget_preset})}
    )
    if canonical_sha256(base.model_dump(mode="json")) != matches[0]:
        raise GoldAccessViolation("staged task differs from authorized base beyond budget")
    validate_base_task(base, config)
    return task
