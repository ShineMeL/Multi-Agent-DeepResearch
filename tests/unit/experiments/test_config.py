from copy import deepcopy
from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from benchmarks.scripts.capture_inference_environment import capture_inference_environment
from benchmarks.scripts.lock_model import lock_model
from experiments.config import FormalExperimentConfig, FormalExperimentTemplate, canonical_sha256


@pytest.fixture
def valid_payload():
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
            name: ["test-t1"]
            for name in (
                "main_test_task_ids",
                "stability_task_ids",
                "cost_subset_task_ids",
                "p0_task_ids",
                "oracle_task_ids",
            )
        }
    )
    payload["internal_runtime_task_hashes"] = {"test-t1": "b" * 64}
    return payload


def test_template_cannot_run_and_sealed_hash_is_stable(valid_payload):
    template = yaml.safe_load(Path("benchmarks/configs/formal.template.yaml").read_text())
    FormalExperimentTemplate.model_validate(template)
    with pytest.raises(ValidationError):
        FormalExperimentConfig.model_validate(template)
    config = FormalExperimentConfig.model_validate(valid_payload)
    assert config.experiment_group_id() == canonical_sha256(config.model_dump(mode="json"))[:16]
    assert config == FormalExperimentConfig.model_validate_json(config.model_dump_json())
    with pytest.raises(TypeError):
        config.internal_runtime_task_hashes["test-t1"] = "c" * 64


@pytest.mark.parametrize("field", ["model_snapshot_sha256", "code_tree_sha256", "replication"])
def test_missing_sealed_fields_rejected(valid_payload, field):
    valid_payload.pop(field)
    with pytest.raises(ValidationError):
        FormalExperimentConfig.model_validate(valid_payload)


@pytest.mark.parametrize(
    "patch",
    [
        {"api_key": "secret"},
        {"model_snapshot_sha256": "0" * 64},
        {"model_snapshot_sha256": "A" * 64},
        {"model_snapshot_sha256": "bad"},
        {"base_url": "http://user:secret@localhost/v1"},
        {"base_url": "http://localhost/v1?api_key=secret"},
        {"internal_runtime_task_hashes": {}},
        {"stability_task_ids": ["dev-t1"]},
        {"external_config_sha256": "b" * 64},
        {"budget_sensitivity_presets": ["low", "high"]},
        {"budget_sensitivity_presets": ["medium", "medium"]},
        {
            "external_config_sha256": "b" * 64,
            "external_lock_sha256": "c" * 64,
            "external_runtime_task_hashes": {"unnamespaced": "d" * 64},
        },
    ],
)
def test_invalid_seals_rejected(valid_payload, patch):
    with pytest.raises(ValidationError):
        FormalExperimentConfig.model_validate({**valid_payload, **patch})


def test_pricing_identity_and_replication_are_checked(valid_payload):
    bad = deepcopy(valid_payload)
    bad["pricing_snapshot"]["model_id"] = "other"
    with pytest.raises(ValidationError, match="pricing identity"):
        FormalExperimentConfig.model_validate(bad)
    bad = deepcopy(valid_payload)
    bad["replication"]["seed_values"] = [7, 7]
    with pytest.raises(ValidationError):
        FormalExperimentConfig.model_validate(bad)
    base = FormalExperimentConfig.model_validate(valid_payload)
    changed = base.model_copy(
        update={"replication": base.replication.model_copy(update={"seed_values": (7, 8, 9)})}
    )
    budgets = base.model_copy(update={"budget_sensitivity_presets": ("medium", "high")})
    assert (
        len(
            {
                base.experiment_group_id(),
                changed.experiment_group_id(),
                budgets.experiment_group_id(),
            }
        )
        == 3
    )


def test_model_lock_metadata_sorted_verified_and_never_overwritten(tmp_path):
    revision = "1" * 40
    response = {
        "id": "org/model",
        "sha": revision,
        "siblings": [
            {
                "rfilename": "weights.safetensors",
                "size": 100,
                "blobId": "2" * 40,
                "lfs": {"sha256": "3" * 64, "size": 100},
            },
            {"rfilename": "config.json", "size": 10, "blobId": "4" * 40},
        ],
    }
    output = tmp_path / "model.lock.json"
    result = lock_model(
        repository_id="org/model", revision=revision, output_path=output, metadata=response
    )
    assert [item.path for item in result.files] == ["config.json", "weights.safetensors"]
    assert result.files[1].git_blob_or_lfs_oid == "3" * 64
    assert result.snapshot_sha256 == canonical_sha256(
        [item.model_dump(mode="json") for item in result.files]
    )
    with pytest.raises(FileExistsError):
        lock_model(
            repository_id="org/model", revision=revision, output_path=output, metadata=response
        )
    with pytest.raises(ValueError):
        lock_model(
            repository_id="org/model",
            revision="main",
            output_path=tmp_path / "bad.json",
            metadata=response,
        )
    with pytest.raises(ValueError):
        lock_model(
            repository_id="org/model",
            revision="5" * 40,
            output_path=tmp_path / "bad.json",
            metadata=response,
        )
    assert not (tmp_path / "bad.json").exists()


def test_environment_capture_is_deterministic_and_redacts_credentials(tmp_path):
    facts = {
        "python_version": "3.12.10",
        "platform": "linux",
        "cuda_version": "12.8",
        "driver_version": "570",
        "gpu_model": "fixture GPU",
        "distributions": [{"name": "vllm", "version": "0.28.0", "artifact_sha256": "b" * 64}],
    }
    kwargs = {
        "environment": facts,
        "launch_arguments": ["vllm", "--api-key", "secret"],
        "model_snapshot_sha256": "a" * 64,
    }
    first = capture_inference_environment(output_path=tmp_path / "one.json", **kwargs)
    second = capture_inference_environment(output_path=tmp_path / "two.json", **kwargs)
    assert first == second
    assert "secret" not in first.model_dump_json()
    assert (tmp_path / "one.json").read_bytes() == (tmp_path / "two.json").read_bytes()
    with pytest.raises(FileExistsError):
        capture_inference_environment(output_path=tmp_path / "one.json", **kwargs)


def test_staged_budget_arm_preserves_authorized_base(valid_payload):
    from benchmarks.datasets.isolation import GoldAccessViolation, GoldIsolationGuard
    from benchmarks.datasets.models import AnnotatedQuestion, RuntimeTask
    from experiments.config import authorized_staged_task

    question = AnnotatedQuestion.model_validate_json(
        Path("benchmarks/datasets/templates/question.example.json").read_bytes()
    )
    base = GoldIsolationGuard.runtime_view(question).model_copy(
        update={
            "task_id": "test-t1",
            "corpus_version": valid_payload["corpus_version"],
            "index_version": valid_payload["index_version"],
            "request": question.request.model_copy(
                update={
                    "execution_mode": "hybrid",
                    "access_profile": "local",
                    "run_purpose": "benchmark",
                    "provider_profile_id": valid_payload["provider_profile_id"],
                    "budget_preset": "medium",
                }
            ),
        }
    )
    valid_payload["internal_runtime_task_hashes"] = {
        base.task_id: canonical_sha256(base.model_dump(mode="json"))
    }
    config = FormalExperimentConfig.model_validate(valid_payload)
    staged = RuntimeTask.model_validate_json(
        base.model_copy(
            update={"request": base.request.model_copy(update={"budget_preset": "low"})}
        ).model_dump_json()
    )
    assert (
        authorized_staged_task(
            config,
            staged,
            staged_sha256=canonical_sha256(staged.model_dump(mode="json")),
            budget_preset="low",
        )
        == staged
    )
    tampered = staged.model_copy(
        update={"request": staged.request.model_copy(update={"question": "Unsealed question?"})}
    )
    with pytest.raises(GoldAccessViolation):
        authorized_staged_task(
            config,
            tampered,
            staged_sha256=canonical_sha256(tampered.model_dump(mode="json")),
            budget_preset="low",
        )
    with pytest.raises(GoldAccessViolation):
        authorized_staged_task(config, staged, staged_sha256="c" * 64, budget_preset="low")


def test_code_hash_includes_benchmark_python_excludes_sealed_output_and_results(tmp_path):
    import subprocess

    from experiments.config import code_tree_sha256

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "benchmarks/datasets/schema.py"
    source.parent.mkdir(parents=True)
    source.write_text("schema = 1\n")
    template = tmp_path / "benchmarks/configs/formal.template.yaml"
    template.parent.mkdir()
    template.write_text("template: 1\n")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    first = code_tree_sha256(tmp_path)
    (template.parent / "formal.yaml").write_text("sealed: true\n")
    results = tmp_path / "benchmarks/results/data.json"
    results.parent.mkdir()
    results.write_text('{"result": 1}')
    assert code_tree_sha256(tmp_path) == first
    source.write_text("schema = 2\n")
    assert code_tree_sha256(tmp_path) != first


def test_code_hash_excludes_private_raw_python_and_tracked_result_files(tmp_path):
    import subprocess

    from experiments.config import code_tree_sha256

    subprocess.run(["git", "init", "-q", str(tmp_path)], check=True)
    source = tmp_path / "experiments/config.py"
    source.parent.mkdir()
    source.write_text("schema = 1\n")
    first = code_tree_sha256(tmp_path)
    for relative in (
        "benchmarks/private/raw.py",
        "benchmarks/snapshots/generated.py",
        "experiments/group/result.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("private data")
    subprocess.run(["git", "add", "."], cwd=tmp_path, check=True)
    assert code_tree_sha256(tmp_path) == first


def test_capture_rejects_tampered_environment_hash(tmp_path):
    import json

    from experiments.models import InferenceEnvironmentLock

    facts = {
        "python_version": "3.12",
        "platform": "linux",
        "cuda_version": "12.8",
        "driver_version": "570",
        "gpu_model": "fixture",
        "distributions": [{"name": "vllm", "version": "0.28.0", "artifact_sha256": "b" * 64}],
    }
    output = tmp_path / "environment.json"
    capture_inference_environment(
        output_path=output,
        environment=facts,
        launch_arguments=["vllm", "serve"],
        model_snapshot_sha256="a" * 64,
    )
    payload = json.loads(output.read_bytes())
    payload["gpu_model"] = "different GPU"
    with pytest.raises(ValidationError, match="environment hash"):
        InferenceEnvironmentLock.model_validate(payload)


def test_launch_hash_redacts_secrets_but_preserves_tokenizer_configuration(tmp_path):
    facts = {
        "python_version": "3.12",
        "platform": "linux",
        "cuda_version": "12.8",
        "driver_version": "570",
        "gpu_model": "fixture",
        "distributions": [{"name": "vllm", "version": "0.28.0", "artifact_sha256": "b" * 64}],
    }
    hashes = []
    for index, (secret, tokenizer) in enumerate(
        (("one", "tok-a"), ("two", "tok-a"), ("two", "tok-b"))
    ):
        lock = capture_inference_environment(
            output_path=tmp_path / f"{index}.json",
            environment=facts,
            model_snapshot_sha256="a" * 64,
            launch_arguments=["vllm", "--api-key", secret, "--tokenizer", tokenizer],
        )
        hashes.append(lock.launch_arguments_sha256)
    assert hashes[0] == hashes[1]
    assert hashes[1] != hashes[2]


def test_environment_wheelhouse_must_match_installed_package_bytes(tmp_path, monkeypatch):
    import importlib.metadata
    import zipfile

    from benchmarks.scripts.capture_inference_environment import _installed_distributions

    installed = tmp_path / "installed"
    metadata_dir = installed / "fixture-1.0.dist-info"
    metadata_dir.mkdir(parents=True)
    metadata = "Name: fixture\nVersion: 1.0\n"
    (metadata_dir / "METADATA").write_bytes(metadata.encode())
    (installed / "fixture.py").write_bytes(b"version = 2\n")
    wheelhouse = tmp_path / "wheels"
    wheelhouse.mkdir()
    with zipfile.ZipFile(wheelhouse / "fixture-1.0-py3-none-any.whl", "w") as archive:
        archive.writestr("fixture-1.0.dist-info/METADATA", metadata)
        archive.writestr("fixture.py", "version = 1\n")
    distribution = importlib.metadata.PathDistribution(metadata_dir)
    monkeypatch.setattr(importlib.metadata, "distributions", lambda: (distribution,))
    with pytest.raises(ValueError, match="installed.*artifact"):
        _installed_distributions(wheelhouse)
    (installed / "fixture.py").write_bytes(b"version = 1\n")
    assert _installed_distributions(wheelhouse)[0].name == "fixture"


@pytest.fixture
def frozen_config_inputs(tmp_path):
    import json
    import subprocess

    from benchmarks.datasets.builder import DatasetBuilder
    from tests.unit.benchmarks.test_builder import _question, _write_complete_private_dataset

    repo = tmp_path / "repo"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", str(repo)], check=True)
    private = tmp_path / "private"
    payload = yaml.safe_load(Path("benchmarks/configs/formal.template.yaml").read_text())
    payload.update(
        corpus_version=_question().corpus_version, index_version=_question().index_version
    )
    snapshot_root = repo / "benchmarks/snapshots" / payload["dataset_id"]
    _write_complete_private_dataset(private, snapshot_root)
    for batch in (private / "batches").glob("*.jsonl"):
        records = [json.loads(line) for line in batch.read_bytes().splitlines()]
        for record in records:
            record["request"].update(
                execution_mode="hybrid",
                access_profile="local",
                run_purpose="benchmark",
                provider_profile_id=payload["provider_profile_id"],
                budget_preset="medium",
            )
        batch.write_text("".join(json.dumps(record) + "\n" for record in records))
    DatasetBuilder(snapshot_root=snapshot_root).finalize(
        dataset_id=payload["dataset_id"],
        version="1.0.0",
        private_root=private,
        public_root=repo / "benchmarks/datasets" / payload["dataset_id"],
        snapshot_root=snapshot_root,
        subset_seed=20260829,
    )
    model_hash = None
    for prefix in ("", "r1_"):
        model_id = payload[prefix + "model_id"]
        revision = payload[prefix + "model_revision"]
        model = lock_model(
            repository_id=model_id,
            revision=revision,
            output_path=repo / payload[prefix + "model_lock_path"],
            metadata={
                "id": model_id,
                "sha": revision,
                "siblings": [{"rfilename": "config.json", "size": 10, "blobId": "b" * 40}],
            },
        )
        if not prefix:
            model_hash = model.snapshot_sha256
    capture_inference_environment(
        output_path=repo / payload["serving_environment_lock_path"],
        model_snapshot_sha256=model_hash,
        launch_arguments=["vllm", "serve"],
        environment={
            "python_version": "3.12",
            "platform": "linux",
            "cuda_version": "12.8",
            "driver_version": "570",
            "gpu_model": "fixture",
            "distributions": [
                {
                    "name": "vllm",
                    "version": payload["serving_runtime_version"],
                    "artifact_sha256": payload["serving_runtime_artifact_sha256"],
                }
            ],
        },
    )
    template = FormalExperimentTemplate.model_validate(payload)
    (repo / "benchmarks/configs/formal.template.yaml").write_text(template.model_dump_json())
    return repo, private, template


def test_freeze_derives_exact_tasks_verifies_seal_and_refuses_overwrite(frozen_config_inputs):
    from experiments.config import freeze_config, load_config, preflight_config

    repo, private, template = frozen_config_inputs
    output = repo / "benchmarks/configs/formal.yaml"
    config = freeze_config(template, repo_root=repo, private_root=private, output_path=output)
    assert len(config.internal_runtime_task_hashes) == 30
    assert tuple(config.internal_runtime_task_hashes) == config.main_test_task_ids
    assert load_config(output) == config
    assert freeze_config(template, repo_root=repo, private_root=private) == config
    preflight_config(config, repo_root=repo, private_root=private)
    with pytest.raises(FileExistsError):
        freeze_config(template, repo_root=repo, private_root=private, output_path=output)
    with pytest.raises(ValueError):
        freeze_config(
            template.model_copy(update={"provider_profile_id": "wrong-profile"}),
            repo_root=repo,
            private_root=private,
        )
    altered = config.model_copy(update={"model_snapshot_sha256": "d" * 64})
    with pytest.raises(ValueError, match="sealed config"):
        preflight_config(altered, repo_root=repo, private_root=private)


def test_freeze_rejects_modified_private_runtime_and_lock_identity(frozen_config_inputs):
    import json

    from experiments.config import freeze_config

    repo, private, template = frozen_config_inputs
    with pytest.raises(ValueError, match="model lock identity"):
        freeze_config(
            template.model_copy(update={"judge_model_id": "wrong-model"}),
            repo_root=repo,
            private_root=private,
        )
    runtime = next((private / "runtime/test").glob("*.jsonl"))
    lines = runtime.read_bytes().splitlines()
    task = json.loads(lines[0])
    task["request"]["question"] = "Tampered question?"
    runtime.write_bytes(json.dumps(task).encode() + b"\n" + b"\n".join(lines[1:]) + b"\n")
    with pytest.raises(ValueError, match="dataset verification"):
        freeze_config(template, repo_root=repo, private_root=private)
