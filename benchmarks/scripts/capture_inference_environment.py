"""Capture actual serving facts; unknown package artifact provenance fails closed."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import subprocess
import sys
import zipfile
from collections.abc import Mapping, Sequence
from email.parser import Parser
from pathlib import Path
from typing import Annotated

import typer
from pydantic import BaseModel, ConfigDict

from benchmarks.datasets.validator import canonical_json_bytes, sha256_bytes
from experiments.config import load_template, write_immutable
from experiments.models import (
    InferenceEnvironmentLock,
    LockedDistribution,
    ModelSnapshotLock,
    canonical_sha256,
)


class _EnvironmentFacts(BaseModel):
    model_config = ConfigDict(extra="forbid")
    python_version: str
    platform: str
    cuda_version: str
    driver_version: str
    gpu_model: str
    distributions: tuple[LockedDistribution, ...]


def _redact_arguments(arguments: Sequence[str]) -> tuple[str, ...]:
    result: list[str] = []
    redact_next = False
    for argument in arguments:
        if redact_next:
            result.append("[REDACTED]")
            redact_next = False
        elif re.match(
            r"(?i)^(?:--)?(?:api[_-]?key|hf[_-]token|access[_-]token|token|password|secret|authorization)(?:=|$)",
            argument,
        ):
            key, separator, _ = argument.partition("=")
            result.append(key + "=[REDACTED]" if separator else key)
            redact_next = not separator
        else:
            argument = re.sub(r"(https?://)[^/@\s]+:[^/@\s]+@", r"\1[REDACTED]@", argument)
            result.append(argument)
    return tuple(result)


def capture_inference_environment(
    *,
    output_path: Path,
    environment: Mapping[str, object],
    launch_arguments: Sequence[str],
    model_snapshot_sha256: str,
) -> InferenceEnvironmentLock:
    facts = _EnvironmentFacts.model_validate(environment)
    if not launch_arguments:
        raise ValueError("actual server launch arguments required")
    payload = facts.model_dump(mode="json")
    payload["distributions"] = [
        item.model_dump(mode="json") for item in sorted(facts.distributions, key=lambda d: d.name)
    ]
    payload.update(
        launch_arguments_sha256=canonical_sha256(_redact_arguments(launch_arguments)),
        model_snapshot_sha256=model_snapshot_sha256,
    )
    lock = InferenceEnvironmentLock.model_validate(
        {**payload, "environment_sha256": canonical_sha256(payload)}
    )
    write_immutable(output_path, canonical_json_bytes(lock.model_dump(mode="json")))
    return lock


def _normalized(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_distributions(artifact_dir: Path | None) -> tuple[LockedDistribution, ...]:
    artifacts: dict[tuple[str, str], str] = {}
    artifact_paths: dict[tuple[str, str], Path] = {}
    if artifact_dir is not None:
        for wheel in sorted(artifact_dir.glob("*.whl")):
            with zipfile.ZipFile(wheel) as archive:
                metadata_names = [
                    n for n in archive.namelist() if n.endswith(".dist-info/METADATA")
                ]
                if len(metadata_names) != 1:
                    raise ValueError("wheel must have exactly one METADATA record")
                message = Parser().parsestr(archive.read(metadata_names[0]).decode("utf-8"))
                identity = _normalized(str(message["Name"])), str(message["Version"])
            digest = sha256_bytes(wheel.read_bytes())
            if identity in artifacts and artifacts[identity] != digest:
                raise ValueError("ambiguous package artifacts")
            artifacts[identity] = digest
            artifact_paths[identity] = wheel
    records: list[LockedDistribution] = []
    for distribution in importlib.metadata.distributions():
        name = _normalized(distribution.metadata["Name"])
        version = distribution.version
        digest = artifacts.get((name, version))
        wheel = artifact_paths.get((name, version))
        if wheel is not None:
            with zipfile.ZipFile(wheel) as archive:
                for member in archive.namelist():
                    if member.endswith(("/", ".dist-info/RECORD")):
                        continue
                    parts = Path(member).parts
                    if not parts or ".." in parts or Path(member).is_absolute():
                        raise ValueError("unsafe artifact member path")
                    installed_member = member
                    if parts[0].endswith(".data"):
                        if len(parts) < 3 or parts[1] not in {"purelib", "platlib"}:
                            raise ValueError("cannot verify relocated installed artifact member")
                        installed_member = "/".join(parts[2:])
                    installed_path = Path(str(distribution.locate_file(installed_member)))
                    if not installed_path.is_file() or installed_path.read_bytes() != archive.read(
                        member
                    ):
                        raise ValueError("installed package bytes do not match artifact")
        direct_url = distribution.read_text("direct_url.json")
        if direct_url:
            payload = json.loads(direct_url)
            provenance = payload.get("archive_info", {}).get("hashes", {}).get("sha256")
            if provenance:
                if digest is not None and digest != provenance:
                    raise ValueError("installed artifact provenance mismatch")
                digest = str(provenance)
        if digest is None:
            raise ValueError(
                f"missing installed artifact SHA-256 for {name}=={version}; supply wheelhouse"
            )
        records.append(LockedDistribution(name=name, version=version, artifact_sha256=digest))
    return tuple(sorted(records, key=lambda item: item.name))


def _command(arguments: list[str]) -> str:
    return subprocess.run(arguments, check=True, capture_output=True, text=True).stdout.strip()


def _server_arguments() -> tuple[str, ...]:
    candidates: list[tuple[str, ...]] = []
    for path in Path("/proc").glob("[0-9]*/cmdline"):
        if path.parent.name == str(os.getpid()):
            continue
        try:
            parts = tuple(p.decode("utf-8") for p in path.read_bytes().split(b"\0") if p)
        except (OSError, UnicodeError):
            continue
        if any(p.endswith("/vllm") or p == "vllm.entrypoints.openai.api_server" for p in parts):
            candidates.append(parts)
    if len(candidates) != 1:
        raise ValueError("exactly one actual vLLM server process required for launch capture")
    return candidates[0]


def main(
    template: Annotated[Path, typer.Option()],
    model_lock: Annotated[Path, typer.Option()],
    output: Annotated[Path, typer.Option()],
    artifact_dir: Annotated[Path | None, typer.Option()] = None,
) -> None:
    config = load_template(template)
    model = ModelSnapshotLock.model_validate_json(model_lock.read_bytes())
    if (model.repository_id, model.requested_revision) != (config.model_id, config.model_revision):
        raise ValueError("model lock identity mismatch")
    facts: dict[str, object] = {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "cuda_version": _command([sys.executable, "-c", "import torch; print(torch.version.cuda)"]),
        "driver_version": _command(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"]
        ),
        "gpu_model": _command(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]),
        "distributions": _installed_distributions(artifact_dir),
    }
    capture_inference_environment(
        output_path=output,
        environment=facts,
        launch_arguments=_server_arguments(),
        model_snapshot_sha256=model.snapshot_sha256,
    )


if __name__ == "__main__":
    typer.run(main)
