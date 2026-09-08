"""Validated deployment provenance; zero hashes explicitly mean unavailable."""

import hashlib
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BuildProvenance:
    code_commit: str
    dependency_lock_sha256: str

    def __post_init__(self) -> None:
        if re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", self.code_commit) is None:
            raise ValueError("build code commit must be a lowercase Git object hash")
        if re.fullmatch(r"[0-9a-f]{64}", self.dependency_lock_sha256) is None:
            raise ValueError("build dependency lock must be a lowercase SHA256")


def resolve_build_provenance(repository: Path | None = None) -> BuildProvenance:
    repository = repository or Path(__file__).resolve().parents[3]
    commit = os.environ.get("DEEPRESEARCH_CODE_COMMIT")
    lock = os.environ.get("DEEPRESEARCH_DEPENDENCY_LOCK_SHA256")
    if commit is None and (repository / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=repository,
                check=True,
                capture_output=True,
                text=True,
                timeout=2,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            pass
    lock_path = repository / "uv.lock"
    if lock_path.is_file():
        actual_lock = hashlib.sha256(lock_path.read_bytes()).hexdigest()
        if lock is not None and lock != actual_lock:
            raise ValueError("build dependency lock does not match packaged uv.lock")
        lock = actual_lock
    # Never fabricate a source identity from unrelated source/lock bytes. The
    # all-zero sentinel is stable and documented as unknown, including wheels.
    return BuildProvenance(
        commit if commit is not None else "0" * 40, lock if lock is not None else "0" * 64
    )
