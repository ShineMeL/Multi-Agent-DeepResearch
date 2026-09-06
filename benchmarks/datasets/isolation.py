from __future__ import annotations

import hashlib
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import cast

from pydantic import JsonValue

from .models import AnnotatedQuestion, RuntimeTask

_FORBIDDEN_FIELDS = frozenset(
    {
        "acceptable_claims",
        "gold_evidence_spans",
        "gold_claim_links",
        "rubric",
        "private_root",
        "gold_root",
    }
)


class GoldAccessViolation(RuntimeError):
    def __init__(self, message: str, *, code: str = "GOLD_ACCESS_VIOLATION") -> None:
        super().__init__(message)
        self.code = code


def _inside(path: Path, root: Path) -> bool:
    return path == root or root in path.parents


def _reject_forbidden_payload(value: object, *, private_root: Path | None = None) -> None:
    if isinstance(value, Mapping):
        mapping = cast("Mapping[object, object]", value)
        for key, item in mapping.items():
            if isinstance(key, str) and key.casefold() in _FORBIDDEN_FIELDS:
                raise GoldAccessViolation(f"gold field is not allowed: {key}")
            _reject_forbidden_payload(item, private_root=private_root)
        return
    if isinstance(value, (list, tuple)):
        for item in cast("list[object] | tuple[object, ...]", value):
            _reject_forbidden_payload(item, private_root=private_root)
        return
    if private_root is not None and isinstance(value, str):
        private_text = str(private_root)
        if private_text and private_text.casefold() in value.casefold():
            raise GoldAccessViolation("private benchmark path is evaluator-only")


class GoldIsolationGuard:
    def __init__(
        self,
        runtime_root: Path,
        snapshot_root: Path,
        private_root: Path,
    ) -> None:
        self.runtime_root = Path(runtime_root).resolve()
        self.snapshot_root = Path(snapshot_root).resolve()
        self.private_root = Path(private_root).resolve()

    @staticmethod
    def runtime_view(question: AnnotatedQuestion) -> RuntimeTask:
        if type(question) is not AnnotatedQuestion:
            raise TypeError("question must be an AnnotatedQuestion")
        return RuntimeTask(
            task_id=question.task_id,
            category=question.category,
            request=question.request,
            evaluation_cutoff=question.evaluation_cutoff,
            snapshot_id=question.snapshot_id,
            corpus_version=question.corpus_version,
            index_version=question.index_version,
        )

    def assert_agent_readable(self, path: Path) -> Path:
        resolved = Path(path).resolve()
        if _inside(resolved, self.private_root):
            raise GoldAccessViolation("private benchmark path is evaluator-only")
        if not any(
            _inside(resolved, root) for root in (self.runtime_root, self.snapshot_root)
        ):
            raise GoldAccessViolation("path is outside runtime benchmark root")
        return resolved

    def validate_run_payload(self, payload: JsonValue) -> None:
        _reject_forbidden_payload(payload, private_root=self.private_root)


class AgentRuntimeGuard:
    """Positive allow-list used inside the agent; it never receives private_root."""

    def __init__(
        self,
        *,
        runtime_root: Path,
        snapshot_root: Path,
        run_root: Path,
    ) -> None:
        # Keep the lexical identity supplied by the child environment.  Calling
        # ``resolve`` here would follow a root symlink before the resolver can
        # reject it, allowing an attacker to substitute an otherwise allowed
        # tree.  Roots are trusted only when their complete existing prefix is
        # free of links/reparse points.
        self.runtime_root = self._root(Path(runtime_root), label="runtime")
        self.snapshot_root = self._root(Path(snapshot_root), label="snapshot")
        self.run_root = self._root(Path(run_root), label="run")

    @classmethod
    def _root(cls, path: Path, *, label: str) -> Path:
        absolute = path.absolute()
        if ".." in absolute.parts:
            raise GoldAccessViolation(f"{label} root contains traversal")
        current = absolute
        while current != Path(current.anchor):
            if cls._is_link_or_reparse(current):
                raise GoldAccessViolation(f"allowed {label} root is a symlink")
            current = current.parent
        if cls._is_link_or_reparse(current):
            raise GoldAccessViolation(f"allowed {label} root is a symlink")
        return absolute

    @staticmethod
    def _is_link_or_reparse(path: Path) -> bool:
        try:
            details = path.lstat()
        except FileNotFoundError:
            return False
        reparse_flag = getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0)
        return stat.S_ISLNK(details.st_mode) or bool(
            getattr(details, "st_file_attributes", 0) & reparse_flag
        )

    def _resolve(
        self,
        path: Path,
        root: Path,
        *,
        label: str,
        regular_file: bool | None = None,
        suffix: str | None = None,
    ) -> Path:
        candidate = Path(path)
        if ".." in candidate.parts:
            raise GoldAccessViolation(f"path traversal is not allowed for {label} input")
        # ``resolve`` alone follows a redirecting child symlink.  Check the
        # lexical chain first so an attacker cannot substitute an allowed
        # input between validation and open.
        absolute = candidate.absolute()
        try:
            relative = absolute.relative_to(root)
        except ValueError:
            raise GoldAccessViolation(f"path is outside allowed {label} root")
        current = root
        if self._is_link_or_reparse(current):
            raise GoldAccessViolation(f"allowed {label} root is a symlink")
        for part in relative.parts:
            current = current / part
            if self._is_link_or_reparse(current):
                raise GoldAccessViolation(f"symlink is not allowed for {label} input")
        resolved = absolute.resolve(strict=False)
        if not _inside(resolved, root):
            raise GoldAccessViolation(f"path is outside allowed {label} root")
        if suffix is not None and resolved.suffix != suffix:
            raise GoldAccessViolation(f"{label} input has an invalid suffix")
        if regular_file is True and (not resolved.exists() or not resolved.is_file()):
            raise GoldAccessViolation(f"{label} file does not exist")
        if regular_file is False and (not resolved.exists() or not resolved.is_dir()):
            raise GoldAccessViolation(f"{label} directory does not exist")
        return resolved

    def resolve_runtime_task(self, path: Path) -> Path:
        return self._resolve(path, self.runtime_root, label="runtime", regular_file=True)

    def resolve_snapshot(self, path: Path) -> Path:
        return self._resolve(path, self.snapshot_root, label="snapshot", regular_file=False)

    def resolve_request(self, path: Path) -> Path:
        root = self.run_root / "requests"
        return self._resolve(path, root, label="request", regular_file=True)

    @staticmethod
    def _verify_hash(path: Path, expected_sha256: str) -> None:
        if (
            type(expected_sha256) is not str
            or len(expected_sha256) != 64
            or expected_sha256 != expected_sha256.lower()
            or expected_sha256 == "0" * 64
            or any(character not in "0123456789abcdef" for character in expected_sha256)
        ):
            raise GoldAccessViolation("input hash is invalid")
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected_sha256:
            raise GoldAccessViolation("input hash mismatch")

    def resolve_staged_config(self, path: Path, *, expected_sha256: str) -> Path:
        expected = self.run_root / "config" / "formal.yaml"
        resolved = self._resolve(path, self.run_root / "config", label="sealed config", regular_file=True)
        if resolved != expected:
            raise GoldAccessViolation("sealed config path is not the staged formal config")
        self._verify_hash(resolved, expected_sha256)
        return resolved

    def resolve_candidate_pool(self, path: Path, *, expected_sha256: str) -> Path:
        resolved = self._resolve(
            path,
            self.run_root / "candidate-pools",
            label="candidate pool",
            regular_file=True,
            suffix=".json",
        )
        if resolved.parent != (self.run_root / "candidate-pools"):
            raise GoldAccessViolation("candidate pool must be a direct run-root input")
        self._verify_hash(resolved, expected_sha256)
        return resolved

    def resolve_resume_checkpoint(self, path: Path, *, expected_sha256: str) -> Path:
        resolved = self._resolve(
            path,
            self.run_root / "resume-checkpoints",
            label="resume checkpoint",
            regular_file=True,
            suffix=".sqlite3",
        )
        if resolved.parent != (self.run_root / "resume-checkpoints"):
            raise GoldAccessViolation("resume checkpoint must be a direct run-root input")
        self._verify_hash(resolved, expected_sha256)
        return resolved

    def resolve_output(self, path: Path) -> Path:
        resolved = self._resolve(path, self.run_root, label="run output")
        if resolved == self.run_root:
            raise GoldAccessViolation("output path must be a file")
        protected = {
            self.run_root / name
            for name in (
                "config",
                "agent-inputs",
                "requests",
                "candidate-pools",
                "resume-checkpoints",
            )
        }
        if any(_inside(resolved, root) for root in protected):
            raise GoldAccessViolation("run output cannot overwrite an authorized input")
        if self._is_link_or_reparse(resolved):
            raise GoldAccessViolation("run output cannot be a symlink")
        return resolved

    def validate_payload(self, payload: JsonValue) -> None:
        _reject_forbidden_payload(payload)


def assert_agent_environment(environment: Mapping[str, str]) -> None:
    if "DEEPRESEARCH_BENCHMARK_GOLD_ROOT" in environment:
        raise GoldAccessViolation(
            "gold environment is forbidden in agent process",
            code="GOLD_ROOT_FORBIDDEN",
        )


__all__ = [
    "AgentRuntimeGuard",
    "GoldAccessViolation",
    "GoldIsolationGuard",
    "assert_agent_environment",
]
