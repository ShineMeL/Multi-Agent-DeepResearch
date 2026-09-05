from __future__ import annotations

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
        self.runtime_root = Path(runtime_root).resolve()
        self.snapshot_root = Path(snapshot_root).resolve()
        self.run_root = Path(run_root).resolve()

    def _resolve(self, path: Path, root: Path, *, label: str) -> Path:
        resolved = Path(path).resolve()
        if not _inside(resolved, root):
            raise GoldAccessViolation(f"path is outside allowed {label} root")
        return resolved

    def resolve_runtime_task(self, path: Path) -> Path:
        resolved = self._resolve(path, self.runtime_root, label="runtime")
        if not resolved.is_file():
            raise GoldAccessViolation("runtime task file does not exist")
        return resolved

    def resolve_snapshot(self, path: Path) -> Path:
        resolved = self._resolve(path, self.snapshot_root, label="snapshot")
        if not resolved.is_dir():
            raise GoldAccessViolation("snapshot directory does not exist")
        return resolved

    def resolve_output(self, path: Path) -> Path:
        resolved = self._resolve(path, self.run_root, label="run output")
        if resolved == self.run_root:
            raise GoldAccessViolation("output path must be a file")
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
