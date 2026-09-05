from __future__ import annotations

import json
from pathlib import Path

import pytest

from benchmarks.datasets.isolation import GoldAccessViolation, GoldIsolationGuard
from benchmarks.datasets.models import AnnotatedQuestion, RuntimeTask
from benchmarks.processes.evaluator import materialize_agent_runtime_task

EXAMPLE = Path(__file__).parents[2] / ".." / "benchmarks" / "datasets" / "templates" / "question.example.json"


def _runtime_task() -> RuntimeTask:
    question = AnnotatedQuestion.model_validate_json(EXAMPLE.resolve().read_bytes())
    return GoldIsolationGuard.runtime_view(question)


def test_runtime_view_serializes_no_gold_fields() -> None:
    payload = _runtime_task().model_dump(mode="json")
    assert set(payload) == {
        "task_id",
        "category",
        "request",
        "evaluation_cutoff",
        "snapshot_id",
        "corpus_version",
        "index_version",
    }
    assert "gold" not in repr(payload).casefold()
    assert "rubric" not in repr(payload).casefold()


def test_agent_process_cannot_resolve_private_gold(tmp_path: Path) -> None:
    guard = GoldIsolationGuard(
        runtime_root=tmp_path / "runtime",
        snapshot_root=tmp_path / "snapshots",
        private_root=tmp_path / "private",
    )
    with pytest.raises(GoldAccessViolation, match="private benchmark path"):
        guard.assert_agent_readable(tmp_path / "private" / "gold" / "test.jsonl")


def test_runtime_manifest_rejects_private_absolute_path(tmp_path: Path) -> None:
    guard = GoldIsolationGuard(
        runtime_root=tmp_path / "runtime",
        snapshot_root=tmp_path / "snapshots",
        private_root=tmp_path / "private",
    )
    with pytest.raises(GoldAccessViolation):
        guard.validate_run_payload({"snapshot_dir": str(tmp_path / "private" / "gold")})


def test_runtime_manifest_rejects_nested_gold_fields(tmp_path: Path) -> None:
    guard = GoldIsolationGuard(
        runtime_root=tmp_path / "runtime",
        snapshot_root=tmp_path / "snapshots",
        private_root=tmp_path / "private",
    )
    with pytest.raises(GoldAccessViolation, match="gold field"):
        guard.validate_run_payload({"config": {"rubric": {"coverage": 1}}})


def test_evaluator_materializes_one_public_task_outside_private_root(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    staged = materialize_agent_runtime_task(
        _runtime_task(),
        agent_input_root=tmp_path / "agent-inputs",
        request_id="request-1",
        forbidden_private_root=private_root,
    )
    assert private_root.resolve() not in staged.resolve().parents
    assert set(json.loads(staged.read_text(encoding="utf-8"))) == {
        "task_id",
        "category",
        "request",
        "evaluation_cutoff",
        "snapshot_id",
        "corpus_version",
        "index_version",
    }


def test_materialization_rejects_private_destination(tmp_path: Path) -> None:
    private_root = tmp_path / "private"
    private_root.mkdir()
    with pytest.raises(GoldAccessViolation):
        materialize_agent_runtime_task(
            _runtime_task(),
            agent_input_root=private_root / "agent-inputs",
            request_id="request-1",
            forbidden_private_root=private_root,
        )
