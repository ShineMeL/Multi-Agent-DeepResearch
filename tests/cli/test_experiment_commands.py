from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from apps.cli.main import app
from benchmarks.scripts.render_results import ResultValidationError, render_results


def test_invalid_formal_config_exits_nonzero(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("not: a formal config\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["experiment", "config", "validate", "--config", str(path)],
    )
    assert result.exit_code != 0


def test_cost_sweep_cannot_accept_cli_budget_override(tmp_path: Path) -> None:
    path = tmp_path / "invalid.yaml"
    path.write_text("not: a formal config\n", encoding="utf-8")
    result = CliRunner().invoke(
        app,
        ["experiment", "run-cost-subset", "--config", str(path), "--budgets", "low,high"],
    )
    assert result.exit_code != 0


def test_runner_stages_the_user_config_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.cli import experiment

    source = tmp_path / "formal.yaml"
    source.write_text("sealed: true\n", encoding="utf-8")
    config = SimpleNamespace(dataset_id="fixture")
    monkeypatch.setattr(experiment, "_load_sealed_tasks", lambda config, repo_root: {})
    runner = experiment._runner(config, source)  # type: ignore[arg-type]
    assert runner.config_source == source.resolve()


def test_runner_receives_sealed_task_category_loader(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from apps.cli import experiment

    source = tmp_path / "formal.yaml"
    source.write_text("sealed: true\n", encoding="utf-8")
    config = SimpleNamespace(dataset_id="dataset-v1")
    sealed_tasks = {"task-a": object()}
    monkeypatch.setattr(
        experiment,
        "_load_sealed_tasks",
        lambda loaded, repo_root: sealed_tasks,
        raising=False,
    )

    runner = experiment._runner(config, source)  # type: ignore[arg-type]

    assert runner._task_loader is sealed_tasks


def test_private_manifest_must_be_a_real_file_not_a_symlink(tmp_path: Path) -> None:
    from apps.cli.experiment import _require_regular_file

    target = tmp_path / "private_manifest.json"
    target.write_text("{}", encoding="utf-8")
    link = tmp_path / "manifest-link.json"
    link.symlink_to(target)
    with pytest.raises(ValueError, match="symlink|reparse|regular file"):
        _require_regular_file(link, label="sealed private manifest")


def test_config_freeze_consumes_all_lock_options(tmp_path: Path, monkeypatch) -> None:
    from apps.cli import experiment

    source = tmp_path / "template.yaml"
    private = tmp_path / "private_manifest.json"
    output = tmp_path / "formal.yaml"
    for path in (source, private):
        path.write_text("fixture\n", encoding="utf-8")
    template = SimpleNamespace(
        model_lock_path="models/main.lock.json",
        r1_model_lock_path="models/r1.lock.json",
        serving_environment_lock_path="models/environment.lock.json",
    )
    called = {"freeze": False}

    class FakeConfig:
        def model_dump(self, *, mode: str):
            assert mode == "json"
            return {"fixture": True}

        def experiment_group_id(self) -> str:
            return "fixture-group"

    def fake_freeze(*args, **kwargs):
        called["freeze"] = True
        return FakeConfig()

    monkeypatch.setattr(experiment, "load_template", lambda path: template)
    monkeypatch.setattr(experiment, "freeze_config", fake_freeze)
    result = CliRunner().invoke(
        app,
        [
            "experiment",
            "config",
            "freeze",
            "--source",
            str(source),
            "--private-manifest",
            str(private),
            "--model-lock",
            str(tmp_path / "wrong-main.lock.json"),
            "--r1-model-lock",
            str(tmp_path / "wrong-r1.lock.json"),
            "--serving-environment-lock",
            str(tmp_path / "wrong-environment.lock.json"),
            "--output",
            str(output),
        ],
    )
    assert result.exit_code != 0
    assert not called["freeze"]


def _renderer_summary(**overrides: object) -> dict[str, object]:
    """Small public-only summary used by renderer contract tests."""

    payload: dict[str, object] = {
        "schema_version": "experiment-summary-v1",
        "group_id": "fixture-group",
        "dataset_version": "frozen-ai-cs-60-v1",
        "evaluation_date": "2026-09-07",
        "ranker_component": {
            "baseline": "R1",
            "candidate": "R2",
            "metric": "citation_support_precision",
            "confidence_interval": {
                "estimate": 0.02,
                "lower": 0.01,
                "upper": 0.03,
            },
            "metrics": {"citation_support_precision": {"mean": 0.82, "n": 4}},
        },
        "planner_policy": {
            "comparisons": {
                "R1": {
                    "baseline": "A",
                    "candidate": "C",
                    "non_inferior": True,
                    "completeness": {"estimate": 0.01, "lower": -0.01, "upper": 0.03},
                },
                "R2": {
                    "baseline": "B",
                    "candidate": "D",
                    "non_inferior": True,
                    "completeness": {"estimate": 0.01, "lower": -0.01, "upper": 0.03},
                },
            },
            "metrics": {"search_calls": {"mean": 3.0, "n": 4}},
        },
        "end_to_end": {
            "baseline": "A",
            "candidate": "D",
            "metric": "information_completeness",
            "confidence_interval": {
                "estimate": 0.01,
                "lower": -0.01,
                "upper": 0.03,
            },
            "metrics": {"information_completeness": {"mean": 0.8, "n": 4}},
        },
        "reference": {
            "status": "not_present",
            "reason": "ORACLE is evaluator-only",
        },
    }
    payload.update(overrides)
    return payload


def test_renderer_requires_all_protocol_sections() -> None:
    payload = _renderer_summary()
    payload.pop("planner_policy")

    with pytest.raises(ResultValidationError, match="planner_policy"):
        render_results(payload)


def test_renderer_preserves_negative_primary_result() -> None:
    page = render_results(
        _renderer_summary(
            ranker_component={
                "baseline": "R1",
                "candidate": "R2",
                "metric": "citation_support_precision",
                "confidence_interval": {
                    "estimate": -0.02,
                    "lower": -0.04,
                    "upper": -0.01,
                },
            },
            planner_policy={"non_inferior": False},
        )
    )

    assert "主假设未成立" in page
    assert "-0.04" in page
    assert "失败分析" in page


def test_renderer_rejects_tampered_public_manifest(tmp_path: Path) -> None:
    experiment_dir = tmp_path / "experiment"
    experiment_dir.mkdir()
    summary_path = experiment_dir / "summary.json"
    summary_path.write_text(json.dumps(_renderer_summary(), ensure_ascii=False), encoding="utf-8")

    digest = hashlib.sha256(summary_path.read_bytes()).hexdigest()
    (experiment_dir / "manifest.sha256").write_text(
        json.dumps(
            {
                "schema_version": "experiment-artifact-manifest-v1",
                "files": {"summary.json": digest},
            }
        ),
        encoding="utf-8",
    )
    summary_path.write_text("{}", encoding="utf-8")

    with pytest.raises(ResultValidationError, match="hash"):
        render_results(experiment_dir=experiment_dir)


def test_renderer_writes_deterministic_accessible_public_artifacts(tmp_path: Path) -> None:
    render_results(_renderer_summary(), docs_dir=tmp_path)
    first = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    render_results(_renderer_summary(), docs_dir=tmp_path)
    second = {
        path.relative_to(tmp_path).as_posix(): path.read_bytes()
        for path in tmp_path.rglob("*")
        if path.is_file()
    }

    assert first == second
    assert set(second) == {
        "results.md",
        "evaluation.md",
        "assets/results/citation-support-vs-usd.svg",
        "assets/results/completeness-vs-search.svg",
        "assets/results/abcd-metrics.svg",
    }
    for name, value in second.items():
        if name.endswith(".svg"):
            text = value.decode("utf-8")
            assert 'role="img"' in text
            assert "<title" in text and "<desc" in text
            assert "no formal result is sealed" in text
