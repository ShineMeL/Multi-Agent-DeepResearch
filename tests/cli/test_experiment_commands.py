from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from typer.testing import CliRunner

from apps.cli.main import app


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


def test_runner_stages_the_user_config_source(tmp_path: Path) -> None:
    from apps.cli.experiment import _runner

    source = tmp_path / "formal.yaml"
    source.write_text("sealed: true\n", encoding="utf-8")
    config = SimpleNamespace()
    runner = _runner(config, source)  # type: ignore[arg-type]
    assert runner.config_source == source.resolve()


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
