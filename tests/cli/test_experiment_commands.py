from __future__ import annotations

from pathlib import Path

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
