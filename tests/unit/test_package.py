from typer.testing import CliRunner

import deepresearch
from apps.cli.main import app


def test_package_exposes_version():
    assert deepresearch.__version__ == "0.1.0"


def test_cli_version():
    result = CliRunner().invoke(app, ["version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == "0.1.0"
