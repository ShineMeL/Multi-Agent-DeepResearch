"""Exercise the built distribution without the editable checkout on sys.path."""

import os
import shutil
import subprocess
import sys
from pathlib import Path


def test_built_wheel_exposes_research_and_experiment_cli(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[3]
    output = tmp_path / "wheels"
    uv_executable = os.environ.get("UV") or shutil.which("uv")
    assert uv_executable, "Run this test through uv so the build tool is available"
    built = subprocess.run(
        [
            uv_executable,
            "build",
            "--wheel",
            "--offline",
            "--out-dir",
            str(output),
        ],
        cwd=repository,
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert built.returncode == 0, built.stderr
    wheels = tuple(output.glob("*.whl"))
    assert len(wheels) == 1
    # Python can import pure Python packages directly from the wheel archive.
    # The child keeps installed dependencies but cannot resolve our editable
    # source tree; wheel modules must win and include the experiment imports.
    code = """
import sys
from pathlib import Path
wheel = sys.argv[1]
checkout = Path(sys.argv[2]).resolve()
sys.path = [wheel, *(p for p in sys.path if p and Path(p).resolve() not in {checkout, checkout / 'src'})]
from apps.cli.main import app
from typer.testing import CliRunner
for arguments in (['--help'], ['research', '--help'], ['experiment', '--help']):
    result = CliRunner().invoke(app, arguments)
    assert result.exit_code == 0, (result.output, result.exception)
import apps, benchmarks, experiments
for package in (apps, benchmarks, experiments):
    assert package.__file__.startswith(wheel), package.__file__
"""
    imported = subprocess.run(
        [sys.executable, "-I", "-c", code, str(wheels[0]), str(repository)],
        cwd=tmp_path,
        env={**os.environ, "PYTHONUTF8": "1"},
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
    )
    assert imported.returncode == 0, imported.stderr
