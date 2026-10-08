"""Git staging policy for local state and credential files."""

import subprocess
from pathlib import Path


def test_git_excludes_runtime_databases_and_real_env_files(tmp_path: Path) -> None:
    repository = Path(__file__).resolve().parents[2]
    (tmp_path / ".gitignore").write_bytes((repository / ".gitignore").read_bytes())
    subprocess.run(["git", "init", "--quiet", str(tmp_path)], check=True)
    paths = (
        "deepresearch.db",
        "deepresearch.db-wal",
        "deepresearch.db-shm",
        "runs.sqlite3",
        "runs.sqlite3-journal",
        "nested/checkpoints.sqlite",
        ".env.production",
        ".env.local",
        "nested/.env.example",
        ".env.example",
        ".env.demo.example",
        "tests/fixtures/replay/baseline/snapshot.json",
    )
    result = subprocess.run(
        ["git", "check-ignore", "--stdin", "-z"],
        input=("\0".join(paths) + "\0").encode(),
        cwd=tmp_path,
        capture_output=True,
        check=False,
    )
    assert result.returncode in (0, 1), result.stderr
    assert {path.decode() for path in result.stdout.split(b"\0") if path} == set(paths[:9])
