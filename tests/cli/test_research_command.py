from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

from apps.cli.main import app


runner = CliRunner()


def test_version_output_is_byte_compatible() -> None:
    result = runner.invoke(app, ["version"])

    assert result.exit_code == 0
    assert result.stdout.replace("\r\n", "\n") == "0.1.0\n"


def test_research_help_lists_only_public_options() -> None:
    result = runner.invoke(app, ["research", "--help"])

    assert result.exit_code == 0
    for option in (
        "--question",
        "--mode",
        "--replay-root",
        "--record-replay-root",
        "--checkpoint-db",
        "--resume-checkpoint",
        "--resume-thread-id",
        "--budget",
        "--output",
    ):
        assert option in result.stdout
    assert "api-key" not in result.stdout.casefold()
    assert "credential" not in result.stdout.casefold()


def test_research_requires_question_and_mode() -> None:
    missing_both = runner.invoke(app, ["research"])
    missing_mode = runner.invoke(app, ["research", "--question", "demo"])
    missing_question = runner.invoke(app, ["research", "--mode", "replay"])

    assert missing_both.exit_code != 0
    assert missing_mode.exit_code != 0
    assert missing_question.exit_code != 0
    assert "question" in missing_both.output.casefold()
    assert "mode" in missing_mode.output.casefold()
    assert "question" in missing_question.output.casefold()


@pytest.mark.parametrize(
    ("extra", "needle"),
    [
        (("--mode", "replay"), "replay mode requires"),
        (("--mode", "live", "--replay-root", "missing"), "live mode"),
        (("--mode", "replay", "--replay-root", "missing"), "replay root"),
        (("--mode", "replay", "--replay-root", "missing", "--record-replay-root", "record"), "replay mode"),
        (("--mode", "replay", "--replay-root", "missing", "--resume-checkpoint", "cp"), "resume requires"),
        (("--mode", "replay", "--replay-root", "missing", "--resume-thread-id", "thread"), "resume requires"),
    ],
)
def test_option_preflight_rejects_invalid_combinations(
    tmp_path: Path, extra: tuple[str, ...], needle: str
) -> None:
    result = runner.invoke(
        app,
        ["research", "--question", "demo", *extra, "--output", str(tmp_path / "out")],
    )

    assert result.exit_code != 0
    assert needle in result.output.casefold()


def test_option_preflight_rejects_existing_targets(tmp_path: Path) -> None:
    replay_root = tmp_path / "replay"
    replay_root.mkdir()
    record_root = tmp_path / "record"
    record_root.mkdir()
    output = tmp_path / "out"
    output.mkdir()

    existing_record = runner.invoke(
        app,
        [
            "research",
            "--question",
            "demo",
            "--mode",
            "replay",
            "--replay-root",
            str(replay_root),
            "--record-replay-root",
            str(record_root),
            "--output",
            str(tmp_path / "new-out"),
        ],
    )
    existing_output = runner.invoke(
        app,
        [
            "research",
            "--question",
            "demo",
            "--mode",
            "replay",
            "--replay-root",
            str(replay_root),
            "--output",
            str(output),
        ],
    )

    assert existing_record.exit_code != 0
    assert "record" in existing_record.output.casefold()
    assert existing_output.exit_code != 0
    assert "output" in existing_output.output.casefold()


def test_secret_option_is_rejected_without_echoing_value(tmp_path: Path) -> None:
    marker = "secret-marker"
    result = runner.invoke(
        app,
        [
            "research",
            "--question",
            "demo",
            "--mode",
            "replay",
            "--replay-root",
            str(tmp_path / "missing"),
            "--output",
            str(tmp_path / "out"),
            "--provider-api-key",
            marker,
        ],
    )

    assert result.exit_code != 0
    assert marker not in result.output
