from __future__ import annotations

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
