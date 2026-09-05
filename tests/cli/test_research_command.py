from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

import apps.cli.main as cli
from deepresearch.config import Settings
from deepresearch.domain import RunBudget
from deepresearch.providers.embeddings import EmbeddingModelFile, EmbeddingModelLock
from deepresearch.providers.replay_schema import ReplayBundle

runner = CliRunner()
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "replay" / "provider_contract"
app = cli.app
_build_provider_profile = getattr(cli, "_build_provider_profile")  # noqa: B009
_build_request_config = getattr(cli, "_build_request_config")  # noqa: B009
_repository_metadata = getattr(cli, "_repository_metadata")  # noqa: B009
_safe_absolute_endpoint = getattr(cli, "_safe_absolute_endpoint")  # noqa: B009


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


def _embedding_lock() -> EmbeddingModelLock:
    return EmbeddingModelLock.create(
        model_id="embed-v1",
        revision="b" * 40,
        vector_dimension=384,
        normalize_embeddings=True,
        files=(
            EmbeddingModelFile(path="model.bin", sha256="a" * 64, size_bytes=1),
        ),
    )


def _settings() -> Settings:
    return Settings(
        model_base_url="https://model.example/",
        model_id="model-v1",
        model_api_key="secret-marker",
        connect_timeout_seconds=10.0,
        read_timeout_seconds=45.0,
    )


def test_provider_profile_is_content_bound_and_secret_free() -> None:
    bundle = ReplayBundle.load(FIXTURE_ROOT)
    replay_profile, replay_id = _build_provider_profile(mode="replay", bundle=bundle)
    live_profile, live_id = _build_provider_profile(
        mode="live", settings=_settings(), embedding_lock=_embedding_lock()
    )
    hybrid_profile, hybrid_id = _build_provider_profile(
        mode="hybrid",
        bundle=bundle,
        settings=_settings(),
        embedding_lock=_embedding_lock(),
    )

    assert set(replay_profile) == {"schema_version", "mode", "replay", "live"}
    assert replay_profile["mode"] == "replay"
    assert replay_profile["live"] is None
    assert set(live_profile) == {"schema_version", "mode", "replay", "live"}
    assert live_profile["replay"] is None
    assert hybrid_profile["mode"] == "hybrid"
    assert replay_id.startswith("baseline-replay-v1:")
    assert live_id.startswith("baseline-live-v1:")
    assert hybrid_id.startswith("baseline-hybrid-v1:")
    for value in (replay_profile, live_profile, hybrid_profile):
        assert "secret-marker" not in repr(value)

    changed_settings = _settings().model_copy(update={"model_id": "model-v2"})
    _, changed_id = _build_provider_profile(
        mode="live", settings=changed_settings, embedding_lock=_embedding_lock()
    )
    assert changed_id != live_id


@pytest.mark.parametrize(
    "endpoint",
    [
        "https://user:pass@example.com",
        "https://example.com/path?api_key=secret-marker",
        "https://example.com/path?arbitrary=value",
        "https://example.com/path#fragment",
        "relative.example/path",
        "ftp://example.com/path",
        "https:///path",
    ],
)
def test_endpoint_policy_rejects_unsafe_values(endpoint: str) -> None:
    with pytest.raises(ValueError, match="provider endpoint is invalid") as error:
        _safe_absolute_endpoint(endpoint)
    assert "secret-marker" not in str(error.value)


def test_request_config_uses_frozen_cli_identity() -> None:
    config = _build_request_config(
        question="Compare planner strategies",
        mode="replay",
        budget_name="medium",
        provider_profile_id="baseline-replay-v1:" + "a" * 64,
        prompt_versions={"planner": "p1", "planner_queries": "p1-q", "writer": "w1"},
    )

    assert config.request.output_requirements == {"answer_shape": "markdown"}
    assert config.request.report_language == "en"
    assert config.request.source_languages == ("en",)
    assert config.request.freshness_requirement.kind == "none"
    assert config.request.access_profile == "local"
    assert config.request.run_purpose == "demo"
    assert config.request.provider_profile_id.endswith("a" * 64)
    assert config.workflow_id == "baseline-v1"
    assert config.planner_id == "P1"
    assert config.ranker_id == "R1"
    assert config.seed == 0
    assert config.budget.max_cost_usd is None
    assert config.budget.max_search_calls == RunBudget.preset("medium").max_search_calls


def test_repository_metadata_is_real_and_value_free() -> None:
    commit, lock_digest = _repository_metadata()

    assert len(commit) == 40
    assert len(lock_digest) == 64
