from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import BaseModel, SecretStr, ValidationError
from typer.testing import CliRunner

import apps.cli.main as cli
from deepresearch.config import Settings
from deepresearch.domain import (
    FreshnessRequirement,
    ResearchRequest,
    ResourceUsage,
    RunBudget,
    RunConfig,
    RunEvent,
    RunResult,
)
from deepresearch.runtime.manifest import RunManifest

ROOT = Path(__file__).parents[1]
BASELINE_ROOT = ROOT / "fixtures" / "replay" / "baseline"
runner = CliRunner()

MODEL_SECRET = "model-key-7f4a9c2e0d1b8a6f"
SEARCH_SECRET = "search-key-3c8e1a5d9b7f2e4c"
URL_SENTINEL = "userinfo-query-6b2d9f4a8c1e"


def _config() -> RunConfig:
    budget = RunBudget.preset("low").model_copy(update={"max_cost_usd": None})
    request = ResearchRequest(
        question="redaction boundary",
        output_requirements={"answer_shape": "markdown"},
        report_language="en",
        source_languages=("en",),
        freshness_requirement=FreshnessRequirement(kind="none"),
        execution_mode="replay",
        access_profile="local",
        provider_profile_id="baseline-replay-v1:" + "a" * 64,
        run_purpose="test",
        budget_preset="low",
    )
    return RunConfig(
        request=request,
        workflow_id="baseline-v1",
        planner_id="P1",
        ranker_id="R1",
        budget=budget,
        prompt_versions={"planner": "fixed-planner-v1"},
        ranker_weights_version=None,
        seed=0,
    )


def _public_models() -> tuple[RunConfig, RunEvent, RunResult]:
    usage = ResourceUsage.zero()
    event = RunEvent(
        seq=1,
        run_id="redaction-run",
        timestamp=datetime(2026, 8, 29, tzinfo=UTC),
        node="Planner",
        kind="node_completed",
        status="completed",
        public_payload={"is_partial": False},
        usage_delta=usage,
        artifact_ids=(),
        error_code=None,
    )
    result = RunResult(
        run_id="redaction-run",
        thread_id="redaction-run",
        status="completed",
        stop_reason="SUFFICIENT",
        is_partial=False,
        report_artifact_id=None,
        evidence_graph_artifact_id=None,
        manifest_artifact_id=None,
        final_usage=usage,
        error_code=None,
    )
    return _config(), event, result


def _serialized(value: object) -> tuple[str, ...]:
    values: list[str] = [repr(value), str(value)]
    if isinstance(value, BaseModel):
        values.append(json.dumps(value.model_dump(mode="python"), default=str, sort_keys=True))
        values.append(json.dumps(value.model_dump(mode="json"), sort_keys=True))
        values.append(value.model_dump_json())
    return tuple(values)


def test_settings_and_public_models_never_serialize_secret_sentinels(
    tmp_path: Path,
) -> None:
    settings = Settings(
        model_base_url="https://model.example/v1",
        model_id="model-v1",
        model_api_key=MODEL_SECRET,
        tavily_api_key=SEARCH_SECRET,
    )
    assert type(settings.model_api_key) is SecretStr
    assert type(settings.tavily_api_key) is SecretStr
    for value in _serialized(settings):
        assert MODEL_SECRET not in value
        assert SEARCH_SECRET not in value

    with pytest.raises(ValidationError) as error:
        Settings(
            model_base_url=f"https://{URL_SENTINEL}@model.example/v1?token={URL_SENTINEL}",
            model_id="model-v1",
            model_api_key=MODEL_SECRET,
        )
    assert URL_SENTINEL not in str(error.value)

    config, event, result = _public_models()
    for value in (config, event, result):
        for serialized in _serialized(value):
            assert MODEL_SECRET not in serialized
            assert SEARCH_SECRET not in serialized
            assert URL_SENTINEL not in serialized

    output = tmp_path / "out"
    cli_result = runner.invoke(
        cli.app,
        [
            "research",
            "--question",
            "Compare planner strategies",
            "--mode",
            "replay",
            "--replay-root",
            str(BASELINE_ROOT),
            "--checkpoint-db",
            str(tmp_path / "checkpoints.sqlite3"),
            "--output",
            str(output),
        ],
    )
    assert cli_result.exit_code == 0, cli_result.output
    assert MODEL_SECRET not in cli_result.output
    assert SEARCH_SECRET not in cli_result.output
    assert URL_SENTINEL not in cli_result.output

    manifest_bytes = (output / "run-manifest.json").read_bytes()
    manifest = RunManifest.model_validate_json(manifest_bytes, strict=True)
    recreated = RunManifest.create(manifest.model_dump(round_trip=True))
    for value in (manifest, recreated):
        for serialized in _serialized(value):
            assert MODEL_SECRET not in serialized
            assert SEARCH_SECRET not in serialized
            assert URL_SENTINEL not in serialized

    runtime_root = tmp_path / ".checkpoints.runtime"
    persisted = [*output.rglob("*")] + [*runtime_root.rglob("*")]
    for path in persisted:
        if path.is_file():
            payload = path.read_bytes()
            for marker in (MODEL_SECRET, SEARCH_SECRET, URL_SENTINEL):
                assert marker.encode("utf-8") not in payload
