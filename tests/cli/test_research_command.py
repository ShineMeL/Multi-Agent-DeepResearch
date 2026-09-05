from __future__ import annotations

import base64
import builtins
import json
import math
import re
import time
from collections import Counter
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import pytest
from typer.testing import CliRunner

import apps.cli.main as cli
from deepresearch.config import Settings
from deepresearch.domain import EvidenceSpan, ResearchPlan, RunBudget, SourceDocument
from deepresearch.providers.embeddings import EmbeddingModelFile, EmbeddingModelLock
from deepresearch.providers.replay_schema import (
    REPLAY_FILES,
    REPLAY_OPERATIONS,
    ReplayBundle,
)
from deepresearch.runtime.manifest import RunManifest
from deepresearch.workflow import BaselineRuntimeHooks

runner = CliRunner()
FIXTURE_ROOT = Path(__file__).parents[1] / "fixtures" / "replay" / "provider_contract"
BASELINE_ROOT = Path(__file__).parents[1] / "fixtures" / "replay" / "baseline"
_DETERMINISTIC_MONOTONIC_BASE = math.floor(time.monotonic()) + 10.0
app = cli.app
_build_provider_profile = getattr(cli, "_build_provider_profile")  # noqa: B009
_build_request_config = getattr(cli, "_build_request_config")  # noqa: B009
_repository_metadata = getattr(cli, "_repository_metadata")  # noqa: B009
_safe_absolute_endpoint = getattr(cli, "_safe_absolute_endpoint")  # noqa: B009


def _jsonl_records(filename: str) -> list[dict[str, Any]]:
    return [
        cast("dict[str, Any]", json.loads(line))
        for line in (BASELINE_ROOT / filename).read_text(encoding="utf-8").splitlines()
    ]


def _contains_private_body_key(value: object) -> bool:
    if isinstance(value, dict):
        mapping = cast("dict[object, object]", value)
        if any(key in {"body", "body_base64", "raw_body"} for key in mapping):
            return True
        return any(_contains_private_body_key(item) for item in mapping.values())
    if isinstance(value, list):
        return any(_contains_private_body_key(item) for item in cast("list[object]", value))
    return False


def _invoke_baseline_replay(
    *,
    checkpoint_db: Path,
    output: Path,
    question: str = "Compare planner strategies",
) -> Any:
    return runner.invoke(
        app,
        [
            "research",
            "--question",
            question,
            "--mode",
            "replay",
            "--replay-root",
            str(BASELINE_ROOT),
            "--checkpoint-db",
            str(checkpoint_db),
            "--output",
            str(output),
        ],
    )


def _deterministic_runtime_hooks() -> BaselineRuntimeHooks:
    origin = datetime(2026, 8, 29, tzinfo=UTC)
    # Use one stable synthetic base for both isolated runs.  A fresh large
    # monotonic float per invocation can make subtraction retain a few ULPs
    # of run-specific noise in persisted wall-time fields and artifact IDs.
    origin_monotonic = _DETERMINISTIC_MONOTONIC_BASE
    state = {"monotonic": origin_monotonic, "utc_calls": 0, "ids": 0}

    def monotonic() -> float:
        state["monotonic"] = round(state["monotonic"] + 0.001, 6)
        return state["monotonic"]

    def utc_now() -> datetime:
        state["utc_calls"] += 1
        slack = 0.0 if state["utc_calls"] == 1 else 0.000001
        return origin + timedelta(
            seconds=state["monotonic"] - origin_monotonic + slack
        )

    def new_id(prefix: str) -> str:
        state["ids"] += 1
        return f"{prefix}-fixture-{state['ids']}"

    return BaselineRuntimeHooks(monotonic=monotonic, utc_now=utc_now, new_id=new_id)


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


def test_baseline_fixture_contract() -> None:
    authorized = {
        *REPLAY_FILES,
        "manifest.sha256",
        "expected-report.md",
        "expected-evidence.json",
    }
    assert {item.name for item in BASELINE_ROOT.iterdir()} == authorized

    bundle = ReplayBundle.load(BASELINE_ROOT)
    verification = bundle.verify()
    assert verification.valid
    assert verification.errors == ()
    assert verification.record_count_by_operation == {
        operation: count
        for operation, count in {
            "model.complete": 2,
            "model.structured": 2,
            "model.stream": 0,
            "search": 4,
            "fetch": 3,
            "embed": 4,
        }.items()
    }
    assert tuple(verification.record_count_by_operation) == REPLAY_OPERATIONS

    signed_manifest = cast(
        "dict[str, Any]",
        json.loads((BASELINE_ROOT / "manifest.sha256").read_text(encoding="utf-8")),
    )
    assert set(cast("dict[str, str]", signed_manifest["file_sha256"])) == set(REPLAY_FILES)
    assert not {
        "expected-report.md",
        "expected-evidence.json",
    } & set(cast("dict[str, str]", signed_manifest["file_sha256"]))

    snapshot = bundle.snapshot
    assert snapshot.run_id == "00000000-0000-4000-8000-000000000009"
    assert snapshot.providers["model"].model_id == "baseline-model-v1"
    assert snapshot.providers["embed"].model_revision == "e" * 40

    model_records = _jsonl_records("model_responses.jsonl")
    plan_records = [
        item
        for item in model_records
        if item["key"]["operation"] == "model.complete"
        and item["key"]["prompt_version"] == "fixed-planner-v1"
    ]
    assert len(plan_records) == 1
    plan_output = plan_records[0]["outcome"]["response"]["output"]
    plan = ResearchPlan.model_validate_json(
        cast("str", plan_output).encode("utf-8"), strict=True
    )
    assert plan.created_by_model == "baseline-model-v1"
    assert tuple(item.id for item in plan.subquestions) == ("sq-1", "sq-2")
    assert tuple(item.information_needs[0].need_id for item in plan.subquestions) == (
        "need-sq-1",
        "need-sq-2",
    )

    query_records = [
        item
        for item in model_records
        if item["key"]["operation"] == "model.structured"
    ]
    assert len(query_records) == 2
    assert {
        tuple(cast("list[str]", item["outcome"]["response"]["output"]["queries"]))
        for item in query_records
    } == {
        ("planner strategy alpha", "planner strategy beta"),
        ("planner comparison gamma", "planner comparison alpha"),
    }
    assert all(item["outcome"]["kind"] == "success" for item in model_records)
    assert all(item["outcome"]["response"]["tool_calls"] == [] for item in model_records)

    search_records = _jsonl_records("search.jsonl")
    assert len(search_records) == 4
    search_urls = {
        hit["url"]
        for record in search_records
        for hit in cast("list[dict[str, Any]]", record["outcome"]["response"])
    }
    assert search_urls == {
        "https://alpha.example/strategy",
        "https://beta.example/strategy",
        "https://gamma.example/strategy.pdf",
    }
    assert sum(
        1
        for record in search_records
        for hit in cast("list[dict[str, Any]]", record["outcome"]["response"])
        if hit["url"] == "https://alpha.example/strategy"
    ) > 1

    document_records = _jsonl_records("documents.jsonl")
    assert len(document_records) == 3
    content_types = Counter(
        record["outcome"]["response"]["content_type"] for record in document_records
    )
    assert content_types == {"text/html": 2, "application/pdf": 1}
    for record in document_records:
        response = cast("dict[str, Any]", record["outcome"]["response"])
        assert response["status"] == 200
        assert base64.b64decode(response["body_base64"], validate=True)

    embedding_records = _jsonl_records("embeddings.jsonl")
    assert len(embedding_records) == 4
    for record in embedding_records:
        vectors = cast("list[list[float]]", record["outcome"]["response"])
        assert vectors
        assert all(len(vector) == 384 for vector in vectors)
        assert all(math.isfinite(value) for vector in vectors for value in vector)

    evidence_payload = cast(
        "dict[str, Any]",
        json.loads((BASELINE_ROOT / "expected-evidence.json").read_text(encoding="utf-8")),
    )
    assert set(evidence_payload) == {"evidence", "sources"}
    evidence = [
        EvidenceSpan.model_validate_json(
            json.dumps(item, ensure_ascii=False).encode("utf-8"), strict=True
        )
        for item in cast("list[dict[str, Any]]", evidence_payload["evidence"])
    ]
    sources = [
        SourceDocument.model_validate_json(
            json.dumps(item, ensure_ascii=False).encode("utf-8"), strict=True
        )
        for item in cast("list[dict[str, Any]]", evidence_payload["sources"])
    ]
    assert len(evidence) >= 4
    assert len({item.source_family_id for item in sources}) >= 2
    evidence_ids = {item.evidence_id for item in evidence}
    report = (BASELINE_ROOT / "expected-report.md").read_text(encoding="utf-8")
    cited_ids = set(re.findall(r"E-[0-9a-f]{64}", report))
    assert cited_ids
    assert cited_ids <= evidence_ids
    assert not _contains_private_body_key(evidence_payload)
    for marker in ("secret-marker", "api_key", "Authorization"):
        assert marker not in report
        assert marker.encode() not in (BASELINE_ROOT / "expected-evidence.json").read_bytes()


def test_replay_writes_all_public_outputs(tmp_path: Path) -> None:
    output = tmp_path / "out"
    result = _invoke_baseline_replay(
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        output=output,
    )

    assert result.exit_code == 0, result.output
    assert "status=completed" in result.stdout
    assert "stop_reason=SUFFICIENT" in result.stdout
    assert {item.name for item in output.iterdir()} == {
        "report.md",
        "evidence.json",
        "run-manifest.json",
    }
    assert (output / "report.md").read_bytes() == (BASELINE_ROOT / "expected-report.md").read_bytes()
    assert (output / "evidence.json").read_bytes() == (BASELINE_ROOT / "expected-evidence.json").read_bytes()
    manifest = RunManifest.model_validate_json(
        (output / "run-manifest.json").read_bytes(), strict=True
    )
    assert manifest.replay_parent == "00000000-0000-4000-8000-000000000009"
    assert manifest.stop_reason == "SUFFICIENT"


def test_replay_is_byte_deterministic_in_isolated_roots(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    run_id = "00000000-0000-4000-8000-000000000010"
    monkeypatch.setattr(cli, "_new_run_id", lambda: run_id)
    monkeypatch.setattr(cli, "_runtime_hooks", _deterministic_runtime_hooks)

    outputs: list[Path] = []
    for index in (1, 2):
        output = tmp_path / f"out-{index}"
        result = _invoke_baseline_replay(
            checkpoint_db=tmp_path / f"checkpoints-{index}.sqlite3",
            output=output,
        )
        assert result.exit_code == 0, result.output
        outputs.append(output)

    filenames = ("report.md", "evidence.json", "run-manifest.json")
    for filename in filenames:
        assert (outputs[0] / filename).read_bytes() == (outputs[1] / filename).read_bytes()
    manifest = RunManifest.model_validate_json(
        (outputs[0] / "run-manifest.json").read_bytes(), strict=True
    )
    assert manifest.run_id == run_id
    assert manifest.thread_id == run_id


def test_replay_shared_root_second_run_uses_cache_without_provider_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    lookup_calls: list[str] = []
    original_lookup = ReplayBundle.lookup

    def lookup(self: ReplayBundle, key: Any) -> Any:
        lookup_calls.append(key.operation)
        return original_lookup(self, key)

    monkeypatch.setattr(ReplayBundle, "lookup", lookup)
    checkpoint_db = tmp_path / "shared.sqlite3"
    first = _invoke_baseline_replay(
        checkpoint_db=checkpoint_db,
        output=tmp_path / "first",
    )
    first_call_count = len(lookup_calls)
    second = _invoke_baseline_replay(
        checkpoint_db=checkpoint_db,
        output=tmp_path / "second",
    )

    assert first.exit_code == 0, first.output
    assert second.exit_code == 0, second.output
    assert first_call_count > 0
    assert len(lookup_calls) == first_call_count
    manifest = RunManifest.model_validate_json(
        (tmp_path / "second" / "run-manifest.json").read_bytes(), strict=True
    )
    assert manifest.cache_hit_count > 0
    assert all(call.cache_hit for call in manifest.provider_calls)


def test_replay_miss_does_not_fallback_to_live_or_publish_output(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_import = builtins.__import__

    def reject_live_import(name: str, *args: Any, **kwargs: Any) -> Any:
        if name in {
            "deepresearch.config",
            "deepresearch.providers.embeddings",
            "deepresearch.providers.openai_compatible",
            "deepresearch.providers.tavily",
            "deepresearch.providers.httpx_fetcher",
        }:
            raise AssertionError(f"unexpected live import: {name}")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", reject_live_import)
    output = tmp_path / "out"
    result = _invoke_baseline_replay(
        checkpoint_db=tmp_path / "checkpoints.sqlite3",
        output=output,
        question="A question absent from the signed replay fixture",
    )

    assert result.exit_code != 0
    assert "research run failed" in str(result.exception).casefold()
    assert not output.exists()
    runtime_root = tmp_path / ".checkpoints.runtime"
    if runtime_root.exists():
        assert b"secret-marker" not in b"".join(
            item.read_bytes() for item in runtime_root.rglob("*") if item.is_file()
        )
