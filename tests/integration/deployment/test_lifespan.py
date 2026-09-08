import asyncio
import io
import json
import logging
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient
from pydantic import SecretStr, ValidationError

from apps.api.identity import OwnerSessionMiddleware
from apps.api.main import create_app
from apps.api.schemas import CreateRunRequest
from apps.api.settings import ServiceSettings
from deepresearch.domain import ResourceUsage
from deepresearch.runtime.runner_factory import (
    DefaultCoreRunnerBuilder,
    FileProviderRouteCatalog,
    LangGraphServiceRunnerFactory,
    ProviderProfileDrift,
)
from deepresearch.storage.migrations.runner import ServiceMigrationError, upgrade_service_schema
from deepresearch.storage.protocols import RunRecord
from tests.integration.replay.test_baseline_graph import config
from tests.unit.runtime.test_manager import ControlledRunner


@pytest.fixture
def settings(tmp_path: Path):
    return ServiceSettings(
        database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
        artifact_root=tmp_path / "artifacts",
        checkpoint_sqlite_path=tmp_path / "artifacts" / "checkpoints.sqlite",
        session_signing_key="test-signing-key-at-least-32-bytes-long",
        langgraph_strict_msgpack=True,
    )


@pytest.fixture
def app(settings):
    return create_app(settings)


def test_uvicorn_factory_can_call_create_app_without_arguments(monkeypatch):
    monkeypatch.setenv("SESSION_SIGNING_KEY", "local-test-signing-key-at-least-32-bytes")
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    assert isinstance(create_app(), FastAPI)


@pytest.mark.parametrize("signing_key", ["", "   ", "too-short"])
def test_session_signing_key_requires_32_nonblank_utf8_bytes(monkeypatch, signing_key):
    monkeypatch.setenv("SESSION_SIGNING_KEY", signing_key)
    monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", "true")
    with pytest.raises(ValidationError, match="SESSION_SIGNING_KEY.*32 bytes"):
        ServiceSettings()


@pytest.mark.parametrize("strict_value", [None, "false"])
@pytest.mark.parametrize(
    "database_url",
    [
        "sqlite+aiosqlite:///./test.db",
        "postgresql+asyncpg://u:p@db/research",
    ],
)
def test_startup_requires_explicit_strict_msgpack(monkeypatch, strict_value, database_url):
    monkeypatch.setenv("DATABASE_URL", database_url)
    monkeypatch.setenv("SESSION_SIGNING_KEY", "test-signing-key-at-least-32-bytes-long")
    monkeypatch.delenv("LANGGRAPH_STRICT_MSGPACK", raising=False)
    if strict_value is not None:
        monkeypatch.setenv("LANGGRAPH_STRICT_MSGPACK", strict_value)
    with pytest.raises(ValidationError):
        ServiceSettings()


@pytest.mark.parametrize(
    "overrides",
    [
        {"deployment_access_profile": "public_live", "cookie_secure": False},
        {
            "deployment_access_profile": "public_live",
            "cookie_secure": True,
            "allowed_budget_presets": ("high",),
        },
        {"allowed_execution_modes": ()},
        {"allowed_provider_profile_ids": ()},
        {"allowed_provider_profile_ids": (" ",)},
        {"allowed_run_purposes": ()},
        {"allowed_budget_presets": ()},
        {"trusted_proxy_cidrs": ("not-a-cidr",)},
        {"provider_credential_env_names": ("SESSION_SIGNING_KEY",)},
        {"provider_credential_env_names": ("UNCOVERED_SECRET",)},
        {"daily_cost_limit_usd": "NaN"},
        {"daily_cost_limit_usd": "-1"},
    ],
)
def test_settings_reject_invalid_policy_and_secret_configuration(settings, overrides):
    with pytest.raises(ValidationError):
        ServiceSettings(**(settings.model_dump() | overrides))


def test_loaded_secrets_deduplicate_and_include_explicit_signing_key(settings, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "provider-private-secret")
    monkeypatch.setenv("SEARCH_API_KEY", "provider-private-secret")
    monkeypatch.setenv("SESSION_SIGNING_KEY", "other-environment-private-secret")
    assert settings.loaded_secret_values() == (
        "test-signing-key-at-least-32-bytes-long",
        "provider-private-secret",
        "other-environment-private-secret",
    )
    assert "provider-private-secret" not in settings.model_dump_json()


def test_public_policy_uses_core_budget_presets(settings):
    public = ServiceSettings(
        **(
            settings.model_dump()
            | {
                "deployment_access_profile": "public_live",
                "cookie_secure": True,
            }
        )
    )
    policy = public.deployment_policy()
    assert policy.forced_access_profile == "public_live"
    assert policy.budget_presets["low"].max_total_tokens == 20_000
    assert policy.budget_presets["medium"].max_total_tokens == 40_000
    assert str(policy.budget_presets["medium"].max_cost_usd) == "0.50"


def test_composition_installs_one_secretstr_owner_middleware(settings):
    app = create_app(settings)
    owners = [item for item in app.user_middleware if item.cls is OwnerSessionMiddleware]
    assert len(owners) == 1
    assert isinstance(owners[0].kwargs["session_secret"], SecretStr)


async def test_lifespan_composes_concrete_resources_with_shared_limits(app, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "provider-private-secret")
    async with app.router.lifespan_context(app):
        manager = app.state.manager
        factory = manager.runner_factory
        assert isinstance(factory, LangGraphServiceRunnerFactory)
        assert isinstance(factory.route_catalog, FileProviderRouteCatalog)
        assert isinstance(factory.builder, DefaultCoreRunnerBuilder)
        assert factory.builder.search_slot.__self__ is manager.admission
        assert factory.builder.host_slot.__self__ is manager.admission
        assert "provider-private-secret" in manager.secrets
        assert "provider-private-secret" in factory.builder._secrets
        assert app.state.accepting_runs is True
        assert app.state.checkpointer_ready is True
        assert (
            await manager.checkpointer.aget_tuple({"configurable": {"thread_id": "absent"}}) is None
        )
    assert app.state.accepting_runs is False
    assert app.state.checkpointer_ready is False


async def test_startup_marks_running_runs_interrupted(app):
    await upgrade_service_schema(app.state.store.engine)
    await app.state.store.create_run(
        RunRecord(
            run_id="r1",
            thread_id="r1",
            status="running",
            config_json={},
            pricing_status="unknown",
            pricing_snapshots=(),
            provider_profile_json={},
            provider_profile_sha256="a" * 64,
            config_sha256="b" * 64,
            owner_scope_sha256="c" * 64,
            idempotency_scope_sha256="c" * 64,
            idempotency_key=None,
            admission_reservation_id=None,
            admission_attempt_no=None,
            stop_reason=None,
            is_partial=False,
            report_artifact_id=None,
            evidence_graph_artifact_id=None,
            manifest_artifact_id=None,
            final_usage=ResourceUsage.zero(),
            error_code=None,
            updated_at=datetime.now(UTC),
            version=7,
        )
    )
    async with app.router.lifespan_context(app):
        recovered = await app.state.store.get_run("r1")
        assert (
            recovered.status,
            recovered.is_partial,
            recovered.error_code,
            recovered.version,
        ) == (
            "interrupted",
            True,
            "PROCESS_RESTART",
            8,
        )
        assert (await app.state.store.list_events_after("r1", 0))[-1].status == "interrupted"


@pytest.mark.parametrize("failure", ["migration", "recovery", "checkpointer"])
async def test_startup_failure_never_opens_admission(app, monkeypatch, failure):
    from apps.api import main

    error = ServiceMigrationError(1, RuntimeError("private-failure"))
    if failure == "migration":
        app.state.schema_upgrader = AsyncMock(side_effect=error)
    elif failure == "recovery":
        monkeypatch.setattr(app.state.store, "reconcile_startup", AsyncMock(side_effect=error))
    else:

        @asynccontextmanager
        async def broken_checkpointer(**kwargs):
            raise error
            yield

        monkeypatch.setattr(main, "open_service_checkpointer", broken_checkpointer)
    with pytest.raises(ServiceMigrationError):
        async with app.router.lifespan_context(app):
            pytest.fail("startup should fail")
    assert app.state.accepting_runs is False
    assert app.state.checkpointer_ready is False
    assert getattr(app.state, "manager", None) is None


async def test_shutdown_stops_admission_before_manager_and_closes_saver_last(app, monkeypatch):
    from apps.api import main

    order = []
    original_open = main.open_service_checkpointer

    @asynccontextmanager
    async def observed_checkpointer(**kwargs):
        async with original_open(**kwargs) as saver:
            order.append("saver_open")
            yield saver
            order.append("saver_close")

    monkeypatch.setattr(main, "open_service_checkpointer", observed_checkpointer)
    async with app.router.lifespan_context(app):
        manager = app.state.manager
        original_shutdown = manager.shutdown

        async def observed_shutdown(grace_seconds):
            assert app.state.accepting_runs is False
            assert grace_seconds == 20.0
            assert (
                await manager.checkpointer.aget_tuple({"configurable": {"thread_id": "x"}}) is None
            )
            await original_shutdown(grace_seconds)
            order.append("manager_shutdown")

        monkeypatch.setattr(manager, "shutdown", observed_shutdown)
    assert order == ["saver_open", "manager_shutdown", "saver_close"]


async def test_checkpoint_path_outside_artifact_root_fails_before_open(settings, tmp_path):
    settings = ServiceSettings(
        **(
            settings.model_dump()
            | {
                "checkpoint_sqlite_path": tmp_path / "outside.sqlite",
            }
        )
    )
    app = create_app(settings)
    with pytest.raises(ValueError, match="inside artifact_root"):
        async with app.router.lifespan_context(app):
            pytest.fail("must reject escaping checkpoint path")
    assert not (tmp_path / "outside.sqlite").exists()
    assert app.state.accepting_runs is False


async def test_service_logs_redact_secrets_on_direct_and_ancestor_handlers(settings, monkeypatch):
    monkeypatch.setenv("MODEL_API_KEY", "private-provider-value")
    outputs = []
    handlers = []
    for name in ("deepresearch", "deepresearch.runtime", "uvicorn", "uvicorn.error", ""):
        output = io.StringIO()
        handler = logging.StreamHandler(output)
        logger = logging.getLogger(name)
        logger.addHandler(handler)
        outputs.append(output)
        handlers.append((logger, handler))
    try:
        app = create_app(settings)
        async with app.router.lifespan_context(app):
            for name in ("deepresearch.runtime", "uvicorn.error"):
                logging.getLogger(name).warning("failure: %s", "private-provider-value")
            assert all("private-provider-value" not in output.getvalue() for output in outputs)
            assert all("[REDACTED]" in output.getvalue() for output in outputs)
    finally:
        for logger, handler in handlers:
            logger.removeHandler(handler)


def test_default_catalog_preserves_names_and_does_not_fabricate_routes():
    catalog = FileProviderRouteCatalog.load(None)
    for name in ("replay", "replay-default"):
        profile = catalog.resolve(name)
        assert profile.profile_id == name
        assert profile.execution_mode == "replay"
        assert profile.routes == ()


@pytest.mark.parametrize(
    "overrides",
    [
        {"allowed_execution_modes": ("live",)},
        {"allowed_provider_profile_ids": ("missing",)},
    ],
)
def test_settings_reject_inconsistent_catalog_allowlists(settings, overrides):
    with pytest.raises(ValidationError):
        ServiceSettings(**(settings.model_dump() | overrides))


def test_settings_reject_catalog_credentials_outside_provider_allowlist(settings, tmp_path):
    path = tmp_path / "profiles.json"
    path.write_text(
        json.dumps(
            {
                "profiles": {
                    "replay-default": {
                        "execution_mode": "replay",
                        "routes": [
                            {
                                "operation": "model",
                                "provider_id": "replay",
                                "endpoint_type": "model",
                                "model_id": None,
                                "model_revision": None,
                                "base_url": None,
                                "credential_ref": "SESSION_SIGNING_KEY",
                                "fallback_rank": 0,
                                "parameters": {},
                            }
                        ],
                    }
                }
            }
        )
    )
    with pytest.raises(ValidationError):
        ServiceSettings(**(settings.model_dump() | {"provider_profile_catalog_path": path}))


async def test_default_empty_routes_fail_closed_before_run_creation(app):
    async with app.router.lifespan_context(app):
        request = config().request.model_copy(
            update={
                "provider_profile_id": "replay-default",
                "execution_mode": "replay",
                "run_purpose": "test",
                "budget_preset": "low",
            }
        )
        conf = CreateRunRequest(
            request=request,
            workflow_id="baseline-v1",
            planner_id="P1",
            ranker_id="R1",
        ).to_run_config(app.state.deployment_policy)
        with pytest.raises(ProviderProfileDrift):
            await app.state.manager.create(conf, client_ip="127.0.0.1", session_id="test")


async def test_lifespan_persists_active_run_interruption_before_saver_close(app, monkeypatch):
    async with app.router.lifespan_context(app):
        manager = app.state.manager
        request = config().request.model_copy(
            update={
                "provider_profile_id": "replay-default",
                "execution_mode": "replay",
                "run_purpose": "test",
                "budget_preset": "low",
            }
        )
        conf = CreateRunRequest(
            request=request,
            workflow_id="baseline-v1",
            planner_id="P1",
            ranker_id="R1",
        ).to_run_config(app.state.deployment_policy)
        runner = ControlledRunner()
        # The controlled provider blocks until the real manager cancels its token.
        monkeypatch.setattr(manager.runner_factory, "create", lambda **kwargs: runner)
        view = await manager.create(conf, client_ip="127.0.0.1", session_id="test")
        await asyncio.wait_for(runner.started.wait(), timeout=2)
        original_shutdown = manager.shutdown

        async def immediate_shutdown(grace_seconds):
            assert grace_seconds == 20.0
            assert app.state.accepting_runs is False
            await original_shutdown(grace_seconds=0)
            record = await manager.store.get_run(view.run_id)
            assert record.status == "interrupted"
            assert record.error_code == "SERVICE_SHUTDOWN"
            assert record.is_partial is True
            events = await manager.store.list_events_after(view.run_id, 0)
            assert events[-1].status == "interrupted"
            assert (
                await manager.checkpointer.aget_tuple({"configurable": {"thread_id": "x"}}) is None
            )

        monkeypatch.setattr(manager, "shutdown", immediate_shutdown)


async def test_hosted_routes_preserve_owner_isolation_and_durable_sse(app, monkeypatch):
    async with app.router.lifespan_context(app):
        manager = app.state.manager
        runner = ControlledRunner()
        runner.finish.set()
        monkeypatch.setattr(manager.runner_factory, "create", lambda **kwargs: runner)
        transport = ASGITransport(app=app, client=("127.0.0.1", 1234))
        async with AsyncClient(transport=transport, base_url="http://test") as owner:
            request = config().request.model_copy(update={"provider_profile_id": "replay-default"})
            created = await owner.post(
                "/runs",
                json={
                    "request": request.model_dump(mode="json"),
                    "workflow_id": "baseline-v1",
                    "planner_id": "P1",
                    "ranker_id": "R1",
                },
            )
            assert created.status_code == 202
            run_id = created.json()["run_id"]
            await manager.wait(run_id)
            events = await owner.get(f"/runs/{run_id}/events")
            assert events.status_code == 200
            assert '"status":"completed"' in events.text
            assert "id: 1" in events.text
            replay = await owner.get(f"/runs/{run_id}/events", headers={"Last-Event-ID": "1"})
            assert replay.status_code == 200
            assert replay.text == ""
        async with AsyncClient(transport=transport, base_url="http://test") as stranger:
            foreign = await stranger.get(f"/runs/{run_id}/events")
            missing = await stranger.get("/runs/absent/events")
            assert foreign.status_code == missing.status_code == 404
            assert foreign.json() == missing.json()
