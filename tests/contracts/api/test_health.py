from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import text

from apps.api.main import create_app
from apps.api.settings import ServiceSettings


@pytest.fixture
def app(tmp_path: Path):
    return create_app(
        ServiceSettings(
            database_url=f"sqlite+aiosqlite:///{tmp_path / 'runs.sqlite'}",
            artifact_root=tmp_path / "artifacts",
            checkpoint_sqlite_path=tmp_path / "artifacts" / "checkpoints.sqlite",
            session_signing_key="health-test-signing-key-at-least-32-bytes",
            langgraph_strict_msgpack=True,
        )
    )


def test_readiness_reports_healthy_dependencies_and_closed_admission(app):
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        assert client.get("/health/live").json() == {"status": "ok", "checks": {}}
        response = client.get("/health/ready")
        assert response.status_code == 200
        assert response.json() == {
            "status": "ok",
            "checks": {
                "database": "ok",
                "artifacts": "ok",
                "schema": "ok",
                "checkpointer": "ok",
                "admission": "ok",
            },
        }
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        app.state.accepting_runs = False
        assert client.get("/health/live").status_code == 200
        assert client.get("/health/ready").status_code == 503


def test_readiness_before_lifespan_is_unavailable(app):
    client = TestClient(app, client=("127.0.0.1", 1234))
    assert client.get("/health/live").status_code == 200
    assert client.get("/health/ready").status_code == 503


def test_readiness_fails_when_database_is_unavailable(app, monkeypatch):
    with TestClient(app, client=("127.0.0.1", 1234)) as client:

        def broken_connect(self):
            raise RuntimeError("private-database-password")

        monkeypatch.setattr(type(app.state.store.engine), "connect", broken_connect)
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"]["database"] == "unavailable"
        assert "private-database-password" not in response.text
        assert client.get("/health/live").status_code == 200


@pytest.mark.parametrize("check", ["artifacts", "schema", "checkpointer"])
def test_readiness_checks_each_required_dependency(app, tmp_path, check):
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        if check == "artifacts":
            app.state.artifact_root = tmp_path / "missing"
        elif check == "checkpointer":
            app.state.checkpointer_ready = False
        else:

            async def remove_schema_version():
                async with app.state.store.engine.begin() as connection:
                    await connection.execute(text("DELETE FROM service_schema_versions"))

            client.portal.call(remove_schema_version)
        response = client.get("/health/ready")
        assert response.status_code == 503
        assert response.json()["checks"][check] == "unavailable"


@pytest.mark.parametrize("path", ["/runs", "/runs/unknown/resume"])
def test_shutdown_gate_rejects_create_and_resume_with_static_error(app, path):
    with TestClient(app, client=("127.0.0.1", 1234)) as client:
        app.state.accepting_runs = False
        response = client.post(path, json={})
        assert response.status_code == 503
        assert response.json()["code"] == "SERVICE_SHUTDOWN"
