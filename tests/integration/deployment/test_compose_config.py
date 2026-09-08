"""Static contract for the local Compose deployment."""

import json
from pathlib import Path, PurePosixPath

import yaml


def test_compose_packages_api_ui_and_postgres_without_embedded_secrets():
    config = yaml.safe_load(Path("docker-compose.yml").read_text(encoding="utf-8"))

    assert {"api", "ui", "postgres"} <= set(config["services"])
    assert {"artifact-data", "postgres-data"} <= set(config["volumes"])

    api = config["services"]["api"]
    ui = config["services"]["ui"]
    postgres = config["services"]["postgres"]
    environment = api["environment"]

    assert api["build"] == ui["build"] == "."
    assert "apps.api.main:create_app" in api["command"]
    assert "apps/ui/app.py" in ui["command"]
    assert api["depends_on"]["postgres"]["condition"] == "service_healthy"
    assert ui["depends_on"]["api"]["condition"] == "service_healthy"
    assert "/health/ready" in " ".join(api["healthcheck"]["test"])
    assert "_stcore/health" in " ".join(ui["healthcheck"]["test"])
    assert "pg_isready" in " ".join(postgres["healthcheck"]["test"])

    assert environment["DATABASE_URL"].startswith("postgresql+asyncpg://deepresearch:")
    assert "${POSTGRES_PASSWORD:?" in environment["DATABASE_URL"]
    assert environment["ARTIFACT_ROOT"] == "/var/lib/deepresearch/artifacts"
    checkpoint = environment["CHECKPOINT_SQLITE_PATH"]
    assert checkpoint == "/var/lib/deepresearch/artifacts/checkpoints.sqlite"
    assert PurePosixPath(checkpoint).is_relative_to(PurePosixPath(environment["ARTIFACT_ROOT"]))
    assert "artifact-data:/var/lib/deepresearch/artifacts" in api["volumes"]

    assert environment["LANGGRAPH_STRICT_MSGPACK"] == "true"
    assert environment["DEPLOYMENT_ACCESS_PROFILE"] == "showcase"
    assert json.loads(environment["ALLOWED_EXECUTION_MODES"]) == ["replay"]
    assert json.loads(environment["ALLOWED_PROVIDER_PROFILE_IDS"]) == ["replay-default"]
    assert environment["SESSION_SIGNING_KEY"].startswith("${SESSION_SIGNING_KEY:?")
    assert postgres["environment"]["POSTGRES_PASSWORD"].startswith("${POSTGRES_PASSWORD:?")
    assert ui["environment"]["DEEPRESEARCH_API_URL"] == "http://api:8000"


def test_docker_context_excludes_local_secrets_and_runtime_data():
    ignored = Path(".dockerignore").read_text(encoding="utf-8").splitlines()

    assert ".env" in ignored
    assert ".env.*" in ignored
    assert "artifacts/" in ignored
    assert ".venv/" in ignored


def test_dockerfile_uses_the_locked_dependencies_as_a_non_root_user():
    dockerfile = Path("Dockerfile").read_text(encoding="utf-8")

    assert "FROM python:3.12-slim" in dockerfile
    assert "COPY pyproject.toml uv.lock ./" in dockerfile
    assert "uv sync --frozen --no-dev" in dockerfile
    assert "USER deepresearch" in dockerfile
    assert "SESSION_SIGNING_KEY" not in dockerfile
    assert "POSTGRES_PASSWORD" not in dockerfile
