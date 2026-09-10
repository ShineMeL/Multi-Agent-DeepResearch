"""Static health responses without exception, path, or connection details."""

from pathlib import Path
from tempfile import TemporaryFile
from typing import Literal

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel
from sqlalchemy import select, text
from starlette.concurrency import run_in_threadpool

from deepresearch.storage.migrations.runner import MIGRATIONS
from deepresearch.storage.models import ServiceSchemaVersion

CheckStatus = Literal["ok", "unavailable"]


class HealthResponse(BaseModel):
    status: CheckStatus
    checks: dict[str, CheckStatus]


router = APIRouter()


@router.get("/health/live", response_model=HealthResponse)
async def live_health() -> HealthResponse:
    return HealthResponse(status="ok", checks={})


def _artifact_access(root: Path) -> bool:
    try:
        if not root.is_dir():
            return False
        with TemporaryFile(dir=root) as probe:
            probe.write(b"health")
            probe.flush()
            probe.seek(0)
            return probe.read() == b"health"
    except OSError:
        return False


@router.get("/health/ready", response_model=HealthResponse)
async def ready_health(request: Request, response: Response) -> HealthResponse:
    state = request.app.state
    checks: dict[str, CheckStatus] = {
        "database": "unavailable",
        "artifacts": "unavailable",
        "schema": "unavailable",
        "checkpointer": "ok" if state.checkpointer_ready else "unavailable",
        "admission": "ok" if state.accepting_runs else "unavailable",
    }
    try:
        async with state.store.engine.connect() as connection:
            await connection.execute(text("SELECT 1"))
            checks["database"] = "ok"
            versions = set(
                (await connection.execute(select(ServiceSchemaVersion.version))).scalars()
            )
            if versions == {migration.version for migration in MIGRATIONS}:
                checks["schema"] = "ok"
    except Exception:  # noqa: BLE001 - probes expose static failure states only
        checks["schema"] = "unavailable"
    if await run_in_threadpool(_artifact_access, state.artifact_root):
        checks["artifacts"] = "ok"
    available = all(value == "ok" for value in checks.values())
    response.status_code = 200 if available else 503
    return HealthResponse(status="ok" if available else "unavailable", checks=checks)
