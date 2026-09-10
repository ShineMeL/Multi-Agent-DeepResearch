"""Owned run lifecycle and fixed-kind artifact downloads."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from pydantic import ValidationError
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from deepresearch.runtime.deployment_policy import DeploymentPolicy
from deepresearch.runtime.manager import RunManager
from deepresearch.storage import LocalArtifactStore
from deepresearch.storage.protocols import RunView

from .dependencies import (
    get_artifact_store,
    get_deployment_policy,
    get_manager,
    get_owned_run,
    get_owner,
)
from .error_handlers import APIError
from .identity import OwnerIdentity
from .schemas import ArtifactKind, CreateRunRequest, RunAccepted, RunViewResponse

router = APIRouter()


@router.post("/runs", status_code=202, response_model=RunAccepted)
async def create_run(
    body: CreateRunRequest,
    request: Request,
    manager: Annotated[RunManager, Depends(get_manager)],
    policy: Annotated[DeploymentPolicy, Depends(get_deployment_policy)],
    owner: Annotated[OwnerIdentity, Depends(get_owner)],
) -> RunAccepted:
    try:
        config = body.to_run_config(policy)
    except ValidationError:
        raise APIError("INVALID_REQUEST") from None
    view = await manager.create(
        config,
        client_ip=owner.client_ip,
        session_id=owner.session_id,
        idempotency_key=request.headers.get("Idempotency-Key"),
    )
    return RunAccepted(
        run_id=view.run_id,
        thread_id=view.thread_id,
        status=view.status,
        events_url=f"/runs/{view.run_id}/events",
    )


@router.get("/runs/{run_id}", response_model=RunViewResponse)
async def get_run(owned: Annotated[RunView, Depends(get_owned_run)]) -> RunViewResponse:
    return RunViewResponse.model_validate(owned)


@router.post("/runs/{run_id}/resume", response_model=RunViewResponse)
async def resume_run(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_manager)],
    owner: Annotated[OwnerIdentity, Depends(get_owner)],
) -> RunViewResponse:
    view = await manager.resume(run_id, client_ip=owner.client_ip, session_id=owner.session_id)
    return RunViewResponse.model_validate(view)


@router.post("/runs/{run_id}/cancel", response_model=RunViewResponse)
async def cancel_run(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_manager)],
    owner: Annotated[OwnerIdentity, Depends(get_owner)],
) -> RunViewResponse:
    view = await manager.cancel(run_id, owner_scope_sha256=owner.owner_scope_sha256)
    return RunViewResponse.model_validate(view)


@router.get("/runs/{run_id}/artifacts/{artifact_kind}")
async def download_artifact(
    artifact_kind: ArtifactKind,
    owned: Annotated[RunView, Depends(get_owned_run)],
    store: Annotated[LocalArtifactStore, Depends(get_artifact_store)],
) -> Response:
    artifact_id, media_type, filename = {
        "report": (owned.report_artifact_id, "text/markdown", "report.md"),
        "evidence": (owned.evidence_graph_artifact_id, "application/json", "evidence.json"),
        "manifest": (owned.manifest_artifact_id, "application/json", "manifest.json"),
    }[artifact_kind]
    if artifact_id is None:
        raise APIError("ARTIFACT_NOT_FOUND")
    try:
        data = await run_in_threadpool(store.get_bytes, artifact_id)
    except FileNotFoundError:
        raise APIError("ARTIFACT_NOT_FOUND") from None
    return Response(
        data,
        media_type=media_type,
        headers={
            "Content-Disposition": f'attachment; filename="{filename}"',
            "X-Content-Type-Options": "nosniff",
        },
    )
