"""Injectable application resources, shared by lifecycle and future SSE routes."""

from typing import Annotated, cast

from fastapi import Depends, Request

from deepresearch.runtime.deployment_policy import DeploymentPolicy
from deepresearch.runtime.manager import RunManager
from deepresearch.storage import LocalArtifactStore
from deepresearch.storage.protocols import RunView

from .identity import OwnerIdentity


def get_manager(request: Request) -> RunManager:
    return cast(RunManager, request.app.state.manager)


def get_deployment_policy(request: Request) -> DeploymentPolicy:
    return cast(DeploymentPolicy, request.app.state.deployment_policy)


def get_artifact_store(request: Request) -> LocalArtifactStore:
    return cast(LocalArtifactStore, request.app.state.artifact_store)


def get_owner(request: Request) -> OwnerIdentity:
    return cast(OwnerIdentity, request.state.owner)


async def get_owned_run(
    run_id: str,
    manager: Annotated[RunManager, Depends(get_manager)],
    owner: Annotated[OwnerIdentity, Depends(get_owner)],
) -> RunView:
    return await manager.get(run_id, owner_scope_sha256=owner.owner_scope_sha256)
