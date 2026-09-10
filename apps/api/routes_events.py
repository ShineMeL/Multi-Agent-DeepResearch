"""Owner-scoped SSE transport for the persisted run event log."""

from typing import Annotated

from fastapi import APIRouter, Depends, Request
from sse_starlette.sse import EventSourceResponse

from deepresearch.runtime.manager import RunManager
from deepresearch.storage.protocols import RunView

from .dependencies import get_manager, get_owned_run, get_owner
from .identity import OwnerIdentity
from .sse import event_stream, parse_last_event_id

router = APIRouter()


@router.get("/runs/{run_id}/events")
async def run_events(
    request: Request,
    owned: Annotated[RunView, Depends(get_owned_run)],
    manager: Annotated[RunManager, Depends(get_manager)],
    owner: Annotated[OwnerIdentity, Depends(get_owner)],
) -> EventSourceResponse:
    # Authorize in a dependency before validation and before SSE sends headers.
    # The generator independently authorizes its subscription before reading.
    cursor = parse_last_event_id(request.headers.get("Last-Event-ID"))
    return EventSourceResponse(
        event_stream(
            owned.run_id,
            last_event_id=cursor,
            owner_scope_sha256=owner.owner_scope_sha256,
            store=manager.store,
            manager=manager,
        )
    )
