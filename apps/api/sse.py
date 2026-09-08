"""Durable cursor replay; manager subscriptions only wake the next store read."""

import asyncio
from collections.abc import AsyncIterator

from deepresearch.runtime.manager import RunManager, RunNotFound
from deepresearch.storage.protocols import RunStore

HEARTBEAT_SECONDS = 15.0
# RunEventRow.seq uses SQLAlchemy Integer (PostgreSQL INT4).
_MAX_STORED_SEQ = (1 << 31) - 1
_TERMINAL_STATUSES = frozenset({"interrupted", "completed", "failed", "cancelled"})


class InvalidLastEventId(ValueError):
    """The submitted SSE cursor is not an unsigned decimal integer."""


def parse_last_event_id(value: str | None) -> int:
    if value is None:
        return 0
    if not value or not value.isascii() or not value.isdecimal():
        raise InvalidLastEventId()
    # Convert in bounded chunks so Python's decimal-string conversion limit
    # does not redefine the unsigned-decimal cursor contract.
    cursor = 0
    for offset in range(0, len(value), 500):
        chunk = value[offset : offset + 500]
        cursor = cursor * 10 ** len(chunk) + int(chunk)
    return cursor


async def event_stream(
    run_id: str,
    *,
    last_event_id: int,
    owner_scope_sha256: str,
    store: RunStore,
    manager: RunManager,
) -> AsyncIterator[dict[str, object]]:
    subscription = await manager.subscribe(run_id, owner_scope_sha256=owner_scope_sha256)
    cursor = last_event_id
    try:
        while True:
            # Preserve the client cursor but keep SQL bindings within the
            # shared schema's signed integer range, including PostgreSQL INT4.
            query_cursor = min(cursor, _MAX_STORED_SEQ)
            durable = sorted(
                await store.list_events_after(run_id, query_cursor), key=lambda item: item.seq
            )
            for event in durable:
                if event.seq <= cursor:
                    continue
                cursor = event.seq
                yield {"id": str(event.seq), "event": event.kind, "data": event.model_dump_json()}

            record = await store.get_owned_run(run_id, owner_scope_sha256)
            if record is None:
                raise RunNotFound(run_id)
            if record.status in _TERMINAL_STATUSES:
                # A terminal row/event can commit after the snapshot. Drain
                # again before EOF; core node statuses never close the stream.
                if await store.list_events_after(run_id, min(cursor, _MAX_STORED_SEQ)):
                    continue
                return
            try:
                await asyncio.wait_for(subscription.wait(), timeout=HEARTBEAT_SECONDS)
            except TimeoutError:
                yield {"comment": "heartbeat"}
    finally:
        await subscription.close()
