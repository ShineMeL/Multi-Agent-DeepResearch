from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol, cast

from deepresearch.domain import RunConfig, RunEvent, RunResult

from .cancellation import CancellationToken


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    thread_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        checkpoint_id = cast("object", self.checkpoint_id)
        thread_id = cast("object", self.thread_id)
        created_at = cast("object", self.created_at)
        if type(checkpoint_id) is not str:
            raise TypeError("checkpoint_id is invalid")
        if not checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if type(thread_id) is not str:
            raise TypeError("thread_id is invalid")
        if not thread_id:
            raise ValueError("thread_id must not be empty")
        if type(created_at) is not datetime:
            raise TypeError("created_at is invalid")
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("created_at must be timezone-aware")


class ResearchRunner(Protocol):
    async def run(
        self,
        *,
        run_id: str,
        thread_id: str,
        config: RunConfig,
        checkpoint: CheckpointRef | None,
        emit: Callable[[RunEvent], Awaitable[None]],
        cancellation_token: CancellationToken,
    ) -> RunResult: ...


__all__ = ["CheckpointRef", "ResearchRunner"]
