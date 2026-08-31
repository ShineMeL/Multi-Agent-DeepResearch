from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

from deepresearch.domain import RunConfig, RunEvent, RunResult

from .cancellation import CancellationToken


@dataclass(frozen=True)
class CheckpointRef:
    checkpoint_id: str
    thread_id: str
    created_at: datetime

    def __post_init__(self) -> None:
        if not self.checkpoint_id:
            raise ValueError("checkpoint_id must not be empty")
        if not self.thread_id:
            raise ValueError("thread_id must not be empty")
        if self.created_at.tzinfo is None or self.created_at.utcoffset() is None:
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
