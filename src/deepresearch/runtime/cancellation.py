from threading import Event
from typing import Literal


class OperationCancelled(RuntimeError):
    code: Literal["CANCELLED"] = "CANCELLED"

    def __init__(self) -> None:
        super().__init__("operation cancelled")


class CancellationToken:
    """A thread-safe, one-way cancellation signal."""

    def __init__(self) -> None:
        self._event = Event()

    def cancel(self) -> None:
        self._event.set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def raise_if_cancelled(self) -> None:
        if self._event.is_set():
            raise OperationCancelled


__all__ = ["CancellationToken", "OperationCancelled"]
