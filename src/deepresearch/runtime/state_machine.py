from typing import Literal

from deepresearch.domain import RunStatus

RunAction = Literal["start", "resume", "interrupt", "complete", "fail", "cancel"]


class InvalidTransition(ValueError):
    """Raised when an action is not permitted from the current run status."""


_TRANSITIONS: dict[tuple[RunStatus, RunAction], RunStatus] = {
    ("queued", "start"): "running",
    ("queued", "interrupt"): "interrupted",
    ("running", "interrupt"): "interrupted",
    ("interrupted", "resume"): "running",
    ("running", "complete"): "completed",
    ("running", "fail"): "failed",
    ("queued", "cancel"): "cancelled",
    ("running", "cancel"): "cancelled",
    ("interrupted", "cancel"): "cancelled",
    ("cancelled", "cancel"): "cancelled",
}


def transition(current: RunStatus, action: RunAction) -> RunStatus:
    try:
        return _TRANSITIONS[(current, action)]
    except KeyError as exc:
        raise InvalidTransition(f"{current} + {action}") from exc


def validate_resume(status: RunStatus) -> None:
    transition(status, "resume")


def validate_cancel(status: RunStatus) -> None:
    transition(status, "cancel")


__all__ = ["InvalidTransition", "RunAction", "transition", "validate_cancel", "validate_resume"]
