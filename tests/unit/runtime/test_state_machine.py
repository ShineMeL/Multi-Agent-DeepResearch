import pytest

from deepresearch.runtime.state_machine import (
    InvalidTransition,
    transition,
    validate_cancel,
    validate_resume,
)


def test_resume_only_allows_interrupted() -> None:
    assert transition("interrupted", "resume") == "running"
    assert validate_resume("interrupted") is None


def test_resume_completed_is_invalid() -> None:
    with pytest.raises(InvalidTransition):
        transition("completed", "resume")
    with pytest.raises(InvalidTransition):
        validate_resume("completed")


def test_cancel_is_idempotent_only_for_cancelled_status() -> None:
    assert transition("cancelled", "cancel") == "cancelled"
    assert validate_cancel("cancelled") is None
    with pytest.raises(InvalidTransition):
        validate_cancel("completed")
