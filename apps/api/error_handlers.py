"""Allowlisted public errors; exception text and validation inputs stay private."""

from typing import Literal

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from starlette.exceptions import HTTPException
from starlette.responses import JSONResponse

from deepresearch.runtime.deployment_policy import PolicyViolation
from deepresearch.runtime.manager import (
    CheckpointResumeUnavailable,
    IdempotencyConflict,
    MissingPricingSnapshot,
    RunNotFound,
    ServiceShuttingDown,
)
from deepresearch.runtime.runner_factory import ProviderProfileDrift, ResearchGraphUnavailable
from deepresearch.runtime.state_machine import InvalidTransition

PublicErrorCode = Literal[
    "INVALID_REQUEST",
    "DEPLOYMENT_POLICY_VIOLATION",
    "PRICING_REQUIRED",
    "INVALID_LAST_EVENT_ID",
    "RUN_NOT_FOUND",
    "ARTIFACT_NOT_FOUND",
    "INVALID_RUN_STATE",
    "IDEMPOTENCY_CONFLICT",
    "CHECKPOINT_RESUME_UNAVAILABLE",
    "PROVIDER_PROFILE_DRIFT",
    "RESEARCH_GRAPH_UNAVAILABLE",
    "RATE_LIMITED",
    "SERVICE_SHUTDOWN",
    "INTERNAL_ERROR",
    "INVALID_FORWARDED_HEADER",
    "INVALID_CLIENT_IP",
    "NOT_FOUND",
    "METHOD_NOT_ALLOWED",
]

_ERRORS: dict[PublicErrorCode, tuple[int, str]] = {
    "INVALID_REQUEST": (422, "Invalid request."),
    "DEPLOYMENT_POLICY_VIOLATION": (422, "Request is not permitted by deployment policy."),
    "PRICING_REQUIRED": (422, "Server pricing is required for this run."),
    "INVALID_LAST_EVENT_ID": (422, "Invalid event cursor."),
    "RUN_NOT_FOUND": (404, "Run not found."),
    "ARTIFACT_NOT_FOUND": (404, "Artifact not found."),
    "INVALID_RUN_STATE": (409, "This action is not permitted in the current run state."),
    "IDEMPOTENCY_CONFLICT": (409, "Idempotency key was used for a different request."),
    "CHECKPOINT_RESUME_UNAVAILABLE": (
        409,
        "Checkpoint continuation is unavailable in this service version.",
    ),
    "PROVIDER_PROFILE_DRIFT": (409, "Provider profile is unavailable or no longer permitted."),
    "RESEARCH_GRAPH_UNAVAILABLE": (
        422,
        "Research workflow is unavailable in this service version.",
    ),
    "RATE_LIMITED": (429, "Service capacity or usage limit reached."),
    "SERVICE_SHUTDOWN": (503, "Service is shutting down."),
    "INTERNAL_ERROR": (500, "An internal error occurred."),
    "INVALID_FORWARDED_HEADER": (400, "Invalid forwarded client address."),
    "INVALID_CLIENT_IP": (400, "Client address is unavailable or invalid."),
    "NOT_FOUND": (404, "Resource not found."),
    "METHOD_NOT_ALLOWED": (405, "Method not allowed."),
}


class APIError(Exception):
    """Extension point for SSE/admission, restricted to static public messages."""

    def __init__(
        self,
        code: PublicErrorCode,
        *,
        run_id: str | None = None,
        retry_after: int | None = None,
    ) -> None:
        if code not in _ERRORS:
            raise ValueError("unknown public error code")
        if retry_after is not None and (type(retry_after) is not int or retry_after < 0):
            raise ValueError("retry_after must be non-negative seconds")
        super().__init__(code)
        self.code: PublicErrorCode = code
        self.run_id = run_id
        self.retry_after = retry_after


def error_response(error: APIError) -> JSONResponse:
    status, message = _ERRORS[error.code]
    headers = {"Cache-Control": "no-store"}
    if error.retry_after is not None:
        headers["Retry-After"] = str(error.retry_after)
    return JSONResponse(
        status_code=status,
        content={
            "code": error.code,
            "message": message,
            "run_id": None if error.code == "RUN_NOT_FOUND" else error.run_id,
            "retry_after": error.retry_after,
        },
        headers=headers,
    )


async def handle_error(request: Request, error: Exception) -> JSONResponse:
    if isinstance(error, APIError):
        return error_response(error)
    code: PublicErrorCode
    if isinstance(error, RunNotFound):
        code = "RUN_NOT_FOUND"
    elif isinstance(error, RequestValidationError):
        code = "INVALID_REQUEST"
    elif isinstance(error, PolicyViolation):
        code = "DEPLOYMENT_POLICY_VIOLATION"
    elif isinstance(error, MissingPricingSnapshot):
        code = "PRICING_REQUIRED"
    elif isinstance(error, InvalidTransition):
        code = "INVALID_RUN_STATE"
    elif isinstance(error, IdempotencyConflict):
        code = "IDEMPOTENCY_CONFLICT"
    elif isinstance(error, CheckpointResumeUnavailable):
        code = "CHECKPOINT_RESUME_UNAVAILABLE"
    elif isinstance(error, ProviderProfileDrift):
        code = "PROVIDER_PROFILE_DRIFT"
    elif isinstance(error, ResearchGraphUnavailable):
        code = "RESEARCH_GRAPH_UNAVAILABLE"
    elif isinstance(error, ServiceShuttingDown):
        code = "SERVICE_SHUTDOWN"
    elif isinstance(error, HTTPException):
        code = (
            "NOT_FOUND"
            if error.status_code == 404
            else "METHOD_NOT_ALLOWED"
            if error.status_code == 405
            else "INTERNAL_ERROR"
        )
    else:
        code = "INTERNAL_ERROR"
    # Never include submitted IDs or exception arguments in a missing/foreign response.
    return error_response(APIError(code))


def install_error_handlers(app: FastAPI) -> None:
    for kind in (
        APIError,
        RunNotFound,
        RequestValidationError,
        PolicyViolation,
        MissingPricingSnapshot,
        InvalidTransition,
        IdempotencyConflict,
        CheckpointResumeUnavailable,
        ProviderProfileDrift,
        ResearchGraphUnavailable,
        ServiceShuttingDown,
        HTTPException,
        Exception,
    ):
        app.add_exception_handler(kind, handle_error)
