from __future__ import annotations

from math import isfinite
from typing import Literal

from deepresearch.domain import ResourceUsage

type ProviderErrorCode = Literal[
    "TIMEOUT",
    "RATE_LIMITED",
    "INVALID_REQUEST",
    "INVALID_RESPONSE",
    "AUTHENTICATION",
    "NETWORK",
    "REPLAY_MISS",
    "INVALID_SNAPSHOT",
    "PARSE_UNSUPPORTED",
    "UPSTREAM_5XX",
]

_ERROR_CODES = frozenset(
    {
        "TIMEOUT",
        "RATE_LIMITED",
        "INVALID_REQUEST",
        "INVALID_RESPONSE",
        "AUTHENTICATION",
        "NETWORK",
        "REPLAY_MISS",
        "INVALID_SNAPSHOT",
        "PARSE_UNSUPPORTED",
        "UPSTREAM_5XX",
    }
)


class ProviderError(RuntimeError):
    code: ProviderErrorCode
    provider: str
    operation: str
    public_message: str
    retryable: bool
    retry_after: float | None
    usage: ResourceUsage | None

    def __init__(
        self,
        *,
        code: ProviderErrorCode,
        provider: str,
        operation: str,
        public_message: str,
        retryable: bool,
        retry_after: float | None = None,
        usage: ResourceUsage | None = None,
    ) -> None:
        if code not in _ERROR_CODES:
            raise ValueError(f"unknown provider error code: {code}")
        if retry_after is not None and (retry_after < 0 or not isfinite(retry_after)):
            raise ValueError("retry_after must be a finite non-negative number")
        super().__init__(public_message)
        self.code = code
        self.provider = provider
        self.operation = operation
        self.public_message = public_message
        self.retryable = retryable
        self.retry_after = retry_after
        self.usage = usage


__all__ = ["ProviderError", "ProviderErrorCode"]
