from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from math import isfinite
from random import Random
from types import MappingProxyType
from typing import Literal, TypeVar

from .errors import ProviderError, ProviderErrorCode

type ProviderOperation = Literal["model", "search", "fetch", "parse", "embed"]
T = TypeVar("T")

_OPERATIONS: tuple[ProviderOperation, ...] = (
    "model",
    "search",
    "fetch",
    "parse",
    "embed",
)
_RETRIED_CODES: frozenset[ProviderErrorCode] = frozenset(
    {"TIMEOUT", "RATE_LIMITED", "NETWORK", "UPSTREAM_5XX"}
)
_NEVER_RETRIED_CODES: frozenset[ProviderErrorCode] = frozenset(
    {
        "AUTHENTICATION",
        "INVALID_REQUEST",
        "INVALID_RESPONSE",
        "PARSE_UNSUPPORTED",
        "REPLAY_MISS",
        "INVALID_SNAPSHOT",
    }
)


@dataclass(frozen=True)
class ProviderCallPolicy:
    default_timeout_seconds: Mapping[ProviderOperation, float]
    max_retries: int = 2
    base_delay_seconds: float = 0.25
    max_delay_seconds: float = 4.0
    jitter_ratio: float = 0.20

    def __post_init__(self) -> None:
        if set(self.default_timeout_seconds) != set(_OPERATIONS):
            raise ValueError("default_timeout_seconds must define every provider operation")
        if any(
            timeout <= 0 or not isfinite(timeout)
            for timeout in self.default_timeout_seconds.values()
        ):
            raise ValueError("provider timeouts must be finite and positive")
        if self.max_retries < 0:
            raise ValueError("max_retries must be non-negative")
        if self.base_delay_seconds < 0 or not isfinite(self.base_delay_seconds):
            raise ValueError("base_delay_seconds must be finite and non-negative")
        if self.max_delay_seconds < 0 or not isfinite(self.max_delay_seconds):
            raise ValueError("max_delay_seconds must be finite and non-negative")
        if self.max_delay_seconds < self.base_delay_seconds:
            raise ValueError("max_delay_seconds must not be less than base_delay_seconds")
        if not 0 <= self.jitter_ratio <= 1 or not isfinite(self.jitter_ratio):
            raise ValueError("jitter_ratio must be finite and between zero and one")
        frozen = MappingProxyType(dict(self.default_timeout_seconds))
        object.__setattr__(self, "default_timeout_seconds", frozen)

    @classmethod
    def defaults(cls) -> ProviderCallPolicy:
        return cls(
            default_timeout_seconds={
                "model": 120.0,
                "search": 30.0,
                "fetch": 30.0,
                "parse": 30.0,
                "embed": 30.0,
            }
        )


@dataclass(frozen=True)
class ProviderCallAttempt:
    operation: ProviderOperation
    invocation_index: int
    attempt_index: int
    deadline: float
    succeeded: bool
    error_code: ProviderErrorCode | None = None


class ProviderCallExecutor:
    def __init__(
        self,
        *,
        policy: ProviderCallPolicy,
        clock: Callable[[], float] = time.monotonic,
        sleeper: Callable[[float], Awaitable[None]] = asyncio.sleep,
        random: Random | None = None,
        strict_replay: bool = False,
    ) -> None:
        self._policy = policy
        self._clock = clock
        self._sleeper = sleeper
        self._random = random or Random()
        self._strict_replay = strict_replay
        self._attempts: list[ProviderCallAttempt] = []

    @property
    def attempts(self) -> tuple[ProviderCallAttempt, ...]:
        return tuple(self._attempts)

    async def call(
        self,
        operation: ProviderOperation,
        invoke: Callable[[float], Awaitable[T]],
        *,
        remaining_deadline: float,
        fallback_invocations: Sequence[Callable[[float], Awaitable[T]]] = (),
    ) -> T:
        if not isfinite(remaining_deadline):
            raise ValueError("remaining_deadline must be finite")
        invocations = (invoke,)
        if not self._strict_replay:
            invocations += tuple(fallback_invocations)

        last_error: ProviderError | None = None
        for invocation_index, candidate in enumerate(invocations):
            for attempt_index in range(self._policy.max_retries + 1):
                if self._clock() >= remaining_deadline:
                    raise self._deadline_error(operation)
                attempt_deadline = min(
                    remaining_deadline,
                    self._clock() + self._policy.default_timeout_seconds[operation],
                )
                try:
                    result = await candidate(attempt_deadline)
                except ProviderError as error:
                    last_error = error
                    self._attempts.append(
                        ProviderCallAttempt(
                            operation=operation,
                            invocation_index=invocation_index,
                            attempt_index=attempt_index,
                            deadline=attempt_deadline,
                            succeeded=False,
                            error_code=error.code,
                        )
                    )
                    if error.code in _NEVER_RETRIED_CODES:
                        raise
                    may_retry = error.retryable and error.code in _RETRIED_CODES
                    if not may_retry:
                        raise
                    if attempt_index == self._policy.max_retries:
                        break
                    delay = self._retry_delay(error, attempt_index)
                    if self._clock() + delay >= remaining_deadline:
                        raise self._deadline_error(operation) from error
                    await self._sleeper(delay)
                    if self._clock() >= remaining_deadline:
                        raise self._deadline_error(operation) from error
                else:
                    self._attempts.append(
                        ProviderCallAttempt(
                            operation=operation,
                            invocation_index=invocation_index,
                            attempt_index=attempt_index,
                            deadline=attempt_deadline,
                            succeeded=True,
                        )
                    )
                    return result

        if last_error is None:
            raise RuntimeError("provider executor reached an impossible state")
        raise last_error

    def _retry_delay(self, error: ProviderError, attempt_index: int) -> float:
        if error.retry_after is not None:
            return error.retry_after
        exponential = min(
            self._policy.base_delay_seconds * (2**attempt_index),
            self._policy.max_delay_seconds,
        )
        jitter = self._random.uniform(
            1 - self._policy.jitter_ratio, 1 + self._policy.jitter_ratio
        )
        return exponential * jitter

    @staticmethod
    def _deadline_error(operation: ProviderOperation) -> ProviderError:
        return ProviderError(
            code="TIMEOUT",
            provider="executor",
            operation=operation,
            public_message="provider call deadline exceeded",
            retryable=True,
        )


__all__ = ["ProviderCallAttempt", "ProviderCallExecutor", "ProviderCallPolicy"]
