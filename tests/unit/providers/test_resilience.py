from random import Random

import pytest

from deepresearch.providers import ProviderCallExecutor, ProviderCallPolicy, ProviderError


class FakeClock:
    def __init__(self, now: float = 10.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeSleeper:
    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.delays: list[float] = []

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)
        self.clock.now += delay


class SequenceCall:
    def __init__(self, outcomes: list[str | ProviderError]) -> None:
        self.outcomes = outcomes
        self.calls = 0
        self.deadlines: list[float] = []

    async def __call__(self, deadline: float) -> str:
        self.calls += 1
        self.deadlines.append(deadline)
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, ProviderError):
            raise outcome
        return outcome


def provider_error(
    code: str, *, retryable: bool, retry_after: float | None = None
) -> ProviderError:
    return ProviderError(
        code=code,
        provider="fake",
        operation="search",
        public_message=code.lower(),
        retryable=retryable,
        retry_after=retry_after,
    )


@pytest.mark.asyncio
async def test_retry_policy_uses_injected_jitter_and_never_retries_auth() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=sleeper,
        random=Random(7),
    )
    transient = SequenceCall(
        [provider_error("UPSTREAM_5XX", retryable=True), "ok"]
    )

    assert await executor.call("search", transient, remaining_deadline=100.0) == "ok"
    assert len(sleeper.delays) == 1
    assert 0.20 <= sleeper.delays[0] <= 0.30
    authentication = SequenceCall(
        [provider_error("AUTHENTICATION", retryable=True), "should-not-run"]
    )
    with pytest.raises(ProviderError) as error:
        await executor.call("model", authentication, remaining_deadline=100.0)
    assert error.value.code == "AUTHENTICATION"
    assert authentication.calls == 1


@pytest.mark.asyncio
async def test_attempt_deadlines_are_absolute_and_capped_per_operation() -> None:
    clock = FakeClock(now=50.0)
    sleeper = FakeSleeper(clock)
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(), clock=clock, sleeper=sleeper, random=Random(1)
    )
    call = SequenceCall(["ok"])

    await executor.call("search", call, remaining_deadline=500.0)

    assert call.deadlines == [80.0]


@pytest.mark.asyncio
async def test_retry_after_is_honored_without_jitter() -> None:
    clock = FakeClock()
    sleeper = FakeSleeper(clock)
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(), clock=clock, sleeper=sleeper, random=Random(2)
    )
    call = SequenceCall(
        [provider_error("RATE_LIMITED", retryable=True, retry_after=1.75), "ok"]
    )

    assert await executor.call("search", call, remaining_deadline=100.0) == "ok"
    assert sleeper.delays == [1.75]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "code",
    [
        "AUTHENTICATION",
        "INVALID_REQUEST",
        "INVALID_RESPONSE",
        "PARSE_UNSUPPORTED",
        "REPLAY_MISS",
        "INVALID_SNAPSHOT",
    ],
)
async def test_never_retry_codes_also_never_fall_back(code: str) -> None:
    clock = FakeClock()
    fallback = SequenceCall(["fallback"])
    primary = SequenceCall([provider_error(code, retryable=True)])
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=FakeSleeper(clock),
        random=Random(1),
    )

    with pytest.raises(ProviderError, match=code.lower()):
        await executor.call(
            "search",
            primary,
            remaining_deadline=100.0,
            fallback_invocations=(fallback,),
        )
    assert primary.calls == 1
    assert fallback.calls == 0


@pytest.mark.asyncio
async def test_retryable_failure_falls_back_only_after_primary_retries() -> None:
    clock = FakeClock()
    primary = SequenceCall(
        [provider_error("NETWORK", retryable=True) for _ in range(3)]
    )
    fallback = SequenceCall(["fallback-ok"])
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=FakeSleeper(clock),
        random=Random(1),
    )

    result = await executor.call(
        "fetch",
        primary,
        remaining_deadline=100.0,
        fallback_invocations=(fallback,),
    )

    assert result == "fallback-ok"
    assert primary.calls == 3
    assert fallback.calls == 1
    assert len(executor.attempts) == 4


@pytest.mark.asyncio
async def test_strict_replay_disables_fallback_and_expired_deadline_skips_work() -> None:
    clock = FakeClock(now=10.0)
    sleeper = FakeSleeper(clock)
    primary = SequenceCall(
        [provider_error("NETWORK", retryable=True) for _ in range(3)]
    )
    fallback = SequenceCall(["forbidden"])
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=sleeper,
        random=Random(1),
        strict_replay=True,
    )

    with pytest.raises(ProviderError) as error:
        await executor.call(
            "fetch",
            primary,
            remaining_deadline=100.0,
            fallback_invocations=(fallback,),
        )
    assert error.value.code == "NETWORK"
    assert fallback.calls == 0

    never_called = SequenceCall(["late"])
    with pytest.raises(ProviderError) as deadline_error:
        await executor.call("fetch", never_called, remaining_deadline=clock.now)
    assert deadline_error.value.code == "TIMEOUT"
    assert never_called.calls == 0


@pytest.mark.asyncio
async def test_retry_allowance_is_shared_across_primary_and_fallback() -> None:
    clock = FakeClock()
    primary = SequenceCall(
        [provider_error("NETWORK", retryable=True) for _ in range(3)]
    )
    fallback = SequenceCall(
        [provider_error("UPSTREAM_5XX", retryable=True) for _ in range(3)]
    )
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=FakeSleeper(clock),
        random=Random(1),
    )

    with pytest.raises(ProviderError) as error:
        await executor.call(
            "fetch",
            primary,
            remaining_deadline=100.0,
            fallback_invocations=(fallback,),
        )

    assert error.value.code == "UPSTREAM_5XX"
    assert primary.calls == 3
    assert fallback.calls == 1
    assert sum(attempt.attempt_index > 0 for attempt in executor.attempts) == 2


@pytest.mark.asyncio
async def test_unfit_primary_retry_advances_to_eligible_fallback() -> None:
    clock = FakeClock(now=0.0)
    sleeper = FakeSleeper(clock)
    primary = SequenceCall(
        [provider_error("NETWORK", retryable=True, retry_after=10.0)]
    )
    fallback = SequenceCall(["fallback-ok"])
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=sleeper,
        random=Random(1),
    )

    result = await executor.call(
        "fetch",
        primary,
        remaining_deadline=1.0,
        fallback_invocations=(fallback,),
    )

    assert result == "fallback-ok"
    assert primary.calls == 1
    assert fallback.calls == 1
    assert sleeper.delays == []


@pytest.mark.asyncio
async def test_unfit_retry_without_fallback_preserves_original_error() -> None:
    clock = FakeClock(now=0.0)
    primary = SequenceCall(
        [provider_error("NETWORK", retryable=True, retry_after=10.0)]
    )
    executor = ProviderCallExecutor(
        policy=ProviderCallPolicy.defaults(),
        clock=clock,
        sleeper=FakeSleeper(clock),
        random=Random(1),
    )

    with pytest.raises(ProviderError) as error:
        await executor.call("fetch", primary, remaining_deadline=1.0)

    assert error.value.code == "NETWORK"
    assert clock.now == 0.0


@pytest.mark.asyncio
async def test_final_jittered_delay_is_capped_by_policy_maximum() -> None:
    clock = FakeClock(now=0.0)
    sleeper = FakeSleeper(clock)
    policy = ProviderCallPolicy(
        default_timeout_seconds=ProviderCallPolicy.defaults().default_timeout_seconds,
        max_retries=1,
        base_delay_seconds=4.0,
        max_delay_seconds=4.0,
        jitter_ratio=1.0,
    )
    call = SequenceCall([provider_error("NETWORK", retryable=True), "ok"])
    executor = ProviderCallExecutor(
        policy=policy,
        clock=clock,
        sleeper=sleeper,
        random=Random(0),
    )

    assert await executor.call("fetch", call, remaining_deadline=100.0) == "ok"
    assert sleeper.delays == [4.0]
