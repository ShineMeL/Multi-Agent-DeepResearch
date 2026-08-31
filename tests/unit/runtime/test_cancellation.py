from concurrent.futures import ThreadPoolExecutor

import pytest

from deepresearch.runtime import CancellationToken, OperationCancelled


def test_token_is_monotonic_and_raises_stable_public_error() -> None:
    token = CancellationToken()

    assert token.is_cancelled() is False
    token.raise_if_cancelled()
    token.cancel()
    token.cancel()

    assert token.is_cancelled() is True
    with pytest.raises(OperationCancelled) as error:
        token.raise_if_cancelled()
    assert error.value.code == "CANCELLED"


def test_cancel_is_thread_safe_under_concurrent_calls() -> None:
    token = CancellationToken()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = tuple(pool.map(lambda _: token.cancel(), range(200)))

    assert results == (None,) * 200
    assert token.is_cancelled() is True
