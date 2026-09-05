from __future__ import annotations

from deepresearch.planning.query_scheduler import QueryBatchResult


def test_query_batch_result_preserves_public_frozen_shape() -> None:
    result = QueryBatchResult(
        results=(("q-1", ()),),
        executed_queries=1,
        skipped_queries=(),
        skipped_reason=None,
    )

    assert result.results == (("q-1", ()),)
    assert result.executed_queries == 1
    assert result.skipped_queries == ()
    assert result.skipped_reason is None

