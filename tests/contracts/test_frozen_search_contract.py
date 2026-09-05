from __future__ import annotations

from pathlib import Path

from deepresearch.providers.frozen_index import FrozenCorpusSnapshot
from deepresearch.providers.frozen_search import (
    FrozenCorpusFetcher,
    FrozenCorpusMaterializer,
    FrozenCorpusSearchProvider,
)
from deepresearch.providers.protocols import Fetcher, SearchProvider


def test_frozen_adapters_share_one_snapshot_identity() -> None:
    snapshot = FrozenCorpusSnapshot.load(
        Path(__file__).parents[1] / "fixtures" / "frozen_corpus" / "task-fixture",
        task_id="task-fixture",
    )
    search = FrozenCorpusSearchProvider(snapshot)
    fetch = FrozenCorpusFetcher(snapshot)
    materializer = FrozenCorpusMaterializer(snapshot)
    assert isinstance(search, SearchProvider)
    assert isinstance(fetch, Fetcher)
    assert search.snapshot is snapshot
    assert fetch.snapshot is snapshot
    assert materializer.snapshot is snapshot
