from __future__ import annotations

import time
from collections.abc import Sequence

import pytest

from deepresearch.domain import EvidenceSpan, HtmlLocator
from deepresearch.evidence import SimilarityRanker
from deepresearch.providers import ProviderError
from deepresearch.retrieval import sha256_text
from deepresearch.runtime import CancellationToken, OperationCancelled


def evidence(evidence_id: str, excerpt: str) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id,
        source_id=f"source-{evidence_id}",
        locator=HtmlLocator(
            paragraph_id="p-1", start_char=0, end_char=len(excerpt)
        ),
        excerpt=excerpt,
        excerpt_hash=sha256_text(excerpt),
        language="en",
        information_need_ids=("need-1",),
    )


class FakeEmbedder:
    provider_id = "fake-embedder"
    model_id = "locked-model"
    model_revision = "revision-1"
    snapshot_sha256 = "a" * 64

    def __init__(
        self,
        vectors: Sequence[Sequence[float]],
        *,
        cancel_after: CancellationToken | None = None,
    ) -> None:
        self.vectors = vectors
        self.cancel_after = cancel_after
        self.calls = 0

    async def embed(
        self,
        texts: Sequence[str],
        *,
        deadline: float,
        cancellation_token: CancellationToken,
    ) -> tuple[tuple[float, ...], ...]:
        del texts, deadline, cancellation_token
        self.calls += 1
        if self.cancel_after is not None:
            self.cancel_after.cancel()
        return tuple(tuple(vector) for vector in self.vectors)


def future_deadline() -> float:
    return time.monotonic() + 30.0


@pytest.mark.asyncio
async def test_similarity_ranker_orders_by_mapped_cosine_then_id() -> None:
    spans = (evidence("ev-2", "second"), evidence("ev-1", "first"))
    embedder = FakeEmbedder(((1.0, 0.0), (1.0, 0.0), (1.0, 0.0)))

    scores = await SimilarityRanker(embedder).score(
        "planner optimization",
        spans,
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert [score.evidence_id for score in scores] == ["ev-1", "ev-2"]
    assert [score.total for score in scores] == [1.0, 1.0]
    assert all(score.feature_scores == {"relevance": 1.0} for score in scores)
    assert all(score.model_id == "locked-model" for score in scores)


@pytest.mark.asyncio
async def test_similarity_ranker_maps_negative_and_orthogonal_cosines_exactly() -> None:
    spans = (evidence("ev-negative", "negative"), evidence("ev-zero", "zero"))
    embedder = FakeEmbedder(((2.0, 0.0), (-3.0, 0.0), (0.0, 4.0)))

    scores = await SimilarityRanker(embedder).score(
        "need",
        spans,
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    by_id = {score.evidence_id: score.total for score in scores}
    assert by_id == {"ev-negative": 0.0, "ev-zero": 0.5}


@pytest.mark.asyncio
async def test_empty_evidence_avoids_embedding_work() -> None:
    embedder = FakeEmbedder(())

    result = await SimilarityRanker(embedder).score(
        "need",
        (),
        deadline=future_deadline(),
        cancellation_token=CancellationToken(),
    )

    assert result == []
    assert embedder.calls == 0


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "vectors",
    [
        ((1.0, 0.0),),
        ((1.0, 0.0), (1.0,)),
        ((), ()),
        ((1.0, float("nan")), (1.0, 0.0)),
        ((0.0, 0.0), (1.0, 0.0)),
        ((1.0, 0.0), (0.0, 0.0)),
    ],
)
async def test_invalid_embeddings_raise_sanitized_provider_error(
    vectors: Sequence[Sequence[float]],
) -> None:
    embedder = FakeEmbedder(vectors)

    with pytest.raises(ProviderError) as error:
        await SimilarityRanker(embedder).score(
            "need",
            (evidence("ev-1", "excerpt"),),
            deadline=future_deadline(),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_RESPONSE"
    assert error.value.provider == "fake-embedder"
    assert error.value.operation == "embed"
    assert error.value.retryable is False
    assert error.value.public_message == "embedding response is invalid"
    assert "nan" not in str(error.value).lower()


@pytest.mark.asyncio
async def test_cancellation_after_embed_prevents_score_publication() -> None:
    token = CancellationToken()
    embedder = FakeEmbedder(((1.0,), (1.0,)), cancel_after=token)

    with pytest.raises(OperationCancelled):
        await SimilarityRanker(embedder).score(
            "need",
            (evidence("ev-1", "excerpt"),),
            deadline=future_deadline(),
            cancellation_token=token,
        )


@pytest.mark.asyncio
async def test_invalid_deadline_fails_before_embedding() -> None:
    embedder = FakeEmbedder(((1.0,), (1.0,)))

    with pytest.raises(ProviderError) as error:
        await SimilarityRanker(embedder).score(
            "need",
            (evidence("ev-1", "excerpt"),),
            deadline=float("inf"),
            cancellation_token=CancellationToken(),
        )

    assert error.value.code == "INVALID_REQUEST"
    assert embedder.calls == 0
