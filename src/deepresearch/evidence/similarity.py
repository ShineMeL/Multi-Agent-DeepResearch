from __future__ import annotations

import math
import time
from collections.abc import Sequence

from deepresearch.domain import EvidenceSpan, RerankScore
from deepresearch.providers import Deadline, ProviderError, TextEmbedder, validate_embeddings
from deepresearch.runtime import CancellationToken


def _check_boundary(
    embedder: TextEmbedder,
    *,
    deadline: Deadline,
    cancellation_token: CancellationToken,
) -> None:
    cancellation_token.raise_if_cancelled()
    if not math.isfinite(deadline):
        raise ProviderError(
            code="INVALID_REQUEST",
            provider=embedder.provider_id,
            operation="embed",
            public_message="deadline must be a finite absolute monotonic deadline",
            retryable=False,
        )
    if time.monotonic() >= deadline:
        raise ProviderError(
            code="TIMEOUT",
            provider=embedder.provider_id,
            operation="embed",
            public_message="embedding deadline expired",
            retryable=False,
        )


def _invalid_embeddings(embedder: TextEmbedder) -> ProviderError:
    return ProviderError(
        code="INVALID_RESPONSE",
        provider=embedder.provider_id,
        operation="embed",
        public_message="embedding response is invalid",
        retryable=False,
    )


def _cosine(left: Sequence[float], right: Sequence[float]) -> float:
    left_norm = math.hypot(*left)
    right_norm = math.hypot(*right)
    if left_norm == 0.0 or right_norm == 0.0:
        raise ArithmeticError
    return math.fsum(
        (left_value / left_norm) * (right_value / right_norm)
        for left_value, right_value in zip(left, right, strict=True)
    )


class SimilarityRanker:
    ranker_id = "R1"

    def __init__(self, embedder: TextEmbedder) -> None:
        self.embedder = embedder

    async def score(
        self,
        information_need: str,
        evidence_spans: Sequence[EvidenceSpan],
        *,
        deadline: Deadline,
        cancellation_token: CancellationToken,
    ) -> list[RerankScore]:
        _check_boundary(
            self.embedder,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        if not evidence_spans:
            _check_boundary(
                self.embedder,
                deadline=deadline,
                cancellation_token=cancellation_token,
            )
            return []
        if not information_need.strip():
            raise ProviderError(
                code="INVALID_REQUEST",
                provider=self.embedder.provider_id,
                operation="embed",
                public_message="information need must not be empty",
                retryable=False,
            )
        texts = [information_need, *(span.excerpt for span in evidence_spans)]
        _check_boundary(
            self.embedder,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        vectors = await self.embedder.embed(
            texts,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        _check_boundary(
            self.embedder,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        try:
            validated = validate_embeddings(texts, vectors)
            query_vector, *evidence_vectors = validated
            scores: list[RerankScore] = []
            for span, vector in zip(evidence_spans, evidence_vectors, strict=True):
                cosine = _cosine(query_vector, vector)
                relevance = min(1.0, max(0.0, (cosine + 1.0) / 2.0))
                scores.append(
                    RerankScore(
                        evidence_id=span.evidence_id,
                        total=relevance,
                        feature_scores={"relevance": relevance},
                        model_id=self.embedder.model_id,
                        prompt_version=None,
                    )
                )
        except (ArithmeticError, TypeError, ValueError):
            raise _invalid_embeddings(self.embedder) from None
        _check_boundary(
            self.embedder,
            deadline=deadline,
            cancellation_token=cancellation_token,
        )
        return sorted(scores, key=lambda item: (-item.total, item.evidence_id))


__all__ = ["SimilarityRanker"]
