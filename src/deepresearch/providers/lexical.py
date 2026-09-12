"""Lightweight lexical similarity for local live demos, not a semantic model.

Token feature hashing preserves shared terms (unlike whole-text test hashes).
The explicit identity keeps this baseline separate from locked neural embeddings
and from benchmark claims about semantic retrieval quality.
"""

import hashlib
import math
import re
from collections import Counter
from typing import override

from deepresearch.retrieval import normalize_text

from .embeddings import DeterministicHashTextEmbedder

_TOKENS = re.compile(r"[\u3400-\u9fff]+|[^\W_]+", re.UNICODE)


class LexicalHashTextEmbedder(DeterministicHashTextEmbedder):
    provider_id = "lexical-hash"
    model_id = "lexical-hash-v1"
    model_revision = "1"

    @override
    def _vector(self, text: str) -> tuple[float, ...]:
        terms: list[str] = []
        for token in _TOKENS.findall(normalize_text(text).casefold()):
            if "\u3400" <= token[0] <= "\u9fff" and len(token) > 1:
                terms.extend(token[i : i + 2] for i in range(len(token) - 1))
            else:
                terms.append(token)
        values = [0.0] * self.dimension
        for term, count in Counter(terms or ["<empty>"]).items():
            index = int.from_bytes(hashlib.sha256(term.encode("utf-8")).digest()[:8])
            values[index % self.dimension] += 1.0 + math.log(count)
        norm = math.hypot(*values)
        return tuple(value / norm for value in values)
