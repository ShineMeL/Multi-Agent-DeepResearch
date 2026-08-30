import hashlib
import re
import unicodedata
from dataclasses import dataclass
from difflib import SequenceMatcher

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_TOKEN_PATTERN = re.compile(r"\w+", flags=re.UNICODE)


def exact_duplicate_key(parsed_content_hash: str) -> str:
    if _SHA256_PATTERN.fullmatch(parsed_content_hash) is None:
        raise ValueError("parsed_content_hash must be a lowercase 64-character SHA-256")
    return parsed_content_hash


def simhash64(text: str) -> int:
    tokens = _TOKEN_PATTERN.findall(unicodedata.normalize("NFKC", text).casefold())
    if not tokens:
        return 0
    weights = [0] * 64
    for token in tokens:
        fingerprint = int.from_bytes(hashlib.sha256(token.encode("utf-8")).digest()[:8], "big")
        for bit in range(64):
            weights[bit] += 1 if fingerprint & (1 << bit) else -1
    result = 0
    for bit, weight in enumerate(weights):
        if weight >= 0:
            result |= 1 << bit
    return result


def hamming_distance(left: int, right: int) -> int:
    if not 0 <= left < 2**64 or not 0 <= right < 2**64:
        raise ValueError("SimHash values must be unsigned 64-bit integers")
    return (left ^ right).bit_count()


def normalize_title(title: str) -> str:
    normalized = unicodedata.normalize("NFKC", title).casefold()
    words = _TOKEN_PATTERN.findall(normalized)
    return " ".join(words)


def normalized_title_similarity(left: str, right: str) -> float:
    normalized_left = normalize_title(left)
    normalized_right = normalize_title(right)
    if not normalized_left and not normalized_right:
        return 1.0
    if not normalized_left or not normalized_right:
        return 0.0
    return SequenceMatcher(None, normalized_left, normalized_right, autojunk=False).ratio()


@dataclass(frozen=True, slots=True)
class NearDuplicateSignals:
    left_simhash: int
    right_simhash: int
    simhash_distance: int
    title_similarity: float
    is_near_duplicate: bool


def near_duplicate_signals(
    *,
    left_text: str,
    right_text: str,
    left_title: str,
    right_title: str,
    max_hamming_distance: int = 3,
    min_title_similarity: float = 0.90,
) -> NearDuplicateSignals:
    if max_hamming_distance < 0 or not 0.0 <= min_title_similarity <= 1.0:
        raise ValueError("near-duplicate thresholds are invalid")
    left_simhash = simhash64(left_text)
    right_simhash = simhash64(right_text)
    distance = hamming_distance(left_simhash, right_simhash)
    title_similarity = normalized_title_similarity(left_title, right_title)
    return NearDuplicateSignals(
        left_simhash=left_simhash,
        right_simhash=right_simhash,
        simhash_distance=distance,
        title_similarity=title_similarity,
        is_near_duplicate=(
            distance <= max_hamming_distance and title_similarity >= min_title_similarity
        ),
    )


__all__ = [
    "NearDuplicateSignals",
    "exact_duplicate_key",
    "hamming_distance",
    "near_duplicate_signals",
    "normalize_title",
    "normalized_title_similarity",
    "simhash64",
]
