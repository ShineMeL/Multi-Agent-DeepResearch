import re
from typing import Annotated, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .normalize import normalize_text, sha256_text

DEFAULT_TARGET_SIZE = 900
DEFAULT_MAX_SIZE = 1_200
DEFAULT_OVERLAP = 120


class TextChunk(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    text: str
    start_char: Annotated[int, Field(ge=0)]
    end_char: Annotated[int, Field(gt=0)]
    text_hash: str

    @model_validator(mode="after")
    def validate_integrity(self) -> Self:
        if self.end_char <= self.start_char:
            raise ValueError("end_char must be greater than start_char")
        if len(self.text) != self.end_char - self.start_char:
            raise ValueError("chunk text length does not match source offsets")
        if self.text_hash != sha256_text(self.text):
            raise ValueError("text_hash does not match chunk text")
        return self


def _paragraph_ends(text: str) -> tuple[int, ...]:
    return tuple(match.start() for match in re.finditer(r"\n{2,}", text))


def chunk_text(
    normalized_text: str,
    *,
    target_size: int = DEFAULT_TARGET_SIZE,
    max_size: int = DEFAULT_MAX_SIZE,
    overlap: int = DEFAULT_OVERLAP,
) -> tuple[TextChunk, ...]:
    """Chunk normalized text using code-point half-open source offsets."""
    if target_size <= 0 or max_size <= 0 or target_size > max_size:
        raise ValueError("target_size must be positive and no greater than max_size")
    if overlap < 0 or overlap >= max_size:
        raise ValueError("overlap must be non-negative and less than max_size")
    if normalize_text(normalized_text) != normalized_text:
        raise ValueError("chunk_text requires normalized text")
    if not normalized_text:
        return ()

    boundaries = _paragraph_ends(normalized_text)
    chunks: list[TextChunk] = []
    start = 0
    text_length = len(normalized_text)
    while start < text_length:
        remaining = text_length - start
        if remaining <= target_size:
            end = text_length
        else:
            target_end = start + target_size
            hard_end = min(start + max_size, text_length)
            before_target = [point for point in boundaries if start < point <= target_end]
            if before_target:
                end = before_target[-1]
            else:
                after_target = [point for point in boundaries if target_end < point <= hard_end]
                end = after_target[0] if after_target else hard_end
        if end <= start:
            end = min(start + max_size, text_length)
        chunk = normalized_text[start:end]
        chunks.append(
            TextChunk(
                text=chunk,
                start_char=start,
                end_char=end,
                text_hash=sha256_text(chunk),
            )
        )
        if end == text_length:
            break
        next_start = end - overlap
        start = next_start if next_start > start else end
    return tuple(chunks)


__all__ = [
    "DEFAULT_MAX_SIZE",
    "DEFAULT_OVERLAP",
    "DEFAULT_TARGET_SIZE",
    "TextChunk",
    "chunk_text",
]
