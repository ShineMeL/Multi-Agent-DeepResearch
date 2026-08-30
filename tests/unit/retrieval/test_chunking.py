from itertools import pairwise

from deepresearch.retrieval import chunk_text, sha256_text


def test_chunk_text_keeps_short_normalized_text_as_one_exact_source_slice() -> None:
    text = "第一段。\n\nSecond paragraph."

    chunks = chunk_text(text)

    assert len(chunks) == 1
    assert chunks[0].text == text
    assert (chunks[0].start_char, chunks[0].end_char) == (0, len(text))
    assert chunks[0].text_hash == sha256_text(text)


def test_chunk_text_prefers_paragraph_boundaries_and_maps_unicode_offsets() -> None:
    first = "雪" * 700
    second = "火" * 350
    text = f"{first}\n\n{second}"

    chunks = chunk_text(text, target_size=900, max_size=1200, overlap=120)

    assert len(chunks) == 2
    assert chunks[0].end_char == len(first)
    assert chunks[0].text == text[chunks[0].start_char : chunks[0].end_char]
    assert chunks[1].text == text[chunks[1].start_char : chunks[1].end_char]
    assert chunks[1].start_char == chunks[0].end_char - 120
    assert chunks[-1].end_char == len(text)


def test_chunk_text_hard_splits_long_paragraph_with_progress_and_maximum() -> None:
    text = "x" * 3_050

    chunks = chunk_text(text)

    assert len(chunks) >= 3
    assert all(0 < chunk.end_char - chunk.start_char <= 1_200 for chunk in chunks)
    assert all(left.start_char < right.start_char for left, right in pairwise(chunks))
    assert all(
        right.start_char == left.end_char - 120 for left, right in pairwise(chunks)
    )
    assert chunks[-1].end_char == len(text)


def test_chunk_text_rejects_non_normalized_text_and_invalid_limits() -> None:
    for kwargs in (
        {"target_size": 0},
        {"target_size": 1_201, "max_size": 1_200},
        {"overlap": 1_200, "max_size": 1_200},
    ):
        try:
            chunk_text("text", **kwargs)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid chunk limits accepted: {kwargs}")

    try:
        chunk_text(" text ")
    except ValueError as error:
        assert "normalized" in str(error)
    else:
        raise AssertionError("non-normalized text accepted")
