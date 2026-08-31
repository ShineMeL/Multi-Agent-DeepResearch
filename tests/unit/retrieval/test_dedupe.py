from deepresearch.retrieval import (
    exact_duplicate_key,
    hamming_distance,
    near_duplicate_signals,
    normalize_title,
    simhash64,
)


def test_exact_duplicate_key_is_only_the_parsed_content_hash() -> None:
    digest = "a" * 64

    assert exact_duplicate_key(digest) == digest


def test_exact_duplicate_key_rejects_non_sha256_values() -> None:
    for value in ("", "A" * 64, "a" * 63, "z" * 64):
        try:
            exact_duplicate_key(value)
        except ValueError:
            pass
        else:
            raise AssertionError(f"invalid parsed content hash accepted: {value!r}")


def test_simhash_is_deterministic_and_hamming_distance_is_symmetric() -> None:
    left = simhash64("alpha beta beta gamma")
    right = simhash64("alpha beta delta gamma")

    assert left == simhash64("alpha beta beta gamma")
    assert 0 <= left < 2**64
    assert hamming_distance(left, right) == hamming_distance(right, left)


def test_title_normalization_and_similarity_ignore_case_width_and_punctuation() -> None:
    signals = near_duplicate_signals(
        left_text="alpha beta gamma",
        right_text="alpha beta gamma",
        left_title="Ａ Study: Results!",
        right_title="a study results",
    )

    assert normalize_title("Ａ Study: Results!") == "a study results"
    assert signals.left_simhash == signals.right_simhash
    assert signals.title_similarity == 1.0
    assert signals.is_near_duplicate


def test_near_duplicate_helper_reports_conflicting_content_without_merging_it() -> None:
    signals = near_duplicate_signals(
        left_text="the treatment increased survival by ten percent",
        right_text="the treatment decreased survival by ten percent",
        left_title="Trial results",
        right_title="Trial results",
    )

    assert signals.title_similarity == 1.0
    assert isinstance(signals.is_near_duplicate, bool)
    assert not hasattr(signals, "merged_text")
