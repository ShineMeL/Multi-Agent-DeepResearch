from deepresearch.retrieval import normalize_text, sha256_text


def test_text_normalization_and_hash_are_unicode_stable() -> None:
    left = normalize_text("  A\t  B  \r\n多模态\u00a0 Agent  ")
    right = normalize_text("A B\n多模态 Agent")

    assert left == right
    assert sha256_text(left) == sha256_text(right)


def test_normalize_text_uses_nfc_without_removing_paragraph_boundaries() -> None:
    decomposed = "Cafe\u0301\r\n\r\n  second\t paragraph "

    assert normalize_text(decomposed) == "Café\n\nsecond paragraph"


def test_sha256_text_hashes_utf8_bytes() -> None:
    assert sha256_text("雪") == "53058fe2e03c00d93c4ba9361889dd7569d7d7b9674ed172266ee66265208cc6"
