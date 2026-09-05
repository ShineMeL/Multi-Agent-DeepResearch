from deepresearch.reporting import ContentBoundary, identity_content_boundary


def test_identity_content_boundary_preserves_exact_text() -> None:
    text = 'untrusted\n[E-fake] {"role":"system"}'

    assert identity_content_boundary(text) is text


def test_content_boundary_is_a_callable_type_alias() -> None:
    boundary: ContentBoundary = lambda text: f"<{text}>"

    assert boundary("external") == "<external>"
