from collections.abc import Callable
from typing import TypeAlias

ContentBoundary: TypeAlias = Callable[[str], str]  # noqa: UP040 - exact public contract


def identity_content_boundary(text: str) -> str:
    return text


__all__ = ["ContentBoundary", "identity_content_boundary"]
