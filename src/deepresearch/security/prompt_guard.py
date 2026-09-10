"""Label external material as data before Core serializes a prompt."""

from html import escape

UNTRUSTED_OPEN = "<untrusted_web_content>"
UNTRUSTED_CLOSE = "</untrusted_web_content>"


def wrap_untrusted_content(text: str) -> str:
    # Escape all markup, including mixed-case/whitespace closing tags and
    # forged system tags. Original evidence and artifacts are never modified.
    return f"{UNTRUSTED_OPEN}\n{escape(text, quote=False)}\n{UNTRUSTED_CLOSE}"
