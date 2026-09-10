"""Recursive, nonmutating redaction for JSON-compatible public values."""

import re
from collections.abc import Collection, Mapping
from typing import cast
from urllib.parse import unquote_plus, urlsplit, urlunsplit

_MASK = "[REDACTED]"
_CREDENTIAL_NAME = (
    r"authorization|proxy-authorization|(?:x[-_])?api[-_]?key|password|secret|"
    r"(?:access|refresh)[-_]?token|token|cookie|set-cookie|signature|sig|"
    r"x-(?:amz|goog)-(?:signature|credential|security-token)|awsaccesskeyid|googleaccessid"
)
_SENSITIVE_KEY = re.compile(rf"(?i)^(?:{_CREDENTIAL_NAME})$")
_CREDENTIAL = re.compile(
    rf"(?i)(\b(?:{_CREDENTIAL_NAME})[\"']?\s*[:=]\s*)"
    r"(\"[^\"]*\"|'[^']*'|[^\s\"',;}&]+)"
)
_AUTH_TOKEN = re.compile(r"(?i)(\b(?:bearer|basic)\s+)[^\s\"',;}&]+")
_URL = re.compile(r"(?i)\bhttps?://[^\s<>\"']+")


def _redact_url(match: re.Match[str]) -> str:
    """Sanitize display URLs only; never use this to select a fetch target."""
    try:
        parts = urlsplit(match[0])
    except ValueError:
        return "[REDACTED_URL]"

    def query(text: str) -> str:
        fields: list[str] = []
        for field in text.split("&"):
            key, separator, value = field.partition("=")
            if separator and _SENSITIVE_KEY.fullmatch(unquote_plus(key)):
                value = _MASK
            fields.append(key + separator + value)
        return "&".join(fields)

    # Userinfo and signed query/fragment credentials are presentation-only
    # masks; unrelated escaped query values retain their exact spelling.
    netloc = "REDACTED@" + parts.netloc.rsplit("@", 1)[1] if "@" in parts.netloc else parts.netloc
    return urlunsplit((parts.scheme, netloc, parts.path, query(parts.query), query(parts.fragment)))


def _redact_assignment(match: re.Match[str]) -> str:
    value = match[2]
    quote = value[0] if value[0] in {'"', "'"} else ""
    return match[1] + quote + _MASK + quote


def redact(value: object, *, secrets: Collection[str]) -> object:
    ordered = sorted({item for item in secrets if item}, key=len, reverse=True)

    def clean(item: object) -> object:
        if isinstance(item, str):
            safe = item
            for secret in ordered:
                safe = safe.replace(secret, _MASK)
            safe = _URL.sub(_redact_url, safe)
            # Mask the token before an assignment can replace its scheme.
            safe = _AUTH_TOKEN.sub(lambda match: match[1] + _MASK, safe)
            return _CREDENTIAL.sub(_redact_assignment, safe)
        if isinstance(item, Mapping):
            return {
                str(clean(str(key))): _MASK if _SENSITIVE_KEY.fullmatch(str(key)) else clean(part)
                for key, part in cast(Mapping[object, object], item).items()
            }
        if isinstance(item, (list, tuple)):
            return [clean(part) for part in cast(list[object] | tuple[object, ...], item)]
        return item

    return clean(value)
