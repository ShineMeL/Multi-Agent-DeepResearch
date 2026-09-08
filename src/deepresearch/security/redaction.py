"""Recursive, nonmutating redaction for JSON-compatible public values."""

import re
from collections.abc import Collection, Mapping
from typing import cast

_MASK = "[REDACTED]"
_SENSITIVE_KEY = re.compile(
    r"(?i)^(?:authorization|proxy-authorization|(?:x[-_])?api[-_]?key|"
    r"password|secret|(?:access|refresh)[-_]?token|cookie|set-cookie)$"
)
_CREDENTIAL = re.compile(
    r"(?i)(\bbearer\s+|\b(?:[\w-]*api[_-]?key|access[_-]?token|"
    r"refresh[_-]?token|password|secret)\s*[:=]\s*[\"']?)[^\s\"',;}]+"
)


def redact(value: object, *, secrets: Collection[str]) -> object:
    ordered = sorted({item for item in secrets if item}, key=len, reverse=True)

    def clean(item: object) -> object:
        if isinstance(item, str):
            safe = item
            for secret in ordered:
                safe = safe.replace(secret, _MASK)
            return _CREDENTIAL.sub(lambda match: match[1] + _MASK, safe)
        if isinstance(item, Mapping):
            return {
                str(clean(str(key))): _MASK if _SENSITIVE_KEY.fullmatch(str(key)) else clean(part)
                for key, part in cast(Mapping[object, object], item).items()
            }
        if isinstance(item, (list, tuple)):
            return [clean(part) for part in cast(list[object] | tuple[object, ...], item)]
        return item

    return clean(value)
