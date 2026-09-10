"""Python 3.12 handler filter that never forwards original exception objects."""

import logging
from collections.abc import Collection
from copy import copy
from typing import cast

from .redaction import redact


class RedactingFilter(logging.Filter):
    def __init__(self, *, secrets: Collection[str]) -> None:
        super().__init__()
        self._secrets = tuple(secrets)

    def filter(self, record: logging.LogRecord) -> logging.LogRecord:
        safe = copy(record)
        # Preserve structured sensitive keys until recursive redaction has
        # inspected them. Retain the outer %-format argument tuple/mapping.
        # String templates must retain their placeholders until interpolation.
        safe.msg = (
            record.msg if isinstance(record.msg, str) else redact(record.msg, secrets=self._secrets)
        )
        if isinstance(record.args, tuple):
            safe.args = tuple(redact(arg, secrets=self._secrets) for arg in record.args)
        elif record.args:
            safe.args = cast(dict[str, object], redact(record.args, secrets=self._secrets))
        # The second pass covers object representations and interpolated text.
        safe.msg = str(redact(safe.getMessage(), secrets=self._secrets))
        safe.args = ()
        safe.message = safe.msg
        exception = (
            logging.Formatter().formatException(record.exc_info)
            if record.exc_info is not None
            else record.exc_text
        )
        safe.exc_text = None if exception is None else str(redact(exception, secrets=self._secrets))
        safe.exc_info = None
        if record.stack_info is not None:
            safe.stack_info = str(redact(record.stack_info, secrets=self._secrets))
        return safe
