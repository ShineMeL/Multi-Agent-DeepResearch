"""Python 3.12 handler filter that never forwards original exception objects."""

import logging
from collections.abc import Collection
from copy import copy

from .redaction import redact


class RedactingFilter(logging.Filter):
    def __init__(self, *, secrets: Collection[str]) -> None:
        super().__init__()
        self._secrets = tuple(secrets)

    def filter(self, record: logging.LogRecord) -> logging.LogRecord:
        safe = copy(record)
        # Render first: preserves numeric/mapping %-formatting and covers
        # arbitrary objects in msg/args without retaining their credentials.
        safe.msg = str(redact(record.getMessage(), secrets=self._secrets))
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
