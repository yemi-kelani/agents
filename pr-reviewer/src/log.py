from __future__ import annotations

import logging
import os
import sys

from utilities import scrub

_CONFIGURED = False
_DEFAULT_FORMAT = "%(asctime)s %(levelname)-8s %(name)s: %(message)s"


class ScrubbingFormatter(logging.Formatter):
    """Format normally, then strip anything that looks like an API key.

    Secrets reach the log through interpolated args and tracebacks, not just
    the format string, so the redaction happens on the finished line.
    """

    def format(self, record: logging.LogRecord) -> str:
        return scrub(super().format(record))


def configure(level: str | int | None = None, stream=sys.stderr) -> None:
    """Attach one stderr handler to the package logger. Idempotent.

    Level comes from `level`, else `LOG_LEVEL`, else INFO. Logs go to stderr so
    they never mix into anything the tool writes on stdout.
    """
    global _CONFIGURED

    root = logging.getLogger(__package__ or "pr_reviewer")
    if level is None:
        level = os.getenv("LOG_LEVEL", "INFO")
    root.setLevel(level if isinstance(level, int) else level.upper())

    if _CONFIGURED:
        return

    handler = logging.StreamHandler(stream)
    handler.setFormatter(ScrubbingFormatter(_DEFAULT_FORMAT))
    root.addHandler(handler)
    root.propagate = False
    _CONFIGURED = True


def get_logger(name: str | None = None) -> logging.Logger:
    """The logger for a module: `get_logger(__name__)`. Configures on first use."""
    configure()
    base = __package__ or "pr_reviewer"
    if not name or name == base:
        return logging.getLogger(base)
    return logging.getLogger(name if name.startswith(f"{base}.") else f"{base}.{name}")
