from __future__ import annotations

import contextlib
import logging
import re
import tempfile
from typing import Iterator

# `log` imports `scrub` from here, so this module takes the stdlib logger
# directly rather than `log.get_logger` — importing back would be a cycle.
logger = logging.getLogger(__name__)

_SECRET = re.compile(r"(sk-|key-)[A-Za-z0-9_\-]{8,}")

_TRUE = {"1", "true", "t", "yes", "y", "on"}
_FALSE = {"0", "false", "f", "no", "n", "off"}


def parse_int(value, default: int | None = None) -> int | None:
    """Coerce an env value to int, falling back to `default` when it isn't one."""
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        logger.error("Could not parse %r as an integer; using %r", value, default)
        return default


def parse_boolean(value, default: bool | None = None) -> bool | None:
    """Coerce an env value to bool, falling back to `default` when it isn't one."""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in _TRUE:
        return True
    if normalized in _FALSE:
        return False
    logger.error("Could not parse %r as a boolean; using %r", value, default)
    return default


def scrub(s: str) -> str:
    """CLIs echo the key on some auth failures. Never log it raw."""
    return _SECRET.sub("[REDACTED]", s)


@contextlib.contextmanager
def maybe_home(isolate: bool) -> Iterator[str | None]:
    """Yield a throwaway HOME when `isolate`, else None (use the real one)."""
    if not isolate:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="cli-home-") as d:
        yield d
