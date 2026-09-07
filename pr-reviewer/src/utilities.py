from __future__ import annotations

import contextlib
import logging
import re
import tempfile
from typing import Iterator
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

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
        
def token_count(s: str) -> int:
    return count_tokens_approximately([HumanMessage(content=s)])


def trim_text(text: str, max_tokens: int = 50_000) -> tuple[str, str]:
    """Split `text` into the longest prefix within `max_tokens`, and the remainder.

    Token counting is approximate but monotonic in length, so a binary search on
    the split point finds the longest fitting prefix without counting every
    candidate.
    """
    s = text

    if not isinstance(text, str):
        logger.warning(f"'trim_text' method recieved text input with type '{type(text)}' instead of str.")
        s = str(text)

    len_text: int = len(s)
    if len_text == 0 or token_count(s) <= max_tokens:
        return s, ""

    # Invariant: `low` always fits, `high` never does.
    low, high = 0, len_text
    while high - low > 1:
        mid = (low + high) // 2
        if token_count(s[:mid]) <= max_tokens:
            low = mid
        else:
            high = mid

    return s[:low], s[low:]
