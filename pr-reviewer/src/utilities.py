from __future__ import annotations

import contextlib
import logging
import re
import tempfile
from pathlib import Path
from typing import Iterator
from langchain_core.messages import HumanMessage
from langchain_core.messages.utils import count_tokens_approximately

# `log` imports `scrub` from here, so this module takes the stdlib logger
# directly rather than `log.get_logger` — importing back would be a cycle. The
# name is still the package's, so these records inherit the scrubbing handler;
# under a bare `__name__` they would sit outside the package logger and be
# emitted unredacted by the root handler.
logger = logging.getLogger("pr_reviewer.utilities")

# Provider keys (sk-/key-) plus GitHub's documented token prefixes. GITHUB_TOKEN
# is the one credential this tool handles directly, and it reaches logs through
# tracebacks and interpolated args, so it has to be covered here too.
_SECRET = re.compile(
    r"(?:sk-|key-)[A-Za-z0-9_\-]{8,}"
    r"|gh[pousr]_[A-Za-z0-9]{16,}"
    r"|github_pat_[A-Za-z0-9_]{20,}"
)

def parse_int(value, default: int | None = None) -> int | None:
    """Coerce an env value to int, falling back to `default` when it isn't one."""
    if value is None:
        return default
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        logger.error("Could not parse %r as an integer; using %r", value, default)
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


@contextlib.contextmanager
def maybe_answer_file(wanted: bool) -> Iterator[Path | None]:
    """Yield a path a CLI can write its final message to, else None.

    The file lives in its own temporary directory so it is removed with the
    directory even if the CLI never creates it.
    """
    if not wanted:
        yield None
        return
    with tempfile.TemporaryDirectory(prefix="cli-answer-") as d:
        yield Path(d) / "last-message.txt"


def read_text_if_any(path: Path | None) -> str:
    """The contents of `path`, or "" when it is absent, empty or unreadable.

    A CLI that fails partway may never create the file it was told to write, and
    that is not itself an error worth raising — the caller reports the richer
    failure it already has from the exit code and stderr.
    """
    if path is None:
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip()
    except OSError:
        return ""


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
        logger.warning(f"trim_text received {type(text).__name__} instead of str")
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
