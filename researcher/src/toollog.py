"""What each tool was asked and what it answered — the debug channel.

Deliberately *not* the `trace`/`describe` channel (12). That one is the curated
projection a human reads while a run happens, kept short on purpose — a message
per fetch turns the transcript into a wall — and it renders model-written text
through `safe_render` because untrusted page content lands there. Tool payloads
are the opposite shape: long, uncurated, and wanted only when something has gone
wrong. Two audiences, two channels.

So this writes to `logging` instead: silent at the default level, switched on
per run (`run.py -v`, or `RESEARCHER_LOG_LEVEL`), and on **stderr** while the
terminal UI writes stdout — which is what makes `run.py -v "..." 2>debug.log`
separate the report from the diagnostics.

One context manager covers every tool surface, sync and async alike, because an
`await` inside the `with` block is timed by it just as well as a plain call:

    with tool_call("search", query=q) as call:
        hits = await backend.search(q, k)
        call.set(f"{len(hits)} hits", payload=[h.url for h in hits])

It observes and never intervenes. A call that raises is logged and re-raised
unchanged — a logger that swallowed an exception would turn a dead backend into
a silent empty result, which is the exact failure the router (04) exists to
distinguish from a real one.
"""
from __future__ import annotations

import logging
import sys
from contextlib import contextmanager
from time import perf_counter

from settings import DEFAULT_LOG_LEVEL, settings

ROOT_LOGGER_NAME = "researcher"
LOGGER_NAME = f"{ROOT_LOGGER_NAME}.tools"

ENV_LOG_LEVEL = "RESEARCHER_LOG_LEVEL"
"""How the surfaces that never see `run.py`'s argv turn this on — `langgraph
dev` and the eval harness (11) among them. Named here for readers, but *read*
through `settings`, which is the one module that knows a variable's name."""

LEVELS = (DEFAULT_LOG_LEVEL, "INFO", "DEBUG")
"""Silent, then what the tools answered, then what they were asked."""

logger = logging.getLogger(LOGGER_NAME)

PREVIEW_CHARS = 240
"""Enough to recognise what came back, not enough to bury the next line. The
count of what was cut is kept, because "the page was 400KB" is itself the answer
to a whole class of extraction bug."""

REDACTED = "***"

SECRET_HINTS = ("key", "token", "secret", "password", "authorization")
"""Matched as substrings of the argument *name*. Backend options travel into
`tool_call` as keyword arguments and one of them is an API key; a debug log that
writes a live credential to a file is a worse bug than the one being debugged."""


def preview(value) -> str:
    """One line standing in for a payload of any size.

    Flattened, because a log line that spans a fetched article is not a log
    line, and the surrounding format puts the tool's name at the front of it.
    """
    text = ", ".join(str(v) for v in value) if isinstance(value, (list, tuple)) \
        else str(value)
    text = " ".join(text.split())

    cut = len(text) - PREVIEW_CHARS
    return f"{text[:PREVIEW_CHARS]}…(+{cut} chars)" if cut > 0 else text


class _Call:
    """The outcome slot for one wrapped call.

    Handed to the `with` body rather than read off a return value, because the
    interesting number is usually derived (how many hits, how many characters)
    rather than the returned object itself.
    """

    __slots__ = ("summary", "payload")

    def __init__(self):
        self.summary = ""
        self.payload = None

    def set(self, summary: str, payload=None) -> None:
        self.summary = summary
        self.payload = payload

    def rendered(self) -> str:
        parts = [p for p in (self.summary,) if p]
        if self.payload is not None:
            parts.append(preview(self.payload))
        return "  ".join(parts)


@contextmanager
def tool_call(name: str, passthrough: tuple = (), /, **args):
    """Log one tool invocation and whatever it answered.

    Arguments go out at DEBUG and the outcome at INFO, so `-v` answers "what did
    the tools return?" without also dumping every prompt. A failure is WARNING:
    the one thing worth seeing when nothing was turned on.

    `passthrough` names the exception types that are control flow rather than
    fault — `interrupt()` leaves `ask_user` by raising, and a warning per
    clarification would train everyone to ignore the line that means something
    really broke. It is positional-only because `**args` are chosen by a model,
    and a tool is entitled to an argument called `passthrough`.
    """
    logger.debug("→ %s(%s)", name, _rendered_args(args))
    started = perf_counter()
    call = _Call()

    try:
        yield call
    except BaseException as e:
        if isinstance(e, passthrough):
            logger.debug("· %s %s — %s", name, _elapsed(started), type(e).__name__)
        else:
            logger.warning("✗ %s %s — %s: %s", name, _elapsed(started),
                           type(e).__name__, e)
        raise

    logger.info("← %s %s %s", name, _elapsed(started), call.rendered())


def _rendered_args(args: dict) -> str:
    return ", ".join(f"{k}={REDACTED if _secret(k) else preview(v)}"
                     for k, v in args.items())


def _secret(name: str) -> bool:
    return any(hint in name.lower() for hint in SECRET_HINTS)


def _elapsed(started: float) -> str:
    return f"{(perf_counter() - started) * 1000:.0f}ms"


def level_for(verbosity: int) -> str:
    """`-v` -> INFO, `-vv` -> DEBUG. Clamped, so `-vvvv` is not an error: a
    person reaching for a fourth `v` wants everything, not a usage message."""
    return LEVELS[min(max(verbosity, 0), len(LEVELS) - 1)]


def configure(level: str | int | None = None) -> None:
    """Point the tool log at stderr and set how much of it to emit.

    **stderr, not stdout.** The terminal UI (12) streams the report to stdout,
    so this is what makes `run.py -v "..." 2>debug.log` keep a readable report
    and a full diagnostic trace at the same time.

    Idempotent: an entry point that configures twice gets one handler, not two
    copies of every line. `propagate` is turned off for the same reason — the
    root logger's fallback would print every warning a second time.

    An unusable level is ignored rather than raised on: a typo in an environment
    variable must not be able to end a research run.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)

    try:
        root.setLevel(level or settings().log_level)
    except ValueError:
        root.setLevel(LEVELS[0])

    if not any(getattr(h, "_researcher", False) for h in root.handlers):
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(message)s"))
        handler._researcher = True
        root.addHandler(handler)
        root.propagate = False
