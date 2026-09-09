"""Drive a research run in the terminal — no server, no browser.

    python run.py "your question here"          # FAST  (~45s)
    python run.py --full "your question here"   # FULL  (1-4 min)
    python run.py -v "your question here"       # + what each tool answered
    python run.py -vv "your question here"      # + what each tool was asked

The tool log goes to stderr while the report goes to stdout, so

    python run.py -v "your question here" 2>debug.log

leaves a readable report on screen and the full trace in a file. Surfaces that
never see this argv — `langgraph dev`, the eval harness (11) — read
`RESEARCHER_LOG_LEVEL` instead.

Must run with the project venv's python, from the `researcher/` directory.
"""
from __future__ import annotations

import asyncio
import re
import sys
import uuid

from config import load_config
from graph import build_graph_from_config
from state import FAST, FULL
from terminal import TerminalUI, stream_run
from toollog import configure, level_for

__all__ = ["level_for", "main", "question", "verbosity"]

FULL_FLAG = "--full"
VERBOSE_FLAG = re.compile(r"-v+$")
"""Matched rather than compared, so `-vv` is one argument and not a typo. It
also keeps the *word* "vs" — which turns up in real research questions — from
being read as a flag."""


def verbosity(args: list[str]) -> int:
    """How many `v`s were asked for, across however many flags carried them."""
    return sum(len(a) - 1 for a in args if VERBOSE_FLAG.match(a))


def question(args: list[str]) -> str:
    """Everything that was not a flag, joined back into the research question.

    The join is why this is a function: `run.py` reconstructs the question from
    whatever is left of `argv`, so a flag that is not removed here does not
    cause an error — it silently becomes part of what gets researched.
    """
    return " ".join(
        a for a in args if a != FULL_FLAG and not VERBOSE_FLAG.match(a)
    ).strip()


async def main() -> int:
    args = sys.argv[1:]
    full = FULL_FLAG in args
    asked = question(args)

    if not asked:
        print(__doc__)
        return 2

    # Before the graph is built: `build_graph_from_config` constructs the
    # backends, and a misconfigured one raises there (04) — which is exactly the
    # failure worth having the log switched on for.
    configure(level_for(verbosity(args)))

    # served=False, so the builder supplies a checkpointer of its own — which
    # the graph needs regardless, because `ask` interrupts and an interrupt
    # with nowhere to pause is an exception rather than a question.
    graph = build_graph_from_config(load_config())

    state = await stream_run(
        graph,
        {
            "question": asked,
            # The clarification latch (spec 02). Left False the run can stop and
            # ask, and this script has no resume plumbing to answer it with.
            "clarified": True,
            "budget": FULL if full else FAST,
        },
        {"configurable": {"thread_id": f"terminal-{uuid.uuid4().hex[:8]}"}},
        ui=TerminalUI(),
    )

    claims = state.get("claims") or []
    if claims:
        tally: dict[str, int] = {}
        for c in claims:
            tally[c.verdict] = tally.get(c.verdict, 0) + 1
        print("\n\n" + "  ".join(f"{v}: {n}" for v, n in sorted(tally.items())))

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
