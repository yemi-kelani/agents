"""Drive a research run in the terminal — no server, no browser.

    python run.py "your question here"          # FAST  (~45s)
    python run.py --full "your question here"   # FULL  (1-4 min)

Must run with the project venv's python, from the `researcher/` directory.
"""
from __future__ import annotations

import asyncio
import sys
import uuid

from config import load_config
from graph import build_graph_from_config
from state import FAST, FULL
from terminal import TerminalUI, stream_run


async def main() -> int:
    args = sys.argv[1:]
    full = "--full" in args
    question = " ".join(a for a in args if a != "--full").strip()

    if not question:
        print(__doc__)
        return 2

    # served=False, so the builder supplies a checkpointer of its own — which
    # the graph needs regardless, because `ask` interrupts and an interrupt
    # with nowhere to pause is an exception rather than a question.
    graph = build_graph_from_config(load_config())

    state = await stream_run(
        graph,
        {
            "question": question,
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
