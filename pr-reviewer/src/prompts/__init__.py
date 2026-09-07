from __future__ import annotations

from functools import lru_cache
from pathlib import Path

_DIR = Path(__file__).parent


@lru_cache(maxsize=None)
def load(name: str) -> str:
    """Return the prompt named `name`, without the `.txt` suffix.

    Raises FileNotFoundError for an unknown name — a missing prompt is a wiring
    bug, and an empty string would reach the model as a blank instruction.
    """
    return (_DIR / f"{name}.txt").read_text(encoding="utf-8").strip()


PROMPT_NAMES = tuple(sorted(p.stem for p in _DIR.glob("*.txt")))
"""Every prompt shipped beside this module. Discovered rather than listed, so
adding a `.txt` needs no second edit here."""

for _name in PROMPT_NAMES:
    load(_name)
"""Read at import, which is what keeps `load` off the disk at runtime.

Extraction (06) and the verifier (07) both reach for a prompt from inside an
`async def`, and a synchronous read there stalls the event loop — under an ASGI
server the blocking-call detector raises outright. Warming the cache here costs
a few kilobytes once and makes every later call a dictionary lookup.
"""
