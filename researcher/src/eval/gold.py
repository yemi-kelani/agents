"""The hand-labelled ground truth — spec 11.

This is the part that cannot be automated away, and the part with the highest
leverage: every number the harness prints is a statement about how well the
verifier agrees with these labels, so a sloppy label is a wrong headline.

**The shipped set is synthetic and the report says so.** Its documents live in
`corpus/` and were written for it, over `example.com` URLs that are reserved
precisely so nothing here reads as a claim about anybody's product. That buys
two things the eval needs: the quotes are provably in their sources, and the
whole run reproduces offline. It buys nothing at all about the real world —
replace it with claims the agent actually made and labels you actually checked
before quoting a number, which is easy in practice: run the agent once, look at
what it cited, and label honestly. Its own mistakes are the best test data.

Build a replacement deliberately — roughly five clearly supported, five clearly
unsupported, and three to five hard cases (related-but-not-supporting,
correct-but-different-number, right-topic-wrong-subject). The hard cases are
what the verifier's judgment turns on; an all-easy set saturates and shows
nothing.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field

DEFAULT_GOLD_PATH = Path(__file__).parent / "gold.jsonl"
CORPUS_DIR = Path(__file__).parent / "corpus"


class GoldClaim(BaseModel):
    """One labelled (claim, source, quote) triple."""
    claim_id: str
    claim: str
    source_url: str
    quote: str
    label: Literal["supported", "unsupported"]
    """Two labels, matching the two the verifier can mean. "Partially supported"
    is a real failure mode and a documented known unknown (07) — a gold row
    carrying one could never be matched by anything."""

    note: str = Field(description="Why the label is what it is.")
    """The hand-labelling itself. An unauditable ground truth is an opinion."""

    corpus: str | None = None
    """A document in `corpus/` that travels with this row, so the shipped set
    runs offline. Omit it and the source is fetched from `source_url` instead —
    which is what a real gold set does."""

    hard: bool = False
    """Marks the cases the verifier's judgment actually turns on, so a set can be
    checked for having any."""


def load_gold(path: str | Path | None = None) -> list[GoldClaim]:
    """Read a JSONL gold set, in file order.

    Order is load-bearing: `pass_at_1` reads the first run and the report lists
    claims as they were labelled, so a set that loaded differently each time
    could not produce the same numbers twice.
    """
    text = Path(path or DEFAULT_GOLD_PATH).read_text(encoding="utf-8")
    return [GoldClaim(**json.loads(line)) for line in text.splitlines() if line.strip()]
