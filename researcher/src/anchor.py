"""Locating a quote in the source it was copied from — spec 06.

A citation you cannot mechanically check is not a citation. This module turns a
quote the model *wrote* into a span into the text that was actually fetched, so
`source_text[start:end] == quote.text` settles the question with no model in the
loop. Verification (spec 07) leans on the same function for its cheap tier.

Three tiers, and **which tier answered is part of the answer**:

1. `exact` — what the extraction prompt asks for.
2. `normalized` — re-flowed whitespace and changed case. Models do this
   constantly and it does not alter what the source said.
3. `fuzzy` — a model that silently "corrected" the source. Accepted, because
   refusing it loses a real citation over a typo, but reported: a corrected
   quote is not a verbatim one, and a high fuzzy rate means the extraction
   prompt is the problem rather than the matcher.

Anchoring nowhere is the *useful* outcome. It is how a fabricated quote is
caught, and it is the single most valuable thing this pipeline detects.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Literal, NamedTuple

DEFAULT_FUZZY_THRESHOLD = 0.92


class Anchor(NamedTuple):
    """Where a quote landed, and how hard the matcher had to work to land it."""
    start: int
    end: int
    method: Literal["exact", "normalized", "fuzzy"]


def anchor(
    quote_text: str,
    source_text: str,
    *,
    fuzzy_threshold: float = DEFAULT_FUZZY_THRESHOLD,
) -> Anchor | None:
    """Locate `quote_text` in `source_text`, or answer None for a quote that is
    not in there at all."""
    normalized_quote, _ = _normalize(quote_text)
    if not normalized_quote:
        # `"".find` answers 0. An empty or whitespace-only quote would otherwise
        # anchor at the top of every document ever fetched.
        return None

    idx = source_text.find(quote_text)
    if idx != -1:
        return Anchor(idx, idx + len(quote_text), "exact")

    normalized_source, origin = _normalize(source_text)

    idx = normalized_source.find(normalized_quote)
    if idx != -1:
        return Anchor(*_span(idx, len(normalized_quote), origin), "normalized")

    best = _best_window(normalized_quote, normalized_source)
    if best and best[1] >= fuzzy_threshold:
        return Anchor(*_span(best[0], len(normalized_quote), origin), "fuzzy")

    return None


def _normalize(text: str) -> tuple[str, list[int]]:
    """Lowercase and collapse whitespace runs, recording where every surviving
    character came from.

    The index is the load-bearing half: a hit in the normalized copy is useless
    unless it maps back to offsets into the text the report will cite. Leading
    and trailing whitespace is dropped the way `strip` would, by never emitting
    a collapsed space with nothing before or after it.
    """
    out: list[str] = []
    origin: list[int] = []
    pending_ws: int | None = None

    for i, ch in enumerate(text):
        if ch.isspace():
            if pending_ws is None:
                pending_ws = i          # the run's first character, so a span
            continue                    # that starts on a space starts at the run
        if pending_ws is not None:
            if out:
                out.append(" ")
                origin.append(pending_ws)
            pending_ws = None
        out.append(ch.lower())
        origin.append(i)

    return "".join(out), origin


def _span(start: int, length: int, origin: list[int]) -> tuple[int, int]:
    """A `[start, start + length)` slice of the normalized copy, as offsets into
    the original. The end is exclusive and taken from the last character that
    actually matched, so a collapsed whitespace run is not swept into the span."""
    last = min(start + length, len(origin)) - 1
    return origin[start], origin[last] + 1


def _best_window(needle: str, haystack: str) -> tuple[int, float] | None:
    """The best same-length window of `haystack`, and how well it matches.

    Scoring every window would be one `SequenceMatcher` run per character of the
    source. The longest shared block says where the window has to be instead:
    align it so that block sits at the offset it occupies in the needle, and
    score once.
    """
    if not haystack:
        return None

    # autojunk=False: the heuristic drops characters appearing in more than 1%
    # of a sequence longer than 200, which for prose is most of the alphabet.
    block = SequenceMatcher(None, haystack, needle, autojunk=False).find_longest_match(
        0, len(haystack), 0, len(needle)
    )
    if not block.size:
        return None

    start = max(0, min(block.a - block.b, len(haystack) - len(needle)))
    window = haystack[start:start + len(needle)]
    return start, SequenceMatcher(None, window, needle, autojunk=False).ratio()
