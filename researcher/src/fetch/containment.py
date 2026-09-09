"""Injection containment — spec 05.

**Injection is not solved, and this does not solve it.** No published defense
survives adaptive attack (Nasr & Carlini et al., 2025, arXiv:2503.00061), so the
stance is containment rather than prevention: the fetcher holds no secrets, and
the privileged nodes — planning, sufficiency, synthesis — never see page text at
all, only extracted claims and quotes. That separation is the structural half of
the defense and matters more than the marking below.

The marking is Microsoft's spotlighting (Hines et al., 2024, arXiv:2403.14720):
making the instruction/data boundary legible by marking every untrusted token
dropped attack success from ~50% to under 3% on GPT-3.5-Turbo at negligible task
cost. Cheap, worth doing, not a guarantee.
"""
from __future__ import annotations

from prompts import load

MARK = "░"    # rare in natural text


def datamark(text: str) -> str:
    """Mark every token of untrusted content as data.

    Note for spec 06: quotes the model copies out of a marked block come back
    marked, so anchoring against the stored (unmarked) text has to reverse this
    first — see `undatamark`.
    """
    return text.replace(" ", MARK)


def undatamark(text: str) -> str:
    """Reverse `datamark` on text that came back out of the model.

    A quote the model copied faithfully out of a marked block is marked. Sending
    that to the anchor as-is reports a fabricated quote for content the model
    got exactly right, which is the worst possible false positive here: it is
    indistinguishable from the failure the anchor exists to catch.
    """
    return text.replace(MARK, " ")


UNTRUSTED_PREAMBLE = load("untrusted_preamble").format(mark=MARK)
