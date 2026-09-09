"""HTML -> clean text — spec 05.

**Be honest about what this removes.** `trafilatura` drops navigation,
boilerplate, image `alt` text and (with `include_comments=False`) HTML comments.
It does **not** remove the text content of a CSS-hidden div — that text is real
content as far as any extractor is concerned. Stripping HTML is necessary, not
sufficient, which is what the containment layer and the mandatory verification
node (07) are for.
"""
from __future__ import annotations

import re

import trafilatura

# Zero-width and bidi-control characters: an instruction spelled out in
# characters that render as nothing, which a model still reads.
ZERO_WIDTH = re.compile(r"[\u200b-\u200f\u2028-\u202e\ufeff]")
BLANK_RUN = re.compile(r"\n{3,}")


def to_clean_text(html: str) -> str:
    """Extract the article text. Returns "" for a page with nothing in it —
    `trafilatura` answers None there, and None reaching the content store blows
    up on `len(text)` downstream."""
    text = trafilatura.extract(html, include_comments=False, include_tables=True) or ""
    text = ZERO_WIDTH.sub("", text)                 # strip unicode steganography
    return BLANK_RUN.sub("\n\n", text).strip()
