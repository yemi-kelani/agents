"""Source authority — hand-written on purpose.

An LLM scoring source quality is one more thing to be wrong about, and it costs
a call per hit. The score ranks results and, more usefully, flags **source
laundering**: a blog post citing a paper is a weaker citation than the paper.
"""
from __future__ import annotations

from urllib.parse import urlparse

AUTHORITY = {
    ".gov": 0.9, ".edu": 0.85, "arxiv.org": 0.85,
    "github.com": 0.7, "medium.com": 0.35, "reddit.com": 0.25,
}

NEUTRAL = 0.5


def authority(url: str) -> float:
    """Score `url`'s host, defaulting to neutral.

    Matching is on domain-label boundaries, never a bare substring: the score is
    what marks a laundered source, so `arxiv.org.attacker.example` — a domain
    anyone can register — must not inherit arxiv's standing.
    """
    host = (urlparse(url).hostname or "").lower().rstrip(".")
    for pattern, score in AUTHORITY.items():
        suffix = pattern.lstrip(".")
        if host == suffix or host.endswith(f".{suffix}"):
            return score
    return NEUTRAL
