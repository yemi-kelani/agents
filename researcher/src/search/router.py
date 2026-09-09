"""Ordered fallback across backends — spec 04.

A backend that raises and a backend that returns nothing are the same failure
from the caller's side, and both have to fall through: SearXNG's limiter answers
200 with an empty result set, and a rate-limited provider raises. Only the
router decides what either means, which is why the adapters do not swallow
errors themselves.
"""
from __future__ import annotations

import logging

from search.base import SearchBackend, SearchHit

logger = logging.getLogger(__name__)


class SearchRouter:
    """Ordered fallback. First backend that returns results wins."""

    def __init__(self, backends: list[SearchBackend], *, k: int = 5):
        self.backends = backends
        self.k = k

    async def search(self, query: str, k: int | None = None) -> list[SearchHit]:
        k = self.k if k is None else k
        failures = []

        for b in self.backends:
            try:
                hits = await b.search(query, k)
                if hits:
                    return hits
                failures.append(f"{b.name}: empty")
            except Exception as e:                    # noqa: BLE001 — deliberate
                failures.append(f"{b.name}: {type(e).__name__}")
                logger.warning("search backend %s failed: %s", b.name, e)

        # Not an error to raise on: "nobody found anything" is a research
        # outcome the sufficiency check (09) reads as a `no_results`
        # uncertainty, and one dead query must not end the run.
        logger.error("all backends failed for %r: %s", query, "; ".join(failures))
        return []
