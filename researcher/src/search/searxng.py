"""SearXNG — self-hosted, free, private; SERP-style, so spec 05 does the fetch.

Two things to know before depending on it: its bot-detection limiter rejects
non-browser clients outright (a private instance needs `limiter: false` and
`formats: [html, json]` in `settings.yml`), and it proxies to upstream engines
that may be blocking *it*. A fine free option, not a sole backend — which is
what the router's fallback chain is for.
"""
from __future__ import annotations

import httpx

from http_client import DEFAULT_TIMEOUT, session
from search.base import SearchHit
from settings import DEFAULT_SEARXNG_URL


class SearxngBackend:
    name = "searxng"

    def __init__(
        self,
        base_url: str = DEFAULT_SEARXNG_URL,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ):
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout
        self._client = client

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        async with session(self._client, self._timeout) as c:
            r = await c.get(f"{self.base_url}/search",
                            params={"q": query, "format": "json"},
                            timeout=self._timeout)
            r.raise_for_status()
            # SearXNG has no result-count parameter, so `k` is applied here.
            # Without it the same k would mean a different number of hits per
            # backend, and the fetch budget downstream would swing with it.
            return [
                SearchHit(url=x["url"], title=x.get("title", ""),
                          snippet=x.get("content", ""), backend=self.name)
                for x in r.json().get("results", [])[:k]
            ]
