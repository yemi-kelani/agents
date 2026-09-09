"""Tavily — an LLM-optimized backend: one call returns the page text too."""
from __future__ import annotations

import httpx

from http_client import DEFAULT_TIMEOUT, session
from search.base import SearchHit

ENDPOINT = "https://api.tavily.com/search"


class TavilyBackend:
    name = "tavily"

    def __init__(
        self,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
        client: httpx.AsyncClient | None = None,
    ):
        self._key = api_key
        self._timeout = timeout
        self._client = client

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        async with session(self._client, self._timeout) as c:
            r = await c.post(ENDPOINT, json={
                "api_key": self._key,
                "query": query,
                "max_results": k,
                "include_raw_content": True,
            }, timeout=self._timeout)
            r.raise_for_status()
            return [
                SearchHit(
                    url=x["url"],
                    title=x.get("title", ""),
                    snippet=x.get("content", ""),
                    content=x.get("raw_content"),
                    backend=self.name,
                )
                for x in r.json().get("results", [])
            ]
