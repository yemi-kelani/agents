"""The search interface every backend implements — spec 04.

One `SearchHit` shape over two families of provider:

- **LLM-optimized** (Tavily, Exa) return clean page text with the result.
- **SERP-style** (SearXNG, Brave) return links and snippets only.

`content` is where that difference lives, and normalizing it is the whole job of
this layer: downstream code writes `hit.content or await fetch(hit.url)` and
never branches on which provider answered. The same shape is what will let an
internal OpenSearch index (P2) drop in without touching extraction, verification
or the report.
"""
from __future__ import annotations

from typing import Protocol, runtime_checkable

from pydantic import BaseModel


class SearchHit(BaseModel):
    url: str
    title: str
    snippet: str
    content: str | None = None     # some backends return clean page text; most don't
    published: str | None = None
    backend: str


@runtime_checkable
class SearchBackend(Protocol):
    name: str

    async def search(self, query: str, k: int = 5) -> list[SearchHit]: ...
