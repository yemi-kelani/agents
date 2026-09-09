# 04 — Pluggable Search Backends

**Priority: P0.** Build immediately after state.

## Purpose

One interface, several backends, config-driven selection, ordered fallback. The
point is not "supports many providers" — it's that the *same interface* will later
point at an internal document index without the agent knowing.

## The interface

```python
# search/base.py
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
```

The single most important field is `content`. Backends split into two families:

- **LLM-optimized** (Tavily, Exa, Firecrawl) return clean extracted text with the
  result. One call, no separate fetch, no HTML parsing.
- **SERP-style** (Serper, Brave, DuckDuckGo) return links and snippets. You fetch
  and extract yourself (spec 05).

Normalizing over that difference is the whole job of this layer. Downstream code
does `hit.content or await fetch(hit.url)` and never branches on provider.

## Adapters

```python
# search/tavily.py
import httpx

class TavilyBackend:
    name = "tavily"

    def __init__(self, api_key: str, *, timeout: float = 15.0):
        self._key, self._timeout = api_key, timeout

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=self._timeout) as c:
            r = await c.post("https://api.tavily.com/search", json={
                "api_key": self._key,
                "query": query,
                "max_results": k,
                "include_raw_content": True,
            })
            r.raise_for_status()
            return [
                SearchHit(
                    url=x["url"], title=x.get("title", ""),
                    snippet=x.get("content", ""),
                    content=x.get("raw_content"),
                    backend=self.name,
                )
                for x in r.json().get("results", [])
            ]
```

```python
# search/searxng.py  — self-hosted, free, private
class SearxngBackend:
    name = "searxng"

    def __init__(self, base_url: str = "http://localhost:8080"):
        self._base = base_url.rstrip("/")

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        async with httpx.AsyncClient(timeout=15.0) as c:
            r = await c.get(f"{self._base}/search",
                            params={"q": query, "format": "json"})
            r.raise_for_status()
            return [
                SearchHit(url=x["url"], title=x.get("title", ""),
                          snippet=x.get("content", ""), backend=self.name)
                for x in r.json().get("results", [])[:k]
            ]
```

**SearXNG caveat worth knowing before you demo on it.** Its bot-detection limiter
is designed to reject non-browser clients — with the limiter on, a plain HTTP
client gets 429 on every request. For programmatic use on a private instance set
`limiter: false` and `formats: [html, json]` in `settings.yml`. Even then, SearXNG
is proxying to upstream engines that may block *it*. Fine as a free/private
option; not something to depend on as the only backend.

## Fallback chain

```python
# search/router.py
import logging

class SearchRouter:
    """Ordered fallback. First backend that returns results wins."""

    def __init__(self, backends: list[SearchBackend]):
        self._backends = backends

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        errors = []
        for b in self._backends:
            try:
                hits = await b.search(query, k)
                if hits:
                    return hits
                errors.append(f"{b.name}: empty")
            except Exception as e:                    # noqa: BLE001 — deliberate
                errors.append(f"{b.name}: {type(e).__name__}")
                logging.warning("search backend %s failed: %s", b.name, e)
        logging.error("all backends failed: %s", "; ".join(errors))
        return []
```

Config-driven construction:

```yaml
# config.yaml
search:
  backends: [tavily, duckduckgo]     # ordered; first is primary
  k: 5
```

## Authority heuristic

Transparent and hand-written on purpose — an LLM scoring source quality is one
more thing to be wrong about, and it costs a call per hit.

```python
AUTHORITY = {
    ".gov": 0.9, ".edu": 0.85, "arxiv.org": 0.85,
    "github.com": 0.7, "medium.com": 0.35, "reddit.com": 0.25,
}

def authority(url: str) -> float:
    host = urlparse(url).netloc.lower()
    for pattern, score in AUTHORITY.items():
        if host.endswith(pattern) or pattern in host:
            return score
    return 0.5
```

Used for ranking and, more usefully, for flagging **source laundering** — a blog
post citing a paper is a weaker citation than the paper. Prefer primary sources
when both cover a claim.

## Future: internal index (P2)

The same `SearchBackend` protocol over OpenSearch. Hybrid retrieval — BM25 for
exact identifiers, k-NN dense vectors for semantics, combined by a normalization
processor. The key design point is that an internal hit maps into the *same*
`SearchHit`, so internal-doc citations carry url/title/quote metadata identically
and every downstream component — extraction, verification, the report — works
unchanged. That's the payoff for having built the abstraction on day one.

```python
class OpenSearchBackend:
    name = "opensearch"
    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        # hybrid query: {"hybrid": {"queries": [ {match: ...}, {knn: ...} ]}}
        # map _source -> SearchHit(url=doc_uri, content=body, backend=self.name)
        ...
```

## Acceptance

- Swapping backends is a config change, no code change.
- Primary failure falls through to secondary without killing the run.
- All backends emit identical `SearchHit` shape.
- Zero results from every backend is handled, not crashed on.
