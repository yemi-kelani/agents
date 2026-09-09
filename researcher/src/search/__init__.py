"""The one way to build a search router from configuration.

Backends are named in config and constructed here, so switching provider — or
adding the internal index (P2) — never touches a caller. Secrets are the
exception: an API key never appears in the config file. It comes from the
environment through `settings`, which is also where a self-hosted SearXNG's
address defaults from.
"""
from __future__ import annotations

from typing import Callable

from pydantic import BaseModel, Field

from search.authority import authority
from settings import settings
from search.base import SearchBackend, SearchHit
from search.router import SearchRouter
from search.searxng import SearxngBackend
from search.tavily import TavilyBackend

__all__ = [
    "BACKENDS", "SearchBackend", "SearchConfig", "SearchHit", "SearchRouter",
    "authority", "build_router",
]


def _tavily(**opts) -> TavilyBackend:
    key = settings().tavily_api_key
    if not key:
        raise ValueError("search backend 'tavily' needs TAVILY_API_KEY in the environment")
    return TavilyBackend(api_key=key, **opts)


def _searxng(**opts) -> SearxngBackend:
    """`SEARXNG_BASE_URL` is the default, and `options.searxng.base_url` in the
    config file overrides it — an instance's location is deployment, but a run
    profile may want to pin a specific one."""
    return SearxngBackend(**{"base_url": settings().searxng_base_url, **opts})


BACKENDS: dict[str, Callable[..., SearchBackend]] = {
    "tavily": _tavily,
    "searxng": _searxng,
}
"""Name -> constructor. Every entry must build from its name alone (plus
config options), or "swap the backend in config" stops being true."""


class SearchConfig(BaseModel):
    backends: list[str] = Field(default_factory=lambda: ["tavily"])  # ordered; first is primary
    k: int = 5
    options: dict[str, dict] = Field(default_factory=dict)           # per-backend kwargs


def build_router(cfg: dict | SearchConfig | None = None, *, cache=None) -> SearchRouter:
    """Build the fallback chain named by `cfg` (the `search:` section).

    Construction failures — an unknown name, a missing key — are raised here,
    while the run is being wired, rather than surfacing as a dead backend three
    topics in. A misconfigured backend that silently drops out looks exactly
    like a provider outage, and the fallback chain would hide it.

    `cache` (spec 10) wraps each backend rather than the router: the fallback
    order is a decision about live backends, and caching the chain's answer
    would cache "the primary was down" along with the results.
    """
    cfg = cfg if isinstance(cfg, SearchConfig) else SearchConfig(**(cfg or {}))

    backends = []
    for name in cfg.backends:
        if name not in BACKENDS:
            raise ValueError(
                f"unknown search backend {name!r}; known: {', '.join(BACKENDS)}"
            )
        backend = BACKENDS[name](**cfg.options.get(name, {}))
        backends.append(cache.backend(backend) if cache else backend)

    return SearchRouter(backends, k=cfg.k)
