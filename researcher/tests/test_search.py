"""Spec 04 acceptance: one interface, several backends, ordered fallback.

The adapters are driven through real `httpx` plumbing (`MockTransport`) rather
than a mocked client, because the thing worth testing is the mapping from each
provider's payload shape into one `SearchHit` — a mocked client would only prove
the mock returns what it was told to.

`asyncio.run` rather than pytest-asyncio: nothing here needs an event-loop
fixture, and the suite stays runnable without the plugin installed.
"""
from __future__ import annotations

import asyncio
import json
import logging

import httpx
import pytest

from search import BACKENDS, SearchConfig, build_router
from search.authority import authority
from search.base import SearchBackend, SearchHit
from search.router import SearchRouter
from search.searxng import SearxngBackend
from search.tavily import TavilyBackend

TAVILY_PAYLOAD = {
    "results": [
        {
            "url": "https://arxiv.org/abs/2304.09848",
            "title": "Evaluating Verifiability",
            "content": "a snippet",
            "raw_content": "the full clean page text",
        }
    ]
}

SEARXNG_PAYLOAD = {
    "results": [
        {"url": f"https://example.com/{i}", "title": f"page {i}", "content": f"snippet {i}"}
        for i in range(3)
    ]
}


def responder(payload: dict, status: int = 200):
    """A transport handler returning one canned JSON body, recording requests."""
    seen: list[httpx.Request] = []

    def handle(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(status, json=payload)

    handle.seen = seen
    return handle


def search_with(handler, make_backend, query: str = "q", k: int = 5) -> list[SearchHit]:
    async def go():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            return await make_backend(client).search(query, k)

    return asyncio.run(go())


def tavily(client) -> TavilyBackend:
    return TavilyBackend(api_key="key", client=client)


def searxng(client) -> SearxngBackend:
    return SearxngBackend(client=client)


# --- acceptance: every backend emits the same SearchHit shape ---------------

def test_an_llm_optimized_backend_returns_page_content_with_the_hit():
    """Tavily's family answers `hit.content or fetch(url)` without the fetch."""
    (hit,) = search_with(responder(TAVILY_PAYLOAD), tavily)

    assert hit.url == "https://arxiv.org/abs/2304.09848"
    assert hit.title == "Evaluating Verifiability"
    assert hit.snippet == "a snippet"
    assert hit.content == "the full clean page text"
    assert hit.backend == "tavily"


def test_a_serp_backend_returns_the_same_shape_with_no_content():
    """SearXNG's family leaves `content` unset, so spec 05 fetches the page.
    Downstream code branches on the field, never on the provider."""
    hits = search_with(responder(SEARXNG_PAYLOAD), searxng)

    assert {type(h) for h in hits} == {SearchHit}
    assert hits[0].url == "https://example.com/0"
    assert hits[0].snippet == "snippet 0"
    assert hits[0].content is None
    assert hits[0].backend == "searxng"


@pytest.mark.parametrize("backend", [
    TavilyBackend(api_key="key"),
    SearxngBackend(),
])
def test_backends_satisfy_the_search_protocol(backend):
    assert isinstance(backend, SearchBackend)


def test_a_backend_asks_the_provider_for_k_results():
    handler = responder(TAVILY_PAYLOAD)
    search_with(handler, tavily, k=3)

    assert json.loads(handler.seen[0].content)["max_results"] == 3


def test_a_backend_that_ignores_k_is_truncated_locally():
    """SearXNG has no result-count parameter, so the cap is applied here —
    otherwise `k` means something different depending on the backend."""
    hits = search_with(responder(SEARXNG_PAYLOAD), searxng, k=2)

    assert len(hits) == 2


def test_an_http_error_from_a_provider_surfaces_as_an_exception():
    """The router decides what a failed backend means; the adapter does not
    swallow it and return zero results."""
    with pytest.raises(httpx.HTTPStatusError):
        search_with(responder({}, status=500), tavily)


# --- acceptance: fallback keeps the run alive ------------------------------

class FakeBackend:
    """A backend that hits, returns nothing, or fails — the three cases the
    router exists to tell apart."""

    def __init__(self, name: str, *, hits: int = 0, error: Exception | None = None):
        self.name = name
        self._hits = hits
        self._error = error
        self.calls: list[tuple[str, int]] = []

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.calls.append((query, k))
        if self._error:
            raise self._error
        return [
            SearchHit(url=f"http://{self.name}/{i}", title="t", snippet="s",
                      backend=self.name)
            for i in range(self._hits)
        ]


def test_the_primary_backend_wins_when_it_has_results():
    primary = FakeBackend("primary", hits=2)
    secondary = FakeBackend("secondary", hits=2)

    hits = asyncio.run(SearchRouter([primary, secondary]).search("q"))

    assert {h.backend for h in hits} == {"primary"}
    assert secondary.calls == [], "a working primary must not cost a second call"


def test_a_raising_primary_falls_through_instead_of_killing_the_run():
    primary = FakeBackend("primary", error=httpx.ConnectError("refused"))
    secondary = FakeBackend("secondary", hits=1)

    hits = asyncio.run(SearchRouter([primary, secondary]).search("q"))

    assert [h.backend for h in hits] == ["secondary"]


def test_an_empty_primary_falls_through_too():
    """A backend that returns nothing has failed at its job as surely as one
    that raised — SearXNG's limiter answers 200 with an empty result set."""
    primary = FakeBackend("primary", hits=0)
    secondary = FakeBackend("secondary", hits=1)

    hits = asyncio.run(SearchRouter([primary, secondary]).search("q"))

    assert [h.backend for h in hits] == ["secondary"]


def test_every_backend_failing_yields_no_results_rather_than_an_exception():
    """Zero results is a research outcome (spec 09 logs it as `no_results`),
    not a crash."""
    router = SearchRouter([
        FakeBackend("a", error=RuntimeError("boom")),
        FakeBackend("b", hits=0),
    ])

    assert asyncio.run(router.search("q")) == []


def test_the_configured_k_is_the_default_for_every_search():
    backend = FakeBackend("only", hits=1)

    asyncio.run(SearchRouter([backend], k=3).search("q"))

    assert backend.calls == [("q", 3)]


def test_a_caller_can_override_k_per_search():
    backend = FakeBackend("only", hits=1)

    asyncio.run(SearchRouter([backend], k=3).search("q", k=1))

    assert backend.calls == [("q", 1)]


# --- acceptance: swapping backends is a config change ----------------------

def test_config_selects_the_backends_and_their_fallback_order(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "key")

    router = build_router({"backends": ["searxng", "tavily"], "k": 3})

    assert [b.name for b in router.backends] == ["searxng", "tavily"]
    assert router.k == 3


def test_every_registered_backend_can_be_built_from_its_name_alone(monkeypatch):
    """The registry is what makes "swap the backend in config" true; a name in
    it that needs an undocumented argument breaks that promise."""
    monkeypatch.setenv("TAVILY_API_KEY", "key")

    router = build_router({"backends": list(BACKENDS)})

    assert [b.name for b in router.backends] == list(BACKENDS)


def test_per_backend_options_from_config_reach_the_adapter():
    """A self-hosted SearXNG is never at the default address in a deployment
    that matters, so its location has to be config, not code."""
    router = build_router({
        "backends": ["searxng"],
        "options": {"searxng": {"base_url": "http://searx.internal:8888/"}},
    })

    assert router.backends[0].base_url == "http://searx.internal:8888"


def test_a_backend_queries_the_host_it_was_configured_with():
    handler = responder(SEARXNG_PAYLOAD)
    search_with(handler, lambda c: SearxngBackend("http://searx.internal:8888", client=c))

    assert handler.seen[0].url.host == "searx.internal"
    assert handler.seen[0].url.path == "/search"


def test_an_unknown_backend_name_names_itself_and_the_known_ones():
    with pytest.raises(ValueError) as err:
        build_router({"backends": ["gogle"]})

    assert "gogle" in str(err.value)
    assert "tavily" in str(err.value)


def test_a_backend_missing_its_api_key_fails_at_build_time(monkeypatch):
    """Fail while wiring the run, not three topics into it. A missing key is a
    misconfiguration, and silently dropping the backend hides it."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    with pytest.raises(ValueError) as err:
        build_router({"backends": ["tavily"]})

    assert "TAVILY_API_KEY" in str(err.value)


def test_the_default_config_is_usable_without_any_search_section(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "key")

    assert SearchConfig().k == 5
    assert build_router({}).backends


# --- the authority heuristic ------------------------------------------------

@pytest.mark.parametrize("url,expected", [
    ("https://arxiv.org/abs/2304.09848", 0.85),
    ("https://www.nih.gov/news", 0.9),
    ("https://cs.stanford.edu/paper", 0.85),
    ("https://github.com/org/repo", 0.7),
    ("https://medium.com/@someone/post", 0.35),
    ("https://reddit.com/r/x", 0.25),
])
def test_authority_ranks_primary_sources_above_aggregators(url, expected):
    assert authority(url) == expected


def test_an_unknown_host_gets_the_neutral_default():
    assert authority("https://some-blog.example/post") == 0.5


def test_subdomains_inherit_their_domain_score():
    assert authority("https://export.arxiv.org/abs/1") == 0.85


def test_a_lookalike_host_cannot_borrow_authority():
    """Authority is what flags source laundering, so a domain anyone can
    register must not inherit the score of the domain it impersonates."""
    assert authority("https://arxiv.org.evil.example/abs/1") == 0.5
    assert authority("https://not-nih.gov.attacker.example/") == 0.5


# --- the tool log ----------------------------------------------------------
#
# Which backend actually answered is the fact the trace could never carry: the
# router hands back hits, and a caller counting them cannot tell a primary that
# worked from a fallback that rescued it.

def searched_with_log(router, caplog, query: str = "q"):
    caplog.set_level(logging.DEBUG, logger="researcher.tools")
    hits = asyncio.run(router.search(query))
    return hits, "\n".join(r.getMessage() for r in caplog.records)


def test_a_search_logs_the_backend_that_answered(caplog):
    router = SearchRouter([FakeBackend("primary", hits=2)])

    _, logged = searched_with_log(router, caplog)

    assert "primary" in logged


def test_a_search_logs_the_urls_it_got_back(caplog):
    router = SearchRouter([FakeBackend("primary", hits=1)])

    _, logged = searched_with_log(router, caplog)

    assert "http://primary/0" in logged


def test_a_search_logs_the_query_it_ran(caplog):
    router = SearchRouter([FakeBackend("primary", hits=1)])

    _, logged = searched_with_log(router, caplog, query="pelicans")

    assert "pelicans" in logged


def test_the_log_names_the_backend_that_rescued_a_failed_primary(caplog):
    """The whole point: 'search returned 1 result' reads identically whether
    the primary worked or died, and only one of those needs looking at."""
    router = SearchRouter([
        FakeBackend("primary", error=httpx.ConnectError("refused")),
        FakeBackend("secondary", hits=1),
    ])

    _, logged = searched_with_log(router, caplog)

    assert "secondary" in logged


def test_a_search_that_found_nothing_anywhere_says_so(caplog):
    router = SearchRouter([FakeBackend("only", hits=0)])

    _, logged = searched_with_log(router, caplog)

    assert "0 hits" in logged
    assert "only: empty" in logged, "the log has to say why, not just that"
