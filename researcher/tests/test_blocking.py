"""Nothing on the graph's async path blocks the event loop.

`langgraph dev` runs the graph inside an ASGI server with blockbuster active,
and a synchronous file read there raises `BlockingError` rather than merely
being slow. That matters more than it sounds: the search router catches every
exception a backend raises and falls through to the next one, so a disk write in
the *cache* surfaced as "all backends failed" — a bug in this repo's own I/O
wearing the costume of every search provider being down at once.

The rule this file enforces: an `async def` on the graph's path touches the disk
through a thread, or not at all. It is scoped to this repo's modules, because
what a dependency does inside its own event loop is not something these tests
can fix.
"""
from __future__ import annotations

import asyncio

from blockbuster import blockbuster_ctx

from cache import Cache
from prompts import PROMPT_NAMES, load
from search.base import SearchHit

SCANNED = ["cache", "prompts", "nodes", "search", "fetch", "anchor", "render"]


def without_blocking(factory):
    """Run a coroutine with blockbuster watching this repo's modules.

    A `BlockingError` propagates: the point is to fail the test with the same
    exception the server would have raised, at the line that caused it.
    """
    async def go():
        with blockbuster_ctx(scanned_modules=SCANNED):
            return await factory()

    return asyncio.run(go())


async def a_search(query: str) -> list[SearchHit]:
    return [SearchHit(url="https://example.com", title="t", snippet="s", backend="b")]


class Backend:
    name = "fake"

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return await a_search(query)


# --- the cache --------------------------------------------------------------

def test_a_cache_miss_does_not_block_the_event_loop(tmp_path):
    """The write path, and the one that broke a real run: `mkdir` and
    `write_text` are synchronous filesystem calls inside an async wrapper."""
    cached = Cache(tmp_path).wrap(a_search)

    assert without_blocking(lambda: cached("mimir"))


def test_a_cache_hit_does_not_block_the_event_loop(tmp_path):
    """The read path. Cheaper to miss than to hit would be a strange cache."""
    cached = Cache(tmp_path).wrap(a_search)
    asyncio.run(cached("mimir"))

    assert without_blocking(lambda: cached("mimir"))


def test_a_cached_backend_does_not_block_the_event_loop(tmp_path):
    """The exact path that failed under `langgraph dev`: a search that succeeded
    and then tripped over its own cache on the way back."""
    backend = Cache(tmp_path).backend(Backend())

    assert without_blocking(lambda: backend.search("mimir", 3))


def test_a_corrupt_entry_does_not_block_the_event_loop_either(tmp_path):
    """The miss-by-way-of-a-bad-file path reads before it decides to refetch."""
    cache = Cache(tmp_path)
    cached = cache.wrap(a_search)
    asyncio.run(cached("mimir"))
    for entry in tmp_path.glob("*.json"):
        entry.write_text("{ not json", encoding="utf-8")

    assert without_blocking(lambda: cached("mimir"))


# --- prompts ----------------------------------------------------------------

def test_loading_a_prompt_does_not_block_the_event_loop():
    """`load` is called from inside async nodes — extraction (06) and the
    verifier (07) both reach for a prompt mid-flight."""
    async def load_them():
        return [load(name) for name in PROMPT_NAMES]

    assert without_blocking(load_them)


def test_every_prompt_is_read_before_anything_awaits_it():
    """How the above is true: the files are read at import, so `load` is a
    dictionary lookup by the time a coroutine wants one. A prompt added to the
    directory is picked up without anyone registering it."""
    assert {"extract", "verify", "plan", "synthesize", "sufficiency"} <= set(PROMPT_NAMES)
    assert all(load(name) for name in PROMPT_NAMES)
