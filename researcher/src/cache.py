"""Disk cache for the two calls that leave the process — spec 10.

Eval runs (11) repeat the same queries dozens of times. Uncached, most of the
wall-clock is network wait and most of the quota goes on identical requests, so
the local-vs-frontier comparison the model layer exists to enable is mostly a
measurement of somebody's DNS.

Two rules that are not obvious:

**Nothing falsy is cached.** A dead backend answers `[]` and a failed extraction
answers `""`, and those are indistinguishable here from a real empty answer.
Storing one for a day turns a transient outage into a day of empty runs.

**Entries expire.** A cached page is a snapshot, and this pipeline's whole claim
is that a quote is still in the document it cited. A cache with no TTL quietly
turns that into "was, once".
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

from pydantic_core import to_jsonable_python

from search.base import SearchBackend, SearchHit
from settings import settings

DEFAULT_TTL_HOURS = 24.0


def default_dir() -> Path:
    """`RESEARCHER_CACHE_DIR`, or a well-behaved place under the home directory."""
    return settings().cache_dir or Path.home() / ".cache" / "researcher"


def _identity(value: Any) -> Any:
    return value


class Cache:
    """A directory of JSON entries, keyed by the call that produced them."""

    def __init__(self, dir: str | Path, ttl_hours: float = DEFAULT_TTL_HOURS):
        # Not created here: a cache nothing ever writes to should leave nothing
        # behind, and wiring one up is not the same as using it.
        self.dir = Path(dir)
        self.ttl_hours = ttl_hours

    @classmethod
    def from_config(cls, cfg) -> Cache | None:
        """The `cache:` section, or None when it is switched off. None rather
        than a no-op cache so the call site says which it got."""
        if not cfg.enabled:
            return None
        return cls(cfg.dir or default_dir(), ttl_hours=cfg.ttl_hours)

    def wrap(self, fn: Callable, *, revive: Callable[[Any], Any] = _identity,
             key: str | None = None) -> Callable:
        """The async `fn`, answered from disk when it has been asked before.

        `revive` rebuilds whatever JSON cannot represent — the models a caller
        was written against. Without it a cached `SearchHit` comes back as a
        dict and the second run takes a different code path than the first,
        which is the worst way for a cache to be wrong.
        """
        name = key or fn.__qualname__

        async def cached(*args, **kwargs):
            path = self._path(name, args, kwargs)
            # Off the event loop. `read_text`, `mkdir` and `write_text` are
            # synchronous syscalls, and under an ASGI server they do not merely
            # stall the loop — the blocking-call detector raises. That lands
            # here as an exception from a *backend*, which the router (04)
            # catches and reports as a failed provider, so a disk write in this
            # file reads in the logs as every search engine being down at once.
            stored = await asyncio.to_thread(self._read, path)
            if stored is not None:
                return revive(stored)

            value = await fn(*args, **kwargs)
            if value:
                await asyncio.to_thread(self._write, path, value)
            return value

        return cached

    def backend(self, backend: SearchBackend) -> SearchBackend:
        """`backend`, answering repeated queries from disk.

        Wrapped rather than built in, so spec 04's adapters and the router stay
        unaware: the fallback chain still sees a backend, and a backend that
        raised is still a backend that raised.
        """
        return CachedBackend(backend, self)

    def _path(self, name: str, args: tuple, kwargs: dict) -> Path:
        digest = hashlib.sha256(
            repr((name, args, sorted(kwargs.items()))).encode()
        ).hexdigest()[:32]
        return self.dir / f"{digest}.json"

    def _read(self, path: Path) -> Any | None:
        """The stored value, or None for a miss.

        Unambiguous because nothing falsy is ever written: a stored entry always
        holds something. A corrupt or half-written file reads as a miss —
        re-fetching costs a request, and raising costs the run.
        """
        try:
            entry = json.loads(path.read_text(encoding="utf-8"))
            if time.time() - entry["at"] > self.ttl_hours * 3600:
                return None
            return entry["value"]
        except (OSError, ValueError, KeyError, TypeError):
            return None

    def _write(self, path: Path, value: Any) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps({"at": time.time(), "value": to_jsonable_python(value)}),
            encoding="utf-8",
        )


class CachedBackend:
    """One backend's results, keyed by its own name.

    The name is in the key because the fallback chain asks every backend the
    same query: a shared entry would answer for a backend that was never called,
    and hide which provider the evidence actually came from.
    """

    def __init__(self, backend: SearchBackend, cache: Cache):
        self.name = backend.name
        self._search = cache.wrap(
            backend.search,
            key=f"search:{backend.name}",
            revive=lambda rows: [SearchHit(**row) for row in rows],
        )

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return await self._search(query, k)
