"""The one way a component gets an HTTP client.

Search backends (04) and the fetcher (05) both need "use the client I was given,
or make a short-lived one", and the caller-supplied case is what lets a run share
a connection pool — and what lets these paths be exercised against
`httpx.MockTransport`.
"""
from __future__ import annotations

from contextlib import asynccontextmanager

import httpx

DEFAULT_TIMEOUT = 15.0


@asynccontextmanager
async def session(client: httpx.AsyncClient | None, timeout: float = DEFAULT_TIMEOUT):
    """Yield `client` if given, else a client owned by this block.

    An injected client's lifecycle belongs to whoever passed it, so it is never
    closed here. Per-request options are the caller's job: anything security
    relevant — `follow_redirects` above all — must be set on the request rather
    than assumed from the client (see `fetch.safe_fetch`).
    """
    if client is not None:
        yield client
    else:
        async with httpx.AsyncClient(timeout=timeout) as owned:
            yield owned
