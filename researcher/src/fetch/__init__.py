"""Fetching a page safely — spec 05.

`safe_fetch` is the only way page content enters this system, and it is the
enforcement point for the SSRF guard: resolve, check every address, connect to
the address that was checked, and re-check on every redirect hop.
"""
from __future__ import annotations

import httpx

from fetch.containment import MARK, UNTRUSTED_PREAMBLE, datamark, undatamark
from fetch.extract import to_clean_text
from fetch.guard import BLOCKED_NETS, BlockedURL, resolve_and_check
from http_client import DEFAULT_TIMEOUT, session

__all__ = [
    "BLOCKED_NETS", "BlockedURL", "MARK", "UNTRUSTED_PREAMBLE", "datamark",
    "fetch_clean_text", "resolve_and_check", "safe_fetch", "to_clean_text",
    "undatamark",
]

MAX_REDIRECTS = 3


async def safe_fetch(
    url: str,
    *,
    max_bytes: int = 2_000_000,
    timeout: float = DEFAULT_TIMEOUT,
    client: httpx.AsyncClient | None = None,
) -> str:
    """Fetch `url`, or raise `BlockedURL` before anything leaves the process."""
    async with session(client, timeout) as c:
        for _ in range(MAX_REDIRECTS):
            ip, host = resolve_and_check(url)
            target = httpx.URL(url)

            # Connect to the address that was just checked, not to the hostname:
            # resolving a second time inside the client is the rebinding window
            # the check exists to close. The Host header keeps virtual hosting
            # working, and `sni_hostname` keeps TLS validating against the name
            # rather than the pinned address.
            #
            # `follow_redirects=False` is set per request, never left to the
            # client: a caller whose client follows redirects would otherwise
            # skip every hop check after the first.
            authority = target.netloc.decode("ascii").rsplit("@", 1)[-1]  # no userinfo
            r = await c.get(
                target.copy_with(host=ip),
                headers={"Host": authority},
                extensions={"sni_hostname": host},
                follow_redirects=False,
                timeout=timeout,
            )

            if r.has_redirect_location:
                # Resolved against the original URL, not the pinned one: a
                # relative Location joined onto the IP would drop the hostname
                # the next hop needs for its Host header and TLS validation.
                url = str(target.join(r.headers["location"]))
                continue

            r.raise_for_status()
            # Caps characters after the body is already in memory — enough for
            # the extractor, not a defense against a hostile Content-Length.
            return r.text[:max_bytes]

    raise BlockedURL("too many redirects")


async def fetch_clean_text(url: str, **kwargs) -> str:
    """URL -> article text. The guarded fetch and the HTML cleaner as one step,
    so extraction (06) depends on "get me the text of this page" rather than on
    both halves of how that happens."""
    return to_clean_text(await safe_fetch(url, **kwargs))
