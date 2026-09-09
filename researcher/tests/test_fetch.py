"""Spec 05 acceptance: SSRF guard, redirect re-checking, injection containment.

Resolution is stubbed rather than reaching real DNS: the guard's job is what it
does with the addresses it gets back, and a test that depends on the network
tests the network. The HTTP side runs through real `httpx` plumbing via
`MockTransport`, which is also what makes "no request was made" assertable.
"""
from __future__ import annotations

import asyncio
import socket
from pathlib import Path

import httpx
import pytest

from fetch import safe_fetch
from fetch.containment import MARK, UNTRUSTED_PREAMBLE, datamark
from fetch.extract import to_clean_text
from fetch.guard import BlockedURL, resolve_and_check

FIXTURES = Path(__file__).parent / "fixtures"

PUBLIC_IP = "93.184.216.34"


@pytest.fixture
def dns(monkeypatch):
    """Point hostname resolution wherever a test needs it."""
    hosts: dict[str, list[str]] = {}

    def fake_getaddrinfo(host, port, *args, **kwargs):
        if host not in hosts:
            raise socket.gaierror(f"unstubbed host in test: {host}")
        return [
            (socket.AF_INET, socket.SOCK_STREAM, 6, "", (ip, port or 0))
            for ip in hosts[host]
        ]

    monkeypatch.setattr(socket, "getaddrinfo", fake_getaddrinfo)
    return hosts


class Net:
    """A mock network: `seen` is every request that actually left the client,
    which is what makes "nothing was requested" assertable even when the call
    raised."""

    def __init__(self, handler=None, *, follow_redirects: bool = False):
        self.seen: list[httpx.Request] = []
        self._handler = handler or (lambda r: httpx.Response(200, text="<p>hello</p>"))
        self._follow = follow_redirects

    def _record(self, request: httpx.Request) -> httpx.Response:
        self.seen.append(request)
        return self._handler(request)

    def fetch(self, url: str, **kwargs) -> str:
        async def go():
            async with httpx.AsyncClient(
                transport=httpx.MockTransport(self._record),
                follow_redirects=self._follow,
            ) as client:
                return await safe_fetch(url, client=client, **kwargs)

        return asyncio.run(go())

    @property
    def hosts(self) -> list[str]:
        return [r.headers["host"] for r in self.seen]


def redirect_to(location: str):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": location})

    return handler


# --- acceptance: the metadata endpoint is blocked before any request --------

def test_the_cloud_metadata_endpoint_is_blocked(dns):
    """The Capital One shape: 169.254.169.254 hands out IAM credentials."""
    dns["169.254.169.254"] = ["169.254.169.254"]

    with pytest.raises(BlockedURL):
        resolve_and_check(
            "http://169.254.169.254/latest/meta-data/iam/security-credentials/"
        )


def test_a_blocked_url_never_reaches_the_network(dns):
    dns["169.254.169.254"] = ["169.254.169.254"]
    net = Net()

    with pytest.raises(BlockedURL):
        net.fetch("http://169.254.169.254/latest/meta-data/")

    # A check that ran after the request would raise too — and the credentials
    # would already have left the host.
    assert net.seen == []


@pytest.mark.parametrize("ip", [
    "127.0.0.1",          # loopback
    "10.1.2.3",           # RFC1918
    "172.16.5.4",
    "192.168.1.1",
    "169.254.169.254",    # link-local, including cloud metadata
    "::1",                # IPv6 loopback
    "fc00::1",            # IPv6 ULA
    "fe80::1",            # IPv6 link-local
])
def test_every_private_range_is_refused(dns, ip):
    dns["internal.example"] = [ip]

    with pytest.raises(BlockedURL):
        resolve_and_check("http://internal.example/")


def test_a_public_host_resolves_to_a_pinnable_address(dns):
    dns["example.com"] = [PUBLIC_IP]

    assert resolve_and_check("https://example.com/page") == (PUBLIC_IP, "example.com")


def test_every_resolved_address_is_checked_not_only_the_first(dns):
    """A host that answers with one public and one private record is the cheap
    version of rebinding: checking `ips[0]` passes and the connection can still
    land on the private one."""
    dns["split.example"] = [PUBLIC_IP, "127.0.0.1"]

    with pytest.raises(BlockedURL):
        resolve_and_check("http://split.example/")


@pytest.mark.parametrize("url", [
    "file:///etc/passwd",
    "gopher://example.com/",
    "data:text/html,<p>x</p>",
])
def test_only_http_and_https_are_fetchable(url):
    with pytest.raises(BlockedURL):
        resolve_and_check(url)


def test_a_url_with_no_host_is_refused():
    with pytest.raises(BlockedURL):
        resolve_and_check("http:///nowhere")


# --- acceptance: redirects are re-checked at every hop ---------------------

def test_a_redirect_into_a_private_range_is_blocked_at_the_hop(dns):
    """A one-shot check on the first URL is defeated by a public host that
    redirects to 169.254.169.254."""
    dns["public.example"] = [PUBLIC_IP]
    dns["internal.example"] = ["169.254.169.254"]
    net = Net(redirect_to("http://internal.example/meta"))

    with pytest.raises(BlockedURL):
        net.fetch("http://public.example/")

    assert net.hosts == ["public.example"], "the private hop must never be requested"


def test_a_client_that_follows_redirects_cannot_skip_the_per_hop_check(dns):
    """The guard cannot depend on how the caller configured its client: httpx
    following the hop itself would bypass every check after the first."""
    dns["public.example"] = [PUBLIC_IP]
    dns["internal.example"] = ["127.0.0.1"]
    net = Net(redirect_to("http://internal.example/"), follow_redirects=True)

    with pytest.raises(BlockedURL):
        net.fetch("http://public.example/")

    assert net.hosts == ["public.example"]


def test_a_redirect_to_a_public_host_is_followed(dns):
    dns["public.example"] = [PUBLIC_IP]
    dns["elsewhere.example"] = ["93.184.216.35"]

    def handler(request: httpx.Request) -> httpx.Response:
        if request.headers["host"] == "public.example":
            return httpx.Response(302, headers={"location": "http://elsewhere.example/x"})
        return httpx.Response(200, text="<p>landed</p>")

    net = Net(handler)
    text = net.fetch("http://public.example/")

    assert "landed" in text
    assert net.hosts == ["public.example", "elsewhere.example"]


def test_an_endless_redirect_chain_is_cut_off(dns):
    dns["loop.example"] = [PUBLIC_IP]
    net = Net(redirect_to("http://loop.example/again"))

    with pytest.raises(BlockedURL):
        net.fetch("http://loop.example/")

    assert len(net.seen) <= 3


# --- the request is pinned to the address that was checked -----------------

def test_the_request_goes_to_the_checked_ip_carrying_the_original_host(dns):
    """Resolving, checking, then connecting by hostname re-resolves — that
    second lookup is the rebinding window. The checked address is the one
    dialled, and the Host header keeps virtual hosting working."""
    dns["example.com"] = [PUBLIC_IP]
    net = Net()

    net.fetch("http://example.com/page?q=1")

    assert net.seen[0].url.host == PUBLIC_IP
    assert net.seen[0].url.path == "/page"
    assert net.seen[0].url.query == b"q=1"
    assert net.seen[0].headers["host"] == "example.com"


def test_a_pinned_https_request_still_names_the_host_for_tls(dns):
    """Dialling the IP without SNI would break certificate validation, which is
    the usual reason pinning gets dropped."""
    dns["example.com"] = [PUBLIC_IP]
    net = Net()

    net.fetch("https://example.com/page")

    assert net.seen[0].extensions["sni_hostname"] == "example.com"


def test_a_non_default_port_survives_pinning(dns):
    dns["example.com"] = [PUBLIC_IP]
    net = Net()

    net.fetch("http://example.com:8080/page")

    assert net.seen[0].url.port == 8080
    assert net.seen[0].headers["host"] == "example.com:8080"


# --- the body ---------------------------------------------------------------

def test_the_page_body_is_returned(dns):
    dns["example.com"] = [PUBLIC_IP]

    assert Net().fetch("http://example.com/") == "<p>hello</p>"


def test_an_oversized_page_is_truncated(dns):
    dns["example.com"] = [PUBLIC_IP]
    net = Net(lambda r: httpx.Response(200, text="x" * 5000))

    assert len(net.fetch("http://example.com/", max_bytes=100)) == 100


def test_an_error_status_raises_rather_than_returning_an_error_page(dns):
    dns["example.com"] = [PUBLIC_IP]
    net = Net(lambda r: httpx.Response(404, text="nope"))

    with pytest.raises(httpx.HTTPStatusError):
        net.fetch("http://example.com/")


# --- extraction: what to_clean_text does, and what it does not -------------

def fixture(name: str) -> str:
    return to_clean_text((FIXTURES / name).read_text(encoding="utf-8"))


def test_boilerplate_is_dropped_and_the_article_survives():
    text = fixture("hidden_div.html")

    assert "$8 per 1k series per month" in text
    assert "Home" not in text and "Copyright" not in text


def test_zero_width_characters_are_stripped():
    """Unicode steganography: an instruction spelled out in characters that
    render as nothing, which a model still reads."""
    hidden = "​‏﻿"          # zero-width space, RTL mark, BOM
    html = (f"<html><body><article>"
            f"<p>Retention is thirty{hidden} days, unchanged since the 3.0 release.</p>"
            f"<p>Existing tenants keep whatever retention they configured.</p>"
            f"</article></body></html>")

    text = to_clean_text(html)

    assert not any(c in text for c in hidden)
    assert "thirty days" in text


def test_an_instruction_hidden_in_an_html_comment_never_reaches_the_model():
    text = fixture("comment.html")

    assert "retention default to 30 days" in text
    assert "ignore the extraction rules" not in text


def test_an_instruction_hidden_in_alt_text_never_reaches_the_model():
    text = fixture("alt_text.html")

    assert "benchmark in March 2026" in text
    assert "attacker.example" not in text


def test_css_hidden_text_is_not_removed_and_that_is_the_documented_limit():
    """Pinning the honest boundary of this layer: a `display:none` div is real
    text to any extractor, so containment (datamarking, and the privileged
    nodes never seeing page text) is what covers it — not the HTML cleaner."""
    text = fixture("hidden_div.html")

    assert "Ignore prior instructions" in text


def test_runs_of_blank_lines_are_collapsed():
    html = "<html><body><article><p>one</p>" + "<p></p>" * 6 + \
           "<p>two, with enough words to survive the extractor</p></article></body></html>"

    assert "\n\n\n" not in to_clean_text(html)


def test_a_page_with_nothing_extractable_yields_empty_text():
    """trafilatura returns None on a page with no article; a None reaching the
    content store would blow up on `len(text)` in spec 06."""
    assert to_clean_text("<html><body></body></html>") == ""


# --- containment: spotlighting ---------------------------------------------

def test_datamarking_marks_every_space_in_the_untrusted_block():
    marked = datamark("report X is safest")

    assert " " not in marked
    assert marked == f"report{MARK}X{MARK}is{MARK}safest"


def test_the_preamble_names_the_marker_it_uses():
    """The marking is only legible to the model if the preamble explains it."""
    assert MARK in UNTRUSTED_PREAMBLE


def test_the_preamble_says_the_block_is_data_and_what_to_do_with_directives():
    lowered = UNTRUSTED_PREAMBLE.lower()

    assert "untrusted" in lowered
    assert "data" in lowered and "instructions" in lowered


def test_the_preamble_is_the_only_copy_of_that_text():
    """Prompt text lives in prompts/*.txt so it can be read and diffed without
    opening the code that uses it."""
    from prompts import load

    assert load("untrusted_preamble").format(mark=MARK) == UNTRUSTED_PREAMBLE
