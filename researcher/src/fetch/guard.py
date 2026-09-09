"""SSRF guard — spec 05.

Any `fetch(url)` an agent can steer is a path to the cloud metadata endpoint;
`http://169.254.169.254/latest/meta-data/iam/security-credentials/` returns IAM
credentials, which is the shape of the 2019 Capital One breach.

The detail most implementations get wrong: **resolve the hostname, check every
resolved address, then connect to the address that was checked.** Checking the
hostname and then fetching by hostname resolves twice, and the second lookup is
a DNS-rebinding window — first answer public, second answer 127.0.0.1.

Blocklists alone are a losing game (decimal notation, IPv6 equivalents,
rebinding); an allowlist is stronger but a research agent needs the open web. So:
private-range denial + address pinning + a re-check on every redirect hop. Say
that plainly in the README rather than overclaiming.
"""
from __future__ import annotations

import ipaddress
import socket
from urllib.parse import urlparse

BLOCKED_NETS = [
    ipaddress.ip_network("127.0.0.0/8"),      # loopback
    ipaddress.ip_network("10.0.0.0/8"),       # RFC1918
    ipaddress.ip_network("172.16.0.0/12"),
    ipaddress.ip_network("192.168.0.0/16"),
    ipaddress.ip_network("169.254.0.0/16"),   # link-local INCLUDING cloud metadata
    ipaddress.ip_network("::1/128"),
    ipaddress.ip_network("fc00::/7"),         # IPv6 ULA
    ipaddress.ip_network("fe80::/10"),        # IPv6 link-local
]

ALLOWED_SCHEMES = ("http", "https")


class BlockedURL(Exception): ...


def resolve_and_check(url: str) -> tuple[str, str]:
    """Return (safe_ip, host). Raise BlockedURL if any resolved IP is private."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise BlockedURL(f"scheme not allowed: {parsed.scheme}")
    host = parsed.hostname
    if not host:
        raise BlockedURL("no host")

    infos = socket.getaddrinfo(host, None)
    ips = {i[4][0] for i in infos}
    # Every record, not just the first: a host that answers with one public and
    # one private address defeats a check that stops at ips[0].
    for ip in ips:
        addr = ipaddress.ip_address(ip.split("%")[0])   # drop any zone id
        if any(addr in net for net in BLOCKED_NETS):
            raise BlockedURL(f"{host} resolves to private address {ip}")
    return next(iter(ips)), host        # pin this IP for the request
