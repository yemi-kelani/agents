"""A very small GitHub REST client: enough to count our own reviews and post one.

Only two endpoints are needed, so this uses `urllib` rather than adding an HTTP
dependency. The `request` seam on each function is what the tests drive.
"""

from __future__ import annotations

import json
import urllib.request
from dataclasses import dataclass

from settings import settings

API_ROOT = "https://api.github.com"
PER_PAGE = 100

# An HTML comment: present in the raw review body the API returns, invisible in
# the markdown GitHub renders. This is how a later run recognizes our own work.
MARKER = "<!-- pr-reviewer:v1 -->"


class GitHubError(RuntimeError):
    """The API could not be reached, or answered with something unusable."""


@dataclass(frozen=True)
class PullRequest:
    repository: str  # "owner/repo"
    number: int


def _request(method: str, url: str, token: str, payload=None):
    """Send one authenticated request and decode the JSON body."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url,
        data=data,
        method=method,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
            "User-Agent": "pr-reviewer",
        },
    )
    with urllib.request.urlopen(req) as response:
        body = response.read()
    return json.loads(body) if body else {}


def _reviews_url(pr: PullRequest) -> str:
    return f"{API_ROOT}/repos/{pr.repository}/pulls/{pr.number}/reviews"


def count_agent_reviews(pr: PullRequest, request=_request) -> int:
    """How many reviews on `pr` this agent has already posted."""
    count = 0
    page = 1
    while True:
        url = f"{_reviews_url(pr)}?per_page={PER_PAGE}&page={page}"
        try:
            reviews = request("GET", url, settings().github_token, None)
        except Exception as exc:  # urllib raises a wide family of errors
            raise GitHubError(f"Could not list reviews for {pr.repository}#{pr.number}") from exc

        count += sum(1 for review in reviews if MARKER in (review.get("body") or ""))
        if len(reviews) < PER_PAGE:
            return count
        page += 1


def post_review(pr: PullRequest, body: str, token: str, request=_request) -> None:
    """Post a review comment on `pr`, stamped so later runs can recognize it."""
    payload = {"body": f"{body}\n\n{MARKER}", "event": "COMMENT"}
    try:
        request("POST", _reviews_url(pr), token, payload)
    except Exception as exc:
        raise GitHubError(f"Could not post a review on {pr.repository}#{pr.number}") from exc
