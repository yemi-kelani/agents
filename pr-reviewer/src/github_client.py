"""A very small GitHub REST client: enough to count our own reviews and post one.

Only two endpoints are needed, so this uses `urllib` rather than adding an HTTP
dependency. The `request` seam on each function is what the tests drive.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

from log import get_logger

logger = get_logger(__name__)

API_ROOT = "https://api.github.com"
PER_PAGE = 100

# urlopen has no timeout by default, so a hung connection would block until the
# CI job itself is killed — and the pagination loop below gives it several
# chances to happen.
TIMEOUT_SECONDS = 30

# Retries are for the transient failures GitHub documents — 5xx and rate limits.
# Bounded, because the caller sits inside the run's overall deadline.
MAX_ATTEMPTS = 3
BACKOFF_SECONDS = 2.0
MAX_BACKOFF_SECONDS = 30.0

# An HTML comment: present in the raw review body the API returns, invisible in
# the markdown GitHub renders. This is how a later run recognizes our own work.
MARKER = "<!-- pr-reviewer:v1 -->"


class GitHubError(RuntimeError):
    """The API could not be reached, or answered with something unusable."""


class GitHubUnavailable(GitHubError):
    """A transient failure that outlived its retries.

    Separate from `GitHubError` so the caller can tell "GitHub was down" from
    "this reviewer is misconfigured". They need different exit codes: one is
    worth retrying the job, the other needs a human to fix the wiring.
    """


@dataclass(frozen=True)
class PullRequest:
    repository: str  # "owner/repo"
    number: int


def _retry_after(exc: urllib.error.HTTPError) -> float | None:
    """How long GitHub asked us to wait, or None if it did not say.

    GitHub's REST guidance: honour `retry-after` when present, otherwise wait
    until `x-ratelimit-reset` when the remaining quota is exhausted.
    """
    headers = exc.headers or {}

    explicit = headers.get("retry-after")
    if explicit:
        try:
            return max(0.0, float(explicit))
        except ValueError:
            pass

    if headers.get("x-ratelimit-remaining") == "0":
        reset = headers.get("x-ratelimit-reset")
        if reset is None:
            return None
        try:
            return max(0.0, float(reset) - time.time())
        except (TypeError, ValueError):
            return None
    return None


def _is_transient(exc: Exception) -> bool:
    """Whether retrying `exc` could plausibly succeed.

    A 5xx or a rate limit is worth another attempt. A 401, 403-without-a-rate-
    limit, 404 or 422 is a statement about our request, and repeating it
    verbatim only wastes the budget.
    """
    if isinstance(exc, urllib.error.HTTPError):
        if exc.code >= 500 or exc.code == 429:
            return True
        # A 403 is GitHub's secondary-rate-limit response as well as its
        # permission denial; only the former carries retry guidance.
        return exc.code == 403 and _retry_after(exc) is not None
    # A transport failure never reached GitHub at all.
    return isinstance(exc, urllib.error.URLError)


def _send(method: str, url: str, token: str, payload=None):
    """One authenticated request, decoded. No retry — see `_request`."""
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
    with urllib.request.urlopen(req, timeout=TIMEOUT_SECONDS) as response:
        body = response.read()
    return json.loads(body) if body else None


def _request(method: str, url: str, token: str, payload=None, sleep=time.sleep):
    """Send one request, retrying the failures GitHub says are worth retrying.

    Raises `GitHubUnavailable` when a transient failure survives every attempt,
    and lets anything else propagate to the caller's own error handling.
    """
    for attempt in range(1, MAX_ATTEMPTS + 1):
        try:
            return _send(method, url, token, payload)
        except Exception as exc:
            if not _is_transient(exc) or attempt == MAX_ATTEMPTS:
                if _is_transient(exc):
                    raise GitHubUnavailable(
                        f"{method} {url} failed {MAX_ATTEMPTS} times: {exc}") from exc
                raise

            # Exponential backoff, overridden by whatever GitHub asked for.
            delay = min(BACKOFF_SECONDS * 2 ** (attempt - 1), MAX_BACKOFF_SECONDS)
            if isinstance(exc, urllib.error.HTTPError):
                delay = _retry_after(exc) or delay
            logger.warning(
                f"{method} {url} failed ({exc}); retrying in {delay:.0f}s "
                f"[attempt {attempt}/{MAX_ATTEMPTS}]")
            sleep(delay)


def _reviews_url(pr: PullRequest) -> str:
    return f"{API_ROOT}/repos/{pr.repository}/pulls/{pr.number}/reviews"


def count_agent_reviews(pr: PullRequest, token: str, request=_request) -> int:
    """How many reviews on `pr` this agent has already posted.

    Raises `GitHubError` rather than guessing when the answer cannot be
    established. The caller treats "unknown" as a reason to stay quiet, so
    reporting a wrong zero here would let the agent post duplicate reviews.
    """
    count = 0
    page = 1
    while True:
        url = f"{_reviews_url(pr)}?per_page={PER_PAGE}&page={page}"
        try:
            reviews = request("GET", url, token, None)

            # An unexpected body must not be read as "no previous reviews". This
            # sits inside the try so a malformed response raises GitHubError like
            # any other failure, rather than an AttributeError that would bypass
            # the caller's handling entirely.
            if not isinstance(reviews, list):
                raise GitHubError(
                    f"Expected a list of reviews for {pr.repository}#{pr.number}, "
                    f"got {type(reviews).__name__}")

            count += sum(
                1 for review in reviews
                if isinstance(review, dict) and MARKER in (review.get("body") or ""))
        except GitHubError:
            raise
        except Exception as exc:  # urllib raises a wide family of errors
            raise GitHubError(f"Could not list reviews for {pr.repository}#{pr.number}") from exc

        if len(reviews) < PER_PAGE:
            return count
        page += 1


def post_review(pr: PullRequest, body: str, token: str, request=_request) -> None:
    """Post a review comment on `pr`, stamped so later runs can recognize it."""
    payload = {"body": f"{body}\n\n{MARKER}", "event": "COMMENT"}
    try:
        request("POST", _reviews_url(pr), token, payload)
    except GitHubError:
        raise
    except Exception as exc:
        raise GitHubError(f"Could not post a review on {pr.repository}#{pr.number}") from exc
