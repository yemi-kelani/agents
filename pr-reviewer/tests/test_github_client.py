"""The review-counting guard and the review it posts.

Everything drives the `request=` seam the module already exposes, so no HTTP
happens and no response shape is invented beyond what the API documents.
"""

import email.message
import urllib.error

import pytest

import github_client
from github_client import (
    MARKER,
    GitHubError,
    PullRequest,
    count_agent_reviews,
    post_review,
)

PR = PullRequest(repository="owner/repo", number=7)


def _responder(*pages):
    """A fake `request` returning each page in turn."""
    remaining = list(pages)

    def request(method, url, token, payload=None):
        return remaining.pop(0)

    return request


class TestCountAgentReviews:
    def test_counts_only_our_own_reviews(self):
        reviews = [
            {"body": f"ours\n{MARKER}"},
            {"body": "a human review"},
            {"body": None},
            {"body": f"{MARKER}"},
        ]
        assert count_agent_reviews(PR, token="t", request=_responder(reviews)) == 2

    def test_a_non_list_body_raises_rather_than_reporting_zero(self):
        """A wrong zero would let the agent post duplicate reviews.

        This used to raise AttributeError from outside the try, bypassing the
        caller's `except GitHubError` entirely.
        """
        body = {"message": "API rate limit exceeded"}
        with pytest.raises(GitHubError):
            count_agent_reviews(PR, token="t", request=_responder(body))

    def test_an_empty_body_raises(self):
        with pytest.raises(GitHubError):
            count_agent_reviews(PR, token="t", request=_responder(None))

    def test_transport_failure_becomes_a_github_error(self):
        def request(*args, **kwargs):
            raise OSError("connection reset")

        with pytest.raises(GitHubError):
            count_agent_reviews(PR, token="t", request=request)

    def test_paginates_until_a_short_page(self):
        full = [{"body": MARKER}] * 100
        assert count_agent_reviews(
            PR, token="t", request=_responder(full, [{"body": MARKER}])) == 101

    def test_stops_on_an_exactly_empty_final_page(self):
        full = [{"body": MARKER}] * 100
        assert count_agent_reviews(PR, token="t", request=_responder(full, [])) == 100

    def test_uses_the_token_it_is_given(self):
        seen = {}

        def request(method, url, token, payload=None):
            seen["token"] = token
            return []

        count_agent_reviews(PR, token="secret-token", request=request)
        assert seen["token"] == "secret-token"


class TestPostReview:
    def test_stamps_the_marker_so_later_runs_recognize_it(self):
        sent = {}

        def request(method, url, token, payload=None):
            sent.update(method=method, url=url, token=token, payload=payload)
            return {}

        post_review(PR, "the findings", "tok", request=request)

        assert sent["method"] == "POST"
        assert sent["payload"]["event"] == "COMMENT"
        assert sent["payload"]["body"].startswith("the findings")
        # Without this the dedup guard can never see its own past work.
        assert MARKER in sent["payload"]["body"]

    def test_failure_becomes_a_github_error(self):
        def request(*args, **kwargs):
            raise OSError("503")

        with pytest.raises(GitHubError):
            post_review(PR, "body", "tok", request=request)

    def test_a_posted_review_is_counted_by_the_guard(self):
        """The guard and the poster have to agree, or dedup silently fails."""
        posted = {}

        def post(method, url, token, payload=None):
            posted["body"] = payload["body"]
            return {}

        post_review(PR, "findings", "tok", request=post)

        counted = count_agent_reviews(
            PR, token="tok", request=_responder([{"body": posted["body"]}]))
        assert counted == 1


class TestRetry:
    """Transient failures are retried; statements about our request are not.

    GitHub's REST guidance is to back off on 5xx and rate limits, honouring
    `retry-after` and `x-ratelimit-reset`. Retrying a 404 or a 422 only wastes
    the budget, since repeating the request verbatim cannot change the answer.
    """

    def _http_error(self, code, headers=None):
        """A real HTTPError, headers and all — `_retry_after` reads them."""
        hdrs = email.message.Message()
        for name, value in (headers or {}).items():
            hdrs[name] = value
        return urllib.error.HTTPError(
            "https://api.github.com/x", code, "boom", hdrs, None)

    def test_a_transient_failure_is_retried_and_can_succeed(self, monkeypatch):
        attempts = []
        slept = []

        def flaky(method, url, token, payload):
            attempts.append(1)
            if len(attempts) < 3:
                raise self._http_error(500)
            return [{"body": "ok"}]

        monkeypatch.setattr(github_client, "_send", flaky)
        result = github_client._request("GET", "u", "tok", sleep=slept.append)

        assert result == [{"body": "ok"}]
        assert len(attempts) == 3
        assert len(slept) == 2

    def test_a_persistent_transient_failure_is_distinguishable(self, monkeypatch):
        monkeypatch.setattr(
            github_client, "_send",
            lambda *a, **k: (_ for _ in ()).throw(self._http_error(503)))

        with pytest.raises(github_client.GitHubUnavailable):
            github_client._request("GET", "u", "tok", sleep=lambda _: None)

    def test_a_permanent_failure_is_not_retried(self, monkeypatch):
        attempts = []

        def denied(*args, **kwargs):
            attempts.append(1)
            raise self._http_error(404)

        monkeypatch.setattr(github_client, "_send", denied)
        with pytest.raises(urllib.error.HTTPError):
            github_client._request("GET", "u", "tok", sleep=lambda _: None)
        assert len(attempts) == 1

    def test_retry_after_is_honoured(self, monkeypatch):
        slept = []
        attempts = []

        def limited(*args, **kwargs):
            attempts.append(1)
            if len(attempts) == 1:
                raise self._http_error(403, {"retry-after": "7"})
            return []

        monkeypatch.setattr(github_client, "_send", limited)
        github_client._request("GET", "u", "tok", sleep=slept.append)
        assert slept == [7.0]

    def test_a_plain_403_is_a_permission_denial_not_a_rate_limit(self, monkeypatch):
        """Only a 403 carrying retry guidance is a secondary rate limit. A bare
        one means the token cannot do this, and repeating it will not help."""
        monkeypatch.setattr(
            github_client, "_send",
            lambda *a, **k: (_ for _ in ()).throw(self._http_error(403)))

        with pytest.raises(urllib.error.HTTPError):
            github_client._request("POST", "u", "tok", sleep=lambda _: None)
