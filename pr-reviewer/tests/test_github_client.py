"""The review-counting guard and the review it posts.

Everything drives the `request=` seam the module already exposes, so no HTTP
happens and no response shape is invented beyond what the API documents.
"""

import pytest

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
