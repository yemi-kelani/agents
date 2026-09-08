"""The terminal client: working out what to review, and showing the result."""

import asyncio

import pytest

import review
from github_client import ACCEPT_DIFF, PullRequest, pull_request_diff
from models import Critique

DIFF = (
    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1,2 @@\n context\n+added\n"
)


class TestResolve:
    """Four target shapes, tried most-specific first."""

    def test_a_pull_request_url_is_fetched_from_github(self, monkeypatch, tmp_path):
        seen = {}

        def fake_fetch(pr, token):
            seen.update(repository=pr.repository, number=pr.number, token=token)
            return DIFF

        monkeypatch.setattr(review, "pull_request_diff", fake_fetch)
        monkeypatch.setenv("GITHUB_TOKEN", "tok")

        diff, described = review.resolve(
            "https://github.com/owner/repo/pull/42")

        assert diff == DIFF
        assert described == "owner/repo#42"
        assert seen == {"repository": "owner/repo", "number": 42, "token": "tok"}

    def test_the_short_reference_form_is_equivalent(self, monkeypatch):
        monkeypatch.setattr(review, "pull_request_diff", lambda pr, token: DIFF)
        monkeypatch.setenv("GITHUB_TOKEN", "tok")

        _, described = review.resolve("owner/repo#7")
        assert described == "owner/repo#7"

    def test_a_saved_diff_file_needs_no_network_or_token(self, monkeypatch, tmp_path):
        """The offline path: a demo must not depend on the venue's wifi."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        path = tmp_path / "saved.diff"
        path.write_text(DIFF)

        diff, described = review.resolve(str(path))
        assert diff == DIFF
        assert described == str(path)

    def test_a_revision_range_is_diffed_locally(self, monkeypatch):
        monkeypatch.setattr(review, "local_diff", lambda rev: DIFF)
        diff, described = review.resolve("main..HEAD")
        assert (diff, described) == (DIFF, "main..HEAD")

    def test_a_pull_request_without_a_token_says_so_plainly(self, monkeypatch):
        """Naming the missing variable beats a 401 from three frames down."""
        monkeypatch.delenv("GITHUB_TOKEN", raising=False)
        with pytest.raises(ValueError, match="GITHUB_TOKEN"):
            review.resolve("owner/repo#7")

    def test_a_file_is_preferred_over_a_revision_range(self, monkeypatch, tmp_path):
        """A path that exists is a file, even if git would also accept the name."""
        called = []
        monkeypatch.setattr(review, "local_diff", lambda rev: called.append(rev) or "")
        path = tmp_path / "HEAD"
        path.write_text(DIFF)

        diff, _ = review.resolve(str(path))
        assert diff == DIFF
        assert called == []


class TestRender:
    def test_findings_are_ordered_most_severe_first(self, monkeypatch):
        monkeypatch.setattr(review.sys.stdout, "isatty", lambda: False)
        out = review.render([
            Critique(file="b.py", issue="minor", detail="", severity="low"),
            Critique(file="a.py", issue="crash", detail="", severity="high"),
        ])
        assert out.index("HIGH") < out.index("LOW")

    def test_a_clean_review_says_so(self, monkeypatch):
        monkeypatch.setattr(review.sys.stdout, "isatty", lambda: False)
        assert review.render([]) == "No problems found."

    def test_a_line_number_is_shown_when_known(self, monkeypatch):
        monkeypatch.setattr(review.sys.stdout, "isatty", lambda: False)
        out = review.render(
            [Critique(file="a.py", line=12, issue="x", detail="", severity="high")])
        assert "a.py:12" in out

    def test_no_ansi_codes_when_output_is_redirected(self, monkeypatch):
        """So piping the findings into a file or a pager stays readable."""
        monkeypatch.setattr(review.sys.stdout, "isatty", lambda: False)
        out = review.render(
            [Critique(file="a.py", issue="x", detail="d", severity="high")])
        assert "\033[" not in out


class TestRun:
    def test_an_empty_diff_succeeds_without_calling_the_model(self, monkeypatch):
        monkeypatch.setattr(review, "resolve", lambda t: ("", "nothing"))

        def explode(*args, **kwargs):
            raise AssertionError("the model must not be called")

        monkeypatch.setattr(review, "Agent", explode)
        assert asyncio.run(review.review("x", "codex", ".")) == review.EXIT_OK

    def test_an_unresolvable_target_is_misconfiguration_not_failure(self, monkeypatch):
        def bad(target):
            raise ValueError("no such thing")

        monkeypatch.setattr(review, "resolve", bad)
        assert asyncio.run(
            review.review("x", "codex", ".")) == review.EXIT_MISCONFIGURED


class TestPullRequestDiff:
    def test_the_diff_media_type_is_requested(self):
        """JSON metadata is the default; the diff needs an explicit Accept."""
        seen = {}

        def request(method, url, token, payload, accept=None):
            seen.update(method=method, url=url, accept=accept)
            return DIFF

        got = pull_request_diff(
            PullRequest(repository="owner/repo", number=3), "tok", request=request)

        assert got == DIFF
        assert seen["accept"] == ACCEPT_DIFF
        assert seen["url"].endswith("/repos/owner/repo/pulls/3")
