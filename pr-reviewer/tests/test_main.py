"""Exit codes and posting.

The exit code is the only thing CI sees, so it has to distinguish "reviewed,
clean" from "never ran" — the two were indistinguishable before.
"""

import asyncio

import main as main_module
from agent import SKIPPED_ALREADY_REVIEWED, SKIPPED_MISCONFIGURED
from github_client import GitHubError
from main import EXIT_FAILED, EXIT_MISCONFIGURED, EXIT_OK, main, publish
from models import Critique
from settings import Settings

CRITIQUE = Critique(file="a.py", issue="leak", detail="d", line=3, severity="high")


def _configured(**overrides) -> Settings:
    values = dict(
        github_token="tok", github_repository="owner/repo", pr_number=7,
        branch="feature", target_branch="main",
    )
    values.update(overrides)
    return Settings(**values)


class TestPublish:
    def test_a_clean_review_posts_nothing_and_succeeds(self, monkeypatch):
        posted = []
        monkeypatch.setattr(main_module, "post_review",
                            lambda *a, **k: posted.append(a))

        assert publish([], _configured()) == EXIT_OK
        assert posted == []

    def test_findings_are_posted_and_do_not_fail_the_build(self, monkeypatch):
        """Findings are delivered by the comment, not by a red check."""
        posted = {}

        def fake_post(pr, body, token):
            posted.update(pr=pr, body=body, token=token)

        monkeypatch.setattr(main_module, "post_review", fake_post)

        assert publish([CRITIQUE], _configured()) == EXIT_OK
        assert posted["pr"].repository == "owner/repo"
        assert posted["pr"].number == 7
        assert "a.py" in posted["body"]
        assert posted["token"] == "tok"

    def test_findings_that_cannot_be_posted_fail_loudly(self, monkeypatch):
        """Silently dropping findings is the failure mode being prevented."""
        monkeypatch.setattr(main_module, "post_review",
                            lambda *a, **k: (_ for _ in ()).throw(AssertionError))

        assert publish([CRITIQUE], _configured(github_token=None)) == EXIT_MISCONFIGURED

    def test_a_failed_post_is_reported(self, monkeypatch):
        def fake_post(*args, **kwargs):
            raise GitHubError("nope")

        monkeypatch.setattr(main_module, "post_review", fake_post)
        assert publish([CRITIQUE], _configured()) == EXIT_FAILED


class _Graph:
    def __init__(self, result):
        self._result = result

    async def ainvoke(self, state, config=None):
        return self._result


def _run_main(monkeypatch, result=None, settings=None, raises=None):
    monkeypatch.setattr(main_module, "settings", lambda: settings or _configured())

    class FakeAgent:
        def __init__(self, cli, repo_path):
            pass

        def create_graph(self):
            if raises is not None:
                raise raises
            return _Graph(result)

    monkeypatch.setattr(main_module, "Agent", FakeAgent)
    monkeypatch.setattr(main_module, "post_review", lambda *a, **k: None)
    return asyncio.run(main())


class TestMainExitCodes:
    def test_misconfiguration_fails_the_build(self, monkeypatch):
        """The core regression: this used to exit 0, so a review that never
        happened looked exactly like a PR with no problems."""
        code = _run_main(monkeypatch, {"skipped": SKIPPED_MISCONFIGURED})
        assert code == EXIT_MISCONFIGURED

    def test_an_already_reviewed_pr_succeeds(self, monkeypatch):
        code = _run_main(monkeypatch, {"skipped": SKIPPED_ALREADY_REVIEWED})
        assert code == EXIT_OK

    def test_a_clean_review_succeeds(self, monkeypatch):
        code = _run_main(monkeypatch, {"skipped": None, "critiques": []})
        assert code == EXIT_OK

    def test_a_review_with_findings_succeeds(self, monkeypatch):
        code = _run_main(monkeypatch, {"skipped": None, "critiques": [CRITIQUE]})
        assert code == EXIT_OK

    def test_an_unhandled_failure_is_reported_not_raised(self, monkeypatch):
        code = _run_main(monkeypatch, raises=RuntimeError("codex is not installed"))
        assert code == EXIT_FAILED
