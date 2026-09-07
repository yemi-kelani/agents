"""Resolving configuration from the environment."""

import settings as settings_module
from settings import settings

# Every variable the resolver consults, cleared before each case so a value
# leaking in from the developer's shell or .env cannot make a test pass.
_VARS = (
    "GITHUB_HEAD_REF", "GITHUB_BASE_REF",
    "CI_COMMIT_BRANCH", "CI_MERGE_REQUEST_TARGET_BRANCH_NAME",
    "BASE_SHA", "HEAD_SHA",
    "PR_NUMBER", "GITHUB_REF", "GITHUB_REPOSITORY", "GITHUB_TOKEN",
    "OPENAI_API_KEY", "MAX_REVIEWS_PER_PR", "LLM_MODEL_NAME",
)


def _env(monkeypatch, **values):
    for name in _VARS:
        monkeypatch.delenv(name, raising=False)
    for name, value in values.items():
        monkeypatch.setenv(name, value)


class TestBranches:
    def test_prefers_githubs_own_variables(self, monkeypatch):
        _env(monkeypatch,
             GITHUB_HEAD_REF="feature", GITHUB_BASE_REF="main",
             CI_COMMIT_BRANCH="gitlab-head",
             CI_MERGE_REQUEST_TARGET_BRANCH_NAME="gitlab-base")

        s = settings()
        assert s.branch == "feature"
        assert s.target_branch == "main"

    def test_falls_back_to_the_gitlab_names(self, monkeypatch):
        _env(monkeypatch,
             CI_COMMIT_BRANCH="feature",
             CI_MERGE_REQUEST_TARGET_BRANCH_NAME="main")

        s = settings()
        assert s.branch == "feature"
        assert s.target_branch == "main"

    def test_absent_branches_are_none_not_empty(self, monkeypatch):
        _env(monkeypatch, GITHUB_HEAD_REF="")
        assert settings().branch is None


class TestCommits:
    def test_reads_the_shas(self, monkeypatch):
        _env(monkeypatch, BASE_SHA="aaa111", HEAD_SHA="bbb222")

        s = settings()
        assert s.base_sha == "aaa111"
        assert s.head_sha == "bbb222"

    def test_absent_shas_are_none(self, monkeypatch):
        _env(monkeypatch)
        assert settings().base_sha is None


class TestPrNumber:
    def test_prefers_an_explicit_number(self, monkeypatch):
        _env(monkeypatch, PR_NUMBER="42", GITHUB_REF="refs/pull/99/merge")
        assert settings().pr_number == 42

    def test_falls_back_to_the_actions_ref(self, monkeypatch):
        _env(monkeypatch, GITHUB_REF="refs/pull/99/merge")
        assert settings().pr_number == 99

    def test_a_branch_ref_yields_nothing(self, monkeypatch):
        _env(monkeypatch, GITHUB_REF="refs/heads/main")
        assert settings().pr_number is None

    def test_an_unparseable_number_falls_back(self, monkeypatch):
        _env(monkeypatch, PR_NUMBER="not-a-number", GITHUB_REF="refs/pull/7/merge")
        assert settings().pr_number == 7


class TestGeneral:
    def test_max_reviews_defaults_to_one(self, monkeypatch):
        _env(monkeypatch)
        assert settings().max_reviews_per_pr == 1

    def test_an_unparseable_max_reviews_falls_back(self, monkeypatch):
        _env(monkeypatch, MAX_REVIEWS_PER_PR="lots")
        assert settings().max_reviews_per_pr == 1

    def test_an_empty_secret_is_none_not_empty(self, monkeypatch):
        """An empty key must read as absent, or the CLI reports a confusing
        auth error instead of the obvious missing-key one."""
        _env(monkeypatch, GITHUB_TOKEN="")
        assert settings().github_token is None

    def test_the_openai_key_feeds_the_codex_slot(self, monkeypatch):
        """The codex CLI reads CODEX_API_KEY, but CI supplies OPENAI_API_KEY;
        llm.py bridges the two, so this mapping has to hold."""
        _env(monkeypatch, OPENAI_API_KEY="sk-abc")
        assert settings().codex_api_key == "sk-abc"


def test_pull_ref_pattern_requires_a_trailing_segment():
    """`refs/pull/<n>` alone is not a PR ref Actions produces."""
    assert settings_module._PULL_REF.match("refs/pull/12/merge")
    assert settings_module._PULL_REF.match("refs/pull/12/head")
    assert not settings_module._PULL_REF.match("refs/heads/pull")
