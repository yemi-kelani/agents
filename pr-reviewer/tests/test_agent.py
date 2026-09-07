"""Graph routing and the skip-reason the exit code depends on."""

import asyncio

import agent as agent_module
import pytest
from agent import SKIPPED_ALREADY_REVIEWED, SKIPPED_MISCONFIGURED, Agent
from github_client import GitHubError
from langgraph.graph import END
from settings import Settings


def _settings(**overrides) -> Settings:
    values = dict(
        github_token="tok", github_repository="owner/repo", pr_number=7,
        base_sha="aaa", head_sha="bbb", max_reviews_per_pr=1,
    )
    values.update(overrides)
    return Settings(**values)


@pytest.fixture
def gate(monkeypatch):
    """`should_review`, with its settings and review count under our control."""
    def build(settings, reviewed=0, raises=None):
        monkeypatch.setattr(agent_module, "settings", lambda: settings)

        def count(pr, token, request=None):
            if raises is not None:
                raise raises
            return reviewed

        monkeypatch.setattr(agent_module, "count_agent_reviews", count)
        # __new__ rather than Agent(...): the constructor builds a real model,
        # and the gate under test never touches one.
        instance = Agent.__new__(Agent)
        return instance.should_review({"branch": settings.branch,
                                       "target_branch": settings.target_branch})

    return build


class TestShouldReview:
    def test_proceeds_when_configured(self, gate):
        assert gate(_settings())["skipped"] is None

    def test_shas_alone_are_enough_to_diff(self, gate):
        """A fork PR has no usable branch names, only commits."""
        result = gate(_settings(branch=None, target_branch=None))
        assert result["skipped"] is None

    def test_branch_names_alone_are_enough(self, gate):
        result = gate(_settings(base_sha=None, head_sha=None,
                                branch="feature", target_branch="main"))
        assert result["skipped"] is None

    def test_nothing_to_diff_is_misconfiguration(self, gate):
        result = gate(_settings(base_sha=None, head_sha=None,
                                branch=None, target_branch=None))
        assert result["skipped"] == SKIPPED_MISCONFIGURED

    def test_missing_github_config_is_misconfiguration(self, gate):
        assert gate(_settings(github_token=None))["skipped"] == SKIPPED_MISCONFIGURED
        assert gate(_settings(pr_number=None))["skipped"] == SKIPPED_MISCONFIGURED

    def test_an_uncountable_review_history_stops_the_run(self, gate):
        """Staying quiet beats risking a duplicate review."""
        result = gate(_settings(), raises=GitHubError("api down"))
        assert result["skipped"] == SKIPPED_MISCONFIGURED

    def test_an_already_reviewed_pr_is_skipped_legitimately(self, gate):
        result = gate(_settings(), reviewed=1)
        assert result["skipped"] == SKIPPED_ALREADY_REVIEWED

    def test_an_unlimited_quota_skips_the_api_entirely(self, gate):
        result = gate(_settings(max_reviews_per_pr=None, github_token=None))
        assert result["skipped"] is None


class TestRoute:
    def test_a_skip_ends_the_graph(self):
        assert Agent.route({"skipped": SKIPPED_MISCONFIGURED}) == END

    def test_no_skip_continues_to_the_review(self):
        assert Agent.route({"skipped": None}) == "diff_analyzer"


def test_the_skip_reason_survives_into_the_result(monkeypatch):
    """A conditional edge cannot write state — mutating the dict it receives is
    silently discarded, which is why the gate is a node."""
    monkeypatch.setattr(agent_module, "settings",
                        lambda: _settings(base_sha=None, head_sha=None))
    monkeypatch.setattr(agent_module, "get_model", lambda cli: object())

    graph = Agent(cli="codex", repo_path=".").create_graph()
    result = asyncio.run(graph.ainvoke(
        {"messages": [], "branch": None, "target_branch": None},
        config={"configurable": {"thread_id": "test"}}))

    assert result["skipped"] == SKIPPED_MISCONFIGURED
    assert result["critiques"] == []
