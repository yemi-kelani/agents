"""Entrypoint: build the review graph and run it once. Called from CI."""

import asyncio
import os
import sys

from agent import SKIPPED_MISCONFIGURED, Agent
from github_client import GitHubError, PullRequest, post_review
from log import get_logger
from models import Critique, format_review
from settings import Settings, settings

logger = get_logger(__name__)

# A review that ran is a success whether or not it found anything: the findings
# are delivered by the PR comment, not by the exit code. Only a reviewer that
# could not do its job fails the build, so that a misconfigured run is never
# mistaken for a clean one.
EXIT_OK = 0
EXIT_MISCONFIGURED = 2
EXIT_FAILED = 1


def publish(critiques: list[Critique], s: Settings) -> int:
    """Post the findings to the PR. Returns an exit code."""
    if not critiques:
        logger.info("No problems found; nothing to post")
        return EXIT_OK

    if not s.github_token or not s.github_repository or s.pr_number is None:
        logger.error(
            "Found %d critique(s) but cannot post them. "
            "repository: '%s', pr_number: %s, token: %s",
            len(critiques), s.github_repository, s.pr_number,
            "set" if s.github_token else "missing",
        )
        return EXIT_MISCONFIGURED

    pr = PullRequest(repository=s.github_repository, number=s.pr_number)
    try:
        post_review(pr, format_review(critiques), s.github_token)
    except GitHubError:
        logger.exception("Could not post the review")
        return EXIT_FAILED

    logger.info(f"Posted {len(critiques)} critique(s) to {pr.repository}#{pr.number}")
    return EXIT_OK


async def main() -> int:
    s = settings()
    cli = os.getenv("REVIEW_CLI", "codex")
    # GITHUB_WORKSPACE is the checkout root on Actions; the cwd is the fallback
    # for local runs. The sub-agent explores whatever this points at.
    repo_path = os.getenv("GITHUB_WORKSPACE") or os.getcwd()
    logger.info(f"Reviewing '{s.branch}' against '{s.target_branch}' with cli '{cli}' in {repo_path}")

    try:
        graph = Agent(cli=cli, repo_path=repo_path).create_graph()
        result = await graph.ainvoke(
            {"messages": [], "branch": s.branch, "target_branch": s.target_branch},
            config={"configurable": {"thread_id": "pr-review"}},
        )
    except Exception:
        # An unhandled failure here means no review was produced. Say so plainly
        # rather than letting a bare traceback be the only signal.
        logger.exception("The review failed to run")
        return EXIT_FAILED

    skipped = result.get("skipped")
    if skipped:
        # Broken configuration must fail the build. A run that was correctly
        # skipped because the PR is already reviewed is a success.
        logger.info(f"Review skipped: {skipped}")
        return EXIT_MISCONFIGURED if skipped == SKIPPED_MISCONFIGURED else EXIT_OK

    critiques = result.get("critiques") or []
    logger.info(f"Review finished with {len(critiques)} critique(s)")
    return publish(critiques, s)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
