"""Entrypoint: build the review graph and run it once. Called from CI."""

import asyncio
import sys

from agent import SKIPPED_MISCONFIGURED, SKIPPED_UNAVAILABLE, Agent
from github_client import GitHubError, post_review
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

# The whole run, end to end. Every phase is bounded individually, but those
# bounds multiply: the diff fan-out alone is
# `ceil(MAX_SEGMENTS / MAX_CONCURRENCY) waves x SEGMENT_ATTEMPTS x timeout`,
# which can exceed the workflow's `timeout-minutes`. Being killed by the harness
# produces no comment and no exit code, so the process enforces its own deadline
# below that and reports the overrun itself.
MAX_RUNTIME_SECONDS = 1_500  # 25 minutes, against the workflow's 30


def publish(critiques: list[Critique], s: Settings) -> int:
    """Post the findings to the PR. Returns an exit code."""
    pr = s.pull_request()
    if pr is None:
        logger.error(
            f"Found {len(critiques)} critique(s) but cannot post them. "
            f"{s.github_config_error()}")
        return EXIT_MISCONFIGURED

    try:
        # FIX: a clean review is posted too. Returning early wrote no marker, so
        # `count_agent_reviews` never saw the run and every later push paid for a
        # full re-review — and the author could not tell a clean review from one
        # that never happened.
        post_review(pr, format_review(critiques), s.github_token or "")
    except GitHubError:
        logger.exception("Could not post the review")
        return EXIT_FAILED

    logger.info(f"Posted {len(critiques)} critique(s) to {pr.repository}#{pr.number}")
    return EXIT_OK


async def main() -> int:
    s = settings()
    logger.info(
        f"Reviewing '{s.branch}' against '{s.target_branch}' "
        f"with cli '{s.review_cli}' in {s.repo_path}")

    try:
        graph = Agent(cli=s.review_cli, repo_path=s.repo_path).create_graph()
        result = await asyncio.wait_for(
            graph.ainvoke({
                "messages": [], "branch": s.branch, "target_branch": s.target_branch,
            }),
            timeout=MAX_RUNTIME_SECONDS,
        )
    except asyncio.TimeoutError:
        logger.error(
            f"The review exceeded its {MAX_RUNTIME_SECONDS}s budget and was stopped")
        return EXIT_FAILED
    except Exception:
        # An unhandled failure here means no review was produced. Say so plainly
        # rather than letting a bare traceback be the only signal.
        logger.exception("The review failed to run")
        return EXIT_FAILED

    skipped = result.get("skipped")
    if skipped:
        # Broken configuration must fail the build, and so must an unreachable
        # GitHub — but a run correctly skipped because the PR is already
        # reviewed is a success.
        logger.info(f"Review skipped: {skipped}")
        if skipped == SKIPPED_MISCONFIGURED:
            return EXIT_MISCONFIGURED
        return EXIT_FAILED if skipped == SKIPPED_UNAVAILABLE else EXIT_OK

    critiques = result.get("critiques") or []
    logger.info(f"Review finished with {len(critiques)} critique(s)")
    return publish(critiques, s)


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
