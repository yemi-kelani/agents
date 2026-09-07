"""Entrypoint: build the review graph and run it once. Called from CI."""

import asyncio
import os
import sys

from agent import Agent
from log import get_logger
from settings import settings

logger = get_logger()


async def main() -> int:
    s = settings()
    cli = os.getenv("REVIEW_CLI", "codex")
    # GITHUB_WORKSPACE is the checkout root on Actions; the cwd is the fallback
    # for local runs. The sub-agent explores whatever this points at.
    repo_path = os.getenv("GITHUB_WORKSPACE") or os.getcwd()
    logger.info(f"Reviewing '{s.branch}' against '{s.target_branch}' with cli '{cli}' in {repo_path}")

    graph = Agent(cli=cli, repo_path=repo_path).create_graph()
    result = await graph.ainvoke(
        {"messages": [], "branch": s.branch, "target_branch": s.target_branch},
        config={"configurable": {"thread_id": "pr-review"}},
    )

    critiques = result.get("critiques") or []
    logger.info(f"Review finished with {len(critiques)} critique(s)")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
