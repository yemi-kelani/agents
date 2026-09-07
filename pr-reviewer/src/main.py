"""Entrypoint: build the review graph and run it once. Called from CI."""

import os
import sys

from agent import Agent
from log import get_logger
from settings import settings

logger = get_logger()


def main() -> int:
    s = settings()
    cli = os.getenv("REVIEW_CLI", "codex")
    logger.info(f"Reviewing '{s.branch}' against '{s.target_branch}' with cli '{cli}'")

    graph = Agent(cli=cli).create_graph()
    graph.invoke(
        {
            "messages": [], 
            "branch": s.branch, 
            "target_branch": s.target_branch
        },
        config={"configurable": {"thread_id": "pr-review"}},
    )

    logger.info("Review finished")
    return 0


if __name__ == "__main__":
    sys.exit(main())
