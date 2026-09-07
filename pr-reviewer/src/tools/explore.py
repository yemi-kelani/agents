"""The one tool the review loop gets: ask a sub-CLI about the codebase.

The CLI behind `ShellChatModel` is itself an agent with filesystem access, so
exploration needs no read/list/grep of our own. It also keeps the outer history
small: this returns a digested answer rather than raw file contents, and that
history is re-sent in full on every step of the outer loop.
"""

from __future__ import annotations

from typing import Awaitable, Callable

from llm import get_model
from log import get_logger
from prompts import load

logger = get_logger(__name__)

# A sub-agent question should be answerable quickly; the default 300s is a
# budget for a whole task, and the outer loop can afford several of these.
EXPLORE_TIMEOUT = 120


def create_explore_tool(cli: str, repo_path: str) -> Callable[..., Awaitable[str]]:
    """Build `explore_codebase`, bound to a CLI running inside `repo_path`.

    `ShellChatModel.cwd` defaults to None, which would leave the sub-CLI in
    whatever directory the process happens to be in. It is set explicitly here.
    """
    model = get_model(cli=cli, cwd=repo_path, timeout=EXPLORE_TIMEOUT)
    template = load("explore_codebase")

    async def explore_codebase(question: str) -> str:
        """Ask a sub-agent with file access a question about the codebase."""
        logger.info(f"Exploring: {question}")
        response = await model.ainvoke(template.format(question=question))
        return str(response.content).strip()

    return explore_codebase
