from __future__ import annotations

import subprocess
from log import get_logger
from langchain_core.language_models import BaseChatModel

logger = get_logger()
DIFF_ANALYZER = "diff_analyzer"


class DiffError(RuntimeError):
    """The diff could not be produced."""


def get_diff(branch: str, target_branch: str) -> str:
    """The changes `branch` introduces on top of `target_branch`, as unified diff text.

    Three-dot syntax: diff against the merge base, so commits that landed on the
    target after this branch started are not reported as part of it.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"origin/{target_branch}...origin/{branch}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise DiffError(f"Could not diff {branch} against {target_branch}: {stderr.strip()}") from exc

    return result.stdout

def create_diff_analyzer_node(llm: BaseChatModel):

    def diff_analyzer(state):
        branch = state.get("branch")
        target_branch = state.get("target_branch")
        if not branch or not target_branch:
            # `should_review` gates on this, so reaching here means the graph is miswired.
            raise DiffError(f"Missing branch ('{branch}') or target branch ('{target_branch}')")

        diff = get_diff(branch=branch, target_branch=target_branch)
        logger.info(f"Diff is {len(diff)} characters")

        # TODO: analyze `diff`.
        
        return state

    return diff_analyzer