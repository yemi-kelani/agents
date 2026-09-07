from __future__ import annotations

import subprocess
from log import get_logger
from prompts import load
from utilities import token_count, trim_text
from langchain_core.language_models import BaseChatModel

logger = get_logger()
DIFF_ANALYZER = "diff_analyzer"

# Characters of the previous segment repeated at the start of the next one, so a
# hunk that lands on a boundary is still readable in full in one of them.
GRACE_CHARS = 1_000

# Each call is a CLI subprocess, so this bounds concurrent processes, not threads.
MAX_CONCURRENCY = 4


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


def chunk_diff(diff: str, max_tokens: int = 50_000, grace: int = GRACE_CHARS) -> list[str]:
    """Split `diff` into segments of at most `max_tokens`, overlapping by `grace` chars.

    Every segment after the first repeats the tail of the one before it, so a
    hunk landing on a boundary is still readable in full somewhere. Segments
    therefore overlap and do not reassemble into the original text.
    """
    segments: list[str] = []
    rest = diff
    previous = ""

    while rest:
        overlap = previous[-grace:]
        budget = max_tokens - token_count(overlap)
        if budget <= 0:
            raise DiffError(f"max_tokens ({max_tokens}) leaves no room beyond the {grace} char overlap")

        fresh, rest = trim_text(rest, max_tokens=budget)
        if not fresh:
            raise DiffError(f"Could not fit any of the diff into {budget} tokens")

        segments.append(overlap + fresh)
        previous = fresh

    return segments


async def summarize_segments(llm: BaseChatModel, segments: list[str]) -> str:
    """Summarize every segment, then amalgamate the results into one summary.

    Segments are independent, so they go through `abatch`, which runs the CLI
    subprocesses concurrently on the event loop.
    """
    template = load("diff_analyzer")
    prompts = [
        template.format(index=i, total=len(segments), segment=segment)
        for i, segment in enumerate(segments, start=1)
    ]

    responses = await llm.abatch(prompts, config={"max_concurrency": MAX_CONCURRENCY})

    if len(responses) == 1:
        return str(responses[0].content).strip()

    return "\n\n".join(
        f"## Segment {i} of {len(responses)}\n\n{str(response.content).strip()}"
        for i, response in enumerate(responses, start=1)
    )


def create_diff_analyzer_node(llm: BaseChatModel):

    async def diff_analyzer(state):
        branch = state.get("branch")
        target_branch = state.get("target_branch")
        if not branch or not target_branch:
            # `should_review` gates on this, so reaching here means the graph is miswired.
            raise DiffError(f"Missing branch ('{branch}') or target branch ('{target_branch}')")

        diff = get_diff(branch=branch, target_branch=target_branch)
        if not diff.strip():
            logger.info(f"No changes on '{branch}' against '{target_branch}'")
            return {"diff_summary": ""}

        segments = chunk_diff(diff)
        logger.info(f"Diff is {len(diff)} characters, split into {len(segments)} segment(s)")

        return {"diff_summary": await summarize_segments(llm, segments)}

    return diff_analyzer