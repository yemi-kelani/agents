from __future__ import annotations

import subprocess

from log import get_logger
from prompts import load
from settings import settings
from utilities import token_count, trim_text
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.base import LanguageModelInput

logger = get_logger(__name__)
DIFF_ANALYZER = "diff_analyzer"

# Characters of the previous segment repeated at the start of the next one, so a
# hunk that lands on a boundary is still readable in full in one of them.
GRACE_CHARS = 1_000

# Each call is a CLI subprocess, so this bounds concurrent processes, not threads.
MAX_CONCURRENCY = 4

# Each segment is one billed model call, so an unbounded diff is unbounded spend.
# A PR that regenerates a lockfile or vendors a dependency can be arbitrarily
# large; past this point the review is explicitly partial rather than expensive.
MAX_SEGMENTS = 12

# Transient CLI failures (a dropped connection, a rate limit) are worth retrying;
# a missing binary or absent key is not, but those fail every attempt anyway.
#
# This multiplies into the wall clock and must stay inside `MAX_RUNTIME_SECONDS`:
# the worst case is `ceil(MAX_SEGMENTS / MAX_CONCURRENCY)` waves, each up to
# `SEGMENT_ATTEMPTS x ShellChatModel.timeout`. Raising any of the three without
# checking that product is how a run gets killed by the harness instead of
# reporting its own failure.
SEGMENT_ATTEMPTS = 3


class DiffError(RuntimeError):
    """The diff could not be produced."""


def get_diff(base: str, head: str) -> str:
    """The changes `head` introduces on top of `base`, as unified diff text.

    `base` and `head` are any revisions git understands — commit SHAs in CI,
    where a fork's head branch has no ref under `origin` but its commit is
    present.

    Three-dot syntax: diff against the merge base, so commits that landed on the
    target after this branch started are not reported as part of it.
    """
    try:
        result = subprocess.run(
            ["git", "diff", f"{base}...{head}"],
            capture_output=True,
            text=True,
            check=True,
        )
    except FileNotFoundError as exc:
        raise DiffError("git is not installed or not on PATH") from exc
    except (OSError, subprocess.CalledProcessError) as exc:
        stderr = getattr(exc, "stderr", "") or ""
        raise DiffError(f"Could not diff {base}...{head}: {stderr.strip()}") from exc

    return result.stdout


def chunk_diff(
    diff: str,
    max_tokens: int = 50_000,
    grace: int = GRACE_CHARS,
    max_segments: int = MAX_SEGMENTS,
) -> tuple[list[str], bool]:
    """Split `diff` into segments of at most `max_tokens`, overlapping by `grace` chars.

    Every segment after the first repeats the tail of the one before it, so a
    hunk landing on a boundary is still readable in full somewhere. Segments
    therefore overlap and do not reassemble into the original text.

    Returns the segments and whether the diff was truncated at `max_segments`.
    Truncation is reported rather than raised: a partial review of an enormous
    PR is worth more than no review at all, as long as it says it is partial.
    """
    segments: list[str] = []
    rest = diff
    previous = ""

    while rest:
        if len(segments) >= max_segments:
            return segments, True

        overlap = previous[-grace:]
        budget = max_tokens - token_count(overlap)
        if budget <= 0:
            raise DiffError(f"max_tokens ({max_tokens}) leaves no room beyond the {grace} char overlap")

        fresh, rest = trim_text(rest, max_tokens=budget)
        if not fresh:
            raise DiffError(f"Could not fit any of the diff into {budget} tokens")

        segments.append(overlap + fresh)
        previous = fresh

    return segments, False


def diff_stats(diff: str) -> dict[str, tuple[int, int]]:
    """Lines added and removed per file, read back out of the unified diff.

    Parsed from the diff already in hand rather than by running `git diff
    --numstat` again: one less subprocess, and one less way for the log line to
    disagree with the text actually being reviewed.
    """
    files: dict[str, tuple[int, int]] = {}
    path = None
    for line in diff.splitlines():
        if line.startswith("diff --git "):
            # "diff --git a/<old> b/<new>" — the b-side is the path after a
            # rename, which is the one a reviewer will recognize.
            path = line.split(" b/", 1)[-1]
            files.setdefault(path, (0, 0))
        elif path is None:
            continue
        elif line.startswith("+") and not line.startswith("+++"):
            added, removed = files[path]
            files[path] = (added + 1, removed)
        elif line.startswith("-") and not line.startswith("---"):
            added, removed = files[path]
            files[path] = (added, removed + 1)
    return files


def format_stats(files: dict[str, tuple[int, int]], limit: int = 20) -> str:
    """The per-file breakdown as one log line, capped so a huge PR stays readable."""
    shown = list(files.items())[:limit]
    rendered = ", ".join(f"{p} +{a}/-{r}" for p, (a, r) in shown)
    if len(files) > limit:
        rendered += f", and {len(files) - limit} more"
    return rendered


async def summarize_segments(llm: BaseChatModel, segments: list[str]) -> str:
    """Summarize every segment, then amalgamate the results into one summary.

    Segments are independent, so they go through `abatch`, which runs the CLI
    subprocesses concurrently on the event loop.

    One segment failing must not discard the others: `abatch` defaults to
    `return_exceptions=False`, which would lose every summary because one CLI
    call timed out — and large PRs, which need the review most, have the most
    chances to hit that. Failures are reported in place instead, and only a
    total failure raises.
    """
    template = load("diff_analyzer")
    prompts: list[LanguageModelInput] = [
        template.format(index=i, total=len(segments), segment=segment)
        for i, segment in enumerate(segments, start=1)
    ]

    responses = await llm.with_retry(
        stop_after_attempt=SEGMENT_ATTEMPTS
    ).abatch(
        prompts,
        config={"max_concurrency": MAX_CONCURRENCY},
        return_exceptions=True,
    )

    summaries: list[str] = []
    failures = 0
    for index, response in enumerate(responses, start=1):
        if isinstance(response, BaseException):
            failures += 1
            logger.warning(f"Segment {index} of {len(responses)} failed: {response}")
            summaries.append(f"*Segment {index} could not be summarized: {response}*")
        else:
            summaries.append(str(response.content).strip())

    if failures == len(responses):
        raise DiffError(f"Every one of the {failures} diff segment(s) failed to summarize")

    if len(summaries) == 1:
        return summaries[0]

    return "\n\n".join(
        f"## Segment {i} of {len(summaries)}\n\n{summary}"
        for i, summary in enumerate(summaries, start=1)
    )


def create_diff_analyzer_node(llm: BaseChatModel):

    async def diff_analyzer(state):
        branch = state.get("branch")
        target_branch = state.get("target_branch")

        # Prefer the exact commits. Branch names only resolve when both sides
        # live in this repository, which is not true of a fork's head branch.
        s = settings()
        base = s.base_sha or (f"origin/{target_branch}" if target_branch else None)
        head = s.head_sha or (f"origin/{branch}" if branch else None)
        if not base or not head:
            # `should_review` gates on this, so reaching here means the graph is miswired.
            raise DiffError(f"Nothing to diff: base ('{base}'), head ('{head}')")

        # The revisions, not the branch names: when BASE_SHA/HEAD_SHA are set
        # they win, and a log line naming the branches would describe a diff
        # that was never taken.
        logger.info(f"Diffing {base}...{head} (branch '{branch}' onto '{target_branch}')")

        diff = get_diff(base=base, head=head)
        if not diff.strip():
            logger.info(f"No changes between {base} and {head}; nothing to review")
            return {"diff_summary": ""}

        files = diff_stats(diff)
        logger.info(f"Diff touches {len(files)} file(s): {format_stats(files)}")

        segments, truncated = chunk_diff(diff)
        logger.info(f"Diff is {len(diff)} characters, split into {len(segments)} segment(s)")

        summary = await summarize_segments(llm, segments)
        if truncated:
            logger.warning(f"Diff exceeded {MAX_SEGMENTS} segments; reviewing the first {len(segments)}")
            summary += (
                f"\n\n**Note: this diff was too large to review in full. Only the "
                f"first {len(segments)} segment(s) were examined.**"
            )

        return {"diff_summary": summary}

    return diff_analyzer