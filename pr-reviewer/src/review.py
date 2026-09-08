"""Review a pull request, a local revision range, or a saved diff, from a terminal.

The same graph CI runs, pointed at a target you name:

    python review.py https://github.com/owner/repo/pull/3
    python review.py owner/repo#3
    python review.py main..HEAD
    python review.py saved.diff

Nothing is posted. The gate that stops CI double-commenting is skipped, because
a run that only prints has nothing to double-post.
"""

from __future__ import annotations

import argparse
import asyncio
import re
import subprocess
import sys
import time
from pathlib import Path

from agent import Agent
from github_client import GitHubError, PullRequest, pull_request_diff
from log import configure, get_logger
from models import Critique
from nodes.diff_analyzer import diff_stats, format_stats
from settings import settings

logger = get_logger(__name__)

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_MISCONFIGURED = 2

_PR_URL = re.compile(r"github\.com/([^/]+/[^/]+)/pull/(\d+)")
_PR_SHORT = re.compile(r"^([\w.-]+/[\w.-]+)#(\d+)$")

# Severity -> colour. Only used when stdout is a terminal.
_COLOURS = {"high": "\033[31m", "medium": "\033[33m", "low": "\033[36m"}
_BOLD, _DIM, _RESET = "\033[1m", "\033[2m", "\033[0m"


def _paint(text: str, *codes: str) -> str:
    """Wrap `text` in ANSI codes, unless output is redirected."""
    if not sys.stdout.isatty() or not codes:
        return text
    return f"{''.join(codes)}{text}{_RESET}"


def local_diff(revisions: str) -> str:
    """The diff for a git revision range, e.g. `main..HEAD` or `HEAD~3`."""
    result = subprocess.run(
        ["git", "diff", revisions], capture_output=True, text=True)
    if result.returncode != 0:
        raise ValueError(f"git could not diff {revisions!r}: {result.stderr.strip()}")
    return result.stdout


def resolve(target: str) -> tuple[str, str]:
    """Turn a target into (diff, a description of where it came from).

    Four shapes, tried in order of how specific they are: a pull request URL, an
    `owner/repo#n` reference, a path to a saved diff, and finally anything git
    understands as a revision range.
    """
    match = _PR_URL.search(target) or _PR_SHORT.match(target)
    if match:
        repository, number = match.group(1), int(match.group(2))
        token = settings().github_token
        if not token:
            raise ValueError(
                "GITHUB_TOKEN is not set, and it is needed to fetch a pull "
                "request. Export one, or pass a saved .diff file instead.")
        pr = PullRequest(repository=repository, number=number)
        return pull_request_diff(pr, token), f"{repository}#{number}"

    path = Path(target)
    if path.is_file():
        return path.read_text(encoding="utf-8", errors="replace"), str(path)

    return local_diff(target), target


def render(critiques: list[Critique]) -> str:
    """The findings, formatted for a terminal rather than for GitHub."""
    if not critiques:
        return _paint("No problems found.", _BOLD)

    order = {"high": 0, "medium": 1, "low": 2}
    ordered = sorted(critiques, key=lambda c: (order.get(c.severity, 3), c.file))

    lines = []
    for critique in ordered:
        where = critique.file
        if critique.line is not None:
            where += f":{critique.line}"
        severity = critique.severity.upper().ljust(6)
        lines.append(
            f"{_paint(severity, _BOLD, _COLOURS.get(critique.severity, ''))}  "
            f"{_paint(where, _BOLD)}")
        lines.append(f"        {critique.issue}")
        if critique.detail:
            lines.append(_paint(f"        {critique.detail}", _DIM))
        lines.append("")
    return "\n".join(lines).rstrip()


async def review(target: str, cli: str, repo_path: str) -> int:
    started = time.monotonic()

    try:
        diff, described = resolve(target)
    except (ValueError, GitHubError) as exc:
        logger.error(f"Could not work out what to review: {exc}")
        return EXIT_MISCONFIGURED

    if not diff.strip():
        print(f"\n  {_paint('Nothing to review', _BOLD)} — {described} has no changes.\n")
        return EXIT_OK

    files = diff_stats(diff)
    print()
    print(f"  {_paint('Reviewing', _BOLD)} {described}")
    print(f"  {len(files)} file(s), {len(diff)} characters: {format_stats(files, limit=8)}")
    print(f"  {_DIM if sys.stdout.isatty() else ''}model {settings().llm_model_name} "
          f"via {cli}{_RESET if sys.stdout.isatty() else ''}")
    print()

    try:
        graph = Agent(cli=cli, repo_path=repo_path).create_graph(gated=False)
        result = await graph.ainvoke({"messages": [], "diff": diff})
    except Exception:
        logger.exception("The review failed to run")
        return EXIT_FAILED

    critiques = result.get("critiques") or []
    elapsed = time.monotonic() - started

    print()
    print(render(critiques))
    print()
    count = len(critiques)
    print(f"  {_paint(f'{count} finding(s)', _BOLD)} in {elapsed:.0f}s")
    print()
    return EXIT_OK


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        prog="review",
        description="Review a pull request, a revision range, or a saved diff.",
        epilog=(
            "examples:\n"
            "  review.py https://github.com/owner/repo/pull/3\n"
            "  review.py owner/repo#3\n"
            "  review.py main..HEAD\n"
            "  review.py saved.diff"),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "target",
        help="a pull request URL, owner/repo#n, a git revision range, or a .diff file")
    parser.add_argument("--cli", default=None, help="which agent CLI to drive")
    parser.add_argument(
        "--repo", default=None,
        help="the checkout the sub-agent explores (default: the current directory)")
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="hide progress logging and print only the findings")
    return parser.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    configure(level="WARNING" if args.quiet else "INFO")
    s = settings()
    return asyncio.run(review(
        target=args.target,
        cli=args.cli or s.review_cli,
        repo_path=args.repo or s.repo_path,
    ))


if __name__ == "__main__":
    sys.exit(main())
