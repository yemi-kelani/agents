"""Second step of a review: judge the change, exploring the code as needed."""

from __future__ import annotations

from langchain_core.language_models import BaseChatModel

from log import get_logger
from models import Critique
from parsers import parse_critiques
from prompts import load
from tool_loop import run_tool_loop
from tools.explore import create_explore_tool

logger = get_logger()
CRITIQUE = "critique"


async def extract_critiques(llm: BaseChatModel, review: str) -> list[dict]:
    """Parse findings out of a review, re-asking the model if the format drifted.

    The critique prompt asks for FILE/LINE/SEVERITY/ISSUE/DETAIL blocks, so the
    common case is a plain parse. The model call is the fallback for when it
    answers in prose anyway — which it will, sometimes.
    """
    findings = parse_critiques(review)
    if findings:
        return findings

    if not review.strip():
        return []

    logger.info("Review did not parse; asking the model to reformat it")
    reformatted = await llm.ainvoke(load("critique_extraction").format(review=review))
    findings = parse_critiques(str(reformatted.content))
    if not findings:
        # Either a genuinely clean review or an unusable one. Both end here, but
        # the review text is logged so the difference is visible after the fact.
        logger.info(f"No findings parsed from review:\n{review}")
    return findings


def create_critique_node(llm: BaseChatModel, cli: str, repo_path: str):

    async def critique(state):
        diff_summary = state.get("diff_summary") or ""
        if not diff_summary.strip():
            logger.info("No diff summary to critique")
            return {"critiques": []}

        tools = {"explore_codebase": create_explore_tool(cli=cli, repo_path=repo_path)}
        review = await run_tool_loop(
            model=llm,
            tools=tools,
            task=load("critique").format(diff_summary=diff_summary),
        )

        findings = await extract_critiques(llm, review)
        logger.info(f"Review produced {len(findings)} critique(s)")
        return {"critiques": [Critique(**finding) for finding in findings]}

    return critique
