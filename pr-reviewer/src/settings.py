from log import get_logger
from utilities import parse_int

import os
import re
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()
logger = get_logger()

_PULL_REF = re.compile(r"^refs/pull/(\d+)/")

@dataclass(frozen=True)
class Settings:
    # git
    branch: str | None = None
    target_branch: str | None = None
    
    # model
    anthropic_api_key: str | None = None
    bobshell_api_key: str | None = None
    codex_api_key: str | None = None
    openai_api_key: str | None = None
    llm_model_name: str | None = None

    # github
    github_token: str | None = None
    github_repository: str | None = None  # "owner/repo"
    pr_number: int | None = None

    # general config
    max_reviews_per_pr: int | None = 1


def _pr_number() -> int | None:
    """The PR under review: `PR_NUMBER` if set, else the number in `GITHUB_REF`.

    Actions sets `GITHUB_REF` to `refs/pull/<n>/merge` on a pull_request event,
    so nothing has to be wired up by hand in the common case.
    """
    explicit = parse_int(os.getenv("PR_NUMBER") or None)
    if explicit is not None:
        return explicit

    match = _PULL_REF.match(os.getenv("GITHUB_REF") or "")
    return int(match.group(1)) if match else None


def settings() -> Settings:
    """
    Read the environment now.

    `os.getenv(NAME, default)` throughout, and `or None` on the secrets: an
    empty string is a value, and a provider handed one reports a confusing auth
    error instead of the obvious "you have no key".
    """
    return Settings(
        # git
        branch=(os.getenv("CI_COMMIT_BRANCH") or None),
        target_branch=(os.getenv("CI_MERGE_REQUEST_TARGET_BRANCH_NAME") or None),
        # model
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        bobshell_api_key=os.getenv("BOBSHELL_API_KEY") or None,
        codex_api_key=os.getenv("OPENAI_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-5.4"),
        # github
        github_token=os.getenv("GITHUB_TOKEN") or None,
        github_repository=os.getenv("GITHUB_REPOSITORY") or None,
        pr_number=_pr_number(),
        # general config
        max_reviews_per_pr=parse_int(os.getenv("MAX_REVIEWS_PER_PR"), 1),
    )
    
    
    

    