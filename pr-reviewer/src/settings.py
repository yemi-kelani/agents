from log import get_logger
from utilities import parse_int

import os
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()
logger = get_logger()

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
    
    # general config
    max_reviews_per_pr: int | None = 1
    

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
        # general config
        max_reviews_per_pr=parse_int(os.getenv("MAX_REVIEWS_PER_PR"), 1),
    )
    
    
    

    