from github_client import PullRequest
from utilities import parse_int

import os
import re
from dotenv import load_dotenv
from dataclasses import dataclass

load_dotenv()

_PULL_REF = re.compile(r"^refs/pull/(\d+)/")

# `MAX_REVIEWS_PER_PR` accepts this to mean "review every time". Without a
# sentinel there is no way to reach the unlimited branch from the environment,
# since an unparseable value falls back to the default rather than to None.
_UNLIMITED = {"none", "unlimited", "-1"}


@dataclass(frozen=True)
class Settings:
    # git
    branch: str | None = None
    target_branch: str | None = None
    # The commits to diff. Preferred over the branch names, which cannot be
    # resolved for a pull request opened from a fork.
    base_sha: str | None = None
    head_sha: str | None = None

    # model
    anthropic_api_key: str | None = None
    bobshell_api_key: str | None = None
    # One field, read from OPENAI_API_KEY. The codex CLI wants it under the
    # name CODEX_API_KEY; that translation is ShellSpec.env_key's job, not a
    # second copy of the secret here.
    openai_api_key: str | None = None
    llm_model_name: str | None = None

    # github
    github_token: str | None = None
    github_repository: str | None = None  # "owner/repo"
    pr_number: int | None = None

    # general config
    max_reviews_per_pr: int | None = 1
    review_cli: str = "codex"
    # What the exploring sub-agent is pointed at: the checkout root on Actions,
    # the working directory for a local run.
    repo_path: str = "."

    def pull_request(self) -> PullRequest | None:
        """The PR to read and post to, or None when GitHub is not fully configured.

        Both the pre-review gate and the publish step need this exact triple, so
        the check lives here rather than being spelled out at each call site.
        """
        if not self.github_token or not self.github_repository or self.pr_number is None:
            return None
        return PullRequest(repository=self.github_repository, number=self.pr_number)

    def github_config_error(self) -> str:
        """Which part of the GitHub configuration is missing, for a log line."""
        return (
            f"repository: '{self.github_repository}', pr_number: {self.pr_number}, "
            f"token: {'set' if self.github_token else 'missing'}"
        )


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


def _max_reviews() -> int | None:
    """The review cap: a count, or None for unlimited."""
    raw = (os.getenv("MAX_REVIEWS_PER_PR") or "").strip()
    if raw.lower() in _UNLIMITED:
        return None
    return parse_int(raw or None, 1)


def settings() -> Settings:
    """
    Read the environment now.

    `os.getenv(NAME, default)` throughout, and `or None` on the secrets: an
    empty string is a value, and a provider handed one reports a confusing auth
    error instead of the obvious "you have no key".
    """
    return Settings(
        # git — GitHub's own variables first, the GitLab names as a fallback, the
        # same shape `_pr_number` uses for the PR number.
        branch=(os.getenv("GITHUB_HEAD_REF") or os.getenv("CI_COMMIT_BRANCH") or None),
        target_branch=(
            os.getenv("GITHUB_BASE_REF")
            or os.getenv("CI_MERGE_REQUEST_TARGET_BRANCH_NAME")
            or None
        ),
        base_sha=os.getenv("BASE_SHA") or None,
        head_sha=os.getenv("HEAD_SHA") or None,
        # model
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        bobshell_api_key=os.getenv("BOBSHELL_API_KEY") or None,
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        llm_model_name=os.getenv("LLM_MODEL_NAME", "gpt-5.4"),
        # github
        github_token=os.getenv("GITHUB_TOKEN") or None,
        github_repository=os.getenv("GITHUB_REPOSITORY") or None,
        pr_number=_pr_number(),
        # general config
        max_reviews_per_pr=_max_reviews(),
        review_cli=os.getenv("REVIEW_CLI") or "codex",
        # GITHUB_WORKSPACE is the checkout root on Actions; the cwd is the
        # fallback for local runs.
        repo_path=os.getenv("GITHUB_WORKSPACE") or os.getcwd(),
    )
