from github_client import GitHubError, PullRequest, count_agent_reviews
from llm import get_model
from log import get_logger
from models import Critique
from settings import settings
from nodes.critique import CRITIQUE, create_critique_node
from nodes.diff_analyzer import DIFF_ANALYZER, create_diff_analyzer_node

from typing import Annotated, List, TypedDict

from operator import add
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph

logger = get_logger(__name__)


# Why the graph declined to run. `main` turns these into exit codes, so a run
# that was skipped because of broken configuration can be told apart from one
# that legitimately had nothing to do — from the outside they look identical.
SKIPPED_MISCONFIGURED = "misconfigured"
SKIPPED_ALREADY_REVIEWED = "already_reviewed"

SHOULD_REVIEW = "should_review"


class State(TypedDict):
    messages: Annotated[List, add_messages]

    diff_summary: str

    critiques: Annotated[list[Critique], add]

    # Set only when the graph stops before reviewing; None on a normal run.
    skipped: str | None

    # seeded at entry-point
    branch: str
    target_branch: str


class Agent:

    def __init__(self, cli: str, repo_path: str):
        self.cli = cli
        # The sub-agent that explores the codebase needs an explicit directory;
        # ShellChatModel.cwd is None by default.
        self.repo_path = repo_path
        self.llm = get_model(cli=cli)

    @staticmethod
    def _skip(reason: str) -> dict:
        """Record why the run stopped. Routing reads this back."""
        logger.info("Exiting Agent...")
        return {"skipped": reason}

    def should_review(self, state: State) -> dict:
        """Decide whether to review, recording why not when the answer is no.

        A node rather than a conditional edge: mutating the state a routing
        function receives does not propagate to the graph's result, so the reason
        has to be *returned* as a state update and routed on afterwards.
        """
        s = settings()
        branch = state.get("branch")
        target_branch = state.get("target_branch")
        has_commits = bool(s.base_sha and s.head_sha)
        if not has_commits and (not branch or not target_branch):
            logger.error(
                "Nothing to diff: need BASE_SHA/HEAD_SHA, or both branch names. "
                f"branch: '{branch}', target_branch: '{target_branch}'")
            return self._skip(SKIPPED_MISCONFIGURED)

        if s.max_reviews_per_pr is None:
            return {"skipped": None}

        if not s.github_token or not s.github_repository or s.pr_number is None:
            logger.error(
                "Cannot check for existing reviews. "
                f"repository: '{s.github_repository}', pr_number: {s.pr_number}, "
                f"token: {'set' if s.github_token else 'missing'}"
            )
            return self._skip(SKIPPED_MISCONFIGURED)

        pr = PullRequest(repository=s.github_repository, number=s.pr_number)
        try:
            reviewed = count_agent_reviews(pr, token=s.github_token)
        except GitHubError:
            # Better to stay quiet than to risk piling duplicate reviews onto a
            # PR because we could not read what we had already posted.
            logger.exception("Could not count existing reviews")
            return self._skip(SKIPPED_MISCONFIGURED)

        if reviewed >= s.max_reviews_per_pr:
            logger.info(
                f"Already reviewed {pr.repository}#{pr.number} {reviewed} time(s), "
                f"max_reviews_per_pr is {s.max_reviews_per_pr}"
            )
            return self._skip(SKIPPED_ALREADY_REVIEWED)

        return {"skipped": None}

    @staticmethod
    def route(state: State) -> str:
        """Continue to the review, or stop, based on what `should_review` recorded."""
        return END if state.get("skipped") else DIFF_ANALYZER

    def create_graph(self) -> CompiledStateGraph:

        workflow: StateGraph = StateGraph(state_schema=State)

        workflow.add_node(SHOULD_REVIEW, self.should_review)
        workflow.add_node(DIFF_ANALYZER, create_diff_analyzer_node(llm=self.llm))
        workflow.add_node(
            CRITIQUE,
            create_critique_node(llm=self.llm, cli=self.cli, repo_path=self.repo_path),
        )

        workflow.add_edge(START, SHOULD_REVIEW)
        # The destinations are listed explicitly: without them the graph has no
        # static edges to draw or validate, since `route` is opaque to LangGraph.
        workflow.add_conditional_edges(
            source=SHOULD_REVIEW,
            path=self.route,
            path_map=[DIFF_ANALYZER, END],
        )
        workflow.add_edge(DIFF_ANALYZER, CRITIQUE)
        workflow.add_edge(CRITIQUE, END)

        return workflow.compile(checkpointer=InMemorySaver())