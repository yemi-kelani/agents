from github_client import GitHubError, PullRequest, count_agent_reviews
from llm import get_model
from log import get_logger
from settings import settings
from nodes.diff_analyzer import DIFF_ANALYZER, create_diff_analyzer_node

from typing import Annotated, List, TypedDict

from operator import add
from pydantic import BaseModel
from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph

logger = get_logger()

class Critique(BaseModel):
    pass

class State(TypedDict):
    messages: Annotated[List, add_messages]
    
    diff_summary: str
    
    critiques: Annotated[list[Critique], add]
    
    # seeded at entry-point
    branch: str
    target_branch: str

class Agent:
    
    def __init__(self, cli: str):
        self.llm = get_model(cli=cli)
        
    def should_review(self, state: State):
        
        s = settings()
        branch = state.get("branch")
        target_branch = state.get("target_branch")
        if not branch or not target_branch:
            logger.error(f"Missing branch or target branch. branch: '{branch}', target_branch: '{target_branch}'")
            logger.info(f"Exiting Agent...")
            return END

        if s.max_reviews_per_pr is None:
            return DIFF_ANALYZER

        if not s.github_token or not s.github_repository or s.pr_number is None:
            logger.error(
                "Cannot check for existing reviews. "
                f"repository: '{s.github_repository}', pr_number: {s.pr_number}, "
                f"token: {'set' if s.github_token else 'missing'}"
            )
            logger.info(f"Exiting Agent...")
            return END

        pr = PullRequest(repository=s.github_repository, number=s.pr_number)
        try:
            reviewed = count_agent_reviews(pr)
        except GitHubError:
            # Better to stay quiet than to risk piling duplicate reviews onto a
            # PR because we could not read what we had already posted.
            logger.exception("Could not count existing reviews")
            logger.info(f"Exiting Agent...")
            return END

        if reviewed >= s.max_reviews_per_pr:
            logger.info(
                f"Already reviewed {pr.repository}#{pr.number} {reviewed} time(s), "
                f"max_reviews_per_pr is {s.max_reviews_per_pr}"
            )
            logger.info(f"Exiting Agent...")
            return END

        return DIFF_ANALYZER

    def create_graph(self) -> CompiledStateGraph:
        
        workflow: StateGraph = StateGraph(state_schema=State)
        workflow.add_conditional_edges(source=START, path=self.should_review)
        
        workflow.add_node(DIFF_ANALYZER, create_diff_analyzer_node(llm=self.llm))
        
        
        return workflow.compile(checkpointer=InMemorySaver())