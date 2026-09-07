from llm import get_model
from log import get_logger
from settings import settings

from typing import Annotated, List, TypedDict

from langgraph.graph import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.state import StateGraph, START, END, CompiledStateGraph

logger = get_logger()

class State(TypedDict):
    messages: Annotated[List, add_messages]

class Agent:
    
    def __init__(self, cli: str):
        self.llm = get_model(cli=cli)
        
    def should_review(self, state: State):
        
        s = settings()
        if not s.branch or not s.target_branch:
            logger.error(f"Missing branch or target branch. branch: '{s.branch}', target_branch: '{s.target_branch}'")
            logger.error(f"Exiting Agent")
            return END
        
        
        
        
        
        
    def create_graph() -> CompiledStateGraph:
        
        workflow: StateGraph = StateGraph(state_schema=State)
        workflow.add_conditional_edges(source=START, path=self.should_review)
        
        
        
        return workflow.compile(checkpointer=InMemorySaver())