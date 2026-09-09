"""Planning and fan-out — spec 03.

Decomposition quality sets the ceiling on the run: topics that depend on each
other research a hole, because they run concurrently and cannot see each other's
answers. The constraint is stated in the prompt and bounded structurally here —
`Plan.topics` rejects a sixth topic before it becomes a sixth research flow.
"""
from __future__ import annotations

from typing import TypedDict

from langgraph.types import Send
from pydantic import BaseModel, Field

from prompts import load
from state import ResearchState, Topic, ev

MIN_TOPICS = 2
MAX_TOPICS = 5
RESEARCH_NODE = "research_topic"


class TopicSpec(BaseModel):
    """One topic as the model proposes it, before it gets an id."""
    question: str = Field(description="A specific, search-answerable sub-question.")
    rationale: str = Field(description="What gap in the brief this closes.")


class Plan(BaseModel):
    brief: str = Field(description="2-4 sentences: what a complete answer must cover.")
    topics: list[TopicSpec] = Field(min_length=MIN_TOPICS, max_length=MAX_TOPICS)


class TopicTask(TypedDict):
    """What one research flow receives — its topic and the brief that topic
    serves. A `Send` payload replaces the state a node sees, so anything the
    flow needs travels in here."""
    topic: Topic
    brief: str


def _scope(state: ResearchState) -> str:
    """What spec 02 settled about the question, in the planner's words.

    Answers win when they exist. Otherwise the skip-path assumption is stated
    outright, so a run nobody clarified is planned against a scope on the record
    rather than one the planner quietly invents.
    """
    clarifications = state.get("clarifications") or []
    if clarifications:
        pairs = "\n".join(f"- {c['q']} -> {c['a']}" for c in clarifications)
        return f"Clarifications:\n{pairs}"

    assumption = state.get("_assumption") or ""
    return f"Proceeding on this assumption: {assumption}" if assumption else ""


def plan_node(state: ResearchState, *, llm) -> dict:
    """Question (+ clarifications) -> brief + topics. Runs once, opening round 1;
    later rounds get their topics from the sufficiency check (spec 09)."""
    plan = llm.with_structured_output(Plan).invoke(
        load("plan").format(question=state["question"], context=_scope(state))
    )

    # The budget truncates the model's list rather than the dispatch alone, so
    # state never carries a topic this run will not research.
    topics = [
        Topic(id=f"t{i}", question=t.question, rationale=t.rationale)
        for i, t in enumerate(plan.topics[: state["budget"].max_topics])
    ]

    return {
        "brief": plan.brief,
        "topics": topics,
        "round": 1,
        "trace": [ev("plan", brief=plan.brief, topics=[t.question for t in topics])],
    }


def dispatch(state: ResearchState) -> list[Send]:
    """One concurrent research flow per pending topic.

    `Send` is what makes the fan-out width a runtime value: the topic list is not
    known when the graph is compiled. Filtering on `status` is also what keeps a
    second round from re-researching the first round's topics.
    """
    budget = state["budget"]
    if budget.exhausted():
        return []

    pending = [t for t in state["topics"] if t.status == "pending"]
    return [
        Send(RESEARCH_NODE, TopicTask(topic=t, brief=state["brief"]))
        for t in pending[: budget.max_topics]
    ]
