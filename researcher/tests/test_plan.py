"""Spec 03 acceptance: the brief, topic decomposition, and the fan-out.

Driven through real compiled graphs. Two of the four acceptance criteria —
concurrent fan-out and a fan-in that merges without `InvalidUpdateError` — are
runtime properties of `Send` plus the spec 01 reducers, and asserting on the
node's return value would prove neither.
"""
from __future__ import annotations

import time
from functools import partial

import pytest
from langgraph.graph import START, END, StateGraph
from pydantic import ValidationError

from nodes.plan import Plan, TopicSpec, dispatch, plan_node
from state import Budget, BudgetDelta, Claim, Quote, ResearchState, Source, Topic


class FakeLLM:
    """Records the prompt so the scope context reaching the model is testable."""

    def __init__(self, plan: Plan):
        self.plan = plan
        self.calls: list[str] = []

    def with_structured_output(self, schema):
        assert schema is Plan
        return self

    def invoke(self, prompt: str):
        self.calls.append(prompt)
        return self.plan


def a_plan(n: int = 3) -> Plan:
    return Plan(
        brief="A complete answer must cover cost, ergonomics and lock-in.",
        topics=[
            TopicSpec(question=f"sub-question {i}", rationale=f"gap {i}")
            for i in range(n)
        ],
    )


def build(llm: FakeLLM, research=None):
    """plan -> (Send fan-out) -> research_topic, the wiring from the spec."""
    b = StateGraph(ResearchState)
    b.add_node("plan", partial(plan_node, llm=llm))
    b.add_node("research_topic", research or (lambda task: {}))
    b.add_edge(START, "plan")
    b.add_conditional_edges("plan", dispatch, ["research_topic"])
    b.add_edge("research_topic", END)
    return b.compile()


# --- acceptance 1: a broad question produces 3-5 topics ---------------------

def test_plan_turns_a_question_into_a_brief_and_topics():
    llm = FakeLLM(a_plan(4))
    out = build(llm).invoke({"question": "compare observability vendors"})

    assert out["brief"] == "A complete answer must cover cost, ergonomics and lock-in."
    assert [t.question for t in out["topics"]] == [f"sub-question {i}" for i in range(4)]
    assert [t.rationale for t in out["topics"]] == [f"gap {i}" for i in range(4)]


def test_planned_topics_have_unique_ids_and_start_pending():
    out = build(FakeLLM(a_plan(3))).invoke({"question": "q"})

    assert len({t.id for t in out["topics"]}) == 3
    assert all(t.status == "pending" for t in out["topics"])


def test_plan_opens_round_one():
    out = build(FakeLLM(a_plan())).invoke({"question": "q"})

    assert out["round"] == 1


def test_plan_is_traced_with_the_brief_and_the_topic_questions():
    out = build(FakeLLM(a_plan(3))).invoke({"question": "q"})

    events = [e for e in out["trace"] if e["kind"] == "plan"]
    assert len(events) == 1
    assert events[0]["brief"].startswith("A complete answer")
    assert events[0]["topics"] == [f"sub-question {i}" for i in range(3)]


@pytest.mark.parametrize("n", [1, 6])
def test_the_topic_count_bounds_are_structural(n):
    """"3-5 topics" in a prompt is advisory; this rejects the eleventh topic
    before it becomes eleven concurrent research flows."""
    with pytest.raises(ValidationError):
        a_plan(n)


# --- acceptance 2: topics respect budget.max_topics -------------------------

def test_a_tight_budget_truncates_the_planned_topics():
    """The run profile caps the topic list, not just the dispatch: state must
    never carry more topics than the budget will ever research."""
    out = build(FakeLLM(a_plan(5))).invoke(
        {"question": "q", "budget": Budget(max_topics=2)}
    )

    assert len(out["topics"]) == 2


def test_dispatch_never_sends_more_than_max_topics():
    state = {
        "brief": "b",
        "budget": Budget(max_topics=2),
        "topics": [Topic(id=f"t{i}", question=f"q{i}", rationale="r") for i in range(4)],
    }

    assert len(dispatch(state)) == 2


def test_dispatch_stops_when_the_budget_is_exhausted():
    state = {
        "brief": "b",
        "budget": Budget(max_searches=2, searches_used=2),
        "topics": [Topic(id="t0", question="q0", rationale="r")],
    }

    assert dispatch(state) == []


def test_an_exhausted_budget_fans_out_to_nothing_without_erroring():
    """Dispatching no `Send` at all has to be a clean stop, not a graph error —
    it is the path an already-spent run takes."""
    ran: list[str] = []

    def research(task) -> dict:
        ran.append(task["topic"].id)
        return {}

    graph = build(FakeLLM(a_plan(3)), research=research)
    out = graph.invoke(
        {"question": "q", "budget": Budget(max_searches=2, searches_used=2)}
    )

    assert ran == []
    assert len(out["topics"]) == 3, "the plan still lands; only the fan-out stops"


def test_dispatch_skips_topics_already_researched():
    """Spec 09 leans on this: round 2 fans out over the new topics only."""
    state = {
        "brief": "b",
        "budget": Budget(),
        "topics": [
            Topic(id="t0", question="q0", rationale="r", status="researched"),
            Topic(id="t1", question="q1", rationale="r", status="pending"),
        ],
    }

    sends = dispatch(state)

    assert [s.arg["topic"].id for s in sends] == ["t1"]


def test_each_send_carries_one_topic_and_the_brief():
    """A research flow receives only its payload — not the full state — so the
    brief has to ride along or the topic is researched without its context."""
    state = {
        "brief": "the brief",
        "budget": Budget(),
        "topics": [Topic(id="t0", question="q0", rationale="r")],
    }

    (send,) = dispatch(state)

    assert send.node == "research_topic"
    assert send.arg["topic"].question == "q0"
    assert send.arg["brief"] == "the brief"


# --- acceptance 3: fan-out is concurrent ------------------------------------

def test_three_topics_take_about_as_long_as_one():
    """Wall-clock for 3 topics ~= 1 topic, not 3x. A `Send` per topic that ran
    sequentially would still pass every other test here."""
    delay = 0.2

    def slow_research(task) -> dict:
        time.sleep(delay)
        return {"topics": [task["topic"].model_copy(update={"status": "researched"})]}

    graph = build(FakeLLM(a_plan(3)), research=slow_research)

    started = time.perf_counter()
    graph.invoke({"question": "q"})
    elapsed = time.perf_counter() - started

    assert elapsed < delay * 2, f"fan-out was sequential: {elapsed:.2f}s for 3 topics"


# --- acceptance 4: fan-in merges without InvalidUpdateError -----------------

def test_parallel_research_flows_merge_back_into_state():
    def research(task) -> dict:
        topic = task["topic"]
        return {
            "topics": [topic.model_copy(update={"status": "researched"})],
            "sources": [
                Source(source_id=topic.id, url=f"http://x/{topic.id}",
                       title=topic.question, backend="test")
            ],
            "claims": [
                Claim(id=f"c-{topic.id}", topic_id=topic.id, text=topic.question,
                      quotes=[Quote(source_id=topic.id, text="v", start=0, end=1)])
            ],
            "budget": BudgetDelta(searches=1, fetches=2, tokens=100),
        }

    out = build(FakeLLM(a_plan(3)), research=research).invoke({"question": "q"})

    assert len(out["claims"]) == 3
    assert len(out["sources"]) == 3
    assert {t.status for t in out["topics"]} == {"researched"}
    assert (out["budget"].searches_used, out["budget"].fetches_used) == (3, 6)
    assert out["budget"].max_topics == Budget().max_topics, "spend must not clobber limits"


# --- the prompt: what the planner is told about scope -----------------------

def test_answered_clarifications_reach_the_prompt():
    llm = FakeLLM(a_plan())
    build(llm).invoke({
        "question": "compare observability vendors",
        "clarifications": [{"q": "Which vendors?", "a": "Grafana and Datadog"}],
    })

    assert "Which vendors? -> Grafana and Datadog" in llm.calls[0]


def test_the_skip_assumption_reaches_the_prompt_when_nobody_answered():
    """Spec 02's "proceed anyway" path stores what the run proceeds on. Without
    it here, the planner re-invents a scope the human already declined to give."""
    llm = FakeLLM(a_plan())
    build(llm).invoke({
        "question": "compare observability vendors",
        "_assumption": "Compare Grafana Cloud and Datadog on cost.",
    })

    assert "Compare Grafana Cloud and Datadog on cost." in llm.calls[0]


def test_the_question_reaches_the_prompt():
    llm = FakeLLM(a_plan())
    build(llm).invoke({"question": "compare observability vendors"})

    assert "compare observability vendors" in llm.calls[0]
