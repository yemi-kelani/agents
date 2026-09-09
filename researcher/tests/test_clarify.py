"""Spec 02 acceptance: clarification as a tool loop.

Driven through real compiled graphs, because the two behaviours that matter —
pausing inside a tool and re-executing the node on resume — only exist at
runtime.
"""
from __future__ import annotations

import logging
from functools import partial

from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import START, END, StateGraph
from langgraph.types import Command

from nodes.clarify import MAX_ROUNDS, clarify_node
from state import ResearchState


class FakeToolLLM:
    """Scripts the loop. Each turn is a list of (name, args) tool calls, or a
    string, which ends the loop. `calls` records every turn so a duplicated
    model call on resume is visible."""

    def __init__(self, *turns):
        self._turns = list(turns)
        self._i = 0
        self.calls: list[list] = []

    def bind_tools(self, tools):
        self.tools = tools
        return self

    def invoke(self, msgs):
        self.calls.append(list(msgs))
        # Indexed, not popped: the node re-executes from the top on resume, and
        # a queue that drained would hand the re-run a different script than the
        # first pass — hiding the very re-execution these tests assert on.
        turn = self._turns[self._i] if self._i < len(self._turns) else "done"
        self._i += 1
        if isinstance(turn, str):
            return AIMessage(content=turn)
        return AIMessage(content="", tool_calls=[
            {"name": n, "args": a, "id": f"call{i}"} for i, (n, a) in enumerate(turn)
        ])

    def reset(self):
        """Replay from the top, which is what the node does on resume."""
        self._i = 0


def asks(*questions, assumption="Compare Grafana and Datadog on cost."):
    return FakeToolLLM(
        [("ask_user", {"questions": list(questions),
                       "assumption_if_skipped": assumption})],
        "Scope is clear now.",
    )


def build(llm):
    b = StateGraph(ResearchState)
    b.add_node("clarify", partial(clarify_node, llm=llm))
    b.add_edge(START, "clarify")
    b.add_edge("clarify", END)
    # interrupt() without a checkpointer is an error.
    return b.compile(checkpointer=InMemorySaver())


def cfg(name: str) -> dict:
    return {"configurable": {"thread_id": name}}


def resume(graph, value, config):
    """Resume, replaying the model script the way a re-executed node sees it."""
    return graph.invoke(Command(resume=value), config)


# --- asking -----------------------------------------------------------------

def test_a_vague_question_pauses_with_the_questions_the_tool_was_given():
    llm = asks("Which vendors?", "What workload?")
    graph = build(llm)

    out = graph.invoke({"question": "compare observability vendors"}, cfg("t1"))

    payload = out["__interrupt__"][0].value
    assert payload["questions"] == ["Which vendors?", "What workload?"]


def test_the_interrupt_payload_offers_the_skip_path():
    """`assumption_if_skipped` rides in the payload so the interface can offer
    "proceed anyway". A run is never blocked on a human who isn't there."""
    llm = asks("Which vendors?", assumption="Compare Grafana and Datadog on cost.")
    graph = build(llm)

    out = graph.invoke({"question": "compare vendors"}, cfg("t2"))

    assert out["__interrupt__"][0].value["assumption_if_skipped"] == (
        "Compare Grafana and Datadog on cost.")


def test_a_specific_question_never_interrupts_and_falls_through():
    """No tool call is the fall-through: the question was answerable as stated."""
    llm = FakeToolLLM("PromQL's rate() is specific enough to research directly.")
    graph = build(llm)

    out = graph.invoke({"question": "what is PromQL's rate() for"}, cfg("t3"))

    assert "__interrupt__" not in out
    assert out["clarified"] is True
    assert out.get("clarifications", []) == []


# --- resuming ---------------------------------------------------------------

def test_resuming_records_the_pairs_and_continues():
    llm = asks("Which vendors?", "What workload?")
    graph = build(llm)
    graph.invoke({"question": "compare vendors"}, cfg("t4"))
    llm.reset()

    out = resume(graph, ["Grafana and Datadog", "cost at 10k series"], cfg("t4"))

    assert out["clarifications"] == [
        {"q": "Which vendors?", "a": "Grafana and Datadog"},
        {"q": "What workload?", "a": "cost at 10k series"},
    ]
    assert out["clarified"] is True


def test_fewer_answers_than_questions_still_records_every_question():
    """zip_longest, not zip: a UI returning two answers for three questions must
    not drop the third from the record — `clarifications` is what tells the
    planner the scope was narrowed."""
    llm = asks("Which vendors?", "What workload?", "What decision?")
    graph = build(llm)
    graph.invoke({"question": "compare vendors"}, cfg("t5"))
    llm.reset()

    out = resume(graph, ["Grafana and Datadog"], cfg("t5"))

    assert [c["q"] for c in out["clarifications"]] == [
        "Which vendors?", "What workload?", "What decision?"]
    assert [c["a"] for c in out["clarifications"]] == ["Grafana and Datadog", "", ""]


def test_resuming_with_nothing_proceeds_on_the_assumption():
    llm = asks("Which vendors?", assumption="Compare Grafana and Datadog on cost.")
    graph = build(llm)
    graph.invoke({"question": "compare vendors"}, cfg("t6"))
    llm.reset()

    out = resume(graph, [], cfg("t6"))

    assert out["clarified"] is True
    assert out.get("clarifications", []) == []
    assert out["_assumption"] == "Compare Grafana and Datadog on cost."
    assert [e for e in out["trace"] if e["kind"] == "clarify_skipped"]


def test_a_single_answer_string_is_accepted():
    """A UI resuming one question with a bare string, not a list."""
    llm = asks("Which vendors?")
    graph = build(llm)
    graph.invoke({"question": "compare vendors"}, cfg("t7"))
    llm.reset()

    out = resume(graph, "Grafana and Datadog", cfg("t7"))

    assert out["clarifications"] == [{"q": "Which vendors?", "a": "Grafana and Datadog"}]


# --- bounds -----------------------------------------------------------------

def test_the_loop_is_capped_at_max_rounds():
    """A model that asks forever is worse than one that proceeds on a stated
    assumption. The cap is structural, not a prompt instruction."""
    llm = FakeToolLLM(*[
        [("ask_user", {"questions": [f"q{i}?"], "assumption_if_skipped": "a"})]
        for i in range(MAX_ROUNDS + 3)
    ])
    graph = build(llm)
    config = cfg("t8")

    graph.invoke({"question": "compare vendors"}, config)
    for _ in range(MAX_ROUNDS + 2):
        llm.reset()
        out = resume(graph, ["an answer"], config)
        if "__interrupt__" not in out:
            break

    assert "__interrupt__" not in out, "the loop must stop asking"
    assert out["clarified"] is True


def test_a_preclarified_input_skips_the_model_and_never_interrupts():
    """How the eval harness (11) drives the graph with nobody attached. No
    separate "disable clarification" flag is needed."""
    llm = asks("Which vendors?")
    graph = build(llm)

    out = graph.invoke({"question": "compare vendors", "clarified": True}, cfg("t9"))

    assert llm.calls == [], "the latch must cap the model call, not just the pause"
    assert "__interrupt__" not in out


def test_the_question_cap_is_structural():
    """A fourth question is truncated before it reaches a human, not asked
    politely against in the prompt."""
    llm = asks("q1?", "q2?", "q3?", "q4?", "q5?")
    graph = build(llm)

    out = graph.invoke({"question": "compare vendors"}, cfg("t10"))

    assert len(out["__interrupt__"][0].value["questions"]) == 3


def test_a_hallucinated_tool_name_does_not_crash_the_node():
    llm = FakeToolLLM([("ask_the_human", {"questions": ["?"]})], "done")
    graph = build(llm)

    out = graph.invoke({"question": "compare vendors"}, cfg("t11"))

    assert out["clarified"] is True


def test_the_round_is_traced():
    """`with_progress` projects trace into the UI. Tool use nobody can see is
    tool use nobody can debug."""
    llm = asks("Which vendors?")
    graph = build(llm)
    graph.invoke({"question": "compare vendors"}, cfg("t12"))
    llm.reset()

    out = resume(graph, ["Grafana and Datadog"], cfg("t12"))

    assert [e for e in out["trace"] if e["kind"] == "clarify_asked"]
    assert [e for e in out["trace"] if e["kind"] == "clarify_answered"]


# --- the tool log -----------------------------------------------------------
#
# `ask_user` is the one tool a model actually chooses to call in this pipeline,
# so it is the one place where "what was the tool called with, and what did it
# answer?" is a question about the *model's* behaviour rather than ours.

def tool_logged(caplog):
    caplog.set_level(logging.DEBUG, logger="researcher.tools")
    return lambda: "\n".join(r.getMessage() for r in caplog.records)


def test_a_tool_call_logs_the_name_the_model_chose(caplog):
    read = tool_logged(caplog)
    graph = build(asks("Which vendors?"))

    graph.invoke({"question": "compare vendors"}, cfg("log1"))

    assert "ask_user" in read()


def test_a_tool_call_logs_the_arguments_the_model_passed(caplog):
    read = tool_logged(caplog)
    graph = build(asks("Which vendors?"))

    graph.invoke({"question": "compare vendors"}, cfg("log2"))

    assert "Which vendors?" in read()


def test_the_tool_result_handed_back_to_the_model_is_logged(caplog):
    """What the model *sees* after its tool call is the input to its next turn,
    and it is the one string nothing else on any surface records."""
    llm = asks("Which vendors?")
    graph = build(llm)
    graph.invoke({"question": "compare vendors"}, cfg("log3"))
    llm.reset()
    read = tool_logged(caplog)

    resume(graph, ["Grafana and Datadog"], cfg("log3"))

    assert "Grafana and Datadog" in read()


def test_a_hallucinated_tool_name_is_logged_rather_than_swallowed(caplog):
    """The node answers a bad name instead of raising, which is right — but an
    unrecorded hallucination is a model failure that leaves no evidence."""
    caplog.set_level(logging.WARNING, logger="researcher.tools")
    graph = build(FakeToolLLM([("ask_the_human", {"questions": ["?"]})], "done"))

    graph.invoke({"question": "compare vendors"}, cfg("log4"))

    logged = "\n".join(r.getMessage() for r in caplog.records)
    assert "ask_the_human" in logged
    assert any(r.levelno == logging.WARNING for r in caplog.records)


def test_pausing_to_ask_a_human_is_not_logged_as_a_tool_failure(caplog):
    """The pause is the feature. A warning per clarification would teach
    everyone to ignore the line that means something really broke."""
    caplog.set_level(logging.DEBUG, logger="researcher.tools")
    graph = build(asks("Which vendors?"))

    graph.invoke({"question": "compare vendors"}, cfg("log5"))

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]
