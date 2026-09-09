"""Clarification — spec 02.

One node, one tool. The model decides whether the question is answerable as
stated: calling `ask_user` pauses the graph on an `interrupt()`, and *not*
calling it is the fall-through to planning.

This is deliberately an affordance rather than the structural interrupt spec 02
originally specified, and the trade is on the record there. Two consequences
follow and are bounded here rather than in the prompt:

- **The model turn re-runs on resume.** LangGraph re-executes the node from the
  top; the `interrupt()` inside the tool then resolves from the resume map
  instead of pausing again, so the re-run's turn is computed and discarded.
  Questions and answers still pair correctly because they meet inside one tool
  call — the tool's own `questions` argument against its own resume value. A
  non-deterministic model re-deriving different wording on the re-run is the
  residual risk.
- **A model can ask forever.** `MAX_ROUNDS` stops it, and `clarified` latches so
  a later pass over the thread neither calls the model nor interrupts.
"""
from __future__ import annotations

from itertools import zip_longest

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.types import interrupt

from prompts import load
from state import ResearchState, ev

MAX_QUESTIONS = 3
"""The cap is structural. A prompt saying "no more than 3" is advisory; this
truncates a fourth question before it reaches a human."""

MAX_ROUNDS = 2
"""Rounds of questions, not tool calls. One follow-up conditioned on the first
answer is the capability this design buys; a third is a model that will not
commit, and an agent that keeps asking is worse than one that proceeds on a
stated assumption."""


def _tools(pairs: list[dict], trace: list[dict], box: dict):
    """The tool, closed over this run's accumulators. Built per call because the
    answers belong to one node execution, not to the module."""

    @tool
    def ask_user(questions: list[str], assumption_if_skipped: str = "") -> str:
        """Ask the human up to 3 follow-up questions to scope the research."""
        asked = questions[:MAX_QUESTIONS]
        box["assumption"] = assumption_if_skipped
        trace.append(ev("clarify_asked", n=len(asked)))

        # Nothing above this line has a side effect worth repeating: on resume
        # the node re-executes and this call returns the stored answers instead
        # of pausing again.
        answers = interrupt({
            "questions": asked,
            "assumption_if_skipped": assumption_if_skipped,
        })

        if not answers:
            # The skip path: resuming with nothing means "proceed anyway", and
            # `_assumption` carries what planning states rather than inventing.
            trace.append(ev("clarify_skipped", assumption=assumption_if_skipped))
            return "The human skipped. Proceed on your stated assumption."

        if isinstance(answers, str):
            answers = [answers]

        # zip_longest, not zip: a UI returning two answers for three questions
        # would otherwise drop the third from the record entirely, and
        # `clarifications` is what tells the planner the scope was narrowed.
        pairs.extend(
            {"q": q, "a": a}
            for q, a in zip_longest(asked, list(answers)[:len(asked)], fillvalue="")
        )
        trace.append(ev("clarify_answered", n=len(asked)))
        return "\n".join(f"{p['q']} -> {p['a']}" for p in pairs[-len(asked):])

    return [ask_user]


def clarify_node(state: ResearchState, *, llm) -> dict:
    """Decide whether the question is answerable as stated, asking if not.

    `clarified` latches, so a caller can pre-set it to skip the round entirely —
    which is how the eval harness (11) drives the graph with no human attached:
    no model call, no interrupt.
    """
    if state.get("clarified"):
        return {}

    pairs: list[dict] = []
    trace: list[dict] = []
    box: dict = {"assumption": ""}
    tools = _tools(pairs, trace, box)
    by_name = {t.name: t for t in tools}
    chat = llm.bind_tools(tools)

    # A local variable, never state: `messages` is the progress projection the
    # interface reads (12), and writing this conversation into it would put node
    # narration in the model's context and the model's scratchpad on the screen.
    msgs = [SystemMessage(load("clarify")), HumanMessage(state["question"])]

    for _ in range(MAX_ROUNDS):
        ai = chat.invoke(msgs)
        msgs.append(ai)

        if not ai.tool_calls:
            break                       # answerable as stated — fall through

        for tc in ai.tool_calls:
            impl = by_name.get(tc["name"])
            # A hallucinated tool name is answered, not raised: killing the run
            # over one bad call is worse than proceeding on the assumption.
            content = impl.invoke(tc["args"]) if impl else (
                f"No tool named {tc['name']}. Available: {', '.join(by_name)}.")
            msgs.append(ToolMessage(content=content, tool_call_id=tc["id"]))

    return {
        "clarifications": pairs,
        "clarified": True,
        "_assumption": box["assumption"],
        "trace": trace,
    }
