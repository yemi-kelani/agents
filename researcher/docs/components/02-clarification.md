# 02 — Clarification Tool Loop

**Priority: P1.** Cheap, and it prevents the most expensive failure mode —
spending the entire research budget on a misread question. Build after the core
pipeline works.

## Purpose

Decide whether the question is answerable as stated. If it is, proceed. If it
isn't, ask — and if the answer is still ambiguous, ask once more.

"Compare observability vendors" needs to know: which vendors, for what workload,
and what decision the answer feeds. Guessing wastes the entire research budget on
the wrong axis. And *"the open-source ones"* is an answer that still needs a
follow-up, which is why this is a loop and not a single round.

## Shape

One node, one tool. Calling `ask_user` pauses the graph on an `interrupt()`.
*Not* calling it is the fall-through to planning — the model saying the question
was answerable as stated.

```python
# nodes/clarify.py
MAX_QUESTIONS = 3
MAX_ROUNDS = 2


def _tools(pairs: list[dict], trace: list[dict], box: dict):
    """Closed over this run's accumulators — the answers belong to one node
    execution, not to the module."""

    @tool
    def ask_user(questions: list[str], assumption_if_skipped: str = "") -> str:
        """Ask the human up to 3 follow-up questions to scope the research."""
        asked = questions[:MAX_QUESTIONS]
        box["assumption"] = assumption_if_skipped
        trace.append(ev("clarify_asked", n=len(asked)))

        answers = interrupt({                       # <- pauses here
            "questions": asked,
            "assumption_if_skipped": assumption_if_skipped,
        })

        if not answers:                             # "proceed anyway"
            trace.append(ev("clarify_skipped", assumption=assumption_if_skipped))
            return "The human skipped. Proceed on your stated assumption."

        if isinstance(answers, str):
            answers = [answers]

        # zip_longest, not zip: a UI returning two answers for three questions
        # would otherwise drop the third from the record, and `clarifications`
        # is what tells the planner the scope was narrowed.
        pairs.extend(
            {"q": q, "a": a}
            for q, a in zip_longest(asked, list(answers)[:len(asked)], fillvalue="")
        )
        trace.append(ev("clarify_answered", n=len(asked)))
        return "\n".join(f"{p['q']} -> {p['a']}" for p in pairs[-len(asked):])

    return [ask_user]


def clarify_node(state: ResearchState, *, llm) -> dict:
    if state.get("clarified"):
        return {}

    pairs, trace, box = [], [], {"assumption": ""}
    tools = _tools(pairs, trace, box)
    by_name = {t.name: t for t in tools}
    chat = llm.bind_tools(tools)

    # A local variable, never state — see "The messages channel" below.
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

    return {"clarifications": pairs, "clarified": True,
            "_assumption": box["assumption"], "trace": trace}
```

Wiring:

```python
builder.add_node("clarify", clarify_node)
builder.add_edge(START, "clarify")
builder.add_edge("clarify", "plan")
```

Resuming from the caller is unchanged:

```python
config = {"configurable": {"thread_id": run_id}}

for chunk in graph.stream({"question": q}, config, stream_mode="updates"):
    ...                                    # runs until the interrupt

# graph is now paused; surface the questions, collect answers, then:
graph.invoke(Command(resume=["Grafana and Datadog", "cost at 10k series"]), config)
```

## The re-execution gotcha

**On resume, LangGraph re-executes the node from the top.** The `interrupt()`
inside the tool then resolves from the resume map instead of pausing again — but
everything before it runs a second time, including the model turn that produced
the tool call.

The earlier two-node design (`assess` stores the questions, `ask` does nothing
but interrupt) existed precisely to prevent that, and a tool loop cannot keep it:
the decision to ask *is* the tool call. So the duplicate model call is accepted
here rather than designed away.

What it does **not** break: question/answer pairing. Questions and answers meet
inside a single tool call — the tool's own `questions` argument against its own
resume value — so they cannot drift apart across a re-run. The residual risk is a
non-deterministic model re-deriving *differently worded* questions on the re-run,
so stored answers land against text the human never saw. Low, but real, and worth
knowing before raising `MAX_ROUNDS`.

## The messages channel

The loop's message list is a local variable and must stay one. `messages` in
`ResearchState` is the human-readable projection of `trace` that the interface
reads (12), and `with_progress` writes into it from every node. Sharing the
channel would put node narration into the model's context and the model's
scratchpad onto the screen.

## Requirements

- **A checkpointer is mandatory.** `interrupt()` without one is an error.
  `InMemorySaver` is fine for the demo but loses interrupted threads on restart —
  note that in the README rather than pretending it's durable.
- **Cap at `MAX_ROUNDS`.** Two: one question, one follow-up conditioned on the
  answer. A third is a model that will not commit, and an agent that keeps asking
  is worse than one that proceeds on a stated assumption. `clarified` latches on
  the way out, so a later pass over the thread neither calls the model nor
  interrupts.
- **Cap questions structurally.** `MAX_QUESTIONS` truncates inside the tool. Tool
  arguments are not validated the way a Pydantic response schema was, so the
  fourth question is dropped in the tool body rather than asked against in the
  prompt.
- **Always offer the skip path.** `assumption_if_skipped` rides in the interrupt
  payload so the interface can offer "proceed anyway"; resuming with `[]` takes
  it. The run is never blocked on a human who isn't there. The prompt asks for
  the assumption even when the model is asking, so the option always exists.
- **A headless caller passes `clarified: True`.** That is how the eval harness
  (11) drives the graph with nobody attached: the node returns immediately, with
  no model call and no interrupt. No separate "disable clarification" flag is
  needed.

## Relationship to spec 07

Verification is structural and clarification is not, and that asymmetry is
deliberate.

A claim reaching the report unverified is a correctness failure the system exists
to prevent, so `verify` is a node on the only edge out of research, with no
opt-out. A question that goes unasked costs a research budget — expensive, but
recoverable, and visible in the report's own hedging. So clarification is an
affordance the model may decline, bounded by `MAX_ROUNDS` and the `clarified`
latch rather than by structure.

This section previously argued the opposite, and the earlier reasoning was sound:
"ask the user if unsure" is advisory, and a model skips it exactly when it is
most confidently wrong. That cost was accepted in exchange for adaptive
follow-ups and for the pipeline having a demonstrable tool-calling surface. **Do
not read this section as license to make verification a tool.**

## Acceptance

- A vague question pauses with ≤3 questions.
- A specific question makes no tool call and falls through to planning.
- Resuming with `Command(resume=[...])` records the pairs and continues.
- Resuming with `[]` proceeds on the assumption and records no clarifications.
- Fewer answers than questions still records every question.
- A single answer string, not a list, is accepted.
- The loop stops after `MAX_ROUNDS`, whatever the model does.
- A pre-clarified input calls no model and never interrupts.
- A hallucinated tool name does not crash the node.
- Asking and answering both reach `trace`.

**No longer accepted:** *"no duplicated LLM call on resume."* One node means the
model turn re-runs when the graph resumes; the two-node split that prevented it
is gone. See "The re-execution gotcha" above for what that costs and what it
does not.
