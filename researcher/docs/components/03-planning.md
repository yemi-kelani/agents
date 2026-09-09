# 03 — Planning & Topic Decomposition

**Priority: P0.**

## Purpose

Turn a question (plus any clarifications) into a **research brief** and a set of
3–5 **topics** — specific, independently researchable sub-questions. Then fan out
over them.

Decomposition quality sets the ceiling on the whole run. A flat single-query agent
converges on the first plausible answer; multi-perspective decomposition is the
main defense against that, and it is the mechanism STORM (Shao et al., NAACL 2024)
credits for coverage.

## Design constraints

**Topics must be independent.** They run concurrently, so a topic that depends on
another topic's answer will research a hole. If the question genuinely has
sequential structure ("what happened, then why did it matter"), that dependency is
what round 2 is for — the sufficiency check (spec 09) adds the follow-up topic
once the first is answered.

**Topics must be answerable by search.** "Is X a good idea" is not a topic;
"what tradeoffs do X's adopters report" is.

**3–5, hard-capped at `budget.max_topics`.** More topics is not more insight, it
is more tokens and a slower demo.

## Implementation

```python
# nodes/plan.py
from pydantic import BaseModel, Field


class Plan(BaseModel):
    brief: str = Field(description="2-4 sentences: what a complete answer must cover.")
    topics: list[TopicSpec] = Field(min_length=2, max_length=5)


class TopicSpec(BaseModel):
    question: str = Field(description="A specific, search-answerable sub-question.")
    rationale: str = Field(description="What gap in the brief this closes.")


PLAN_PROMPT = """Scope a research task.

Question: {question}
{clarifications}

Write:
1. A brief — 2-4 sentences stating what a complete answer must cover.
2. 3-5 topics. Each must be:
   - independently researchable (no topic may depend on another's answer)
   - answerable from public sources
   - non-overlapping

Prefer topics that approach the question from different angles over topics that
restate it. If the question is contested, include a topic covering the strongest
opposing case."""


def plan_node(state, *, llm):
    clar = "\n".join(f"- {c['q']} -> {c['a']}" for c in state.get("clarifications", []))
    plan = llm.with_structured_output(Plan).invoke(
        PLAN_PROMPT.format(
            question=state["question"],
            clarifications=f"Clarifications:\n{clar}" if clar else "",
        )
    )
    topics = [
        Topic(id=f"t{i}", question=t.question, rationale=t.rationale)
        for i, t in enumerate(plan.topics)
    ]
    return {
        "brief": plan.brief,
        "topics": topics,
        "round": 1,
        "trace": [ev("plan", brief=plan.brief, topics=[t.question for t in topics])],
    }
```

## Fan-out with `Send`

`Send` dispatches one instance of a node per item, concurrently. This is how you
map over a topic list whose length isn't known at compile time.

```python
from langgraph.types import Send


def dispatch(state) -> list[Send]:
    pending = [t for t in state["topics"] if t.status == "pending"]
    if state["budget"].exhausted():
        return []
    return [
        Send("research_topic", {"topic": t, "brief": state["brief"]})
        for t in pending[: state["budget"].max_topics]
    ]


builder.add_node("plan", plan_node)
builder.add_node("research_topic", research_topic_node)   # spec 04-06
builder.add_conditional_edges("plan", dispatch, ["research_topic"])
builder.add_edge("research_topic", "verify")
```

Each `research_topic` invocation receives only the payload in its `Send` — not the
full state. It returns partial updates that merge back through the reducers in
spec 01. That merge is why those reducers are mandatory.

## On "k topics in parallel" vs "1 topic with k agents"

Pick one. They solve different problems:

- **k topics in parallel** buys *breadth* — more of the question covered per round.
- **1 topic, k agents** buys *consensus* — redundant coverage of one thing, useful
  when you distrust a single pass.

Breadth is what a research report needs; consensus is what you want when the cost
of being wrong on a single fact is high. Doing both multiplies the failure surface
and the token bill for no additional coverage. **Build breadth.** Consensus is a
reasonable extension once the verification signal in spec 07 shows which topics
are unstable enough to warrant redundant passes.

## Acceptance

- A broad question produces 3–5 non-overlapping, independently-researchable topics.
- Topics respect `budget.max_topics`.
- Fan-out runs concurrently (wall-clock for 3 topics ≈ 1 topic, not 3×).
- Fan-in merges without `InvalidUpdateError`.
