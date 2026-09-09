# 09 — Sufficiency Check & Iterative Deepening

**Priority: P2 — designed, not built.** The spec exists so the design is settled
when it gets built; deferring it is a scope decision, not an open question.

## Purpose

After a round, decide whether the accumulated evidence answers the brief. If not,
spawn new topics targeting specific gaps and run another round — bounded.

## The check is structured, not a boolean

A yes/no sufficiency check is useless twice over: it gives you nothing to act on,
and models say "yes, sufficient" almost every time when the alternative is more
work. Make it return gaps, and make every proposed topic justify itself.

```python
class Gap(BaseModel):
    topic_id: str | None                      # None = the brief itself has a hole
    missing: str = Field(description="What specifically is not yet established.")
    evidence: str = Field(description="Which uncertainty or absent claim shows this.")


class NewTopic(BaseModel):
    question: str
    justifying_gap: str = Field(description="Must restate the gap this closes.")


class Sufficiency(BaseModel):
    covered: list[str]                        # topic ids considered answered
    gaps: list[Gap] = Field(max_length=4)
    new_topics: list[NewTopic] = Field(default_factory=list, max_length=3)
    confidence: float = Field(ge=0, le=1)
    should_continue: bool
```

**The rule that keeps this from running forever:** every `new_topic` must cite a
`justifying_gap` that traces to a logged uncertainty or an unanswered part of the
brief. Without it, the model generates plausible-adjacent topics indefinitely —
each one reasonable, the set unbounded.

```python
SUFFICIENCY_PROMPT = """Assess whether the research so far answers the brief.

BRIEF: {brief}
TOPICS RESEARCHED: {researched}
VERIFIED CLAIMS ({n_supported}): {claims_digest}
REJECTED CLAIMS ({n_rejected}): {rejected_digest}
LOGGED UNCERTAINTIES: {uncertainties}

Identify gaps — parts of the brief not yet established by verified evidence.
Propose a new topic ONLY where a specific gap justifies it; restate that gap.

Do not propose topics that are merely interesting. Do not propose a topic already
researched. If the brief is covered, say so and stop.

Note: a topic whose claims were mostly REJECTED is a gap, not a covered topic —
the agent looked and found nothing solid."""
```

That last paragraph matters. A topic that produced five claims, all rejected, has
*not* been answered — but a naive check sees "topic researched" and moves on. The
rejection signal is what makes this loop smarter than one that only counts
coverage.

## Budget is authoritative

```python
def route_after_sufficiency(state) -> str:
    s = state["_sufficiency"]
    b = state["budget"]

    if b.exhausted() or state["round"] >= b.max_rounds:
        return "synthesize"                         # budget wins, always
    if len(state["topics"]) >= b.max_topics:
        return "synthesize"
    if not s.should_continue or not s.new_topics:
        return "synthesize"
    return "dispatch"                               # another round
```

The LLM *proposes*; the budget *decides*. Any design where a model can vote itself
more iterations has no upper bound on cost or latency.

## Uncertainty-driven expansion

This is the substantive difference from a naive loop. Most iterative research
agents ask the model to introspect on its own report — "is this good enough?" — which is
exactly the question models are worst at. Here, round-2 topics are driven by
**signals the research flows emitted during round 1**:

- `kind="unconfirmed"` — a source asserted something no other source corroborated.
- `kind="source_conflict"` — two sources disagreed. Strongest signal; almost always
  worth a targeted follow-up.
- `kind="no_results"` — search found nothing. Suggests the topic was badly phrased,
  so the follow-up should rephrase rather than repeat.

Plus the rejection signal from verification. Together these are *external* evidence
about what's missing, rather than the model's opinion of its own work.

## Wiring

```python
b.add_node("sufficiency", sufficiency_node)
b.add_edge("verify", "sufficiency")
b.add_conditional_edges("sufficiency", route_after_sufficiency,
                        {"dispatch": "dispatch", "synthesize": "synthesize"})
```

`dispatch` (spec 03) already filters to `status == "pending"`, so newly appended
topics fan out and previously-researched ones don't re-run.

## Why it's deferred

1. **It doubles worst-case latency** and introduces a loop whose failure mode
   (runaway topic expansion) is worse than the problem it solves. The bounded
   single-round path is predictable.
2. **It is orthogonal to what the eval measures.** Iterative deepening improves
   coverage; the harness in spec 11 measures grounding. Adding it changes the
   report without changing the number.
3. **The uncertainty and rejection signals it depends on are cheap to collect
   now** (components 06 and 07 already emit them), so deferring the consumer costs
   nothing later.

## Acceptance (when built)

- Round count never exceeds `max_rounds` regardless of model output.
- Every new topic traces to a logged gap.
- A topic with all-rejected claims is treated as a gap, not as covered.
- Researched topics never re-run.
