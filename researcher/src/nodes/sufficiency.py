"""Sufficiency check and iterative deepening — spec 09.

After a round, decide whether the accumulated evidence answers the brief; if it
does not, spawn topics aimed at specific gaps and run another round. Bounded.

Two things make this different from asking a model "is this good enough?", which
is the question models are worst at:

**The check returns gaps, not a boolean.** A yes/no answer gives you nothing to
act on, and a model asked whether it should do more work says no. Gaps are
actionable and every proposed topic has to restate the gap it closes — enforced
here, not requested in the prompt, because "generate plausible-adjacent topics
indefinitely" is exactly what an unenforced version does.

**The gaps are driven by signals the research flows emitted**, not by the model's
opinion of its own report: uncertainties logged during extraction (06), and the
rejection verdicts from verification (07). A topic that produced five claims and
had all five rejected has *not* been answered, and this module computes that
rather than hoping the model notices it.

Two deviations from the spec's sketch, both to put a decision in one place:

- `can_deepen` is the single budget gate, consulted by this node *before* it
  spends an LLM call and implicitly by the route afterwards. The spec puts those
  checks in `route_after_sufficiency`; splitting them across a node and a router
  means `round` is either incremented too early to compare or too late to matter.
  The model still only ever proposes — nothing here can vote itself more rounds.
- The assessment does not land in state. Nothing downstream reads it, and a
  pydantic model in state is a serializer allowlist entry (01) bought for
  nothing. The decision it produced is in `trace`, which is what the UI (12) and
  the eval (11) read.
"""
from __future__ import annotations

from langgraph.types import Send
from pydantic import BaseModel, Field

from anchor import anchor
from nodes.plan import dispatch
from nodes.synthesize import SYNTHESIZE_NODE
from prompts import load
from state import Claim, ResearchState, Topic, Uncertainty, ev

SUFFICIENCY_NODE = "sufficiency"

MAX_GAPS = 4
MAX_NEW_TOPICS = 3


class Gap(BaseModel):
    topic_id: str | None = None               # None = the brief itself has a hole
    missing: str = Field(description="What specifically is not yet established.")
    evidence: str = Field(description="Which uncertainty or absent claim shows this.")


class NewTopic(BaseModel):
    question: str
    justifying_gap: str = Field(description="Must restate the gap this closes.")


class Sufficiency(BaseModel):
    covered: list[str]                        # topic ids considered answered
    gaps: list[Gap] = Field(default_factory=list, max_length=MAX_GAPS)
    new_topics: list[NewTopic] = Field(default_factory=list, max_length=MAX_NEW_TOPICS)
    confidence: float = Field(ge=0, le=1)
    should_continue: bool


def can_deepen(state: ResearchState) -> bool:
    """Has this run got another round in it? The budget's answer, and the only
    one that counts.

    Any design where a model can vote itself more iterations has no upper bound
    on cost or latency, so this is a pure function of the budget and what has
    already been spent — nothing the model returned reaches it.
    """
    budget = state["budget"]
    return (not budget.exhausted()
            and state.get("round", 1) < budget.max_rounds
            and len(state.get("topics") or []) < budget.max_topics)


def sufficiency_node(state: ResearchState, *, llm) -> dict:
    """Assess coverage and, if the budget allows it, open another round."""
    if not can_deepen(state):
        # Asked before the model is: an assessment that cannot be acted on is a
        # call bought for nothing. `max_rounds=1` therefore turns the whole loop
        # off at zero cost, which is what makes this component optional in
        # practice as well as on paper.
        return {"trace": [_event(continuing=False, reason="budget")]}

    unanswered = _unanswered(state)
    assessment = llm.with_structured_output(Sufficiency).invoke(
        load("sufficiency").format(
            brief=state.get("brief", ""),
            researched=_topic_digest(t for t in state["topics"] if t.status != "pending"),
            unanswered=_topic_digest(unanswered),
            n_supported=len(_supported(state)),
            claims_digest=_claim_digest(_supported(state)),
            n_rejected=len(_rejected(state)),
            rejected_digest=_claim_digest(_rejected(state)),
            uncertainties=_uncertainty_digest(state.get("uncertainties") or []),
        )
    )

    topics = _followups(state, assessment)
    # A topic the run looked into and could not stand up is a gap, whatever the
    # model put in `covered`. The naive check sees "researched" and moves on.
    covered = [t for t in assessment.covered if t not in {u.id for u in unanswered}]

    event = _event(
        continuing=bool(topics),
        reason=_reason(assessment, topics),
        covered=covered,
        confidence=assessment.confidence,
        gaps=[g.missing for g in assessment.gaps],
        new_topics=[t.question for t in topics],
    )
    if not topics:
        return {"trace": [event]}

    # `round` is the round being researched, so it moves only when one actually
    # opens. `dispatch` fans the new topics out; the researched ones it filters.
    return {"topics": topics, "round": state.get("round", 1) + 1, "trace": [event]}


def route_after_sufficiency(state: ResearchState) -> list[Send] | str:
    """Another round, or the report.

    Reuses `dispatch` (03) rather than restating its rule, so "researched topics
    never re-run" has one implementation. `or SYNTHESIZE_NODE` is load-bearing:
    `dispatch` answers with no `Send` for a budget that died mid-round, and
    routing that straight through would halt the run holding pending topics and
    no report.
    """
    return dispatch(state) or SYNTHESIZE_NODE


def _followups(state: ResearchState, assessment: Sufficiency) -> list[Topic]:
    """The proposals that survive: justified by a listed gap, and affordable.

    Ids continue the plan's numbering rather than restarting, because
    `merge_topics` merges by id — a collision would overwrite a finished topic
    with a pending one and research it a second time.
    """
    if not assessment.should_continue:
        return []

    existing = state.get("topics") or []
    keep = [t for t in assessment.new_topics
            if _justified(t, assessment.gaps) and _unasked(t, existing)]
    room = state["budget"].max_topics - len(existing)

    return [Topic(id=f"t{len(existing) + i}", question=t.question,
                  rationale=t.justifying_gap)
            for i, t in enumerate(keep[:room])]


def _justified(topic: NewTopic, gaps: list[Gap]) -> bool:
    """Does this proposal restate a gap the assessment actually listed?

    The rule that keeps the loop from running forever, and the reason it is code
    rather than a prompt line: unenforced, the model proposes plausible-adjacent
    topics indefinitely — each one reasonable, the set unbounded.
    """
    return any(_restates(topic.justifying_gap, gap.missing) for gap in gaps)


def _unasked(topic: NewTopic, existing: list[Topic]) -> bool:
    """Is this actually a new question?

    "Researched topics never re-run" is about the question, not the id. The
    prompt asks the check not to repeat itself; a check that does anyway would
    otherwise buy the same search a second time under a fresh id, which is
    re-running the topic with the bookkeeping hidden.
    """
    return not any(_restates(topic.question, t.question) for t in existing)


def _restates(text: str, other: str) -> bool:
    """Do these two say the same thing?

    The quote anchor (06), in both directions. Models reflow and recase text
    constantly, so byte equality would miss a restatement while adding nothing —
    a model inclined to fabricate a justification will fabricate a verbatim one.
    """
    return anchor(text, other) is not None or anchor(other, text) is not None


def _unanswered(state: ResearchState) -> list[Topic]:
    """Researched topics with nothing verified to show for it.

    Computed rather than inferred: which topics came back empty is a fact about
    the run, and asking a model to derive it from a rejection digest is asking
    it to do arithmetic it is bad at.
    """
    answered = {c.topic_id for c in _supported(state)}
    return [t for t in state.get("topics") or []
            if t.status != "pending" and t.id not in answered]


def _supported(state: ResearchState) -> list[Claim]:
    return [c for c in state.get("claims") or [] if c.verdict == "supported"]


def _rejected(state: ResearchState) -> list[Claim]:
    return [c for c in state.get("claims") or [] if c.verdict != "supported"]


def _reason(assessment: Sufficiency, topics: list[Topic]) -> str:
    """Why the loop did what it did. "The brief is covered" and "the check
    proposed topics that justified nothing" both end the run, and telling them
    apart is the difference between research being done and this node
    misbehaving."""
    if topics:
        return "deepening"
    if not assessment.should_continue or not assessment.new_topics:
        return "covered"
    return "unjustified"


def _event(**fields) -> dict:
    return ev("sufficiency", **fields)


def _topic_digest(topics) -> str:
    return "\n".join(f"- ({t.id}) {t.question}" for t in topics) or "none"


def _claim_digest(claims: list[Claim]) -> str:
    return "\n".join(f"- ({c.topic_id}) {c.text} [{c.verdict}]" for c in claims) or "none"


def _uncertainty_digest(uncertainties: list[Uncertainty]) -> str:
    """Kind first: `source_conflict` is the strongest signal in the set and the
    model can only weigh it if it can see which one it is looking at."""
    return "\n".join(f"- [{u.kind}] ({u.topic_id}) {u.description}"
                     for u in uncertainties) or "none"
