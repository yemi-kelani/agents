"""Spec 09 acceptance: the loop is bounded by the budget, not by the model.

Four criteria, and each is asserted where it is actually enforced rather than
where it is described:

- the round count never exceeds `max_rounds` whatever the model returns,
- every new topic traces to a gap the assessment listed,
- a topic whose claims were all rejected is a gap, not a covered topic,
- researched topics never re-run.

The first and third are the ones worth stating plainly. "Budget wins" is tested
against a model that votes for another round every single time it is asked, and
"all-rejected is not covered" is tested against a model that puts the topic in
`covered` anyway — because a criterion that only holds when the model cooperates
is not a criterion.

`asyncio.run` rather than pytest-asyncio, matching the rest of the suite.
"""
from __future__ import annotations

import asyncio

import pytest
from langchain_core.messages import AIMessage
from pydantic import ValidationError

from content_store import ContentStore
from graph import build_graph
from models import Models
from nodes.plan import RESEARCH_NODE, Plan, TopicSpec
from nodes.research import ExtractedClaim, ExtractedQuote, Extraction
from nodes.sufficiency import (
    SUFFICIENCY_NODE, Gap, NewTopic, Sufficiency, can_deepen,
    route_after_sufficiency, sufficiency_node,
)
from nodes.synthesize import SYNTHESIZE_NODE
from nodes.verify import VERIFY_NODE
from prompts import load
from search.base import SearchHit
from state import Budget, Claim, Quote, Topic, Uncertainty
from verifiers import Verdict

SOURCE_TEXT = (
    "Mimir scales to over 1 billion active series in a single cluster.\n\n"
    "It is deployed by large enterprises including several banks."
)
SCALES = "Mimir scales to over 1 billion active series"

URL = "https://grafana.com/mimir"
SID = ContentStore.source_id(URL)

GAP = "no source gives a 2025 series count"


# --- fakes ------------------------------------------------------------------

class ScriptedLLM:
    """One model for the whole pipeline: structured calls pick their reply by
    schema, the plain call is synthesis (08). Records every prompt, which is how
    "the budget refused before the model was asked" becomes assertable."""

    model_name = "fake-model"

    def __init__(self, **replies):
        self.prompts: list[tuple[str, str]] = []
        self.replies = {
            "Plan": Plan(brief="what a complete answer must cover",
                         topics=[TopicSpec(question=f"sub-question {i}", rationale="gap")
                                 for i in range(2)]),
            "Extraction": Extraction(claims=[ExtractedClaim(
                text="Mimir scales to over 1 billion active series.",
                quotes=[ExtractedQuote(text=SCALES)])]),
            "Sufficiency": an_assessment(),
        } | replies

    def with_structured_output(self, schema):
        return _Bound(self, schema)

    def invoke(self, prompt: str):
        self.prompts.append(("text", prompt))
        return AIMessage(content="A report. [1]")

    def prompt_for(self, schema: str) -> str:
        (prompt,) = [p for name, p in self.prompts if name == schema]
        return prompt


class _Bound:
    def __init__(self, llm: ScriptedLLM, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, prompt: str):
        self.llm.prompts.append((self.schema.__name__, prompt))
        reply = self.llm.replies[self.schema.__name__]
        return reply(prompt) if callable(reply) else reply

    async def ainvoke(self, prompt: str):
        return self.invoke(prompt)


class FakeVerifier:
    name = "fake:verifier"

    def __init__(self, verdict: str = "supported"):
        self._verdict = verdict
        self.checked: list[str] = []

    async def check(self, claim: str, quote: str) -> Verdict:
        self.checked.append(claim)
        return Verdict(verdict=self._verdict, reason="the quote states it")


class FakeRouter:
    """Records what it was asked, which is how "researched topics never re-run"
    is asserted against the searches that actually happened."""

    def __init__(self):
        self.queries: list[str] = []

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.queries.append(query)
        return [SearchHit(url=URL, title="Mimir", snippet="s",
                          content=SOURCE_TEXT, backend="test")]


# --- helpers ----------------------------------------------------------------

def an_assessment(**overrides) -> Sufficiency:
    """An assessment that proposes one properly justified follow-up."""
    return Sufficiency(**{
        "covered": ["t0"],
        "gaps": [Gap(topic_id="t1", missing=GAP, evidence="uncertainty on t1")],
        "new_topics": [NewTopic(question="What is Mimir's 2025 series count?",
                                justifying_gap=GAP)],
        "confidence": 0.4,
        "should_continue": True,
    } | overrides)


def an_endless_proposer():
    """A check that votes for another round every single time it is asked, with
    a fresh question each time so that nothing but the budget can stop it."""
    questions = iter(f"follow-up {i}" for i in range(1, 99))
    return lambda _: an_assessment(new_topics=[
        NewTopic(question=next(questions), justifying_gap=GAP)])


def a_topic(id: str = "t0", status: str = "researched") -> Topic:
    return Topic(id=id, question=f"question {id}", rationale="gap", status=status)


def a_claim(topic_id: str = "t0", id: str = "t0-c0",
            verdict: str = "supported") -> Claim:
    return Claim(id=id, topic_id=topic_id, text=f"a claim from {topic_id}",
                 quotes=[Quote(source_id=SID, text=SCALES, start=0, end=len(SCALES))],
                 verdict=verdict, verdict_reason="reason", verified_by="fake:verifier")


def a_state(**overrides) -> dict:
    return {"question": "How far does Mimir scale?",
            "brief": "Cover scale and who runs it.",
            "round": 1,
            "budget": Budget(),
            "topics": [a_topic("t0"), a_topic("t1")],
            "claims": [a_claim("t0"), a_claim("t1", id="t1-c0")],
            "uncertainties": [], **overrides}


def assess(state: dict | None = None, llm: ScriptedLLM | None = None) -> dict:
    return sufficiency_node(state if state is not None else a_state(),
                            llm=llm or ScriptedLLM())


def a_graph(*, llm=None, router=None, verifier=None):
    return build_graph(router=router or FakeRouter(), store=ContentStore(),
                       models=Models.uniform(llm or ScriptedLLM()),
                       verifier=verifier or FakeVerifier())


def run_graph(graph, **state) -> dict:
    config = {"configurable": {"thread_id": "test"}}
    payload = {"question": "How far does Mimir scale?", "clarified": True,
               "budget": Budget(), **state}
    return asyncio.run(graph.ainvoke(payload, config))


# --- the check is structured, not a boolean ---------------------------------

def test_the_assessment_cannot_propose_an_unbounded_number_of_topics():
    """A yes/no check gives you nothing to act on; an unbounded list of gaps
    gives you a second round the size of the first. Both ends are capped."""
    with pytest.raises(ValidationError):
        an_assessment(new_topics=[NewTopic(question=f"q{i}", justifying_gap=GAP)
                                  for i in range(4)])


def test_the_assessment_cannot_report_more_than_four_gaps():
    with pytest.raises(ValidationError):
        an_assessment(gaps=[Gap(topic_id=None, missing=f"gap {i}", evidence="e")
                            for i in range(5)])


def test_confidence_is_a_probability():
    with pytest.raises(ValidationError):
        an_assessment(confidence=1.4)


# --- acceptance: every new topic traces to a logged gap ---------------------

def test_a_topic_that_restates_a_listed_gap_is_kept():
    out = assess()

    (new,) = out["topics"]
    assert new.question == "What is Mimir's 2025 series count?"


def test_a_topic_that_cites_no_listed_gap_is_dropped():
    """The rule that keeps this from running forever. Without it the model
    generates plausible-adjacent topics indefinitely — each one reasonable, the
    set unbounded."""
    llm = ScriptedLLM(Sufficiency=an_assessment(new_topics=[NewTopic(
        question="How does Loki compare?",
        justifying_gap="it would be interesting to know")]))

    out = assess(llm=llm)

    assert "topics" not in out


def test_a_topic_proposed_against_no_gaps_at_all_is_dropped():
    """"Keep going" with nothing to point at is the failure mode this check
    exists to refuse."""
    llm = ScriptedLLM(Sufficiency=an_assessment(gaps=[]))

    assert "topics" not in assess(llm=llm)


def test_a_gap_restated_with_reflowed_whitespace_still_justifies_its_topic():
    """Models reflow text constantly. Demanding byte equality would drop real
    follow-ups over a line break, which is how a structural rule turns into a
    bug that looks like the model being unhelpful."""
    llm = ScriptedLLM(Sufficiency=an_assessment(new_topics=[NewTopic(
        question="What is Mimir's 2025 series count?",
        justifying_gap="no source gives\n  a 2025 SERIES count")]))

    assert len(assess(llm=llm)["topics"]) == 1


def test_a_new_topic_carries_the_gap_that_justified_it():
    """`rationale` is "why the brief needs this" (spec 01). For a round-two
    topic that is the gap, so the reason it exists survives into state."""
    (new,) = assess()["topics"]

    assert new.rationale == GAP


def test_a_new_topic_cannot_take_the_id_of_a_researched_one():
    """`merge_topics` merges by id, so a collision would overwrite a finished
    topic with a pending one — and re-run it."""
    (new,) = assess()["topics"]

    assert new.id not in {"t0", "t1"}
    assert new.status == "pending"


def test_a_topic_restating_one_already_researched_is_dropped():
    """"Researched topics never re-run" means the question, not the id. A check
    that repeats itself would otherwise buy the same search a second time under
    a fresh id — which is re-running the topic with the bookkeeping hidden."""
    llm = ScriptedLLM(Sufficiency=an_assessment(new_topics=[NewTopic(
        question="question t1", justifying_gap=GAP)]))

    assert "topics" not in assess(llm=llm)


def test_a_model_that_says_stop_proposes_nothing():
    llm = ScriptedLLM(Sufficiency=an_assessment(should_continue=False))

    assert "topics" not in assess(llm=llm)


# --- acceptance: the budget decides, the model only proposes ----------------

def test_the_last_allowed_round_is_not_even_asked():
    """Budget-first, and cheap: an assessment that cannot be acted on is an LLM
    call bought for nothing. The same reasoning as spec 08's empty report — do
    not ask a question whose answer cannot change what happens."""
    llm = ScriptedLLM()

    out = assess(a_state(round=5, budget=Budget(max_rounds=5)), llm)

    assert llm.prompts == []
    assert "topics" not in out


def test_an_exhausted_budget_stops_the_loop():
    spent = Budget(max_searches=10, searches_used=10)
    llm = ScriptedLLM()

    assert not can_deepen(a_state(budget=spent))
    assert "topics" not in assess(a_state(budget=spent), llm)
    assert llm.prompts == []


def test_a_full_topic_list_stops_the_loop():
    state = a_state(budget=Budget(max_topics=2))

    assert not can_deepen(state)


def test_new_topics_are_truncated_to_the_room_left_in_the_topic_budget():
    llm = ScriptedLLM(Sufficiency=an_assessment(new_topics=[
        NewTopic(question=f"follow-up {i}", justifying_gap=GAP) for i in range(3)]))

    out = assess(a_state(budget=Budget(max_topics=3)), llm)

    assert len(out["topics"]) == 1, "two topics researched, room for one more"


def test_a_round_that_appends_topics_opens_the_next_one():
    assert assess()["round"] == 2


def test_a_round_that_appends_nothing_leaves_the_round_where_it_is():
    """`round` is the round being researched. A proposal the budget or the
    justification rule refused did not open one."""
    llm = ScriptedLLM(Sufficiency=an_assessment(should_continue=False))

    assert "round" not in assess(llm=llm)


# --- acceptance: all-rejected is a gap, not a covered topic -----------------

def test_a_topic_whose_every_claim_was_rejected_is_not_recorded_as_covered():
    """The signal that makes this loop smarter than one counting coverage. A
    naive check sees "topic researched" and moves on; the agent looked and found
    nothing solid. Asserted against a model that claims it *was* covered,
    because a rule the model can opt out of is not a rule."""
    state = a_state(claims=[a_claim("t0"), a_claim("t1", id="t1-c0",
                                                   verdict="unsupported")])
    llm = ScriptedLLM(Sufficiency=an_assessment(covered=["t0", "t1"]))

    out = assess(state, llm)

    (event,) = [e for e in out["trace"] if e["kind"] == "sufficiency"]
    assert event["covered"] == ["t0"]


def test_a_topic_whose_quotes_were_all_fabricated_is_not_covered_either():
    state = a_state(claims=[a_claim("t1", id="t1-c0", verdict="quote_not_found")])
    llm = ScriptedLLM(Sufficiency=an_assessment(covered=["t1"]))

    (event,) = [e for e in assess(state, llm)["trace"] if e["kind"] == "sufficiency"]
    assert event["covered"] == []


def test_a_topic_that_produced_a_supported_claim_stays_covered():
    llm = ScriptedLLM(Sufficiency=an_assessment(covered=["t0", "t1"]))

    (event,) = [e for e in assess(llm=llm)["trace"] if e["kind"] == "sufficiency"]
    assert event["covered"] == ["t0", "t1"]


def test_a_topic_that_found_nothing_solid_is_shown_to_the_model_as_a_gap():
    """Computed here rather than inferred by the model: which topics came back
    empty is a fact about the run, and asking a model to derive it from a
    rejection digest is asking it to do arithmetic it is bad at."""
    state = a_state(claims=[a_claim("t0"), a_claim("t1", id="t1-c0",
                                                   verdict="unsupported")])
    llm = ScriptedLLM()

    assess(state, llm)

    prompt = llm.prompt_for("Sufficiency")
    found_nothing = prompt.split("FOUND NOTHING SOLID")[1].split("VERIFIED")[0]
    assert "t1" in found_nothing
    assert "t0" not in found_nothing


# --- uncertainty-driven expansion ------------------------------------------

def test_the_logged_uncertainties_reach_the_model_with_their_kind():
    """The substantive difference from a naive loop: round-two topics are driven
    by signals the round-one flows emitted, not by the model's opinion of its
    own work. `source_conflict` is the strongest of those and the model can only
    weigh it if it can see which kind it is."""
    state = a_state(uncertainties=[
        Uncertainty(topic_id="t1", description="Two sources disagree on the count.",
                    kind="source_conflict")])
    llm = ScriptedLLM()

    assess(state, llm)

    prompt = llm.prompt_for("Sufficiency")
    assert "source_conflict" in prompt
    assert "Two sources disagree on the count." in prompt


def test_the_rejected_claims_reach_the_model_with_their_count():
    state = a_state(claims=[a_claim("t0"), a_claim("t1", id="t1-c0",
                                                   verdict="unsupported")])
    llm = ScriptedLLM()

    assess(state, llm)

    prompt = llm.prompt_for("Sufficiency")
    assert "REJECTED CLAIMS (1)" in prompt
    assert "VERIFIED CLAIMS (1)" in prompt


def test_the_researched_topics_reach_the_model():
    llm = ScriptedLLM()

    assess(llm=llm)

    prompt = llm.prompt_for("Sufficiency")
    assert "question t0" in prompt and "question t1" in prompt


def test_the_brief_reaches_the_model():
    llm = ScriptedLLM()

    assess(llm=llm)

    assert "Cover scale and who runs it." in llm.prompt_for("Sufficiency")


# --- the trace --------------------------------------------------------------

def test_a_refusal_by_the_budget_is_traced_as_such():
    out = assess(a_state(round=5, budget=Budget(max_rounds=5)))

    (event,) = [e for e in out["trace"] if e["kind"] == "sufficiency"]
    assert event["continuing"] is False
    assert event["reason"] == "budget"


def test_an_unjustified_proposal_is_traced_as_such():
    """Distinguishable from "the brief is covered" on purpose: one means the
    research is done, the other means the check misbehaved."""
    llm = ScriptedLLM(Sufficiency=an_assessment(new_topics=[NewTopic(
        question="How does Loki compare?", justifying_gap="idle curiosity")]))

    (event,) = [e for e in assess(llm=llm)["trace"] if e["kind"] == "sufficiency"]
    assert event["reason"] == "unjustified"


def test_a_decision_to_deepen_is_traced_with_what_drove_it():
    (event,) = [e for e in assess()["trace"] if e["kind"] == "sufficiency"]

    assert event["continuing"] is True
    assert event["reason"] == "deepening"
    assert event["gaps"] == [GAP]
    assert event["confidence"] == 0.4


# --- routing ----------------------------------------------------------------

def test_the_route_fans_out_only_the_pending_topics():
    """Acceptance: researched topics never re-run. `dispatch` (03) already
    filters on status, so the loop reuses it rather than restating the rule."""
    state = a_state(topics=[a_topic("t0"), a_topic("t2", status="pending")])

    sends = route_after_sufficiency(state)

    assert [s.node for s in sends] == [RESEARCH_NODE]
    assert sends[0].arg["topic"].id == "t2"


def test_the_route_ends_at_synthesis_when_nothing_is_pending():
    assert route_after_sufficiency(a_state()) == SYNTHESIZE_NODE


def test_the_route_still_reaches_synthesis_when_the_budget_died_mid_round():
    """`dispatch` answers with no `Send` for a spent budget. Routing that
    straight through would halt the run with pending topics and no report — the
    one outcome worse than a short report."""
    state = a_state(topics=[a_topic("t2", status="pending")],
                    budget=Budget(max_searches=10, searches_used=10))

    assert route_after_sufficiency(state) == SYNTHESIZE_NODE


# --- the graph --------------------------------------------------------------

def test_sufficiency_sits_between_verify_and_synthesis():
    """Spec 07's cut is untouched: the loop hangs off the far side of verify, so
    a second round's claims are verified before anything can cite them."""
    edges = a_graph().get_graph().edges

    assert {e.target for e in edges if e.source == VERIFY_NODE} == {SUFFICIENCY_NODE}
    assert {e.target for e in edges if e.source == SUFFICIENCY_NODE} == \
        {RESEARCH_NODE, SYNTHESIZE_NODE}


def test_a_run_the_check_calls_covered_reaches_synthesis_in_one_round():
    router = FakeRouter()
    llm = ScriptedLLM(Sufficiency=an_assessment(should_continue=False, new_topics=[]))

    out = run_graph(a_graph(llm=llm, router=router))

    assert router.queries == ["sub-question 0", "sub-question 1"]
    assert out["round"] == 1
    assert out["report"]


def test_a_second_round_researches_the_new_topic_and_nothing_else():
    """Acceptance: researched topics never re-run."""
    router = FakeRouter()

    run_graph(a_graph(router=router), budget=Budget(max_rounds=2))

    assert router.queries == ["sub-question 0", "sub-question 1",
                              "What is Mimir's 2025 series count?"]


def test_a_model_that_always_wants_another_round_still_stops_at_max_rounds():
    """Acceptance, and the whole point of "budget is authoritative": the model
    votes to continue every time it is asked and gets exactly as many rounds as
    the budget allows."""
    router = FakeRouter()
    llm = ScriptedLLM(Sufficiency=an_endless_proposer())

    out = run_graph(a_graph(llm=llm, router=router), budget=Budget(max_rounds=3))

    assert out["round"] == 3
    assert router.queries == ["sub-question 0", "sub-question 1",
                              "follow-up 1", "follow-up 2"]


def test_one_round_is_a_run_with_no_loop_and_no_sufficiency_call():
    """`max_rounds=1` turns the loop off outright, and costs nothing to do so."""
    llm = ScriptedLLM()

    out = run_graph(a_graph(llm=llm), budget=Budget(max_rounds=1))

    assert [name for name, _ in llm.prompts if name == "Sufficiency"] == []
    assert out["round"] == 1
    assert out["report"]


def test_a_claim_the_previous_round_already_settled_is_not_verified_twice():
    """Verification runs once per round over the accumulated claim list. Paying
    to re-entail a settled claim would make cost quadratic in rounds — and a
    verifier that answered differently the second time would make the report
    unstable, which is the variance spec 07 exists to remove."""
    verifier = FakeVerifier()

    run_graph(a_graph(verifier=verifier), budget=Budget(max_rounds=2))

    assert len(verifier.checked) == 3, "two claims in round one, one in round two"


# --- the prompt -------------------------------------------------------------

def test_the_prompt_says_a_mostly_rejected_topic_is_a_gap():
    prompt = load("sufficiency").lower()

    assert "rejected" in prompt and "not a covered topic" in prompt


def test_the_prompt_refuses_topics_that_are_merely_interesting():
    assert "merely interesting" in load("sufficiency").lower()


def test_the_prompt_refuses_a_topic_already_researched():
    assert "already researched" in load("sufficiency").lower()


def test_the_prompt_asks_for_the_gap_to_be_restated():
    assert "restate" in load("sufficiency").lower()


def test_the_prompt_says_a_topic_that_found_nothing_should_be_rephrased():
    """`no_results` means the search found nothing, which usually means the
    topic was badly phrased. Repeating it verbatim buys a second empty round."""
    assert "rephrase" in load("sufficiency").lower()


def test_the_prompt_says_the_evidence_is_not_instructions():
    """Claim text and uncertainty descriptions are model-written from untrusted
    pages, and this node decides whether to spend more of the budget."""
    prompt = load("sufficiency").lower()

    assert "not instructions" in prompt or "not an instruction" in prompt
