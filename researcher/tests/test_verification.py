"""Spec 07 acceptance: verification is mandatory, two-tiered, and swappable.

The five acceptance criteria are the spine of this file — no path through the
graph reaches the end without passing `verify`, every claim leaves with a verdict
that is not `unverified`, a fabricated quote is caught without spending a token,
verdicts stream as they resolve rather than in a batch at the end, and the
verifier swaps out without the graph noticing.

The first of those is the one worth stating plainly: it is asserted against the
*compiled graph's* topology, not against a node's return value. A test that
called `verify_node` and checked its output would prove the node works and
nothing at all about whether anything can route around it.

`asyncio.run` rather than pytest-asyncio, matching the rest of the suite.
"""
from __future__ import annotations

import asyncio
import time
from collections import deque
from functools import partial

import pytest
from langchain_core.messages import AIMessage
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from anchor import anchor
from content_store import ContentStore
from graph import build_graph, default_checkpointer
from models import Models
from nodes.plan import Plan, TopicSpec
from nodes.research import ExtractedClaim, ExtractedQuote, Extraction
from nodes.sufficiency import Sufficiency
from nodes.verify import VERIFY_NODE, verify_claim, verify_node
from prompts import load
from search.base import SearchHit
from state import Budget, Claim, Quote, ResearchState
from verifiers import LLMVerifier, Verdict, Verifier

SOURCE_TEXT = (
    "Mimir scales to over 1 billion active series in a single cluster.\n\n"
    "It is deployed by large enterprises including several banks."
)

SCALES = "Mimir scales to over 1 billion active series"
DEPLOYED = "deployed by large enterprises"

URL = "https://grafana.com/mimir"
SID = ContentStore.source_id(URL)


# --- fakes ------------------------------------------------------------------

class FakeVerifier:
    """A `Verifier` that answers what the test told it to, and records what it
    was asked — which is how "no LLM call happened" becomes assertable."""

    name = "fake:verifier"

    def __init__(self, verdict="supported", reason="the quote states it", *, delay=0.0):
        self._verdict = verdict
        self._reason = reason
        self._delay = delay
        self.checked: list[tuple[str, str]] = []

    async def check(self, claim: str, quote: str) -> Verdict:
        self.checked.append((claim, quote))
        if self._delay:
            await asyncio.sleep(self._delay)
        verdict = self._verdict(claim) if callable(self._verdict) else self._verdict
        return Verdict(verdict=verdict, reason=self._reason)


class ScriptedLLM:
    """One model for the whole pipeline. `with_structured_output` picks the
    reply by schema, which is how a single injected LLM serves assess, plan,
    extract and verify — and how the end-to-end test stays one fake wide."""

    model_name = "fake-model"

    def __init__(self, **replies):
        self.replies = replies                       # schema name -> value or callable
        self.prompts: list[tuple[str, str]] = []

    def with_structured_output(self, schema):
        return _Bound(self, schema)

    def invoke(self, prompt: str):
        """Synthesis (08) is a plain call, not a structured one. Its output is
        not what this file is about — it is here so the end-to-end runs reach
        the end of the graph."""
        self.prompts.append(("text", prompt))
        return AIMessage(content=self.replies.get("text", "A report. [1]"))


class _Bound:
    def __init__(self, llm: ScriptedLLM, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, prompt: str):
        self.llm.prompts.append((self.schema.__name__, prompt))
        reply = self.llm.replies[self.schema.__name__]
        return reply(prompt) if callable(reply) else reply

    async def ainvoke(self, prompt: str):
        return self.invoke(prompt)


class FakeRouter:
    def __init__(self, *hits: SearchHit):
        self._hits = list(hits)

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return self._hits[:k]


# --- helpers ----------------------------------------------------------------

def a_store(text: str = SOURCE_TEXT, source_id: str = SID) -> ContentStore:
    store = ContentStore()
    store.put(source_id, text)
    return store


def quote_of(text: str, *, source: str = SOURCE_TEXT, source_id: str = SID) -> Quote:
    """A quote anchored the way spec 06 anchors it."""
    span = anchor(text, source)
    assert span is not None, "fixture bug: the quote is not in the source"
    return Quote(source_id=source_id, text=text, start=span.start, end=span.end)


def a_claim(text: str = "Mimir scales to over 1 billion active series.",
            *quotes: Quote, id: str = "t0-c0") -> Claim:
    return Claim(id=id, topic_id="t0", text=text,
                 quotes=list(quotes) or [quote_of(SCALES)])


def verify(claim: Claim, *, store: ContentStore | None = None, verifier=None) -> Claim:
    return asyncio.run(verify_claim(
        claim, store=store or a_store(), verifier=verifier or FakeVerifier()
    ))


def verify_graph(store: ContentStore, verifier):
    """The verify node alone, so streaming and the `Overwrite` are testable
    without dragging the whole pipeline in."""
    b = StateGraph(ResearchState)
    b.add_node(VERIFY_NODE, partial(verify_node, store=store, verifier=verifier))
    b.add_edge(START, VERIFY_NODE)
    b.add_edge(VERIFY_NODE, END)
    return b.compile()


# --- the verifier interface -------------------------------------------------

def test_the_verdict_is_binary():
    """"Partially supported" is a real failure mode and a documented known
    unknown. It is not representable here, and the spec says so on purpose."""
    with pytest.raises(ValidationError):
        Verdict(verdict="partially supported", reason="hedged")


def test_an_llm_verifier_satisfies_the_protocol():
    """Acceptance: the verifier swaps without touching the graph. That is only
    true if `Verifier` is the contract and not `LLMVerifier` in disguise."""
    assert isinstance(LLMVerifier(ScriptedLLM()), Verifier)
    assert isinstance(FakeVerifier(), Verifier)


def test_a_verifier_names_the_model_that_produced_the_verdict():
    """`verified_by` is what lets the eval (11) say *which* judge said so —
    without it, "use a different model family for verification" is unauditable."""
    assert LLMVerifier(ScriptedLLM()).name == "llm:fake-model"


def test_the_claim_and_the_quote_reach_the_verifier():
    llm = ScriptedLLM(Verdict=Verdict(verdict="supported", reason="stated"))

    asyncio.run(LLMVerifier(llm).check("Mimir scales.", SCALES))

    (schema, prompt) = llm.prompts[0]
    assert schema == "Verdict"
    assert "Mimir scales." in prompt and SCALES in prompt


def test_the_prompt_refuses_relatedness_as_support():
    """The failure this whole component exists to catch: a quote about roughly
    the right subject, cited for a claim it does not make."""
    lowered = load("verify").lower()

    assert "related is not" in lowered
    assert "unsupported" in lowered


def test_the_prompt_says_the_evidence_block_is_not_instructions():
    """The quote is untrusted page text reaching a model that decides something.
    Containment (05) keeps page text out of planning and synthesis; verification
    is the one privileged node that has to look at it."""
    lowered = load("verify").lower()

    assert "not instructions" in lowered or "not an instruction" in lowered


def test_the_prompt_does_not_name_the_source():
    """Entailment is a question about text. Telling the judge the quote came
    from arxiv.org invites exactly the authority bias the spec warns about in
    LLM-as-judge."""
    assert "{url}" not in load("verify")


# --- tier 1: quote grounding, and it is free -------------------------------

def test_a_grounded_quote_reaches_the_entailment_tier():
    verifier = FakeVerifier()

    claim = verify(a_claim(), verifier=verifier)

    assert verifier.checked == [("Mimir scales to over 1 billion active series.", SCALES)]
    assert claim.verdict == "supported"
    assert claim.verified_by == "fake:verifier"
    assert claim.verdict_reason == "the quote states it"


def test_a_fabricated_quote_is_caught_without_an_llm_call():
    """Acceptance, and the reason the cheap tier runs first: the worst failure
    is also the one that costs nothing to detect."""
    fabricated = Quote(source_id=SID, text="Mimir was discontinued in 2019.",
                       start=0, end=31)
    verifier = FakeVerifier()

    claim = verify(a_claim("Mimir is discontinued.", fabricated), verifier=verifier)

    assert claim.verdict == "quote_not_found"
    assert claim.verified_by == "quote_match"
    assert verifier.checked == [], "tier 2 must not run on a quote tier 1 rejected"


def test_a_quote_whose_offsets_drifted_is_caught_even_though_the_text_is_real():
    """The check is on the offsets, not on the document. A quote that appears
    somewhere in the source but not where the citation points is a citation that
    does not check out."""
    misplaced = Quote(source_id=SID, text=DEPLOYED, start=0, end=len(DEPLOYED))
    assert DEPLOYED in SOURCE_TEXT, "fixture: the text really is in the document"

    claim = verify(a_claim("Mimir is deployed by enterprises.", misplaced))

    assert claim.verdict == "quote_not_found"


def test_a_quote_the_model_reflowed_still_grounds():
    """Spec 06 anchors through a whitespace- and case-normalised match, so its
    offsets legitimately span text that differs from `quote.text` by a newline.
    A byte-equality check here would reject every one of those as fabricated."""
    source = "Mimir scales to over\n1 billion active series."
    store = a_store(source)
    reflowed = quote_of("scales to over 1 billion", source=source)
    assert source[reflowed.start:reflowed.end] != reflowed.text, "fixture: they differ"

    claim = verify(a_claim("Mimir scales.", reflowed), store=store)

    assert claim.verdict == "supported"


def test_a_claim_with_no_quotes_at_all_never_reaches_the_llm():
    """Spec 06 drops these before they leave extraction. Defence in depth: a
    claim arriving from anywhere else must not be entailed against nothing."""
    verifier = FakeVerifier()
    quoteless = Claim(id="t0-c0", topic_id="t0", text="Mimir scales.", quotes=[])

    claim = verify(quoteless, verifier=verifier)

    assert claim.verdict == "quote_not_found"
    assert verifier.checked == []


def test_the_first_grounded_quote_is_the_one_entailed():
    """A claim carrying one bad quote and one good one still has real evidence.
    Tier 1 is free, so checking both costs nothing and saves the claim."""
    fabricated = Quote(source_id=SID, text="Mimir was discontinued.", start=0, end=23)
    verifier = FakeVerifier()

    claim = verify(a_claim("Mimir scales.", fabricated, quote_of(SCALES)),
                   verifier=verifier)

    assert [q for c, q in verifier.checked] == [SCALES]
    assert claim.verdict == "supported"


def test_a_claim_whose_every_quote_is_ungrounded_is_quote_not_found():
    bad = [Quote(source_id=SID, text=f"fabrication {i}", start=0, end=14) for i in range(2)]
    verifier = FakeVerifier()

    claim = verify(a_claim("Mimir scales.", *bad), verifier=verifier)

    assert claim.verdict == "quote_not_found"
    assert verifier.checked == []


# --- tier 2: entailment -----------------------------------------------------

def test_an_unsupported_verdict_carries_the_reason_it_failed():
    """"Unsupported" is a first-class outcome, and a report that says why is
    more useful than one that only says no."""
    verifier = FakeVerifier(verdict="unsupported", reason="the quote says 1 billion, not 1 million")

    claim = verify(a_claim("Mimir scales to 1 million series."), verifier=verifier)

    assert claim.verdict == "unsupported"
    assert claim.verdict_reason == "the quote says 1 billion, not 1 million"
    assert claim.verified_by == "fake:verifier"


def test_verification_does_not_mutate_the_claim_it_was_handed():
    """`claims` is replaced wholesale by the node. A verifier that mutated in
    place would also alter the copy the reducer is about to overwrite, hiding
    whether the replacement actually happened."""
    original = a_claim()

    verify(original)

    assert original.verdict == "unverified" and original.verified_by is None


# --- the node ---------------------------------------------------------------

def test_every_claim_leaves_the_node_with_a_verdict():
    """Acceptance: nothing stays `unverified`. This is the property the whole
    component exists to make true."""
    claims = [a_claim(f"claim {i}", id=f"t0-c{i}") for i in range(3)]
    graph = verify_graph(a_store(), FakeVerifier())

    out = asyncio.run(graph.ainvoke({"claims": claims}))

    assert [c.verdict for c in out["claims"]] == ["supported"] * 3
    assert all(c.verdict != "unverified" for c in out["claims"])


def test_verified_claims_replace_the_originals_rather_than_appending():
    """`claims` carries an `add` reducer for the fan-in, so a plain return would
    leave state holding both the unverified and the verified copy of every
    claim — and synthesis would cite whichever it reached first."""
    claims = [a_claim(f"claim {i}", id=f"t0-c{i}") for i in range(2)]
    graph = verify_graph(a_store(), FakeVerifier())

    out = asyncio.run(graph.ainvoke({"claims": claims}))

    assert len(out["claims"]) == 2


def test_every_verdict_is_traced():
    graph = verify_graph(a_store(), FakeVerifier(verdict="unsupported", reason="nope"))

    out = asyncio.run(graph.ainvoke({"claims": [a_claim()]}))

    (event,) = [e for e in out["trace"] if e["kind"] == "verdict"]
    assert event["claim_id"] == "t0-c0"
    assert event["verdict"] == "unsupported"
    assert event["reason"] == "nope"
    assert event["verified_by"] == "fake:verifier"


def test_claims_are_verified_concurrently():
    """Wall-clock for 4 claims ~= 1 claim. Verification is the one node that
    runs an LLM call per claim; doing them in series is the difference between
    a slow pipeline and an unusable one."""
    delay = 0.2
    claims = [a_claim(f"claim {i}", id=f"t0-c{i}") for i in range(4)]
    graph = verify_graph(a_store(), FakeVerifier(delay=delay))

    started = time.perf_counter()
    asyncio.run(graph.ainvoke({"claims": claims}))
    elapsed = time.perf_counter() - started

    assert elapsed < delay * 2, f"verification was serial: {elapsed:.2f}s for 4 claims"


def test_a_run_that_found_no_claims_verifies_cleanly():
    """The empty set is a real outcome — every source was off topic, or every
    fetch failed. It must not be an error on the mandatory path."""
    out = asyncio.run(verify_graph(a_store(), FakeVerifier()).ainvoke({"claims": []}))

    assert out["claims"] == []


# --- acceptance: verdicts stream as they resolve ---------------------------

def test_verdicts_stream_as_they_resolve_not_in_a_batch_at_the_end():
    """A progress indicator that arrives all at once is not progress. The slow
    claim is first in state; if its verdict is also first out of the stream, the
    node is flushing at the end rather than emitting per verdict."""
    slow, fast = a_claim("slow claim", id="t0-c0"), a_claim("fast claim", id="t0-c1")

    class Staggered(FakeVerifier):
        async def check(self, claim: str, quote: str) -> Verdict:
            if claim == "slow claim":
                await asyncio.sleep(0.15)
            return await super().check(claim, quote)

    graph = verify_graph(a_store(), Staggered())

    async def collect() -> list[dict]:
        return [
            chunk async for chunk in
            graph.astream({"claims": [slow, fast]}, stream_mode="custom")
        ]

    events = asyncio.run(collect())

    assert [e["claim_id"] for e in events] == ["t0-c1", "t0-c0"]
    assert all(e["kind"] == "verdict" for e in events)


def test_a_streamed_verdict_is_the_same_shape_the_trace_records():
    """The UI (12) reads one event shape whether it is replaying `trace` or
    following a live run."""
    graph = verify_graph(a_store(), FakeVerifier())

    async def collect():
        streamed = [c async for c in graph.astream({"claims": [a_claim()]},
                                                   stream_mode="custom")]
        final = await graph.ainvoke({"claims": [a_claim()]})
        return streamed, final

    streamed, final = asyncio.run(collect())
    traced = [e for e in final["trace"] if e["kind"] == "verdict"]

    assert [e["claim_id"] for e in streamed] == [e["claim_id"] for e in traced]
    assert streamed[0].keys() == traced[0].keys()


# --- acceptance: nothing routes around verify ------------------------------

def a_graph(*, verifier=None, llm=None, store=None, fetch=None):
    return build_graph(
        router=FakeRouter(SearchHit(url=URL, title="Mimir", snippet="s",
                                    content=SOURCE_TEXT, backend="test")),
        store=store if store is not None else ContentStore(),
        models=Models.uniform(llm or a_scripted_llm()),
        verifier=verifier or FakeVerifier(),
        fetch=fetch,
    )


def a_scripted_llm(**overrides) -> ScriptedLLM:
    replies = {
        "Plan": Plan(brief="what a complete answer must cover",
                     topics=[TopicSpec(question=f"sub-question {i}", rationale="gap")
                             for i in range(2)]),
        "Extraction": Extraction(claims=[ExtractedClaim(
            text="Mimir scales to over 1 billion active series.",
            quotes=[ExtractedQuote(text=SCALES)])]),
        # The loop (09) sits on the path now. These runs are about verification,
        # so the check calls the brief covered and they stay one round.
        "Sufficiency": Sufficiency(covered=[], gaps=[], new_topics=[],
                                   confidence=0.9, should_continue=False),
    }
    return ScriptedLLM(**(replies | overrides))


def reachable_without(graph, forbidden: str) -> set[str]:
    """Every node reachable from START without ever entering `forbidden`."""
    drawn = graph.get_graph()
    adjacency: dict[str, list[str]] = {}
    for edge in drawn.edges:
        adjacency.setdefault(edge.source, []).append(edge.target)

    seen, queue = set(), deque([START])
    while queue:
        node = queue.popleft()
        for target in adjacency.get(node, []):
            if target == forbidden or target in seen:
                continue
            seen.add(target)
            queue.append(target)
    return seen


def test_nothing_reaches_the_end_of_the_graph_without_passing_verify():
    """Acceptance, and the entire thesis: verification is a property of the
    topology, not of what a model chose to do. Spec 08 hangs `synthesize` off
    the far side of `verify`, so this cut stays the only way through."""
    assert END not in reachable_without(a_graph(), VERIFY_NODE)


def test_research_leads_to_verify_and_nowhere_else():
    edges = a_graph().get_graph().edges
    targets = {e.target for e in edges if e.source == "research_topic"}

    assert targets == {VERIFY_NODE}


def test_the_verifier_is_swappable_without_touching_the_graph():
    """Acceptance. `build_graph` is handed a `Verifier`; it does not know or
    care that the NLI cross-encoder is the planned replacement."""
    class CountingVerifier:
        name = "nli:deberta"

        def __init__(self):
            self.calls = 0

        async def check(self, claim: str, quote: str) -> Verdict:
            self.calls += 1
            return Verdict(verdict="unsupported", reason="entailment score below threshold")

    verifier = CountingVerifier()
    out = run_graph(a_graph(verifier=verifier))

    assert verifier.calls == len(out["claims"]) > 0
    assert {c.verified_by for c in out["claims"]} == {"nli:deberta"}


def test_a_verifier_is_required_rather_than_defaulted_to_the_extraction_model():
    """Defaulting would silently ask the model that wrote the claim whether the
    claim is right. Self-preference bias is a documented LLM-as-judge failure,
    and a default is how it gets shipped by accident."""
    with pytest.raises(TypeError):
        build_graph(router=FakeRouter(), store=ContentStore(),
                    models=Models.uniform(a_scripted_llm()))


# --- end to end -------------------------------------------------------------

def run_graph(graph, **state) -> dict:
    """Drive the compiled graph with the clarification round pre-latched, the
    way the eval harness (11) drives it with no human attached."""
    config = {"configurable": {"thread_id": "test"}}
    payload = {"question": "How far does Mimir scale?", "clarified": True,
               "budget": Budget(), **state}
    return asyncio.run(graph.ainvoke(payload, config))


def test_a_claim_travels_from_a_search_hit_to_a_verdict():
    """The pipeline composes: plan fans out, both topics research the same page,
    the fan-in merges, and every claim comes back with a verdict on it."""
    store = ContentStore()
    out = run_graph(a_graph(store=store))

    assert len(out["claims"]) == 2, "one claim per topic"
    assert {c.verdict for c in out["claims"]} == {"supported"}
    assert all(c.verified_by == "fake:verifier" for c in out["claims"])
    for c in out["claims"]:
        for q in c.quotes:
            assert store.get(q.source_id)[q.start:q.end] == q.text


def test_a_fabricated_quote_survives_the_whole_pipeline_as_quote_not_found():
    """End to end, the failure that matters: the extractor invents a quote, the
    anchor drops it, the claim is dropped with it — and if one did slip through
    with bad offsets, verification is still there to catch it."""
    llm = a_scripted_llm(Extraction=Extraction(claims=[ExtractedClaim(
        text="Mimir was discontinued.",
        quotes=[ExtractedQuote(text="Mimir was discontinued in 2019.")])]))

    out = run_graph(a_graph(llm=llm))

    assert out["claims"] == []
    assert [e["kind"] for e in out["trace"] if e["kind"] == "quote_not_found"]


# --- the checkpointer spec 01 asked for ------------------------------------

def test_the_default_checkpointer_revives_state_models_rather_than_dicts():
    """Spec 01 leaves this to whoever builds the checkpointer. Without the
    allowlist a resumed run gets a plain dict where the sufficiency router (09)
    calls `budget.exhausted()`."""
    serde = default_checkpointer().serde
    budget = Budget(max_topics=3, searches_used=1)

    revived = serde.loads_typed(serde.dumps_typed(budget))

    assert isinstance(revived, Budget)
    assert revived.exhausted() is False and revived.max_topics == 3


def test_a_model_outside_the_allowlist_is_refused_rather_than_revived():
    """Proves the allowlist is doing something: an arbitrary class arriving from
    a checkpoint is not reconstructed just because it was asked for."""
    serde = default_checkpointer().serde

    revived = serde.loads_typed(serde.dumps_typed(Verdict(verdict="supported", reason="r")))

    assert isinstance(revived, dict)
