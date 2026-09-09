"""Spec 06 acceptance: claims anchored to verbatim quotes with character offsets.

The four acceptance criteria are the spine of this file — every claim carries an
anchored quote or is dropped, `source_text[q.start:q.end]` reproduces `q.text`, a
hallucinated quote yields `quote_not_found` rather than a silent pass, and an
off-topic source produces nothing.

The model, the search router and the fetcher are injected fakes. What is *not*
faked is the anchoring: it is the half of this spec that has to be right, and a
test that stubbed it would assert nothing.

`asyncio.run` rather than pytest-asyncio, matching the rest of the suite.
"""
from __future__ import annotations

import asyncio
from functools import partial

import httpx
import pytest
from langgraph.graph import END, START, StateGraph
from pydantic import ValidationError

from anchor import anchor
from fetch.containment import UNTRUSTED_PREAMBLE, datamark, undatamark
from fetch.guard import BlockedURL
from nodes.plan import RESEARCH_NODE, TopicTask, dispatch
from nodes.research import (
    MAX_CLAIMS,
    MAX_EXTRACT_CHARS,
    MAX_QUOTES,
    MAX_SOURCES_PER_TOPIC,
    SEARCH_K,
    Extraction,
    ExtractedClaim,
    ExtractedQuote,
    research_topic_node,
)
from content_store import ContentStore
from prompts import load
from search.base import SearchHit
from state import Budget, BudgetDelta, ResearchState, Topic

SOURCE_TEXT = (
    "Mimir scales to over 1 billion active series in a single cluster.\n\n"
    "It is deployed by large enterprises including several banks."
)

SCALES = "Mimir scales to over 1 billion active series"
DEPLOYED = "deployed by large enterprises"

URL = "https://grafana.com/mimir"


# --- fakes ------------------------------------------------------------------

class FakeLLM:
    """One `Extraction` per source, in order, recording every prompt so what
    reaches the model is assertable."""

    def __init__(self, *extractions: Extraction):
        self._queue = list(extractions)
        self.prompts: list[str] = []

    def with_structured_output(self, schema):
        assert schema is Extraction
        return self

    async def ainvoke(self, prompt: str) -> Extraction:
        self.prompts.append(prompt)
        return self._queue.pop(0) if self._queue else Extraction()


class FakeRouter:
    def __init__(self, *hits: SearchHit):
        self._hits = list(hits)
        self.queries: list[tuple[str, int]] = []

    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        self.queries.append((query, k))
        return self._hits[:k]


class FakeFetcher:
    """url -> clean text, or an exception to raise. `requested` is what actually
    left the node, which is what makes "this was not fetched" assertable."""

    def __init__(self, pages: dict[str, str | Exception] | None = None):
        self.pages = pages if pages is not None else {URL: SOURCE_TEXT}
        self.requested: list[str] = []

    async def __call__(self, url: str) -> str:
        self.requested.append(url)
        page = self.pages[url]
        if isinstance(page, Exception):
            raise page
        return page


# --- helpers ----------------------------------------------------------------

def a_hit(url: str = URL, title: str = "Mimir", content: str | None = None) -> SearchHit:
    return SearchHit(url=url, title=title, snippet="s", content=content, backend="test")


def a_claim(text: str, *quotes: str) -> ExtractedClaim:
    return ExtractedClaim(text=text, quotes=[ExtractedQuote(text=q) for q in quotes])


def a_task(question: str = "How far does Mimir scale?") -> TopicTask:
    return TopicTask(topic=Topic(id="t0", question=question, rationale="r"), brief="the brief")


def research(task=None, *, router=None, llm=None, fetch=None, store=None) -> dict:
    """Run the node once and hand back its state update."""
    return asyncio.run(research_topic_node(
        task or a_task(),
        router=router or FakeRouter(a_hit()),
        store=store if store is not None else ContentStore(),
        llm=llm or FakeLLM(Extraction(claims=[a_claim("Mimir scales.", SCALES)])),
        fetch=fetch or FakeFetcher(),
    ))


def kinds(out: dict, kind: str) -> list[dict]:
    return [e for e in out["trace"] if e["kind"] == kind]


# --- anchoring: exact -------------------------------------------------------

def test_an_exact_quote_anchors_to_the_span_that_reproduces_it():
    """The acceptance criterion, stated as code: the offsets are what a verifier
    checks, so `source_text[start:end]` has to come back as the quote itself."""
    span = anchor(SCALES, SOURCE_TEXT)

    assert span is not None
    assert SOURCE_TEXT[span.start:span.end] == SCALES
    assert span.method == "exact"


def test_the_offsets_point_past_the_start_of_the_document():
    """A matcher that always answered (0, len(quote)) would pass a test anchored
    at the top of the source and nothing else."""
    span = anchor(DEPLOYED, SOURCE_TEXT)

    assert span is not None and span.start > 0
    assert SOURCE_TEXT[span.start:span.end] == DEPLOYED


# --- anchoring: whitespace and case ----------------------------------------

def test_a_quote_whose_whitespace_the_model_normalised_still_anchors():
    """Models re-flow line breaks into spaces constantly. The offsets must still
    land on the original span, newline and all."""
    source = "Mimir scales to over\n1 billion active series."

    span = anchor("scales to over 1 billion", source)

    assert span is not None and span.method == "normalized"
    assert source[span.start:span.end] == "scales to over\n1 billion"


def test_a_quote_the_model_recased_still_anchors():
    span = anchor("MIMIR SCALES TO OVER 1 BILLION", SOURCE_TEXT)

    assert span is not None and span.method == "normalized"
    assert SOURCE_TEXT[span.start:span.end] == "Mimir scales to over 1 billion"


# --- anchoring: fuzzy, and its boundary ------------------------------------

def test_a_trivial_model_edit_anchors_fuzzily_and_says_so():
    """Accepted, but never silently: a "corrected" quote is not verbatim, and
    the eval reports the fuzzy rate separately (spec 11)."""
    span = anchor("Mimir scales to over 1 billion activ series", SOURCE_TEXT)

    assert span is not None and span.method == "fuzzy"
    assert SOURCE_TEXT[span.start:span.end].startswith("Mimir scales to over")


def test_the_fuzzy_threshold_is_a_knob_not_a_constant():
    """Raise it and the same near-miss is refused — which is what an operator
    who wants strictly verbatim citations turns."""
    near_miss = "Mimir scales to over 1 billion activ series"

    assert anchor(near_miss, SOURCE_TEXT, fuzzy_threshold=0.999) is None


def test_a_fabricated_quote_anchors_nowhere():
    """The single most useful thing this pipeline detects."""
    assert anchor("Mimir was discontinued in 2019 and replaced.", SOURCE_TEXT) is None


def test_a_quote_that_only_reuses_the_vocabulary_anchors_nowhere():
    """The hard fabrication: same words, different assertion. Fuzzy matching is
    where a threshold set too loose would wave this through."""
    assert anchor("Mimir scales to over 1 million active tenants", SOURCE_TEXT) is None


@pytest.mark.parametrize("empty", ["", "   ", "\n\n"])
def test_an_empty_quote_anchors_nowhere(empty):
    """`"".find` answers 0 — an empty quote would otherwise anchor at the top of
    every document ever fetched."""
    assert anchor(empty, SOURCE_TEXT) is None


def test_a_quote_longer_than_the_source_anchors_nowhere():
    assert anchor(SOURCE_TEXT * 3, SOURCE_TEXT) is None


# --- anchoring: the datamarking round trip ---------------------------------

def test_undatamarking_reverses_the_marking():
    assert undatamark(datamark(SCALES)) == SCALES


def test_a_quote_copied_out_of_a_marked_block_anchors_after_unmarking():
    """Spotlighting replaces every space in the untrusted block, so a quote the
    model copied faithfully comes back marked. Anchoring it as-is would report a
    fabricated quote for content the model got exactly right."""
    marked = datamark(SCALES)

    assert anchor(marked, SOURCE_TEXT) is None, "the marked form is not in the source"
    assert anchor(undatamark(marked), SOURCE_TEXT).method == "exact"


# --- the extraction schema --------------------------------------------------

def test_a_claim_with_no_quote_is_rejected_structurally():
    """"Each claim carries a quote" in a prompt is advisory; this makes an
    unquoted claim unrepresentable."""
    with pytest.raises(ValidationError):
        ExtractedClaim(text="Mimir scales.", quotes=[])


def test_the_claim_and_quote_counts_are_bounded():
    with pytest.raises(ValidationError):
        Extraction(claims=[a_claim(f"c{i}", SCALES) for i in range(MAX_CLAIMS + 1)])

    with pytest.raises(ValidationError):
        a_claim("c", *[SCALES] * (MAX_QUOTES + 1))


def test_the_quote_field_description_repeats_the_verbatim_rule():
    """Models paraphrase quotes by default, and the field description is read at
    the moment the quote is being written — closer to the failure than the
    prompt is."""
    described = ExtractedQuote.model_fields["text"].description.lower()

    assert "exact" in described
    assert "paraphrase" in described


# --- the prompt -------------------------------------------------------------

def test_the_prompt_demands_verbatim_quotes():
    assert "exactly" in load("extract").lower()


def test_the_prompt_says_the_empty_set_is_a_valid_answer():
    """Without this, a model manufactures relevance from an off-topic page."""
    assert "no claims" in load("extract").lower()


def test_the_prompt_asks_for_atomic_claims_and_separate_uncertainties():
    lowered = load("extract").lower()

    assert "atomic" in lowered
    assert "uncertaint" in lowered


def test_the_claim_cap_is_stated_once_and_shared_with_the_schema():
    """The number the prompt asks for and the number the schema enforces are the
    same number, interpolated rather than typed twice, so they cannot drift."""
    llm = FakeLLM()
    research(llm=llm)

    assert "{max_claims}" in load("extract")
    assert f"at most {MAX_CLAIMS} claims" in llm.prompts[0]


def test_the_sub_question_and_the_marked_page_reach_the_model():
    llm = FakeLLM()
    research(a_task("How far does Mimir scale?"), llm=llm)

    prompt = llm.prompts[0]
    assert "How far does Mimir scale?" in prompt
    assert UNTRUSTED_PREAMBLE in prompt
    assert datamark(SOURCE_TEXT) in prompt, "page text must reach the model marked"


def test_the_page_text_is_truncated_before_it_reaches_the_model():
    """A truncation, not a chunking strategy — the documented limitation. What
    matters here is that it is bounded at all."""
    tail = "TAIL-MARKER"
    long_page = "Mimir scales. " * 2000 + tail
    assert len(long_page) > MAX_EXTRACT_CHARS

    llm = FakeLLM()
    research(llm=llm, fetch=FakeFetcher({URL: long_page}))

    assert tail not in llm.prompts[0]
    assert datamark(long_page[:MAX_EXTRACT_CHARS]) in llm.prompts[0]


# --- the node: sources and the content store -------------------------------

def test_fetched_text_lands_in_the_store_and_only_metadata_in_state():
    """No blobs in state: LangGraph serializes state on every checkpoint write."""
    store = ContentStore()
    out = research(store=store)

    (source,) = out["sources"]
    assert store.get(source.source_id) == SOURCE_TEXT
    assert source.url == URL
    assert source.char_count == len(SOURCE_TEXT)
    assert source.backend == "test"
    assert SOURCE_TEXT not in repr(out["sources"])


def test_a_source_is_scored_for_authority():
    out = research(router=FakeRouter(a_hit(url="https://arxiv.org/abs/2304.09848")),
                   fetch=FakeFetcher({"https://arxiv.org/abs/2304.09848": SOURCE_TEXT}))

    assert out["sources"][0].authority == 0.85


def test_a_backend_that_already_returned_clean_text_is_not_fetched():
    """Tavily returns page text with the result; SearXNG does not. Fetching
    anyway would pay twice for the same page."""
    fetcher = FakeFetcher({})
    out = research(router=FakeRouter(a_hit(content=SOURCE_TEXT)), fetch=fetcher)

    assert fetcher.requested == []
    assert out["sources"][0].char_count == len(SOURCE_TEXT)


def test_a_page_already_in_the_store_is_not_fetched_again():
    """Two topics that turn up the same page fetch it once."""
    store = ContentStore()
    store.put(ContentStore.source_id(URL), SOURCE_TEXT)
    fetcher = FakeFetcher({})

    out = research(store=store, fetch=fetcher)

    assert fetcher.requested == []
    assert len(out["sources"]) == 1


def test_a_source_with_nothing_extractable_is_skipped():
    """`to_clean_text` answers "" for a page with no article. Extracting claims
    from an empty document is how a model gets asked to invent them."""
    llm = FakeLLM()
    out = research(llm=llm, fetch=FakeFetcher({URL: ""}))

    assert out["sources"] == []
    assert llm.prompts == [], "an empty document must never reach the model"


@pytest.mark.parametrize("error", [BlockedURL("private range"), httpx.ConnectError("down")])
def test_a_fetch_failure_is_traced_and_the_rest_of_the_topic_continues(error):
    router = FakeRouter(a_hit(url="https://bad.example/x"), a_hit())
    fetcher = FakeFetcher({"https://bad.example/x": error, URL: SOURCE_TEXT})

    out = research(router=router, fetch=fetcher)

    failed = kinds(out, "fetch_failed")
    assert [e["url"] for e in failed] == ["https://bad.example/x"]
    assert failed[0]["error"] == type(error).__name__
    assert len(out["sources"]) == 1, "the healthy source still researches"


def test_only_a_bounded_number_of_sources_is_researched():
    urls = [f"https://example.com/{i}" for i in range(SEARCH_K + 2)]
    router = FakeRouter(*[a_hit(url=u) for u in urls])

    out = research(router=router, fetch=FakeFetcher({u: SOURCE_TEXT for u in urls}))

    assert len(out["sources"]) == MAX_SOURCES_PER_TOPIC


def test_a_failed_fetch_gives_up_its_slot_to_the_next_hit():
    """The search asked for more hits than it will read. That margin exists so a
    dead link costs a source rather than the topic."""
    urls = [f"https://example.com/{i}" for i in range(SEARCH_K)]
    pages: dict[str, str | Exception] = {u: SOURCE_TEXT for u in urls}
    pages[urls[0]] = httpx.ConnectError("down")

    out = research(router=FakeRouter(*[a_hit(url=u) for u in urls]),
                   fetch=FakeFetcher(pages))

    assert len(out["sources"]) == MAX_SOURCES_PER_TOPIC


def test_the_search_runs_on_the_topic_question():
    router = FakeRouter(a_hit())
    research(a_task("How far does Mimir scale?"), router=router)

    assert router.queries == [("How far does Mimir scale?", SEARCH_K)]


# --- the node: claims and their anchors ------------------------------------

def test_every_claim_comes_back_anchored_into_the_stored_text():
    """The acceptance criterion, end to end."""
    store = ContentStore()
    llm = FakeLLM(Extraction(claims=[
        a_claim("Mimir scales to over 1 billion active series.", SCALES),
        a_claim("Mimir is deployed by large enterprises.", DEPLOYED),
    ]))

    out = research(store=store, llm=llm)

    assert len(out["claims"]) == 2
    for c in out["claims"]:
        assert c.quotes
        for q in c.quotes:
            assert store.get(q.source_id)[q.start:q.end] == q.text


def test_claims_carry_their_topic_and_unique_ids():
    llm = FakeLLM(Extraction(claims=[a_claim(f"c{i}", SCALES) for i in range(3)]))

    out = research(llm=llm)

    assert {c.topic_id for c in out["claims"]} == {"t0"}
    assert len({c.id for c in out["claims"]}) == 3


def test_claims_start_unverified():
    """Spec 06 anchors; spec 07 decides. A claim leaving extraction with a
    verdict already on it would make the mandatory node decorative."""
    out = research()

    assert [c.verdict for c in out["claims"]] == ["unverified"]
    assert out["claims"][0].verified_by is None


def test_a_kept_claim_is_traced():
    out = research()

    (event,) = kinds(out, "claim")
    assert event["claim_id"] == out["claims"][0].id
    assert "Mimir scales" in event["text"]


def test_a_quote_the_model_copied_out_of_the_marked_block_still_anchors():
    """The end-to-end version of the round trip: the model is shown marked text,
    so it copies marked text back. That must not read as a fabrication."""
    llm = FakeLLM(Extraction(claims=[a_claim("Mimir scales.", datamark(SCALES))]))

    out = research(llm=llm)

    (claim,) = out["claims"]
    assert claim.quotes[0].text == SCALES
    assert SOURCE_TEXT[claim.quotes[0].start:claim.quotes[0].end] == SCALES
    assert kinds(out, "quote_not_found") == []


def test_a_fuzzy_anchor_is_recorded_so_the_eval_can_report_the_rate():
    """A high fuzzy rate means the extraction prompt is the problem. That is
    only actionable if the rate is visible."""
    llm = FakeLLM(Extraction(claims=[
        a_claim("Mimir scales.", "Mimir scales to over 1 billion activ series")
    ]))

    out = research(llm=llm)

    (event,) = kinds(out, "quote_fuzzy")
    assert event["claim_id"] == out["claims"][0].id


# --- the node: fabricated quotes -------------------------------------------

def test_a_hallucinated_quote_is_reported_not_silently_passed():
    fabricated = "Mimir was discontinued in 2019 and replaced."
    llm = FakeLLM(Extraction(claims=[
        a_claim("Mimir is discontinued.", fabricated),
        a_claim("Mimir scales.", SCALES),
    ]))

    out = research(llm=llm)

    (event,) = kinds(out, "quote_not_found")
    assert event["quote"].startswith("Mimir was discontinued")
    assert event["claim"].startswith("Mimir is discontinued")


def test_a_claim_whose_every_quote_is_fabricated_is_dropped():
    """Acceptance: every claim carries >=1 quote or is dropped. A quoteless
    claim reaching synthesis is an uncitable assertion in the report."""
    llm = FakeLLM(Extraction(claims=[
        a_claim("Mimir is discontinued.", "Mimir was discontinued in 2019."),
        a_claim("Mimir scales.", SCALES),
    ]))

    out = research(llm=llm)

    assert [c.text for c in out["claims"]] == ["Mimir scales."]
    assert kinds(out, "claim_dropped")[0]["text"].startswith("Mimir is discontinued")


def test_a_claim_keeps_the_quotes_that_did_anchor():
    """One bad quote of two demotes the citation, not the claim."""
    llm = FakeLLM(Extraction(claims=[
        a_claim("Mimir scales.", "Mimir was discontinued in 2019.", SCALES)
    ]))

    out = research(llm=llm)

    (claim,) = out["claims"]
    assert [q.text for q in claim.quotes] == [SCALES]


def test_an_off_topic_source_produces_no_claims_rather_than_stretched_ones():
    out = research(llm=FakeLLM(Extraction(claims=[], uncertainties=[])))

    assert out["claims"] == []
    assert out["sources"], "the source was still read; it just said nothing useful"


# --- the node: uncertainties -----------------------------------------------

def test_uncertainties_are_collected_against_the_topic():
    """Cheap to collect, and what gives spec 09's round 2 a concrete reason to
    spawn a topic."""
    llm = FakeLLM(Extraction(
        claims=[a_claim("Mimir scales.", SCALES)],
        uncertainties=["The source does not say which release added the limit."],
    ))

    out = research(llm=llm)

    (u,) = out["uncertainties"]
    assert u.topic_id == "t0"
    assert u.kind == "unconfirmed"
    assert "which release" in u.description


def test_a_topic_nobody_could_search_records_a_no_results_uncertainty():
    """The router answers [] when every backend failed or found nothing. Someone
    has to turn that into a research outcome, and this is the only node that
    sees it."""
    out = research(router=FakeRouter())

    assert out["sources"] == [] and out["claims"] == []
    assert [u.kind for u in out["uncertainties"]] == ["no_results"]
    assert out["topics"][0].status == "failed"


def test_a_topic_whose_every_source_failed_to_fetch_is_marked_failed():
    router = FakeRouter(a_hit(url="https://bad.example/x"))
    fetcher = FakeFetcher({"https://bad.example/x": httpx.ConnectError("down")})

    out = research(router=router, fetch=fetcher)

    assert out["topics"][0].status == "failed"
    assert [u.kind for u in out["uncertainties"]] == ["no_results"]


# --- the node: what it hands back to the graph ------------------------------

def test_the_topic_comes_back_researched_without_being_mutated_in_place():
    """`merge_topics` merges by id, and a node that mutated the Topic it was
    handed would flip the status of the copy still sitting in state."""
    task = a_task()

    out = research(task)

    assert out["topics"][0].status == "researched"
    assert task["topic"].status == "pending"


def test_the_node_reports_what_it_spent():
    """A `BudgetDelta`, never a `Budget`: two topics reporting in the same
    super-step must sum rather than clobber the configured limits."""
    urls = [f"https://example.com/{i}" for i in range(2)]
    router = FakeRouter(*[a_hit(url=u) for u in urls])

    out = research(router=router, fetch=FakeFetcher({u: SOURCE_TEXT for u in urls}))

    assert out["budget"] == BudgetDelta(searches=1, fetches=2)


def test_a_failed_fetch_still_counts_against_the_budget():
    """It cost a request. A budget that only counted successes would let a
    domain of dead links run forever."""
    router = FakeRouter(a_hit(url="https://bad.example/x"))
    fetcher = FakeFetcher({"https://bad.example/x": httpx.ConnectError("down")})

    out = research(router=router, fetch=fetcher)

    assert out["budget"].fetches == 1


def test_the_search_is_traced_with_its_result_count():
    out = research()

    (event,) = kinds(out, "search")
    assert event["topic_id"] == "t0"
    assert event["n_results"] == 1


def test_the_fetch_is_traced_with_the_size_of_what_came_back():
    out = research()

    (event,) = kinds(out, "fetch")
    assert (event["url"], event["status"], event["chars"]) == (URL, "ok", len(SOURCE_TEXT))


# --- under the real fan-out -------------------------------------------------

def test_the_node_runs_under_the_spec_03_fan_out():
    """The payload contract is only real if a `Send` built by spec 03's dispatch
    is what this node actually receives, and the writes of two concurrent
    branches have to merge rather than raise `InvalidUpdateError`."""
    store = ContentStore()
    pages: dict[str, str | Exception] = {
        f"https://example.com/{i}": SOURCE_TEXT for i in range(2)
    }

    def seed(_state: ResearchState) -> dict:
        return {
            "brief": "the brief",
            "topics": [Topic(id=f"t{i}", question=f"q{i}", rationale="r") for i in range(2)],
        }

    def node_for(i: int):
        return partial(
            research_topic_node,
            router=FakeRouter(a_hit(url=f"https://example.com/{i}")),
            store=store,
            llm=FakeLLM(Extraction(claims=[a_claim(f"claim {i}", SCALES)])),
            fetch=FakeFetcher(pages),
        )

    async def route(task: TopicTask) -> dict:
        i = int(task["topic"].id[1:])
        return await node_for(i)(task)

    b = StateGraph(ResearchState)
    b.add_node("seed", seed)
    b.add_node(RESEARCH_NODE, route)
    b.add_edge(START, "seed")
    b.add_conditional_edges("seed", dispatch, [RESEARCH_NODE])
    b.add_edge(RESEARCH_NODE, END)

    out = asyncio.run(b.compile().ainvoke({"question": "q", "budget": Budget()}))

    assert sorted(c.text for c in out["claims"]) == ["claim 0", "claim 1"]
    assert {t.status for t in out["topics"]} == {"researched"}
    assert (out["budget"].searches_used, out["budget"].fetches_used) == (2, 2)
    assert out["budget"].max_topics == Budget().max_topics, "spend must not clobber limits"
    for c in out["claims"]:
        for q in c.quotes:
            assert store.get(q.source_id)[q.start:q.end] == q.text
