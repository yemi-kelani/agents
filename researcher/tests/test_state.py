"""Spec 01 acceptance: reducers, checkpoint size, dedup.

These drive real StateGraphs rather than asserting on the annotations, because
the failure mode the spec cares about (InvalidUpdateError on concurrent writes)
only appears at runtime.
"""
from __future__ import annotations

import pytest
from pydantic import ValidationError
from langgraph.graph import START, END, StateGraph
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from state import (
    STATE_MODELS,
    Budget,
    BudgetDelta,
    Claim,
    Quote,
    ResearchState,
    Source,
    Topic,
    ev,
    merge_budget,
    merge_topics,
)
from content_store import ContentStore


def claim(cid: str, topic_id: str = "t1") -> Claim:
    return Claim(
        id=cid,
        topic_id=topic_id,
        text=f"claim {cid}",
        quotes=[Quote(source_id="s1", text="verbatim", start=0, end=8)],
    )


# --- acceptance 1: parallel branches merge -----------------------------------

def test_two_branches_writing_claims_in_one_superstep_merge():
    def branch_a(state: ResearchState):
        return {"claims": [claim("a")]}

    def branch_b(state: ResearchState):
        return {"claims": [claim("b")]}

    b = StateGraph(ResearchState)
    b.add_node("a", branch_a)
    b.add_node("b", branch_b)
    b.add_edge(START, "a")
    b.add_edge(START, "b")
    b.add_edge("a", END)
    b.add_edge("b", END)

    out = b.compile().invoke({"question": "q"})

    assert {c.id for c in out["claims"]} == {"a", "b"}


def test_two_branches_writing_sources_and_uncertainties_merge():
    from state import Uncertainty

    def a(state: ResearchState):
        return {
            "sources": [Source(source_id="s1", url="http://a", title="A", backend="x")],
            "uncertainties": [Uncertainty(topic_id="t1", description="d1", kind="unconfirmed")],
        }

    def b_(state: ResearchState):
        return {
            "sources": [Source(source_id="s2", url="http://b", title="B", backend="x")],
            "uncertainties": [Uncertainty(topic_id="t2", description="d2", kind="no_results")],
        }

    b = StateGraph(ResearchState)
    b.add_node("a", a)
    b.add_node("b", b_)
    b.add_edge(START, "a")
    b.add_edge(START, "b")
    b.add_edge("a", END)
    b.add_edge("b", END)

    out = b.compile().invoke({"question": "q"})

    assert {s.source_id for s in out["sources"]} == {"s1", "s2"}
    assert len(out["uncertainties"]) == 2


# --- acceptance 2: checkpoint size independent of fetched bytes --------------

def _run_and_measure_checkpoint(doc: str) -> int:
    """Fetch `doc` into the content store, keep metadata in state, return
    the serialized size of the resulting checkpoint."""
    store = ContentStore()
    serde = JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS)

    def fetch(state: ResearchState):
        url = "http://example.com/doc"
        sid = ContentStore.source_id(url)
        store.put(sid, doc)
        return {
            "sources": [
                Source(source_id=sid, url=url, title="doc", backend="test",
                       char_count=len(doc))
            ]
        }

    b = StateGraph(ResearchState)
    b.add_node("fetch", fetch)
    b.add_edge(START, "fetch")
    b.add_edge("fetch", END)

    saver = InMemorySaver(serde=serde)
    cfg = {"configurable": {"thread_id": "t"}}
    b.compile(checkpointer=saver).invoke({"question": "q"}, cfg)

    tup = saver.get_tuple(cfg)
    _type, blob = serde.dumps_typed(tup.checkpoint)
    return len(blob)


def test_checkpoint_size_is_independent_of_fetched_bytes():
    small = _run_and_measure_checkpoint("x" * 100)
    huge = _run_and_measure_checkpoint("x" * 2_000_000)

    # Only char_count digits differ; allow generous slack for serializer noise.
    assert abs(huge - small) < 2_000, (
        f"checkpoint grew with document size: {small} -> {huge} bytes"
    )


# --- acceptance 3: same URL from two topics yields one Source ---------------

def test_same_url_from_two_topics_yields_one_source_id():
    url = "https://example.com/paper"
    assert ContentStore.source_id(url) == ContentStore.source_id(url)


def test_content_store_dedupes_same_url_across_topics():
    store = ContentStore()
    url = "https://example.com/paper"

    store.put(ContentStore.source_id(url), "text from topic A")
    store.put(ContentStore.source_id(url), "text from topic A")

    assert len(store) == 1


# --- merge_topics reducer ---------------------------------------------------

def test_merge_topics_lets_right_win_on_same_id():
    left = [Topic(id="t1", question="q1", rationale="r1", status="pending")]
    right = [Topic(id="t1", question="q1", rationale="r1", status="researched")]

    merged = merge_topics(left, right)

    assert len(merged) == 1
    assert merged[0].status == "researched"


def test_merge_topics_appends_new_ids():
    left = [Topic(id="t1", question="q1", rationale="r1")]
    right = [Topic(id="t2", question="q2", rationale="r2")]

    merged = merge_topics(left, right)

    assert {t.id for t in merged} == {"t1", "t2"}


def test_parallel_branches_flipping_topic_status_do_not_lose_topics():
    """The failure the spec warns about: two branches both rewriting `topics`."""
    def a(state: ResearchState):
        return {"topics": [Topic(id="t1", question="q1", rationale="r", status="researched")]}

    def b_(state: ResearchState):
        return {"topics": [Topic(id="t2", question="q2", rationale="r", status="researched")]}

    b = StateGraph(ResearchState)
    b.add_node("a", a)
    b.add_node("b", b_)
    b.add_edge(START, "a")
    b.add_edge(START, "b")
    b.add_edge("a", END)
    b.add_edge("b", END)

    seed = [Topic(id="t1", question="q1", rationale="r"),
            Topic(id="t2", question="q2", rationale="r")]
    out = b.compile().invoke({"question": "q", "topics": seed})

    assert {t.id for t in out["topics"]} == {"t1", "t2"}
    assert all(t.status == "researched" for t in out["topics"])


# --- messages channel (spec 12 depends on this being here) ------------------

def test_messages_channel_appends_across_nodes():
    from langchain.messages import AIMessage

    def a(state: ResearchState):
        return {"messages": [AIMessage(content="from a")]}

    def b_(state: ResearchState):
        return {"messages": [AIMessage(content="from b")]}

    b = StateGraph(ResearchState)
    b.add_node("a", a)
    b.add_node("b", b_)
    b.add_edge(START, "a")
    b.add_edge("a", "b")
    b.add_edge("b", END)

    out = b.compile().invoke({"question": "q"})

    assert [m.content for m in out["messages"]] == ["from a", "from b"]


# --- Budget -----------------------------------------------------------------

def test_budget_is_not_exhausted_when_fresh():
    assert Budget().exhausted() is False


@pytest.mark.parametrize(
    "field,limit",
    [("searches_used", "max_searches"),
     ("fetches_used", "max_fetches"),
     ("tokens_used", "max_total_tokens")],
)
def test_budget_exhausts_on_each_limit(field, limit):
    b = Budget()
    setattr(b, field, getattr(b, limit))
    assert b.exhausted() is True


def test_budget_tokens_are_cumulative_not_a_context_window():
    """max_total_tokens sums every model call in the run, so it exhausts by
    accumulation rather than by any single call being large."""
    b = Budget(max_total_tokens=100)
    for _ in range(10):
        b.tokens_used += 10
    assert b.exhausted() is True


# --- merge_budget reducer ---------------------------------------------------

def test_budget_channel_defaults_to_a_usable_budget():
    """Nothing wrote `budget`, so the router (spec 09) still gets real limits."""
    def noop(state: ResearchState):
        return {}

    b = StateGraph(ResearchState)
    b.add_node("noop", noop)
    b.add_edge(START, "noop")
    b.add_edge("noop", END)

    out = b.compile().invoke({"question": "q"})

    assert out["budget"].max_searches == Budget().max_searches
    assert out["budget"].exhausted() is False


def test_a_budget_write_configures_limits():
    """The entry payload sets the run profile (FAST/FULL in spec 12)."""
    fast = Budget(max_rounds=1, max_topics=2, max_searches=4, max_fetches=4)

    b = StateGraph(ResearchState)
    b.add_node("noop", lambda state: {})
    b.add_edge(START, "noop")
    b.add_edge("noop", END)

    out = b.compile().invoke({"question": "q", "budget": fast})

    assert out["budget"].max_searches == 4
    assert out["budget"].searches_used == 0


def test_two_branches_reporting_spend_in_one_superstep_sum():
    """The InvalidUpdateError case: parallel branches both spending budget."""
    fast = Budget(max_searches=4, max_fetches=4)

    def a(state: ResearchState):
        return {"budget": BudgetDelta(searches=1, fetches=2, tokens=100)}

    def b_(state: ResearchState):
        return {"budget": BudgetDelta(searches=1, fetches=1, tokens=250)}

    b = StateGraph(ResearchState)
    b.add_node("a", a)
    b.add_node("b", b_)
    b.add_edge(START, "a")
    b.add_edge(START, "b")
    b.add_edge("a", END)
    b.add_edge("b", END)

    out = b.compile().invoke({"question": "q", "budget": fast})

    assert (out["budget"].searches_used, out["budget"].fetches_used) == (2, 3)
    assert out["budget"].tokens_used == 350
    assert out["budget"].max_searches == 4, "a spend report must not clobber limits"


def test_spend_accumulates_across_rounds():
    delta = BudgetDelta(searches=2)
    b = Budget(max_searches=4)

    for _ in range(2):
        b = merge_budget(b, delta)

    assert b.searches_used == 4
    assert b.exhausted() is True


def test_a_budget_arriving_as_json_is_still_a_budget():
    """A served graph (12) is handed its input over HTTP, so the run profile
    arrives as a plain dict rather than as a `Budget`. Left as one it reaches
    `dispatch`, which calls `budget.exhausted()` — and the whole run dies on the
    first fan-out with `'dict' object has no attribute`."""
    merged = merge_budget(Budget(), {"max_rounds": 1, "max_topics": 2,
                                     "max_searches": 4, "max_fetches": 4})

    assert isinstance(merged, Budget)
    assert merged.max_topics == 2
    assert merged.exhausted() is False


def test_a_json_budget_survives_the_channel_and_not_just_the_reducer():
    """The reported failure was not in `merge_budget` alone — it was raised from
    `apply_writes`, when the channel folded the entry payload in. The reducer
    tests above call the function directly and so would still pass if the
    channel stopped routing a first write through it, which is precisely the
    arrangement that put a dict in front of `right.searches`. This drives the
    served graph's path instead: JSON in at START, a real `Budget` in the node,
    and a delta written back as JSON the way a revived checkpoint supplies it."""
    seen = {}

    def dispatch(state: ResearchState):
        seen["budget"] = state["budget"]
        seen["exhausted"] = state["budget"].exhausted()
        return {"budget": {"searches": 1}}

    b = StateGraph(ResearchState)
    b.add_node("dispatch", dispatch)
    b.add_edge(START, "dispatch")
    b.add_edge("dispatch", END)

    out = b.compile().invoke({"question": "q", "budget": {
        "max_rounds": 1, "max_topics": 2, "max_searches": 4, "max_fetches": 4}})

    assert isinstance(seen["budget"], Budget), "the node must not be handed a dict"
    assert seen["exhausted"] is False
    assert out["budget"].max_searches == 4, "the run profile must survive the delta"
    assert out["budget"].searches_used == 1


def test_a_spend_report_arriving_as_json_still_accumulates():
    """The other side of the same coin: a delta revived from a checkpoint the
    server persisted comes back as a dict too, and summing it must not silently
    replace the limits it does not carry."""
    budget = merge_budget(Budget(max_searches=10), {"searches": 3})

    assert budget.searches_used == 3
    assert budget.max_searches == 10, "a delta must never overwrite the limits"


def test_the_two_shapes_are_told_apart_by_what_they_carry():
    """`Budget` and `BudgetDelta` share no field names, so a dict is
    unambiguous. Guessing wrong in either direction is silent: a delta read as
    configuration resets the run's limits, and configuration read as a delta
    adds nothing and loses them."""
    assert not set(Budget.model_fields) & set(BudgetDelta.model_fields)


def test_an_unrecognised_budget_shape_is_refused_rather_than_guessed():
    """Better a loud failure at the first write than a run silently given
    default limits nobody chose."""
    with pytest.raises(ValidationError):
        merge_budget(Budget(), {"max_rnods": 1})


# --- trace events -----------------------------------------------------------

def test_ev_carries_kind_timestamp_and_fields():
    e = ev("verdict", claim_id="c1", verdict="unsupported", reason="no support")

    assert e["kind"] == "verdict"
    assert isinstance(e["ts"], float)
    assert e["claim_id"] == "c1"
    assert e["verdict"] == "unsupported"


def test_trace_accumulates_across_parallel_branches():
    def a(state: ResearchState):
        return {"trace": [ev("search", query="a")]}

    def b_(state: ResearchState):
        return {"trace": [ev("search", query="b")]}

    b = StateGraph(ResearchState)
    b.add_node("a", a)
    b.add_node("b", b_)
    b.add_edge(START, "a")
    b.add_edge(START, "b")
    b.add_edge("a", END)
    b.add_edge("b", END)

    out = b.compile().invoke({"question": "q"})

    assert {e["query"] for e in out["trace"]} == {"a", "b"}


# --- claim defaults ---------------------------------------------------------

def test_claim_starts_unverified():
    assert claim("c1").verdict == "unverified"
    assert claim("c1").verified_by is None


# --- checkpoint round trip --------------------------------------------------

def test_state_models_survive_a_checkpoint_round_trip():
    """A resumed run must get its Pydantic models back as objects, not dicts —
    the sufficiency router calls `budget.exhausted()` and synthesis reads
    `claim.verdict`.

    Registering STATE_MODELS is what keeps that true: the serializer revives by
    module path, and an unregistered type is a warning today and blocked under
    LANGGRAPH_STRICT_MSGPACK.
    """
    def seed(state: ResearchState):
        return {
            "claims": [claim("c1")],
            "topics": [Topic(id="t1", question="q1", rationale="r")],
            "budget": Budget(max_searches=4),
        }

    b = StateGraph(ResearchState)
    b.add_node("seed", seed)
    b.add_edge(START, "seed")
    b.add_edge("seed", END)

    serde = JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS)
    cfg = {"configurable": {"thread_id": "roundtrip"}}
    graph = b.compile(checkpointer=InMemorySaver(serde=serde))
    graph.invoke({"question": "q"}, cfg)

    restored = graph.get_state(cfg).values

    assert isinstance(restored["claims"][0], Claim)
    assert isinstance(restored["claims"][0].quotes[0], Quote)
    assert restored["claims"][0].verdict == "unverified"
    assert isinstance(restored["topics"][0], Topic)
    assert isinstance(restored["budget"], Budget)
    assert restored["budget"].exhausted() is False
