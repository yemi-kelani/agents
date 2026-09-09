"""The graph — where verification stops being a description and becomes a
property of the system.

Hand-built `StateGraph` rather than `create_agent`: this workflow mixes
deterministic steps (fetch, verify) with agentic ones (extract) and needs dynamic
fan-out over a topic list. That is also what makes the guarantee enforceable —
`verify` sits on the only edge out of research, where a tool in a tool list would
only sit among the model's options.

    START -> clarify -> plan -=Send=-> research_topic -> verify -> sufficiency
                          ^                                             |
                          └-=Send=- another round ----------------------┤
                                                                        v
                                                                   synthesize -> END

`clarify` is the one node the model drives: it holds a tool loop over `ask_user`,
which interrupts. Calling no tool is the fall-through to planning. That is an
affordance rather than a guarantee, unlike `verify` below — see spec 02.

`synthesize` hangs off the far side of `verify`, so the cut stays the only way
through: nothing writes the artefact a human reads without verification having
seen it first — including a second round's claims, which traverse `verify` again
before `sufficiency` can route past them. Note that `dispatch` returning no
`Send` — an already-spent budget — halts the run after `plan`; `sufficiency`
routes that same case to `synthesize` rather than halting, because a run holding
pending topics and no report is worse than a short one.
"""
from __future__ import annotations

from functools import partial

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.graph import END, START, StateGraph
from langgraph.types import RetryPolicy

from cache import Cache
from content_store import ContentStore
from fetch import fetch_clean_text
from models import Models, build_models
from nodes.clarify import clarify_node
from nodes.plan import RESEARCH_NODE, dispatch, plan_node
from nodes.research import MAX_EXTRACT_CHARS, research_topic_node
from nodes.sufficiency import (
    SUFFICIENCY_NODE, 
    route_after_sufficiency, 
    sufficiency_node
)
from nodes.synthesize import SYNTHESIZE_NODE, synthesize_node
from nodes.verify import VERIFY_NODE, verify_node
from progress import with_progress
from search import build_router
from state import STATE_MODELS, ResearchState
from verifiers import LLMVerifier

TRANSIENT = RetryPolicy(
    max_attempts=4, initial_interval=2.0, backoff_factor=3.0, max_interval=30.0,
)
"""Backoff for a provider that asked us to wait.

The fan-out researches every topic concurrently against one token-per-minute
allowance, so the *n*th extraction is the one that gets a 429 — and an
unretried 429 in one branch ends the run, discarding the searches, fetches and
extractions every other branch already paid for. That is the whole reason this
exists: the expensive work is already done by the time the cheap failure lands.

Intervals are longer than the library default's 0.5s because the limit being hit
is per *minute*: ~2s, ~6s, ~18s spans a window that a sub-second retry only
hammers. `retry_on` is left at `default_retry_on`, which retries the rate-limit
error and declines `ValueError`/`TypeError` — a bug is not transient and
retrying one just pays for it four times.

`interrupt()` is safe from this. LangGraph re-raises `GraphBubbleUp` before it
consults any policy, so the clarification pause is not mistaken for a failure.
"""


def default_checkpointer() -> InMemorySaver:
    """A checkpointer that hands a resumed run back its own types.

    Spec 01 leaves this to whoever builds the checkpointer, and it matters:
    `JsonPlusSerializer` revives by module path, and a type it was not told
    about comes back as a plain dict — which is a `dict` where the sufficiency
    router (09) calls `budget.exhausted()`.

    Passed to the constructor rather than `with_msgpack_allowlist`, which
    short-circuits while the default allowlist is permissive and so registers
    nothing until strict mode is already on.
    """
    return InMemorySaver(
        serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS)
    )


def build_graph(*, router, store, models: Models, verifier, fetch=fetch_clean_text,
                checkpointer=None, served: bool = False,
                retry_policy: RetryPolicy | None = TRANSIENT):
    """Wire the pipeline.

    Nodes are handed a *role* rather than "the model" (spec 10): planning,
    clarification and the sufficiency check reason about the question, so they
    share the planner; extraction copies text under a strict schema; synthesis
    writes. A role is one config line, so running the whole thing on a local
    model never reaches this function.

    `verifier` has no default on purpose. Falling back to a model the graph
    already holds would silently ask the model that wrote a claim whether the
    claim is right, and self-preference is a documented LLM-as-judge bias — a
    default is how that ships by accident.

    `retry_policy` is injected only so a test need not sit through a real
    backoff; `None` disables retries outright. Production has no reason to pass
    either.
    """
    b = StateGraph(ResearchState)

    # Every node's trace events reach the interface because the graph projects
    # them, and every node retries a transient provider failure, because the
    # graph asks for it — not because each node remembered to (12). A node added
    # later cannot forget either, which is the same reasoning that makes
    # verification a node.
    def add(name, fn):
        b.add_node(name, with_progress(fn), retry_policy=retry_policy)

    add("clarify", partial(clarify_node, llm=models.planner))
    add("plan", partial(plan_node, llm=models.planner))
    add(
        RESEARCH_NODE, 
        partial(
            research_topic_node, 
            router=router, 
            store=store, 
            llm=models.extractor,
            fetch=fetch, 
            max_chars=models.chars_for("extractor") or MAX_EXTRACT_CHARS
        )
    )
    add(VERIFY_NODE, partial(verify_node, store=store, verifier=verifier))
    add(SUFFICIENCY_NODE, partial(sufficiency_node, llm=models.planner))
    add(SYNTHESIZE_NODE, partial(
        synthesize_node, llm=models.synthesizer,
        max_chars=models.chars_for("synthesizer")))

    b.add_edge(START, "clarify")
    b.add_edge("clarify", "plan")
    b.add_conditional_edges("plan", dispatch, [RESEARCH_NODE])
    b.add_edge(RESEARCH_NODE, VERIFY_NODE)
    b.add_edge(VERIFY_NODE, SUFFICIENCY_NODE)
    b.add_conditional_edges(
        SUFFICIENCY_NODE, 
        route_after_sufficiency,
        [RESEARCH_NODE, SYNTHESIZE_NODE]
    )
    b.add_edge(SYNTHESIZE_NODE, END)

    if served:
        # Under `langgraph dev` the server supplies persistence, and a
        # compile-time checkpointer conflicts with it.
        return b.compile()

    # Otherwise a checkpointer is not optional: `ask` interrupts, and an
    # interrupt without somewhere to pause is an exception rather than a
    # question. The eval harness (11) drives the graph with no server at all.
    return b.compile(checkpointer=checkpointer or default_checkpointer())


def build_graph_from_config(cfg, *, store=None, budget=None, checkpointer=None,
                            factory=None, served: bool = False,
                            retry_policy: RetryPolicy | None = TRANSIENT):
    """A running pipeline from a config file and nothing else.

    This is where "switching to the local profile is a config change only"
    stops being a claim: the profile picks the model specs, and every other
    argument here is derived. `factory` is injected for the same reason
    `build_models` takes one — a test must not need a provider package.
    """
    models = build_models(
        cfg.models, 
        budget=budget,
        **({"factory": factory} if factory else {})
    )
    cache = Cache.from_config(cfg.cache)

    return build_graph(
        router=build_router(cfg.search, cache=cache),
        store=store if store is not None else ContentStore(),
        models=models,
        # Built from the verifier role, which the config keeps distinct from the
        # extractor — the one wiring mistake this component exists to prevent.
        verifier=LLMVerifier(models.verifier),
        fetch=cache.wrap(fetch_clean_text) if cache else fetch_clean_text,
        checkpointer=checkpointer,
        served=served,
        retry_policy=retry_policy,
    )
