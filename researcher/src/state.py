"""Graph state, the types it carries, and the reducers that merge parallel writes.

Two LangGraph failure modes shape this module:

1. Parallel branches writing the same un-annotated key raise `InvalidUpdateError`.
   The fan-out over topics means several nodes write `claims`, `sources`,
   `uncertainties`, `trace` and `budget` in the same super-step, so every one of
   those keys carries a reducer.
2. State is serialized on every checkpoint write. Fetched page text therefore
   lives in the ContentStore keyed by `source_id`; state carries only metadata.
"""
from __future__ import annotations

import time
from operator import add
from typing import Annotated, Literal, TypedDict

from langgraph.graph import add_messages
from pydantic import BaseModel, ConfigDict


# --- evidence ---------------------------------------------------------------

class Source(BaseModel):
    """Metadata only. The text lives in the ContentStore, keyed by source_id."""
    source_id: str                    # sha256(url)[:16]
    url: str
    title: str
    backend: str                      # which search adapter produced it
    fetched_at: str | None = None
    char_count: int = 0
    authority: float = 0.5            # 0-1, heuristic; see 04-search.md


class Quote(BaseModel):
    """A verbatim span. The offsets are what make a citation checkable."""
    source_id: str
    text: str
    start: int                        # char offset into the stored clean text
    end: int


class Claim(BaseModel):
    id: str
    topic_id: str
    text: str                         # one atomic assertion
    quotes: list[Quote]
    verdict: Literal["unverified", "supported", "unsupported", "quote_not_found"] = "unverified"
    verdict_reason: str | None = None
    verified_by: str | None = None    # "quote_match" | "llm:gpt-4.1-mini" | "nli:deberta"


class Topic(BaseModel):
    id: str
    question: str                     # a specific answerable sub-question
    rationale: str                    # why the brief needs this
    status: Literal["pending", "researched", "failed"] = "pending"


class Uncertainty(BaseModel):
    """Emitted by a research flow. Feeds the sufficiency check (spec 09)."""
    topic_id: str
    description: str
    kind: Literal["unconfirmed", "source_conflict", "no_results"]


# --- budget -----------------------------------------------------------------

class Budget(BaseModel):
    """Limits plus the spend against them. The authoritative stop: the LLM
    sufficiency check *proposes*, this *decides* (spec 09)."""

    model_config = ConfigDict(extra="forbid")
    """A served graph (12) is handed this over HTTP by a client somebody typed.
    Pydantic ignores unknown keys by default, so `max_rnods: 1` would silently
    buy the default five rounds and the bill that goes with them — the whole
    point of this model is that a limit is a limit."""

    max_rounds: int = 5
    max_topics: int = 20
    max_searches: int = 20
    max_fetches: int = 250
    max_total_tokens: int = 500_000    # cumulative spend for the whole run

    searches_used: int = 0
    fetches_used: int = 0
    tokens_used: int = 0

    def exhausted(self) -> bool:
        return (self.searches_used >= self.max_searches
                or self.fetches_used >= self.max_fetches
                or self.tokens_used >= self.max_total_tokens)


FAST = Budget(max_rounds=1, max_topics=2, max_searches=4, max_fetches=4)
"""~45s. Interactive use and smoke tests. `max_rounds=1` switches the deepening
loop (09) off outright, and off costs nothing — the sufficiency check never
asks."""

FULL = Budget(max_rounds=2, max_topics=5, max_searches=12, max_fetches=18)
"""1-4 minutes. A profile is a `Budget`, so it travels on the input payload —
which is what a served graph (12) needs, since the server imports a module
attribute and cannot pass constructor arguments."""


class BudgetDelta(BaseModel):
    """What one node reports it spent. Never carries limits."""
    searches: int = 0
    fetches: int = 0
    tokens: int = 0


# --- reducers ---------------------------------------------------------------

def merge_topics(left: list[Topic], right: list[Topic]) -> list[Topic]:
    """Merge by id, right wins. Plain `add` would duplicate a topic whose status
    a branch flipped, and a bare overwrite would lose the other branch's flip."""
    by_id = {t.id: t for t in left}
    for t in right:
        by_id[t.id] = t
    return list(by_id.values())


def merge_budget(left: Budget | dict, right: Budget | BudgetDelta | dict) -> Budget:
    """Limits are configured once; spend accumulates.

    A `Budget` is a configuration write and replaces the value — only the entry
    payload does this. A `BudgetDelta` is a spend report and sums into the
    counters, so two branches reporting in one super-step merge rather than
    raising InvalidUpdateError.

    The two update types exist because one cannot serve both writes: taking
    limits from the right lets a branch's defaults clobber the configured run
    profile, taking them from the left means the entry payload's limits never
    land.

    Either side may arrive as a plain dict, and both really do. A served graph
    (12) receives its input over HTTP, so the run profile reaches this reducer
    as JSON; and a value revived from a checkpointer that was not told about
    these models comes back the same way. Left as a dict it flows into state and
    the run dies at the first fan-out, where `dispatch` calls
    `budget.exhausted()`.

    The two shapes share no field names, so a dict is unambiguous — which is
    what makes coercing one safe. Guessing would not be: a delta read as
    configuration resets the run's limits, and configuration read as a delta
    adds nothing and loses them.
    """
    left = _budget(left)
    delta = _delta(right)
    if delta is None:
        return _budget(right)

    return left.model_copy(update={
        "searches_used": left.searches_used + delta.searches,
        "fetches_used": left.fetches_used + delta.fetches,
        "tokens_used": left.tokens_used + delta.tokens,
    })


def _budget(value) -> Budget:
    """A dict of limits is validated rather than trusted: an unrecognised key is
    a misconfigured run, and failing loudly at the first write beats a run given
    default limits nobody chose."""
    return value if isinstance(value, Budget) else Budget(**value)


def _delta(value) -> BudgetDelta | None:
    """The spend report in `value`, or None if it is a configuration write."""
    if isinstance(value, BudgetDelta):
        return value
    if isinstance(value, dict) and set(value) <= set(BudgetDelta.model_fields):
        return BudgetDelta(**value)
    return None


# --- graph state ------------------------------------------------------------

class ResearchState(TypedDict, total=False):
    # --- chat channel (see 12-interface.md) ---
    messages: Annotated[list, add_messages]      # human-readable projection of `trace`

    # --- input / clarification ---
    question: str
    clarifications: Annotated[list[dict], add]
    clarified: bool                              # latch; caps the clarify loop (02)
    _assumption: str                             # internal: proceed on this if unanswered

    # --- plan ---
    brief: str
    topics: Annotated[list[Topic], merge_topics]

    # --- accumulating evidence (written by PARALLEL branches -> reducers required) ---
    sources: Annotated[list[Source], add]
    claims: Annotated[list[Claim], add]
    uncertainties: Annotated[list[Uncertainty], add]

    # --- control ---
    round: int
    budget: Annotated[Budget, merge_budget]
    trace: Annotated[list[dict], add]            # progress events for the UI

    # --- output ---
    report: str


# --- serialization ----------------------------------------------------------

STATE_MODELS = (Source, Quote, Claim, Topic, Uncertainty, Budget, BudgetDelta)
"""Every model that reaches a checkpoint.

`JsonPlusSerializer` revives by module path. A type it was not told about is
permitted with a warning today and blocked once `LANGGRAPH_STRICT_MSGPACK`
becomes the default, which would hand a resumed run plain dicts where the
sufficiency router expects to call `budget.exhausted()`. Whoever builds the
checkpointer registers these:

    InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS))

Pass them to the constructor, not to `with_msgpack_allowlist` — that method
short-circuits while the default allowlist is permissive, so it registers
nothing until strict mode is already on. LangChain message types are allowed
regardless; only our own models need naming.

The list lives here so it cannot drift from the models it names.
"""


# --- trace events -----------------------------------------------------------

def ev(kind: str, **fields) -> dict:
    """One shape for every progress event. The eval harness (11) and the
    terminal renderer both read `trace`; the chat UI (12) reads `messages`,
    which is derived from these same events."""
    return {"kind": kind, "ts": time.time(), **fields}

# ev("search", topic_id=..., query=..., n_results=3)
# ev("fetch", url=..., status="ok", chars=8123)
# ev("claim", claim_id=..., text=...)
# ev("verdict", claim_id=..., verdict="unsupported", reason=...)
