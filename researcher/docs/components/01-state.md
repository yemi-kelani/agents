# 01 — State Schema & Content Store

**Priority: P0.** Everything else depends on these types. Build first.

## Why this is its own spec

Two LangGraph failure modes bite here and both are silent until they aren't:

1. **Parallel branches writing the same un-annotated key raise
   `InvalidUpdateError`.** The fan-out over topics means several nodes return
   `{"findings": [...]}` in the same super-step. Without a reducer, LangGraph
   cannot merge them and the run dies.
2. **State is serialized on every checkpoint write.** Putting fetched page text
   in state makes each checkpoint carry every document fetched so far. Quadratic
   growth, visibly slow demo.

## Core types

```python
# state.py
from __future__ import annotations
from operator import add
from typing import Annotated, Literal, TypedDict
from pydantic import BaseModel, Field


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
```

## Graph state

```python
class ResearchState(TypedDict, total=False):
    # --- chat channel (see 12-interface.md) ---
    messages: Annotated[list, add_messages]      # human-readable projection of `trace`

    # --- input / clarification ---
    question: str
    clarifications: Annotated[list[dict], add]
    clarified: bool

    # --- plan ---
    brief: str
    topics: Annotated[list[Topic], merge_topics] # merged by id; see below

    # --- accumulating evidence (written by PARALLEL branches -> reducers required) ---
    sources: Annotated[list[Source], add]
    claims: Annotated[list[Claim], add]
    uncertainties: Annotated[list[Uncertainty], add]

    # --- control ---
    round: int
    budget: Annotated[Budget, merge_budget]      # see below
    trace: Annotated[list[dict], add]            # progress events for the UI

    # --- output ---
    report: str
```

`Annotated[..., add]` uses `operator.add`, which concatenates lists. Anything a
fan-out branch writes needs it. Scalars (`round`, `brief`, `report`) are
overwrite-by-default and are only written by single nodes — that is fine and
intentional.

**Note on `topics`:** appending is safe under `add`; mutating status is not,
because two branches could both rewrite the list. The alternative — a separate
`Annotated[list[str], add]` of researched ids with status derived from it — keeps
two things in sync that want to be one. A reducer that merges by `id` is
cheaper:

```python
def merge_topics(left: list[Topic], right: list[Topic]) -> list[Topic]:
    by_id = {t.id: t for t in left}
    for t in right:
        by_id[t.id] = t          # right wins on conflict
    return list(by_id.values())

topics: Annotated[list[Topic], merge_topics]
```

## Budget — the authoritative stop

```python
class Budget(BaseModel):
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
```

The LLM sufficiency check *proposes*; the budget *decides*. Any design where a
model can vote itself more iterations has no upper bound. See 09.

### Two update types, because one cannot serve both writes

`budget` is written from inside the fan-out — every search, fetch and model call
in a parallel branch spends against it — so it needs a reducer for exactly the
reason `claims` does. But it takes two *kinds* of write: the entry payload sets
limits (`FAST`/`FULL` in 12), branches report spend. A single type makes them
indistinguishable, and either resolution is wrong: take limits from the right and
a branch's defaults clobber the configured run profile; take them from the left
and the entry payload's limits never land.

So spend has its own type, and the ambiguity stops being representable:

```python
class BudgetDelta(BaseModel):
    """What one node reports it spent. Never carries limits."""
    searches: int = 0
    fetches: int = 0
    tokens: int = 0


def merge_budget(left: Budget, right: Budget | BudgetDelta) -> Budget:
    if isinstance(right, Budget):
        return right                 # configuration write; only the entry payload
    return left.model_copy(update={  # spend report; sums under fan-out
        "searches_used": left.searches_used + right.searches,
        "fetches_used": left.fetches_used + right.fetches,
        "tokens_used": left.tokens_used + right.tokens,
    })
```

`Budget` keeps both limits and counters, so `Budgeted` (10) and the sufficiency
router (09) read it unchanged. Rejected: one `Budget` type with `Overwrite(FAST)`
for the config write. It works, but a plain `invoke({"budget": FAST})` is then
silently read as a delta and the run keeps default limits.

**`max_total_tokens` is cumulative spend, not a context window.** `tokens_used`
sums `total_tokens` over *every* model call in the run — planner, each extractor,
each verifier, synthesizer, across every round (the `Budgeted` wrapper in 10 is
what increments it). It is the dial to lower for a cheaper run. It is never the
size of a single prompt. Whether one call *fits* in a given model's window is a
separate constraint, configured per role in 10.

Keeping it model-independent is deliberate. The local-vs-frontier comparison in
10 only measures model quality if both profiles run under an identical budget.
Scale the budget to the model's context window and the frontier profile simply
gets more research — the table would then be measuring budget, not the model.

## Content store

Out-of-state storage for fetched text. In-memory dict for the demo, trivially
swappable for disk or Redis.

```python
# content_store.py
import hashlib

class ContentStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    @staticmethod
    def source_id(url: str) -> str:
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def put(self, source_id: str, text: str) -> None:
        self._data[source_id] = text

    def get(self, source_id: str) -> str:
        return self._data[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._data

    def __len__(self) -> int:
        return len(self._data)
```

`get` raises `KeyError` on an unknown id rather than returning `""` — a missing
document is a bug, not an empty page to extract claims from.

Pass it via `context_schema` / a closure over the node functions rather than
through state. Deriving `source_id` from the URL hash also gives free
deduplication: the same URL found by two topics resolves to one entry.

No `Protocol` in front of it. "Trivially swappable for disk or Redis" is a
property of the four-method surface, not something a one-implementation
interface adds.

## Serialization

The checkpointer revives state models by module path, and `JsonPlusSerializer`
allows a type it was not told about only with a warning — blocked once
`LANGGRAPH_STRICT_MSGPACK` becomes the default. An unregistered model comes back
as a plain dict, which is the moment `budget.exhausted()` in the 09 router stops
being a method call. So the models are named in one place:

```python
STATE_MODELS = (Source, Quote, Claim, Topic, Uncertainty, Budget, BudgetDelta)

InMemorySaver(serde=JsonPlusSerializer(allowed_msgpack_modules=STATE_MODELS))
```

Pass them to the constructor. `JsonPlusSerializer().with_msgpack_allowlist(...)`
looks equivalent and is not: it returns `self` unchanged while the default
allowlist is permissive, so it registers nothing until strict mode is already on.
LangChain message types are allowed either way; only our own models need naming.

## Trace events

The UI (spec 12) and the eval (spec 11) both read this. One shape:

```python
def ev(kind: str, **fields) -> dict:
    return {"kind": kind, "ts": time.time(), **fields}

# ev("search", topic_id=..., query=..., n_results=3)
# ev("fetch", url=..., status="ok", chars=8123)
# ev("claim", claim_id=..., text=...)
# ev("verdict", claim_id=..., verdict="unsupported", reason=...)
```

`verdict` events are the ones worth watching live — a rejection rate that spikes
mid-run usually means a bad extraction prompt or a backend returning junk.

`trace` is the machine-readable channel: the eval harness (spec 11) and the
terminal renderer both read it. The chat interface reads `messages` instead, which
spec 12 derives from these same events. Two channels, one source — so the UI
choice stays out of the graph.

## Acceptance

- Two nodes returning `{"claims": [...]}` in the same super-step merge without error.
- A checkpoint's serialized size is independent of total fetched bytes.
- The same URL discovered by two topics yields one `Source`.
- Two nodes reporting `BudgetDelta` in the same super-step sum, and neither
  overwrites the configured limits.
- A checkpointed run resumes with `Claim`/`Budget` as objects, not dicts —
  under `LANGGRAPH_STRICT_MSGPACK` as well as the default.
