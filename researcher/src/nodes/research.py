"""The research-topic flow — spec 06.

search -> fetch -> extract, for one topic. This is the unit `Send` fans out over
(spec 03): the payload replaces the state the node sees, so everything the flow
needs arrives in it and everything the flow learned leaves through the spec 01
reducers.

Extraction is where citations become checkable or don't. The model returns quote
*text*; `anchor` turns that into offsets into the stored source. A quote that
anchors nowhere is a fabricated quote, and it is reported rather than dropped
quietly — that report is the most useful signal this pipeline produces.

Nothing here decides whether a claim is *true*. Claims leave with
`verdict="unverified"`; the mandatory verification node (07) is what settles it,
and a node that pre-empted it would make that guarantee decorative.
"""
from __future__ import annotations

from datetime import datetime, timezone

import httpx
from pydantic import BaseModel, Field

from anchor import anchor
from content_store import ContentStore
from fetch import BlockedURL, UNTRUSTED_PREAMBLE, datamark, fetch_clean_text, undatamark
from nodes.plan import TopicTask
from prompts import load
from search.authority import authority
from state import BudgetDelta, Claim, Quote, Source, Uncertainty, ev

SEARCH_K = 4
MAX_SOURCES_PER_TOPIC = 3
"""More hits than sources on purpose: the margin is what a dead link costs, so a
fetch that fails gives up its slot to the next hit instead of costing the topic
a third of its evidence."""

MAX_CLAIMS = 6
MAX_QUOTES = 2

MAX_EXTRACT_CHARS = 12_000
"""The cut when no context limit is configured — conservative enough for any
provider. Spec 10 derives it from `context_limits[extractor]` instead, which is
the number that has to grow when map-reduce chunking replaces the truncation.

A truncation, not a chunking strategy. Known limitation: the tail of a long
document is ignored. The real fix is map-reduce over chunks with per-chunk
offsets rebased into the full document — offsets are anchored against the whole
stored text here, so that change stays local to this node."""


class ExtractedQuote(BaseModel):
    text: str = Field(
        description="EXACT substring copied from the source. Do not paraphrase, "
                    "fix typos, or add ellipses."
    )


class ExtractedClaim(BaseModel):
    """`min_length=1` is the structural half of "every claim carries a quote".
    The prompt asks; this makes an unquoted claim unrepresentable."""
    text: str = Field(description="One atomic assertion, in your own words.")
    quotes: list[ExtractedQuote] = Field(min_length=1, max_length=MAX_QUOTES)


class Extraction(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=MAX_CLAIMS)
    uncertainties: list[str] = Field(default_factory=list)


async def research_topic_node(
    task: TopicTask,
    *,
    router,
    store: ContentStore,
    llm,
    fetch=fetch_clean_text,
    max_chars: int = MAX_EXTRACT_CHARS,
) -> dict:
    """Research one topic and hand back what it found.

    `fetch` is injected rather than imported at the call site so the eval
    harness (11) can replay a fixed corpus without a network.
    """
    topic = task["topic"]
    hits = await router.search(topic.question, SEARCH_K)
    trace = [ev("search", topic_id=topic.id, query=topic.question, n_results=len(hits))]

    sources: list[Source] = []
    claims: list[Claim] = []
    uncertainties: list[Uncertainty] = []
    fetches = 0

    for hit in hits:
        if len(sources) >= MAX_SOURCES_PER_TOPIC:
            break

        source_id = ContentStore.source_id(hit.url)
        if source_id not in store:
            # LLM-optimized backends return clean page text with the result;
            # SERP-style backends don't. Fetching anyway pays twice for a page
            # the search already read.
            if hit.content:
                store.put(source_id, hit.content)
            else:
                # Counted before the call: a request that failed still cost one,
                # and a budget that only counted successes would let a domain of
                # dead links run forever.
                fetches += 1
                try:
                    text = await fetch(hit.url)
                except (BlockedURL, httpx.HTTPError) as e:
                    trace.append(ev("fetch_failed", url=hit.url, error=type(e).__name__))
                    continue
                store.put(source_id, text)
                trace.append(ev("fetch", url=hit.url, status="ok", chars=len(text)))

        text = store.get(source_id)
        if not text:
            # `to_clean_text` answers "" for a page with no article in it.
            # Handing a model an empty document is how it gets asked to invent.
            trace.append(ev("fetch", url=hit.url, status="empty", chars=0))
            continue

        sources.append(Source(
            source_id=source_id, url=hit.url, title=hit.title, backend=hit.backend,
            fetched_at=datetime.now(timezone.utc).isoformat(),
            char_count=len(text), authority=authority(hit.url),
        ))

        extraction = await llm.with_structured_output(Extraction).ainvoke(
            load("extract").format(
                topic=topic.question,
                max_claims=MAX_CLAIMS,
                untrusted_block=UNTRUSTED_PREAMBLE + "\n\n"
                                + datamark(text[:max_chars]),
            )
        )

        for extracted in extraction.claims:
            # Assigned before the claim is kept so a fuzzy anchor can name it.
            # A claim that ends up dropped anchored nothing, so it emits no
            # event carrying this id and the next claim reuses it cleanly.
            claim_id = f"{topic.id}-c{len(claims)}"
            quotes = _anchor_quotes(extracted, source_id, text, claim_id, trace)

            if not quotes:
                trace.append(ev("claim_dropped", topic_id=topic.id,
                                text=extracted.text[:120], reason="no_anchored_quote"))
                continue

            claims.append(Claim(id=claim_id, topic_id=topic.id,
                                text=extracted.text, quotes=quotes))
            trace.append(ev("claim", claim_id=claim_id, text=extracted.text[:120]))

        uncertainties += [
            Uncertainty(topic_id=topic.id, description=u, kind="unconfirmed")
            for u in extraction.uncertainties
        ]

    if not sources:
        # The router answers [] when every backend failed or found nothing, and
        # a topic whose every source died reads the same from here. Spec 09 uses
        # this to decide whether a second round has anything concrete to chase.
        uncertainties.append(Uncertainty(
            topic_id=topic.id,
            description=f"No source could be read for: {topic.question}",
            kind="no_results",
        ))

    return {
        "sources": sources,
        "claims": claims,
        "uncertainties": uncertainties,
        # A copy, never a mutation: `merge_topics` merges by id, and flipping
        # the status of the Topic this node was handed would also flip the one
        # still sitting in state.
        "topics": [topic.model_copy(
            update={"status": "researched" if sources else "failed"}
        )],
        "trace": trace,
        "budget": BudgetDelta(searches=1, fetches=fetches),
    }


def _anchor_quotes(
    extracted: ExtractedClaim,
    source_id: str,
    text: str,
    claim_id: str,
    trace: list[dict],
) -> list[Quote]:
    """Turn the quotes the model wrote into spans into `text`, dropping the ones
    that are not in there. Appends to `trace`, which is the point: a quote that
    anchored nowhere has to be loud, and a quote that only anchored fuzzily has
    to be countable (spec 11 reports the fuzzy rate separately)."""
    quotes: list[Quote] = []

    for q in extracted.quotes:
        # Spotlighting replaced every space in the block the model read, so a
        # faithfully copied quote comes back marked.
        quote_text = undatamark(q.text)
        span = anchor(quote_text, text)

        if span is None:
            trace.append(ev("quote_not_found", claim=extracted.text[:80],
                            quote=quote_text[:80]))
            continue

        if span.method == "fuzzy":
            trace.append(ev("quote_fuzzy", claim_id=claim_id, quote=quote_text[:80]))

        quotes.append(Quote(source_id=source_id, text=quote_text,
                            start=span.start, end=span.end))

    return quotes
