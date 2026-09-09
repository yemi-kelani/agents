# 06 — Claim & Quote Extraction

**Priority: P0.** This is where citations become checkable — or don't.

## Purpose

Turn fetched text into **atomic claims**, each anchored to one or more **verbatim
quotes** with character offsets into the stored source text.

The offsets are the whole point. A citation that says "source 3" is unverifiable;
a citation that says "source 3, chars 1840–1955, text: '...'" can be checked by
`source_text[1840:1955] == quote.text` with no model in the loop.

## Atomicity

One claim = one assertion that can be independently true or false.

Bad: *"Mimir scales to over a billion active series and is used by large
enterprises."* — two assertions; if one fails the verdict is meaningless.

Good, as two claims: *"Mimir scales to over 1 billion active series."* and
*"Mimir is deployed by large enterprises."*

This mirrors FactScore's decomposition into atomic facts (Min et al., EMNLP 2023),
which exists for the same reason: you cannot score a compound sentence.

## Extraction

```python
# nodes/extract.py
from pydantic import BaseModel, Field


class ExtractedQuote(BaseModel):
    text: str = Field(description="EXACT substring copied from the source. Do not paraphrase, fix typos, or add ellipses.")


class ExtractedClaim(BaseModel):
    text: str = Field(description="One atomic assertion, in your own words.")
    quotes: list[ExtractedQuote] = Field(min_length=1, max_length=2)


class Extraction(BaseModel):
    claims: list[ExtractedClaim] = Field(default_factory=list, max_length=6)
    uncertainties: list[str] = Field(default_factory=list)


EXTRACT_PROMPT = """Extract claims relevant to this sub-question from the source below.

Sub-question: {topic}

Rules:
- Each claim is ONE atomic assertion.
- Each claim carries at least one quote copied EXACTLY from the source —
  character for character. Do not paraphrase, trim, or normalise the quote.
- Only extract what the source actually says. If the source does not address the
  sub-question, return no claims.
- Record uncertainties separately: things the source hedges, contradicts, or
  leaves open.

Return at most 6 claims. Fewer good claims beats more weak ones.

{untrusted_block}"""
```

Two prompt details that matter more than they look:

- **"copied EXACTLY"** — repeat it in the field description as well as the prompt.
  Models paraphrase quotes by default, and a paraphrased quote fails offset
  matching, which is indistinguishable from a fabricated one downstream.
- **"return no claims"** — models will manufacture relevance from an off-topic
  page unless explicitly told the empty set is a valid answer.

## Anchoring — finding the offsets

The model returns quote *text*. This code turns it into offsets, and this is where
fabricated quotes die.

```python
# extract/anchor.py
import re
from difflib import SequenceMatcher

WS = re.compile(r"\s+")


def _norm(s: str) -> str:
    return WS.sub(" ", s).strip().lower()


def anchor(quote_text: str, source_text: str, *, fuzzy_threshold: float = 0.92
           ) -> tuple[int, int] | None:
    """Locate quote_text in source_text. Returns (start, end) or None."""
    # 1. exact
    idx = source_text.find(quote_text)
    if idx != -1:
        return idx, idx + len(quote_text)

    # 2. whitespace/case-insensitive over a normalised copy
    nq, ns = _norm(quote_text), _norm(source_text)
    idx = ns.find(nq)
    if idx != -1:
        return _map_back(idx, len(nq), source_text)

    # 3. fuzzy — catches trivial model edits, but flag it
    best = _best_window(nq, ns, len(nq))
    if best and best.ratio >= fuzzy_threshold:
        return _map_back(best.start, len(nq), source_text)

    return None          # -> verdict = "quote_not_found"
```

Tier 3 is a judgment call. A model that silently "corrects" a typo produced a
quote that is *not* verbatim, and treating that as a match weakens the guarantee.
Recommendation: accept fuzzy matches but record `verified_by="quote_match:fuzzy"`
and report the fuzzy rate separately in the eval. If the fuzzy rate is high, the
extraction prompt is the problem, not the matcher.

A quote that anchors nowhere is a **fabricated quote**, and it is the single most
useful thing this pipeline detects. Log it loudly:

```python
if span is None:
    trace.append(ev("quote_not_found", claim=claim.text[:80], quote=q.text[:80]))
```

## The research-topic node

Assembles search → fetch → extract for one topic. This is the unit `Send` fans out
over (spec 03).

```python
async def research_topic_node(payload, *, router, store, llm, budget):
    topic = payload["topic"]
    hits = await router.search(topic.question, k=4)

    sources, claims, uncerts, trace = [], [], [], []
    for hit in hits[:3]:
        sid = ContentStore.source_id(hit.url)
        if sid not in store:
            try:
                text = hit.content or to_clean_text(await safe_fetch(hit.url))
            except (BlockedURL, httpx.HTTPError) as e:
                trace.append(ev("fetch_failed", url=hit.url, error=type(e).__name__))
                continue
            store.put(sid, text)

        text = store.get(sid)
        sources.append(Source(source_id=sid, url=hit.url, title=hit.title,
                              backend=hit.backend, char_count=len(text),
                              authority=authority(hit.url)))

        result = await llm.with_structured_output(Extraction).ainvoke(
            EXTRACT_PROMPT.format(
                topic=topic.question,
                untrusted_block=UNTRUSTED_PREAMBLE + "\n\n" + datamark(text[:12000]),
            )
        )

        for c in result.claims:
            quotes = []
            for q in c.quotes:
                span = anchor(q.text, text)
                if span is None:
                    trace.append(ev("quote_not_found", quote=q.text[:80]))
                    continue
                quotes.append(Quote(source_id=sid, text=q.text,
                                    start=span[0], end=span[1]))
            claims.append(Claim(id=new_id(), topic_id=topic.id,
                                text=c.text, quotes=quotes))

        uncerts += [Uncertainty(topic_id=topic.id, description=u, kind="unconfirmed")
                    for u in result.uncertainties]

    topic.status = "researched"
    return {"sources": sources, "claims": claims, "uncertainties": uncerts,
            "topics": [topic], "trace": trace}
```

Note `text[:12000]` — a truncation, not a chunking strategy. Known limitation:
long documents get their tail ignored. The real fix is map-reduce over chunks with
per-chunk offsets rebased into the full document.

## Uncertainties

Cheap to collect, and they are what makes the sufficiency check (spec 09) do
something better than introspect on its own report. A flow that reports "sources
disagreed on the threshold" gives round 2 a concrete reason to spawn a topic.

## Acceptance

- Every claim carries ≥1 quote or is dropped.
- `source_text[q.start:q.end]` reproduces `q.text` for exact matches.
- A hallucinated quote yields `quote_not_found`, not a silent pass.
- Off-topic sources produce zero claims rather than stretched ones.
