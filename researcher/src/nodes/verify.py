"""The mandatory verification node — spec 07.

Everything else in this repo is a competent pipeline. This is the component that
turns grounding from a description of typical behaviour into a property of the
system, and it does so by *placement*: `verify` sits on the only edge between
research and synthesis (see `graph.py`). There is no configuration, prompt or
model decision that routes around it.

That placement is the whole argument. Handed a `verify_claim` tool instead, the
model would call it *sometimes* — and the claims it skips are the ones it is
confidently wrong about. Sampling keeps run-to-run variance high even when mean
accuracy looks fine; as a node the decision leaves the model entirely and that
variance goes to zero. The eval (11) reports Pass@1 alongside Pass^3 so the
consistency cost of ever changing this is visible rather than inferred.

Two tiers, cheap one first:

1. **Quote grounding** — deterministic, free, no model. Do the offsets the
   citation names still hold the quote it claims?
2. **Claim entailment** — does that verbatim quote support the claim? Only runs
   on quotes that passed tier 1, so token spend scales with real evidence rather
   than with model enthusiasm.
"""
from __future__ import annotations

import asyncio

from langgraph.config import get_stream_writer
from langgraph.types import Overwrite

from anchor import anchor
from content_store import ContentStore
from state import Claim, Quote, ResearchState, ev
from verifiers import Verifier

VERIFY_NODE = "verify"


async def verify_node(state: ResearchState, *, store: ContentStore,
                      verifier: Verifier) -> dict:
    """Verify every claim in state, concurrently, and replace them."""
    writer = _stream_writer()

    async def resolve(claim: Claim) -> Claim:
        verified = await verify_claim(claim, store=store, verifier=verifier)
        # Emitted the moment this claim resolves rather than with the node's
        # return: a progress indicator that arrives all at once is not progress.
        writer(_verdict(verified))
        return verified

    claims = state.get("claims") or []
    # Spec 09 runs this node once per round over the *accumulated* claim list.
    # A claim the previous round already settled keeps its verdict: paying to
    # re-entail it would make cost quadratic in rounds, and a verifier that
    # answered differently the second time would put back exactly the
    # run-to-run variance this node exists to remove.
    settled = [c for c in claims if c.verdict != "unverified"]
    verified = await asyncio.gather(
        *(resolve(c) for c in claims if c.verdict == "unverified"))

    return {
        # `claims` carries an `add` reducer for the fan-in, so a plain return
        # would leave state holding the unverified *and* the verified copy of
        # every claim — and synthesis would cite whichever it reached first.
        "claims": Overwrite(settled + list(verified)),
        "trace": [_verdict(c) for c in verified],
    }


async def verify_claim(claim: Claim, *, store: ContentStore,
                       verifier: Verifier) -> Claim:
    """Settle one claim. Never mutates the claim it was handed."""
    # Spec 06 drops quoteless claims before they leave extraction. Defence in
    # depth: a claim arriving from anywhere else must not be entailed against
    # nothing, and "no evidence" is not a supported claim.
    if not claim.quotes:
        return claim.model_copy(update={
            "verdict": "quote_not_found",
            "verdict_reason": "no anchored quote",
        })

    # Tier 1 over *every* quote, not just the first. It costs nothing, and a
    # claim carrying one bad quote and one good one still has real evidence.
    grounded = next((q for q in claim.quotes if _grounded(q, store)), None)
    if grounded is None:
        return claim.model_copy(update={
            "verdict": "quote_not_found",
            "verdict_reason": "quote does not match source at offset",
            "verified_by": "quote_match",
        })

    verdict = await verifier.check(claim.text, grounded.text)
    return claim.model_copy(update={
        # The grounded quote leads, so synthesis (08) shows the evidence this
        # verdict was reached against rather than whichever quote the extractor
        # happened to write down first.
        "quotes": [grounded] + [q for q in claim.quotes if q is not grounded],
        "verdict": verdict.verdict,
        "verdict_reason": verdict.reason,
        "verified_by": verifier.name,
    })


def _grounded(quote: Quote, store: ContentStore) -> bool:
    """Does the span this citation names still hold the quote it claims?

    Anchored *within the cited span* rather than across the document, so a quote
    that appears somewhere in the source but not where the citation points fails
    — a citation to the wrong place is not a citation.

    Re-anchoring rather than comparing strings is what keeps this consistent with
    spec 06: its offsets legitimately span text differing from `quote.text` by a
    newline or a capital, and byte equality would call every one of those a
    fabrication.

    A missing document raises rather than returning False. "We lost the page" is
    an infrastructure failure, and reporting it as `quote_not_found` would be
    this pipeline lying about grounding — the one thing it exists not to do.
    """
    return anchor(quote.text, store.get(quote.source_id)[quote.start:quote.end]) is not None


def _verdict(claim: Claim) -> dict:
    """One event shape, whether the UI (12) is replaying `trace` or following a
    live run. `verified_by` rides along so the eval can attribute the call, and
    the claim text so the interface can show *what* was settled — an id is not
    something a human can read a verdict about."""
    return ev("verdict", claim_id=claim.id, text=claim.text, verdict=claim.verdict,
              reason=claim.verdict_reason, verified_by=claim.verified_by)


def _stream_writer():
    """The custom-stream writer, or a no-op when this node is called outside a
    graph run — which is how the eval harness (11) drives a single node."""
    try:
        return get_stream_writer()
    except RuntimeError:
        return lambda _: None
