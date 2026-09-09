# 07 — Verification

**Priority: P0.** Everything else is a competent pipeline; this is the component
that makes grounding a guarantee rather than an aspiration.

## Purpose

Decide, for every claim, whether its cited quote actually supports it — and make
that decision unskippable:

> **Every claim traverses the verify node.** There is no configuration, prompt, or
> model decision that routes around it. Enforced structure is the only thing that
> turns "claims are checked" from a description of typical behaviour into a
> property of the system.

## Two tiers

Verification is two questions, not one, and separating them is what keeps it cheap.

**Tier 1 — quote grounding.** Does the quote exist verbatim in the source?
Deterministic, free, already computed by the anchoring step in spec 06. Catches
fabricated quotes outright.

**Tier 2 — claim entailment.** Given that verbatim quote, does it support the
claim? Only runs on quotes that passed tier 1, so token spend scales with real
evidence rather than model enthusiasm.

```python
# verify/verifier.py
from pydantic import BaseModel, Field
from typing import Literal


class Verdict(BaseModel):
    verdict: Literal["supported", "unsupported"]
    reason: str = Field(description="One sentence. Cite the specific mismatch if unsupported.")


VERIFY_PROMPT = """Does the evidence support the claim?

CLAIM: {claim}

EVIDENCE (verbatim from {url}):
\"\"\"{quote}\"\"\"

Answer "supported" only if the evidence states or directly entails the claim.
Answer "unsupported" if the evidence is merely related, if the claim generalises
beyond it, if numbers or qualifiers differ, or if it addresses a different subject.

Being related is not being supported. Default to "unsupported" when unsure."""


async def verify_claim(claim: Claim, store: ContentStore, llm) -> Claim:
    if not claim.quotes:
        return claim.model_copy(update={
            "verdict": "quote_not_found", "verdict_reason": "no anchored quote"})

    q = claim.quotes[0]
    # Tier 1 — deterministic re-check. Cheap insurance against a bad anchor.
    if store.get(q.source_id)[q.start:q.end].strip() != q.text.strip():
        return claim.model_copy(update={
            "verdict": "quote_not_found",
            "verdict_reason": "quote does not match source at offset",
            "verified_by": "quote_match"})

    # Tier 2 — entailment.
    v = await llm.with_structured_output(Verdict).ainvoke(
        VERIFY_PROMPT.format(claim=claim.text, quote=q.text, url=q.source_id))
    return claim.model_copy(update={
        "verdict": v.verdict, "verdict_reason": v.reason,
        "verified_by": f"llm:{llm.model_name}"})
```

## Placement in the graph

`verify` sits on the only edge between research and synthesis. Every claim produced
by the fan-in passes through it; nothing reaches `synthesize` that has not.

```python
# graph.py
def build_graph(**deps):
    b = StateGraph(ResearchState)
    b.add_node("clarify", clarify_node)                # tool loop, spec 02
    b.add_node("plan", plan_node)
    b.add_node("research_topic", research_topic_node)
    b.add_node("verify", verify_node)                  # MANDATORY: on the path
    b.add_node("sufficiency", sufficiency_node)
    b.add_node("synthesize", synthesize_node)

    b.add_edge(START, "clarify")
    b.add_edge("clarify", "plan")
    b.add_conditional_edges("plan", dispatch, ["research_topic"])
    b.add_edge("research_topic", "verify")
    b.add_edge("verify", "sufficiency")
    b.add_conditional_edges("sufficiency", route_after_sufficiency,
                            ["research_topic", "synthesize"])
    b.add_edge("synthesize", END)
    return b.compile(checkpointer=InMemorySaver())
```

The `verify` node runs over every claim in state:

```python
async def verify_node(state, *, store, llm):
    verified = await asyncio.gather(*[
        verify_claim(c, store, llm) for c in state["claims"]])
    return {
        "claims": Overwrite(verified),        # replace, don't append
        "trace": [ev("verdict", claim_id=c.id, verdict=c.verdict,
                     reason=c.verdict_reason) for c in verified],
    }
```

Note `Overwrite` — `claims` has an `add` reducer for the fan-in, so a plain return
would append duplicates. LangGraph 1.x provides `Overwrite` to bypass a reducer
for a single update.

## Why a node, not a tool

The obvious alternative is to hand the extraction agent a `verify_claim` tool and a
prompt telling it to check claims it is unsure about. That is rejected, and the
reason is mechanical rather than stylistic.

With a tool, whether a given claim gets verified is a *sampled decision* — the model
calls it sometimes. Sampling is a per-run coin flip, so run-to-run variance stays
high even when the mean looks acceptable, and the claims most likely to be skipped
are the ones the model is confidently wrong about. As a node, the decision is
removed from the model entirely and that source of variance goes to zero.

This is the gap Pass@k and Pass^k are built to separate. τ-bench (Yao et al., 2024)
introduced `pass^k` — the probability of succeeding on *all* k attempts — and found
agents that look adequate on mean accuracy collapse on consistency: GPT-4o at ~61%
on τ-retail single-run falls to roughly 25% at `pass^8`. Since `pass^k = p^k`, a
90% agent is ~73% at k=3. The eval in spec 11 reports both, so the consistency cost
of any future change to this node is visible rather than inferred.

**The honest caveat.** Enforced structure amplifies whatever the agent decided:
a mandatory node that verifies a claim against a badly-chosen source makes the
agent reliably wrong instead of occasionally wrong. Consistency is not correctness,
which is why Pass@1 is reported alongside Pass^3 rather than replaced by it.

## On using an LLM as the verifier

It is the weak link and the README should say so.

LLM-as-judge has documented position, verbosity, and self-preference biases. On
attribution specifically, NLI-fine-tuned models match or beat much larger general
LLMs — ALCE (Gao et al., EMNLP 2023) computes citation precision/recall with the
TRUE NLI model and reports ~85% agreement with human labels on recall, ~78% on
precision.

Mitigations, in order of cost:
1. **Report verifier-vs-human agreement** on the labeled eval set (spec 11). If
   agreement is under ~80%, the automated numbers don't carry weight and you say
   so.
2. **Use a different model family for verification than for extraction** —
   removes self-preference.
3. **Swap in an NLI cross-encoder** (`cross-encoder/nli-deberta-v3-base`, ~180MB,
   runs on CPU). This is the correct answer and the first planned upgrade; the
   only reason it isn't the default is the model download in the setup path.

The verifier interface is deliberately swappable so this is a one-line change:

```python
class Verifier(Protocol):
    async def check(self, claim: str, quote: str) -> tuple[str, str]: ...
```

## Downstream contract

Unverified claims never reach the report as citations. Synthesis (spec 08) filters
on `verdict == "supported"` and renders the rest in an explicit limitations
section. "Not supported" is a first-class outcome, not a failure to hide.

## Acceptance

- No path through the graph reaches `synthesize` without passing `verify`.
- Every claim in state carries a non-`unverified` verdict.
- A fabricated quote yields `quote_not_found` without an LLM call.
- Verdicts stream to the UI as they resolve.
- The verifier can be swapped without touching the graph.
