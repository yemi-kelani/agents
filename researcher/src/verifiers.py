"""Tier-2 entailment checkers — spec 07.

Does this verbatim quote actually *support* this claim? That is the question the
cheap deterministic tier cannot answer, and it is the weakest link in the whole
pipeline, so the interface is one method wide on purpose.

**The LLM is the weak link and this file should say so.** LLM-as-judge carries
documented position, verbosity and self-preference biases. On attribution
specifically, NLI-fine-tuned models match or beat much larger general LLMs — ALCE
(Gao et al., EMNLP 2023) computes citation precision and recall with the TRUE NLI
model at ~85%/78% agreement with human labels. A `cross-encoder/nli-deberta-v3-base`
verifier is the planned upgrade and the correct answer; the only reason it is not
the default is the model download in the setup path.

`Verifier` exists so that swap is one line at the call site and nothing in the
graph moves. `name` is not decoration: it lands in `Claim.verified_by`, which is
what lets the eval (11) report agreement per judge and what makes "verify with a
different model family than you extracted with" auditable rather than assumed.
"""
from __future__ import annotations

from typing import Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field

from prompts import load


class Verdict(BaseModel):
    """Binary on purpose. "Partially supported" is a real failure mode and a
    documented known unknown — collapsing it here undercounts it, and inventing
    a middle grade the eval cannot label would hide that instead."""
    verdict: Literal["supported", "unsupported"]
    reason: str = Field(
        description="One sentence. Cite the specific mismatch if unsupported."
    )


@runtime_checkable
class Verifier(Protocol):
    name: str
    """Lands in `Claim.verified_by`: "llm:gpt-4.1-mini", "nli:deberta"."""

    async def check(self, claim: str, quote: str) -> Verdict: ...


class LLMVerifier:
    """The default verifier: a structured-output call per grounded claim.

    The prompt is the whole implementation, which is the honest summary of how
    much this tier is worth.
    """

    def __init__(self, llm):
        self.name = f"llm:{_model_name(llm)}"
        # Bound once rather than per claim: `with_structured_output` rebuilds
        # the schema binding on every call, and this runs once per claim already.
        self._llm = llm.with_structured_output(Verdict)

    async def check(self, claim: str, quote: str) -> Verdict:
        return await self._llm.ainvoke(load("verify").format(claim=claim, quote=quote))


def _model_name(llm) -> str:
    """Whatever the provider calls its model. Falls back to the class name so a
    verdict is always attributable to *something* — an unattributable verdict is
    one the eval cannot report agreement for."""
    return (
        getattr(llm, "model_name", None)
        or getattr(llm, "model", None)
        or type(llm).__name__
    )
