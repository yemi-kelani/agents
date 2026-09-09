"""Report generation — spec 08.

Synthesis does not get to introduce new facts. The model sees the verified
claims and their quotes and nothing else: no page text, no store, no fetch. That
is the privileged half of the split described in spec 05 — untrusted content
reaches extraction, quarantined and datamarked; the node that writes what the
human reads never touches it, so compromising the report requires an injection
that survives extraction *and* verification.

It is a weaker version of the dual-LLM / CaMeL pattern, and worth being clear
about how much weaker: a claim that passes both gates still carries the
adversary's framing. The prompt says so; the renderer strips images anyway.

The one decision here that is not in the spec's sketch: a run with nothing
citable does not reach the model at all. Handed a question and an empty evidence
block, a model writes the answer from its own weights and the result reads
exactly like a grounded report. The spec asks for that case in a prompt rule
("say that plainly"); a rule the model may ignore is not the guarantee this
repo is built on, so the empty case is answered by this module instead.
"""
from __future__ import annotations

from prompts import load
from render import citation, number_sources, render_bibliography, safe_render
from state import Claim, Quote, ResearchState, ev

SYNTHESIZE_NODE = "synthesize"

EVIDENCE_CHARS = 200
"""Enough to see what the quote says. The full span is in the source, and the
offsets that prove it are in the claim."""

NOTHING_CITABLE = (
    "No claim from this run is citable: nothing extracted survived "
    "verification. There is no grounded answer to give, and an ungrounded one "
    "would be worse than none."
)


def synthesize_node(state: ResearchState, *, llm, max_chars: int | None = None) -> dict:
    """Verified claims -> a report whose every citation was checked first.

    No `store` parameter, and that absence is the containment guarantee: a node
    that cannot reach the page text cannot smuggle an assertion out of it.

    `max_chars` is the synthesizer's window (spec 10). This is the one call in
    the pipeline that can actually overflow — it holds every verified claim at
    once, so its size grows with `max_topics x max_rounds`.
    """
    claims = state.get("claims") or []
    numbering = number_sources(state.get("sources") or [])

    cited = [(n, c, q) for c in claims if (cite := citation(c, numbering))
             for n, q in [cite]]
    rejected = [c for c in claims if c.verdict != "supported"]

    body = (_write(state, cited, rejected, llm=llm, max_chars=max_chars)
            if cited else _nothing_found(rejected))

    return {
        # Rendered once, over the model's prose *and* the bibliography: source
        # titles come from search results, so the half this repo wrote is as
        # untrusted as the half the model wrote.
        "report": safe_render(f"{body}\n\n{render_bibliography(numbering, claims)}"),
        "trace": [ev("report", n_supported=len(claims) - len(rejected),
                     n_rejected=len(rejected))],
    }


def _write(state: ResearchState, cited: list[tuple[int, Claim, Quote]],
           rejected: list[Claim], *, llm, max_chars: int | None = None) -> str:
    return llm.invoke(load("synthesize").format(
        question=state.get("question", ""),
        brief=state.get("brief", ""),
        claims_block=_claims_block(cited, max_chars),
        rejected_block=_rejected_block(rejected),
        uncertainty_block="\n".join(f"- {u.description}"
                                    for u in state.get("uncertainties") or []) or "none",
    )).text


def _nothing_found(rejected: list[Claim]) -> str:
    """The no-evidence report, written here rather than by the model.

    It still ends in Limitations naming what was discarded: on this path the
    rejected claims are the only thing the run has to say, and "we found these
    and could not stand them up" is a real answer.
    """
    return f"{NOTHING_CITABLE}\n\n## Limitations\n{_rejected_block(rejected)}"


def _claims_block(cited: list[tuple[int, Claim, Quote]],
                  max_chars: int | None = None) -> str:
    """Every citable claim, with the number the bibliography gave its source and
    the evidence verification checked it against.

    Compressed rather than truncated when it will not fit the window (spec 10):
    the evidence snippets go first, then the per-claim citation numbers collapse
    into one line per source. **No claim is ever dropped** — a report quietly
    built from less than the run verified is the failure this repo exists to
    avoid, and a call that fails is better than one that lies by omission.
    """
    block = "\n".join(f'[{n}] {c.text}\n    evidence: "{q.text[:EVIDENCE_CHARS]}"'
                      for n, c, q in cited)
    if max_chars is None or len(block) <= max_chars:
        return block

    # The claim is what the model writes from; the quote is what verification
    # already checked it against, and the model is not being asked to re-check.
    block = "\n".join(f"[{n}] {c.text}" for n, c, _ in cited)
    if len(block) <= max_chars:
        return block

    by_source: dict[int, list[str]] = {}
    for n, claim, _ in cited:
        by_source.setdefault(n, []).append(claim.text)
    return "\n".join(f"[{n}] " + "; ".join(texts)
                     for n, texts in sorted(by_source.items()))


def _rejected_block(rejected: list[Claim]) -> str:
    """What failed, and why. The model needs the reason to write Limitations,
    and a reader needs it more than they need the claim."""
    return "\n".join(f"- {c.text}  (rejected: {c.verdict} — {c.verdict_reason})"
                     for c in rejected) or "none"
