"""The harness — spec 11.

One command, a fixed corpus, and k runs of the verifier over hand-labelled
claims. What it measures is narrow on purpose: whether the automated judge
agrees with a human about whether a quote supports a claim. That is the weakest
link in the pipeline (07) and the one number the rest of the design rests on.

Two things the sketch in the spec leaves out, both of which would quietly
produce wrong numbers:

**Gold quotes are anchored, not assumed.** A `Quote` built with `start=0` names
the first characters of the document, and tier 1 checks the span the citation
names — so every gold claim would come back `quote_not_found` and the harness
would report a broken verifier.

**A quote that no longer anchors is excluded and named.** Corpus drift is not a
verifier failure, and scoring it as one blames the judge for the page having
changed underneath the label — this pipeline's own thesis, turned on its own
harness.
"""
from __future__ import annotations

import argparse
import asyncio
import sys
from dataclasses import dataclass, field
from pathlib import Path

from anchor import anchor
from content_store import ContentStore
from eval.gold import CORPUS_DIR, DEFAULT_GOLD_PATH, GoldClaim, load_gold
from eval.metrics import Scores, agreement, score
from state import Claim, Quote

DEFAULT_K = 3


@dataclass(frozen=True)
class EvalResult:
    scores: Scores
    agreement: float
    unanchored: list[str]
    gold_path: Path | None
    """None when the caller handed over gold rows directly — there is no file to
    name, and no grounds to call it the shipped set."""

    verifier: str
    run_state: dict | None = field(default=None)
    """A finished pipeline run, if the caller has one. The secondary rates
    (`quote_not_found`, `fuzzy`) are properties of extraction, so they cannot be
    computed from a gold set — they need a run that actually extracted."""

    @property
    def is_shipped_gold(self) -> bool:
        return self.gold_path == DEFAULT_GOLD_PATH


async def build_store(gold: list[GoldClaim], *, fetch) -> ContentStore:
    """The corpus, read once and shared by every run.

    Shared because k runs of a *verifier* should cost k verifications, not k
    crawls — and because a corpus that changed between runs would show up as
    instability the verifier did not cause.
    """
    store = ContentStore()

    for g in gold:
        source_id = ContentStore.source_id(g.source_url)
        if source_id in store:
            continue
        # A shipped document travels with the row; a real gold set names a URL
        # and gets the cached fetch (10), which is what makes the second run of
        # the day deterministic and free.
        text = ((CORPUS_DIR / g.corpus).read_text(encoding="utf-8")
                if g.corpus else await fetch(g.source_url))
        store.put(source_id, text)

    return store


async def run_eval(*, verifier, k: int = DEFAULT_K,
                   gold: list[GoldClaim] | None = None,
                   gold_path: str | Path | None = None,
                   fetch=None, run_state: dict | None = None) -> EvalResult:
    """Verify every gold claim k times and score the agreement."""
    from nodes.verify import verify_claim

    # A caller who handed over rows directly gets no path: naming the default
    # would report their set as the shipped one and warn about the wrong thing.
    path = Path(gold_path) if gold_path else (None if gold is not None
                                              else DEFAULT_GOLD_PATH)
    gold = gold if gold is not None else load_gold(path)
    store = await build_store(gold, fetch=fetch or _no_fetch)

    scorable, unanchored = _anchor_gold(gold, store)
    per_claim: dict[str, list[bool]] = {g.claim_id: [] for g, _ in scorable}

    for _ in range(k):
        for g, claim in scorable:
            settled = await verify_claim(claim, store=store, verifier=verifier)
            # The verifier answers in three verdicts and the labels in two: a
            # fabricated quote is not support, so it collapses rather than being
            # scored as an outcome the gold set cannot express.
            predicted = "supported" if settled.verdict == "supported" else "unsupported"
            per_claim[g.claim_id].append(predicted == g.label)

    return EvalResult(
        scores=score(per_claim, k),
        agreement=agreement(per_claim),
        unanchored=unanchored,
        gold_path=path,
        verifier=getattr(verifier, "name", type(verifier).__name__),
        run_state=run_state,
    )


def _anchor_gold(gold: list[GoldClaim],
                 store: ContentStore) -> tuple[list[tuple[GoldClaim, Claim]], list[str]]:
    """Locate every gold quote in its source, and set aside the ones that moved."""
    scorable, unanchored = [], []

    for g in gold:
        source_id = ContentStore.source_id(g.source_url)
        span = anchor(g.quote, store.get(source_id))
        if span is None:
            unanchored.append(g.claim_id)
            continue

        scorable.append((g, Claim(
            id=g.claim_id, topic_id="eval", text=g.claim,
            quotes=[Quote(source_id=source_id, text=g.quote,
                          start=span.start, end=span.end)],
        )))

    return scorable, unanchored


async def _no_fetch(url: str) -> str:
    raise RuntimeError(
        f"gold row for {url} names no corpus file and no fetcher was given"
    )


def main(argv: list[str] | None = None, *, verifier=None, fetch=None) -> int:
    """One command: `researcher-eval`, or `python -m eval.run`.

    `verifier` and `fetch` are injected for the same reason every other seam in
    this repo is — a test, and the harness's own reproducibility, must not need
    a provider key. Left out, the verifier is built from the config's verifier
    role (10), which the config keeps distinct from the extractor.
    """
    from eval.report import render

    args = _parser().parse_args(argv)
    result = asyncio.run(run_eval(
        verifier=verifier if verifier is not None else _configured_verifier(args.profile),
        k=args.k, gold_path=args.gold, fetch=fetch,
    ))
    print(render(result))
    return 0


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="researcher-eval",
        description="Citation-faithfulness eval (spec 11).",
    )
    p.add_argument("--k", type=int, default=DEFAULT_K,
                   help="runs per claim; Pass^k is the reliability number")
    p.add_argument("--gold", default=None, help="path to a gold JSONL set")
    p.add_argument("--profile", default=None, help="config profile, e.g. local")
    return p


def _configured_verifier(profile: str | None):
    from config import load_config
    from models import build_models
    from verifiers import LLMVerifier

    cfg = load_config(profile=profile)
    return LLMVerifier(build_models(cfg.models).verifier)


if __name__ == "__main__":                                   # pragma: no cover
    sys.exit(main())
