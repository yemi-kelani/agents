"""The numbers, and what they are worth — spec 11.

Metric definitions follow tau-bench (Yao et al., 2024, arXiv:2406.12045), which
introduced `pass^k`: the probability an agent succeeds on *all* k attempts.
`Pass^k = p^k` decays exponentially, so a 90%-accurate agent is only ~73%
consistent at k=3. That amplification is the point — a pipeline that is right on
average and unstable run-to-run shows up here and nowhere else.

Everything in this module is a pure function of results already collected. The
harness decides what to run; this decides what it meant.
"""
from __future__ import annotations

from dataclasses import dataclass

from state import Claim

ALCE_AGREEMENT = 0.80
"""The bar the automated verifier has to clear to be worth reporting on its own.

ALCE's automatic metrics reach ~85% agreement with human labels on citation
recall and ~78% on precision (Gao et al., EMNLP 2023). Under roughly this, the
automated numbers do not carry weight and the human labels are the primary
result — which the report says out loud rather than printing both at the same
size.
"""

CREDIBLE_RUNS = 5
"""Below this, per-claim stability is a crude estimate. Credible papers use 5-10."""

MIN_SEPARATION_CLAIMS = 2
"""Report a difference only with roughly this much separation. At n=12 one claim
is 8.3 points, so a two-point difference is noise wearing a decimal."""


@dataclass(frozen=True)
class Scores:
    """The headline, and the list that explains it."""
    n: int
    k: int
    pass_at_1: float
    pass_at_k: float
    pass_hat_k: float
    unstable: list[str]
    """Built in gold-file order. Frozen dataclasses compare by field, so two
    equal score sets would compare unequal if that order ever drifted."""


def score(per_claim: dict[str, list[bool]], k: int) -> Scores:
    """Turn per-claim, per-run correctness into the three headline numbers.

    `unstable` is the most informative output in the harness: those are the
    claims where judgment flipped between identical runs. A count says the
    pipeline is unstable; the names say where to look. Consistently wrong is a
    different problem and is deliberately not in this list.
    """
    n = len(per_claim)
    if not n:
        return Scores(n=0, k=k, pass_at_1=0.0, pass_at_k=0.0, pass_hat_k=0.0,
                      unstable=[])

    return Scores(
        n=n,
        k=k,
        pass_at_1=sum(runs[0] for runs in per_claim.values()) / n,
        pass_at_k=sum(any(runs) for runs in per_claim.values()) / n,
        pass_hat_k=sum(all(runs) for runs in per_claim.values()) / n,
        unstable=[cid for cid, runs in per_claim.items() if any(runs) and not all(runs)],
    )


def agreement(per_claim: dict[str, list[bool]]) -> float:
    """How often the automated verifier agreed with the human label.

    Pooled over every run rather than read off the first, which is what keeps it
    from being `pass_at_1` under a second name. It is still not independent
    corroboration — this eval's "correct" *is* "agrees with the human" — so the
    report says what it is rather than presenting two numbers as two witnesses.
    """
    verdicts = [ok for runs in per_claim.values() for ok in runs]
    return sum(verdicts) / len(verdicts) if verdicts else 0.0


def faithfulness(claims: list[Claim]) -> float:
    """Supported claims over cited claims, one run. The ratio the report renders
    for a reader (08), read here as a number for a table."""
    if not claims:
        return 0.0
    return sum(c.verdict == "supported" for c in claims) / len(claims)


def anchor_rates(claims: list[Claim], trace: list[dict]) -> dict[str, float | None]:
    """What extraction did to the quotes it was told to copy verbatim.

    Both rates come off a real pipeline run rather than the gold set, because
    both are properties of the extractor: `quote_not_found` counts fabricated or
    paraphrased quotes, and `fuzzy` counts quotes that only matched once the
    anchor gave up on the text as written. A high fuzzy rate means the
    extraction prompt is the problem rather than the matcher.

    Spec 10 tracks these per model. They are a direct measurement of what a
    smaller extractor costs, and a much better answer than "the local model was
    worse".
    """
    anchored = sum(len(c.quotes) for c in claims)
    not_found = sum(e["kind"] == "quote_not_found" for e in trace)
    fuzzy = sum(e["kind"] == "quote_fuzzy" for e in trace)

    attempted = anchored + not_found
    if not attempted:
        return {"quote_not_found": None, "fuzzy": None, "n_quotes": 0}

    return {"quote_not_found": not_found / attempted,
            "fuzzy": fuzzy / attempted,
            "n_quotes": attempted}


def caveats(n: int, k: int) -> list[str]:
    """What this sample size can and cannot support.

    Computed from the set that actually ran, not written down once: a fixed
    paragraph goes stale the moment somebody adds a claim, and a stale caveat is
    worse than none because it reads as though somebody checked.
    """
    if not n:
        return ["No claim could be scored: every gold quote failed to anchor."]

    per_claim = 100 / n
    return [
        f"n={n} claims x {k} runs is underpowered for anything but a large effect.",
        f"One claim moves every rate by {per_claim:.1f} percentage points, so a "
        f"difference smaller than that is not a difference.",
        f"Report a result only with about {MIN_SEPARATION_CLAIMS}+ claims of "
        f"separation ({MIN_SEPARATION_CLAIMS * per_claim:.1f} points).",
        f"{k} runs estimates per-claim stability crudely; credible papers use "
        f"{CREDIBLE_RUNS}-10 runs.",
        "No significance test at this n: the raw counts and the unstable list "
        "are reported instead of a p-value the sample cannot support.",
    ]
