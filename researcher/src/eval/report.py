"""Rendering the result — spec 11.

The three headline numbers are printed together because they are only meaningful
together: Pass@k above Pass@1 means the pipeline is *capable* of getting a claim
right but does not always; Pass^k near Pass@1 means it is *consistent*. The gap
between them is the instability budget, and the named claims are exactly who
spent it.

The caveats are not a footer. At n=12 one claim is 8.3 percentage points, and a
headline printed without that is a headline that will be quoted without it.
"""
from __future__ import annotations

from eval.metrics import ALCE_AGREEMENT, anchor_rates, caveats, faithfulness

SYNTHETIC_WARNING = (
    "NOTE: this is the shipped gold set. Its corpus is synthetic — written so "
    "the harness reproduces offline, over example.com URLs that are claims "
    "about nobody's product. Replace it with claims the agent actually made and "
    "labels you actually checked before quoting any of these numbers."
)


def render(result) -> str:
    """The whole report, as text. One function so nothing can be printed without
    the part that says what it is worth."""
    scores = result.scores
    k = scores.k

    lines = [
        f"{'':<12}{'Pass@1':>8}{f'Pass@{k}':>9}{f'Pass^{k}':>9}{'unstable':>10}",
        f"{'verifier':<12}{scores.pass_at_1:>8.2f}{scores.pass_at_k:>9.2f}"
        f"{scores.pass_hat_k:>9.2f}{len(scores.unstable):>10}",
        "",
        f"gold set:  {result.gold_path or '(supplied in-process)'}  (n={scores.n}, k={k})",
        f"verifier:  {result.verifier}",
        "",
        *_unstable(scores),
        *_unanchored(result),
        *_agreement(result),
        *_secondary(result),
        "Sample size:",
        *(f"  - {line}" for line in caveats(scores.n, k)),
    ]

    if result.is_shipped_gold:
        lines += ["", SYNTHETIC_WARNING]

    return "\n".join(lines)


def _unstable(scores) -> list[str]:
    """Enumerated, never only counted: a count says the pipeline is unstable,
    the names say where to look."""
    if not scores.unstable:
        return ["unstable claims: none — every claim got the same verdict in "
                f"all {scores.k} runs.", ""]
    return [f"unstable claims ({len(scores.unstable)}): "
            + ", ".join(scores.unstable),
            "  judgment flipped between identical runs; this is the gap between "
            f"Pass@{scores.k} and Pass^{scores.k}.", ""]


def _unanchored(result) -> list[str]:
    """A gold quote that no longer appears in its source is a corpus problem.
    Reporting it as a verifier failure would blame the judge for the page."""
    if not result.unanchored:
        return []
    return [f"excluded ({len(result.unanchored)}): " + ", ".join(result.unanchored),
            "  their quotes no longer appear in the cited source, so the "
            "verifier was never asked. Fix the gold set, not the verifier.", ""]


def _agreement(result) -> list[str]:
    """The number that says what all the others are worth."""
    lines = [
        f"verifier-human agreement: {result.agreement:.0%} "
        f"(pooled over all {result.scores.k} runs; benchmark to beat is "
        f"~{ALCE_AGREEMENT:.0%}, ALCE)",
    ]
    if result.agreement < ALCE_AGREEMENT:
        lines.append(
            "  Below the benchmark: the automated numbers above do not carry "
            "weight on their own. Report the human labels as primary and say so."
        )
    lines.append(
        "  Not independent corroboration — this eval's \"correct\" is \"agrees "
        "with the human label\", so agreement and Pass@1 measure one thing twice."
    )
    return lines + [""]


def _secondary(result) -> list[str]:
    """What extraction did to its quotes. Only available when the caller has an
    actual pipeline run — a gold set has hand-written quotes, so measuring the
    extractor against it would measure nothing."""
    if not result.run_state:
        return []

    claims = result.run_state.get("claims") or []
    rates = anchor_rates(claims, result.run_state.get("trace") or [])
    if rates["quote_not_found"] is None:
        return ["pipeline run: no quotes were extracted.", ""]

    return [
        f"faithfulness (supported/cited, one run): {faithfulness(claims):.0%}",
        f"quote_not_found rate: {rates['quote_not_found']:.0%} "
        f"of {rates['n_quotes']} quotes — fabricated or paraphrased",
        f"fuzzy-match rate: {rates['fuzzy']:.0%} — high means the extraction "
        "prompt is the problem, not the matcher",
        "",
    ]
