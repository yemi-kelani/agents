"""Spec 11 acceptance: the number, and what it is worth.

Five criteria — one command reproduces the run, results are deterministic given
a cached corpus, unstable claims are enumerated rather than counted, verifier–
human agreement is reported, and the sample-size limits are in the output rather
than buried.

The last two are the ones this file leans on. An eval that reports a headline
without saying how much it is worth is the failure mode the component exists to
avoid: the audit this repo cites found that the better-reading systems were the
less grounded ones, and any harness that could be gamed by a nicer number would
reproduce exactly that.

`asyncio.run` rather than pytest-asyncio, matching the rest of the suite.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from anchor import anchor
from eval.gold import CORPUS_DIR, GoldClaim, load_gold
from eval.injection import INJECTIONS, injection_report
from eval.metrics import (
    ALCE_AGREEMENT, agreement, anchor_rates, caveats, faithfulness, score,
)
from eval.report import render
from eval.run import main, run_eval
from state import Claim, Quote, ev
from verifiers import Verdict

FIXTURES = Path(__file__).parent / "fixtures"

MIMIR_TEXT = (
    "Mimir is a horizontally scalable time series database. A single cluster "
    "has been demonstrated to scale to 1 billion active series.\n\n"
    "The project is licensed under AGPLv3."
)
URL = "https://example.com/mimir"


# --- fakes ------------------------------------------------------------------

class ScriptedVerifier:
    """Answers what the test told it to, per claim text."""

    name = "fake:verifier"

    def __init__(self, verdicts: dict[str, str] | None = None,
                 default: str = "supported"):
        self._verdicts = verdicts or {}
        self._default = default
        self.checked: list[str] = []

    async def check(self, claim: str, quote: str) -> Verdict:
        self.checked.append(claim)
        return Verdict(verdict=self._verdicts.get(claim, self._default), reason="r")


class FlipFlop:
    """A verifier whose judgment changes between identical runs — the variance a
    headline average hides, and the whole reason Pass^k is reported."""

    name = "fake:flipflop"

    def __init__(self, *verdicts: str):
        self._verdicts = list(verdicts)
        self._calls: dict[str, int] = {}

    async def check(self, claim: str, quote: str) -> Verdict:
        turn = self._calls.get(claim, 0)
        self._calls[claim] = turn + 1
        return Verdict(verdict=self._verdicts[turn % len(self._verdicts)], reason="r")


def a_gold(claim_id: str = "g01", label: str = "supported", *,
           claim: str = "Mimir scales to 1 billion active series.",
           quote: str = "scale to 1 billion active series",
           url: str = URL, **overrides) -> GoldClaim:
    return GoldClaim(claim_id=claim_id, claim=claim, source_url=url, quote=quote,
                     label=label, note="stated directly", **overrides)


def a_fetcher(text: str = MIMIR_TEXT):
    """Stands in for the cached corpus. Counts calls, which is how "the same
    source is read once" becomes assertable."""
    async def fetch(url: str) -> str:
        fetch.urls.append(url)
        return text
    fetch.urls = []
    return fetch


def evaluate(gold: list[GoldClaim], verifier, *, k: int = 1, fetch=None, **kwargs):
    return asyncio.run(run_eval(gold=gold, verifier=verifier, k=k,
                                fetch=fetch or a_fetcher(), **kwargs))


# --- the gold set is the ground truth and it has to be true -----------------

def test_every_shipped_gold_quote_really_appears_in_its_source():
    """The one thing that makes the shipped set usable: a quote that is not in
    the corpus scores the verifier for a mistake the corpus made."""
    for g in load_gold():
        text = (CORPUS_DIR / g.corpus).read_text(encoding="utf-8")
        assert anchor(g.quote, text) is not None, f"{g.claim_id}: quote not in source"


def test_the_shipped_gold_set_is_big_enough_to_say_anything():
    assert len(load_gold()) >= 12


def test_the_shipped_gold_set_is_balanced():
    """An all-supported set saturates and shows nothing. The recipe is roughly
    five clearly supported, five clearly unsupported, and a few hard cases."""
    labels = [g.label for g in load_gold()]

    assert labels.count("supported") >= 5
    assert labels.count("unsupported") >= 5


def test_the_shipped_gold_set_carries_hard_cases():
    """Related-but-not-supporting, correct-but-different-number,
    right-topic-wrong-subject. These are what the verifier's judgment actually
    turns on."""
    assert sum(g.hard for g in load_gold()) >= 3


def test_every_shipped_label_says_why_it_is_what_it_is():
    """The note is the hand-labelling. Without it nobody can audit the ground
    truth, and an unauditable ground truth is an opinion."""
    assert all(g.note for g in load_gold())


def test_a_label_the_verifier_cannot_produce_is_refused():
    """"Partially supported" is a documented known unknown (07) and not
    representable. A gold row carrying one could never be matched."""
    with pytest.raises(ValidationError):
        a_gold(label="partially supported")


def test_gold_rows_keep_the_order_of_the_file(tmp_path):
    """Determinism starts here: a set that loads in a different order every run
    cannot produce the same numbers twice."""
    path = tmp_path / "gold.jsonl"
    path.write_text("\n".join(
        json.dumps(a_gold(f"g{i:02}").model_dump()) for i in range(5)), encoding="utf-8")

    assert [g.claim_id for g in load_gold(path)] == [f"g{i:02}" for i in range(5)]


# --- the metrics ------------------------------------------------------------

def test_pass_at_one_is_the_first_run():
    scores = score({"g01": [True, False, False], "g02": [False, True, True]}, k=3)

    assert scores.pass_at_1 == 0.5


def test_pass_at_k_is_correct_in_at_least_one_run():
    """Capability: the pipeline can get this right."""
    scores = score({"g01": [False, False, True], "g02": [False, False, False]}, k=3)

    assert scores.pass_at_k == 0.5


def test_pass_hat_k_is_correct_in_every_run():
    """Reliability, and the number that decays: `Pass^k = p^k`, so a 90%-accurate
    agent scores about 73% at k=3. That amplification is why k=3 is sensitive
    enough to expose verifier instability on a set this small."""
    scores = score({"g01": [True, True, True], "g02": [True, True, False]}, k=3)

    assert scores.pass_hat_k == 0.5
    assert scores.pass_at_k == 1.0


def test_the_claims_whose_judgment_flipped_are_named_not_just_counted():
    """Acceptance, and the most informative output in the harness. A count says
    the pipeline is unstable; the names say where to look."""
    scores = score({"g01": [True, True, True], "g02": [True, False, True],
                    "g03": [False, False, False]}, k=3)

    assert scores.unstable == ["g02"]


def test_a_claim_that_never_passes_is_wrong_rather_than_unstable():
    """Consistently wrong is a different problem from sometimes wrong, and
    conflating them sends you to fix the wrong thing."""
    scores = score({"g01": [False, False, False]}, k=3)

    assert scores.unstable == []


def test_an_empty_gold_set_scores_zero_rather_than_dividing_by_zero():
    assert score({}, k=3).pass_at_1 == 0.0


# --- acceptance: verifier-human agreement is reported -----------------------

def test_agreement_is_the_share_of_verdicts_matching_the_human_label():
    assert agreement({"g01": [True, True], "g02": [True, False]}) == 0.75


def test_agreement_pools_every_run_rather_than_reading_only_the_first():
    """Otherwise it is Pass@1 under a second name. Pooling every prediction is
    both a different number and a steadier one."""
    per_claim = {"g01": [True, False, False]}

    assert agreement(per_claim) == pytest.approx(1 / 3)
    assert score(per_claim, k=3).pass_at_1 == 1.0


def test_the_benchmark_the_agreement_has_to_clear_is_stated():
    """ALCE's automatic metrics reach ~85%/78% agreement with human labels. Under
    roughly 80%, the automated numbers do not carry weight and the human labels
    are the primary result."""
    assert 0.75 <= ALCE_AGREEMENT <= 0.85


# --- secondary rates, computed off a real run -------------------------------

def test_the_quote_not_found_rate_counts_fabrications_against_every_quote():
    """Fabricated or paraphrased quotes. Spec 10 tracks this per model — it is a
    direct measurement of what a smaller extractor costs, and a much better
    answer than "the local model was worse"."""
    claims = [Claim(id="c0", topic_id="t0", text="a claim", quotes=[
        Quote(source_id="s", text="q", start=0, end=1)])]
    trace = [ev("quote_not_found", claim="a", quote="b")]

    assert anchor_rates(claims, trace)["quote_not_found"] == 0.5


def test_the_fuzzy_rate_counts_quotes_that_needed_the_third_tier():
    """A high fuzzy rate means the extraction prompt is the problem rather than
    the matcher — the model is "correcting" the source it was told to copy."""
    claims = [Claim(id=f"c{i}", topic_id="t0", text="a claim", quotes=[
        Quote(source_id="s", text="q", start=0, end=1)]) for i in range(3)]
    trace = [ev("quote_fuzzy", claim_id="c0", quote="q")]

    assert anchor_rates(claims, trace)["fuzzy"] == pytest.approx(1 / 3)


def test_a_run_that_produced_no_quotes_reports_no_rate():
    assert anchor_rates([], [])["quote_not_found"] is None


def test_faithfulness_is_the_supported_share_of_a_single_run():
    claims = [Claim(id="c0", topic_id="t", text="a", quotes=[], verdict="supported"),
              Claim(id="c1", topic_id="t", text="b", quotes=[], verdict="unsupported")]

    assert faithfulness(claims) == 0.5


# --- acceptance: the sample size is stated, not buried ----------------------

def test_the_output_says_what_a_single_claim_is_worth():
    """With 12 claims one claim is 8.3 percentage points, so a two-point
    difference is noise. Stating it is what stops the headline being read as
    more precise than it is."""
    lines = " ".join(caveats(n=12, k=3))

    assert "8.3" in lines


def test_the_caveats_are_computed_from_the_set_that_actually_ran():
    """A fixed paragraph goes stale the moment somebody adds a claim."""
    assert "5.0 percentage points" in " ".join(caveats(n=20, k=3))


def test_the_caveats_say_how_crude_three_runs_is():
    """Credible papers use five to ten."""
    lines = " ".join(caveats(n=12, k=3))

    assert "credible papers use 5-10 runs" in lines


def test_the_caveats_refuse_a_significance_test_this_sample_cannot_support():
    lines = " ".join(caveats(n=12, k=3)).lower()

    assert "p-value" in lines or "significance" in lines


# --- the harness ------------------------------------------------------------

def test_a_verdict_matching_the_human_label_scores():
    result = evaluate([a_gold(label="supported")], ScriptedVerifier(default="supported"))

    assert result.scores.pass_at_1 == 1.0


def test_a_verdict_contradicting_the_human_label_does_not():
    result = evaluate([a_gold(label="unsupported")], ScriptedVerifier(default="supported"))

    assert result.scores.pass_at_1 == 0.0


def test_a_quote_not_found_verdict_counts_as_unsupported():
    """The verifier answers in three verdicts and the human labels in two. A
    fabricated quote is not support, so it collapses to "unsupported" rather
    than being scored as a third outcome the gold set cannot express."""
    gold = a_gold(label="unsupported", quote="scale to 1 billion active series")
    result = evaluate([gold], ScriptedVerifier(default="unsupported"))

    assert result.scores.pass_at_1 == 1.0


def test_every_gold_claim_is_verified_once_per_run():
    gold = [a_gold("g01"), a_gold("g02", claim="Mimir is AGPLv3 licensed.",
                                  quote="licensed under AGPLv3")]
    verifier = ScriptedVerifier()

    evaluate(gold, verifier, k=3)

    assert len(verifier.checked) == 6


def test_a_verifier_that_flips_between_runs_is_named_as_unstable():
    """Acceptance, end to end: identical inputs, different answers, and the
    harness says which claim spent the instability budget."""
    result = evaluate([a_gold(label="supported")],
                      FlipFlop("supported", "unsupported", "supported"), k=3)

    assert result.scores.unstable == ["g01"]
    assert result.scores.pass_at_k == 1.0
    assert result.scores.pass_hat_k == 0.0


def test_acceptance_the_same_corpus_and_verifier_give_the_same_numbers():
    """Acceptance: deterministic given a cached corpus. The harness must not add
    variance of its own on top of whatever the model has."""
    gold = [a_gold(f"g{i:02}") for i in range(4)]

    first = evaluate(gold, ScriptedVerifier(), k=3)
    second = evaluate(gold, ScriptedVerifier(), k=3)

    assert first.scores == second.scores


def test_a_source_cited_by_several_claims_is_read_once():
    """The corpus is fetched once and reused across every run, which is what
    makes k runs cost k verifications rather than k crawls."""
    fetch = a_fetcher()

    evaluate([a_gold("g01"), a_gold("g02")], ScriptedVerifier(), k=3, fetch=fetch)

    assert fetch.urls == [URL]


def test_a_gold_quote_that_no_longer_appears_in_its_source_is_excluded_and_named():
    """Corpus drift is not a verifier failure. Scoring it as one would blame the
    judge for the page having changed underneath the label — this pipeline's own
    thesis, applied to its own harness."""
    gold = [a_gold("g01"), a_gold("g02", quote="a sentence nobody wrote")]

    result = evaluate(gold, ScriptedVerifier())

    assert result.unanchored == ["g02"]
    assert result.scores.n == 1


def test_a_run_where_the_whole_corpus_drifted_says_so_rather_than_scoring_zero():
    result = evaluate([a_gold("g01", quote="not in the source")], ScriptedVerifier())

    assert result.scores.n == 0
    assert result.unanchored == ["g01"]


def test_the_shipped_corpus_is_read_from_disk_rather_than_fetched():
    """The shipped set is reproducible offline: every row names a document that
    travels with it, so the default run needs no network at all."""
    fetch = a_fetcher()

    result = asyncio.run(run_eval(verifier=ScriptedVerifier(), k=1, fetch=fetch))

    assert fetch.urls == []
    assert result.scores.n >= 12


# --- the report -------------------------------------------------------------

def a_report(**kwargs) -> str:
    """Two claims a flip-flopping verifier gets right half the time. Distinct
    claim texts, or the two rows share the flip-flop's turn counter and neither
    of them ever flips."""
    return render(evaluate(
        [a_gold("g01"),
         a_gold("g02", label="unsupported", claim="Mimir is Apache licensed.",
                quote="licensed under AGPLv3")],
        FlipFlop("supported", "unsupported"), k=3, **kwargs))


def test_the_report_shows_the_three_headline_numbers_together():
    """Read together or not at all: Pass@3 above Pass@1 means capable but
    inconsistent; Pass^3 near Pass@1 means consistent."""
    out = a_report()

    assert "Pass@1" in out and "Pass@3" in out and "Pass^3" in out


def test_the_report_names_the_unstable_claims_rather_than_only_counting_them():
    assert "unstable claims (2): g01, g02" in a_report()


def test_the_report_states_the_sample_size_limits():
    """Acceptance: in the output, not buried."""
    assert "percentage points" in a_report()


def test_the_report_gives_the_agreement_and_what_it_has_to_beat():
    out = a_report()

    assert "agreement" in out.lower()
    assert "80%" in out


def test_the_report_says_when_the_agreement_is_too_low_to_carry_weight():
    """Under roughly 80%, the automated numbers do not stand on their own and
    the report has to say so rather than printing them at the same size."""
    result = evaluate([a_gold(label="unsupported")], ScriptedVerifier(default="supported"))

    assert "human labels" in render(result).lower()


def test_the_report_warns_that_the_shipped_gold_set_is_synthetic():
    """The corpus that ships is written to make the harness reproducible, not
    to be a claim about anybody's product. Quoting a number off it as though it
    were hand-labelled field data is exactly the dishonesty this repo measures
    against."""
    result = asyncio.run(run_eval(verifier=ScriptedVerifier(), k=1))

    assert "synthetic" in render(result).lower()


def test_a_report_over_someone_elses_gold_set_carries_no_such_warning(tmp_path):
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(a_gold().model_dump()), encoding="utf-8")

    result = evaluate(load_gold(path), ScriptedVerifier(), gold_path=path)

    assert "synthetic" not in render(result).lower()


# --- injection results ------------------------------------------------------

def test_an_injection_that_survives_cleaning_reached_extraction():
    """The hidden div is styled invisible, and `display:none` is a browser
    instruction rather than a parser one — the text is in the document."""
    rows = {r.fixture: r for r in injection_report(FIXTURES)}

    assert rows["hidden_div.html"].reached_extraction is True


def test_an_injection_the_cleaner_strips_never_reaches_the_model():
    """Two of the four never get that far. Reporting them as "contained" rather
    than as "caught" is the difference between the layers doing independent work
    and one layer carrying the whole defense."""
    rows = {r.fixture: r for r in injection_report(FIXTURES)}

    assert rows["comment.html"].reached_extraction is False
    assert rows["alt_text.html"].reached_extraction is False


def test_every_fixture_names_the_payload_it_is_trying_to_land():
    """A fixture with no declared payload silently passes: "the injection did
    not survive" is indistinguishable from "there was nothing to survive"."""
    assert set(INJECTIONS) == {f.name for f in FIXTURES.glob("*.html")}


def test_the_verdict_column_stays_empty_without_a_model_to_fill_it():
    """The containment half is deterministic and always runs; the half that
    needs a model says so rather than implying a check nobody made."""
    assert all(r.verdict is None for r in injection_report(FIXTURES))


def test_verification_catching_a_landed_injection_is_reported_as_the_catch():
    """The more informative outcome of the two: the payload reached the model,
    a claim came out, and the second layer refused it."""
    async def verify(text: str) -> tuple[bool, str | None]:
        return True, "unsupported"

    rows = {r.fixture: r for r in injection_report(FIXTURES, verify=verify)}

    assert rows["hidden_div.html"].claim_produced is True
    assert rows["hidden_div.html"].verdict == "unsupported"
    assert rows["comment.html"].verdict is None, "it never reached the model"


# --- acceptance: one command --------------------------------------------------

def test_one_command_reproduces_the_full_run(capsys):
    """Acceptance. No arguments, no environment, no network — the shipped gold
    set and corpus are enough to print a complete report."""
    exit_code = main(["--k", "2"], verifier=ScriptedVerifier())

    assert exit_code == 0
    out = capsys.readouterr().out
    assert "Pass^2" in out and "percentage points" in out


def test_the_command_takes_a_gold_set_of_your_own(tmp_path, capsys):
    path = tmp_path / "gold.jsonl"
    path.write_text(json.dumps(a_gold().model_dump()), encoding="utf-8")

    main(["--k", "1", "--gold", str(path)], verifier=ScriptedVerifier(),
         fetch=a_fetcher())

    assert "synthetic" not in capsys.readouterr().out.lower()
