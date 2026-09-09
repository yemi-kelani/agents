"""The terminal entry point's argument handling.

Small surface, one sharp edge: `run.py` builds the research question by joining
whatever is left of `argv`, so every flag it grows has to be taken out of that
join. A `-v` that ends up inside the question is not a crash — it is a run that
quietly researches the wrong thing, which is worse.
"""
from __future__ import annotations

import logging

import pytest

import run
from toollog import LEVELS, ROOT_LOGGER_NAME


# --- verbosity --------------------------------------------------------------

def test_no_flag_is_silence():
    assert run.verbosity([]) == 0


def test_v_is_one_level():
    assert run.verbosity(["-v"]) == 1


def test_vv_is_two():
    assert run.verbosity(["-vv"]) == 2


def test_repeated_flags_add_up():
    assert run.verbosity(["-v", "-v"]) == 2


def test_a_v_inside_the_question_is_not_a_flag():
    """"vs" is a word people put in research questions."""
    assert run.verbosity(["grafana", "vs", "datadog"]) == 0


def test_the_verbosity_reaches_a_level_name():
    assert run.level_for(run.verbosity(["-v"])) in LEVELS


# --- the question -----------------------------------------------------------

def test_the_question_survives_intact():
    assert run.question(["how", "far", "does", "mimir", "scale"]) == (
        "how far does mimir scale")


def test_the_verbosity_flag_never_lands_in_the_question():
    assert run.question(["-v", "how far does mimir scale"]) == (
        "how far does mimir scale")


def test_the_full_flag_never_lands_in_the_question():
    assert run.question(["--full", "how far does mimir scale"]) == (
        "how far does mimir scale")


def test_every_flag_comes_out_at_once():
    assert run.question(["--full", "-vv", "how far does mimir scale"]) == (
        "how far does mimir scale")


def test_a_question_that_is_only_flags_is_empty():
    """`run.py` prints usage on an empty question, and "-v" alone must reach
    that path rather than being researched."""
    assert run.question(["-v"]) == ""


# --- the surfaces that never see argv ---------------------------------------
#
# `RESEARCHER_LOG_LEVEL` is the only switch `langgraph dev` and the eval harness
# have, and a switch nothing reads is not a switch. Both entry points have to
# call `configure` themselves.

@pytest.fixture
def unconfigured(monkeypatch):
    root = logging.getLogger(ROOT_LOGGER_NAME)
    before = (list(root.handlers), root.level, root.propagate)
    root.handlers = []
    monkeypatch.setenv("RESEARCHER_LOG_LEVEL", "DEBUG")
    yield root
    root.handlers, root.level, root.propagate = before


def test_the_served_graph_turns_the_log_on(unconfigured, monkeypatch):
    import app

    monkeypatch.setattr(app, "build_graph_from_config", lambda *a, **k: "graph")

    app.make_graph()

    assert unconfigured.isEnabledFor(logging.DEBUG)


def test_the_eval_harness_turns_the_log_on(unconfigured, monkeypatch):
    import eval.report
    import eval.run

    async def no_eval(**kwargs):
        return "result"

    monkeypatch.setattr(eval.run, "run_eval", no_eval)
    monkeypatch.setattr(eval.report, "render", lambda result: "")

    eval.run.main([], verifier=object())

    assert unconfigured.isEnabledFor(logging.DEBUG)
