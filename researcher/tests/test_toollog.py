"""The tool-call log — the debug channel, not the human one.

These tests pin the two properties the rest of the wiring depends on: a call
that raises is still logged *and still raises*, and nothing that looks like a
credential is ever written out.
"""
from __future__ import annotations

import asyncio
import logging

import pytest

from toollog import (
    ENV_LOG_LEVEL, PREVIEW_CHARS, ROOT_LOGGER_NAME, configure, level_for,
    preview, tool_call,
)


@pytest.fixture
def logged(caplog):
    """Everything the tool logger emitted, at every level."""
    caplog.set_level(logging.DEBUG, logger="researcher.tools")
    return caplog


def messages(logged) -> str:
    return "\n".join(r.getMessage() for r in logged.records)


# --- preview ---------------------------------------------------------------

def test_short_text_is_previewed_whole():
    assert preview("3 hits") == "3 hits"


def test_long_text_is_cut_and_says_how_much_it_cut():
    text = "x" * (PREVIEW_CHARS + 500)

    line = preview(text)

    assert len(line) < PREVIEW_CHARS + 40
    assert "+500 chars" in line


def test_preview_flattens_newlines_so_one_call_stays_one_line():
    assert "\n" not in preview("first line\nsecond line\n\n\tthird")


def test_preview_renders_a_list_of_results_rather_than_its_repr():
    line = preview(["https://a.example", "https://b.example"])

    assert "https://a.example" in line
    assert "https://b.example" in line


# --- tool_call: the happy path ---------------------------------------------

def test_a_call_logs_its_name_and_what_it_returned(logged):
    with tool_call("search", query="pelicans") as call:
        call.set("3 hits", payload=["https://a.example"])

    logged_text = messages(logged)
    assert "search" in logged_text
    assert "pelicans" in logged_text
    assert "3 hits" in logged_text
    assert "https://a.example" in logged_text


def test_a_call_reports_how_long_it_took(logged):
    with tool_call("fetch", url="https://a.example") as call:
        call.set("ok")

    assert "ms" in messages(logged)


def test_the_result_is_logged_at_info_so_v_alone_shows_what_tools_returned(logged):
    with tool_call("search") as call:
        call.set("3 hits")

    levels = {r.levelno for r in logged.records if "3 hits" in r.getMessage()}
    assert levels == {logging.INFO}


def test_the_arguments_are_logged_at_debug_so_they_stay_out_of_the_way(logged):
    with tool_call("search", query="pelicans") as call:
        call.set("3 hits")

    levels = {r.levelno for r in logged.records if "pelicans" in r.getMessage()}
    assert levels == {logging.DEBUG}


def test_a_call_that_sets_no_outcome_is_still_logged(logged):
    """A wrapped call whose author forgot `set` must not vanish from the log."""
    with tool_call("fetch", url="https://a.example"):
        pass

    assert "fetch" in messages(logged)


# --- tool_call: the failure path -------------------------------------------

def test_a_failing_call_is_logged_with_its_error(logged):
    with pytest.raises(ValueError):
        with tool_call("fetch", url="https://a.example"):
            raise ValueError("dns went sideways")

    logged_text = messages(logged)
    assert "ValueError" in logged_text
    assert "dns went sideways" in logged_text


def test_a_failing_call_still_raises(logged):
    """The log observes; it must never swallow. A tool logger that ate an
    exception would turn a dead backend into a silent empty result."""
    with pytest.raises(ValueError):
        with tool_call("fetch"):
            raise ValueError("boom")


def test_a_failure_is_logged_at_warning_so_it_shows_without_v(logged):
    with pytest.raises(ValueError):
        with tool_call("fetch"):
            raise ValueError("boom")

    assert any(r.levelno == logging.WARNING for r in logged.records)


def test_an_await_inside_the_block_is_timed_and_logged(logged):
    """The same context manager has to serve the async call sites, which is the
    only reason one manager covers search, fetch and every model call."""
    async def run():
        with tool_call("fetch", url="https://a.example") as call:
            await asyncio.sleep(0)
            call.set("ok", payload="hello")

    asyncio.run(run())

    assert "hello" in messages(logged)


# --- secrets ---------------------------------------------------------------

@pytest.mark.parametrize("name", ["api_key", "token", "SECRET", "password",
                                  "authorization"])
def test_credential_shaped_arguments_are_redacted(logged, name):
    with tool_call("search", **{name: "sk-live-abcdef123456"}) as call:
        call.set("ok")

    assert "sk-live-abcdef123456" not in messages(logged)


def test_redaction_keeps_the_argument_name_so_you_can_see_it_was_passed(logged):
    with tool_call("search", api_key="sk-live-abcdef123456") as call:
        call.set("ok")

    assert "api_key" in messages(logged)


def test_ordinary_arguments_survive_redaction(logged):
    with tool_call("search", query="pelicans", api_key="sk-live-x") as call:
        call.set("ok")

    assert "pelicans" in messages(logged)


# --- control flow is not failure -------------------------------------------
#
# `interrupt()` leaves a tool by raising. Pausing to ask a human is the feature,
# not a fault, and a log that cried wolf on every clarification would train
# people to ignore the line that matters.

class Pause(Exception):
    """Stands in for langgraph's `GraphBubbleUp`."""


def test_a_passthrough_exception_is_not_logged_as_a_failure(logged):
    with pytest.raises(Pause):
        with tool_call("ask_user", (Pause,)):
            raise Pause("asking")

    assert not any(r.levelno >= logging.WARNING for r in logged.records)


def test_a_passthrough_exception_still_propagates(logged):
    """Passing through means the log keeps quiet, never that it interferes."""
    with pytest.raises(Pause):
        with tool_call("ask_user", (Pause,)):
            raise Pause("asking")


def test_a_real_failure_is_still_logged_when_passthrough_is_set(logged):
    with pytest.raises(ValueError):
        with tool_call("ask_user", (Pause,)):
            raise ValueError("boom")

    assert any(r.levelno == logging.WARNING for r in logged.records)


def test_an_argument_named_passthrough_is_still_an_argument(logged):
    """`passthrough` is positional-only, so a tool is free to have an argument
    of that name — tool arguments are chosen by a model, not by us."""
    with tool_call("ask_user", passthrough="a real argument value") as call:
        call.set("ok")

    assert "a real argument value" in messages(logged)


# --- turning it on ----------------------------------------------------------

@pytest.fixture
def unconfigured():
    """Leave the `researcher` logger exactly as it was found.

    `configure` is deliberately idempotent, so a test that installed a handler
    would otherwise hand the next test a handler bound to a stale stderr.
    """
    root = logging.getLogger(ROOT_LOGGER_NAME)
    before = (list(root.handlers), root.level, root.propagate)
    root.handlers = []
    yield root
    root.handlers, root.level, root.propagate = before


def test_the_default_is_silence(unconfigured, monkeypatch):
    """A run that asked for nothing prints what it always printed."""
    monkeypatch.delenv(ENV_LOG_LEVEL, raising=False)

    configure()

    assert not unconfigured.isEnabledFor(logging.INFO)


def test_v_shows_what_the_tools_returned(unconfigured):
    configure(level_for(1))

    assert unconfigured.isEnabledFor(logging.INFO)
    assert not unconfigured.isEnabledFor(logging.DEBUG)


def test_vv_adds_the_arguments_and_the_prompts(unconfigured):
    configure(level_for(2))

    assert unconfigured.isEnabledFor(logging.DEBUG)


def test_more_vs_than_levels_is_still_just_debug(unconfigured):
    configure(level_for(9))

    assert unconfigured.isEnabledFor(logging.DEBUG)


def test_the_environment_sets_the_level_for_surfaces_with_no_flags(
        unconfigured, monkeypatch):
    """`langgraph dev` and the eval harness never see `run.py`'s argv."""
    monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")

    configure()

    assert unconfigured.isEnabledFor(logging.DEBUG)


def test_an_explicit_level_beats_the_environment(unconfigured, monkeypatch):
    monkeypatch.setenv(ENV_LOG_LEVEL, "DEBUG")

    configure(level_for(1))

    assert not unconfigured.isEnabledFor(logging.DEBUG)


def test_configuring_twice_does_not_log_everything_twice(unconfigured):
    """Counted as a delta rather than an absolute: pytest attaches capture
    handlers of its own, and the property under test is only that `configure`
    never stacks a second copy of *its* handler."""
    configure(level_for(1))
    after_one = len(unconfigured.handlers)

    configure(level_for(2))

    assert len(unconfigured.handlers) == after_one


def test_the_log_goes_to_stderr_so_the_report_on_stdout_stays_clean(
        unconfigured, capsys):
    """`run.py -v "..." 2>debug.log` is the whole point: the terminal UI writes
    the report to stdout, and diagnostics must not land in the middle of it."""
    configure(level_for(1))

    with tool_call("search") as call:
        call.set("3 hits")

    captured = capsys.readouterr()
    assert "3 hits" in captured.err
    assert "3 hits" not in captured.out


def test_a_bad_level_name_does_not_take_the_run_down(unconfigured, monkeypatch):
    """A typo in an environment variable must not be fatal to a research run."""
    monkeypatch.setenv(ENV_LOG_LEVEL, "LOUD")

    configure()

    assert unconfigured.handlers
