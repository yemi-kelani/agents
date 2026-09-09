"""A transient provider failure must not cost the whole run.

The failure this file exists for: the fan-out researches every topic
concurrently against one provider's token-per-minute allowance, so the *n*th
extraction is the one handed a 429 telling it to retry in under three seconds.
With no retry policy that exception bubbles through `_panic_or_proceed` and ends
the run — discarding every search, fetch and extraction the other branches had
already paid for, because one branch was told to wait.

Retries are configured graph-wide rather than per node, so a node added later
cannot forget to be resilient. Same reasoning as verification being a node and
the progress projection being a wrapper.

The one thing a graph-wide policy must not retry is `interrupt()`. LangGraph
re-raises `GraphBubbleUp` before consulting any policy, so it does not — and the
last test here is what notices if that ever stops being true.

The fakes come from `test_interface` rather than a second copy: a scripted model
that drifts between files is a test suite that agrees with itself and not with
the code.
"""
from __future__ import annotations

import asyncio

import pytest
from langgraph.types import Command

from state import Budget
from test_interface import ScriptedLLM, _Bound, a_graph, run_graph


class Boom(Exception):
    """Stands in for `OpenAIRateLimitError` — a 429 asking to be retried.

    Deliberately not a `RuntimeError`, `ValueError` or `OSError`:
    `default_retry_on` excludes those, so picking one would make these tests
    pass or fail for a reason unrelated to the policy under test.
    """


class FlakyLLM(ScriptedLLM):
    """Fails its first `failures` structured-output calls, then behaves.

    The failure is injected at `ainvoke`, which is the extraction call in the
    research node — the one the real traceback died in.
    """

    def __init__(self, failures: int = 1, **kw):
        super().__init__(**kw)
        self.remaining = failures
        self.calls = 0

    def with_structured_output(self, schema):
        return _FlakyBound(self, schema)


class _FlakyBound(_Bound):
    async def ainvoke(self, prompt, **kwargs):
        self.llm.calls += 1
        if self.llm.remaining > 0:
            self.llm.remaining -= 1
            raise Boom("rate limit reached; try again in 2.659s")
        return self.invoke(prompt)


def test_a_transient_extraction_failure_is_retried_rather_than_ending_the_run():
    llm = FlakyLLM(failures=1)

    out = run_graph(a_graph(llm=llm))

    assert llm.calls > 1, "the failed extraction was never retried"
    assert out["report"], "a retryable failure still ended the run"


def test_a_retried_topic_still_produces_its_claims():
    """The retry re-runs the node, so the topic that stumbled contributes the
    same evidence as one that did not. A run that completed but quietly dropped
    the retried topic's claims would pass the test above and still be wrong."""
    clean = run_graph(a_graph())
    retried = run_graph(a_graph(llm=FlakyLLM(failures=1)))

    assert len(retried["claims"]) == len(clean["claims"])


def test_a_failure_that_outlasts_the_retries_still_surfaces():
    """Retries hide a blip, not an outage. A provider that is genuinely down has
    to reach the caller: a run that swallowed that and reported nothing would be
    worse than one that stopped."""
    with pytest.raises(BaseException) as excinfo:
        run_graph(a_graph(llm=FlakyLLM(failures=99)))

    assert "Boom" in repr(excinfo.value) or "rate limit" in str(excinfo.value)


def test_the_clarification_interrupt_is_not_retried_into_oblivion():
    """`interrupt()` raises in order to pause the graph. A policy that treated
    that as a transient failure would re-run the model turn until the attempts
    ran out and then bubble the interrupt anyway — having asked nobody."""
    graph = a_graph(llm=ScriptedLLM(clarify=(["which vendors?"], "assume open-source")))
    config = {"configurable": {"thread_id": "resilience-resume"}}

    paused = asyncio.run(graph.ainvoke(
        {"question": "compare observability vendors", "budget": Budget()}, config))
    assert "__interrupt__" in paused, "the graph did not pause to ask"

    out = asyncio.run(graph.ainvoke(Command(resume=["the open-source ones"]), config))
    assert out["clarifications"] == [{"q": "which vendors?", "a": "the open-source ones"}]
