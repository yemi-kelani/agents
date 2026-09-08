"""The Python-driven tool-calling loop."""

import asyncio
import json
import time

from langchain_core.messages import AIMessage

from tool_loop import run_tool_loop


class _ScriptedModel:
    """Replies with each scripted string in turn, then repeats the last."""

    def __init__(self, *replies, delay=0.0):
        self.replies = list(replies)
        self.delay = delay
        self.calls = 0

    async def ainvoke(self, history, config=None):
        self.calls += 1
        if self.delay:
            await asyncio.sleep(self.delay)
        index = min(self.calls - 1, len(self.replies) - 1)
        return AIMessage(content=self.replies[index])


async def _echo(question: str) -> str:
    """Echo the question back."""
    return f"answer to {question}"


async def _explode(question: str) -> str:
    """Always fail."""
    raise RuntimeError("tool is broken")


TOOLS = {"explore_codebase": _echo}


def _run(model, tools=None, **kwargs):
    return asyncio.run(
        run_tool_loop(model, tools if tools is not None else TOOLS, "task", **kwargs))


class TestAnswering:
    def test_an_immediate_answer_is_returned(self):
        assert _run(_ScriptedModel('{"answer": "all good"}')) == "all good"

    def test_a_tool_call_is_dispatched_then_answered(self):
        model = _ScriptedModel(
            json.dumps({"tool": "explore_codebase", "args": {"question": "q"}}),
            '{"answer": "done"}',
        )
        assert _run(model) == "done"
        assert model.calls == 2

    def test_prose_on_the_final_step_is_taken_as_the_answer(self):
        """Better than raising: the exploration already happened."""
        assert _run(_ScriptedModel("just some prose"), max_steps=1) == "just some prose"

    def test_unparseable_output_is_retried(self):
        model = _ScriptedModel("not json at all", '{"answer": "recovered"}')
        assert _run(model) == "recovered"


class TestToolFailures:
    def test_an_unknown_tool_comes_back_as_a_result(self):
        model = _ScriptedModel(
            json.dumps({"tool": "nope", "args": {}}), '{"answer": "moved on"}')
        assert _run(model) == "moved on"

    def test_a_raising_tool_does_not_end_the_run(self):
        """A tool failing is an observation, not a crash."""
        model = _ScriptedModel(
            json.dumps({"tool": "explore_codebase", "args": {"question": "q"}}),
            '{"answer": "moved on"}')
        assert _run(model, tools={"explore_codebase": _explode}) == "moved on"

    def test_wrong_arguments_come_back_as_a_result(self):
        model = _ScriptedModel(
            json.dumps({"tool": "explore_codebase", "args": {"wrong": "x"}}),
            '{"answer": "moved on"}')
        assert _run(model) == "moved on"

    def test_non_object_args_are_rejected_without_raising(self):
        model = _ScriptedModel(
            json.dumps({"tool": "explore_codebase", "args": ["not", "a", "dict"]}),
            '{"answer": "moved on"}')
        assert _run(model) == "moved on"


class TestBudgets:
    def test_the_loop_always_returns_an_answer(self):
        """Never raises AssertionError, however uncooperative the model is."""
        model = _ScriptedModel(
            json.dumps({"tool": "explore_codebase", "args": {"question": "q"}}))
        assert _run(model, max_steps=3)

    def test_the_time_budget_is_enforced_not_merely_checked(self):
        """A step already in flight used to run to the model's own timeout, so
        the wall clock could reach several times max_seconds. The loop must cut
        the call off and still return something usable."""
        model = _ScriptedModel('{"answer": "too late"}', delay=5.0)

        started = time.monotonic()
        result = _run(model, max_seconds=0.2)
        elapsed = time.monotonic() - started

        assert elapsed < 3.0, f"budget not enforced: took {elapsed:.1f}s"
        assert result  # degraded, but still an answer


class TestDeadline:
    """`max_seconds` has to bound the whole loop, not just the model turns.

    Bounding only `ainvoke` let a run overshoot by the full cost of every tool
    call it made along the way — the outer budget was the sum of the model
    timeouts, with the tool timeouts stacked invisibly on top.
    """

    def test_a_slow_tool_cannot_outlive_the_budget(self):
        async def slow(**kwargs):
            """A tool that takes longer than the loop has left."""
            await asyncio.sleep(10)
            return "never"

        model = _ScriptedModel('{"tool": "slow", "args": {}}')
        started = time.monotonic()
        answer = _run(model, tools={"slow": slow}, max_seconds=0.1)

        assert time.monotonic() - started < 5
        assert isinstance(answer, str) and answer
