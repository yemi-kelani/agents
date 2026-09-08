"""A tool-calling loop driven from Python.

`ShellChatModel.bind_tools` raises: the CLIs behind it have no native tool
calling. So the protocol is text — the model replies with a JSON object naming a
tool or carrying a final answer, and this loop dispatches it.

Mechanism, not a node: a node picks the task and the tools, this runs the turns.
"""

from __future__ import annotations

import asyncio
import inspect
import time
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from log import get_logger
from parsers import parse_tool_step
from prompts import load
from utilities import trim_text

logger = get_logger(__name__)

Tool = Callable[..., Awaitable[str]]

# Tool results carry the bulk of the history, and the history is re-sent in full
# on every step (the CLIs are stateless), so this is what keeps a run bounded.
MAX_RESULT_TOKENS = 4_000


def render_docs(tools: dict[str, Tool]) -> str:
    """One line per tool: its signature and the first line of its docstring."""
    lines = []
    for name, fn in tools.items():
        args = ", ".join(p for p in inspect.signature(fn).parameters)
        doc = (inspect.getdoc(fn) or "").strip().splitlines()
        lines.append(f"  {name}({args}) — {doc[0] if doc else 'no description'}")
    return "\n".join(lines)


async def _call(tools: dict[str, Tool], step: dict) -> str:
    """Dispatch one tool call. Every failure comes back as a result, not a raise."""
    name = str(step.get("tool"))
    fn = tools.get(name)
    if fn is None:
        return f"unknown tool {name!r}. Available tools: {', '.join(tools)}"

    args = step.get("args") or {}
    if not isinstance(args, dict):
        return f"'args' must be a JSON object, got {type(args).__name__}"

    try:
        return await fn(**args)
    except TypeError as exc:
        # Wrong argument names — recoverable, the signature is in the prompt.
        return f"could not call {name}: {exc}"
    except Exception as exc:
        # A tool failing is an observation. "File not found" is worth knowing.
        logger.warning(f"Tool {name} raised: {exc}")
        return f"{name} failed: {exc}"


def _last_answer(history: list[tuple[str, str]]) -> str:
    """The most recent thing the model said, for when the loop runs out of time.

    Prefers a tool result over a bare tool call: the result is information the
    review can use, whereas the call is just a request the loop never served.
    """
    for role, content in reversed(history):
        if role == "human" and content.startswith("TOOL RESULT:"):
            return content[len("TOOL RESULT:"):].strip()
    for role, content in reversed(history):
        if role == "ai":
            return content.strip()
    return "The review ran out of time before reaching an answer."


async def run_tool_loop(
    model: BaseChatModel,
    tools: dict[str, Tool],
    task: str,
    max_steps: int = 6,
    max_seconds: float = 600.0,
) -> str:
    """Run `task` to a final answer, letting the model call `tools` along the way.

    Always returns an answer. Running out of steps or time is an expected
    outcome — the model is told to answer now with what it has, rather than
    losing the exploration to an exception.
    """
    history: list[tuple[str, str]] = [
        ("system", load("tool_loop").format(tool_docs=render_docs(tools))),
        ("human", task),
    ]
    deadline = time.monotonic() + max_seconds

    def remaining() -> float:
        """Whatever is left of the budget. Raises when it is spent."""
        left = deadline - time.monotonic()
        if left <= 0:
            raise asyncio.TimeoutError
        return left

    async def ask() -> str:
        """One model turn, bounded by whatever remains of the budget.

        Checking the clock between steps is not enough to enforce `max_seconds`:
        a step already in flight runs to the model's own timeout, so six steps
        could take several times the budget. Bounding the await is what makes the
        limit real.
        """
        return str((await asyncio.wait_for(model.ainvoke(history), timeout=remaining())).content)

    for step_number in range(1, max_steps + 1):
        last_step = step_number == max_steps or time.monotonic() >= deadline
        if last_step:
            history.append(("human", "No steps remain. Answer now with what you have, as {\"answer\": \"...\"}."))

        try:
            raw = await ask()
        except asyncio.TimeoutError:
            # Out of time. The exploration so far is worth more than an
            # exception, so fall back to the last thing the model said.
            logger.warning(f"Ran out of time at step {step_number}; using what we have")
            return _last_answer(history)

        try:
            step = parse_tool_step(raw)
        except ValueError as exc:
            if last_step:
                # It was asked for an answer and gave prose. Prose is the answer.
                logger.warning(f"Final reply was not protocol JSON ({exc}); taking it as the answer")
                return raw.strip()
            logger.warning(f"Could not parse step {step_number} ({exc}); asking again")
            history += [
                ("ai", raw),
                ("human", f"That was not valid protocol JSON ({exc}). Reply with ONLY the JSON object."),
            ]
            continue

        if "answer" in step:
            logger.info(f"Answered after {step_number} step(s)")
            return str(step["answer"]).strip()

        if last_step:
            # Asked for an answer, called a tool anyway. Returning the raw reply
            # would hand back a JSON tool call as the answer, so spend one more
            # call insisting — the exploration so far is still in the history.
            logger.warning("Model called a tool on the final step; asking once more for an answer")
            history += [
                ("ai", raw),
                ("human", "There is no budget to run that tool. Answer now, in plain prose, using what you already know."),
            ]
            try:
                final = (await ask()).strip()
            except asyncio.TimeoutError:
                logger.warning("Ran out of time asking for a final answer")
                return _last_answer(history)
            try:
                return str(parse_tool_step(final).get("answer", final)).strip()
            except ValueError:
                return final

        logger.info(f"Step {step_number} of {max_steps}: calling {step.get('tool')!r}")
        try:
            # FIX: the tool call sits inside the deadline too. Bounding only the
            # model turns let a run overshoot `max_seconds` by the full cost of
            # every tool call it made along the way.
            called = await asyncio.wait_for(_call(tools, step), timeout=remaining())
        except asyncio.TimeoutError:
            logger.warning(f"Ran out of time running {step.get('tool')!r}; using what we have")
            return _last_answer(history)

        result, dropped = trim_text(called, max_tokens=MAX_RESULT_TOKENS)
        logger.info(f"Step {step_number} returned {len(result)} characters")
        if dropped:
            result += f"\n\n[trimmed {len(dropped)} characters]"

        history += [("ai", raw), ("human", f"TOOL RESULT:\n{result}")]

    # Unreachable: the final iteration always returns.
    raise AssertionError("tool loop ended without an answer")
