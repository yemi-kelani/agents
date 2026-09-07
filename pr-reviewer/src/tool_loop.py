"""A tool-calling loop driven from Python.

`ShellChatModel.bind_tools` raises: the CLIs behind it have no native tool
calling. So the protocol is text — the model replies with a JSON object naming a
tool or carrying a final answer, and this loop dispatches it.

Mechanism, not a node: a node picks the task and the tools, this runs the turns.
"""

from __future__ import annotations

import inspect
import time
from typing import Awaitable, Callable

from langchain_core.language_models import BaseChatModel

from log import get_logger
from parsers import parse_tool_step
from prompts import load
from utilities import trim_text

logger = get_logger()

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

    for step_number in range(1, max_steps + 1):
        last_step = step_number == max_steps or time.monotonic() >= deadline
        if last_step:
            history.append(("human", "No steps remain. Answer now with what you have, as {\"answer\": \"...\"}."))

        raw = str((await model.ainvoke(history)).content)

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
            final = str((await model.ainvoke(history)).content).strip()
            try:
                return str(parse_tool_step(final).get("answer", final)).strip()
            except ValueError:
                return final

        logger.info(f"Step {step_number}: calling {step.get('tool')!r}")
        result, dropped = trim_text(await _call(tools, step), max_tokens=MAX_RESULT_TOKENS)
        if dropped:
            result += f"\n\n[trimmed {len(dropped)} characters]"

        history += [("ai", raw), ("human", f"TOOL RESULT:\n{result}")]

    # Unreachable: the final iteration always returns.
    raise AssertionError("tool loop ended without an answer")
