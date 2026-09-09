"""The bridge from `trace` to `messages` — spec 12.

Agent Chat UI is message-centric: it renders `messages`, tool calls and
interrupts, and knows nothing about the `trace` and `claims` lists spec 01
carries. With no bridge the interface sits idle for the entire retrieval phase
and then emits a report.

There is a sharper version of that problem, and it is the reason this module
exists rather than being a nicety. **Verification is a graph node, and a node
produces no message at all** — so the step the whole design turns on is the one
that looks like nothing is happening.

`trace` stays the machine-readable channel: the eval harness (11) and any future
non-chat surface read it. `messages` is the human-readable projection of the same
events. Keeping both means the UI choice is not baked into the graph — and it
means there is exactly one place that knows how to describe an event to a human,
which is what keeps the chat transcript and the terminal (`terminal.py`) saying
the same thing.
"""
from __future__ import annotations

import inspect
from functools import wraps

from langchain_core.messages import AIMessage

from render import safe_render

VERDICT_ICONS = {
    "supported": "✓",
    "unsupported": "✗",
    "quote_not_found": "⚠",
    "unverified": "…",
}
"""One symbol per outcome, shared by both surfaces. Rejections are rendered
distinctly on purpose: the rate at which claims fail verification is the health
signal for the pipeline, and a spike usually means a bad extraction prompt or a
backend returning junk."""

CLAIM_CHARS = 90


def describe(event: dict) -> str | None:
    """One line for one trace event, or None for an event a human cannot use.

    Plain text with symbols and backticks rather than rich markup or heavy
    markdown, because the same string is rendered by a markdown chat UI and by a
    terminal. Anything that looks good in only one of them would have to be
    written twice.

    An unrecognised kind is skipped rather than raising: `trace` is the channel
    that grows, and a node added tomorrow must not be able to take the interface
    down with an event nobody taught this function about.
    """
    match event.get("kind"):
        case "plan":
            topics = "\n".join(f"  · {t}" for t in event.get("topics") or [])
            return f"Brief: {event['brief']}" + (f"\n{topics}" if topics else "")
        case "search":
            # `.get`, not `[...]`: a run checkpointed before this field existed
            # is resumed against this code, and a surface that subscripts a key
            # a node did not write dies on exactly that resume.
            backend = event.get("backend")
            line = f"`search` {event['query']} → {event['n_results']} results"
            return line + (f" ({backend})" if backend else "")
        case "fetch":
            return f"`fetch` {event['url'][:80]} ({event['status']})"
        case "fetch_failed":
            return f"`fetch` {event['url'][:80]} — failed: {event['error']}"
        case "quote_not_found":
            return (f"{VERDICT_ICONS['quote_not_found']} quote not in source: "
                    f"\"{event['quote']}\"")
        case "claim_dropped":
            return f"  ↳ dropped: {event['text']} ({event['reason']})"
        case "verdict":
            icon = VERDICT_ICONS.get(event["verdict"], "·")
            line = f"{icon} {(event.get('text') or event['claim_id'])[:CLAIM_CHARS]}"
            # The reason only earns a line when the claim did not survive. On a
            # pass it restates the quote nobody disputed.
            if event["verdict"] != "supported" and event.get("reason"):
                line += f"\n  ↳ {event['reason']}"
            return line
        case "sufficiency":
            if not event.get("continuing"):
                return f"Coverage: done ({event['reason']})."
            return ("Coverage: gaps found, another round — "
                    + "; ".join(event.get("gaps") or []))
        case "clarify_asked":
            # The one step that stops the run and waits on a human. It has been
            # traced since spec 02 and rendered nowhere, so the pause looked
            # like the pipeline had simply stopped.
            return f"Asking {event['n']} follow-up question(s)…"
        case "clarify_skipped":
            return f"Proceeding on the assumption: {event['assumption']}"
        case "clarify_answered":
            return f"Clarified ({event['n']} answered)."
        case "report":
            return (f"Report: {event['n_supported']} verified claim(s) cited, "
                    f"{event['n_rejected']} rejected.")
    return None


def progress(events: list[dict]) -> list[AIMessage]:
    """One compact status message for one node's events.

    One message per *node*, not per event: `add_messages` appends, and a message
    per fetch turns the transcript into a wall.

    Run through `safe_render` because claim text and verdict reasons are written
    by a model out of untrusted page content, and the chat UI renders markdown —
    an injected image here is a live outbound GET the moment the transcript
    displays, exactly as it would be in the report (08).
    """
    lines = [line for line in map(describe, events) if line]
    return [AIMessage(content=safe_render("\n".join(lines)))] if lines else []


def with_progress(node):
    """A node that also tells the interface what it just did.

    Applied by `build_graph` to every node rather than written into each one:
    a projection each author has to remember is a projection somebody will
    forget, and the node they forget it in is the one whose progress mattered.
    Same reasoning as verification being a node instead of a tool.
    """
    if inspect.iscoroutinefunction(node):
        @wraps(node)
        async def wrapper(*args, **kwargs):
            return _projected(await node(*args, **kwargs))
    else:
        @wraps(node)
        def wrapper(*args, **kwargs):
            return _projected(node(*args, **kwargs))

    return wrapper


def _projected(update):
    """Add the transcript to a node's update, when it has anything to say.

    An empty list is still a write to a channel with a reducer, so a node that
    traced nothing writes no key at all.
    """
    if not isinstance(update, dict):
        return update

    messages = progress(update.get("trace") or [])
    return update | {"messages": messages} if messages else update
