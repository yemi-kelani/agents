"""The terminal fallback — spec 12.

Kept because it has fewer moving parts than a served graph and a chat frontend,
and because the eval harness (11) drives the graph directly with no server
involved. It renders the same trace events the chat transcript does, through the
same `describe` — two surfaces, one description, so they cannot drift into
telling different stories about the same run.
"""
from __future__ import annotations

from langchain_core.messages import AIMessageChunk
from rich.console import Console
from rich.markup import escape

from nodes.synthesize import SYNTHESIZE_NODE
from progress import VERDICT_ICONS, describe

ICON_STYLES = {
    VERDICT_ICONS["supported"]: "green",
    VERDICT_ICONS["unsupported"]: "red",
    VERDICT_ICONS["quote_not_found"]: "yellow",
}


class TerminalUI:
    """Rich rendering of a live run."""

    def __init__(self, console: Console | None = None):
        self.console = console or Console()

    def emit(self, event: dict) -> None:
        line = describe(event)
        if line:
            self.console.print(_styled(line))

    def stream_token(self, text: str) -> None:
        """The report as the synthesizer writes it. No newline and no markup:
        tokens arrive mid-word, and a partial token is not valid anything."""
        self.console.print(escape(text), end="", markup=False, highlight=False)


async def stream_run(graph, payload: dict, config: dict, *,
                     ui: TerminalUI | None = None) -> dict:
    """Drive a run to completion, rendering as it goes, and return the state.

    Reads `updates` for trace events and `messages` for report tokens. `trace`
    rather than the `messages` channel, deliberately: the terminal is the
    non-chat surface, and having it consume the chat projection would make the
    projection load-bearing for both — the thing spec 01 keeps two channels to
    avoid.

    Two filters, because two different things end up on that channel.

    Only *chunks* count as report tokens. The `messages` stream carries anything
    written to that channel, and since every node now writes its progress there
    (`with_progress`), a terminal that took all of it would print each line
    twice — once from the trace it rendered itself, once from the projection
    meant for the chat UI. A model streaming its answer yields
    `AIMessageChunk`; a node writing a finished line yields `AIMessage`.

    And only chunks *from the synthesizer*. Asking for `stream_mode="messages"`
    attaches a streaming callback handler, which makes langchain stream every
    model call in the graph — so the extractor's structured output arrives here
    token by token as raw schema JSON, in the same `AIMessageChunk` type the
    report comes in. The chunk type cannot separate them; `langgraph_node` can.
    Without this the report is interleaved with JSON fragments from however many
    topics are being researched concurrently.

    `.get`, not `[...]`: metadata is filled in by the runtime, and a surface that
    subscripts it dies on the one chunk that arrives without a node.
    """
    ui = ui or TerminalUI()

    async for mode, chunk in graph.astream(payload, config,
                                           stream_mode=["updates", "messages"]):
        if mode == "updates":
            for _node, delta in chunk.items():
                if not isinstance(delta, dict):
                    continue
                for event in delta.get("trace") or []:
                    ui.emit(event)
        elif mode == "messages":
            token, meta = chunk
            if (isinstance(token, AIMessageChunk)
                    and meta.get("langgraph_node") == SYNTHESIZE_NODE):
                ui.stream_token(token.text)

    return (await graph.aget_state(config)).values


def _styled(line: str) -> str:
    """Colour the verdict symbol, and escape everything else.

    Escaped because Rich reads square brackets as style tags and these lines
    carry URLs and model-written claim text — an unescaped `[` from a search
    result is a crash the agent handed itself.
    """
    style = ICON_STYLES.get(line[:1])
    return f"[{style}]{line[0]}[/]{escape(line[1:])}" if style else escape(line)
