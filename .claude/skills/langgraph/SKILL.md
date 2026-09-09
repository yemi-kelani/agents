---
name: langgraph
description: Use when building, debugging, or reviewing LangGraph code — graph or functional API, state schemas and reducers, checkpointers, stores, short- and long-term memory, interrupts and human-in-the-loop, streaming, subgraphs, time travel, retries and timeouts, or the React useStream frontend. Also for langgraph.json config, LangSmith Studio and tracing, deployment, and testing graphs.
---

# LangGraph

Reference documentation for LangGraph, trimmed from the official docs. Examples are Python
unless noted; the four `frontend-*` / `agent-chat-ui` references are React/TypeScript.

**Read the one file that matches the task** — they are self-contained and range from 55 to 2,200
lines. Do not read the whole directory.

## Reference map

### Choosing and writing graphs

| Reference | Covers | Read when |
|---|---|---|
| [graph-api-choosing-apis.md](references/graph-api-choosing-apis.md) | Graph vs Functional decision guide; combining and migrating between them | Starting a new graph, or unsure which API fits |
| [graph-api-overview.md](references/graph-api-overview.md) | `StateGraph`, state schema, reducers, `Overwrite`, nodes, edges, `Send`, `Command`, node caching, recursion limit | You need the concepts behind the building blocks |
| [graph-api-usage.md](references/graph-api-usage.md) | Recipes: input/output schemas, private state, pydantic state, sequences, branches, loops, map-reduce, runtime config, retries, timeouts, async | Implementing a specific graph structure |
| [functional-api-overview.md](references/functional-api-overview.md) | `@entrypoint`, `@task`, injectable params, resuming, determinism, idempotency, side-effect pitfalls | Using `@entrypoint`/`@task` instead of a graph |
| [functional-api-usage.md](references/functional-api-usage.md) | Recipes: parallel execution, calling graphs and other entrypoints, retries, timeouts, task caching, resuming after an error, HITL, memory | Implementing something with the functional API |
| [langgraph-runtime.md](references/langgraph-runtime.md) | Actors and channels: `LastValue`, `Topic`, `BinaryOperatorAggregate`, `DeltaChannel` | Debugging how state merged; writing a custom channel |

### State and persistence

| Reference | Covers | Read when |
|---|---|---|
| [persistence.md](references/persistence.md) | Overview plus troubleshooting: `thread_id` too long, `MemorySaver` not persisting, unbounded checkpoint growth | Orienting, or hitting a persistence error |
| [checkpointers.md](references/checkpointers.md) | Threads, checkpoints, `get_state`/`update_state`/history, replay, durability modes, checkpointer libraries, building a custom one | Configuring or implementing a checkpointer |
| [stores.md](references/stores.md) | `BaseStore` put/get/search, namespaces, semantic search, custom stores | Storing data across threads |
| [memory.md](references/memory.md) | Short-term (thread) and long-term (store) memory in production, semantic search, trimming/deleting/summarizing messages, managing checkpoints | Adding memory or managing conversation history |

### Control flow and human-in-the-loop

| Reference | Covers | Read when |
|---|---|---|
| [interrupts.md](references/interrupts.md) | `interrupt()`, resuming, approve/reject, review-and-edit, interrupts in tools, multiple interrupts, and the rules (no try/except around it, no reordering, idempotent side effects) | Any human-in-the-loop pause |
| [time-travel.md](references/time-travel.md) | Replay from a checkpoint, forking, forking across interrupts and subgraphs | Debugging a past run or exploring alternate paths |
| [subgraphs.md](references/subgraphs.md) | Subgraph as a node vs called inside a node, shared vs different schemas, stateful/stateless persistence, viewing and streaming subgraph state | Composing graphs, or wrapping subagents as tools |
| [fault-tolerance.md](references/fault-tolerance.md) | Retry policies, run/idle timeouts, `NodeError`, routing errors with `Command`, graph-wide defaults, graceful drain | Hardening nodes against flaky I/O |

### Streaming

| Reference | Covers | Read when |
|---|---|---|
| [streaming.md](references/streaming.md) | `stream_mode` (values, updates, messages, custom, checkpoints, tasks, debug), v2 format, LLM tokens, subgraph output, arbitrary LLMs, v1→v2 migration | Streaming from a graph, server-side |
| [event-streaming.md](references/event-streaming.md) | Typed projections, `StreamChannel`, transformers, resuming after an interrupt, `ToolCallTransformer` | Building a custom typed stream projection |

### Frontend (React / TypeScript)

| Reference | Covers | Read when |
|---|---|---|
| [frontend-overview.md](references/frontend-overview.md) | Architecture for rendering a graph agent in a UI; how it differs from a chat stream | Starting frontend work |
| [frontend-graph-execution.md](references/frontend-graph-execution.md) | `useStream` setup, mapping nodes to UI cards, node status, progress bar, `NodeCard`, dynamic pipelines | Building a pipeline-progress UI |
| [frontend-custom-stream-channels.md](references/frontend-custom-stream-channels.md) | Custom channels; `useExtension` vs `useChannel` | Pushing custom server-side data to the UI |
| [agent-chat-ui.md](references/agent-chat-ui.md) | Prebuilt Next.js chat UI: quick start, local dev, connecting an agent | You want a chat UI without building one |

### Build and operate

| Reference | Covers | Read when |
|---|---|---|
| [application-structure.md](references/application-structure.md) | Repo layout, `langgraph.json`, dependencies, graph entries, env vars | Setting up or fixing `langgraph.json` |
| [test.md](references/test.md) | Testing graphs, individual nodes and edges, partial execution | Writing tests for a graph |
| [langsmith-studio.md](references/langsmith-studio.md) | CLI install, local agent server, viewing a graph in Studio | Running the graph in Studio locally |
| [langgraph-observability.md](references/langgraph-observability.md) | Enabling tracing, selective tracing, projects, metadata, anonymizers | Setting up LangSmith tracing |
| [deployment.md](references/deployment.md) | Deploying to LangSmith Cloud | Shipping to production |
| [backwards-compatibility.md](references/backwards-compatibility.md) | Changing graph code without breaking in-flight threads; detecting them; non-determinism | Editing a graph that has live threads |

## Routing hints

| Symptom or task | Start with |
|---|---|
| "Pause and ask the user something" | `interrupts.md` |
| "My list state got overwritten / duplicated" | `graph-api-overview.md` (reducers), then `langgraph-runtime.md` (channels) |
| "Stream tokens to a UI" | `streaming.md` server-side, `frontend-graph-execution.md` client-side |
| "Agent forgets across sessions" | `memory.md` (long-term), `stores.md` |
| "Re-run from an earlier step" | `time-travel.md` |
| "A node calls a flaky API" | `fault-tolerance.md` |
| "Resuming an interrupt runs a side effect twice" | `interrupts.md` (rules), `functional-api-overview.md` (idempotency) |

## Notes

- These files are trimmed copies. Doc-site markup (tabs, accordions, callouts, embedded widgets)
  has been flattened to plain Markdown, and repeated per-backend and sync/async variants
  collapsed into one example plus a table. Async usage is noted inline rather than duplicated.
- Cross-references point at `https://docs.langchain.com/...`. For anything not covered here,
  the full documentation index is at `https://docs.langchain.com/llms.txt`.
