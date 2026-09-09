# Grounded Research Agent — Architecture

A research agent that answers a question by searching the web, extracting claims
anchored to verbatim source quotes, and **verifying every claim against its source
before it reaches the report**.

The differentiator is not the pipeline. It is the measurement: this repo ships an
eval harness for **citation faithfulness**, and every claim traverses a mandatory
verification node before it can be cited.

---

## The thesis

Most research agents are judged on whether the report "reads well." That is the
wrong metric, and the literature says so: an audit of four commercial generative
search engines found only ~51.5% of generated statements were fully supported by
their citations, and citation precision was *inversely* correlated with perceived
utility (Liu, Zhang & Liang, 2023, arXiv:2304.09848). A better-reading report was
a less-grounded one.

So this agent measures the thing that actually matters — **does the cited source
support the claim** — and it puts that check on the execution path rather than in a
prompt:

> Every claim passes through a **mandatory verification node**. There is no
> affordance for skipping it, because an advisory affordance changes what a model
> *can* do and structure changes what it *must* do.

The measurement reports both mean accuracy and consistency, because a pipeline that
is right on average and unstable run-to-run is not a grounded one. Metric
definitions follow τ-bench (Yao et al., 2024, arXiv:2406.12045), which introduced
`pass^k` — the probability an agent succeeds on *all* k attempts. `Pass^k = p^k`
decays exponentially, so a 90%-accurate agent is only ~73% consistent at k=3. That
sensitivity is the point.

---

## Shape of the system

```
                    ┌─────────────────┐
  user question ───►│ clarify (HITL)  │  tool loop over `ask_user`, which
                    │   tool loop     │  interrupts — at most MAX_ROUNDS
                    └────────┬────────┘  (no tool call = fall through)
                             ▼
                    ┌─────────────────┐
                    │      plan       │  question ──► brief + 3-5 topics
                    └────────┬────────┘
                             ▼
                    ┌─────────────────┐
                    │   fan-out (Send)│
                    └─┬──────┬──────┬─┘
          ┌───────────┘      │      └───────────┐
          ▼                  ▼                  ▼
   ┌────────────┐     ┌────────────┐     ┌────────────┐
   │ research   │     │ research   │     │ research   │   one per topic,
   │  topic A   │     │  topic B   │     │  topic C   │   run concurrently
   └─────┬──────┘     └─────┬──────┘     └─────┬──────┘
         │ search → fetch → extract(claims+quotes)
         └───────────┬──────┴───────────────────┘
                     ▼
            ┌───────────────────────────┐
            │  verify (MANDATORY node)  │  ◄── on the path for every claim;
            └────────┬──────────────────┘      no opt-out exists
                     ▼
            ┌──────────────────┐
            │   sufficiency    │  ─── gaps? ──► back to fan-out (bounded)
            └────────┬─────────┘
                     ▼
            ┌──────────────────┐
            │    synthesize    │  only verified claims are citable
            └──────────────────┘
```

---

## Decisions, and why

**Hand-built `StateGraph`, not `create_agent`.** `create_agent` is the right
default for a tool-calling loop, but this workflow mixes deterministic steps
(fetch, verify, render) with agentic ones (extract, synthesize) and needs dynamic
fan-out/fan-in over a topic list. That is exactly the case the LangGraph docs
carve out for a custom graph. It is also what makes verification enforceable: a
node sits on the edge between research and synthesis, where a tool in a tool list
would only sit in the model's options.

**Tool-calling lives at clarification, and only there.** The pipeline is
otherwise deterministic steps and structured-output calls; `clarify` is the one
node where the model drives control flow, and it is bounded by `MAX_ROUNDS` and
the `clarified` latch rather than by trust. What it buys is a follow-up
conditioned on the previous answer — "compare observability vendors" → "which
ones?" → "the open-source ones" is still too vague to research, and a single
blind round cannot chase it.

The asymmetry with `verify` is the point, not an oversight: an unasked question
wastes a budget and shows up as hedging in the report, while an unverified claim
corrupts it silently. Only the second gets a node with no opt-out. Spec
[02](components/02-clarification.md) records what that trade cost.

**Verification is a graph node, not a prompt instruction.** You cannot prompt your
way to a guarantee. If the requirement is "every claim in the report was checked,"
the check has to be on the execution path. This is the same reasoning behind
enforced conclusion gates in incident-investigation harnesses — a "consider
alternatives" instruction the model may ignore is not a control.

**Two-tier verification, cheap tier first.**
1. **Quote grounding** — deterministic. Does the quoted span appear verbatim in
   the fetched source text? Normalized substring match. Zero LLM cost, catches
   fabricated quotes outright.
2. **Claim entailment** — does that verbatim quote actually support the claim?
   LLM or NLI model.

Tier 1 is free and catches the worst failure. Tier 2 only runs on quotes that
passed tier 1, so verification cost scales with real evidence, not with model
enthusiasm.

**Claims carry quotes with character offsets.** A citation you cannot mechanically
check is not a citation. Anchoring to `(source_id, start, end)` is what makes the
eval possible at all.

**"Unsupported" is a first-class outcome.** A claim that fails verification is
demoted, flagged in the report, or dropped — never silently re-cited to a
different source. An agent that says "I could not confirm this" is more useful
than one that always finds something to point at.

**No blobs in state.** LangGraph serializes state on every checkpoint write.
Fetched page text lives in a content store keyed by `source_id`; state carries the
key. Ignoring this makes checkpointing quadratic and the whole run visibly slow.

**Reducers on every list written by parallel branches.** Concurrent branches
writing an un-annotated key raise `InvalidUpdateError`. `Annotated[list[X], add]`
is not optional here.

**Additional input.** Organize the repository efficiently and in a common sense fashion.
Topics should be grouped, e.g. steaming logic goes in `streaming.py`. Prompts go in 
`prompts/xyz.txt ... prompts/abc.txt` with a `utilities.py` or `prompts.py` handler to fetch them.
EVERYTHING MUST OBEY DRY AND YAGNI PRINCIPLES.

---

## Components

| Spec | Component | Priority |
|---|---|---|
| [01](components/01-state.md) | State schema, content store, reducers | **P0** |
| [02](components/02-clarification.md) | Clarification tool loop (HITL) | P1 |
| [03](components/03-planning.md) | Brief + topic decomposition | **P0** |
| [04](components/04-search.md) | Pluggable search backends | **P0** |
| [05](components/05-fetch.md) | Fetch, SSRF guard, injection containment | **P0** |
| [06](components/06-extraction.md) | Claim + quote extraction with offsets | **P0** |
| [07](components/07-verification.md) | Mandatory claim verification | **P0** |
| [08](components/08-synthesis.md) | Report generation | **P0** |
| [09](components/09-sufficiency.md) | Iterative deepening loop | P2 |
| [10](components/10-models.md) | Multi-provider + local models | P1 |
| [11](components/11-eval.md) | Citation-faithfulness harness | **P0** |
| [12](components/12-interface.md) | Agent Chat UI + terminal fallback | **P0** |

**P0 is the minimum coherent system** — it answers a question and measures whether
it was grounded. P1 is quality of life. P2 is designed but deferred; the spec
exists so the design is settled when it gets built.

Build order: 01 → 04 → 05 → 06 → 07 → 03 → 08 → 12 → 11. Get one topic flowing
end-to-end before adding fan-out; parallelism on a broken pipeline is just
concurrent failure.

---

## What is deliberately not built

- **Multi-agent orchestration.** Parallel topic research inside one process gets
  the same breadth without a 15× token bill or an unbounded subagent tree.
- **Vector store / embeddings retrieval.** The corpus is what this run fetched.
  Semantic search over a handful of documents is machinery without a purpose.
- **MCP server per search backend.** Architecturally clean, genuinely right for a
  multi-team deployment, pure overhead at this size. A Python `Protocol` is the
  same boundary without the transport.
- **Fine-grained sandboxing (gVisor, per-run containers).** The SSRF guard is
  where the real risk is for a fetch-only agent. Kernel isolation starts to matter
  once code execution is added.
- **Prompt-injection classifiers.** Defense-in-depth at best; the containment
  approach in [05](components/05-fetch.md) is the load-bearing part.

---

## Known unknowns

- **Whether the consistency numbers survive a larger eval.** n=12 claims × 3 runs
  is underpowered for anything but a large effect.
- **Whether the LLM verifier agrees with human labels.** The harness reports
  verifier-vs-human agreement precisely because the verifier is the weakest link.
  NLI entailment models beat LLM judges on this task; swapping one in is the first
  upgrade.
- **Partial support.** Entailment checking is bad at "the source kind of implies
  this." The current design collapses it to supported/unsupported and undercounts
  a real failure mode.
- **Whether enforced structure locks in wrong interpretations.** Consistency
  amplifies whatever the agent decided. A mandatory node that verifies against a
  badly-chosen source makes the agent reliably wrong.
