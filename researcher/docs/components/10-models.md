# 10 — Model Layer: Multi-Provider & Local

**Priority: P1.**

## Purpose

Route each task to an appropriate model, and make local models a first-class path
so the whole pipeline can be debugged for free before spending the provided API
key.

## Roles, not "a model"

Four distinct jobs with different requirements:

| Role | Needs | Default | Local option |
|---|---|---|---|
| `planner` | reasoning, decomposition | `gpt-4.1` | weak locally |
| `extractor` | volume, strict JSON, verbatim copying | `gpt-4.1-mini` | Qwen3 works |
| `verifier` | careful judgment, different family than extractor | `gpt-4.1-mini` | Qwen3 / NLI model |
| `synthesizer` | long-form writing, instruction adherence | `gpt-4.1` | weak locally |

```yaml
# config.yaml
models:
  planner:     openai:gpt-4.1
  extractor:   openai:gpt-4.1-mini
  verifier:    openai:gpt-4.1-mini
  synthesizer: openai:gpt-4.1

# local profile — same shape, zero cost
models_local:
  planner:     ollama:qwen3:32b
  extractor:   ollama:qwen3:32b
  verifier:    ollama:qwen3:32b
  synthesizer: ollama:qwen3:32b

# how much fits in ONE call, per role. Distinct from Budget.max_total_tokens,
# which caps the whole run. Optional — omit a role to use the provider default.
context_limits:
  extractor:   128000
  synthesizer: 128000

context_limits_local:
  extractor:   32000
  synthesizer: 32000
```

```python
# models.py
from langchain.chat_models import init_chat_model

def build_models(cfg: dict) -> dict[str, BaseChatModel]:
    return {role: init_chat_model(spec, temperature=0)
            for role, spec in cfg.items()}
```

`init_chat_model` handles provider dispatch. Ollama, vLLM, and LM Studio all speak
the OpenAI-compatible `/v1` protocol, so a local model is a `base_url` change —
that's the whole local story. LiteLLM as a proxy is the heavier version if you
want centralized key handling, rate limiting, and a PII-redaction hook in one
place; note it in the README, don't build it.

## Verifier ≠ extractor

Deliberate: LLM-as-judge exhibits **self-preference bias** — models rate their own
output more favorably. Using the same model to extract a claim and then judge
whether it's supported inflates the pass rate. Running the verifier on a different
family (or an NLI model) removes that.

Cheap version if budget is tight: same family, different size. Better than
identical.

## Local models — honest limits

- **Tool calling degrades on multi-step work.** Roughly 90%+ well-formed calls on
  simple workloads, but 80–90% end-to-end on multi-step flows once selection and
  argument errors compound. Under ~7B, coherence collapses past 2–3 steps.
- **The failure nobody advertises is self-termination** — small models struggle to
  recognize "the tool succeeded, I'm done" and loop. This is a real argument for
  the mandatory-node design in spec 07: a node has no termination decision to get
  wrong, so a weak model costs accuracy without also costing control flow.
- **JSON validity is solvable** — Ollama and llama.cpp constrain output at the
  token level via grammars. JSON *correctness* (right fields, faithful quotes) is
  not.
- **Verbatim quoting is where local models hurt most here.** They paraphrase.
  Track your `quote_not_found` rate per model — it's a direct, measurable
  indicator, and reporting it is a much better answer than "the small model was
  worse."

Because the eval is model-agnostic, the local-vs-frontier comparison produces
numbers rather than impressions.

## Workflow

1. Build and debug on Qwen3 via Ollama. Free, and it surfaces tool-calling and
   JSON bugs *faster* than a frontier model, which papers over sloppy prompts.
2. Run the eval on the hosted model.
3. Run the same eval on the local profile and report both.

| | Pass@3 | Pass^3 | quote_not_found | cost | wall-clock |
|---|---|---|---|---|---|
| gpt-4.1-mini | | | | | |
| qwen3:32b | | | | | |

The second row quantifies what the local model costs in accuracy.

## Context limits are not the token budget

`Budget.max_total_tokens` (spec 01) caps what the whole run *spends*. A
`context_limit` caps what one call can *hold*. The two move independently: a 200k
run budget is perfectly fine on a 32k local model, because no single call in this
pipeline comes near 32k.

Two consumers, neither built on day one:

- **Extraction (06)** hard-truncates page text at `text[:12000]` — about 3k
  tokens, safe everywhere, but a magic number rather than a derived one. It
  should come from `context_limits[extractor]`, and it is the number that has to
  grow when map-reduce chunking replaces the truncation.
- **Synthesis (08)** is the call that can actually overflow. It receives every
  verified claim plus the rejected block in a single prompt, so its size grows
  with `max_topics × max_rounds` — the low tens of thousands of tokens at the
  default budget, which is tight on a 32k local window and irrelevant on a
  frontier one. Past `context_limits[synthesizer]`, compress the claim block
  (drop evidence snippets first, then group claims by source) rather than letting
  the call fail.

Both live in the model config rather than in `Budget` for the same reason: the
window size is a property of the model, and the spend cap deliberately is not.

## Cost control

```python
class Budgeted:
    """Wrapper that counts tokens and raises past the cap."""
    def __init__(self, model, budget: Budget):
        self._m, self._b = model, budget

    async def ainvoke(self, *a, **kw):
        if self._b.tokens_used >= self._b.max_total_tokens:
            raise BudgetExceeded(f"{self._b.tokens_used} tokens")
        r = await self._m.ainvoke(*a, **kw)
        self._b.tokens_used += (r.usage_metadata or {}).get("total_tokens", 0)
        return r
```

Also cache search and fetch results to disk. Eval runs repeat the same queries
dozens of times; without a cache most of the wall-clock is network wait and most
of the quota goes on identical requests.

```python
@disk_cache(ttl_hours=24)
async def cached_search(backend: str, query: str, k: int): ...
```

## Acceptance

- Switching to the local profile is a config change only.
- Verifier and extractor default to different models.
- Token spend is capped and enforced.
- Search/fetch results are cached across runs.
