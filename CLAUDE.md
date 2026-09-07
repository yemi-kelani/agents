# Working in this repository

## Never write a parser against a schema you have not seen

This codebase drives external agent CLIs by parsing their output, and that is
where its worst bugs have come from. Both parsers were originally written from
assumptions about the output format, and both were wrong:

- `parse_codex` assumed `codex exec --json` emitted flat records with a
  top-level `message`. The real stream is typed events, where a top-level
  `message` belongs to an **error** event and the agent's text is nested under
  `item`. Trusting the assumption would have posted a 401 error string to a pull
  request as if it were a code review.
- `parse_bob` still carries `UNVERIFIED: field names are inferred` in its own
  docstring. It has never been run against real output.

Before writing or changing any code that parses CLI output: capture a real
transcript and read it. If you cannot capture one (no credentials, no network),
say so and stop — do not infer the schema from the flag name, the SDK docs, or
the surrounding code. Prefer an output contract that needs no parsing at all,
the way `--output-last-message` writes the final message to a file.

## Verify framework behavior before building on it

Assumptions about LangGraph and LangChain have been wrong here too:

- Mutating the state dict inside a conditional-edge routing function does **not**
  propagate to the graph's result. State updates must be *returned* from a node.
  The gate is a node plus a router for exactly this reason.
- `Runnable.abatch` defaults to `return_exceptions=False`, so one failing item
  discards every other result.

Both took one short script to check. Run the script rather than reasoning about
what the library probably does.

## Failure must never look like success

This is a review gate: a run that did not happen must not be indistinguishable
from a pull request with no problems. Misconfiguration exits 2, an actual failure
exits 1, and only a review that genuinely ran exits 0. Findings are delivered by
the posted comment, not by the exit code — do not fail the build for them.
