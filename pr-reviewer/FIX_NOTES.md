# Known issues not addressed by the reliability fix

Found while fixing the CI/output/exit-code defects. Each is real but out of
scope; none blocks a working review.

## 1. The `bob` CLI backend is entirely unverified

`get_model("bob")` is reachable through `REVIEW_CLI`, but nothing exercises it
and `parse_bob` says so itself:

> UNVERIFIED: field names are inferred from a third-party SDK. Check against
> real output before trusting.

This is exactly the fault that broke the codex path: a parser written against a
guessed schema. The codex fix worked only because a real `--json` transcript was
captured first and showed the guess was wrong — top-level `message` belongs to
*error* events, so the old parser would have reported a CLI failure as the
review body.

**Recommended:** capture one real `bob --output-format stream-json` transcript
and write tests against it, or remove `"bob"` from `CLI_NAMES` until someone
does. Do not "fix" `parse_bob` by reasoning about what the fields probably are.
The `--max-coins 30` budget flag is likewise unexplained.

## 2. `parse_critiques` splits blocks on any line-initial `FILE:`

`_BLOCK` matches `^[ \t]*FILE:` anywhere, including inside a `DETAIL` body. A
model that quotes the report format back — plausible when reviewing this very
repository — has its finding cut in two, and the malformed remainder is silently
dropped by the `not fields["FILE"]` guard.

Reproduces today:

```python
parse_critiques("FILE: a.py\nISSUE: x\nDETAIL: see\nFILE: b.py\nmore")
# -> one finding, detail truncated to "see"
```

**Recommended:** require a blank line before a block delimiter, or have the
prompt ask for an explicit separator between findings. Narrow enough to leave
alone until it is observed in practice.

## 3. `max_reviews_per_pr` cannot be set to "unlimited" from the environment

`parse_int(os.getenv("MAX_REVIEWS_PER_PR"), 1)` never returns `None`, so the
`if s.max_reviews_per_pr is None` branch in `should_review` — the one that skips
the GitHub API call entirely — is unreachable from configuration. Only direct
construction of `Settings` can trigger it.

**Recommended:** treat a sentinel such as `MAX_REVIEWS_PER_PR=none` as unlimited,
or drop the unreachable branch. Note `MAX_REVIEWS_PER_PR=0` currently means
"never review", which is a coherent setting worth keeping.

## 4. Re-review semantics after new commits

The dedup marker counts reviews across the PR's entire lifetime, so once the
agent has reviewed a PR it will never review it again — even after the author
pushes substantial changes in response to that very review. This is a product
decision rather than a defect, but it is almost certainly not what a reviewer
should do.

**Recommended:** stamp the reviewed head SHA into the marker
(`<!-- pr-reviewer:v1 sha=... -->`) and re-review when the head has moved,
keeping `max_reviews_per_pr` as the ceiling.

## 5. Generated files consume the model budget

`get_diff` returns the whole diff. A PR that regenerates a lockfile, vendors a
dependency, or applies a bulk reformat spends segments — and money — summarizing
churn no reviewer wants. `MAX_SEGMENTS` now bounds the cost, but it does so by
truncating, which means the generated noise can crowd out the real changes.

**Recommended:** exclude common generated paths via pathspec arguments to
`git diff` (`':(exclude)*.lock'`, `':(exclude)vendor/**'`), ideally configurable.

## 6. `README.md` documents only how to create a virtualenv

It says nothing about required environment variables, the two CLI backends, how
to run the reviewer locally, or what the exit codes mean — all of which now
matter, since exit code 2 (misconfigured) is distinct from 1 (failed) and 0.
