# pr-reviewer

An agent that reviews a diff and posts findings to a pull request.

## Setup

```bash
cd agents
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
npm install -g @openai/codex@0.153.4
```

Required environment:

Create a `.env` folder at root:

```bash
touch .env && chmod 600 .env
```

Set the following variables:

| Variable | Purpose |
|---|---|
| `OPENAI_API_KEY` | The model key. Passed to the codex CLI as `CODEX_API_KEY`. |
| `GITHUB_TOKEN` | (Optional) For reading and posting reviews. Not needed for a saved diff. |

On macOS, python.org builds cannot verify TLS until you run
`/Applications/Python\ 3.10/Install\ Certificates.command` once. Otherwise
export `SSL_CERT_FILE=$(python3 -c "import certifi;print(certifi.where())")`.

## Run the demo yourself

Fork or clone this repo. The `demo-flaws` branch adds a file with deliberate
bugs for the reviewer to find.

Fastest, no pull request and no `GITHUB_TOKEN`:

```bash
git clone https://github.com/yemi-kelani/agents && cd agents/pr-reviewer/src
python review.py origin/main..origin/demo-flaws
```

To drive it from a real pull request in your own fork:

1. Open a PR in your fork, `demo-flaws` into `main`. It is a same repo PR, so
   it is not treated as a fork PR and the workflow guard passes.
2. Review it locally with `python review.py <your-fork>#<n>`, or let CI do it.
3. For CI, enable Actions on the fork (disabled by default on forks) and add
   `OPENAI_API_KEY` to the fork's repository secrets. Set `MAX_REVIEWS_PER_PR`
   to `none` if you want every push reviewed.

## Test locally

No CI, nothing posted. Run from `pr-reviewer/src`. If you need a test repository, for

```bash
python review.py main..HEAD                 # a local revision range
python review.py owner/repo#3               # a pull request, fetched over the API
python review.py https://github.com/owner/repo/pull/3
python review.py saved.diff                 # offline, no token needed
```

Save a diff for offline use:

```bash
gh api repos/owner/repo/pulls/3 -H "Accept: application/vnd.github.v3.diff" > saved.diff
```

Options: `--cli` picks the agent CLI, `--repo` sets the checkout the sub-agent
explores, `-q` prints findings only.

## Test via pull request

Push a branch and open a PR against `main`. The workflow runs on
`pull_request` and posts one comment.

Two things decide whether you see a review:

1. The reviewer only sees the diff. If the code you want reviewed is already on
   `main`, it is not in the diff and nothing will be reported. Add it on the
   branch instead.
2. `MAX_REVIEWS_PER_PR` (default `1`) counts comments already posted. Once a PR
   has been reviewed it is skipped on every later push. Set it to `none` while
   iterating.

Exit codes:

| Code | Meaning |
|---|---|
| 0 | Reviewed, or correctly skipped as already reviewed |
| 1 | Ran and failed: model error, GitHub unreachable, or budget exceeded |
| 2 | Misconfigured: missing token, repository, or PR number |

A green check with no comment means the run was skipped, not that the code is
clean.

## Test via fork

Fork pull requests are skipped on purpose. GitHub withholds secrets from a
fork's `pull_request` run and downgrades `GITHUB_TOKEN` to read only, so the
reviewer would have no model key and no write access. The job guard is:

```yaml
if: github.event.pull_request.head.repo.full_name == github.repository
```

To confirm the behaviour, open a PR from a fork and check that the job does not
run. To review the same diff anyway, fetch it and run locally:

```bash
python review.py https://github.com/owner/repo/pull/<n>
```

Supporting forks in CI needs a second `workflow_run` job that runs in the base
repository's context. It must treat the head commit as untrusted and never
execute code from it.

## Tests

```bash
cd pr-reviewer
python -m pytest tests -q
```
