"""The served graph — spec 12.

`langgraph dev` imports this module and asks it for a graph. A factory rather
than a module-level compiled attribute, and that is not a style preference:
building the graph eagerly would construct the models and the search router at
*import* time, so merely importing this module would need a provider key and a
search key. The server has them; a test collecting the package does not.

    pip install "langgraph-cli[inmem]"
    langgraph dev                      # http://localhost:2024

Then either point the hosted UI at the local server (no Node toolchain needed) —
agentchat.vercel.app, deployment URL http://localhost:2024, graph id `research`,
no LangSmith key required for a local server — or run the frontend locally:

    npx create-agent-chat-app --project-name research-ui
    cd research-ui && pnpm install && pnpm dev

The server cannot pass constructor arguments, so anything that varies per run
travels on the input payload instead: the run profile is a `Budget` (`FAST` /
`FULL` in `state.py`) sent with the question.
"""
from __future__ import annotations

from config import load_config
from graph import build_graph_from_config


def make_graph(config=None):
    """Build the graph the server serves.

    Takes the server's run config and ignores it: the profile that matters here
    is the model profile, which comes from `config.yaml` and
    `RESEARCHER_PROFILE` (10) — so pointing the whole deployment at local models
    is an environment variable and a restart.
    """
    return build_graph_from_config(load_config(), served=True)
