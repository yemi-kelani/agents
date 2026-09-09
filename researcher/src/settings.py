"""Environment and secrets — the one place that knows a variable's name.

Every key the pipeline needs is read here, through `os.getenv(NAME, default)`,
and nowhere else. That is not tidiness: a key read inline at three call sites is
a key that gets a different default at two of them, and a `.env` half the entry
points forget to load.

**This module imports nothing from the project.** It sits at the bottom of the
stack so that `config`, `search`, `cache` and `models` can all reach it without
importing each other — `config` already imports `search` and `cache`, so putting
the environment layer there directly would close a cycle. `config` re-exports
`settings` for readers who look for configuration under that name.

**Two kinds of variable, and only one of them is here.** Variables the provider
SDKs read for themselves — `LANGSMITH_API_KEY`, `LANGCHAIN_TRACING_V2`,
`OPENAI_BASE_URL` — start working the moment `.env` is loaded and need no entry
of their own. `Settings` names the ones *this code* reads.

`settings()` reads the environment on every call rather than caching a snapshot.
A process that changes its environment should see the change, and a cached
snapshot is the kind of thing that is only discovered after an hour of looking
somewhere else.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, fields
from pathlib import Path

from dotenv import load_dotenv

DEFAULT_SEARXNG_URL = "http://localhost:8080"
DEFAULT_PROFILE = "default"
DEFAULT_LOG_LEVEL = "WARNING"
"""Silent. The tool log (`toollog`) is a thing you turn on, not a thing you
turn off — a run that asked for nothing prints what it always printed."""

PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google_genai": "GOOGLE_API_KEY",
    "groq": "GROQ_API_KEY",
    "mistralai": "MISTRAL_API_KEY",
    "fireworks": "FIREWORKS_API_KEY",
}
"""Provider prefix -> the variable holding its key.

Keyed by the prefix of a `provider:model` spec (10), so a model config names its
own credential without anything having to restate the mapping. Providers absent
from this table need no key — `ollama` above all, which is the entire point of
the local profile.
"""

SECRET_FIELDS = frozenset({
    "openai_api_key", "anthropic_api_key", "tavily_api_key",
})
"""Fields masked in `repr`. Settings end up in a traceback or a debug log
eventually, and the default dataclass repr would print the key in it."""


def find_env_file(start: Path | None = None) -> Path | None:
    """The nearest `.env`, searching upward from this file.

    Upward rather than at a fixed depth: the file sits at the repository root,
    above the project, and a hard-coded `parents[2]` breaks the first time
    somebody moves a directory.
    """
    for directory in (start or Path(__file__).resolve()).parents:
        candidate = directory / ".env"
        if candidate.is_file():
            return candidate
    return None


def load_env(path: str | Path | None = None, *, override: bool = False) -> bool:
    """Populate `os.environ` from a `.env` file. True if one was read.

    `override=False` is standard dotenv semantics and load-bearing in
    deployment: a server sets real variables, and a stale file baked into an
    image must not win over them.

    A missing file is not an error. Deployment sets real variables and ships no
    file, so raising here would make the working configuration the one that
    crashes.
    """
    path = Path(path) if path is not None else ENV_FILE
    if path is None or not Path(path).is_file():
        return False
    return bool(load_dotenv(path, override=override))


@dataclass(frozen=True)
class Settings:
    """What this code reads out of the environment."""

    openai_api_key: str | None = None
    anthropic_api_key: str | None = None
    tavily_api_key: str | None = None

    searxng_base_url: str = DEFAULT_SEARXNG_URL
    profile: str = DEFAULT_PROFILE
    cache_dir: Path | None = None
    log_level: str = DEFAULT_LOG_LEVEL

    def provider_key(self, spec: str) -> str | None:
        """The API key for the provider named by a `provider:model` spec.

        Split on the first colon only: `ollama:qwen3:32b` is a provider and a
        model that happens to contain one.
        """
        provider = spec.split(":", 1)[0]
        variable = PROVIDER_KEY_ENV.get(provider)
        return os.getenv(variable) or None if variable else None

    def __repr__(self) -> str:
        shown = [
            f"{f.name}=" + (
                _mask(getattr(self, f.name)) if f.name in SECRET_FIELDS
                else repr(getattr(self, f.name))
            )
            for f in fields(self)
        ]
        return f"Settings({', '.join(shown)})"


def settings() -> Settings:
    """Read the environment now.

    `os.getenv(NAME, default)` throughout, and `or None` on the secrets: an
    empty string is a value, and a provider handed one reports a confusing auth
    error instead of the obvious "you have no key".
    """
    cache_dir = os.getenv("RESEARCHER_CACHE_DIR")
    return Settings(
        openai_api_key=os.getenv("OPENAI_API_KEY") or None,
        anthropic_api_key=os.getenv("ANTHROPIC_API_KEY") or None,
        tavily_api_key=os.getenv("TAVILY_API_KEY") or None,
        searxng_base_url=os.getenv("SEARXNG_BASE_URL", DEFAULT_SEARXNG_URL),
        profile=os.getenv("RESEARCHER_PROFILE", DEFAULT_PROFILE),
        cache_dir=Path(cache_dir) if cache_dir else None,
        log_level=os.getenv("RESEARCHER_LOG_LEVEL", DEFAULT_LOG_LEVEL),
    )


def _mask(value: str | None) -> str:
    """"set" and "unset" are what you are debugging; the value never is."""
    return "'<set>'" if value else "'<unset>'"


ENV_FILE = find_env_file()
"""Resolved once, at import: where the file is does not change during a run.

Loaded once too, and here rather than in each entry point — the CLI, the eval
harness and the served graph all reach the environment through this module, so
this is the only place that can guarantee all three see the same one.
"""

load_env()
