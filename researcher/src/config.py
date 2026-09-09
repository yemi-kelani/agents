"""One config file, two profiles — spec 10.

The shape is `models:` plus `models_local:`: a section may be shadowed by a
profile-suffixed twin, and anything without one is shared. That is what makes
"switching to the local profile is a config change only" literally true —
`search:` has no local variant because a local run still searches the web.

Secrets are not in this *file*. Every key comes from the environment, loaded
from `.env` and read through `settings` — which this module re-exports, so
configuration is one import whether what you need is a YAML section or a key.
`settings` lives next door rather than here because this module imports `cache`,
`models` and `search`, and all three need the environment: putting it here would
close the cycle.
"""
from __future__ import annotations

from pathlib import Path

import yaml
from pydantic import BaseModel, Field

from cache import DEFAULT_TTL_HOURS
from models import ModelConfig
from search import SearchConfig
from settings import DEFAULT_PROFILE, ENV_FILE, Settings, load_env, settings

__all__ = [
    "CacheConfig", "Config", "DEFAULT_CONFIG_PATH", "DEFAULT_PROFILE", "ENV_FILE",
    "Settings", "load_config", "load_env", "settings",
]

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config.yaml"


class CacheConfig(BaseModel):
    enabled: bool = True
    ttl_hours: float = DEFAULT_TTL_HOURS
    dir: str | None = None            # None -> RESEARCHER_CACHE_DIR, then ~/.cache


class Config(BaseModel):
    profile: str = DEFAULT_PROFILE
    """Recorded rather than discarded: the eval (11) reports both profiles side
    by side, and a row that cannot say which one produced it is unreadable."""

    models: ModelConfig = Field(default_factory=ModelConfig)
    search: SearchConfig = Field(default_factory=SearchConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)


def load_config(path: str | Path | None = None, *, profile: str | None = None) -> Config:
    """Load `path` (default `config.yaml` beside the package) for one profile.

    The profile comes from the argument, then `RESEARCHER_PROFILE`, then the
    shared sections — so taking a run local needs neither a code edit nor a
    second config file.
    """
    profile = profile or settings().profile
    raw = yaml.safe_load(Path(path or DEFAULT_CONFIG_PATH).read_text(encoding="utf-8")) or {}

    return Config(
        profile=profile,
        models=ModelConfig(**_section(raw, "models", profile),
                           context_limits=_section(raw, "context_limits", profile)),
        search=SearchConfig(**_section(raw, "search", profile)),
        cache=CacheConfig(**_section(raw, "cache", profile)),
    )


def _section(raw: dict, key: str, profile: str) -> dict:
    """`key_<profile>` if the file has one, else `key`, else nothing."""
    if profile != DEFAULT_PROFILE:
        scoped = raw.get(f"{key}_{profile}")
        if scoped is not None:
            return scoped
    return raw.get(key) or {}
