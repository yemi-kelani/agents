"""Environment and secrets — one place that knows a variable's name.

Every key the pipeline needs is read here, through `os.getenv(NAME, default)`,
and nowhere else. The point is not tidiness: a key read inline at three call
sites is a key that gets a different default at two of them, and a `.env` that
half the entry points forget to load.

`settings()` reads the environment on every call rather than caching it. A
process that changed its environment — a test, a notebook, a server reloading —
should see the change, and a cached snapshot is the kind of thing that is only
noticed once somebody has spent an hour on it.
"""
from __future__ import annotations

import pytest

from cache import default_dir
from models import ModelConfig, build_models
from settings import (
    DEFAULT_LOG_LEVEL, ENV_FILE, PROVIDER_KEY_ENV, load_env, settings,
)


class FakeChatModel:
    def __init__(self, spec: str, **kwargs):
        self.spec, self.kwargs = spec, kwargs


def recording_factory():
    def factory(spec: str, **kwargs) -> FakeChatModel:
        return FakeChatModel(spec, **kwargs)
    return factory


# --- reading the environment ------------------------------------------------

def test_a_variable_that_is_set_is_read(monkeypatch):
    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")

    assert settings().tavily_api_key == "tvly-xxx"


def test_a_missing_secret_is_none_rather_than_an_empty_string(monkeypatch):
    """An empty string is a value, and a provider handed one reports a confusing
    auth error instead of the obvious "you have no key"."""
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)

    assert settings().tavily_api_key is None


def test_a_missing_setting_with_a_default_gets_the_default(monkeypatch):
    monkeypatch.delenv("SEARXNG_BASE_URL", raising=False)

    assert settings().searxng_base_url.startswith("http")


def test_the_environment_is_read_fresh_on_every_call(monkeypatch):
    """Cached settings and a `.env` reloaded at import are the two ways this
    goes wrong quietly."""
    monkeypatch.setenv("RESEARCHER_PROFILE", "local")
    assert settings().profile == "local"

    monkeypatch.setenv("RESEARCHER_PROFILE", "default")
    assert settings().profile == "default"


# --- the .env file ----------------------------------------------------------

def test_the_env_file_populates_the_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("TAVILY_API_KEY", raising=False)
    env = tmp_path / ".env"
    env.write_text("TAVILY_API_KEY=from-the-file\n", encoding="utf-8")

    load_env(env)

    assert settings().tavily_api_key == "from-the-file"


def test_a_variable_already_exported_beats_the_file(tmp_path, monkeypatch):
    """Standard dotenv semantics, and load-bearing in deployment: a server sets
    real variables, and a stale `.env` in the image must not override them."""
    monkeypatch.setenv("TAVILY_API_KEY", "from-the-shell")
    env = tmp_path / ".env"
    env.write_text("TAVILY_API_KEY=from-the-file\n", encoding="utf-8")

    load_env(env)

    assert settings().tavily_api_key == "from-the-shell"


def test_the_env_file_is_found_by_walking_up_from_the_package():
    """It sits at the repo root, above the project — so the search goes up
    rather than assuming a fixed depth."""
    assert ENV_FILE is not None
    assert ENV_FILE.name == ".env"
    assert ENV_FILE.is_file()


def test_a_missing_env_file_is_not_an_error(tmp_path):
    """Deployment sets real variables and ships no file. Raising here would make
    the working configuration the one that crashes."""
    assert load_env(tmp_path / "nope.env") is False


# --- secrets do not leak into logs -----------------------------------------

def test_a_secret_is_masked_in_the_repr(monkeypatch):
    """Settings will end up in a traceback or a debug log eventually. The
    default dataclass repr would print the key in it."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-supersecret")

    printed = repr(settings())

    assert "supersecret" not in printed
    assert "openai_api_key" in printed


def test_a_non_secret_setting_is_still_readable_in_the_repr(monkeypatch):
    monkeypatch.setenv("RESEARCHER_PROFILE", "local")

    assert "local" in repr(settings())


def test_masking_says_whether_a_key_is_there_at_all(monkeypatch):
    """The useful half of the secret: "set" and "unset" are what you are
    debugging, and neither of them is the value."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-proj-supersecret")
    assert "set" in repr(settings())

    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    assert "unset" in repr(settings())


# --- provider keys ----------------------------------------------------------

def test_a_provider_key_is_found_from_the_model_spec(monkeypatch):
    """Specs are `provider:model` (10), so the provider half names the variable
    without anything having to restate the mapping."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    assert settings().provider_key("openai:gpt-4.1") == "sk-test"


def test_a_model_spec_carrying_a_colon_in_its_name_still_resolves(monkeypatch):
    """`ollama:qwen3:32b` is a provider and a model with a colon in it."""
    monkeypatch.setenv("OLLAMA_API_KEY", "should-not-matter")

    assert settings().provider_key("ollama:qwen3:32b") is None


def test_a_provider_that_needs_no_key_has_none():
    """Local models are the point of the local profile. A key looked up for one
    would be a key nobody has."""
    assert "ollama" not in PROVIDER_KEY_ENV


def test_the_known_providers_cover_the_defaults():
    assert {"openai", "anthropic"} <= set(PROVIDER_KEY_ENV)


# --- the call sites ---------------------------------------------------------

def test_a_hosted_model_is_built_with_the_key_for_its_provider(monkeypatch):
    """The key reaches the provider explicitly rather than by hoping the SDK
    finds the same variable we did."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")

    models = build_models(ModelConfig(), factory=recording_factory())

    assert models.planner.kwargs["api_key"] == "sk-test"


def test_a_local_model_is_not_handed_an_api_key(monkeypatch):
    """Ollama takes no key, and passing `api_key=None` into a provider that has
    no such field is how the free path breaks on a keyword argument."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    local = ModelConfig(planner="ollama:qwen3:32b", extractor="ollama:qwen3:32b",
                        verifier="ollama:qwen3:32b", synthesizer="ollama:qwen3:32b")

    with pytest.warns(UserWarning):
        models = build_models(local, factory=recording_factory())

    assert "api_key" not in models.planner.kwargs


def test_a_model_built_without_a_key_configured_says_nothing_about_one(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    models = build_models(ModelConfig(), factory=recording_factory())

    assert "api_key" not in models.planner.kwargs


def test_the_key_settings_resolved_is_the_key_the_backend_holds(monkeypatch):
    """Reaches into the adapter's private field on purpose: what this checks is
    that the credential travelled, and there is no public surface that would
    show it without also putting it somewhere a log could reach.

    Spec 04 already covers the missing-key failure, so that is not repeated."""
    from search import build_router

    monkeypatch.setenv("TAVILY_API_KEY", "tvly-xxx")
    router = build_router({"backends": ["tavily"]})

    assert router.backends[0]._key == "tvly-xxx"


def test_the_searxng_backend_takes_its_location_from_settings(monkeypatch):
    from search import build_router

    monkeypatch.setenv("SEARXNG_BASE_URL", "http://searx.internal:8888")
    router = build_router({"backends": ["searxng"]})

    assert router.backends[0].base_url == "http://searx.internal:8888"


def test_the_cache_directory_comes_from_settings(monkeypatch, tmp_path):
    monkeypatch.setenv("RESEARCHER_CACHE_DIR", str(tmp_path))

    assert default_dir() == tmp_path


def test_the_config_profile_comes_from_settings(monkeypatch, tmp_path):
    from config import load_config

    monkeypatch.setenv("RESEARCHER_PROFILE", "local")
    path = tmp_path / "config.yaml"
    path.write_text("models_local:\n  planner: ollama:qwen3:32b\n", encoding="utf-8")

    assert load_config(path).profile == "local"


def test_the_config_module_still_offers_the_settings(monkeypatch):
    """`config` is where a reader looks for configuration, so it re-exports the
    environment half rather than making them know it lives next door."""
    import config

    monkeypatch.setenv("RESEARCHER_PROFILE", "local")
    assert config.settings().profile == "local"


def test_the_log_level_comes_from_the_environment(monkeypatch):
    """The surfaces with no argv — `langgraph dev`, the eval harness — turn the
    tool log on with this and nothing else."""
    monkeypatch.setenv("RESEARCHER_LOG_LEVEL", "DEBUG")

    assert settings().log_level == "DEBUG"


def test_a_run_that_asked_for_nothing_logs_nothing(monkeypatch):
    monkeypatch.delenv("RESEARCHER_LOG_LEVEL", raising=False)

    assert settings().log_level == DEFAULT_LOG_LEVEL
