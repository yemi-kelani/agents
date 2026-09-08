"""The subprocess environment handed to an agent CLI.

`_child_env` builds an allowlist rather than copying `os.environ`, which is what
keeps one CLI's key away from the other. The cost of that choice is that a
credential under the wrong name does not fall back to anything — the child
simply has no key. That is exactly the fault these tests exist to catch.
"""

import pytest

import llm
from llm import CLIError, build_specs, get_model
from settings import Settings


def _settings(**overrides) -> Settings:
    import dataclasses
    return dataclasses.replace(Settings(openai_api_key="sk-live"), **overrides)


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(llm, "settings", _settings)


class TestCodexCredential:
    def test_the_key_is_exported_under_the_name_the_cli_reads(self, configured):
        """CI supplies OPENAI_API_KEY; the codex CLI reads CODEX_API_KEY. This
        bridge is the whole reason `env_key` and `secret_attr` are separate.

        Probed against 0.153.4: with only CODEX_API_KEY set the CLI reports
        "Incorrect API key provided: sk-x***" (it sent the key), and with only
        OPENAI_API_KEY set it reports "Missing bearer ... in header" (it sent
        nothing). The previous test asserted the mapping inside Settings, which
        would not have caught the bridge being wired to the wrong variable.
        """
        model = get_model("codex")
        env = model._child_env(home=None)

        assert env["CODEX_API_KEY"] == "sk-live"

    def test_a_missing_key_is_a_named_failure_not_a_401(self, monkeypatch):
        """Better to say "you have no key" than to let the CLI report a
        confusing auth error against a key we never sent."""
        monkeypatch.setattr(llm, "settings", lambda: _settings(openai_api_key=None))

        with pytest.raises(CLIError, match="no API key"):
            get_model("codex")._child_env(home=None)

    def test_the_other_cli_s_key_is_never_exported(self, configured):
        """The reason the allowlist exists in the first place."""
        env = get_model("codex")._child_env(home=None)
        assert "BOBSHELL_API_KEY" not in env
        assert "ANTHROPIC_API_KEY" not in env


class TestChildEnvironment:
    def test_a_missing_path_does_not_raise_a_bare_keyerror(self, configured, monkeypatch):
        """Every other setup fault in this module is a CLIError; an absent PATH
        used to be an uncaught KeyError instead."""
        monkeypatch.delenv("PATH", raising=False)
        env = get_model("codex")._child_env(home=None)
        assert env["PATH"]

    def test_proxy_settings_reach_the_child(self, configured, monkeypatch):
        """On a runner behind a proxy, a CLI that cannot see these cannot reach
        the network — and the allowlist is what would have hidden them."""
        monkeypatch.setenv("HTTPS_PROXY", "http://proxy:3128")
        env = get_model("codex")._child_env(home=None)
        assert env["HTTPS_PROXY"] == "http://proxy:3128"

    def test_codex_home_is_only_set_when_home_is_isolated(self, configured):
        model = get_model("codex")
        assert "CODEX_HOME" not in model._child_env(home=None)
        assert model._child_env(home="/tmp/x")["CODEX_HOME"] == "/tmp/x/.codex"


class TestSpecs:
    def test_every_spec_names_an_attribute_settings_actually_has(self):
        """A `secret_attr` that does not resolve is the same class of bug as a
        wrong `env_key`, and would fail only at call time."""
        for name, spec in build_specs().items():
            assert hasattr(Settings(), spec.secret_attr), name

    def test_an_unknown_cli_is_rejected(self):
        with pytest.raises(ValueError, match="unknown CLI"):
            get_model("nope")
