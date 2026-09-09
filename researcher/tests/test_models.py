"""Spec 10 acceptance: roles, profiles, caps and caching.

Four criteria: switching to the local profile is a config change only, the
verifier and the extractor default to different models, token spend is capped
and enforced, and search and fetch results survive across runs.

The first is the one that carries the spec's argument. It is asserted by loading
two profiles from the *same* config file through the *same* assembly function
and checking which model specs were asked for — because "local models are a
first-class path" means exactly that no code moves when you take it.

Models are built through an injected factory. `init_chat_model` needs a provider
package installed and a key to do anything, and a test that reached a provider
would be testing the provider.
"""
from __future__ import annotations

import asyncio
import json
import time
import warnings
from pathlib import Path

import pytest
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from cache import Cache
from config import load_config
from graph import build_graph_from_config
from models import (
    DEFAULT_MODELS, ROLES, BudgetExceeded, Budgeted, ModelConfig, Models,
    TokenMeter, build_models,
)
from search.base import SearchHit
from state import Budget
from verifiers import LLMVerifier

CONFIG = """
models:
  planner:     openai:gpt-4.1
  extractor:   openai:gpt-4.1-mini
  verifier:    openai:gpt-4.1
  synthesizer: openai:gpt-4.1

models_local:
  planner:     ollama:qwen3:32b
  extractor:   ollama:qwen3:32b
  verifier:    ollama:qwen3:32b
  synthesizer: ollama:qwen3:32b

context_limits:
  extractor:   128000
  synthesizer: 128000

context_limits_local:
  extractor:   32000
  synthesizer: 32000

search:
  backends: [searxng]
  k: 5

cache:
  ttl_hours: 24
"""


# --- fakes ------------------------------------------------------------------

class FakeChatModel:
    """Stands in for a provider model. Records how it was built, which is how
    "the local profile asked for ollama" becomes assertable without a provider."""

    def __init__(self, spec: str, **kwargs):
        self.spec = spec
        self.kwargs = kwargs
        self.prompts: list[str] = []

    @property
    def model_name(self) -> str:
        return self.spec

    def with_structured_output(self, schema, **kwargs):
        return _Bound(self, schema)

    def invoke(self, prompt, **kwargs):
        self.prompts.append(prompt)
        return AIMessage(content="a report [1]")

    async def ainvoke(self, prompt, **kwargs):
        return self.invoke(prompt)


class _Bound:
    def __init__(self, model: FakeChatModel, schema):
        self.model, self.schema = model, schema

    def invoke(self, prompt, **kwargs):
        self.model.prompts.append(prompt)
        return self.schema.model_construct()

    async def ainvoke(self, prompt, **kwargs):
        return self.invoke(prompt)


def recording_factory(built: list[str] | None = None):
    """A stand-in for `init_chat_model` that records every spec it was asked for."""
    built = [] if built is None else built

    def factory(spec: str, **kwargs) -> FakeChatModel:
        built.append(spec)
        return FakeChatModel(spec, **kwargs)

    factory.built = built
    return factory


def a_config_file(tmp_path: Path, text: str = CONFIG) -> Path:
    path = tmp_path / "config.yaml"
    path.write_text(text, encoding="utf-8")
    return path


def usage(total: int, model: str = "fake") -> LLMResult:
    """A real provider response carrying real usage metadata."""
    message = AIMessage(content="hi", usage_metadata={
        "input_tokens": total, "output_tokens": 0, "total_tokens": total})
    message.response_metadata = {"model_name": model}
    return LLMResult(generations=[[ChatGeneration(message=message)]])


# --- roles ------------------------------------------------------------------

def test_every_role_the_pipeline_asks_for_has_a_default():
    """Four jobs with different requirements, not "a model". A missing default
    is a role nobody can run without editing config."""
    assert set(ROLES) == {"planner", "extractor", "verifier", "synthesizer"}
    assert set(DEFAULT_MODELS) == set(ROLES)


def test_the_verifier_and_the_extractor_default_to_different_models():
    """Acceptance. LLM-as-judge shows self-preference bias: a model asked
    whether its own claim is supported inflates the pass rate, which is the one
    number this repo exists to report honestly."""
    cfg = ModelConfig()

    assert cfg.verifier != cfg.extractor


def test_a_profile_that_judges_with_the_extraction_model_says_so_out_loud():
    """The local profile shares one model across every role deliberately, and
    that is a real cost to the eval's pass rate rather than a free convenience.
    Warned rather than refused: refusing would ban the local profile the spec
    calls first-class."""
    shared = ModelConfig(extractor="ollama:qwen3:32b", verifier="ollama:qwen3:32b")

    with pytest.warns(UserWarning, match="self-preference"):
        build_models(shared, factory=recording_factory())


def test_distinct_models_pass_without_a_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        build_models(ModelConfig(), factory=recording_factory())


# --- building the models ----------------------------------------------------

def test_each_role_is_built_from_the_spec_the_config_names():
    factory = recording_factory()

    models = build_models(ModelConfig(), factory=factory)

    assert models.planner.spec == DEFAULT_MODELS["planner"]
    assert models.extractor.spec == DEFAULT_MODELS["extractor"]
    assert models.verifier.spec == DEFAULT_MODELS["verifier"]
    assert models.synthesizer.spec == DEFAULT_MODELS["synthesizer"]


def test_models_are_built_deterministic():
    """Temperature zero everywhere. The eval reports run-to-run consistency
    (Pass^k); sampling noise the pipeline chose for itself would show up there
    as inconsistency the pipeline cannot explain."""
    models = build_models(ModelConfig(), factory=recording_factory())

    assert models.extractor.kwargs["temperature"] == 0


def test_a_role_keeps_the_name_of_the_model_behind_it():
    """`verified_by` (07) is what lets the eval attribute a verdict to a judge.
    A wrapper that swallowed the model name would make every verdict read as
    having come from the wrapper."""
    models = build_models(ModelConfig(), factory=recording_factory())

    assert LLMVerifier(models.verifier).name == f"llm:{DEFAULT_MODELS['verifier']}"


def test_one_model_can_serve_every_role():
    """The local profile's shape, and what a test wants: `uniform` is the
    single-model case stated once rather than repeated at four call sites."""
    model = FakeChatModel("ollama:qwen3:32b")

    models = Models.uniform(model)

    assert {models.planner, models.extractor, models.synthesizer} == {model}


# --- acceptance: the local switch is a config change only -------------------

def test_the_local_profile_comes_from_the_same_file_and_the_same_code(tmp_path):
    """Acceptance, and the whole point of the component. Same config file, same
    assembly function, one argument different — and every call goes to a model
    running on the developer's machine."""
    path = a_config_file(tmp_path)
    hosted, local = recording_factory(), recording_factory()

    build_graph_from_config(load_config(path), factory=hosted)
    with pytest.warns(UserWarning, match="self-preference"):
        build_graph_from_config(load_config(path, profile="local"), factory=local)

    assert all(spec.startswith("openai:") for spec in hosted.built)
    assert all(spec.startswith("ollama:") for spec in local.built)
    assert len(local.built) == len(ROLES)


def test_the_profile_can_be_chosen_from_the_environment(tmp_path, monkeypatch):
    """A profile switch that needs a code edit is not a config change."""
    monkeypatch.setenv("RESEARCHER_PROFILE", "local")

    assert load_config(a_config_file(tmp_path)).models.planner == "ollama:qwen3:32b"


def test_an_unnamed_profile_falls_back_to_the_shared_section(tmp_path):
    """`search:` has no `search_local:`, and a local run still has to search."""
    cfg = load_config(a_config_file(tmp_path), profile="local")

    assert cfg.search.backends == ["searxng"]


def test_an_absent_section_leaves_the_defaults_alone(tmp_path):
    path = a_config_file(tmp_path, "models:\n  planner: openai:gpt-4.1\n")

    cfg = load_config(path)

    assert cfg.models.extractor == DEFAULT_MODELS["extractor"]
    assert cfg.models.context_limits == {}


def test_the_profile_travels_with_the_config_it_produced(tmp_path):
    """The eval (11) reports both profiles side by side; a row that cannot say
    which profile produced it is a row nobody can read."""
    assert load_config(a_config_file(tmp_path), profile="local").profile == "local"


# --- context limits are not the token budget --------------------------------

def test_a_context_limit_is_how_much_fits_in_one_call_not_what_a_run_may_spend():
    """`Budget.max_total_tokens` (01) caps what the whole run spends; a context
    limit caps what one call can hold. They move independently — a 500k run
    budget is fine on a 32k local model, because no single call in this pipeline
    comes near 32k."""
    cfg = ModelConfig(context_limits={"extractor": 32000})
    generous = build_models(cfg, budget=Budget(max_total_tokens=500_000),
                            factory=recording_factory())
    frugal = build_models(cfg, budget=Budget(max_total_tokens=1_000),
                          factory=recording_factory())

    assert generous.chars_for("extractor") == frugal.chars_for("extractor")
    assert generous.chars_for("synthesizer") is None


def test_a_bigger_window_buys_a_bigger_slice_of_the_document():
    small = Models.uniform(FakeChatModel("m"), context_limits={"extractor": 32000})
    large = Models.uniform(FakeChatModel("m"), context_limits={"extractor": 128000})

    assert large.chars_for("extractor") == 4 * small.chars_for("extractor")


def test_a_role_with_no_configured_limit_gets_no_limit():
    """Optional by design: omit a role and the provider default applies. A
    fabricated limit would truncate a frontier model's prompt for no reason."""
    assert Models.uniform(FakeChatModel("m")).chars_for("extractor") is None


def test_only_part_of_the_window_is_offered_to_the_document():
    """The prompt, the schema and the answer share the window with the text.
    Handing the whole limit to the document is how a call overflows on the
    model that had exactly enough room."""
    models = Models.uniform(FakeChatModel("m"), context_limits={"extractor": 1000})

    assert 0 < models.chars_for("extractor") < 1000 * 4


# --- acceptance: token spend is capped and enforced -------------------------

def test_the_meter_counts_what_the_provider_reported():
    meter = TokenMeter(max_total_tokens=100)

    meter.on_llm_end(usage(30))
    meter.on_llm_end(usage(12))

    assert meter.spent == 42


def test_a_call_past_the_cap_is_refused_before_it_is_made():
    """Refused before, not after: a wrapper that noticed afterwards has already
    bought the call it was there to prevent."""
    meter = TokenMeter(max_total_tokens=10)
    model = FakeChatModel("m")
    budgeted = Budgeted(model, meter)
    meter.on_llm_end(usage(11))

    with pytest.raises(BudgetExceeded):
        budgeted.invoke("anything")

    assert model.prompts == []


def test_a_call_within_the_cap_goes_through():
    meter = TokenMeter(max_total_tokens=100)

    assert Budgeted(FakeChatModel("m"), meter).invoke("hello").text == "a report [1]"


def test_the_cap_covers_structured_calls_too():
    """Most of this pipeline talks to models through `with_structured_output`.
    A cap that only saw plain calls would be a cap on the report alone."""
    meter = TokenMeter(max_total_tokens=10)
    model = FakeChatModel("m")
    meter.on_llm_end(usage(11))

    with pytest.raises(BudgetExceeded):
        asyncio.run(Budgeted(model, meter).with_structured_output(Budget).ainvoke("x"))

    assert model.prompts == []


def test_the_cap_comes_from_the_run_budget():
    """One number, in `Budget` (01), where the rest of the spend limits live."""
    models = build_models(ModelConfig(), budget=Budget(max_total_tokens=7),
                          factory=recording_factory())

    assert models.meter.max_total_tokens == 7


def test_every_role_is_metered_by_the_same_meter():
    """The cap is on the run, not on a role. Four separate meters would each
    allow the full budget."""
    models = build_models(ModelConfig(), factory=recording_factory())

    assert {id(m.meter) for m in
            (models.planner, models.extractor, models.verifier, models.synthesizer)} \
        == {id(models.meter)}


def test_the_meter_is_wired_into_the_models_it_meters():
    """Counting happens through the callback the model was built with, so a
    structured call — which never returns an `AIMessage` to read usage off —
    is counted like any other."""
    models = build_models(ModelConfig(), factory=recording_factory())

    assert models.meter in models.extractor.kwargs["callbacks"]


# --- acceptance: search and fetch are cached across runs --------------------

def test_a_second_run_answers_a_repeated_call_from_disk(tmp_path):
    """Eval runs repeat the same queries dozens of times. Without this most of
    the wall-clock is network wait and most of the quota goes on identical
    requests. A *second* `Cache` object stands in for the second process."""
    calls = []

    async def fetch(url: str) -> str:
        calls.append(url)
        return "the page text"

    first = Cache(tmp_path).wrap(fetch)
    assert asyncio.run(first("https://example.com")) == "the page text"

    second = Cache(tmp_path).wrap(fetch)
    assert asyncio.run(second("https://example.com")) == "the page text"
    assert calls == ["https://example.com"], "the second run refetched"


def test_different_arguments_are_different_entries(tmp_path):
    calls = []

    async def fetch(url: str) -> str:
        calls.append(url)
        return f"text of {url}"

    cached = Cache(tmp_path).wrap(fetch)
    asyncio.run(cached("https://a.example"))

    assert asyncio.run(cached("https://b.example")) == "text of https://b.example"
    assert len(calls) == 2


def test_an_entry_past_its_ttl_is_refetched(tmp_path):
    """A cached page is a snapshot of a source, and this pipeline's whole claim
    is that a quote is still in the document it cited."""
    calls = []

    async def fetch(url: str) -> str:
        calls.append(url)
        return f"read {len(calls)}"

    cached = Cache(tmp_path, ttl_hours=1).wrap(fetch)
    asyncio.run(cached("https://example.com"))
    for entry in tmp_path.glob("*.json"):
        stale = json.loads(entry.read_text()) | {"at": time.time() - 7200}
        entry.write_text(json.dumps(stale))

    assert asyncio.run(cached("https://example.com")) == "read 2"


def test_an_empty_answer_is_not_cached(tmp_path):
    """A dead backend and a failed fetch both answer with nothing. Caching that
    for a day turns a transient outage into a day of empty runs."""
    answers = iter([[], [SearchHit(url="u", title="t", snippet="s", backend="b")]])

    async def search(query: str):
        return next(answers)

    cached = Cache(tmp_path).wrap(search, revive=lambda rows: [SearchHit(**r) for r in rows])
    assert asyncio.run(cached("mimir")) == []

    assert len(asyncio.run(cached("mimir"))) == 1


def test_a_cached_search_comes_back_as_search_hits_not_as_dictionaries(tmp_path):
    """The cache sits under a `SearchBackend`, so what it returns has to satisfy
    the same protocol the router and extraction were written against."""
    async def search(query: str):
        return [SearchHit(url="https://example.com", title="t", snippet="s", backend="b")]

    cached = Cache(tmp_path).wrap(search, revive=lambda rows: [SearchHit(**r) for r in rows])
    asyncio.run(cached("mimir"))

    (hit,) = asyncio.run(cached("mimir"))
    assert isinstance(hit, SearchHit)
    assert hit.url == "https://example.com"


def test_a_cached_backend_is_still_a_search_backend(tmp_path):
    from search.base import SearchBackend

    class Backend:
        name = "fake"

        async def search(self, query: str, k: int = 5) -> list[SearchHit]:
            return [SearchHit(url="u", title="t", snippet="s", backend="fake")]

    cached = Cache(tmp_path).backend(Backend())

    assert isinstance(cached, SearchBackend)
    assert cached.name == "fake"
    assert len(asyncio.run(cached.search("mimir", 3))) == 1


def test_two_backends_do_not_share_a_cache_entry(tmp_path):
    """The fallback chain asks each backend the same query. One cache entry for
    all of them would answer for a backend that was never called."""
    class Backend:
        def __init__(self, name):
            self.name = name

        async def search(self, query: str, k: int = 5) -> list[SearchHit]:
            return [SearchHit(url=f"https://{self.name}", title="t", snippet="s",
                              backend=self.name)]

    cache = Cache(tmp_path)
    first = asyncio.run(cache.backend(Backend("a")).search("mimir"))
    second = asyncio.run(cache.backend(Backend("b")).search("mimir"))

    assert first[0].url != second[0].url


def test_caching_is_off_when_the_config_turns_it_off(tmp_path):
    assert Cache.from_config(_cache_config(tmp_path, enabled=False)) is None
    assert Cache.from_config(_cache_config(tmp_path)) is not None


def _cache_config(tmp_path: Path, **overrides):
    from config import CacheConfig
    return CacheConfig(dir=str(tmp_path), **overrides)


# --- wiring -----------------------------------------------------------------

def test_the_graph_built_from_config_asks_for_every_role_exactly_once(tmp_path):
    """Four roles, four models, one assembly. A role built twice is a role whose
    two copies can drift; a role never built is a node wired to nothing."""
    factory = recording_factory()

    graph = build_graph_from_config(load_config(a_config_file(tmp_path)), factory=factory)

    assert sorted(factory.built) == sorted(DEFAULT_MODELS[role] for role in ROLES)
    assert {"plan", "verify", "sufficiency", "synthesize"} <= set(graph.get_graph().nodes)


def test_the_context_limit_reaches_the_node_that_truncates_the_document(tmp_path):
    """Extraction's 12,000-character cut was a magic number. It is the
    extractor's window now, and it is the number that has to grow when
    map-reduce chunking replaces the truncation."""
    from nodes.research import MAX_EXTRACT_CHARS

    cfg = load_config(a_config_file(tmp_path))
    models = build_models(cfg.models, factory=recording_factory())

    assert models.chars_for("extractor") > MAX_EXTRACT_CHARS


def test_a_run_with_no_configured_limits_keeps_the_conservative_truncation(tmp_path):
    """Omitting the limits must not mean "no truncation" — an unbounded document
    is how a 12,000-character safety net becomes a 400,000-token call."""
    from nodes.research import MAX_EXTRACT_CHARS

    models = Models.uniform(FakeChatModel("m"))

    assert (models.chars_for("extractor") or MAX_EXTRACT_CHARS) == MAX_EXTRACT_CHARS


# --- streaming, per role ----------------------------------------------------

def test_only_the_synthesizer_is_built_to_stream():
    """The synthesizer writes the report a human watches arrive. Every other
    role returns structured output, and a graph streamed with
    `stream_mode="messages"` makes langchain stream those calls too — so the
    extractor's schema JSON is emitted token by token to a surface that wants
    prose.

    Turning it off at the model is the fix at source rather than a filter
    downstream: it also skips langchain_openai's streaming path, which (unlike
    its non-streaming one) does not exclude the `parsed` field from the response
    dump and so warns on every structured-output call.
    """
    models = build_models(ModelConfig(), factory=recording_factory())

    assert models.synthesizer.kwargs.get("disable_streaming") is not True
    for role in ("planner", "extractor", "verifier"):
        assert getattr(models, role).kwargs["disable_streaming"] is True, role
