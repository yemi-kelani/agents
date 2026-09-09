"""The model layer — spec 10.

Four jobs, not "a model". They have genuinely different requirements: the
planner decomposes, the extractor copies text verbatim under a strict schema,
the verifier judges, the synthesizer writes. Naming them separately is what
makes "run the verifier on a different family than the extractor" a config line
rather than a refactor.

**Verifier != extractor.** LLM-as-judge shows documented self-preference bias:
a model asked whether its own claim is supported rates it more favourably, which
inflates exactly the pass rate this repo exists to report honestly. The defaults
differ; a profile that shares them warns rather than raises, because the local
profile shares one model across every role deliberately and banning it would ban
the free path the component is for.

**Local models are the first-class path.** `init_chat_model` dispatches on a
`provider:model` string, and Ollama, vLLM and LM Studio all speak the
OpenAI-compatible protocol — so a local run is a config change and nothing else.
LiteLLM is the heavier version of this, worth reaching for when key handling,
rate limiting and a PII-redaction hook should live in one place; it is
deliberately not built here.

**Context limits are not the token budget.** `Budget.max_total_tokens` (01) caps
what the whole run *spends*; `context_limits` caps what one call can *hold*. The
two move independently, which is why the window lives here with the model it
describes and the spend cap stays in `Budget`.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any

from langchain.chat_models import init_chat_model
from langchain_core.callbacks import UsageMetadataCallbackHandler
from pydantic import BaseModel, Field

from settings import settings
from state import Budget
from toollog import tool_call

ROLES = ("planner", "extractor", "verifier", "synthesizer")

STREAMING_ROLE = "synthesizer"
"""The one role whose tokens a human reads as they arrive.

Every other role returns structured output, and nothing renders it — but asking
a graph for `stream_mode="messages"` attaches a streaming callback handler, and
langchain streams *every* call once one is attached. So the extractor's schema
JSON is emitted token by token to a surface that wants prose, and the terminal
(12) has to filter it back out.

Turning it off per role is the fix at source. It also avoids a second cost:
langchain_openai's non-streaming path dumps the provider response with the
`parsed` field excluded, precisely because it may hold an arbitrary model; its
streaming path does not, so every structured-output call made this way emits a
`PydanticSerializationUnexpectedValue` warning about a field nobody reads.
"""

DEFAULT_MODELS = {
    "planner": "openai:gpt-4.1",          # reasoning, decomposition
    "extractor": "openai:gpt-4.1-mini",   # volume, strict JSON, verbatim copying
    "verifier": "openai:gpt-4.1",         # careful judgment — and not the extractor
    "synthesizer": "openai:gpt-4.1",      # long-form writing, instruction adherence
}

CHARS_PER_TOKEN = 4
"""Rough, and rough is enough: this converts a window into a truncation point,
and every use of it leaves half the window spare."""

PROMPT_SHARE = 0.5
"""How much of the window the variable part of a prompt may take. The
instructions, the schema and the answer share the window with it, so handing the
whole limit to the document is how a call overflows on the model that had
exactly enough room."""


class BudgetExceeded(RuntimeError):
    """The run tried to spend past `Budget.max_total_tokens`."""


class TokenMeter(UsageMetadataCallbackHandler):
    """Counts what the providers reported, and refuses to authorize more.

    A callback rather than a return-value inspection, because most of this
    pipeline talks to models through `with_structured_output` — which answers
    with a parsed schema and never hands back a message to read usage off. A cap
    that only saw plain calls would be a cap on the report alone.
    """

    def __init__(self, max_total_tokens: int):
        super().__init__()
        self.max_total_tokens = max_total_tokens

    @property
    def spent(self) -> int:
        return sum(u.get("total_tokens", 0) for u in self.usage_metadata.values())

    def check(self) -> None:
        if self.spent >= self.max_total_tokens:
            raise BudgetExceeded(
                f"{self.spent} tokens spent of {self.max_total_tokens}"
            )


class Budgeted:
    """A model that will not be called once the run is out of budget.

    Checked *before* the call: a wrapper that noticed afterwards has already
    bought the call it exists to prevent. Everything else is delegated, so the
    model still answers to `model_name` — which is what `verified_by` (07)
    records and what lets the eval attribute a verdict to a judge.
    """

    def __init__(self, model, meter: TokenMeter, *, role: str = "model",
                 schema: str = "", label: str | None = None):
        self._model = model
        self._meter = meter
        # The role, the model and the schema all travel with the wrapper rather
        # than being looked up at the call site, because `with_structured_output`
        # returns a bound runnable that no longer answers to `model_name` — and
        # a log line that cannot say *which* role took four seconds points at
        # nothing a config change could fix.
        self._role = role
        self._schema = schema
        self._label = label or _label(model)

    @property
    def meter(self) -> TokenMeter:
        return self._meter

    def with_structured_output(self, schema, **kwargs) -> Budgeted:
        return Budgeted(
            self._model.with_structured_output(schema, **kwargs), self._meter,
            role=self._role, label=self._label,
            schema=getattr(schema, "__name__", str(schema)),
        )

    def invoke(self, *args, **kwargs):
        with self._logged(args) as call:
            self._meter.check()
            before = self._meter.spent
            result = self._model.invoke(*args, **kwargs)
            self._record(call, result, before)
            return result

    async def ainvoke(self, *args, **kwargs):
        with self._logged(args) as call:
            self._meter.check()
            before = self._meter.spent
            result = await self._model.ainvoke(*args, **kwargs)
            self._record(call, result, before)
            return result

    def _logged(self, args):
        """The tool-log entry for one model call.

        The prompt rides in as an argument, so it lands at DEBUG: `-v` is for
        "what did the tools answer", and an extraction prompt is a whole page of
        source text. A refused call raises inside this block, which is what puts
        `BudgetExceeded` in the log — a run that stops dead is the last thing
        that should stop silently.
        """
        return tool_call(
            self._role,
            model=self._label,
            **({"schema": self._schema} if self._schema else {}),
            prompt=args[0] if args else "",
        )

    def _record(self, call, result, before: int) -> None:
        spent = self._meter.spent - before
        call.set(f"{spent} tokens", payload=getattr(result, "text", None) or result)

    def __getattr__(self, name: str):
        # Private names are this wrapper's own; delegating them would recurse
        # through `_model` before `__init__` had set it.
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._model, name)


class ModelConfig(BaseModel):
    """Which model does which job, and how much fits in one call."""
    planner: str = DEFAULT_MODELS["planner"]
    extractor: str = DEFAULT_MODELS["extractor"]
    verifier: str = DEFAULT_MODELS["verifier"]
    synthesizer: str = DEFAULT_MODELS["synthesizer"]

    context_limits: dict[str, int] = Field(default_factory=dict)
    """Tokens per call, per role. Optional — omit a role and the provider
    default applies; a fabricated limit would truncate a frontier model's prompt
    for no reason."""


@dataclass(frozen=True)
class Models:
    """The built models, addressed by role.

    A container rather than the spec's bare dict because the context limits
    travel with the models they describe. Threading a parallel limits dict
    through the same call sites is how the two drift apart.
    """
    planner: Any
    extractor: Any
    verifier: Any
    synthesizer: Any
    context_limits: dict[str, int] = field(default_factory=dict)
    meter: TokenMeter | None = None

    @classmethod
    def uniform(cls, model, *, context_limits: dict[str, int] | None = None,
                meter: TokenMeter | None = None) -> Models:
        """One model in every role — the local profile's shape, and a test's."""
        return cls(model, model, model, model, dict(context_limits or {}), meter)

    def chars_for(self, role: str) -> int | None:
        """How many characters of variable prompt this role can hold, or None
        when nothing was configured for it."""
        limit = self.context_limits.get(role)
        return int(limit * CHARS_PER_TOKEN * PROMPT_SHARE) if limit else None


def build_models(cfg: ModelConfig, *, budget: Budget | None = None,
                 factory=init_chat_model) -> Models:
    """Build one model per role, metered by one meter.

    `factory` is injected so a test — and the eval harness (11) — can build the
    pipeline without a provider package or a key. One meter for all four roles
    because the cap is on the run: four meters would each allow the full budget.
    """
    if cfg.verifier == cfg.extractor:
        warnings.warn(
            f"verifier and extractor are both {cfg.verifier!r}: LLM-as-judge "
            "self-preference bias inflates the pass rate this pipeline reports",
            stacklevel=2,
        )

    meter = TokenMeter((budget or Budget()).max_total_tokens)
    built = {}
    for role in ROLES:
        spec = getattr(cfg, role)
        built[role] = Budgeted(
            factory(spec, temperature=0, callbacks=[meter],
                    disable_streaming=role != STREAMING_ROLE,
                    **_credentials(spec)),
            meter,
            role=role,
            label=spec,
        )

    return Models(**built, context_limits=dict(cfg.context_limits), meter=meter)


def _credentials(spec: str) -> dict:
    """The API key for this spec's provider, if that provider takes one.

    Passed explicitly rather than left to the SDK to find, so the key this run
    uses is the key `settings` resolved — one place to look when a provider
    answers 401. Omitted entirely for providers with no key: `api_key=None`
    handed to Ollama is a keyword argument error, and the free path breaking on
    a credential it never needed is a poor joke.
    """
    key = settings().provider_key(spec)
    return {"api_key": key} if key else {}


def _label(model) -> str:
    """Whatever the provider calls this model, for the log line only.

    Falls back to the class name: a call that cannot name its model is still
    worth timing, and `verifiers._model_name` makes the same trade for the same
    reason — an unattributable record beats no record.
    """
    return str(
        getattr(model, "model_name", None)
        or getattr(model, "model", None)
        or type(model).__name__
    )
