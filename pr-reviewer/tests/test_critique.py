"""Turning a written review into findings.

The regression here was observed on a live run. `critique_extraction.txt` told
the model to "output nothing at all" when a review is clean; the model obeyed,
and `ShellChatModel._to_result` raised "exited 0 but produced no parseable
output" — killing a review that had already completed successfully.

The fix is a token for "clean", not a looser check in the CLI layer: an empty
reply is how that layer reports the subprocess produced nothing, and making it
mean "no problems" would let a dead CLI post a clean review.
"""

import asyncio

from langchain_core.messages import AIMessage

from nodes.critique import NO_FINDINGS, extract_critiques
from prompts import load


class _Replies:
    """Returns one canned reply, counting how often it was asked."""

    def __init__(self, reply):
        self.reply = reply
        self.calls = 0

    async def ainvoke(self, prompt, config=None):
        self.calls += 1
        return AIMessage(content=self.reply)


class TestCleanReviews:
    def test_a_clean_review_is_reported_without_falling_silent(self):
        model = _Replies(NO_FINDINGS)
        assert asyncio.run(extract_critiques(model, "Looks fine to me.")) == []
        assert model.calls == 1

    def test_the_prompt_asks_for_the_token_the_code_checks_for(self):
        """Prompt and constant must not drift, so the prompt is formatted with
        the constant rather than repeating the literal."""
        rendered = load("critique_extraction").format(
            review="r", no_findings=NO_FINDINGS)
        assert NO_FINDINGS in rendered
        assert "output nothing at all" not in rendered


class TestReformatting:
    def test_prose_is_reformatted_into_findings(self):
        model = _Replies(
            "FILE: a.py\nLINE: 3\nSEVERITY: high\nISSUE: leaks\nDETAIL: always")
        findings = asyncio.run(extract_critiques(model, "a.py leaks, it's bad"))
        assert [f["file"] for f in findings] == ["a.py"]
        assert findings[0]["line"] == 3

    def test_an_already_formatted_review_needs_no_model_call(self):
        model = _Replies("should not be called")
        findings = asyncio.run(
            extract_critiques(model, "FILE: b.py\nISSUE: boom\nDETAIL: d"))
        assert [f["file"] for f in findings] == ["b.py"]
        assert model.calls == 0

    def test_an_empty_review_needs_no_model_call(self):
        model = _Replies("should not be called")
        assert asyncio.run(extract_critiques(model, "   ")) == []
        assert model.calls == 0

    def test_an_unusable_reformat_yields_no_findings_rather_than_raising(self):
        """A reply that is neither the token nor parseable findings is not a
        crash — but it must not be mistaken for a clean review either, which is
        why it is logged rather than silently dropped."""
        model = _Replies("I could not do that.")
        assert asyncio.run(extract_critiques(model, "something is wrong")) == []
