"""Chunking a diff and surviving partial summarization failures."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from nodes.diff_analyzer import (
    MAX_SEGMENTS,
    DiffError,
    chunk_diff,
    summarize_segments,
)


class _FakeModel:
    """Stands in for the CLI-backed model, failing whichever calls we name.

    Mirrors the real contract that matters here: `abatch` with
    `return_exceptions=True` returns exceptions in place rather than raising.
    """

    def __init__(self, fail_indexes=()):
        self.fail_indexes = set(fail_indexes)
        self.attempts = 0

    def with_retry(self, **kwargs):
        return self

    async def abatch(self, prompts, config=None, return_exceptions=False):
        self.attempts += 1
        results = []
        for index, _ in enumerate(prompts):
            if index in self.fail_indexes:
                error = RuntimeError(f"segment {index} failed")
                if not return_exceptions:
                    raise error
                results.append(error)
            else:
                results.append(AIMessage(content=f"summary {index}"))
        return results


class TestChunkDiff:
    def test_a_small_diff_is_one_segment(self):
        segments, truncated = chunk_diff("diff --git a/x b/x\n+one line\n")
        assert len(segments) == 1
        assert truncated is False

    def test_segments_overlap_so_boundary_hunks_survive(self):
        diff = "".join(f"+line {i}\n" for i in range(4000))
        segments, _ = chunk_diff(diff, max_tokens=500, grace=200)

        assert len(segments) > 1
        # Each later segment repeats the tail of the one before it.
        assert segments[1].startswith(segments[0][-200:])

    def test_truncates_rather_than_running_up_unbounded_cost(self):
        diff = "".join(f"+line {i}\n" for i in range(200_000))
        segments, truncated = chunk_diff(diff, max_tokens=500, grace=100)

        assert truncated is True
        assert len(segments) == MAX_SEGMENTS

    def test_an_overlap_larger_than_the_budget_is_an_error(self):
        diff = "".join(f"+line {i}\n" for i in range(2000))
        with pytest.raises(DiffError):
            chunk_diff(diff, max_tokens=1, grace=5000)


class TestSummarizeSegments:
    def test_one_failure_does_not_discard_the_others(self):
        """`abatch` defaults to return_exceptions=False, which would lose every
        summary because a single CLI call timed out."""
        model = _FakeModel(fail_indexes=[1])
        summary = asyncio.run(summarize_segments(model, ["a", "b", "c"]))

        assert "summary 0" in summary
        assert "summary 2" in summary
        assert "could not be summarized" in summary

    def test_total_failure_raises(self):
        model = _FakeModel(fail_indexes=[0, 1])
        with pytest.raises(DiffError):
            asyncio.run(summarize_segments(model, ["a", "b"]))

    def test_a_single_segment_is_returned_without_a_heading(self):
        summary = asyncio.run(summarize_segments(_FakeModel(), ["only"]))
        assert summary == "summary 0"

    def test_several_segments_are_labelled(self):
        summary = asyncio.run(summarize_segments(_FakeModel(), ["a", "b"]))
        assert "## Segment 1 of 2" in summary
        assert "## Segment 2 of 2" in summary
