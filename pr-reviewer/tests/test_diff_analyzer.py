"""Chunking a diff and surviving partial summarization failures."""

import asyncio

import pytest
from langchain_core.messages import AIMessage

from nodes.diff_analyzer import (
    MAX_SEGMENTS,
    DiffError,
    chunk_diff,
    diff_stats,
    format_stats,
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


class TestDiffStats:
    """The per-file breakdown that explains, in the log, what was reviewed.

    A run that reports "no problems" is indistinguishable from a run that was
    handed the wrong diff unless the log says which files it actually saw.
    """

    DIFF = (
        "diff --git a/a.py b/a.py\n"
        "--- a/a.py\n+++ b/a.py\n"
        "@@ -1,2 +1,3 @@\n context\n+added one\n+added two\n-removed one\n"
        "diff --git a/b.py b/b.py\n"
        "--- a/b.py\n+++ b/b.py\n"
        "@@ -1 +1 @@\n-gone\n"
    )

    def test_counts_additions_and_removals_per_file(self):
        assert diff_stats(self.DIFF) == {"a.py": (2, 1), "b.py": (0, 1)}

    def test_the_file_header_lines_are_not_counted_as_changes(self):
        """`+++`/`---` start with + and -, and would inflate every file by one."""
        assert diff_stats(self.DIFF)["b.py"] == (0, 1)

    def test_an_empty_diff_has_no_files(self):
        assert diff_stats("") == {}

    def test_a_rename_is_reported_under_its_new_path(self):
        renamed = (
            "diff --git a/old.py b/new.py\n"
            "similarity index 90%\nrename from old.py\nrename to new.py\n"
        )
        assert list(diff_stats(renamed)) == ["new.py"]

    def test_the_summary_is_capped_for_a_large_diff(self):
        files = {f"f{i}.py": (1, 1) for i in range(30)}
        rendered = format_stats(files, limit=5)
        assert rendered.count("+1/-1") == 5
        assert "and 25 more" in rendered

    def test_the_summary_names_paths_and_counts(self):
        assert format_stats({"a.py": (2, 1)}) == "a.py +2/-1"
