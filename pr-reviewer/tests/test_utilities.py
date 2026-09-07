"""Secret redaction, trimming, and the answer-file helpers."""

from pathlib import Path

from models import Critique, format_review
from utilities import maybe_answer_file, read_text_if_any, scrub, trim_text


class TestScrub:
    def test_redacts_provider_keys(self):
        assert "sk-abcd1234efgh" not in scrub("auth failed for sk-abcd1234efgh")

    def test_redacts_github_tokens(self):
        """GITHUB_TOKEN is the one credential this tool handles directly, and it
        reaches logs through tracebacks."""
        token = "ghp_" + "a" * 36
        assert token not in scrub(f"401 using {token}")

    def test_redacts_fine_grained_tokens(self):
        token = "github_pat_" + "b" * 30
        assert token not in scrub(f"denied: {token}")

    def test_leaves_ordinary_text_alone(self):
        message = "Could not diff main...feature: unknown revision"
        assert scrub(message) == message


class TestTrimText:
    def test_short_text_passes_through_whole(self):
        head, rest = trim_text("short", max_tokens=1000)
        assert head == "short"
        assert rest == ""

    def test_long_text_splits_without_losing_anything(self):
        text = "x" * 100_000
        head, rest = trim_text(text, max_tokens=100)

        assert head + rest == text
        assert head

    def test_empty_text(self):
        assert trim_text("", max_tokens=10) == ("", "")


class TestAnswerFile:
    def test_yields_nothing_when_not_wanted(self):
        with maybe_answer_file(False) as path:
            assert path is None

    def test_the_path_is_usable_and_cleaned_up(self):
        with maybe_answer_file(True) as path:
            assert path is not None
            path.write_text("the answer", encoding="utf-8")
            assert read_text_if_any(path) == "the answer"
            directory = path.parent
        assert not directory.exists()

    def test_a_file_the_cli_never_wrote_reads_as_empty(self):
        """A CLI that fails partway leaves no file; the caller reports the
        richer failure it already has from the exit code."""
        with maybe_answer_file(True) as path:
            assert read_text_if_any(path) == ""

    def test_none_reads_as_empty(self):
        assert read_text_if_any(None) == ""

    def test_a_missing_path_reads_as_empty(self):
        assert read_text_if_any(Path("/nonexistent/nope.txt")) == ""


class TestFormatReview:
    def test_orders_by_severity(self):
        body = format_review([
            Critique(file="c.py", issue="low one", detail="d", severity="low"),
            Critique(file="a.py", issue="high one", detail="d", severity="high"),
            Critique(file="b.py", issue="medium one", detail="d", severity="medium"),
        ])
        assert body.index("high one") < body.index("medium one") < body.index("low one")

    def test_includes_the_line_when_known(self):
        body = format_review(
            [Critique(file="a.py", issue="x", detail="d", line=42, severity="high")])
        assert "line 42" in body

    def test_omits_the_line_when_unknown(self):
        body = format_review(
            [Critique(file="a.py", issue="x", detail="d", severity="high")])
        assert "line" not in body.lower().replace("a.py", "")

    def test_unknown_severity_sorts_last_rather_than_being_dropped(self):
        body = format_review([
            Critique(file="a.py", issue="mystery", detail="d", severity="unknown"),
            Critique(file="b.py", issue="known", detail="d", severity="high"),
        ])
        assert "mystery" in body
        assert body.index("known") < body.index("mystery")

    def test_a_clean_review_says_so(self):
        assert format_review([]) == "No problems found."

    def test_counts_are_pluralized(self):
        one = format_review(
            [Critique(file="a.py", issue="x", detail="d", severity="high")])
        assert "1 issue." in one
