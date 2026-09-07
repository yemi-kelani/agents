"""Parsing the CLI's output and the model's review format."""

import json

from parsers import parse_codex, parse_critiques, parse_tool_step


def _event(**kwargs) -> str:
    return json.dumps(kwargs)


class TestParseCodex:
    """The event stream shape here is taken from a real `codex exec --json` run."""

    def test_takes_the_agent_message_from_a_completed_item(self):
        stdout = "\n".join([
            _event(type="thread.started", thread_id="abc"),
            _event(type="turn.started"),
            _event(type="item.completed",
                   item={"id": "item_0", "type": "agent_message",
                         "text": "FILE: a.py\nISSUE: bug"}),
            _event(type="turn.completed"),
        ])
        assert parse_codex(stdout) == "FILE: a.py\nISSUE: bug"

    def test_never_reports_a_cli_error_as_the_answer(self):
        """The regression that made this rewrite necessary.

        `error` events carry a top-level `message`. A parser scanning top-level
        keys returns the 401 text as though the model had written it — reporting
        a CLI failure as a code review.
        """
        stdout = "\n".join([
            _event(type="item.completed",
                   item={"type": "agent_message", "text": "the real answer"}),
            _event(type="error", message="401 Unauthorized: Missing bearer"),
            _event(type="turn.failed",
                   error={"message": "401 Unauthorized: Missing bearer"}),
        ])

        result = parse_codex(stdout)
        assert result == "the real answer"
        assert "401" not in result

    def test_ignores_error_items(self):
        stdout = "\n".join([
            _event(type="item.completed",
                   item={"type": "agent_message", "text": "the answer"}),
            _event(type="item.completed",
                   item={"id": "item_1", "type": "error",
                         "message": "Falling back from WebSockets"}),
        ])
        assert parse_codex(stdout) == "the answer"

    def test_plain_text_keeps_every_line(self):
        """Without --json the CLI prints prose; none of it may be dropped.

        The original parser assigned rather than accumulated, collapsing a whole
        review to its final line.
        """
        stdout = "FILE: a.py\nISSUE: first\nFILE: b.py\nISSUE: second"
        result = parse_codex(stdout)

        assert "a.py" in result
        assert "b.py" in result
        assert result.count("ISSUE:") == 2

    def test_last_agent_message_wins(self):
        stdout = "\n".join([
            _event(type="item.completed", item={"type": "agent_message", "text": "first"}),
            _event(type="item.completed", item={"type": "agent_message", "text": "second"}),
        ])
        assert parse_codex(stdout) == "second"

    def test_empty_output(self):
        assert parse_codex("") == ""


class TestParseCritiques:
    def test_reads_a_well_formed_block(self):
        [finding] = parse_critiques(
            "FILE: a.py\nLINE: 12\nSEVERITY: high\nISSUE: leak\nDETAIL: it leaks")

        assert finding == {
            "file": "a.py", "line": 12, "severity": "high",
            "issue": "leak", "detail": "it leaks",
        }

    def test_a_non_numeric_line_becomes_none(self):
        """The prompt allows "unknown", and models also write ranges."""
        [finding] = parse_critiques(
            "FILE: a.py\nLINE: 12-15\nSEVERITY: high\nISSUE: x\nDETAIL: y")
        assert finding["line"] is None

    def test_an_unrecognized_severity_becomes_unknown(self):
        [finding] = parse_critiques(
            "FILE: a.py\nLINE: 1\nSEVERITY: critical\nISSUE: x\nDETAIL: y")
        assert finding["severity"] == "unknown"

    def test_a_clean_review_yields_nothing(self):
        assert parse_critiques("No problems found.") == []

    def test_a_block_without_an_issue_is_dropped(self):
        assert parse_critiques("FILE: a.py\nDETAIL: no issue named") == []

    def test_fields_may_be_reordered(self):
        [finding] = parse_critiques(
            "FILE: a.py\nISSUE: x\nSEVERITY: low\nDETAIL: y\nLINE: 3")
        assert finding["line"] == 3
        assert finding["severity"] == "low"

    def test_several_findings(self):
        findings = parse_critiques(
            "FILE: a.py\nISSUE: one\nDETAIL: d\n"
            "FILE: b.py\nISSUE: two\nDETAIL: d")
        assert [f["file"] for f in findings] == ["a.py", "b.py"]


class TestParseToolStep:
    def test_finds_the_object_inside_prose_and_fences(self):
        raw = 'Here you go:\n```json\n{"tool": "explore_codebase", "args": {"question": "q"}}\n```'
        assert parse_tool_step(raw)["tool"] == "explore_codebase"

    def test_finds_an_answer(self):
        assert parse_tool_step('{"answer": "done"}')["answer"] == "done"

    def test_raises_without_a_protocol_key(self):
        import pytest
        with pytest.raises(ValueError):
            parse_tool_step('{"unrelated": 1}')
