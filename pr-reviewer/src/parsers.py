from __future__ import annotations

import json
import re


def parse_codex(stdout: str) -> str:
    """`codex exec --json` emits JSON lines; keep the last agent message."""
    last = ""
    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            last = line
            continue
        if not isinstance(obj, dict):
            last = line
            continue
        for key in ("last_agent_message", "message", "result", "text"):
            if isinstance(obj.get(key), str):
                last = obj[key]
    return last or stdout.strip()


_FENCE = re.compile(r"^\s*```(?:json)?|```\s*$", re.MULTILINE)


def parse_tool_step(raw: str) -> dict:
    """Pull the tool-protocol JSON object out of a model's reply.

    A coding CLI prefaces its answer with prose, wraps it in fences, or emits
    several objects. So rather than trusting the whole reply to be JSON, scan for
    the first object that decodes and carries a protocol key.

    Raises ValueError when there is none — the caller turns that into another
    turn rather than an exception.
    """
    text = _FENCE.sub("", raw).strip()
    decoder = json.JSONDecoder()

    for start in (i for i, ch in enumerate(text) if ch == "{"):
        try:
            obj, _ = decoder.raw_decode(text[start:])
        except json.JSONDecodeError:
            continue
        if isinstance(obj, dict) and ("tool" in obj or "answer" in obj):
            return obj

    raise ValueError("no JSON object with a 'tool' or 'answer' key")


_FIELDS = ("FILE", "LINE", "SEVERITY", "ISSUE", "DETAIL")

# Splits the review into one chunk per finding. A finding starts at FILE:.
_BLOCK = re.compile(r"^[ \t]*FILE:", re.MULTILINE | re.IGNORECASE)

# A field runs to the next field label or the end of its block, so DETAIL may
# wrap over several lines.
_FIELD = {
    name: re.compile(
        rf"^[ \t]*{name}:[ \t]*(.+?)(?=^[ \t]*(?:{'|'.join(_FIELDS)}):|\Z)",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    for name in _FIELDS
}

_SEVERITIES = ("high", "medium", "low")


def parse_critiques(text: str) -> list[dict]:
    """Pull FILE/LINE/SEVERITY/ISSUE/DETAIL blocks out of a review.

    Fields are read independently within each block, so a model that reorders
    them still parses. Returns dicts rather than models to keep this module free
    of the graph's types. An empty list is either a clean review or a model that
    ignored the format — the caller has to tell those apart.
    """
    findings = []

    for chunk in _split_blocks(text):
        fields = {}
        for name, pattern in _FIELD.items():
            match = pattern.search(chunk)
            fields[name] = " ".join(match.group(1).split()) if match else ""

        if not fields["FILE"] or not fields["ISSUE"]:
            # Without a file and a problem there is nothing to report on.
            continue

        line = fields["LINE"].lstrip("#")
        severity = fields["SEVERITY"].lower()
        findings.append(
            {
                "file": fields["FILE"],
                "line": int(line) if line.isdigit() else None,
                "severity": severity if severity in _SEVERITIES else "unknown",
                "issue": fields["ISSUE"],
                "detail": fields["DETAIL"],
            }
        )

    return findings


def _split_blocks(text: str) -> list[str]:
    starts = [m.start() for m in _BLOCK.finditer(text)]
    return [text[a:b] for a, b in zip(starts, starts[1:] + [len(text)])]


_THINKING = re.compile(r"<thinking>.*?</thinking>", re.DOTALL)


def parse_bob(stdout: str) -> str:
    """Bob's stream-json is stateful.

    Reasoning arrives inline as <thinking>...</thinking>; the answer comes via
    an `attempt_completion` tool call spread over many lines. Accumulate,
    prefer attempt_completion, strip thinking from any fallback text.

    UNVERIFIED: field names are inferred from a third-party SDK. Check against
    real output before trusting.
    """
    completion: list[str] = []
    text: list[str] = []

    for line in stdout.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            text.append(line)
            continue

        if obj.get("name") == "attempt_completion" or obj.get("tool") == "attempt_completion":
            payload = obj.get("input") or obj.get("params") or {}
            if isinstance(payload, dict):
                completion.append(payload.get("result") or payload.get("text") or "")
            elif isinstance(payload, str):
                completion.append(payload)
            continue

        for key in ("text", "content", "delta"):
            if isinstance(obj.get(key), str):
                text.append(obj[key])

    if completion:
        return "".join(completion).strip()
    return _THINKING.sub("", "".join(text)).strip()
