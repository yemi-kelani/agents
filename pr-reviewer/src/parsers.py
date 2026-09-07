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
