"""Types carried in graph state.

Their own module because both `agent` (which declares the state) and the nodes
(which produce them) need them — importing the graph from a node would be a
cycle.
"""

from __future__ import annotations

from pydantic import BaseModel


class Critique(BaseModel):
    file: str
    issue: str
    detail: str
    # Anchors an inline PR comment when the model can name a line; plenty of
    # real findings are about a file as a whole, so this stays optional.
    line: int | None = None
    severity: str = "unknown"


# Most severe first, so the reader meets the worst finding at the top. Anything
# the model did not label sorts last rather than being dropped.
_SEVERITY_ORDER = {"high": 0, "medium": 1, "low": 2, "unknown": 3}

# GitHub rejects an oversized review body with a 422, which would throw away an
# entire completed review. The platform limit is not documented on the reviews
# endpoint, so this sits well under the 65536 characters comment bodies are
# known to accept, leaving room for the marker `post_review` appends.
MAX_BODY_CHARS = 60_000

# Headroom reserved for the omission notice, so reporting truncation can never
# itself overrun the cap.
_OMISSION_HEADROOM = 200


def format_review(critiques: list[Critique], max_chars: int = MAX_BODY_CHARS) -> str:
    """Render findings as the markdown body of a GitHub review.

    Lives beside the type it formats so `github_client` stays transport-only.

    Findings past `max_chars` are dropped and counted rather than raising:
    severity-ordered output means the ones that survive are the ones that matter
    most, and a partial review that posts beats a complete one that 422s.
    """
    if not critiques:
        return "No problems found."

    ordered = sorted(
        critiques,
        key=lambda c: (_SEVERITY_ORDER.get(c.severity, 3), c.file, c.line or 0),
    )

    count = len(ordered)
    header = f"Found {count} issue{'s' if count != 1 else ''}."
    lines = [header, ""]
    used = len(header) + 1
    shown = 0

    for critique in ordered:
        location = f"`{critique.file}`"
        if critique.line is not None:
            location += f" line {critique.line}"
        block = [f"### {critique.severity.upper()} — {location}", "", f"**{critique.issue}**"]
        if critique.detail:
            block += ["", critique.detail]
        block.append("")

        # Leave headroom for the omission notice, so reporting the truncation can
        # never itself overrun the cap.
        size = sum(len(line) + 1 for line in block)
        if shown and used + size > max_chars - _OMISSION_HEADROOM:
            break
        lines += block
        used += size
        shown += 1

    if shown < count:
        lines.append(f"*{count - shown} further finding(s) omitted: the review was too long to post.*")

    body = "\n".join(lines).strip()
    if len(body) > max_chars:
        # The first finding is always kept so the review is never empty, which
        # means one enormous DETAIL can still overrun on its own. Clamp so the
        # cap is a guarantee rather than a best effort.
        notice = "\n\n*This review was truncated because it was too long to post.*"
        body = body[: max_chars - len(notice)] + notice
    return body
