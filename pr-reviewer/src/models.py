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


def format_review(critiques: list[Critique]) -> str:
    """Render findings as the markdown body of a GitHub review.

    Lives beside the type it formats so `github_client` stays transport-only.
    """
    if not critiques:
        return "No problems found."

    ordered = sorted(
        critiques,
        key=lambda c: (_SEVERITY_ORDER.get(c.severity, 3), c.file, c.line or 0),
    )

    count = len(ordered)
    lines = [f"Found {count} issue{'s' if count != 1 else ''}.", ""]
    for critique in ordered:
        location = f"`{critique.file}`"
        if critique.line is not None:
            location += f" line {critique.line}"
        lines.append(f"### {critique.severity.upper()} — {location}")
        lines.append("")
        lines.append(f"**{critique.issue}**")
        if critique.detail:
            lines.append("")
            lines.append(critique.detail)
        lines.append("")

    return "\n".join(lines).strip()
