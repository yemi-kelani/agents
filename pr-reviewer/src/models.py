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
