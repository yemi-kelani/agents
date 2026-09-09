"""Out-of-state storage for fetched page text.

LangGraph serializes state on every checkpoint write, so fetched documents must
not live in state — a run that fetched 50 pages would carry all 50 in every
subsequent checkpoint. State carries the `source_id`; the text lives here.

In-memory for now, trivially swappable for disk or Redis.
"""
from __future__ import annotations

import hashlib


class ContentStore:
    def __init__(self) -> None:
        self._data: dict[str, str] = {}

    @staticmethod
    def source_id(url: str) -> str:
        """Hash of the URL, so the same page found by two topics is one entry."""
        return hashlib.sha256(url.encode()).hexdigest()[:16]

    def put(self, source_id: str, text: str) -> None:
        self._data[source_id] = text

    def get(self, source_id: str) -> str:
        """Raises KeyError for an unknown id — a missing document is a bug, not
        an empty string to silently extract claims from."""
        return self._data[source_id]

    def __contains__(self, source_id: str) -> bool:
        return source_id in self._data

    def __len__(self) -> int:
        return len(self._data)
