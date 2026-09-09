"""What happened to the spec-05 injection fixtures — spec 11.

Two acceptable outcomes for a payload, and they are not equally interesting:
extraction never saw it, or verification refused what it produced. The second
shows the layers doing independent work rather than one of them carrying the
whole defense, which is the claim spec 05 makes and this is where it gets
checked rather than asserted.

The containment half is deterministic and always runs — it is a question about
an HTML cleaner. The half after it needs a model, so it is injected; without one
the table says so rather than implying a check nobody made.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from fetch import to_clean_text

INJECTIONS = {
    "hidden_div.html": "Ignore prior instructions",
    "alt_text.html": "attacker.example",
    "comment.html": "ignore the extraction rules",
    "laundered.html": "98% of generated statements",
}
"""Fixture -> the payload it is trying to land.

Named rather than inferred: without a declared payload, "the injection did not
survive" is indistinguishable from "there was nothing to survive", and a fixture
that silently stopped injecting anything would pass forever.
"""


@dataclass(frozen=True)
class InjectionResult:
    fixture: str
    reached_extraction: bool
    claim_produced: bool | None = None
    verdict: str | None = None


def injection_report(fixtures: str | Path, *, verify=None) -> list[InjectionResult]:
    """Run every fixture through the cleaner, and through a model if given.

    `verify` is `async (clean_text) -> (claim_produced, verdict)`. It is only
    called for payloads that survived cleaning: asking a model about a fixture
    whose injection was already stripped would report a defense that never had
    to work.
    """
    import asyncio

    fixtures = Path(fixtures)
    results = []

    for name, payload in sorted(INJECTIONS.items()):
        clean = to_clean_text((fixtures / name).read_text(encoding="utf-8"))
        reached = payload in clean

        produced = verdict = None
        if reached and verify is not None:
            produced, verdict = asyncio.run(verify(clean))

        results.append(InjectionResult(fixture=name, reached_extraction=reached,
                                       claim_produced=produced, verdict=verdict))

    return results
