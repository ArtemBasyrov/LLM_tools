"""
Hypothesis ledger — the atomic anti-loop primitive for debug mode.

The ledger is a small, in-process list of candidate explanations for the
current failure, each with a status: ``untested`` | ``confirmed`` |
``refuted`` | ``inconclusive``. It exists to solve one specific pathology:
a mid-sized local model, when debugging, tends to re-propose an idea it
already ruled out two turns ago because the evidence has fallen out of
its attention window or been paraphrased into something that no longer
obviously matches.

Writing down each hypothesis and its outcome — and re-reading the list —
gives the model an external memory of what has been tried.

Scope
-----
Ledger lives in memory for the current process. It is NOT persisted across
restarts (debugging sessions are typically short; persistence is overkill
and risks carrying stale claims between unrelated bugs). Use
``hypothesis_clear`` when starting a new bug.

These tools are only exposed while the active mode is ``debug`` — the
orchestrator activates them on entry to debug mode and deactivates them on
exit.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from typing import Literal

from tools import register


Status = Literal["untested", "confirmed", "refuted", "inconclusive"]
_VALID_STATUSES = ("untested", "confirmed", "refuted", "inconclusive")


@dataclass
class _Hypothesis:
    id: int
    claim: str
    status: str = "untested"
    evidence: str = ""
    notes: list[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)


_ledger: list[_Hypothesis] = []
_next_id: int = 1


def _reset() -> None:
    """Test hook."""
    global _ledger, _next_id
    _ledger = []
    _next_id = 1


def _find(hid: int) -> _Hypothesis | None:
    for h in _ledger:
        if h.id == hid:
            return h
    return None


def _serialize(h: _Hypothesis) -> dict:
    d = asdict(h)
    # Drop timestamps from the model-facing payload — they are noise.
    d.pop("created_at", None)
    d.pop("updated_at", None)
    return d


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@register(
    description=(
        "Record a new debugging hypothesis BEFORE testing it. The ledger is "
        "your external memory of what has been tried, so each candidate root "
        "cause goes here first. Returns the hypothesis id — use it with "
        "`hypothesis_update` once you've checked the claim. "
        "Examples: "
        "hypothesis_record(claim='The test fails because config.yaml is missing the `timeout` key'); "
        "hypothesis_record(claim='The regex in parse_line() drops leading whitespace, breaking indented inputs')."
    ),
    always_on=False,
    parameters={
        "type": "object",
        "properties": {
            "claim": {
                "type": "string",
                "description": "Concrete, testable claim about the root cause. One sentence. Avoid compound claims — split into separate hypotheses.",
            },
        },
        "required": ["claim"],
    },
)
def hypothesis_record(claim: str) -> str:
    global _next_id
    claim = (claim or "").strip()
    if not claim:
        return json.dumps({"error": "empty_claim"})

    # Duplicate guard — if this claim is already refuted, refuse so the model
    # doesn't silently re-explore it. This is the whole point of the ledger.
    for h in _ledger:
        if h.claim.strip().lower() == claim.lower():
            return json.dumps(
                {
                    "error": "duplicate",
                    "existing_id": h.id,
                    "existing_status": h.status,
                    "message": (
                        f"This hypothesis already exists as #{h.id} "
                        f"(status={h.status}). Do not re-test it — pick a "
                        "different angle. Call `hypothesis_list` to see "
                        "everything tried so far."
                    ),
                }
            )

    h = _Hypothesis(id=_next_id, claim=claim)
    _next_id += 1
    _ledger.append(h)
    return json.dumps({"recorded": True, "id": h.id, "total": len(_ledger)})


@register(
    description=(
        "Update a recorded hypothesis with the outcome of testing it. Status "
        "must be one of: `confirmed` (evidence proves the claim), `refuted` "
        "(evidence disproves it), `inconclusive` (check was ambiguous — you "
        "can revisit later). Attach concrete evidence: file paths, command "
        "output excerpts, exit codes. "
        "Once a hypothesis is `refuted`, the ledger will block you from "
        "re-recording the same claim."
    ),
    always_on=False,
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "integer",
                "description": "Hypothesis id returned by `hypothesis_record`.",
            },
            "status": {
                "type": "string",
                "enum": ["confirmed", "refuted", "inconclusive"],
                "description": "Outcome of the test.",
            },
            "evidence": {
                "type": "string",
                "description": "Concrete evidence: what you ran, what you saw. e.g. 'ran `pytest tests/test_parse.py::test_indent`; exit=1; AssertionError on line 42 shows leading spaces stripped.'",
            },
        },
        "required": ["id", "status", "evidence"],
    },
)
def hypothesis_update(id: int, status: str, evidence: str) -> str:
    if status not in _VALID_STATUSES or status == "untested":
        return json.dumps(
            {
                "error": "invalid_status",
                "valid": ["confirmed", "refuted", "inconclusive"],
            }
        )
    h = _find(int(id))
    if h is None:
        return json.dumps({"error": "unknown_id", "id": id})
    h.status = status
    h.evidence = evidence or h.evidence
    h.updated_at = time.time()
    return json.dumps({"updated": True, "id": h.id, "status": h.status})


@register(
    description=(
        "List all hypotheses recorded this debugging session with their "
        "statuses and evidence. Call this whenever you feel uncertain about "
        "what has already been tried — that feeling is a loop warning. "
        "Also useful when summarizing findings back to the user."
    ),
    always_on=False,
)
def hypothesis_list() -> str:
    return json.dumps(
        {
            "count": len(_ledger),
            "hypotheses": [_serialize(h) for h in _ledger],
        },
        ensure_ascii=False,
    )


@register(
    description=(
        "Clear the hypothesis ledger. Call this when starting a new, "
        "unrelated bug — old hypotheses from a different problem would be "
        "misleading. Do NOT call this mid-investigation."
    ),
    always_on=False,
)
def hypothesis_clear() -> str:
    n = len(_ledger)
    _reset()
    return json.dumps({"cleared": True, "removed": n})
