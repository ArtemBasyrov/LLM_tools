"""
Triage: classify a user request as 'simple' or 'complex'.

Keyword + structural heuristic only — we do NOT spend a separate LLM call
on triage (the user explicitly chose in-conversation reflection over
separate model calls).

A 'complex' classification emits a planning hint that the orchestrator
injects into the conversation; the model is then expected to call
``plan_create`` before answering.
"""

from __future__ import annotations

from dataclasses import dataclass


_COMPLEX_KEYWORDS = (
    "build ",
    "implement ",
    "refactor",
    "create ",
    "add a",
    "add new",
    "add support",
    "design ",
    "fix all",
    "set up",
    "setup ",
    "write tests",
    "add tests",
    "migrate",
    "integrate",
    "rework",
    "overhaul",
    "generate ",
    "scaffold",
    "port ",
)

_MULTI_STEP_MARKERS = (
    "and then ",
    "after that",
    "finally ",
    "step 1",
    "first,",
    "1. ",
    "1) ",
)


@dataclass
class Triage:
    complex: bool
    reasons: list[str]

    @property
    def simple(self) -> bool:
        return not self.complex


def classify(user_msg: str) -> Triage:
    msg = user_msg.lower()
    reasons: list[str] = []

    for kw in _COMPLEX_KEYWORDS:
        if kw in msg:
            reasons.append(f"keyword:{kw.strip()}")

    for marker in _MULTI_STEP_MARKERS:
        if marker in msg:
            reasons.append(f"multi-step:{marker.strip()}")

    long_enough = len(user_msg) > 160
    many_ands = msg.count(" and ") >= 2
    if long_enough and many_ands:
        reasons.append("length+conjunctions")

    return Triage(complex=bool(reasons), reasons=reasons[:5])
