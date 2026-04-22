"""
Verifier: build injection text for a completed-but-unverified step.

The orchestrator consults this module when ``plan_complete_step`` returns
``verification_requested=true`` — it pulls the step's details from the
plan on disk and produces the ``[SYSTEM VERIFIER]`` message that gets
appended to the conversation.

Actual verification is performed BY THE MODEL on its next turn (reading
files, running bash, etc.) and reported via the ``verify_report`` tool.
This module only builds the prompt.
"""

from __future__ import annotations

from typing import Optional

from agent import plan as _plan
from agent import prompts


def build_injection(step_id: int) -> Optional[str]:
    """Return the verifier prompt for step_id, or None if no such step."""
    p = _plan.load_active()
    if p is None:
        return None
    step = p.get_step(step_id)
    if step is None:
        return None
    return prompts.verifier(
        step_id=step.id,
        description=step.description,
        verification=step.verification,
        evidence=step.evidence or "",
    )
