"""
Verification tool: ``verify_report``.

The model calls this in response to a ``[SYSTEM VERIFIER]`` injection.
It reports whether independent checks confirmed the previously claimed
evidence. The tool updates the plan on disk accordingly:

- ``verified=true``  → step is marked verified; current_step advances.
- ``verified=false`` → step is rolled back to ``in_progress`` so the
                       model can try again with better evidence.
"""

import json

from tools import register
from agent import plan as _plan


@register(
    description=(
        "Report the result of a [SYSTEM VERIFIER] check. Call this exactly once after "
        "independently confirming (or failing to confirm) a step's evidence using "
        "read_file / bash / search_file / file_info. "
        "verified=true marks the step truly done (and advances the plan); "
        "verified=false rolls the step back to in_progress so you can retry. "
        "Be strict: don't accept your own prior claims — use tool evidence only. "
        "Examples: "
        "verify_report(step_id=1, verified=true, notes='read_file shows tools/wordcount.py exists with word_count defined; bash ran function and returned 3 for 3-word input.'); "
        "verify_report(step_id=2, verified=false, notes='tests/test_wordcount.py does not exist; previous completion was premature.')."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "step_id": {
                "type": "integer",
                "description": "The step being verified (from the preceding [SYSTEM VERIFIER] injection).",
            },
            "verified": {
                "type": "boolean",
                "description": "Your honest assessment after independent tool-based checks. If uncertain, pass false.",
            },
            "notes": {
                "type": "string",
                "description": "What you checked and what you found. Cite file paths and command output. Brief but specific.",
            },
        },
        "required": ["step_id", "verified", "notes"],
    },
)
def verify_report(step_id: int, verified: bool, notes: str) -> str:
    p = _plan.load_active()
    if p is None:
        return json.dumps({"error": "no_active_plan"})
    step = p.get_step(int(step_id))
    if step is None:
        return json.dumps({"error": "unknown_step", "step_id": step_id})

    if verified:
        p.mark_verified(step.id, notes=notes)
        p._advance_current()
        _plan.save_active(p)
        out = {
            "verified": True,
            "step_id": step.id,
            "plan_complete": p.is_complete(),
            "next_step": p.current_step,
        }
    else:
        p.rollback_step(step.id, reason=notes)
        _plan.save_active(p)
        out = {
            "verified": False,
            "step_id": step.id,
            "rolled_back": True,
            "notes": notes,
        }
    return json.dumps(out, ensure_ascii=False)
