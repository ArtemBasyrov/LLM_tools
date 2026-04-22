"""
Plan-management tools exposed to the LLM.

These tools let the model create, inspect, and update a persistent plan
for multi-step work. All of them return JSON strings (same convention as
``tools/session.py``).

The plan itself lives on disk under ``$LLM_PLAN_DIR`` (see ``agent.plan``),
so it survives context-window trims AND process restarts — the core
feature we need for long-horizon tasks.

Completing a step with ``plan_complete_step`` marks it as ``completed``
but leaves ``verified=False``. The orchestrator (Phase 2) watches for
this and injects a ``[SYSTEM VERIFIER]`` prompt demanding independent
evidence before the step is truly considered done.
"""

import json

from tools import register
from agent import plan as _plan


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _load_or_err() -> tuple[_plan.Plan | None, str | None]:
    p = _plan.load_active()
    if p is None:
        return None, json.dumps(
            {
                "error": "no_active_plan",
                "message": "No active plan. Call plan_create(goal=...) first.",
            }
        )
    return p, None


def _ok(**kwargs) -> str:
    return json.dumps(kwargs, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@register(
    description=(
        "Create a new plan for a multi-step task. "
        "Use when the user's request requires 3+ distinct actions, is risky/irreversible, "
        "or spans more than a single tool call. "
        "Only one plan is active at a time — this will FAIL if another plan is active "
        "unless ``replace=true``. "
        "After creating the plan, add steps via plan_add_step, then work them one at a time. "
        "Examples: "
        "user asks 'refactor the auth module and add tests' → plan_create(goal='Refactor auth module and add tests'); "
        "user asks 'what time is it?' → do NOT plan_create, just answer; "
        "user says 'start over on the plan' → plan_create(goal='...', replace=true)."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "goal": {
                "type": "string",
                "description": "Top-level user intent in one sentence. e.g. 'Refactor auth module and add tests covering token expiry'.",
            },
            "replace": {
                "type": "boolean",
                "description": "If true, archive any existing active plan before creating the new one. Default false.",
            },
        },
        "required": ["goal"],
    },
)
def plan_create(goal: str, replace: bool = False) -> str:
    existing = _plan.load_active()
    if existing is not None and not replace:
        return json.dumps(
            {
                "error": "active_plan_exists",
                "active_plan_id": existing.id,
                "goal": existing.goal,
                "message": "An active plan exists. Pass replace=true to archive it first, "
                "or use plan_status to inspect it.",
            }
        )
    if existing is not None:
        _plan.archive(existing, outcome="replaced")
    p = _plan.new_plan(goal)
    _plan.save_active(p)
    return _ok(created=True, plan_id=p.id, goal=p.goal, steps=0)


@register(
    description=(
        "Append a step to the active plan. Each step should be ATOMIC "
        "(one clear action), VERIFIABLE (a named completion criterion), and independently testable. "
        "Call this repeatedly after plan_create to populate the step list. "
        "Examples: "
        "plan_add_step(description='Create tools/wordcount.py with a word_count function', verification='File exists and word_count(\"a b c\") returns 3'); "
        "plan_add_step(description='Add tests/test_wordcount.py covering empty/unicode/punctuation', verification='pytest tests/test_wordcount.py passes')."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Concrete action to take. Start with a verb: 'Create...', 'Refactor...', 'Verify...'.",
            },
            "verification": {
                "type": "string",
                "description": "How to confirm the step is done. Should be checkable by reading a file, running a command, or running tests.",
            },
        },
        "required": ["description", "verification"],
    },
)
def plan_add_step(description: str, verification: str) -> str:
    p, err = _load_or_err()
    if err:
        return err
    step = p.add_step(description, verification)
    _plan.save_active(p)
    return _ok(added=True, step_id=step.id, total_steps=len(p.steps))


@register(
    description=(
        "Mark a step as in_progress. Call this BEFORE starting work on a step "
        "so the harness knows what you're currently doing. "
        "Increments the attempt counter on the step."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "step_id": {
                "type": "integer",
                "description": "The numeric ID of the step (as returned by plan_add_step or shown in plan_status).",
            },
        },
        "required": ["step_id"],
    },
)
def plan_start_step(step_id: int) -> str:
    p, err = _load_or_err()
    if err:
        return err
    step = p.start_step(int(step_id))
    if step is None:
        return json.dumps({"error": "unknown_step", "step_id": step_id})
    _plan.save_active(p)
    return _ok(
        started=True,
        step_id=step.id,
        description=step.description,
        attempts=step.attempts,
    )


@register(
    description=(
        "Mark a step as completed AND provide concrete evidence of completion. "
        "The harness will automatically request an independent verification pass — "
        "the step is NOT truly done until verify_report(verified=true, ...) follows. "
        "If your evidence is weak or wrong, the verifier will roll the step back. "
        "Examples: "
        "plan_complete_step(step_id=1, evidence='Created tools/wordcount.py:1-25; file_info shows size 612 bytes; word_count(\"a b c\")=3 confirmed.'); "
        "plan_complete_step(step_id=2, evidence='pytest tests/test_wordcount.py exit=0, 5 tests passed.')."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "step_id": {
                "type": "integer",
                "description": "The numeric ID of the step to complete.",
            },
            "evidence": {
                "type": "string",
                "description": "Concrete evidence: file paths, command output excerpts, exit codes, test counts. Be specific — vague evidence is rejected by the verifier.",
            },
        },
        "required": ["step_id", "evidence"],
    },
)
def plan_complete_step(step_id: int, evidence: str) -> str:
    p, err = _load_or_err()
    if err:
        return err
    step = p.complete_step(int(step_id), evidence)
    if step is None:
        return json.dumps({"error": "unknown_step", "step_id": step_id})
    _plan.save_active(p)
    # The ``verification_requested`` flag is the signal the orchestrator
    # uses to inject a [SYSTEM VERIFIER] message next.
    return _ok(
        completed=True,
        step_id=step.id,
        verification_requested=True,
        message=(
            "Step marked complete. A [SYSTEM VERIFIER] prompt will be injected; "
            "you must independently confirm the evidence before the step is considered done."
        ),
    )


@register(
    description=(
        "Return the full state of the active plan: goal, all steps with status/verified/attempts, "
        "current step pointer, and whether any steps are pending verification. "
        "Call this at the START of every turn when a plan is active so you know exactly "
        "which step to work on next. Also call after any plan mutation to confirm state."
    ),
    always_on=True,
)
def plan_status() -> str:
    p = _plan.load_active()
    if p is None:
        return json.dumps({"active": False, "message": "No active plan."})
    return json.dumps({"active": True, **_plan.detailed_summary(p)}, ensure_ascii=False)


@register(
    description=(
        "Append a note/observation to a step WITHOUT changing its status. "
        "Use this to record discoveries or partial progress that don't yet warrant completion. "
        "Notes persist to disk with the plan."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "step_id": {"type": "integer", "description": "Step ID to annotate."},
            "note": {
                "type": "string",
                "description": "Short observation. e.g. 'Found existing helper in tools/_util.py — reuse.'",
            },
        },
        "required": ["step_id", "note"],
    },
)
def plan_log(step_id: int, note: str) -> str:
    p, err = _load_or_err()
    if err:
        return err
    step = p.log_note(int(step_id), note)
    if step is None:
        return json.dumps({"error": "unknown_step", "step_id": step_id})
    _plan.save_active(p)
    return _ok(logged=True, step_id=step.id, note_count=len(step.notes))


@register(
    description=(
        "Abandon the active plan. The plan is archived to disk with the given reason as outcome. "
        "Use ONLY when: (a) the user explicitly cancels, (b) the plan is obsolete due to new info, "
        "or (c) the task has been proven impossible with current tools. "
        "After abandoning, the next plan_create call will succeed without needing replace=true."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "reason": {
                "type": "string",
                "description": "Why you're abandoning. e.g. 'User switched to a different task.' or 'Blocked: tool X not available.'",
            },
        },
        "required": ["reason"],
    },
)
def plan_abandon(reason: str) -> str:
    p = _plan.load_active()
    if p is None:
        return json.dumps({"error": "no_active_plan"})
    dest = _plan.archive(p, outcome=f"abandoned: {reason}")
    return _ok(abandoned=True, plan_id=p.id, archived_to=str(dest))
