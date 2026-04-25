"""
Prompt templates for harness-injected synthetic messages.

All injected messages use a distinctive ``[SYSTEM <ROLE>]`` prefix so the
model knows they are authoritative orchestrator instructions, not user
speech. The system prompt teaches this convention.
"""

from __future__ import annotations


TRIAGE_HINT = (
    "[SYSTEM TRIAGE] This request looks multi-step (reasons: {reasons}). "
    "Before answering, create a plan: call `plan_create(goal=...)` then "
    "`plan_add_step(...)` for each atomic, verifiable step. Work the plan one "
    "step at a time — do NOT emit a final answer until every step is completed "
    "and verified."
)


VERIFIER = (
    "[SYSTEM VERIFIER] Step #{step_id} was just marked complete.\n"
    "  description: {description}\n"
    "  verification criterion: {verification}\n"
    "  claimed evidence: {evidence}\n\n"
    "Independently verify this evidence now. Use the appropriate tools "
    "(read_file / bash / search_file / file_info) to check the claim without "
    "relying on your prior reasoning. Then call "
    "`verify_report(step_id={step_id}, verified=<bool>, notes=<str>)`. "
    "Be strict: if you cannot concretely confirm the evidence, report "
    "verified=false so the step is rolled back."
)


PLAN_NUDGE = (
    "[SYSTEM ORCHESTRATOR] The active plan is not yet complete — current step "
    "#{current_step} is '{status}'. Do not emit a final answer to the user yet. "
    "Continue executing the plan: call `plan_status` to inspect state, work the "
    "current step, then call `plan_complete_step` with evidence."
)


CRITIC = (
    "[SYSTEM CRITIC] Review the final response you just drafted. Check for:\n"
    "  (a) unsupported claims — anything you stated without tool-verified evidence\n"
    "  (b) skipped plan steps — any pending or unverified steps\n"
    "  (c) missing verification — claims of completion not backed by read_file/bash checks\n"
    "  (d) edge cases or caveats you should have surfaced\n\n"
    "Reply with a JSON object on its own line, no other prose:\n"
    '  {{"accept": true, "issues": []}}  — if the response is solid\n'
    '  {{"accept": false, "issues": ["...", "..."]}}  — otherwise\n\n'
    "This is round {round} of {max_rounds}."
)


CRITIC_ISOLATED_SYSTEM = (
    "You are a strict reviewer. Your only job is to evaluate an assistant's "
    "drafted reply and return a JSON verdict. Do not call tools. Do not write "
    "prose beyond the JSON object."
)


CRITIC_ISOLATED = (
    "[SYSTEM CRITIC] You are reviewing an assistant's drafted final reply.\n\n"
    "User's most recent message:\n"
    "---\n{user_question}\n---\n\n"
    "Drafted reply:\n"
    "---\n{drafted_response}\n---\n\n"
    "Evaluate whether the reply:\n"
    "  (a) actually answers the user's question\n"
    "  (b) makes claims without evidence\n"
    "  (c) misses important caveats or edge cases\n"
    "  (d) is internally inconsistent\n\n"
    "Reply with a single JSON object, nothing else:\n"
    '  {{"accept": true, "issues": []}}  — if the reply is solid\n'
    '  {{"accept": false, "issues": ["...", "..."]}}  — otherwise\n\n'
    "Round {round} of {max_rounds}."
)


SNAPSHOT_NUDGE = (
    "[SYSTEM ORCHESTRATOR] Context is {pct}% full and an active plan is in "
    "flight. Before continuing, call `session_save` to snapshot your current "
    "progress (include the plan id, current step #{current_step}, and any "
    "in-flight findings). This lets the plan survive context trimming."
)


CRITIC_REVISE = (
    "[SYSTEM CRITIC] You identified issues: {issues}\n\n"
    "Revise your final response addressing each issue. Use tools to confirm any claim "
    "you repeat. Then emit the revised response as plain text (no tool calls). "
    "The critic will review again."
)


def triage_hint(reasons: list[str]) -> str:
    return TRIAGE_HINT.format(reasons=", ".join(reasons) or "multi-step")


def verifier(step_id: int, description: str, verification: str, evidence: str) -> str:
    return VERIFIER.format(
        step_id=step_id,
        description=description,
        verification=verification,
        evidence=evidence or "(no evidence provided)",
    )


def plan_nudge(current_step: int, status: str) -> str:
    return PLAN_NUDGE.format(current_step=current_step, status=status)


def snapshot_nudge(pct: int, current_step: int) -> str:
    return SNAPSHOT_NUDGE.format(pct=pct, current_step=current_step)


def critic(round_num: int, max_rounds: int) -> str:
    return CRITIC.format(round=round_num, max_rounds=max_rounds)


def critic_isolated(
    user_question: str, drafted_response: str, round_num: int, max_rounds: int
) -> str:
    return CRITIC_ISOLATED.format(
        user_question=user_question or "(no user question recoverable)",
        drafted_response=drafted_response or "(empty)",
        round=round_num,
        max_rounds=max_rounds,
    )


def critic_revise(issues: list[str]) -> str:
    return CRITIC_REVISE.format(issues="; ".join(issues) if issues else "(none)")
