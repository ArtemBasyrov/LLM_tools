"""
Mode-switching tool exposed to the model.

The model can call ``set_mode`` at any time to declare a mode change. The
orchestrator picks up the new mode at the start of the next inference cycle:
it swaps the sampling profile, injects the mode addendum, and (un)activates
mode-specific tools.

The user can also switch modes via ``/mode <name>`` in ``main.py`` — same
underlying state.
"""

from __future__ import annotations

import json

from tools import register
from agent import modes as _modes


@register(
    description=(
        "Declare a change of operating mode. Modes bundle sampling parameters "
        "(temperature, repeat_penalty, etc.) and behaviour rules tuned for "
        "different kinds of task. "
        "Call this when you recognize the current work would benefit from a "
        "different mode — e.g. a chat turn turns into a debugging session. "
        "The new mode takes effect on the NEXT inference cycle. "
        "Available modes: "
        "`chat` (default, general conversation), "
        "`code` (producing/editing code, low temperature), "
        "`debug` (diagnosing failures, anti-loop discipline, activates the "
        "hypothesis ledger tools), "
        "`research` (gathering and synthesizing info with citations), "
        "`plan` (drafting a plan without executing it). "
        "Examples: "
        "user reports 'the test fails with TypeError' → set_mode(mode='debug', "
        "reason='user surfaced a concrete test failure'); "
        "user asks 'design the new auth flow' → set_mode(mode='plan', "
        "reason='design task, should produce steps not code')."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "mode": {
                "type": "string",
                "enum": ["chat", "code", "debug", "research", "plan"],
                "description": "The mode to switch to.",
            },
            "reason": {
                "type": "string",
                "description": "One-sentence justification for the switch. Kept in logs; helps the user understand why behaviour changed.",
            },
        },
        "required": ["mode", "reason"],
    },
)
def set_mode(mode: str, reason: str) -> str:
    m = _modes.Mode.parse(mode)
    if m is None:
        return json.dumps(
            {
                "error": "unknown_mode",
                "message": f"Unknown mode '{mode}'. Valid: chat, code, debug, research, plan.",
            }
        )
    prev = _modes.set_current_mode(m)
    return json.dumps(
        {
            "switched": True,
            "from": prev.value,
            "to": m.value,
            "reason": reason,
            "message": (
                f"Mode is now '{m.value}'. The orchestrator will apply the new "
                "sampling profile and mode rules on the next inference cycle."
            ),
        }
    )
