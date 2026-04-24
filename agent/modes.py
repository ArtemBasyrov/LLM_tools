"""
Operating modes + sampling profiles.

Each turn runs in exactly one Mode. The Mode bundles:
  * a ``SamplingProfile`` — Ollama options (temperature, repeat_penalty, ...)
    passed per-request, so parameters retune dynamically without reloading
    the model.
  * a system addendum — injected as a sticky ``[SYSTEM MODE]`` message on
    mode change so the model knows the active playbook.
  * an optional tool set activated on entry and deactivated on exit
    (currently used only by ``debug`` for the hypothesis ledger).

Mode transitions happen in three ways:
  1. Heuristic on turn start (``suggest_mode``) — only auto-switches when
     the current mode is the default ``chat`` and we have a strong signal.
  2. User command ``/mode <name>`` handled in ``main.py``.
  3. Model self-declaration via the ``set_mode`` tool (``tools/mode_tools``).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Optional


class Mode(str, Enum):
    CHAT = "chat"
    CODE = "code"
    DEBUG = "debug"
    RESEARCH = "research"
    PLAN = "plan"

    @classmethod
    def parse(cls, name: str) -> Optional["Mode"]:
        if not name:
            return None
        key = name.strip().lower()
        for m in cls:
            if m.value == key:
                return m
        return None


@dataclass(frozen=True)
class SamplingProfile:
    """Per-request Ollama options. Passed as ``options=`` to ``ollama.chat``.

    NOTE: ``num_ctx`` is intentionally excluded — changing it forces Ollama
    to rebuild the KV cache, which is far too expensive to do per turn.
    """

    temperature: float
    top_p: float
    top_k: int
    repeat_penalty: float
    num_predict: int

    def to_ollama_options(self) -> dict:
        return {
            "temperature": self.temperature,
            "top_p": self.top_p,
            "top_k": self.top_k,
            "repeat_penalty": self.repeat_penalty,
            "num_predict": self.num_predict,
        }


# Hand-tuned starting points. The goal is NOT "best for benchmark X" — it is
# "characteristic behaviour that matches the mode's job". Tune by observation.
PROFILES: dict[Mode, SamplingProfile] = {
    # Default conversational — balanced creativity and coherence.
    Mode.CHAT: SamplingProfile(
        temperature=0.7, top_p=0.90, top_k=40, repeat_penalty=1.10, num_predict=1024
    ),
    # Code generation — lower temperature for syntactic determinism.
    Mode.CODE: SamplingProfile(
        temperature=0.30, top_p=0.85, top_k=40, repeat_penalty=1.10, num_predict=2048
    ),
    # Debugging — aggressive anti-loop settings. Low temperature to commit to
    # hypotheses, high repeat_penalty to actively discourage re-exploring the
    # same dead ends, tighter top_k to prune the "plausible but already-tried"
    # tail.
    Mode.DEBUG: SamplingProfile(
        temperature=0.20, top_p=0.80, top_k=30, repeat_penalty=1.20, num_predict=1536
    ),
    # Research — moderate temperature, slightly tighter top_p, enough budget
    # to synthesize across sources.
    Mode.RESEARCH: SamplingProfile(
        temperature=0.50, top_p=0.90, top_k=40, repeat_penalty=1.12, num_predict=2048
    ),
    # Planning — low-ish temperature so steps are concrete and atomic, not
    # creative; strong repeat penalty so step descriptions don't collapse
    # into rephrasings of each other.
    Mode.PLAN: SamplingProfile(
        temperature=0.40, top_p=0.85, top_k=40, repeat_penalty=1.15, num_predict=2048
    ),
}


MODE_ADDENDA: dict[Mode, str] = {
    Mode.CHAT: (
        "[SYSTEM MODE] Active mode: chat. General conversation. No special "
        "constraints beyond the base system prompt."
    ),
    Mode.CODE: (
        "[SYSTEM MODE] Active mode: code. You are producing or editing code. "
        "Rules:\n"
        "  - Prefer small, verifiable edits over big rewrites.\n"
        "  - After writing a file, verify it (read_file or bash syntax check).\n"
        "  - If a test suite exists, run it before declaring work complete.\n"
        "  - Do NOT speculate about code — read it first."
    ),
    Mode.DEBUG: (
        "[SYSTEM MODE] Active mode: debug. You are diagnosing a concrete "
        "failure. Anti-loop discipline is mandatory:\n"
        "  1. Form ONE hypothesis at a time. Call `hypothesis_record(claim=...)` "
        "BEFORE running tools to test it.\n"
        "  2. After each check, call `hypothesis_update(id=..., status=...)` "
        "with one of: confirmed | refuted | inconclusive.\n"
        "  3. NEVER re-test a hypothesis already marked `refuted`. If you "
        "catch yourself about to, call `hypothesis_list` and pick something new.\n"
        "  4. Call `hypothesis_list` whenever you feel uncertain about what "
        "you've already tried — that feeling is the loop warning you.\n"
        "  5. Once a root cause is `confirmed`, state it plainly, then fix.\n"
        "  6. If all current hypotheses are refuted and you are stuck, say so "
        "and ask the user for a new angle rather than spinning."
    ),
    Mode.RESEARCH: (
        "[SYSTEM MODE] Active mode: research. Gather information before "
        "synthesizing. Rules:\n"
        "  - Cite concrete sources (URLs, file paths, memory IDs).\n"
        "  - Distinguish verified claims from speculation explicitly.\n"
        "  - Prefer multiple sources over a single authoritative-sounding one."
    ),
    Mode.PLAN: (
        "[SYSTEM MODE] Active mode: plan. You are producing a plan, not "
        "executing it. Rules:\n"
        "  - Draft `plan_create` + `plan_add_step` calls now.\n"
        "  - Each step must be atomic and independently verifiable.\n"
        "  - Do NOT begin executing steps in this mode — switch to code or "
        "debug mode (via `set_mode`) once the plan is approved."
    ),
}


# Tools activated on entry to a mode and deactivated on exit. Keyed by mode.
MODE_TOOLSETS: dict[Mode, tuple[str, ...]] = {
    Mode.DEBUG: (
        "hypothesis_record",
        "hypothesis_update",
        "hypothesis_list",
        "hypothesis_clear",
    ),
}


# ---------------------------------------------------------------------------
# Module-level mode state
# ---------------------------------------------------------------------------
#
# The orchestrator reads this each turn; tools (set_mode) and the /mode user
# command mutate it. Kept module-level (not orchestrator-instance) so tools
# can change mode without needing a handle to the orchestrator.

_current: Mode = Mode.CHAT


def get_current_mode() -> Mode:
    return _current


def set_current_mode(mode: Mode) -> Mode:
    """Set the module-level current mode. Returns the previous mode."""
    global _current
    prev = _current
    _current = mode
    return prev


def profile_for(mode: Mode) -> SamplingProfile:
    return PROFILES[mode]


def addendum_for(mode: Mode) -> str:
    return MODE_ADDENDA[mode]


def toolset_for(mode: Mode) -> tuple[str, ...]:
    return MODE_TOOLSETS.get(mode, ())


# ---------------------------------------------------------------------------
# Heuristic suggestion
# ---------------------------------------------------------------------------


_DEBUG_KW = (
    "debug",
    "bug",
    "error",
    "traceback",
    "exception",
    "crash",
    "failing",
    "broken",
    "not working",
    "doesn't work",
    "doesnt work",
    "stack trace",
    "stacktrace",
    "segfault",
    "panic",
)
_CODE_KW = (
    "implement ",
    "refactor",
    "add function",
    "write code",
    "write a function",
    "build ",
    "scaffold",
    "edit ",
    "rewrite",
)
_RESEARCH_KW = (
    "research",
    "find out",
    "look up",
    "compare ",
    "survey ",
    "overview of",
    "what is ",
    "what are ",
    "how does ",
    "how do ",
)
_PLAN_KW = (
    "plan ",
    "outline ",
    "design ",
    "draft a plan",
    "sketch ",
)


def suggest_mode(user_msg: str) -> Optional[Mode]:
    """Return a suggested mode based on keyword heuristics, or None if no
    confident signal. Intentionally conservative — when unsure we return
    None so the current mode is preserved."""
    if not user_msg:
        return None
    msg = user_msg.lower()

    def _any(kws):
        return any(kw in msg for kw in kws)

    # Debug has priority — it's the costliest mode to mis-miss (circular thinking).
    if _any(_DEBUG_KW):
        return Mode.DEBUG
    if _any(_PLAN_KW):
        return Mode.PLAN
    if _any(_CODE_KW):
        return Mode.CODE
    if _any(_RESEARCH_KW):
        return Mode.RESEARCH
    return None
