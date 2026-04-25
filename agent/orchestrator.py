"""
Orchestrator — state machine that wraps a single user turn.

Responsibilities
----------------
1. Append the user's message with a context-window prefix.
2. Run triage; if complex and no active plan, inject a planning hint.
3. Drive the ollama <-> tool loop.
4. Watch tool results: on ``plan_complete_step`` with
   ``verification_requested=true``, enqueue a ``[SYSTEM VERIFIER]``
   injection that fires on the next inference cycle.
5. When the model emits a final (no-tool-call) response:
     a. If there are pending verifier injections, fire them and loop.
     b. If the active plan is not yet fully verified-complete, nudge
        the model back to work.
     c. Otherwise run the critic loop (Phase 3; ``CRITIC_MAX_ROUNDS``).
6. Return when the model produces an accepted final answer OR the
   critic has exhausted its rounds.

Configuration (environment variables)
-------------------------------------
AGENTIC_MODE=true|false        — master switch (default: true).
CRITIC_MAX_ROUNDS=<int>         — max critic revise rounds (default: 2).
ENFORCE_PLANNING=true|false    — block final answer while plan
                                  incomplete (default: true).

Dependency injection
--------------------
``chat_fn`` and ``renderer`` are arguments so tests can supply fakes
without touching Ollama or terminal I/O.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Optional

from agent import critic as _critic
from agent import modes as _modes
from agent import plan as _plan
from agent import prompts
from agent import triage as _triage
from agent import verifier as _verifier


# ---------------------------------------------------------------------------
# Renderer protocol (duck-typed)
# ---------------------------------------------------------------------------
#
# Any object with these methods works. We don't import ``rendering`` here so
# this module stays unit-testable.  ``main.py`` builds a Renderer that simply
# delegates to rendering.py's existing print_* functions.


class NullRenderer:
    """No-op renderer for tests."""

    def thinking_start(self):
        pass

    def thinking_token(self, text: str):
        pass

    def thinking_end(self):
        pass

    def response_start(self):
        pass

    def response_token(self, text: str):
        pass

    def response_end(self):
        pass

    def tool_call(self, name: str, args: dict):
        pass

    def tool_result(self, result: str):
        pass

    def stats(self, elapsed, pt, et, cu, cw):
        pass

    def orchestrator_event(self, kind: str, detail: str = ""):
        pass

    def blank_line(self):
        pass


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _env_bool(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None:
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Turn stats
# ---------------------------------------------------------------------------


@dataclass
class TurnStats:
    prompt_tokens: int = 0
    eval_tokens: int = 0
    context_used: int = 0
    elapsed: float = 0.0
    critic_rounds: int = 0
    verifications: int = 0
    rollbacks: int = 0


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


class Orchestrator:
    """State machine that runs one user turn with planning/verify/critic."""

    MAX_INNER_ITERATIONS = 40  # hard cap on model-tool loop to prevent runaway
    AUTO_SNAPSHOT_PCT = 0.70  # context fill% at which we nudge the model to snapshot

    def __init__(
        self,
        messages: list[dict],
        context_window: int,
        model: str,
        *,
        chat_fn: Optional[Callable[..., Any]] = None,
        tool_call_fn: Optional[Callable[[str, dict], str]] = None,
        tool_schemas_fn: Optional[Callable[[], list[dict]]] = None,
        renderer: Optional[Any] = None,
        compact_fn: Optional[Callable[[list, int, int], bool]] = None,
        trim_fn: Optional[Callable[[list], None]] = None,
        context_prefix_fn: Optional[Callable[[int, int], str]] = None,
    ) -> None:
        self.messages = messages
        self.context_window = context_window
        self.model = model
        self.context_used = 0

        # Injected deps
        if chat_fn is None:
            import backend

            chat_fn = backend.chat
        self._chat = chat_fn

        if tool_call_fn is None or tool_schemas_fn is None:
            from tools import call as _default_call, schemas as _default_schemas

            tool_call_fn = tool_call_fn or _default_call
            tool_schemas_fn = tool_schemas_fn or _default_schemas
        self._call = tool_call_fn
        self._schemas = tool_schemas_fn

        self._renderer = renderer or NullRenderer()

        if compact_fn is None or trim_fn is None or context_prefix_fn is None:
            from context_window import (
                compact_messages as _default_compact,
                trim_messages as _default_trim,
                context_prefix as _default_prefix,
            )

            compact_fn = compact_fn or _default_compact
            trim_fn = trim_fn or _default_trim
            context_prefix_fn = context_prefix_fn or _default_prefix
        self._compact = compact_fn
        self._trim = trim_fn
        self._context_prefix = context_prefix_fn

        # Config
        self.agentic = _env_bool("AGENTIC_MODE", True)
        self.enforce_planning = _env_bool("ENFORCE_PLANNING", True)
        self.critic_max_rounds = max(0, int(os.environ.get("CRITIC_MAX_ROUNDS", "2")))

        # Per-turn mutable state (reset at the start of each turn)
        self._pending_verifications: list[int] = []
        self._critic_round = 0
        self._last_response: str = ""
        self._snapshot_nudged: bool = False

        # Mode state — last applied mode, so we can detect transitions and
        # (de)activate mode-specific tools / inject the mode addendum exactly
        # once per change. Initialized to whatever modes.get_current_mode()
        # reports so the first turn injects the correct addendum.
        self._applied_mode: Optional[_modes.Mode] = None  # one nudge per turn max

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def turn(self, user_input: str) -> TurnStats:
        """Process one full user turn to completion."""
        self._pending_verifications = []
        self._critic_round = 0
        self._last_response = ""
        self._snapshot_nudged = False

        self.messages.append(
            {
                "role": "user",
                "content": self._context_prefix(self.context_used, self.context_window)
                + user_input,
            }
        )
        self._renderer.blank_line()

        # Mode suggestion — conservative auto-switch. We only override the
        # current mode if (a) the current mode is the default CHAT (so we
        # don't fight a user/model explicit choice), and (b) the heuristic
        # produces a confident non-None suggestion. Model can always call
        # set_mode to correct us; user can use /mode.
        if self.agentic and _modes.get_current_mode() == _modes.Mode.CHAT:
            suggested = _modes.suggest_mode(user_input)
            if suggested is not None and suggested != _modes.Mode.CHAT:
                _modes.set_current_mode(suggested)
                self._renderer.orchestrator_event(
                    "mode-auto", f"{suggested.value} (heuristic)"
                )

        # Triage (only when no plan is already active — an active plan implies
        # we're already in execution and planning has been done).
        if self.agentic and _plan.load_active() is None:
            hint = _triage.classify(user_input)
            if hint.complex:
                self.messages.append(
                    {"role": "user", "content": prompts.triage_hint(hint.reasons)}
                )
                self._renderer.orchestrator_event(
                    "triage", f"complex ({', '.join(hint.reasons)})"
                )

        stats = TurnStats()
        t_start = time.perf_counter()

        for _ in range(self.MAX_INNER_ITERATIONS):
            # Fire any queued verifier injections BEFORE the next inference
            while self._pending_verifications:
                step_id = self._pending_verifications.pop(0)
                self._inject_verifier(step_id)
                stats.verifications += 1

            # Auto-snapshot nudge: if we're nearing the context limit and a
            # plan is active, ask the model to snapshot so the plan survives
            # the next trim. Fires at most once per turn.
            self._maybe_nudge_snapshot()

            # Refresh the [SYSTEM FILES] sticky so the model sees its current
            # working set + any external staleness before the next inference.
            self._refresh_files_message()

            self._compact(self.messages, self.context_used, self.context_window)

            had_tool_calls = self._run_inference(stats)

            # If the model made tool calls, always loop — model will see results
            # on the next inference.
            if had_tool_calls:
                continue

            # Model emitted a final response this cycle. Check for pending work:

            # 1) Any new verifier prompts queued during the just-completed
            #    inference's tool calls? Loop to fire them.
            if self._pending_verifications:
                continue

            # 2) Is a plan active and still incomplete? Nudge.
            if self.agentic and self.enforce_planning:
                active = _plan.load_active()
                if active is not None and not active.is_complete():
                    if self._inject_plan_nudge(active):
                        continue
                    # If nudge wasn't emitted (already nudged this turn too
                    # many times), fall through to critic.

            # 3) Critic pass.
            if (
                self.agentic
                and self._critic_round < self.critic_max_rounds
                and self._should_critique()
            ):
                if self._run_critic_round():
                    stats.critic_rounds = self._critic_round
                    continue

            # Done.
            break

        stats.elapsed = time.perf_counter() - t_start
        stats.context_used = self.context_used
        self._renderer.stats(
            stats.elapsed,
            stats.prompt_tokens,
            stats.eval_tokens,
            self.context_used,
            self.context_window,
        )
        return stats

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def _run_inference(self, stats: TurnStats, suppress_response: bool = False) -> bool:
        """Run one ollama.chat cycle + tool dispatch. Returns True if tool_calls fired."""
        # Apply current mode BEFORE assembling the request so the sampling
        # profile, tool list, and any sticky addendum are up to date.
        options = self._apply_mode()

        stream = self._chat(
            model=self.model,
            messages=self.messages,
            tools=self._schemas(),
            think=True,
            stream=True,
            keep_alive="15m",
            options=options,
        )

        thinking_open = False
        response_open = False
        thinking_parts: list[str] = []
        content_parts: list[str] = []
        msg_tool_calls: list = []

        try:
            for chunk in stream:
                cmsg = chunk.message
                if getattr(cmsg, "thinking", None):
                    if not thinking_open:
                        self._renderer.thinking_start()
                        thinking_open = True
                    self._renderer.thinking_token(cmsg.thinking)
                    thinking_parts.append(cmsg.thinking)
                if getattr(cmsg, "content", None):
                    if thinking_open:
                        self._renderer.thinking_end()
                        thinking_open = False
                    if not suppress_response:
                        if not response_open:
                            self._renderer.response_start()
                            response_open = True
                        self._renderer.response_token(cmsg.content)
                    content_parts.append(cmsg.content)
                if getattr(cmsg, "tool_calls", None):
                    msg_tool_calls = cmsg.tool_calls
                if getattr(chunk, "done", False):
                    last_prompt = getattr(chunk, "prompt_eval_count", 0) or 0
                    stats.prompt_tokens += last_prompt
                    stats.eval_tokens += getattr(chunk, "eval_count", 0) or 0
                    self.context_used = last_prompt
        except KeyboardInterrupt:
            if thinking_open:
                self._renderer.thinking_end()
            if response_open and not suppress_response:
                self._renderer.response_end()
            self._renderer.orchestrator_event("interrupted", "")
            raise

        if thinking_open:
            self._renderer.thinking_end()
        if response_open and not suppress_response:
            self._renderer.response_end()

        full_content = "".join(content_parts)
        self._last_response = full_content

        self.messages.append(
            {
                "role": "assistant",
                "content": full_content,
                "tool_calls": msg_tool_calls or [],
            }
        )

        if not msg_tool_calls:
            return False

        for tc in msg_tool_calls:
            fn_name = tc.function.name
            fn_args = tc.function.arguments
            self._renderer.tool_call(fn_name, fn_args)

            result = self._call(fn_name, fn_args)
            result_str = str(result)
            self._renderer.tool_result(result_str)
            self.messages.append({"role": "tool", "content": result_str})

            self._post_tool_hook(fn_name, result_str)

        return True

    # ------------------------------------------------------------------
    # Post-tool hooks
    # ------------------------------------------------------------------

    def _post_tool_hook(self, fn_name: str, result_str: str) -> None:
        """React to specific tool results."""
        # Hook 1: plan_complete_step queues a verifier injection.
        if fn_name == "plan_complete_step":
            data = _safe_json(result_str)
            if data.get("completed") and data.get("verification_requested"):
                step_id = int(data.get("step_id", 0))
                if step_id:
                    self._pending_verifications.append(step_id)
                    self._renderer.orchestrator_event(
                        "verifier-queued", f"step #{step_id}"
                    )

        # Hook 2: session_save triggers context trim (preserve existing behavior).
        if fn_name == "session_save":
            data = _safe_json(result_str)
            if data.get("saved"):
                self._trim(self.messages)
                self.context_used = 0

    # ------------------------------------------------------------------
    # Injections
    # ------------------------------------------------------------------

    def _apply_mode(self) -> dict:
        """Ensure the current mode's addendum + toolset are live, return the
        sampling options to pass to ``ollama.chat``.

        Runs at the top of every ``_run_inference`` so model-initiated mode
        switches (via ``set_mode``) take effect on the very next cycle.
        """
        try:
            from tools import (
                activate as _activate,
                deactivate as _deactivate,
                is_registered as _is_registered,
            )
        except ImportError:
            _activate = _deactivate = _is_registered = None

        current = _modes.get_current_mode()

        if self._applied_mode != current:
            # Inject the addendum as a sticky [SYSTEM MODE] message so the
            # context_window._is_sticky check preserves it across trims.
            addendum = _modes.addendum_for(current)
            if addendum:
                self.messages.append({"role": "user", "content": addendum})

            # Deactivate previous mode's toolset
            if self._applied_mode is not None and _deactivate is not None:
                for t in _modes.toolset_for(self._applied_mode):
                    _deactivate(t)

            # Activate new mode's toolset
            if _activate is not None and _is_registered is not None:
                for t in _modes.toolset_for(current):
                    if _is_registered(t):
                        _activate(t)

            self._renderer.orchestrator_event(
                "mode",
                f"{self._applied_mode.value if self._applied_mode else 'init'} → {current.value}",
            )
            self._applied_mode = current

        return _modes.profile_for(current).to_ollama_options()

    def _refresh_files_message(self) -> None:
        """Maintain a single [SYSTEM FILES] message at the tail of self.messages
        that reflects the current open-files registry. Updated in-place each
        cycle so it never accumulates copies, and treated as sticky by
        context_window._is_sticky."""
        try:
            from tools.file_tools import _state as _fs
        except ImportError:
            return
        snap = _fs.open_files_snapshot()
        if not snap:
            return

        lines = ["[SYSTEM FILES] Open files (read or written this session):"]
        for e in snap[:20]:
            tag = "STALE" if e["stale"] else ("missing" if not e["exists"] else "ok")
            ago = e["last_read_ago_s"]
            lines.append(f"- {e['path']}  ({e['size_bytes']} B, {ago}s ago, {tag})")
        if len(snap) > 20:
            lines.append(f"  …and {len(snap) - 20} more.")
        lines.append(
            "Stale files MUST be re-read with read_file before edit_file/write_file."
        )
        text = "\n".join(lines)

        # Replace existing [SYSTEM FILES] msg or append a new one. Keep at most one.
        for i, m in enumerate(self.messages):
            if m.get("role") == "user" and (m.get("content") or "").startswith(
                "[SYSTEM FILES]"
            ):
                m["content"] = text
                return
        self.messages.append({"role": "user", "content": text})

    def _inject_verifier(self, step_id: int) -> None:
        text = _verifier.build_injection(step_id)
        if not text:
            return
        self.messages.append({"role": "user", "content": text})
        self._renderer.orchestrator_event("verifier", f"step #{step_id}")

    def _maybe_nudge_snapshot(self) -> None:
        """Inject a session_save reminder when context ≥70% AND a plan is active.

        Only fires once per turn. The rendered prompt asks the model to capture
        the plan's current step so session_recall can resume cleanly after the
        next compact/trim cycle.
        """
        if self._snapshot_nudged or not self.agentic:
            return
        if self.context_window <= 0 or self.context_used <= 0:
            return
        frac = self.context_used / self.context_window
        if frac < self.AUTO_SNAPSHOT_PCT:
            return
        active = _plan.load_active()
        if active is None or active.is_complete():
            return
        cur = active.current_step or 0
        text = prompts.snapshot_nudge(pct=int(frac * 100), current_step=cur)
        self.messages.append({"role": "user", "content": text})
        self._snapshot_nudged = True
        self._renderer.orchestrator_event(
            "snapshot-nudge", f"ctx {int(frac * 100)}% / step #{cur}"
        )

    def _inject_plan_nudge(self, active_plan: _plan.Plan) -> bool:
        """
        Push the model back to plan execution when it tried to finalize
        while steps remained. Cap so we don't loop forever if the model
        is stuck — after MAX_NUDGES we give up and let the critic run.
        """
        MAX_NUDGES = 2
        count = sum(
            1
            for m in self.messages
            if m.get("role") == "user"
            and "[SYSTEM ORCHESTRATOR]" in m.get("content", "")
        )
        if count >= MAX_NUDGES:
            return False
        cur = active_plan.current_step
        step = active_plan.get_step(cur) if cur else None
        status = step.status if step else "unknown"
        text = prompts.plan_nudge(current_step=cur or 0, status=status)
        self.messages.append({"role": "user", "content": text})
        self._renderer.orchestrator_event("plan-nudge", f"step #{cur}")
        return True

    # ------------------------------------------------------------------
    # Critic
    # ------------------------------------------------------------------

    def _should_critique(self) -> bool:
        """
        Critic only fires when the model has truly delivered a final reply
        to the user — i.e. the turn is about to switch back to the user.

        Skip if:
          - response is empty
          - response ends with a question mark (model is asking the user)
          - an active plan still has unverified/incomplete steps (model
            is still working, even if plan-nudges have been exhausted)
        """
        text = (self._last_response or "").strip()
        if not text:
            return False
        if text.endswith("?"):
            return False
        if self.agentic:
            active = _plan.load_active()
            if active is not None and not active.is_complete():
                return False
        return True

    def _run_critic_round(self) -> bool:
        """
        Run the critic on the drafted reply via an *isolated* chat call —
        the request goes to the same model (so no extra VRAM beyond the
        existing slot's KV cache), but uses a fresh, minimal message list
        instead of being appended to the running conversation. This
        prevents the critic exchange from polluting the model's chain of
        thought.

        If the verdict says ``accept``, return False (no loop continuation).
        If it says ``revise``, append a revise prompt to the *main*
        conversation so the next main-loop inference produces a revised
        reply, and return True.
        """
        self._critic_round += 1
        drafted = self._last_response

        iso_messages = [
            {"role": "system", "content": _critic.isolated_system_prompt()},
            {
                "role": "user",
                "content": _critic.build_isolated_injection(
                    user_question=self._find_last_user_question(),
                    drafted_response=drafted,
                    round_num=self._critic_round,
                    max_rounds=self.critic_max_rounds,
                ),
            },
        ]

        self._renderer.orchestrator_event(
            "critic",
            f"round {self._critic_round}/{self.critic_max_rounds} (isolated)",
        )

        verdict_text = self._isolated_inference(iso_messages)
        verdict = _critic.parse_verdict(verdict_text)

        if verdict.accept:
            self._renderer.orchestrator_event("critic", "accepted")
            return False

        if self._critic_round >= self.critic_max_rounds:
            self._renderer.orchestrator_event(
                "critic",
                f"max rounds reached; {len(verdict.issues)} issue(s) remain",
            )
            return False

        revise_text = _critic.build_revision(verdict.issues)
        self.messages.append({"role": "user", "content": revise_text})
        self._renderer.orchestrator_event(
            "critic-revise", f"{len(verdict.issues)} issue(s)"
        )
        return True

    def _isolated_inference(self, messages: list[dict]) -> str:
        """
        Run a one-off chat call against the same model with the given
        messages, returning only the visible content. No tools, no
        rendering, no mutation of ``self.messages``.

        We keep ``stream=True`` so the existing ``chat_fn`` interface
        (and its test fakes, which always yield iterators) works
        unchanged. Thinking tokens and tool-call deltas are discarded.
        """
        stream = self._chat(
            model=self.model,
            messages=messages,
            tools=None,
            think=False,
            stream=True,
            keep_alive="15m",
            options={"temperature": 0.0},
        )
        parts: list[str] = []
        try:
            for chunk in stream:
                cmsg = getattr(chunk, "message", None)
                if cmsg is None:
                    continue
                piece = getattr(cmsg, "content", None)
                if piece:
                    parts.append(piece)
        except KeyboardInterrupt:
            self._renderer.orchestrator_event("interrupted", "critic")
            raise
        return "".join(parts)

    def _find_last_user_question(self) -> str:
        """Return the most recent genuine user message (skipping orchestrator
        injections), with the ``<context_window>`` prefix stripped."""
        for m in reversed(self.messages):
            if m.get("role") != "user":
                continue
            content = m.get("content", "") or ""
            head = content.lstrip()[:32]
            if head.startswith("[SYSTEM "):
                continue
            if content.startswith("<context_window>"):
                end = content.find("</context_window>")
                if end != -1:
                    content = content[end + len("</context_window>") :]
            return content.strip()
        return ""


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _safe_json(text: str) -> dict:
    try:
        data = json.loads(text)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}
