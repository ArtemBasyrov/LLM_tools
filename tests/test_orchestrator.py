"""
Tests for agent/orchestrator.py — turn state machine with fake Ollama.

We build a scripted ``FakeChat`` that yields chunks mimicking Ollama's
streaming format, then assert on the resulting message history and
orchestrator events.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Callable

import pytest

from agent.orchestrator import NullRenderer, Orchestrator
from agent import plan as P
import tools.plan_tools as PT
import tools.verify_tools as VT


# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


def _chunk(
    thinking: str = "",
    content: str = "",
    tool_calls: list | None = None,
    done: bool = False,
    prompt_eval_count: int = 0,
    eval_count: int = 0,
):
    return SimpleNamespace(
        message=SimpleNamespace(
            thinking=thinking,
            content=content,
            tool_calls=tool_calls or [],
        ),
        done=done,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def _tool_call(name: str, arguments: dict):
    return SimpleNamespace(function=SimpleNamespace(name=name, arguments=arguments))


def _final_response(text: str = "ok", prompt_tokens: int = 100, eval_tokens: int = 20):
    """A single-inference response with no tool calls."""
    return [
        _chunk(
            content=text,
            done=True,
            prompt_eval_count=prompt_tokens,
            eval_count=eval_tokens,
        )
    ]


def _tool_response(tool_calls: list, prompt_tokens: int = 100, eval_tokens: int = 5):
    """A single-inference response that triggers tool calls."""
    return [
        _chunk(
            tool_calls=tool_calls,
            done=True,
            prompt_eval_count=prompt_tokens,
            eval_count=eval_tokens,
        )
    ]


class FakeChat:
    """Scripted replacement for ollama.chat. Each call consumes one script entry."""

    def __init__(self, scripts: list[list]):
        self._scripts = list(scripts)
        self.calls: list[list[dict]] = []  # snapshot of messages passed in

    def __call__(self, *, model, messages, tools, think, stream, keep_alive):
        # Snapshot messages at call time
        self.calls.append([dict(m) for m in messages])
        if not self._scripts:
            raise AssertionError("FakeChat ran out of scripted responses")
        return iter(self._scripts.pop(0))


class CapturingRenderer(NullRenderer):
    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self.stats_calls: list[tuple] = []

    def orchestrator_event(self, kind: str, detail: str = ""):
        self.events.append((kind, detail))

    def stats(self, *args):
        self.stats_calls.append(args)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def agentic_env(monkeypatch):
    """Baseline: agentic on, critic OFF (simpler assertions)."""
    monkeypatch.setenv("AGENTIC_MODE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "0")
    monkeypatch.setenv("ENFORCE_PLANNING", "true")


@pytest.fixture
def no_op_compact():
    def noop(messages, used, total):
        return False

    def noop_trim(messages):
        pass

    def noop_prefix(used, total):
        return ""

    return noop, noop_trim, noop_prefix


def _build(
    messages,
    scripts,
    renderer=None,
    tool_names_to_results=None,
    compact=None,
    trim=None,
    prefix=None,
):
    """Assemble an Orchestrator with injected fakes."""
    chat = FakeChat(scripts)

    def fake_schemas():
        return []

    # Real tool dispatcher backed by the registry (so plan/verify tools actually run)
    from tools import call as real_call

    return Orchestrator(
        messages=messages,
        context_window=10_000,
        model="test",
        chat_fn=chat,
        tool_call_fn=real_call,
        tool_schemas_fn=fake_schemas,
        renderer=renderer or NullRenderer(),
        compact_fn=compact or (lambda m, u, t: False),
        trim_fn=trim or (lambda m: None),
        context_prefix_fn=prefix or (lambda u, t: ""),
    ), chat


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_simple_turn_with_no_plan_or_tools(plan_dir, agentic_env):
    """Simple input → one inference → no plan, no tool, critic off → done."""
    messages = [{"role": "system", "content": "sys"}]
    orch, chat = _build(
        messages, scripts=[_final_response("42", prompt_tokens=50, eval_tokens=2)]
    )
    stats = orch.turn("what is 6 times 7?")
    assert stats.prompt_tokens == 50
    assert stats.eval_tokens == 2
    assert len(chat.calls) == 1
    # User msg + assistant msg appended
    assert messages[-2]["role"] == "user"
    assert messages[-1]["role"] == "assistant"
    assert "42" in messages[-1]["content"]


def test_triage_injects_hint_for_complex_request(plan_dir, agentic_env):
    messages = [{"role": "system", "content": "sys"}]
    renderer = CapturingRenderer()
    orch, _ = _build(
        messages,
        scripts=[_final_response("ok", 50, 2)],
        renderer=renderer,
    )
    orch.turn("build me a word counter CLI in tools/wordcount.py with tests")
    triage_events = [e for e in renderer.events if e[0] == "triage"]
    assert triage_events, f"expected triage event, got {renderer.events}"
    # The triage hint message should be appended to messages
    triage_msgs = [m for m in messages if "[SYSTEM TRIAGE]" in m.get("content", "")]
    assert len(triage_msgs) == 1


def test_no_triage_when_plan_already_active(plan_dir, agentic_env):
    """If a plan is active (resumed), triage should not re-fire."""
    PT.plan_create(goal="ongoing work")
    PT.plan_add_step(description="s1", verification="v1")
    messages = [{"role": "system", "content": "sys"}]
    renderer = CapturingRenderer()
    # Plan is active → nudge fires because plan incomplete. Disable enforcement
    # for this test to isolate triage behavior.
    import os

    os.environ["ENFORCE_PLANNING"] = "false"
    orch, _ = _build(messages, [_final_response("ok", 50, 2)], renderer=renderer)
    orch.turn("build me a CLI thing and tests")
    assert not any(e[0] == "triage" for e in renderer.events)


def test_plan_complete_step_triggers_verifier_injection(plan_dir, agentic_env):
    """When the model calls plan_complete_step, the orchestrator must
    inject a [SYSTEM VERIFIER] on the NEXT inference."""
    # Preseed an active plan with a pending step so the tool call succeeds.
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    PT.plan_start_step(step_id=1)

    messages = [{"role": "system", "content": "sys"}]
    renderer = CapturingRenderer()

    # Script:
    #   Inference 1: model calls plan_complete_step(step_id=1, evidence=...)
    #   Inference 2: model calls verify_report(step_id=1, verified=true, notes=...)
    #   Inference 3: model emits final response "all done"
    scripts = [
        _tool_response(
            [
                _tool_call(
                    "plan_complete_step", {"step_id": 1, "evidence": "looked solid"}
                )
            ]
        ),
        _tool_response(
            [
                _tool_call(
                    "verify_report",
                    {"step_id": 1, "verified": True, "notes": "checked"},
                )
            ]
        ),
        _final_response("all done", 80, 5),
    ]
    orch, chat = _build(messages, scripts, renderer=renderer)
    orch.turn("complete the step")
    assert len(chat.calls) == 3, f"expected 3 inferences, got {len(chat.calls)}"

    # The 2nd call's message list must contain the [SYSTEM VERIFIER] prompt
    # as a user-role message (tool results also contain the literal string,
    # so we filter by role).
    second_call_msgs = chat.calls[1]
    verifier_msgs = [
        m
        for m in second_call_msgs
        if m.get("role") == "user"
        and m.get("content", "").startswith("[SYSTEM VERIFIER]")
    ]
    assert len(verifier_msgs) == 1
    assert "Step #1" in verifier_msgs[0]["content"]

    # Orchestrator events include verifier-queued and verifier
    kinds = [e[0] for e in renderer.events]
    assert "verifier-queued" in kinds
    assert "verifier" in kinds

    # Plan is verified-complete
    p = P.load_active()
    assert p.is_complete()


def test_verify_report_false_keeps_step_in_progress(plan_dir, agentic_env):
    """If the verifier rejects, the step rolls back and the plan stays active."""
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    PT.plan_start_step(step_id=1)

    # Turn off nudges and critic so we don't loop infinitely — we just want to
    # observe that rollback worked.
    import os

    os.environ["ENFORCE_PLANNING"] = "false"

    messages = [{"role": "system", "content": "sys"}]
    scripts = [
        _tool_response(
            [_tool_call("plan_complete_step", {"step_id": 1, "evidence": "bogus"})]
        ),
        _tool_response(
            [
                _tool_call(
                    "verify_report", {"step_id": 1, "verified": False, "notes": "no"}
                )
            ]
        ),
        _final_response("acknowledged, will retry", 80, 5),
    ]
    orch, _ = _build(messages, scripts)
    orch.turn("complete step 1")
    p = P.load_active()
    step = p.get_step(1)
    assert step.status == "in_progress"
    assert step.verified is False


def test_plan_nudge_fires_when_model_tries_to_finalize_too_early(plan_dir, agentic_env):
    """With a pending plan and agentic ON, emitting a final response
    before all steps verified triggers a nudge and another inference."""
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")

    messages = [{"role": "system", "content": "sys"}]
    renderer = CapturingRenderer()

    # Model tries to finalize on turn 1, gets nudged, then "completes" on turn 2
    # by calling plan_start_step + plan_complete_step, verifies, then responds.
    scripts = [
        _final_response("I'm done!", 50, 3),
        _tool_response(
            [
                _tool_call("plan_start_step", {"step_id": 1}),
                _tool_call("plan_complete_step", {"step_id": 1, "evidence": "did it"}),
            ]
        ),
        _tool_response(
            [
                _tool_call(
                    "verify_report", {"step_id": 1, "verified": True, "notes": "ok"}
                )
            ]
        ),
        _final_response("now truly done", 80, 5),
    ]
    orch, _ = _build(messages, scripts, renderer=renderer)
    orch.turn("do the step")
    assert any(e[0] == "plan-nudge" for e in renderer.events)


def test_snapshot_nudge_fires_when_context_high_and_plan_active(plan_dir, monkeypatch):
    """With an active plan and high context use, orchestrator injects a
    `[SYSTEM ORCHESTRATOR]` snapshot nudge before the first inference."""
    monkeypatch.setenv("AGENTIC_MODE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "0")
    monkeypatch.setenv("ENFORCE_PLANNING", "false")
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")

    messages = [{"role": "system", "content": "sys"}]
    renderer = CapturingRenderer()
    scripts = [
        _tool_response(
            [_tool_call("plan_start_step", {"step_id": 1})],
            prompt_tokens=7_500,
            eval_tokens=3,
        ),
        _final_response("done", 7_800, 3),
    ]
    orch, _ = _build(messages, scripts, renderer=renderer)
    orch.turn("work on the step")
    assert any(e[0] == "snapshot-nudge" for e in renderer.events)
    assert any(
        m.get("role") == "user"
        and "[SYSTEM ORCHESTRATOR]" in m.get("content", "")
        and "session_save" in m.get("content", "")
        for m in messages
    )


def test_snapshot_nudge_fires_at_most_once_per_turn(plan_dir, monkeypatch):
    """Even if every iteration is over the threshold, the nudge fires once."""
    monkeypatch.setenv("AGENTIC_MODE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "0")
    monkeypatch.setenv("ENFORCE_PLANNING", "false")
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")

    messages = [{"role": "system", "content": "sys"}]
    renderer = CapturingRenderer()
    scripts = [
        _tool_response(
            [_tool_call("plan_start_step", {"step_id": 1})],
            prompt_tokens=8_000,
            eval_tokens=2,
        ),
        _tool_response(
            [_tool_call("plan_log", {"step_id": 1, "note": "hm"})],
            prompt_tokens=8_200,
            eval_tokens=2,
        ),
        _final_response("done", 8_500, 2),
    ]
    orch, _ = _build(messages, scripts, renderer=renderer)
    orch.turn("work")
    nudge_events = [e for e in renderer.events if e[0] == "snapshot-nudge"]
    assert len(nudge_events) == 1


def test_max_nudges_bounded(plan_dir, agentic_env):
    """If the model never recovers, the orchestrator gives up nudging
    after MAX_NUDGES so we don't loop forever."""
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")

    messages = [{"role": "system", "content": "sys"}]
    # Model keeps emitting final responses — we cap at a finite script.
    scripts = [_final_response("still no plan work", 50, 3) for _ in range(5)]
    orch, chat = _build(messages, scripts)
    orch.turn("do the step")
    # Must not have consumed all 5 — nudges cap at 2 so turn ends after 3 infs.
    assert len(chat.calls) <= 3
