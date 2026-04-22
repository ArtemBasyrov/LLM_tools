"""
Tests for agent.critic — verdict parsing and orchestrator integration.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from agent import critic
from agent.orchestrator import NullRenderer, Orchestrator


# ---------------------------------------------------------------------------
# parse_verdict
# ---------------------------------------------------------------------------


def test_parse_accept_true():
    v = critic.parse_verdict('Here is my review.\n{"accept": true, "issues": []}')
    assert v.accept is True
    assert v.issues == []
    assert v.parsed is True


def test_parse_accept_false_with_issues():
    v = critic.parse_verdict(
        'Reviewing...\n{"accept": false, "issues": ["missing verification", "claim X unsupported"]}'
    )
    assert v.accept is False
    assert v.issues == ["missing verification", "claim X unsupported"]
    assert v.parsed is True


def test_parse_picks_last_json_block():
    """If multiple JSON objects, the last one wins (the verdict)."""
    text = '{"accept": false, "issues": ["draft"]}\n\nFinal: {"accept": true, "issues": []}'
    v = critic.parse_verdict(text)
    assert v.accept is True


def test_parse_no_json_returns_reject_with_error():
    v = critic.parse_verdict("I think the response looks fine.")
    assert v.accept is False
    assert v.parsed is False
    assert any("valid JSON" in i for i in v.issues)


def test_parse_critic_accept_literal_fallback():
    v = critic.parse_verdict("CRITIC_ACCEPT — all good")
    assert v.accept is True
    assert v.parsed is False  # fell back to heuristic


def test_parse_non_list_issues_coerced():
    v = critic.parse_verdict('{"accept": false, "issues": "just one string"}')
    assert v.accept is False
    assert v.issues == ["just one string"]


def test_build_injection_includes_round_info():
    msg = critic.build_injection(round_num=1, max_rounds=2)
    assert "[SYSTEM CRITIC]" in msg
    assert "1" in msg and "2" in msg


def test_build_revision_includes_issues():
    msg = critic.build_revision(["foo", "bar"])
    assert "[SYSTEM CRITIC]" in msg
    assert "foo" in msg and "bar" in msg


# ---------------------------------------------------------------------------
# Orchestrator integration — critic loop
# ---------------------------------------------------------------------------


def _chunk(
    content="",
    thinking="",
    tool_calls=None,
    done=False,
    prompt_eval_count=0,
    eval_count=0,
):
    return SimpleNamespace(
        message=SimpleNamespace(
            thinking=thinking, content=content, tool_calls=tool_calls or []
        ),
        done=done,
        prompt_eval_count=prompt_eval_count,
        eval_count=eval_count,
    )


def _final(text, pt=50, et=5):
    return [_chunk(content=text, done=True, prompt_eval_count=pt, eval_count=et)]


class FakeChat:
    def __init__(self, scripts):
        self._scripts = list(scripts)
        self.calls = []

    def __call__(self, **kw):
        self.calls.append([dict(m) for m in kw["messages"]])
        if not self._scripts:
            raise AssertionError("FakeChat exhausted")
        return iter(self._scripts.pop(0))


class Capturing(NullRenderer):
    def __init__(self):
        self.events = []

    def orchestrator_event(self, kind, detail=""):
        self.events.append((kind, detail))


def _orch(messages, scripts, **extra):
    from tools import call as real_call

    chat = FakeChat(scripts)
    orch = Orchestrator(
        messages=messages,
        context_window=10_000,
        model="test",
        chat_fn=chat,
        tool_call_fn=real_call,
        tool_schemas_fn=lambda: [],
        compact_fn=lambda m, u, t: False,
        trim_fn=lambda m: None,
        context_prefix_fn=lambda u, t: "",
        **extra,
    )
    return orch, chat


@pytest.fixture
def agentic_critic_2(monkeypatch):
    monkeypatch.setenv("AGENTIC_MODE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "2")
    monkeypatch.setenv("ENFORCE_PLANNING", "false")  # isolate critic behavior


def test_critic_accepts_first_round(plan_dir, agentic_critic_2):
    """Model answers, critic immediately accepts → 2 inferences total."""
    messages = [{"role": "system", "content": "sys"}]
    renderer = Capturing()
    scripts = [
        _final("The answer is 42."),
        _final('{"accept": true, "issues": []}'),
    ]
    orch, chat = _orch(messages, scripts, renderer=renderer)
    orch.turn("what is the meaning of life?")
    assert len(chat.calls) == 2
    kinds = [e[0] for e in renderer.events]
    assert "critic" in kinds
    # Second 'critic' event with detail 'accepted' indicates acceptance
    accepts = [d for k, d in renderer.events if k == "critic" and "accept" in d]
    assert accepts


def test_critic_rejects_then_accepts(plan_dir, agentic_critic_2):
    """Round 1 rejects → revise → round 2 accepts."""
    messages = [{"role": "system", "content": "sys"}]
    renderer = Capturing()
    scripts = [
        _final("draft answer"),
        _final('{"accept": false, "issues": ["not specific enough"]}'),
        _final("revised, more specific answer with citations"),
        _final('{"accept": true, "issues": []}'),
    ]
    orch, chat = _orch(messages, scripts, renderer=renderer)
    stats = orch.turn("a question")
    assert len(chat.calls) == 4
    assert stats.critic_rounds >= 1
    kinds = [e[0] for e in renderer.events]
    assert "critic-revise" in kinds


def test_critic_exhausts_max_rounds(plan_dir, agentic_critic_2):
    """If critic never accepts, loop stops at CRITIC_MAX_ROUNDS."""
    messages = [{"role": "system", "content": "sys"}]
    # CRITIC_MAX_ROUNDS=2 → 2 rounds, then give up.
    # Each round consumes 2 inferences (response + verdict).
    # Plus the original response. So: 1 (original) + 2*2 (rounds) = 5 max.
    reject = _final('{"accept": false, "issues": ["still wrong"]}')
    scripts = [
        _final("draft 1"),
        reject,
        _final("draft 2"),
        reject,
        _final("draft 3"),  # defensive extra — should not be consumed
    ]
    orch, chat = _orch(messages, scripts)
    orch.turn("a question")
    # 1 original + 2 rounds × 2 inferences each = 5, but the flow is:
    # inf1=draft1, inf2=verdict1(reject), inf3=draft2(revise), inf4=verdict2(reject)
    # At round 2, critic still rejects; _critic_round (now 2) is NOT < max (2),
    # so we break without injecting another revise.
    assert len(chat.calls) == 4


def test_critic_disabled_with_zero_rounds(plan_dir, monkeypatch):
    """CRITIC_MAX_ROUNDS=0 skips critic entirely — 1 inference only."""
    monkeypatch.setenv("AGENTIC_MODE", "true")
    monkeypatch.setenv("CRITIC_MAX_ROUNDS", "0")
    monkeypatch.setenv("ENFORCE_PLANNING", "false")
    messages = [{"role": "system", "content": "sys"}]
    scripts = [_final("direct answer")]
    orch, chat = _orch(messages, scripts)
    orch.turn("a question")
    assert len(chat.calls) == 1
