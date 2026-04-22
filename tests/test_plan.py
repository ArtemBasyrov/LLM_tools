"""Unit tests for agent.plan — dataclasses, state transitions, disk roundtrip."""

import json

from agent import plan as P


def test_new_plan_has_unique_id_and_empty_steps():
    a = P.new_plan("goal a")
    b = P.new_plan("goal b")
    assert a.id.startswith("plan_") and len(a.id) > 5
    assert a.id != b.id
    assert a.steps == []
    assert a.current_step is None
    assert not a.is_complete()


def test_add_step_auto_assigns_sequential_ids(plan_dir):
    p = P.new_plan("g")
    s1 = p.add_step("do one", "check one")
    s2 = p.add_step("do two", "check two")
    assert s1.id == 1
    assert s2.id == 2
    # First add should set current_step; later adds should not move it
    assert p.current_step == 1


def test_save_and_load_roundtrip(plan_dir):
    p = P.new_plan("roundtrip goal")
    p.add_step("step a", "verify a")
    p.add_step("step b", "verify b")
    p.start_step(1)
    p.complete_step(1, "evidence a")
    P.save_active(p)

    loaded = P.load_active()
    assert loaded is not None
    assert loaded.id == p.id
    assert loaded.goal == "roundtrip goal"
    assert len(loaded.steps) == 2
    step1 = loaded.get_step(1)
    assert step1 is not None
    assert step1.status == "completed"
    assert step1.evidence == "evidence a"
    assert step1.verified is False  # completion is NOT verification
    assert step1.attempts == 1


def test_load_active_returns_none_when_no_file(plan_dir):
    assert P.load_active() is None


def test_load_active_handles_corrupt_json(plan_dir):
    P.active_path().write_text("{ not valid json", encoding="utf-8")
    assert P.load_active() is None


def test_complete_then_verify_advances_current(plan_dir):
    p = P.new_plan("g")
    p.add_step("a", "va")
    p.add_step("b", "vb")
    p.start_step(1)
    p.complete_step(1, "ev")
    # current_step stays on 1 until verified — completion alone doesn't advance
    assert p.current_step == 1
    p.mark_verified(1)
    p._advance_current()
    assert p.current_step == 2


def test_rollback_after_failed_verification(plan_dir):
    p = P.new_plan("g")
    p.add_step("a", "va")
    p.start_step(1)
    p.complete_step(1, "bogus evidence")
    p.rollback_step(1, "verifier could not confirm file exists")
    step = p.get_step(1)
    assert step.status == "in_progress"
    assert step.verified is False
    assert step.completed_at is None
    assert any("rollback" in n for n in step.notes)


def test_is_complete_requires_verified(plan_dir):
    p = P.new_plan("g")
    p.add_step("a", "va")
    p.start_step(1)
    p.complete_step(1, "ev")
    assert not p.is_complete()  # completed but not verified
    p.mark_verified(1)
    assert p.is_complete()


def test_archive_moves_plan_and_clears_active(plan_dir):
    p = P.new_plan("g")
    p.add_step("a", "va")
    P.save_active(p)
    assert P.active_path().exists()

    dest = P.archive(p, outcome="test-archive")
    assert dest.exists()
    assert not P.active_path().exists()
    archived = json.loads(dest.read_text())
    assert archived["outcome"] == "test-archive"
    assert archived["id"] == p.id


def test_detailed_summary_shape(plan_dir):
    p = P.new_plan("g")
    p.add_step("a", "va")
    p.add_step("b", "vb")
    p.start_step(1)
    d = P.detailed_summary(p)
    assert d["goal"] == "g"
    assert d["current_step"] == 1
    assert len(d["steps"]) == 2
    assert d["steps"][0]["status"] == "in_progress"
    assert d["steps"][1]["status"] == "pending"


def test_log_note_persists(plan_dir):
    p = P.new_plan("g")
    p.add_step("a", "va")
    p.log_note(1, "hello note")
    P.save_active(p)
    loaded = P.load_active()
    assert loaded.get_step(1).notes == ["hello note"]
