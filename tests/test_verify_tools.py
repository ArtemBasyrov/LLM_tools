"""Tests for tools/verify_tools.py and agent/verifier.py."""

import json

from agent import plan as P
from agent import verifier as V
import tools.plan_tools as PT
import tools.verify_tools as VT


def _loads(s: str) -> dict:
    return json.loads(s)


def _setup_plan_with_completed_step(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="file exists at X")
    PT.plan_start_step(step_id=1)
    PT.plan_complete_step(step_id=1, evidence="wrote the file")


def test_verify_report_no_plan(plan_dir):
    out = _loads(VT.verify_report(step_id=1, verified=True, notes=""))
    assert out["error"] == "no_active_plan"


def test_verify_report_unknown_step(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    out = _loads(VT.verify_report(step_id=99, verified=True, notes=""))
    assert out["error"] == "unknown_step"


def test_verify_report_true_marks_verified(plan_dir):
    _setup_plan_with_completed_step(plan_dir)
    out = _loads(
        VT.verify_report(step_id=1, verified=True, notes="checked file exists")
    )
    assert out["verified"] is True
    assert out["plan_complete"] is True  # only one step
    p = P.load_active()
    assert p.get_step(1).verified is True
    assert p.is_complete()


def test_verify_report_false_rolls_back(plan_dir):
    _setup_plan_with_completed_step(plan_dir)
    out = _loads(
        VT.verify_report(
            step_id=1, verified=False, notes="file does not exist at the claimed path"
        )
    )
    assert out["verified"] is False
    assert out["rolled_back"] is True
    p = P.load_active()
    step = p.get_step(1)
    assert step.status == "in_progress"
    assert step.verified is False
    assert step.completed_at is None


def test_verify_report_advances_current_step(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    PT.plan_add_step(description="s2", verification="v2")
    PT.plan_start_step(step_id=1)
    PT.plan_complete_step(step_id=1, evidence="ev1")
    out = _loads(VT.verify_report(step_id=1, verified=True, notes="confirmed"))
    assert out["verified"] is True
    assert out["plan_complete"] is False
    assert out["next_step"] == 2


def test_verifier_injection_contains_step_details(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="Create tools/foo.py", verification="file exists")
    PT.plan_start_step(step_id=1)
    PT.plan_complete_step(step_id=1, evidence="wrote the file, 50 lines")
    msg = V.build_injection(step_id=1)
    assert msg is not None
    assert "[SYSTEM VERIFIER]" in msg
    assert "Create tools/foo.py" in msg
    assert "file exists" in msg
    assert "wrote the file, 50 lines" in msg


def test_verifier_injection_returns_none_for_missing(plan_dir):
    assert V.build_injection(step_id=42) is None
