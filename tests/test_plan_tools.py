"""Integration-ish tests for tools/plan_tools.py — JSON shape + disk side-effects."""

import json

from agent import plan as P
import tools.plan_tools as PT


def _loads(s: str) -> dict:
    return json.loads(s)


def test_plan_create_with_no_active_plan(plan_dir):
    out = _loads(PT.plan_create(goal="do the thing"))
    assert out["created"] is True
    assert out["plan_id"].startswith("plan_")
    assert P.active_path().exists()


def test_plan_create_blocks_when_active_exists(plan_dir):
    PT.plan_create(goal="first")
    out = _loads(PT.plan_create(goal="second"))
    assert out["error"] == "active_plan_exists"


def test_plan_create_with_replace_archives_old(plan_dir):
    first = _loads(PT.plan_create(goal="first"))
    old_id = first["plan_id"]
    second = _loads(PT.plan_create(goal="second", replace=True))
    assert second["created"] is True
    # Old plan should be archived
    archive = P.archive_dir() / f"{old_id}.json"
    assert archive.exists()


def test_plan_add_step_without_plan_returns_error(plan_dir):
    out = _loads(PT.plan_add_step(description="x", verification="y"))
    assert out["error"] == "no_active_plan"


def test_plan_add_step_succeeds(plan_dir):
    PT.plan_create(goal="g")
    out = _loads(PT.plan_add_step(description="s1", verification="v1"))
    assert out["added"] is True
    assert out["step_id"] == 1
    assert out["total_steps"] == 1


def test_plan_complete_step_flags_verification_requested(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    PT.plan_start_step(step_id=1)
    out = _loads(
        PT.plan_complete_step(step_id=1, evidence="ran the thing, output matched")
    )
    assert out["completed"] is True
    assert out["verification_requested"] is True
    # Plan on disk reflects completed-but-not-verified
    p = P.load_active()
    step = p.get_step(1)
    assert step.status == "completed"
    assert step.verified is False


def test_plan_status_when_no_plan(plan_dir):
    out = _loads(PT.plan_status())
    assert out["active"] is False


def test_plan_status_shape(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    out = _loads(PT.plan_status())
    assert out["active"] is True
    assert out["goal"] == "g"
    assert len(out["steps"]) == 1


def test_plan_log_appends_note(plan_dir):
    PT.plan_create(goal="g")
    PT.plan_add_step(description="s1", verification="v1")
    out = _loads(PT.plan_log(step_id=1, note="discovered a helper"))
    assert out["logged"] is True
    assert out["note_count"] == 1


def test_plan_abandon_archives(plan_dir):
    PT.plan_create(goal="g")
    out = _loads(PT.plan_abandon(reason="user cancelled"))
    assert out["abandoned"] is True
    assert not P.active_path().exists()
    # Archive file exists
    archive_files = list(P.archive_dir().glob("plan_*.json"))
    assert len(archive_files) == 1


def test_plan_tools_registered_as_always_on():
    """Sanity: all seven plan tools are registered and always-on."""
    from tools import _registry

    for name in [
        "plan_create",
        "plan_add_step",
        "plan_start_step",
        "plan_complete_step",
        "plan_status",
        "plan_log",
        "plan_abandon",
    ]:
        assert name in _registry, f"{name} not registered"
        assert _registry[name]["always_on"] is True, f"{name} should be always_on"
