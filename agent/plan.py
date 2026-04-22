"""
Plan/Step dataclasses and disk-backed persistence.

A single active plan lives at ``$LLM_PLAN_DIR/active.json`` (default
``~/.llm_plans/active.json``). Completed or abandoned plans are moved to
``$LLM_PLAN_DIR/archive/<id>.json``.

All functions return concrete values (``Plan``, ``None``, ``bool``) — the
tool layer in ``tools/plan_tools.py`` handles JSON serialization for the
model.

The module is intentionally framework-free so it can be unit-tested
without importing anything from ``tools`` or ``ollama``.
"""

from __future__ import annotations

import datetime
import json
import os
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

_STATUS_VALUES = {"pending", "in_progress", "completed", "failed", "skipped"}


def _plan_dir() -> Path:
    """Resolve the plan directory each call so tests can override via env var."""
    return Path(os.path.expanduser(os.environ.get("LLM_PLAN_DIR", "~/.llm_plans")))


def active_path() -> Path:
    return _plan_dir() / "active.json"


def archive_dir() -> Path:
    return _plan_dir() / "archive"


def _now() -> str:
    return datetime.datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class Step:
    id: int
    description: str
    verification: str
    status: str = "pending"
    evidence: Optional[str] = None
    verified: bool = False
    attempts: int = 0
    notes: list[str] = field(default_factory=list)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Step":
        return cls(
            id=int(d["id"]),
            description=str(d["description"]),
            verification=str(d.get("verification", "")),
            status=str(d.get("status", "pending")),
            evidence=d.get("evidence"),
            verified=bool(d.get("verified", False)),
            attempts=int(d.get("attempts", 0)),
            notes=list(d.get("notes", []) or []),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
        )


@dataclass
class Plan:
    id: str
    goal: str
    created_at: str
    updated_at: str
    steps: list[Step] = field(default_factory=list)
    current_step: Optional[int] = None
    outcome: Optional[str] = None

    # ---- mutation helpers ----

    def add_step(self, description: str, verification: str) -> Step:
        step = Step(
            id=self._next_step_id(),
            description=description,
            verification=verification,
        )
        self.steps.append(step)
        if self.current_step is None:
            self.current_step = step.id
        self.updated_at = _now()
        return step

    def get_step(self, step_id: int) -> Optional[Step]:
        for s in self.steps:
            if s.id == step_id:
                return s
        return None

    def start_step(self, step_id: int) -> Optional[Step]:
        step = self.get_step(step_id)
        if step is None:
            return None
        step.status = "in_progress"
        step.started_at = step.started_at or _now()
        step.attempts += 1
        self.current_step = step_id
        self.updated_at = _now()
        return step

    def complete_step(self, step_id: int, evidence: str) -> Optional[Step]:
        """Mark step as completed (pending verification). Does NOT set verified=True."""
        step = self.get_step(step_id)
        if step is None:
            return None
        step.status = "completed"
        step.evidence = evidence
        step.completed_at = _now()
        step.verified = False  # verifier must still confirm
        self.updated_at = _now()
        self._advance_current()
        return step

    def rollback_step(self, step_id: int, reason: str = "") -> Optional[Step]:
        """Return a step to in_progress (e.g. after failed verification)."""
        step = self.get_step(step_id)
        if step is None:
            return None
        step.status = "in_progress"
        step.verified = False
        step.completed_at = None
        if reason:
            step.notes.append(f"[rollback] {reason}")
        self.current_step = step_id
        self.updated_at = _now()
        return step

    def mark_verified(self, step_id: int, notes: str = "") -> Optional[Step]:
        step = self.get_step(step_id)
        if step is None:
            return None
        step.verified = True
        if notes:
            step.notes.append(f"[verified] {notes}")
        self.updated_at = _now()
        return step

    def log_note(self, step_id: int, note: str) -> Optional[Step]:
        step = self.get_step(step_id)
        if step is None:
            return None
        step.notes.append(note)
        self.updated_at = _now()
        return step

    def _advance_current(self) -> None:
        """Point current_step at the next incomplete step, or None if all done."""
        for s in self.steps:
            if s.status in ("pending", "in_progress") or (
                s.status == "completed" and not s.verified
            ):
                self.current_step = s.id
                return
        self.current_step = None

    def _next_step_id(self) -> int:
        return (max((s.id for s in self.steps), default=0)) + 1

    # ---- status queries ----

    def is_complete(self) -> bool:
        """True iff every step is completed AND verified."""
        return bool(self.steps) and all(
            s.status == "completed" and s.verified for s in self.steps
        )

    def pending_verification(self) -> list[Step]:
        return [s for s in self.steps if s.status == "completed" and not s.verified]

    # ---- serialization ----

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "goal": self.goal,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "current_step": self.current_step,
            "outcome": self.outcome,
            "steps": [s.to_dict() for s in self.steps],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "Plan":
        return cls(
            id=str(d["id"]),
            goal=str(d["goal"]),
            created_at=str(d.get("created_at", _now())),
            updated_at=str(d.get("updated_at", _now())),
            steps=[Step.from_dict(s) for s in d.get("steps", [])],
            current_step=(
                int(d["current_step"]) if d.get("current_step") is not None else None
            ),
            outcome=d.get("outcome"),
        )


# ---------------------------------------------------------------------------
# Constructors
# ---------------------------------------------------------------------------


def new_plan(goal: str) -> Plan:
    ts = _now()
    return Plan(
        id=f"plan_{uuid.uuid4().hex[:8]}",
        goal=goal,
        created_at=ts,
        updated_at=ts,
        steps=[],
        current_step=None,
        outcome=None,
    )


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def load_active() -> Optional[Plan]:
    path = active_path()
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return Plan.from_dict(data)
    except (json.JSONDecodeError, KeyError, ValueError):
        return None


def save_active(plan: Plan) -> None:
    path = active_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    plan.updated_at = _now()
    path.write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )


def clear_active() -> bool:
    path = active_path()
    if path.exists():
        path.unlink()
        return True
    return False


def archive(plan: Plan, outcome: str) -> Path:
    """Move the plan to the archive dir with the given outcome string."""
    plan.outcome = outcome
    plan.updated_at = _now()
    archive_dir().mkdir(parents=True, exist_ok=True)
    dest = archive_dir() / f"{plan.id}.json"
    dest.write_text(
        json.dumps(plan.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    clear_active()
    return dest


# ---------------------------------------------------------------------------
# Presentation helpers (shared between tools and system prompt)
# ---------------------------------------------------------------------------


def summary(plan: Plan) -> str:
    """Compact human/LLM-readable summary suitable for system-prompt injection."""
    total = len(plan.steps)
    done = sum(1 for s in plan.steps if s.status == "completed" and s.verified)
    cur = plan.get_step(plan.current_step) if plan.current_step else None
    lines = [
        f"Plan {plan.id}: {plan.goal}",
        f"Progress: {done}/{total} steps verified-complete.",
    ]
    if cur:
        lines.append(f"Current step #{cur.id} ({cur.status}): {cur.description}")
    else:
        lines.append("No current step (plan complete or empty).")
    return "\n".join(lines)


def detailed_summary(plan: Plan) -> dict:
    """Structured summary for tool output."""
    return {
        "id": plan.id,
        "goal": plan.goal,
        "current_step": plan.current_step,
        "is_complete": plan.is_complete(),
        "pending_verification": [s.id for s in plan.pending_verification()],
        "steps": [
            {
                "id": s.id,
                "description": s.description,
                "status": s.status,
                "verified": s.verified,
                "attempts": s.attempts,
            }
            for s in plan.steps
        ],
    }
