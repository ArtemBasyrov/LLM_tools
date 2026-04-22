"""Shared pytest fixtures.

Every test that touches disk must use the ``plan_dir`` fixture so writes
land in a tmp dir — never in the user's real ``~/.llm_plans``.
"""

import os
import sys
from pathlib import Path

import pytest

# Ensure project root is importable when pytest runs from tests/
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


@pytest.fixture
def plan_dir(tmp_path, monkeypatch):
    """Redirect LLM_PLAN_DIR to a per-test tmp dir."""
    d = tmp_path / "plans"
    d.mkdir()
    monkeypatch.setenv("LLM_PLAN_DIR", str(d))
    return d
