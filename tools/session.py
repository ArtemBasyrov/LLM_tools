"""
In-session memory tools using a local Markdown file.

Preserves conversation context when the context window fills up:
  LLM saves a snapshot → Python trims old messages from the list →
  LLM calls session_recall() to reload the summary into the active window.

Note: model offloads (GPU/RAM cleared after idle) do NOT cause context loss —
the full messages list is resent to Ollama on every request. Session memory is
only needed when the messages list itself grows beyond the context window limit.

The session file is wiped at chat.py startup and managed entirely by the LLM.
File: ~/.llm_sessions/current.md  (override with LLM_SESSION_DIR env var)

Tools:
    session_save    — overwrite snapshot with a new Markdown summary
    session_recall  — read the current snapshot
    session_clear   — erase the snapshot (start fresh within the same run)
"""

import datetime
import json
import os
from pathlib import Path

from tools import register

_SESSION_DIR = Path(
    os.path.expanduser(os.environ.get("LLM_SESSION_DIR", "~/.llm_sessions"))
)
_SESSION_FILE = _SESSION_DIR / "current.md"


def _clear_session_file() -> None:
    """Called by chat.py at startup to wipe any leftover snapshot from a prior run."""
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.unlink(missing_ok=True)


@register(
    description=(
        "Save a session snapshot so context can survive a context-window trim. "
        "Overwrites any previous snapshot — include everything important. "
        "Call when <context_window> in the user message shows usage above ~75%, "
        "or when the current task is complex and must survive a context trim. "
        "Write in Markdown. Recommended sections: "
        "## Current Task, ## Key Facts & Decisions, ## Pending Work, ## Outcomes."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": (
                    "Markdown-formatted snapshot. Must be self-contained — "
                    "it will be the only context available after a trim. "
                    "Include: what the user is trying to do, decisions already made, "
                    "facts discovered, pending steps, and any failure/success outcomes."
                ),
            },
        },
        "required": ["content"],
    },
)
def session_save(content: str) -> str:
    ts = datetime.datetime.now().isoformat(timespec="seconds")
    _SESSION_DIR.mkdir(parents=True, exist_ok=True)
    _SESSION_FILE.write_text(f"<!-- saved: {ts} -->\n{content}", encoding="utf-8")
    return json.dumps({"saved": True, "timestamp": ts, "chars": len(content)})


@register(
    description=(
        "Read the current session snapshot to restore task context after a context-window trim. "
        "Call this immediately after session_save — the trim will have removed old messages "
        "and the snapshot is now the only record of prior work. "
        "Returns the full snapshot text, or a 'not_found' message if none exists yet."
    ),
)
def session_recall() -> str:
    if not _SESSION_FILE.exists():
        return json.dumps({"found": False, "message": "No session snapshot saved yet."})
    raw = _SESSION_FILE.read_text(encoding="utf-8")
    lines = raw.splitlines()
    if lines and lines[0].startswith("<!-- saved:"):
        ts = lines[0].replace("<!-- saved: ", "").replace(" -->", "").strip()
        body = "\n".join(lines[1:])
    else:
        ts = None
        body = raw
    return json.dumps({"found": True, "timestamp": ts, "content": body})


@register(
    description=(
        "Erase the current session snapshot. "
        "Use when starting a completely new task where prior context is no longer relevant."
    ),
)
def session_clear() -> str:
    if _SESSION_FILE.exists():
        _SESSION_FILE.unlink()
        return json.dumps({"cleared": True})
    return json.dumps({"cleared": False, "message": "No snapshot to clear."})
