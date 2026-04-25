"""
Session-side state for file tools.

Tracks:
- Read ledger: the on-disk mtime captured at the last successful read of each
  file, so write_file/edit_file can refuse to act on stale views.
- Backup ring: the previous content of each file just before each successful
  edit/write, capped at _MAX_BACKUPS per path, used by undo_last_edit.
- Open files: the set of paths the model currently "knows about" (read or
  written this session), exposed to the orchestrator so it can refresh a
  sticky [SYSTEM FILES] message each turn.

All state is in-process. Cleared with reset_session_state(); main.py calls
this in the same place it clears the session snapshot.
"""

from __future__ import annotations

import os
import time
from collections import deque
from typing import Any

# Tolerance for filesystem mtime jitter (network drives, fsync timing).
_MTIME_TOLERANCE_S = 0.05

# Backups per path (ring). Plenty for typical multi-step refactors without
# exploding RAM on bulky files.
_MAX_BACKUPS = 10

# How big a backup file we are willing to retain in memory.
_MAX_BACKUP_BYTES = 2_000_000  # 2 MB


# ---------------------------------------------------------------------------
# Read ledger (path -> last-seen mtime)
# ---------------------------------------------------------------------------

_read_mtimes: dict[str, float] = {}
_read_times: dict[str, float] = {}  # wall-clock time of last read (for display)


def record_read(path: str) -> None:
    """Stamp the on-disk mtime as 'last seen' for staleness comparison."""
    try:
        _read_mtimes[path] = os.path.getmtime(path)
        _read_times[path] = time.time()
    except OSError:
        pass


def is_stale(path: str) -> tuple[bool, str | None]:
    """
    Returns (stale, reason).
    - (False, None): file is not in the ledger (never read this session) — caller
      decides whether to require a probe.
    - (False, "fresh"): file is in the ledger and on-disk mtime matches.
    - (True, "<reason>"): file changed (or vanished) since last read.
    """
    if path not in _read_mtimes:
        return False, None
    try:
        current = os.path.getmtime(path)
    except FileNotFoundError:
        return True, "file no longer exists on disk"
    except OSError as e:
        return True, f"could not stat file: {e}"

    if abs(current - _read_mtimes[path]) <= _MTIME_TOLERANCE_S:
        return False, "fresh"
    return True, (
        f"file changed on disk since last read "
        f"(mtime moved by {current - _read_mtimes[path]:+.2f}s)"
    )


def is_known(path: str) -> bool:
    """True if this path has been read this session."""
    return path in _read_mtimes


# ---------------------------------------------------------------------------
# Backup ring (for undo_last_edit)
# ---------------------------------------------------------------------------

_backups: dict[str, deque] = {}
_undo_log: deque = deque(maxlen=64)  # global order of edits for undo_last_edit()


def push_backup(path: str, original_content: str | None, label: str) -> None:
    """
    Save the pre-edit content of a file. ``original_content=None`` means the
    file did not exist before the operation (so undo restores deletion).
    """
    if original_content is not None and len(original_content) > _MAX_BACKUP_BYTES:
        # Refuse to keep huge backups; user can always git-revert.
        return
    ring = _backups.setdefault(path, deque(maxlen=_MAX_BACKUPS))
    entry = {
        "ts": time.time(),
        "label": label,
        "content": original_content,  # None -> file did not exist
    }
    ring.append(entry)
    _undo_log.append((path, len(ring) - 1))


def pop_last_backup() -> tuple[str, dict] | None:
    """Return (path, backup_entry) for the most recent edit across all files."""
    while _undo_log:
        path, _idx = _undo_log.pop()
        ring = _backups.get(path)
        if not ring:
            continue
        entry = ring.pop()
        return path, entry
    return None


def peek_backups() -> list[dict[str, Any]]:
    """List every retained backup, newest first, for status reporting."""
    out: list[dict[str, Any]] = []
    for path, ring in _backups.items():
        for entry in ring:
            out.append(
                {
                    "path": path,
                    "ts": entry["ts"],
                    "label": entry["label"],
                    "size": (
                        len(entry["content"]) if entry["content"] is not None else 0
                    ),
                    "was_new_file": entry["content"] is None,
                }
            )
    out.sort(key=lambda e: e["ts"], reverse=True)
    return out


# ---------------------------------------------------------------------------
# Open files (for [SYSTEM FILES] sticky message)
# ---------------------------------------------------------------------------


def open_files_snapshot() -> list[dict[str, Any]]:
    """
    Compact list of files the model has touched this session, with current
    staleness state, for the [SYSTEM FILES] message. Sorted by most-recent.
    """
    out: list[dict[str, Any]] = []
    for path, last_read in _read_times.items():
        stale, reason = is_stale(path)
        try:
            size = os.path.getsize(path)
            exists = True
        except OSError:
            size = 0
            exists = False
        out.append(
            {
                "path": path,
                "exists": exists,
                "size_bytes": size,
                "last_read_ago_s": int(time.time() - last_read),
                "stale": bool(stale),
                "stale_reason": reason if stale else None,
            }
        )
    out.sort(key=lambda e: e["last_read_ago_s"])
    return out


# ---------------------------------------------------------------------------
# Pinned slices (survive surgical_clear)
# ---------------------------------------------------------------------------

_pinned: dict[str, list[tuple[int, int]]] = {}  # path -> [(start_line, end_line), ...]


def pin_slice(path: str, start_line: int, end_line: int) -> None:
    ranges = _pinned.setdefault(path, [])
    ranges.append((start_line, end_line))


def unpin_all(path: str) -> int:
    return len(_pinned.pop(path, []))


def pinned_slices() -> dict[str, list[tuple[int, int]]]:
    return {p: list(rs) for p, rs in _pinned.items()}


# ---------------------------------------------------------------------------
# Lifecycle
# ---------------------------------------------------------------------------


def reset_session_state() -> None:
    _read_mtimes.clear()
    _read_times.clear()
    _backups.clear()
    _undo_log.clear()
    _pinned.clear()
