"""
undo_last_edit — restore the file content captured by the last successful
write_file/edit_file/format_file/apply_patch operation.

Backups live in the in-process session ledger (see _state.py); a 10-deep ring
per path and a 64-deep global order. ``undo_last_edit`` pops the most recent
entry from the global ring and restores the file to that snapshot. Repeated
calls walk further back in time.

Edge case: if the most recent backup represents a "new file" (content=None),
undo deletes the file rather than restoring an empty one.
"""

from __future__ import annotations

import json
import os

from tools import register
from tools.file_tools import _state
from tools.file_tools._helpers import confirm, show_write_diff


@register(
    description=(
        "Undo the most recent file change made by write_file / edit_file / "
        "format_file / apply_patch in this session. Walks back through a per-path "
        "ring of backups (max 10 per file) ordered by recency across the whole session. "
        "If the change was the creation of a new file, undo deletes the file. "
        "Examples: "
        "revert the last bad edit → undo_last_edit(); "
        "undo a series → call undo_last_edit() repeatedly; "
        "list available backups first → undo_list()."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def undo_last_edit() -> str:
    snap = _state.pop_last_backup()
    if snap is None:
        return json.dumps({"error": "No backups available in this session."})
    path, entry = snap
    prior = entry["content"]

    if prior is None:
        # File was newly created — undo means delete.
        if not os.path.exists(path):
            return json.dumps(
                {"success": True, "path": path, "action": "already_absent"}
            )
        if not confirm(f"Allow undo (delete) → {path}?"):
            return json.dumps({"error": "User denied undo — no changes made."})
        try:
            os.remove(path)
        except OSError as e:
            return json.dumps({"error": str(e)})
        return json.dumps(
            {
                "success": True,
                "path": path,
                "action": "deleted",
                "label": entry["label"],
            }
        )

    # File existed; restore it.
    show_write_diff(path, prior, label="undo_last_edit")
    if not confirm(f"Allow undo → {path}?"):
        return json.dumps({"error": "User denied undo — no changes made."})
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(prior)
    except OSError as e:
        return json.dumps({"error": str(e)})
    _state.record_read(path)
    return json.dumps(
        {
            "success": True,
            "path": path,
            "action": "restored",
            "restored_chars": len(prior),
            "label": entry["label"],
        },
        ensure_ascii=False,
    )


@register(
    description=(
        "List in-session edit backups (most recent first) so you can decide whether to undo. "
        "Each entry shows path, the operation that produced it, size, and whether it was a new file."
    ),
    parameters={"type": "object", "properties": {}, "required": []},
)
def undo_list() -> str:
    return json.dumps({"backups": _state.peek_backups()}, ensure_ascii=False)
