"""
edit_file — replace occurrences of old_string with new_string in a file.

Supports:
- Single edit (default): old_string must appear exactly once.
- replace_all=true: rename-style replacements across the whole file.
- anchor_line=N: disambiguate when old_string appears multiple times by
  picking the occurrence whose first line is closest to line N.
- edits=[{old, new}, ...]: batch atomic edits applied in order; if any fails
  the whole batch is rolled back before any disk write.

After a successful edit the response includes ±5 lines of post-edit context
around the change and a syntax-check verdict (py_compile / json / yaml on
known extensions). The on-disk mtime is compared with the ledger captured at
the last read; a mismatch refuses the edit and tells the model to re-read.
"""

import json
import os

from tools import register
from tools.file_tools import _state
from tools.file_tools._helpers import (
    _MAX_BYTES,
    closest_lines,
    confirm,
    show_edit_diff,
    with_line_numbers,
)
from tools.file_tools.check import _check_content


def _post_context(updated: str, old: str, new: str, radius: int = 5) -> dict:
    """Return ±radius lines around the first occurrence of `new` in `updated`."""
    if new and new in updated:
        idx = updated.index(new)
    elif old and old in updated:  # rare — replacement was identical
        idx = updated.index(old)
    else:
        return {}
    line_start = updated.count("\n", 0, idx) + 1
    lines = updated.splitlines()
    lo = max(1, line_start - radius)
    hi = min(len(lines), line_start + radius + new.count("\n"))
    block = "\n".join(lines[lo - 1 : hi])
    return {
        "preview_lines": f"{lo}–{hi}",
        "preview": with_line_numbers(block + ("\n" if block else ""), lo),
    }


def _apply_one(
    original: str,
    old: str,
    new: str,
    *,
    replace_all: bool,
    anchor_line: int | None,
) -> tuple[str | None, str | None]:
    """
    Apply a single (old, new) edit to ``original`` and return ``(updated, error)``.
    ``error`` is None on success.
    """
    if not old:
        return None, "old_string must not be empty."
    count = original.count(old)
    if count == 0:
        misses = closest_lines(original, old, k=2)
        if misses:
            hint = "; ".join(
                f"line {m['line_number']} ({m['similarity']}): {m['text']!r}"
                for m in misses
            )
            return None, (
                f"old_string not found in file. Closest matches: {hint}. "
                "Re-read those lines and adjust whitespace/indentation."
            )
        return None, "old_string not found in file."

    if count > 1 and not replace_all and anchor_line is None:
        return None, (
            f"old_string appears {count} times; it must be unique. "
            "Add more surrounding context, or pass anchor_line=<N> to pick the "
            "occurrence nearest line N, or replace_all=true."
        )

    if replace_all:
        return original.replace(old, new), None

    if anchor_line is not None and count > 1:
        # Find the occurrence whose first line is closest to anchor_line.
        best_pos = None
        best_dist = None
        pos = 0
        while True:
            i = original.find(old, pos)
            if i == -1:
                break
            line_at = original.count("\n", 0, i) + 1
            d = abs(line_at - int(anchor_line))
            if best_dist is None or d < best_dist:
                best_dist = d
                best_pos = i
            pos = i + 1
        if best_pos is None:
            return None, "anchor lookup failed."
        return original[:best_pos] + new + original[best_pos + len(old) :], None

    return original.replace(old, new, 1), None


@register(
    description=(
        "Edit a file by replacing one or more exact occurrences of old_string with new_string. "
        "By default old_string must match the file content exactly (whitespace included) and "
        "appear exactly once. "
        "Disambiguation when it appears multiple times: pass anchor_line=<N> to pick the "
        "occurrence closest to line N, or replace_all=true to substitute every occurrence. "
        "Batch mode: pass edits=[{old, new}, ...] to apply many edits atomically — if any fails "
        "the whole batch is rolled back. "
        "Failure on missing old_string returns the closest existing lines so you can correct typos. "
        "After a successful edit the response includes a post-edit preview and a syntax check. "
        "Examples: "
        "(1) Rename a function once → edit_file(path='a.py', old_string='def foo(', new_string='def bar('); "
        "(2) Project-wide rename in one file → edit_file(path='a.py', old_string='self.cfg', new_string='self.config', replace_all=true); "
        "(3) Disambiguate by anchor → edit_file(path='a.py', old_string='return None', new_string='return 0', anchor_line=42); "
        "(4) Batch atomic edits → edit_file(path='a.py', edits=[{'old':'foo','new':'bar'},{'old':'baz','new':'qux'}]); "
        "(5) Delete a line → old_string='debug = True\\n', new_string='' "
        "(6) NOT for complete rewrites — use write_file."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": "Exact text to find. Required for single edits; ignore when using 'edits'.",
            },
            "new_string": {
                "type": "string",
                "description": "Replacement text. Required for single edits; pass '' to delete old_string.",
            },
            "replace_all": {
                "type": "boolean",
                "description": "If true, replace every occurrence instead of requiring uniqueness. Default false.",
            },
            "anchor_line": {
                "type": "integer",
                "description": "Disambiguator when old_string appears multiple times: pick the occurrence whose first line is closest to this line number.",
            },
            "edits": {
                "type": "array",
                "description": "Batch mode: list of {old, new} pairs applied in order. If any fails the file is left unchanged.",
                "items": {
                    "type": "object",
                    "properties": {
                        "old": {"type": "string"},
                        "new": {"type": "string"},
                        "replace_all": {"type": "boolean"},
                        "anchor_line": {"type": "integer"},
                    },
                    "required": ["old", "new"],
                },
            },
            "skip_stale_check": {
                "type": "boolean",
                "description": "Override the staleness guard (NOT recommended — bypasses external-change detection).",
            },
        },
        "required": ["path"],
    },
)
def edit_file(
    path: str,
    old_string: str | None = None,
    new_string: str | None = None,
    replace_all: bool = False,
    anchor_line: int | None = None,
    edits: list[dict] | None = None,
    skip_stale_check: bool = False,
    **_kwargs,
) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    # Stale-read guard
    if not skip_stale_check:
        stale, reason = _state.is_stale(path)
        if stale:
            return json.dumps(
                {
                    "error": f"Stale read: {reason}. Re-read with read_file before editing.",
                    "path": path,
                }
            )
        if not _state.is_known(path):
            return json.dumps(
                {
                    "error": (
                        "File has not been read this session — call read_file first so "
                        "edits are grounded in the current content. Pass skip_stale_check=true "
                        "to override (not recommended)."
                    ),
                    "path": path,
                }
            )

    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            original = fh.read(_MAX_BYTES)
    except UnicodeDecodeError:
        return json.dumps(
            {"error": "File appears to be binary and cannot be edited as text."}
        )
    except OSError as e:
        return json.dumps({"error": str(e)})

    # Build the unified work list
    if edits:
        work = list(edits)
    else:
        if old_string is None or new_string is None:
            return json.dumps(
                {"error": "Provide old_string + new_string, or edits=[...]"}
            )
        work = [
            {
                "old": old_string,
                "new": new_string,
                "replace_all": replace_all,
                "anchor_line": anchor_line,
            }
        ]

    updated = original
    applied_summaries: list[dict] = []
    for i, e in enumerate(work, start=1):
        old = e.get("old", "")
        new = e.get("new", "")
        ra = bool(e.get("replace_all", False))
        al = e.get("anchor_line")
        next_text, err = _apply_one(updated, old, new, replace_all=ra, anchor_line=al)
        if err:
            return json.dumps(
                {
                    "error": f"Edit {i}/{len(work)} failed — batch rolled back: {err}",
                    "path": path,
                    "edits_total": len(work),
                    "edits_failed_at": i,
                }
            )
        applied_summaries.append(
            {"index": i, "old_chars": len(old), "new_chars": len(new)}
        )
        updated = next_text

    if updated == original:
        return json.dumps(
            {
                "error": "No changes — old_string and new_string are equivalent.",
                "path": path,
            }
        )

    show_edit_diff(path, original, updated)

    if not confirm(f"Allow edit_file → {path}?"):
        return json.dumps({"error": "User denied edit_file — no changes made."})

    _state.push_backup(path, original, label=f"edit_file ({len(work)} edits)")

    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(updated)
    except OSError as e:
        return json.dumps({"error": str(e)})

    # Refresh staleness ledger so subsequent edits in the same turn don't trip the guard.
    _state.record_read(path)

    # Post-edit context (around the last applied edit for batches).
    last = work[-1]
    preview = _post_context(updated, last.get("old", ""), last.get("new", ""))

    checks = _check_content(path, updated)

    return json.dumps(
        {
            "success": True,
            "path": path,
            "edits_applied": len(work),
            "summaries": applied_summaries,
            "post_edit": preview,
            "checks": checks,
        },
        ensure_ascii=False,
    )
