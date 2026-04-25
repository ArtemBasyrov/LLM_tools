"""
format_file — run an autoformatter on a file in-place.

Picks the formatter based on extension. No-op (with a clear message) if the
binary is not on PATH. The on-disk content is replaced only on a clean run;
the prior content is captured for undo_last_edit.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from tools import register
from tools.file_tools import _state
from tools.file_tools._helpers import confirm, show_edit_diff


_FORMATTERS: list[tuple[tuple[str, ...], list[str]]] = [
    ((".py",), ["ruff", "format"]),
    ((".js", ".ts", ".tsx", ".jsx", ".css", ".html", ".md"), ["prettier", "--write"]),
    ((".rs",), ["rustfmt"]),
    ((".go",), ["gofmt", "-w"]),
]


def _pick(path: str) -> list[str] | None:
    ext = os.path.splitext(path)[1].lower()
    for exts, cmd in _FORMATTERS:
        if ext in exts and shutil.which(cmd[0]):
            return cmd
    return None


@register(
    description=(
        "Format a file in-place using the language's standard formatter "
        "(ruff for .py, prettier for JS/TS/CSS/HTML/MD, rustfmt for .rs, gofmt for .go). "
        "Returns the diff and a backup is captured for undo_last_edit. "
        "Examples: "
        "tidy a Python file → format_file(path='module.py'); "
        "tidy JS → format_file(path='app.js'); "
        "NOT for: files in a binary format or unknown extensions (no-op with explanation)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to format.",
            },
        },
        "required": ["path"],
    },
)
def format_file(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return json.dumps({"error": f"File not found: {path}"})

    cmd = _pick(path)
    if cmd is None:
        return json.dumps(
            {
                "error": f"No formatter available for {os.path.splitext(path)[1] or '(no extension)'} "
                "— check that ruff/prettier/rustfmt/gofmt is installed and on PATH."
            }
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            original = fh.read()
    except OSError as e:
        return json.dumps({"error": str(e)})

    try:
        p = subprocess.run(cmd + [path], capture_output=True, text=True, timeout=15)
    except FileNotFoundError:
        return json.dumps({"error": f"Formatter not found: {cmd[0]}"})
    except subprocess.TimeoutExpired:
        return json.dumps({"error": "Formatter timed out"})

    if p.returncode != 0:
        return json.dumps(
            {
                "error": f"Formatter exited {p.returncode}: {(p.stderr or p.stdout).strip()[:500]}"
            }
        )

    try:
        with open(path, "r", encoding="utf-8") as fh:
            updated = fh.read()
    except OSError as e:
        return json.dumps({"error": str(e)})

    if updated == original:
        return json.dumps({"success": True, "path": path, "changed": False})

    show_edit_diff(path, original, updated, label="format_file")
    if not confirm(f"Keep formatter changes to {path}?"):
        # Revert
        try:
            with open(path, "w", encoding="utf-8") as fh:
                fh.write(original)
        except OSError:
            pass
        return json.dumps({"error": "User reverted format_file — no changes made."})

    _state.push_backup(path, original, label="format_file")
    _state.record_read(path)
    return json.dumps({"success": True, "path": path, "changed": True, "tool": cmd[0]})
