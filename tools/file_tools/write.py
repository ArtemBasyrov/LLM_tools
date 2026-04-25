"""
write_file — create or overwrite a file with given content.
write_json — validate, format, and write JSON to a file.

Both refuse to overwrite an existing file unless either the file has been read
this session (so the model is grounded) or overwrite=true is set explicitly.
On success they capture a backup for undo_last_edit and run a fast syntax
check so the model gets immediate feedback.
"""

import json
import os
import shutil
import subprocess

from tools import register
from tools.file_tools import _state
from tools.file_tools._helpers import confirm, show_write_diff
from tools.file_tools.check import _check_content


def _maybe_format(path: str, content: str) -> str:
    """Best-effort autoformat for known extensions when the formatter is on PATH."""
    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".py" and shutil.which("ruff"):
            r = subprocess.run(
                ["ruff", "format", "-"],
                input=content,
                capture_output=True,
                text=True,
                timeout=8,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
        if ext == ".json":
            return json.dumps(json.loads(content), indent=2, ensure_ascii=False) + "\n"
        if ext in (
            ".js",
            ".ts",
            ".tsx",
            ".jsx",
            ".css",
            ".html",
            ".md",
        ) and shutil.which("prettier"):
            r = subprocess.run(
                ["prettier", f"--stdin-filepath={path}"],
                input=content,
                capture_output=True,
                text=True,
                timeout=8,
            )
            if r.returncode == 0 and r.stdout:
                return r.stdout
    except Exception:
        pass
    return content


@register(
    description=(
        "Write text content to a file, creating it if it does not exist or "
        "overwriting it entirely if it does. "
        "REQUIRED: both 'path' and 'content' must be provided in one call. "
        "SAFETY: refuses to overwrite an existing file that has not been read this "
        "session (call read_file first), unless overwrite=true. "
        "Optional autoformat=true runs ruff/prettier on the content before writing "
        "(no-op if formatter unavailable). "
        "Returns a syntax check (py_compile/ruff for .py, JSON/YAML/TOML parsers) so "
        "you can iterate on errors immediately. "
        "Use edit_file instead for targeted changes to an existing file. "
        "Examples: "
        "create a new script → write_file(path='deploy.sh', content='#!/bin/bash\\n...'); "
        "complete rewrite (existing file already read) → write_file(path='config.yaml', content='...'); "
        "force overwrite → write_file(path='gen.py', content='...', overwrite=true); "
        "with autoformat → write_file(path='out.py', content='...', autoformat=true)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path. Parent dirs are created automatically.",
            },
            "content": {
                "type": "string",
                "description": "The full text to write. Must be provided.",
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow overwriting an existing file that has NOT been read this session. Default false (safer).",
            },
            "autoformat": {
                "type": "boolean",
                "description": "Run ruff format / prettier on the content before writing, when available. Default false.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(
    path: str,
    content: str,
    overwrite: bool = False,
    autoformat: bool = False,
) -> str:
    path = os.path.expanduser(path)
    existed = os.path.exists(path)

    # Probe-first guard: don't blindly overwrite a file the model never read.
    if existed and not overwrite and not _state.is_known(path):
        return json.dumps(
            {
                "error": (
                    f"Refusing to overwrite '{path}' — file has not been read this "
                    "session. Call read_file first, or pass overwrite=true if you "
                    "really mean to discard the existing content."
                )
            }
        )

    # Stale guard: if the model has read the file, mtime must still match.
    stale, reason = _state.is_stale(path)
    if stale and not overwrite:
        return json.dumps(
            {
                "error": (
                    f"Stale read: {reason}. Re-read with read_file before writing, "
                    "or pass overwrite=true to clobber."
                )
            }
        )

    if autoformat:
        content = _maybe_format(path, content)

    show_write_diff(path, content, label="write_file")

    if not confirm(f"Allow write_file → {path}?"):
        return json.dumps({"error": "User denied write_file — no changes made."})

    # Snapshot the prior content for undo_last_edit.
    if existed:
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                prior = fh.read()
        except OSError:
            prior = ""
        _state.push_backup(path, prior, label="write_file (overwrite)")
    else:
        _state.push_backup(path, None, label="write_file (new)")

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as e:
        return json.dumps({"error": str(e)})

    _state.record_read(path)

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return json.dumps(
        {
            "success": True,
            "path": path,
            "lines_written": lines,
            "created": not existed,
            "checks": _check_content(path, content),
        },
        ensure_ascii=False,
    )


@register(
    description=(
        "Validate and write a JSON value to a file with consistent formatting (2-space indent). "
        "Use instead of write_file when output must be well-formed JSON — rejects malformed "
        "input before touching the file. Same overwrite-safety rules as write_file."
        "Examples: "
        "write a config → write_json(path='config.json', content='{\"host\": \"localhost\"}'); "
        "write a list → write_json(path='items.json', content='[{\"id\": 1}]'); "
        "NOT for: non-JSON formats (use write_file)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the JSON file.",
            },
            "content": {
                "type": "string",
                "description": 'Valid JSON text. e.g. \'{"key": "value"}\'.',
            },
            "overwrite": {
                "type": "boolean",
                "description": "Allow overwriting an unread existing file. Default false.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_json(path: str, content: str, overwrite: bool = False) -> str:
    path = os.path.expanduser(path)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON content: {e}"})

    formatted = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    return write_file(path=path, content=formatted, overwrite=overwrite)
