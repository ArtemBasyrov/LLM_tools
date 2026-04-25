"""
apply_patch — apply a unified diff across one or more files.

Lets the model express a coherent multi-file change as a single tool call
(Aider/Cursor pattern), avoiding N round-trips of edit_file. Each hunk is
applied with the standard ``patch`` algorithm via Python's ``difflib`` /
``unidiff``-compatible parser.

Strategy: shell out to the system ``patch`` binary when available (most
reliable), with a Python fallback for environments without it. Stale-read
guard runs per file; backups are captured per file before writes.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile

from tools import register
from tools.file_tools import _state
from tools.file_tools._helpers import confirm, show_edit_diff
from tools.file_tools.check import _check_content


def _files_in_diff(diff: str) -> list[str]:
    paths: list[str] = []
    for line in diff.splitlines():
        if line.startswith("+++ "):
            tail = line[4:].split("\t", 1)[0].strip()
            if tail.startswith("b/"):
                tail = tail[2:]
            if tail and tail != "/dev/null" and tail not in paths:
                paths.append(tail)
    return paths


def _read(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return fh.read()
    except OSError:
        return None


@register(
    description=(
        "Apply a unified diff to one or more files in one atomic operation. "
        "Use when a coherent change spans multiple files (refactors, renames across files, "
        "coordinated config + code changes). Diff format: standard `diff -u` output with "
        "`--- a/path` and `+++ b/path` headers and @@ hunks. "
        "Per file: refuses if the file is stale (re-read first). On any per-file failure the "
        "whole patch is rolled back. Each modified file gets a syntax check on success. "
        "Examples: "
        "rename + import update across 2 files → apply_patch(diff='--- a/foo.py\\n+++ b/foo.py\\n@@ ...'); "
        "NOT for: single-file edits (use edit_file — simpler, smaller surface)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "diff": {
                "type": "string",
                "description": "Unified diff text (the output of `git diff` or `diff -u`). Must include --- a/<path> and +++ b/<path> headers.",
            },
        },
        "required": ["diff"],
    },
)
def apply_patch(diff: str) -> str:
    paths = _files_in_diff(diff)
    if not paths:
        return json.dumps(
            {"error": "No file paths found in diff (need '+++ b/<path>' headers)."}
        )

    # Stale check per file
    for p in paths:
        if os.path.isfile(p):
            stale, reason = _state.is_stale(p)
            if stale:
                return json.dumps(
                    {"error": f"Stale read on {p}: {reason}. Re-read first."}
                )

    # Snapshot pre-state for rollback
    pre_state: dict[str, str | None] = {p: _read(p) for p in paths}

    # Show preview using a dry-run if `patch` is available.
    show_diff_text = "(diff preview unavailable)"
    if shutil.which("patch"):
        try:
            with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as tmp:
                tmp.write(diff)
                tmp_path = tmp.name
            r = subprocess.run(
                ["patch", "-p1", "--dry-run", "-i", tmp_path],
                capture_output=True,
                text=True,
                timeout=10,
            )
            show_diff_text = (
                r.stdout + r.stderr
            ).strip() or "(dry-run produced no output)"
        except Exception:
            pass

    print("\n── apply_patch dry-run ──────────────────────────────")
    print(show_diff_text[:2000])
    print("─────────────────────────────────────────────────────\n")

    if not confirm(f"Allow apply_patch ({len(paths)} files)?"):
        return json.dumps({"error": "User denied apply_patch — no changes made."})

    # Use system patch when available (handles fuzzy matching, line numbers).
    if not shutil.which("patch"):
        return json.dumps(
            {
                "error": "System `patch` binary not available; install GNU patch or use edit_file."
            }
        )

    try:
        with tempfile.NamedTemporaryFile("w", suffix=".diff", delete=False) as tmp:
            tmp.write(diff)
            tmp_path = tmp.name
        r = subprocess.run(
            ["patch", "-p1", "-i", tmp_path],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except Exception as e:
        return json.dumps({"error": f"patch failed: {e}"})

    if r.returncode != 0:
        # Roll back any partial writes
        for p, prior in pre_state.items():
            if prior is None:
                if os.path.exists(p):
                    try:
                        os.remove(p)
                    except OSError:
                        pass
            else:
                try:
                    with open(p, "w", encoding="utf-8") as fh:
                        fh.write(prior)
                except OSError:
                    pass
        return json.dumps(
            {
                "error": f"patch exited {r.returncode}; rolled back. {(r.stdout + r.stderr).strip()[:500]}"
            }
        )

    # Capture backups + run syntax checks
    checks: dict[str, dict] = {}
    for p, prior in pre_state.items():
        _state.push_backup(p, prior, label="apply_patch")
        _state.record_read(p)
        if os.path.isfile(p):
            checks[p] = _check_content(p)

    return json.dumps(
        {
            "success": True,
            "files_changed": list(pre_state.keys()),
            "checks": checks,
        },
        ensure_ascii=False,
    )
