"""
git_status — porcelain-style summary of working tree changes.
git_diff   — diff between two refs (or working tree vs HEAD by default).
git_blame  — author/sha for a specific line of a file.

These are convenience wrappers around `git`. The model is far more likely to
use named git intelligence tools than to remember to call bash for the same
purpose, and typed JSON output is easier for it to reason about.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from tools import register


_TIMEOUT = 10
_MAX_DIFF_CHARS = 12_000


def _git(
    *args: str, cwd: str | None = None, timeout: int = _TIMEOUT
) -> tuple[int, str, str]:
    if not shutil.which("git"):
        return 127, "", "git not on PATH"
    try:
        p = subprocess.run(
            ["git", *args],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cwd or None,
        )
        return p.returncode, p.stdout, p.stderr
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"
    except Exception as e:
        return 1, "", str(e)


@register(
    description=(
        "Return the current git working-tree state: branch, ahead/behind counts, "
        "and per-file porcelain status (modified/added/deleted/untracked). "
        "Use before committing or to understand what's changed since you started. "
        "Examples: "
        "see what's pending → git_status(); "
        "from a subdir → git_status(cwd='backend/')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "cwd": {
                "type": "string",
                "description": "Directory to run from (default: current).",
            },
        },
        "required": [],
    },
)
def git_status(cwd: str | None = None) -> str:
    rc, so, se = _git("status", "--branch", "--porcelain=v2", cwd=cwd)
    if rc != 0:
        return json.dumps({"error": (se or so).strip() or f"git exit {rc}"})

    branch = None
    upstream = None
    ahead = behind = 0
    files: list[dict] = []
    for line in so.splitlines():
        if line.startswith("# branch.head "):
            branch = line.split(" ", 2)[2]
        elif line.startswith("# branch.upstream "):
            upstream = line.split(" ", 2)[2]
        elif line.startswith("# branch.ab "):
            parts = line.split()
            for p in parts:
                if p.startswith("+"):
                    ahead = int(p[1:])
                elif p.startswith("-"):
                    behind = int(p[1:])
        elif line.startswith("1 ") or line.startswith("2 "):
            tokens = line.split(" ", 8)
            files.append({"status": tokens[1], "path": tokens[-1]})
        elif line.startswith("? "):
            files.append({"status": "??", "path": line[2:]})

    return json.dumps(
        {
            "branch": branch,
            "upstream": upstream,
            "ahead": ahead,
            "behind": behind,
            "files": files,
        },
        ensure_ascii=False,
    )


@register(
    description=(
        "Return a unified diff. Defaults to working tree vs HEAD. "
        "Pass ref_a/ref_b to compare two commits, or path to scope to a single file. "
        "Output is capped at ~12K chars; refine with path or refs if truncated. "
        "Examples: "
        "what have I changed? → git_diff(); "
        "diff one file → git_diff(path='main.py'); "
        "compare branches → git_diff(ref_a='main', ref_b='HEAD'); "
        "staged only → git_diff(staged=true)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Limit diff to this path."},
            "ref_a": {
                "type": "string",
                "description": "First ref/commit (default: HEAD).",
            },
            "ref_b": {
                "type": "string",
                "description": "Second ref/commit (default: working tree).",
            },
            "staged": {"type": "boolean", "description": "Show only staged changes."},
            "cwd": {"type": "string", "description": "Directory to run from."},
        },
        "required": [],
    },
)
def git_diff(
    path: str | None = None,
    ref_a: str | None = None,
    ref_b: str | None = None,
    staged: bool = False,
    cwd: str | None = None,
) -> str:
    args = ["diff", "--no-color"]
    if staged:
        args.append("--cached")
    if ref_a and ref_b:
        args.extend([ref_a, ref_b])
    elif ref_a:
        args.append(ref_a)
    if path:
        args.extend(["--", os.path.expanduser(path)])

    rc, so, se = _git(*args, cwd=cwd)
    if rc != 0:
        return json.dumps({"error": (se or so).strip() or f"git exit {rc}"})

    truncated = False
    if len(so) > _MAX_DIFF_CHARS:
        so = so[:_MAX_DIFF_CHARS]
        truncated = True
    return json.dumps(
        {"diff": so, "truncated": truncated, "command": "git " + " ".join(args)},
        ensure_ascii=False,
    )


@register(
    description=(
        "Return git blame for a range of lines in a file: author, sha, date, summary "
        "for each line. Useful for 'why is this here?' questions. "
        "Examples: "
        "who wrote this fn → git_blame(path='main.py', start_line=42, end_line=50); "
        "single line → git_blame(path='main.py', start_line=42)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to the file."},
            "start_line": {
                "type": "integer",
                "description": "First line to blame (1-indexed).",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to blame (default: same as start_line).",
            },
            "cwd": {"type": "string", "description": "Directory to run from."},
        },
        "required": ["path", "start_line"],
    },
)
def git_blame(
    path: str,
    start_line: int,
    end_line: int | None = None,
    cwd: str | None = None,
) -> str:
    s = int(start_line)
    e = int(end_line) if end_line else s
    rc, so, se = _git(
        "blame",
        "--porcelain",
        "-L",
        f"{s},{e}",
        os.path.expanduser(path),
        cwd=cwd,
    )
    if rc != 0:
        return json.dumps({"error": (se or so).strip() or f"git exit {rc}"})

    blocks: list[dict] = []
    current: dict = {}
    for line in so.splitlines():
        if not line:
            continue
        if line.startswith("\t"):
            if current:
                current["code"] = line[1:]
                blocks.append(current)
                current = {}
        else:
            parts = line.split(" ", 1)
            tag = parts[0]
            val = parts[1] if len(parts) > 1 else ""
            if len(tag) == 40 and all(c in "0123456789abcdef" for c in tag):
                # New commit chunk
                current = {"sha": tag[:12]}
            elif tag == "author":
                current["author"] = val
            elif tag == "author-time":
                current["author_time"] = int(val)
            elif tag == "summary":
                current["summary"] = val

    return json.dumps(
        {"path": path, "range": f"{s}–{e}", "lines": blocks},
        ensure_ascii=False,
    )
