"""
find_files — typed glob/find with a hard result cap.

Safer than `bash find /` — won't burn tokens on huge result sets, won't follow
symlinks by default, and ignores common heavy directories (node_modules,
.venv, .git, dist, build, __pycache__) unless include_hidden=true.
"""

from __future__ import annotations

import fnmatch
import json
import os

from tools import register


_DEFAULT_IGNORE = {
    ".git",
    ".hg",
    ".svn",
    "node_modules",
    ".venv",
    "venv",
    "env",
    "__pycache__",
    ".pytest_cache",
    ".mypy_cache",
    ".ruff_cache",
    "dist",
    "build",
    ".next",
    ".turbo",
    "target",
}
_MAX_RESULTS = 500


@register(
    description=(
        "Find files by glob pattern under a root directory. Returns a structured list of paths "
        "with size and mtime. Ignores common heavy directories (node_modules, .venv, .git, dist, "
        "build, caches) unless include_hidden=true. Hard cap of 500 matches; tighten the pattern "
        "or root if you hit it. "
        "Examples: "
        "all Python files in the project → find_files(pattern='*.py'); "
        "tests under a subtree → find_files(root='tests/', pattern='test_*.py'); "
        "config files → find_files(pattern='*.yaml'); "
        "NOT for: searching file contents (use search_file or grep via bash)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern matched against the basename. e.g. '*.py' or 'test_*.py'.",
            },
            "root": {
                "type": "string",
                "description": "Directory to search under (default: current working dir).",
            },
            "include_hidden": {
                "type": "boolean",
                "description": "Descend into common heavy/hidden directories (default false).",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum entries to return (default 200, hard cap 500).",
            },
        },
        "required": ["pattern"],
    },
)
def find_files(
    pattern: str,
    root: str = ".",
    include_hidden: bool = False,
    max_results: int = 200,
) -> str:
    root = os.path.expanduser(root)
    if not os.path.isdir(root):
        return json.dumps({"error": f"Not a directory: {root}"})
    cap = max(1, min(int(max_results), _MAX_RESULTS))

    matches: list[dict] = []
    truncated = False

    for cwd, dirs, files in os.walk(root, followlinks=False):
        if not include_hidden:
            dirs[:] = [
                d for d in dirs if d not in _DEFAULT_IGNORE and not d.startswith(".")
            ]
        for f in files:
            if not include_hidden and f.startswith("."):
                continue
            if fnmatch.fnmatch(f, pattern):
                p = os.path.join(cwd, f)
                try:
                    st = os.stat(p)
                    matches.append(
                        {"path": p, "size_bytes": st.st_size, "mtime": int(st.st_mtime)}
                    )
                except OSError:
                    continue
                if len(matches) >= cap:
                    truncated = True
                    break
        if truncated:
            break

    matches.sort(key=lambda e: -e["mtime"])
    return json.dumps(
        {
            "root": root,
            "pattern": pattern,
            "match_count": len(matches),
            "truncated": truncated,
            "matches": matches,
        },
        ensure_ascii=False,
    )
