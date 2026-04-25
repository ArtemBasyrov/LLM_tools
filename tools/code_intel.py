"""
go_to_definition — locate a Python symbol's def/class line in a file or project.
find_references  — list call/usage sites of a symbol within a file or project.

Lightweight AST-based code intelligence. Python only (the project's primary
language). Limits: no cross-module type resolution; ``find_references`` matches
on Name + Attribute nodes by ident — good enough for a local model to navigate
without an LSP. Falls back to regex over non-.py files when given an explicit
glob.
"""

from __future__ import annotations

import ast
import json
import os
from typing import Iterable

from tools import register
from tools.find import find_files as _find_files


def _iter_python_files(root: str) -> Iterable[str]:
    """Walk root and yield .py files, skipping common heavy dirs."""
    skip = {
        ".git",
        "node_modules",
        ".venv",
        "venv",
        "__pycache__",
        ".pytest_cache",
        ".ruff_cache",
        "dist",
        "build",
    }
    for cwd, dirs, files in os.walk(root, followlinks=False):
        dirs[:] = [d for d in dirs if d not in skip and not d.startswith(".")]
        for f in files:
            if f.endswith(".py"):
                yield os.path.join(cwd, f)


def _scan_definitions(path: str, name: str) -> list[dict]:
    """Return every top-level or nested def/class named ``name`` in ``path``."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        tree = ast.parse(source, path)
    except (OSError, SyntaxError):
        return []
    out: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            if node.name == name:
                kind = "class" if isinstance(node, ast.ClassDef) else "function"
                try:
                    sig = ast.unparse(node).splitlines()[0].rstrip(":")
                except Exception:
                    sig = node.name
                out.append(
                    {"path": path, "line": node.lineno, "kind": kind, "signature": sig}
                )
    return out


def _scan_references(path: str, name: str) -> list[dict]:
    """Return Name/Attribute usages of ``name`` in ``path``."""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            source = fh.read()
        tree = ast.parse(source, path)
    except (OSError, SyntaxError):
        return []
    src_lines = source.splitlines()
    out: list[dict] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and node.id == name:
            out.append(
                {
                    "path": path,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "context": src_lines[node.lineno - 1].strip()
                    if 0 < node.lineno <= len(src_lines)
                    else "",
                }
            )
        elif isinstance(node, ast.Attribute) and node.attr == name:
            out.append(
                {
                    "path": path,
                    "line": node.lineno,
                    "col": node.col_offset,
                    "context": src_lines[node.lineno - 1].strip()
                    if 0 < node.lineno <= len(src_lines)
                    else "",
                    "attribute": True,
                }
            )
    return out


@register(
    description=(
        "Find a Python symbol's definition (def or class) by name. Searches a single file when "
        "path is given, else walks the project tree. Returns every match (overloads, classes vs "
        "methods of the same name). "
        "Examples: "
        "find a fn project-wide → go_to_definition(name='process_order'); "
        "limit to one file → go_to_definition(name='Orchestrator', path='agent/orchestrator.py'); "
        "limit to a subtree → go_to_definition(name='build_index', root='tools/'); "
        "NOT for: non-Python languages (use search_file with a regex)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Symbol name (function or class).",
            },
            "path": {"type": "string", "description": "Search inside this file only."},
            "root": {
                "type": "string",
                "description": "Search under this directory (default: cwd).",
            },
        },
        "required": ["name"],
    },
)
def go_to_definition(
    name: str,
    path: str | None = None,
    root: str | None = None,
) -> str:
    if path:
        path = os.path.expanduser(path)
        results = _scan_definitions(path, name)
        return json.dumps({"name": name, "path": path, "results": results})

    base = os.path.expanduser(root) if root else "."
    results: list[dict] = []
    for p in _iter_python_files(base):
        results.extend(_scan_definitions(p, name))
    return json.dumps(
        {"name": name, "root": base, "match_count": len(results), "results": results},
        ensure_ascii=False,
    )


@register(
    description=(
        "Find all references to a Python symbol by name across a file or project. "
        "Matches both bare names (foo) and attribute accesses (obj.foo). Ideal for impact "
        "analysis before a rename or for understanding who calls a function. "
        "Examples: "
        "who calls this fn → find_references(name='build_index'); "
        "limit to a dir → find_references(name='session_save', root='agent/'); "
        "in one file → find_references(name='Plan', path='agent/plan.py'); "
        "NOT for: non-Python text search (use search_file)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "name": {
                "type": "string",
                "description": "Symbol name to find references for.",
            },
            "path": {"type": "string", "description": "Search inside this file only."},
            "root": {
                "type": "string",
                "description": "Search under this directory (default: cwd).",
            },
            "max_results": {
                "type": "integer",
                "description": "Cap on returned references (default 200).",
            },
        },
        "required": ["name"],
    },
)
def find_references(
    name: str,
    path: str | None = None,
    root: str | None = None,
    max_results: int = 200,
) -> str:
    cap = max(1, min(int(max_results), 1000))

    if path:
        path = os.path.expanduser(path)
        results = _scan_references(path, name)[:cap]
        return json.dumps({"name": name, "path": path, "results": results})

    base = os.path.expanduser(root) if root else "."
    results: list[dict] = []
    for p in _iter_python_files(base):
        results.extend(_scan_references(p, name))
        if len(results) >= cap:
            results = results[:cap]
            break
    return json.dumps(
        {"name": name, "root": base, "match_count": len(results), "results": results},
        ensure_ascii=False,
    )


# Silence unused-import nag (find_files is a sibling we deliberately
# don't depend on at runtime — the iterator above is faster).
_ = _find_files
