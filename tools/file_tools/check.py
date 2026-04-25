"""
check_syntax — fast static checks for written files.

Implements the "generate-and-check" feedback loop. Runs the cheapest
appropriate validator for each extension and returns structured findings the
model can act on. Called automatically by write_file/edit_file; also exposed
as a standalone tool so the model can re-validate after manual edits.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from tools import register


def _run(cmd: list[str], timeout: int = 8) -> tuple[int, str, str]:
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, p.stdout.strip(), p.stderr.strip()
    except FileNotFoundError:
        return 127, "", f"missing binary: {cmd[0]}"
    except subprocess.TimeoutExpired:
        return 124, "", f"timed out after {timeout}s"


def _check_python(path: str) -> dict:
    out: dict = {}
    rc, _so, se = _run(["python3", "-m", "py_compile", path])
    out["py_compile"] = "ok" if rc == 0 else se or "syntax error"
    if shutil.which("ruff"):
        rc, so, se = _run(["ruff", "check", "--output-format=concise", "--quiet", path])
        out["ruff"] = "ok" if rc == 0 and not so else (so or se).splitlines()[:20]
    return out


def _check_json(path: str, content: str | None = None) -> dict:
    try:
        text = content if content is not None else open(path, encoding="utf-8").read()
        json.loads(text)
        return {"json": "ok"}
    except json.JSONDecodeError as e:
        return {"json": f"invalid JSON: {e}"}


def _check_yaml(path: str, content: str | None = None) -> dict:
    try:
        import yaml  # type: ignore
    except ImportError:
        return {"yaml": "skipped (pyyaml not installed)"}
    try:
        text = content if content is not None else open(path, encoding="utf-8").read()
        yaml.safe_load(text)
        return {"yaml": "ok"}
    except Exception as e:
        return {"yaml": f"invalid YAML: {e}"}


def _check_toml(path: str, content: str | None = None) -> dict:
    try:
        import tomllib  # type: ignore
    except ImportError:
        return {"toml": "skipped (tomllib not available)"}
    try:
        text = content if content is not None else open(path, encoding="utf-8").read()
        tomllib.loads(text)
        return {"toml": "ok"}
    except Exception as e:
        return {"toml": f"invalid TOML: {e}"}


def _check_content(path: str, content: str | None = None) -> dict:
    """
    Used inline by write_file/edit_file. Best-effort, never raises. Returns
    {} for unsupported extensions so callers can render the section conditionally.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext == ".py":
        # py_compile reads from disk; for in-memory content fall back to compile().
        if content is not None:
            try:
                compile(content, path, "exec")
                out: dict = {"py_compile": "ok"}
            except SyntaxError as e:
                out = {"py_compile": f"line {e.lineno}: {e.msg}"}
            if shutil.which("ruff"):
                rc, so, se = _run(
                    ["ruff", "check", "--output-format=concise", "--quiet", path]
                )
                out["ruff"] = (
                    "ok" if rc == 0 and not so else (so or se).splitlines()[:20]
                )
            return out
        return _check_python(path)
    if ext == ".json":
        return _check_json(path, content)
    if ext in (".yaml", ".yml"):
        return _check_yaml(path, content)
    if ext == ".toml":
        return _check_toml(path, content)
    return {}


@register(
    description=(
        "Run fast static checks on a file (syntax + linter where available). "
        "Use after write_file/edit_file to confirm a change is sound, or after "
        "manual edits via bash/heredoc. Per-extension: "
        "• .py → py_compile (always) and ruff check (if installed). "
        "• .json/.yaml/.toml → strict parser. "
        "Other extensions return an empty result. "
        "Examples: "
        "verify a script compiles → check_syntax(path='build.py'); "
        "lint after a refactor → check_syntax(path='module.py'); "
        "validate JSON → check_syntax(path='config.json')."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file to check.",
            },
        },
        "required": ["path"],
    },
)
def check_syntax(path: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        return json.dumps({"error": f"File not found: {path}"})
    return json.dumps(
        {"path": path, "checks": _check_content(path)}, ensure_ascii=False
    )
