"""
run_tests — run pytest on a focused path with a tight output cap.

Wraps `pytest -x --tb=short` and trims output so a failing test doesn't blow
out the context window. Falls back to running the full suite if no path is
given. Honours the project's micromamba environment when available.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess

from tools import register


_MAX_OUTPUT_CHARS = 6_000  # cap on tool result size


def _resolve_pytest() -> list[str]:
    if shutil.which("pytest"):
        return ["pytest"]
    if shutil.which("python3"):
        return ["python3", "-m", "pytest"]
    return ["python", "-m", "pytest"]


@register(
    description=(
        "Run pytest, optionally scoped to a file, directory, or test node id. "
        "Returns the trimmed last part of pytest output plus the exit code. Uses "
        "`-x` (stop on first failure) and `--tb=short` so failures are compact. "
        "Examples: "
        "run all tests → run_tests(); "
        "run one file → run_tests(target='tests/test_critic.py'); "
        "run one test → run_tests(target='tests/test_plan.py::test_serialize'); "
        "extra args → run_tests(target='tests/', extra='-q -k smoke'); "
        "NOT for: starting a long-running dev server (use bash)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "target": {
                "type": "string",
                "description": "File / dir / nodeid passed to pytest. Omit to run the whole suite.",
            },
            "extra": {
                "type": "string",
                "description": "Extra pytest args, e.g. '-q -k smoke'. Will be tokenized on whitespace.",
            },
            "timeout": {
                "type": "integer",
                "description": "Hard timeout in seconds. Default 60, max 300.",
            },
        },
        "required": [],
    },
)
def run_tests(
    target: str | None = None,
    extra: str | None = None,
    timeout: int = 60,
) -> str:
    timeout = max(5, min(int(timeout), 300))
    cmd = _resolve_pytest() + ["-x", "--tb=short"]
    if extra:
        cmd.extend(extra.split())
    if target:
        cmd.append(os.path.expanduser(target))

    try:
        p = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return json.dumps({"error": "pytest not available"})
    except subprocess.TimeoutExpired as e:
        partial = (e.stdout or "") + (e.stderr or "")
        return json.dumps(
            {
                "error": f"timed out after {timeout}s",
                "tail": partial[-_MAX_OUTPUT_CHARS:],
            }
        )

    out = (p.stdout or "") + (("\nSTDERR:\n" + p.stderr) if p.stderr else "")
    if len(out) > _MAX_OUTPUT_CHARS:
        out = "…[truncated head]…\n" + out[-_MAX_OUTPUT_CHARS:]

    return json.dumps(
        {
            "exit_code": p.returncode,
            "passed": p.returncode == 0,
            "command": " ".join(cmd),
            "output": out,
        },
        ensure_ascii=False,
    )
