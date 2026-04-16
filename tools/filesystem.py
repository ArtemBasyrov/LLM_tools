"""
Bash-based filesystem operations.
"""

import json
import subprocess
import os
import sys
from pathlib import Path

from tools import register

# Context files to look for, in priority order
_CONTEXT_FILES = [
    "CLAUDE.md",
    "AGENT.md",
    "AGENTS.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
]


def _is_file_writing_command(command: str) -> bool:
    """
    Detect bash commands that write or overwrite files via shell redirection or
    common write-oriented utilities.
    """
    import re

    patterns = [
        r">\s*\S",  # any > redirect (> file or >> file)
        r"\btee\b",
        r"\bdd\b.*\bof=",
        r"\bsponge\b",
    ]
    return any(re.search(p, command) for p in patterns)


def _is_dangerous_command(command: str) -> bool:
    """
    Determine if a bash command is potentially dangerous/destructive.
    """
    dangerous_patterns = [
        "rm ",
        "mv ",
        "cp ",
        "chmod ",
        "chown ",
        "chgrp ",
        "dd ",
        "mkfs ",
        "mount ",
        "umount ",
        "truncate ",
        "find .* -delete",
        "find .* -exec rm",
    ]

    command_lower = command.lower().strip()
    return any(pattern in command_lower for pattern in dangerous_patterns)


def _confirm_dangerous_command(command: str) -> bool:
    """
    Ask user for confirmation before executing a potentially dangerous command.
    """
    print(f"\n⚠️  DANGEROUS COMMAND DETECTED")
    print(f"Command: {command}")
    print("This command may modify or delete files/directories.")
    print("Do you want to proceed? (y/N)")

    try:
        answer = input().strip().lower()
        return answer in ("y", "yes")
    except (EOFError, KeyboardInterrupt):
        return False


@register(
    description=(
        "Execute a bash command for filesystem operations. "
        "Use this to list directories, create directories, remove files/directories, "
        "and perform other filesystem operations. "
        "For potentially destructive commands (like rm, mv, cp, etc.), "
        "manual confirmation will be required before execution. "
        "IMPORTANT: writing files via bash (>, >>, tee, dd of=, etc.) is BLOCKED — "
        "use write_file or edit_file instead."
    ),
    parameters={
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": (
                    "The bash command to execute. Examples: "
                    "'ls -la', 'mkdir -p new_folder', 'rm -rf old_folder', "
                    "'pwd', 'find . -name '*.py' -type f'"
                ),
            },
        },
        "required": ["command"],
    },
)
def bash(command: str) -> str:
    """
    Execute a bash command and return the result.

    Args:
        command (str): The bash command to execute

    Returns:
        str: JSON-encoded result with stdout, stderr, and return code
    """
    # Block file-writing via bash — use write_file / edit_file instead
    if _is_file_writing_command(command):
        return json.dumps(
            {
                "error": (
                    "Writing files via bash is not allowed. "
                    "Use write_file to create or overwrite a file, "
                    "or edit_file to modify an existing one. "
                    "Those tools require user approval and keep an audit trail."
                ),
                "command": command,
                "status": "blocked",
            }
        )

    # Check if command is potentially dangerous
    if _is_dangerous_command(command):
        if not _confirm_dangerous_command(command):
            return json.dumps(
                {
                    "error": "User denied dangerous command execution",
                    "command": command,
                    "status": "cancelled",
                }
            )

    try:
        # Execute the bash command
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True, cwd=os.getcwd()
        )

        # Prepare the response
        response = {
            "stdout": result.stdout,
            "stderr": result.stderr,
            "returncode": result.returncode,
            "command": command,
        }

        # If there's an error, we might want to include it in the response
        if result.returncode != 0:
            response["error"] = f"Command failed with return code {result.returncode}"

        return json.dumps(response)

    except Exception as e:
        return json.dumps({"error": str(e), "command": command})


@register(
    description=(
        "Read agent/project context files (CLAUDE.md, AGENT.md, AGENTS.md, "
        ".cursorrules, .windsurfrules, .clinerules) from a directory and its "
        "parents. Call this when starting work in an unfamiliar directory so you "
        "understand any project-specific instructions or conventions before acting."
    ),
    parameters={
        "type": "object",
        "properties": {
            "directory": {
                "type": "string",
                "description": (
                    "Absolute or relative path to search. "
                    "Defaults to the current working directory."
                ),
            },
            "walk_parents": {
                "type": "boolean",
                "description": (
                    "If true (default), also read context files found in parent "
                    "directories up to the filesystem root."
                ),
            },
        },
        "required": [],
    },
)
def read_context_files(directory: str = ".", walk_parents: bool = True) -> str:
    """
    Find and return the contents of agent context files in *directory* (and
    optionally its parents).  Results are ordered from outermost directory
    (highest in the tree) to innermost so that more-specific instructions
    override more-general ones.
    """
    start = Path(directory).expanduser().resolve()

    # Collect candidate directories: walk up from start to root
    dirs: list[Path] = []
    current = start
    while True:
        dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
        if not walk_parents:
            break

    # Reverse so outermost (root) comes first → innermost last
    dirs.reverse()

    found: list[dict] = []
    for d in dirs:
        for name in _CONTEXT_FILES:
            path = d / name
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8")
                    found.append({"file": str(path), "content": content})
                except Exception as exc:
                    found.append({"file": str(path), "error": str(exc)})

    if not found:
        return json.dumps(
            {
                "message": "No context files found.",
                "searched_in": str(start),
                "looked_for": _CONTEXT_FILES,
            }
        )

    return json.dumps({"context_files": found, "count": len(found)})
