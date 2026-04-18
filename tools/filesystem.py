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
        r">\s*(?!/dev/null\b)\S",  # > redirect, but not to /dev/null
        r"\btee\b",
        r"\bdd\b.*\bof=",
        r"\bsponge\b",
    ]
    return any(re.search(p, command) for p in patterns)


def _is_dangerous_command(command: str) -> bool:
    """
    Determine if a bash command is potentially dangerous/destructive.
    """
    import re

    dangerous_patterns = [
        # File / directory destruction
        r"\brm\b",
        r"\bshred\b",
        r"\btruncate\b",
        r"\bsrm\b",
        # File / directory mutation
        r"\bmv\b",
        r"\bcp\b",
        r"\bln\b",
        r"\brename\b",
        # Permission / ownership changes
        r"\bchmod\b",
        r"\bchown\b",
        r"\bchgrp\b",
        r"\bumask\b",
        # Disk / filesystem
        r"\bdd\b",
        r"\bmkfs\b",
        r"\bfdisk\b",
        r"\bparted\b",
        r"\bdiskutil\b",
        r"\bmount\b",
        r"\bumount\b",
        r"\bfsck\b",
        r"\bformat\b",
        # Process termination
        r"\bkill\b",
        r"\bkillall\b",
        r"\bpkill\b",
        r"\bxkill\b",
        # System power / reboot
        r"\bshutdown\b",
        r"\breboot\b",
        r"\bhalt\b",
        r"\bpoweroff\b",
        r"\binit\s+[016]\b",
        # Privilege escalation
        r"\bsudo\b",
        r"\bsu\b\s",
        r"\bdoas\b",
        # Network / firewall
        r"\biptables\b",
        r"\bip6tables\b",
        r"\bnftables\b",
        r"\bpfctl\b",
        r"\bufw\b",
        r"\bfirewall-cmd\b",
        r"\bnc\b\s",
        r"\bnetcat\b",
        # User / group management
        r"\buseradd\b",
        r"\buserdel\b",
        r"\busermod\b",
        r"\bgroupadd\b",
        r"\bgroupdel\b",
        r"\bgroupmod\b",
        r"\bpasswd\b",
        # Cron / scheduled tasks
        r"\bcrontab\b",
        r"\bat\b\s",
        # Package removal
        r"\bpip\s+uninstall\b",
        r"\bpip3\s+uninstall\b",
        r"\bnpm\s+uninstall\b",
        r"\byarn\s+remove\b",
        r"\bbrew\s+uninstall\b",
        r"\bbrew\s+remove\b",
        r"\bapt[-\s]get\s+remove\b",
        r"\bapt[-\s]get\s+purge\b",
        r"\bapt\s+remove\b",
        r"\bapt\s+purge\b",
        r"\byum\s+remove\b",
        r"\bdnf\s+remove\b",
        # Destructive git operations
        r"\bgit\s+reset\b",
        r"\bgit\s+clean\b",
        r"\bgit\s+push\s+.*--force\b",
        r"\bgit\s+push\s+.*-f\b",
        r"\bgit\s+branch\s+.*-[Dd]\b",
        r"\bgit\s+tag\s+.*-d\b",
        r"\bgit\s+stash\s+drop\b",
        r"\bgit\s+stash\s+clear\b",
        r"\bgit\s+reflog\s+delete\b",
        # Shell history
        r"\bhistory\s+-[cw]\b",
        # Environment / variable manipulation
        r"\bunset\b",
        r"\bexport\b",
        # xargs piped destructive
        r"\bxargs\b.*\brm\b",
        r"\bxargs\b.*\bkill\b",
        # find with destructive actions
        r"\bfind\b.*-delete\b",
        r"\bfind\b.*-exec\b.*\brm\b",
        # curl / wget writing to disk
        r"\bcurl\b.*\s-[a-zA-Z]*[oO]\b",
        r"\bwget\b",
    ]

    command_stripped = command.strip()
    return any(re.search(p, command_stripped) for p in dangerous_patterns)


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
        "Execute an arbitrary bash command — filesystem ops, git, grep, process inspection, "
        "package management, or any shell task. "
        "Use this whenever a built-in tool does not cover the operation. "
        "Potentially destructive commands (rm, mv, kill, shutdown, sudo, git reset, "
        "pip uninstall, iptables, crontab, wget, etc.) require manual user "
        "confirmation before execution. "
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
