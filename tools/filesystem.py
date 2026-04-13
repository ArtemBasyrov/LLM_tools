"""
Bash-based filesystem operations.
"""

import json
import subprocess
import os
import sys

from tools import register


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
        "This tool provides access to the full power of bash for filesystem tasks. "
        "Use this to list directories, create directories, remove files/directories, "
        "and perform other filesystem operations. "
        "For potentially destructive commands (like rm, mv, cp, etc.), "
        "manual confirmation will be required before execution."
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
