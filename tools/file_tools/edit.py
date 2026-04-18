"""
edit_file — replace one exact occurrence of old_string with new_string in a file
"""

import json
import os

from tools import register
from tools.file_tools._helpers import _MAX_BYTES, confirm, show_edit_diff


@register(
    description=(
        "Edit a file by replacing one exact occurrence of old_string with new_string. "
        "old_string must match the file content exactly (including whitespace and indentation). "
        "Returns an error if old_string is not found or appears more than once. "
        "Use read_file first if you are unsure of the exact text."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "old_string": {
                "type": "string",
                "description": "The exact text to find and replace. Must be unique in the file.",
            },
            "new_string": {
                "type": "string",
                "description": "The text to substitute in place of old_string.",
            },
        },
        "required": ["path", "old_string", "new_string"],
    },
)
def edit_file(path: str, old_string: str, new_string: str) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            original = fh.read(_MAX_BYTES)
    except UnicodeDecodeError:
        return json.dumps(
            {"error": "File appears to be binary and cannot be edited as text."}
        )
    except OSError as e:
        return json.dumps({"error": str(e)})

    count = original.count(old_string)
    if count == 0:
        return json.dumps({"error": "old_string not found in file."})
    if count > 1:
        return json.dumps(
            {
                "error": (
                    f"old_string appears {count} times; it must be unique. "
                    "Add more surrounding context to make it unambiguous."
                )
            }
        )

    show_edit_diff(path, original, old_string, new_string)

    if not confirm(f"Allow edit_file → {path}?"):
        return json.dumps({"error": "User denied edit_file — no changes made."})

    updated = original.replace(old_string, new_string, 1)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(updated)
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"success": True, "path": path})
