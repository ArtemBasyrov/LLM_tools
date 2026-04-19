"""
write_file — create or overwrite a file with given content
write_json — validate, format, and write JSON to a file
"""

import json
import os

from tools import register
from tools.file_tools._helpers import confirm, show_write_diff


@register(
    description=(
        "Write text content to a file, creating it if it does not exist or "
        "overwriting it entirely if it does. "
        "REQUIRED: both 'path' and 'content' must be provided in one call — "
        "there is no separate open/write step. "
        "Use for new files or complete rewrites. "
        "Use edit_file instead for targeted changes to an existing file. "
        "Examples: "
        "create a new script → write_file(path='scripts/deploy.sh', content='#!/bin/bash\\n...'); "
        "completely replace a config → write_file(path='config.yaml', content='...full new content...'); "
        "NOT for: small targeted changes to existing files (use edit_file to avoid overwriting unchanged content)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file. Parent directories are created automatically.",
            },
            "content": {
                "type": "string",
                "description": "The full text to write. Must be provided — omitting it is an error.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    show_write_diff(path, content, label="write_file")

    if not confirm(f"Allow write_file → {path}?"):
        return json.dumps({"error": "User denied write_file — no changes made."})

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(content)
    except OSError as e:
        return json.dumps({"error": str(e)})

    lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
    return json.dumps({"success": True, "path": path, "lines_written": lines})


@register(
    description=(
        "Validate and write a JSON value to a file with consistent formatting (2-space indent). "
        "Use this instead of write_file when the output must be well-formed JSON — "
        "it validates and rejects malformed input before touching the file. "
        "Examples: "
        'write a config → write_json(path=\'config.json\', content=\'{"host": "localhost", "port": 5432}\'); '
        "write a list → write_json(path='items.json', content='[{\"id\": 1}, {\"id\": 2}]'); "
        "NOT for: non-JSON formats like YAML, TOML, or plain text (use write_file for those)."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the JSON file.",
            },
            "content": {
                "type": "string",
                "description": "A valid JSON string to write. e.g. '{\"key\": \"value\"}' or '[1, 2, 3]'. Will be rejected if not valid JSON.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_json(path: str, content: str) -> str:
    path = os.path.expanduser(path)

    try:
        data = json.loads(content)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON content: {e}"})

    formatted = json.dumps(data, indent=2, ensure_ascii=False) + "\n"
    show_write_diff(path, formatted, label="write_json")

    if not confirm(f"Allow write_json → {path}?"):
        return json.dumps({"error": "User denied write_json — no changes made."})

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(formatted)
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"success": True, "path": path})
