"""
File write, edit, and JSON tools.

write_file — create or overwrite a file with given content
edit_file  — replace an exact string in a file (targeted edit)
read_json  — parse a JSON file and return structured data
write_json — validate, format, and write JSON to a file

Both write_file and write_json show a diff and ask for user confirmation.
"""

import difflib
import json
import os
import sys

from tools import register

_MAX_BYTES = 100_000  # ~100 KB — same limit as read_file

# ---------------------------------------------------------------------------
# ANSI helpers (only emit escape codes when stdout is a TTY)
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()


def _a(*codes: int) -> str:
    return f"\033[{';'.join(map(str, codes))}m" if _IS_TTY else ""


_RESET = _a(0)
_BOLD = _a(1)
_DIM = _a(2)
_RED = _a(31)
_GREEN = _a(32)
_CYAN = _a(36)
_YELLOW = _a(33)


# ---------------------------------------------------------------------------
# Diff + confirmation helpers
# ---------------------------------------------------------------------------


def _colorize_diff(lines: list[str]) -> str:
    """Return a single string with ANSI-coloured unified diff lines."""
    parts = []
    for line in lines:
        if line.startswith("+") and not line.startswith("+++"):
            parts.append(f"{_GREEN}{line}{_RESET}")
        elif line.startswith("-") and not line.startswith("---"):
            parts.append(f"{_RED}{line}{_RESET}")
        elif line.startswith("@@"):
            parts.append(f"{_CYAN}{line}{_RESET}")
        else:
            parts.append(line)
    return "".join(parts)


def _show_write_diff(path: str, new_content: str) -> None:
    """Print a unified diff for write_file (existing → new, or new file)."""
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8", errors="replace") as fh:
                old_lines = fh.readlines()
        except OSError:
            old_lines = []
        label_old = f"a/{path}"
        label_new = f"b/{path}"
    else:
        old_lines = []
        label_old = "/dev/null"
        label_new = f"b/{path}"

    new_lines = new_content.splitlines(keepends=True)
    diff = list(
        difflib.unified_diff(old_lines, new_lines, fromfile=label_old, tofile=label_new)
    )

    print(
        f"\n{_BOLD}{_YELLOW}── write_file diff ─────────────────────────────────{_RESET}"
    )
    if diff:
        print(_colorize_diff(diff), end="")
    else:
        print(f"{_DIM}  (no changes){_RESET}")
    print(
        f"{_BOLD}{_YELLOW}─────────────────────────────────────────────────────{_RESET}\n"
    )


def _show_edit_diff(path: str, original: str, old_string: str, new_string: str) -> None:
    """Print a unified diff for edit_file (old_string → new_string in context)."""
    old_lines = original.splitlines(keepends=True)
    updated = original.replace(old_string, new_string, 1)
    new_lines = updated.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
        )
    )

    print(
        f"\n{_BOLD}{_YELLOW}── edit_file diff ──────────────────────────────────{_RESET}"
    )
    if diff:
        print(_colorize_diff(diff), end="")
    else:
        print(f"{_DIM}  (no changes){_RESET}")
    print(
        f"{_BOLD}{_YELLOW}─────────────────────────────────────────────────────{_RESET}\n"
    )


def _confirm(prompt: str) -> bool:
    """Ask the user y/n. Returns True if approved."""
    try:
        answer = input(f"{_BOLD}{prompt} [y/N] {_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@register(
    description=(
        "Write text content to a file, creating it if it does not exist or "
        "overwriting it if it does. Use this to create new files or completely "
        "replace an existing file's contents."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "content": {
                "type": "string",
                "description": "The full text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    },
)
def write_file(path: str, content: str) -> str:
    path = os.path.expanduser(path)
    _show_write_diff(path, content)

    if not _confirm(f"Allow write_file → {path}?"):
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

    _show_edit_diff(path, original, old_string, new_string)

    if not _confirm(f"Allow edit_file → {path}?"):
        return json.dumps({"error": "User denied edit_file — no changes made."})

    updated = original.replace(old_string, new_string, 1)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(updated)
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"success": True, "path": path})


# ---------------------------------------------------------------------------
# JSON tools
# ---------------------------------------------------------------------------


@register(
    description=(
        "Read and parse a JSON file, returning its contents as structured data. "
        "Optionally extract a nested value using a dot-separated key path "
        "(e.g. 'users.0.name' to get the name of the first user). "
        "Prefer this over read_file when you need to inspect or query JSON data."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the JSON file.",
            },
            "key_path": {
                "type": "string",
                "description": (
                    "Dot-separated path to extract a nested value, e.g. 'a.b.c' or 'items.0'. "
                    "Omit to return the entire document."
                ),
            },
        },
        "required": ["path"],
    },
)
def read_json(path: str, key_path: str | None = None) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read(_MAX_BYTES)
            truncated = fh.read(1) != ""
    except OSError as e:
        return json.dumps({"error": str(e)})

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    if key_path:
        try:
            for key in key_path.split("."):
                if isinstance(data, list):
                    data = data[int(key)]
                else:
                    data = data[key]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            return json.dumps({"error": f"Key path '{key_path}' not found: {e}"})

    result: dict = {"path": path, "data": data}
    if truncated:
        result["warning"] = (
            f"File truncated at {_MAX_BYTES} bytes — JSON may be incomplete."
        )
    return json.dumps(result, ensure_ascii=False)


@register(
    description=(
        "Write a JSON value to a file with consistent formatting (2-space indent). "
        "Validates that the content is valid JSON before writing. "
        "Use this instead of write_file when the output must be well-formed JSON."
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
                "description": "The JSON content to write (as a JSON-encoded string).",
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
    _show_write_diff(path, formatted)

    if not _confirm(f"Allow write_json → {path}?"):
        return json.dumps({"error": "User denied write_json — no changes made."})

    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            fh.write(formatted)
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"success": True, "path": path})
