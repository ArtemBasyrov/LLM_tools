"""
Directory and filesystem navigation tools.

list_directory   — list the contents of a directory
make_directory   — create a new folder (requires user confirmation)
remove_file      — delete a file or directory tree (requires user confirmation)
get_working_context — show cwd and a one-level overview of its contents
"""

import json
import os
import shutil
import sys

from tools import register

# ---------------------------------------------------------------------------
# ANSI helpers (mirrors files.py)
# ---------------------------------------------------------------------------

_IS_TTY = sys.stdout.isatty()


def _a(*codes: int) -> str:
    return f"\033[{';'.join(map(str, codes))}m" if _IS_TTY else ""


_RESET = _a(0)
_BOLD = _a(1)
_DIM = _a(2)
_RED = _a(31)
_GREEN = _a(32)
_YELLOW = _a(33)
_CYAN = _a(36)


def _confirm(prompt: str) -> bool:
    """Ask the user y/n. Returns True only if approved."""
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
        "List the contents of a directory. Returns each entry's name, type "
        "(file or directory), and size in bytes (files only). "
        "Use this to explore what is inside a folder before reading or editing files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": (
                    "Absolute or relative path to the directory to list. "
                    "Defaults to the current working directory if omitted."
                ),
            },
        },
        "required": [],
    },
)
def list_directory(path: str = ".") -> str:
    path = os.path.expanduser(path)
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return json.dumps({"error": f"Path not found: {abs_path}"})
    if not os.path.isdir(abs_path):
        return json.dumps({"error": f"Not a directory: {abs_path}"})

    try:
        entries = sorted(
            os.scandir(abs_path), key=lambda e: (not e.is_dir(), e.name.lower())
        )
    except PermissionError as e:
        return json.dumps({"error": str(e)})

    items = []
    for entry in entries:
        info: dict = {
            "name": entry.name,
            "type": "directory" if entry.is_dir() else "file",
        }
        if entry.is_file():
            try:
                info["size_bytes"] = entry.stat().st_size
            except OSError:
                info["size_bytes"] = None
        items.append(info)

    return json.dumps({"path": abs_path, "entries": items, "count": len(items)})


@register(
    description=(
        "Show the current working directory and a one-level summary of its contents: "
        "how many files and subdirectories it contains, plus their names. "
        "Use this to orient yourself before exploring or editing files."
    ),
    parameters={
        "type": "object",
        "properties": {},
        "required": [],
    },
)
def get_working_context() -> str:
    cwd = os.getcwd()

    try:
        entries = list(os.scandir(cwd))
    except PermissionError as e:
        return json.dumps({"error": str(e)})

    dirs = sorted(e.name for e in entries if e.is_dir())
    files = sorted(e.name for e in entries if e.is_file())

    return json.dumps(
        {
            "cwd": cwd,
            "subdirectories": dirs,
            "files": files,
            "total_entries": len(entries),
        }
    )


@register(
    description=(
        "Create a new directory (and any missing parent directories). "
        "Always asks the user for permission before creating anything. "
        "Returns an error if the path already exists."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path of the directory to create.",
            },
        },
        "required": ["path"],
    },
)
def make_directory(path: str) -> str:
    path = os.path.expanduser(path)
    abs_path = os.path.abspath(path)

    if os.path.exists(abs_path):
        return json.dumps({"error": f"Path already exists: {abs_path}"})

    print(
        f"\n{_BOLD}{_YELLOW}── make_directory ───────────────────────────────────{_RESET}"
    )
    print(f"  Create directory: {_CYAN}{abs_path}{_RESET}")
    print(
        f"{_BOLD}{_YELLOW}─────────────────────────────────────────────────────{_RESET}\n"
    )

    if not _confirm(f"Allow make_directory → {abs_path}?"):
        return json.dumps({"error": "User denied make_directory — no changes made."})

    try:
        os.makedirs(abs_path)
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"success": True, "path": abs_path})


@register(
    description=(
        "Permanently delete a file or an entire directory tree. "
        "Always asks the user for permission before deleting anything. "
        "This action is irreversible — use with caution."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file or directory to remove.",
            },
        },
        "required": ["path"],
    },
)
def remove_file(path: str) -> str:
    path = os.path.expanduser(path)
    abs_path = os.path.abspath(path)

    if not os.path.exists(abs_path):
        return json.dumps({"error": f"Path not found: {abs_path}"})

    is_dir = os.path.isdir(abs_path)
    kind = "directory tree" if is_dir else "file"

    # Show what will be deleted
    print(
        f"\n{_BOLD}{_RED}── remove_file ─────────────────────────────────────{_RESET}"
    )
    print(f"  Delete {kind}: {_CYAN}{abs_path}{_RESET}")
    if is_dir:
        try:
            n_files = sum(len(fs) for _, _, fs in os.walk(abs_path))
            n_dirs = sum(len(ds) for _, ds, _ in os.walk(abs_path))
            print(
                f"  Contains: {_DIM}{n_dirs} subdirector{'y' if n_dirs == 1 else 'ies'}, "
                f"{n_files} file{'s' if n_files != 1 else ''}{_RESET}"
            )
        except OSError:
            pass
    print(
        f"{_BOLD}{_RED}─────────────────────────────────────────────────────{_RESET}\n"
    )

    if not _confirm(f"Allow remove_file → permanently delete {abs_path}?"):
        return json.dumps({"error": "User denied remove_file — nothing deleted."})

    try:
        if is_dir:
            shutil.rmtree(abs_path)
        else:
            os.remove(abs_path)
    except OSError as e:
        return json.dumps({"error": str(e)})

    return json.dumps({"success": True, "deleted": abs_path, "type": kind})
