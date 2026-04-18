"""
Shared helpers for file tools: ANSI formatting, diff display, user confirmation.
"""

import difflib
import sys

_MAX_BYTES = 100_000  # ~100 KB — keeps context manageable

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


def _colorize_diff(lines: list[str]) -> str:
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


def show_write_diff(path: str, new_content: str, label: str = "write_file") -> None:
    import os

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

    print(f"\n{_BOLD}{_YELLOW}── {label} diff ──────────────────────────────{_RESET}")
    if diff:
        print(_colorize_diff(diff), end="")
    else:
        print(f"{_DIM}  (no changes){_RESET}")
    print(
        f"{_BOLD}{_YELLOW}─────────────────────────────────────────────────────{_RESET}\n"
    )


def show_edit_diff(path: str, original: str, old_string: str, new_string: str) -> None:
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


def confirm(prompt: str) -> bool:
    try:
        answer = input(f"{_BOLD}{prompt} [y/N] {_RESET}").strip().lower()
    except (EOFError, KeyboardInterrupt):
        return False
    return answer in ("y", "yes")
