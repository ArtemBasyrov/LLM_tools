"""
Shared helpers for file tools: ANSI formatting, diff display, user confirmation,
line-number gutters.
"""

import difflib
import os
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


def show_edit_diff(
    path: str, original: str, updated: str, label: str = "edit_file"
) -> None:
    """Diff between two full strings (post-replacement). Replaces the older
    ``show_edit_diff(path, original, old, new)`` form so batch edits and
    ``replace_all`` use the same path."""
    old_lines = original.splitlines(keepends=True)
    new_lines = updated.splitlines(keepends=True)

    diff = list(
        difflib.unified_diff(
            old_lines, new_lines, fromfile=f"a/{path}", tofile=f"b/{path}"
        )
    )

    print(
        f"\n{_BOLD}{_YELLOW}── {label} diff ──────────────────────────────────{_RESET}"
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


# ---------------------------------------------------------------------------
# Line-number gutter (used by read_file output)
# ---------------------------------------------------------------------------


def with_line_numbers(text: str, start_line: int = 1) -> str:
    """
    Prefix each line with a right-aligned line number and a separator, so the
    model can copy line numbers verbatim into edit_file/start_line calls.

    Format:  '   42 | def foo():'
    """
    if not text:
        return text
    lines = text.splitlines(keepends=True)
    last = start_line + len(lines) - 1
    width = max(4, len(str(last)))
    out = []
    for i, line in enumerate(lines):
        n = start_line + i
        # Preserve whether the original line ended with \n
        body = line.rstrip("\n")
        suffix = "\n" if line.endswith("\n") else ""
        out.append(f"{n:>{width}} | {body}{suffix}")
    return "".join(out)


# ---------------------------------------------------------------------------
# Near-miss search (for edit_file failures)
# ---------------------------------------------------------------------------


def closest_lines(haystack: str, needle: str, k: int = 2) -> list[dict]:
    """
    When ``needle`` is not found in ``haystack``, return the top-k file lines
    that are closest in similarity, with their line numbers, so the model can
    correct whitespace / typo errors on the next try.
    """
    needle_first = needle.splitlines()[0].strip() if needle.strip() else ""
    if not needle_first:
        return []
    scores: list[tuple[float, int, str]] = []
    for i, line in enumerate(haystack.splitlines(), start=1):
        ratio = difflib.SequenceMatcher(None, line.strip(), needle_first).ratio()
        if ratio >= 0.6:
            scores.append((ratio, i, line))
    scores.sort(reverse=True)
    return [
        {"line_number": ln, "text": text, "similarity": round(r, 2)}
        for r, ln, text in scores[:k]
    ]
