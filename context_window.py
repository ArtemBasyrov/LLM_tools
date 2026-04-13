"""
Context window management for the LLM tool calling interface.
"""

import os
import ollama
import sys
import shutil

from system_prompt import MODEL

_IS_TTY = sys.stdout.isatty()


def _ansi(*codes: int) -> str:
    return f"\033[{';'.join(map(str, codes))}m" if _IS_TTY else ""


_RESET = _ansi(0)
_BOLD = _ansi(1)
_DIM = _ansi(2)
_ITALIC = _ansi(3)
_FG_CYAN = _ansi(36)
_FG_GREEN = _ansi(32)
_FG_YELLOW = _ansi(33)
_FG_BLUE = _ansi(34)
_FG_GRAY = _ansi(90)
_BG_NONE = ""

# Composed styles
STYLE_USER = _BOLD + _FG_GREEN
STYLE_ASSISTANT = _BOLD + _FG_CYAN
STYLE_TOOL_CALL = _FG_YELLOW
STYLE_TOOL_RES = _FG_GRAY
STYLE_THINK = _DIM
STYLE_STATS = _DIM
STYLE_HEADER = _BOLD + _FG_BLUE

_TOOL_RESULT_MAX = 300  # chars shown in tool-result lines
_CTX_BAR_WIDTH = 24  # characters in the context fill bar


def _term_width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


def get_context_window() -> int:
    """Query Ollama for the model's context length; fall back to 32 768."""
    try:
        info = ollama.show(MODEL)
        modelinfo = getattr(info, "modelinfo", {}) or {}
        for key in modelinfo:
            if key.endswith(".context_length"):
                val = modelinfo[key]
                if val:
                    return int(val)
        for key in ("context_length", "num_ctx"):
            val = modelinfo.get(key)
            if val:
                return int(val)
    except Exception:
        pass
    return 32_768


def context_bar(used: int, total: int) -> str:
    """Return an ANSI-coloured fill bar with usage fraction."""
    if total <= 0:
        return ""
    frac = min(used / total, 1.0)
    filled = round(frac * _CTX_BAR_WIDTH)
    bar = "█" * filled + "░" * (_CTX_BAR_WIDTH - filled)
    pct = frac * 100

    if pct >= 90:
        colour = _ansi(31)  # red
    elif pct >= 70:
        colour = _ansi(33)  # yellow
    else:
        colour = _ansi(32)  # green

    return f"{colour}{bar}{_RESET} {used:,} / {total:,} tok  ({pct:.1f}%)"


def context_prefix(used: int, total: int) -> str:
    """One-line context status prepended to every user message so the LLM can see it."""
    if total <= 0:
        return ""
    pct = used / total * 100
    return f"<context_window>{used:,}/{total:,} tok ({pct:.0f}%)</context_window>\n"


def trim_messages(messages: list, keep_recent: int = 8) -> None:
    """Keep the system message and the most recent *keep_recent* messages.

    Called automatically after a successful session_save so the LLM can load
    its fresh snapshot into a much smaller context window.
    """
    if len(messages) <= keep_recent + 1:
        return
    messages[:] = [messages[0]] + messages[-keep_recent:]
    print(
        f"\n{_DIM}  [context trimmed — kept last {keep_recent} messages; "
        f"call session_recall() to reload the snapshot]{_RESET}\n"
    )


def offload() -> None:
    """Unload the model from GPU/RAM by sending keep_alive=0."""
    msg = f"  Offloading {MODEL}…"
    print(f"{_DIM}{msg}{_RESET}", end="", flush=True)
    try:
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": ""}],
            keep_alive=0,
        )
    except Exception:
        pass
    print(f"\r{' ' * len(msg)}\r", end="", flush=True)


def warmup() -> None:
    msg = f"  Loading {MODEL}…"
    print(f"{_DIM}{msg}{_RESET}", end="", flush=True)
    try:
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            think=False,
            options={"num_predict": 1},
            keep_alive=15,
        )
    except Exception:
        pass
    print(f"\r{' ' * len(msg)}\r", end="", flush=True)
