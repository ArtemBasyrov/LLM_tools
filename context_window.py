"""
Context window management for the LLM tool calling interface.
"""

import os
import shutil
import sys

import ollama

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


def warmup(system_prompt: str = "") -> None:
    msg = f"  Loading {MODEL}…"
    print(f"{_DIM}{msg}{_RESET}", end="", flush=True)
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": "hi"})
    try:
        ollama.chat(
            model=MODEL,
            messages=messages,
            think=False,
            options={"num_predict": 1},
            keep_alive="15m",
        )
    except Exception:
        pass
    print(f"\r{' ' * len(msg)}\r", end="", flush=True)


# ---------------------------------------------------------------------------
# Staged compaction
# ---------------------------------------------------------------------------


def compact_messages(messages: list, used: int, total: int) -> bool:
    """
    Graduated compaction pipeline triggered by context fill %.

    Stage 1 (≥80%): Surgical clearing — replace bulky old tool results with pointers.
    Stage 2 (≥85%): Fast pruning — drop middle user/assistant messages.
    Stage 3 (≥92%): LLM summarization — compress history into an episodic block.

    Returns True if any compaction was applied.
    """
    if total <= 0 or used <= 0:
        return False
    pct = used / total * 100
    if pct < 80:
        return False

    compacted = False
    if pct >= 80:
        compacted |= _surgical_clear(messages)
    if pct >= 85:
        compacted |= _fast_prune(messages)
    if pct >= 92:
        compacted |= _llm_compact(messages)
    return compacted


def _surgical_clear(
    messages: list,
    keep_recent: int = 8,
    min_chars: int = 500,
) -> bool:
    """Replace bulky tool-result bodies outside the recent tail with a pointer."""
    cleared = False
    safe_start = max(1, len(messages) - keep_recent)

    for i, msg in enumerate(messages):
        if i >= safe_start:
            continue
        if msg.get("role") == "tool" and len(msg.get("content", "")) > min_chars:
            original_len = len(msg["content"])
            msg["content"] = (
                f"[Tool result cleared — {original_len:,} chars removed to free context. "
                "Re-run the tool to get the result again.]"
            )
            cleared = True

    if cleared:
        print(f"\n{_DIM}  [context: surgical tool-result clearing applied]{_RESET}\n")
    return cleared


def _fast_prune(messages: list, keep_recent: int = 10) -> bool:
    """Drop middle user/assistant messages, keeping system message + last keep_recent."""
    if len(messages) <= keep_recent + 1:
        return False
    system_msg = messages[0]
    tail = messages[-keep_recent:]
    messages[:] = [system_msg] + tail
    print(
        f"\n{_DIM}  [context: fast pruning — kept system + last {keep_recent} messages]{_RESET}\n"
    )
    return True


def _llm_compact(messages: list, keep_recent: int = 6) -> bool:
    """Use the local model to summarize history into a compact episodic block."""
    middle = messages[1 : max(1, len(messages) - keep_recent)]
    if not middle:
        return False

    history_text = []
    for msg in middle:
        role = msg.get("role", "unknown").upper()
        content = (msg.get("content") or "")[
            :2_000
        ]  # cap each msg for the compact call
        if content:
            history_text.append(f"[{role}]: {content}")

    if not history_text:
        return False

    compact_prompt = [
        {
            "role": "system",
            "content": (
                "You are a context compaction assistant. "
                "Summarize the conversation history below into a compact Markdown block. "
                "Include exactly these sections:\n"
                "## Session Intent\n"
                "## Key Facts & Decisions\n"
                "## Pending Work\n"
                "## Artifact Index (files read/created/modified with full paths)\n\n"
                "Rules: be terse; preserve exact file paths, function names, and error codes; "
                "no commentary; output only the Markdown block."
            ),
        },
        {"role": "user", "content": "History:\n\n" + "\n\n".join(history_text)},
    ]

    try:
        response = ollama.chat(
            model=MODEL,
            messages=compact_prompt,
            think=False,
            options={"num_predict": 1024},
            keep_alive="15m",
        )
        summary = (response.message.content or "").strip()
    except Exception:
        return False

    if not summary:
        return False

    system_msg = messages[0]
    tail = messages[-keep_recent:]
    compact_block = {
        "role": "assistant",
        "content": f"[COMPACTED HISTORY — auto-generated summary]\n\n{summary}",
        "tool_calls": [],
    }
    messages[:] = [system_msg, compact_block] + tail
    print(f"\n{_DIM}  [context: LLM compaction applied — history summarized]{_RESET}\n")
    return True
