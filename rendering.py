"""
Rendering helpers for the LLM tool calling interface.
"""

import sys
import shutil
import textwrap
from tools import schemas

from context_window import (
    _ansi,
    _IS_TTY,
    _RESET,
    _BOLD,
    _DIM,
    _ITALIC,
    _FG_CYAN,
    _FG_GREEN,
    _FG_YELLOW,
    _FG_BLUE,
    _FG_GRAY,
    _BG_NONE,
    STYLE_USER,
    STYLE_ASSISTANT,
    STYLE_TOOL_CALL,
    STYLE_TOOL_RES,
    STYLE_THINK,
    STYLE_STATS,
    STYLE_HEADER,
    _TOOL_RESULT_MAX,
    _CTX_BAR_WIDTH,
    context_bar,
    _term_width,
)


def hr(char: str = "─", style: str = _DIM) -> str:
    return f"{style}{char * _term_width()}{_RESET}"


def print_header(context_window: int) -> None:
    from system_prompt import MODEL

    w = _term_width()
    title = f"  {MODEL}  "
    pad = (w - len(title)) // 2
    print()
    print(hr("━", STYLE_HEADER))
    print(f"{STYLE_HEADER}{' ' * pad}{title}{_RESET}")
    print(hr("━", STYLE_HEADER))
    tools_list = "  ".join(
        f"{STYLE_TOOL_CALL}⬡{_RESET} {s['function']['name']}" for s in schemas()
    )
    print(f"\n{_DIM}Tools:{_RESET}  {tools_list}")
    print(f"{_DIM}Context window:{_RESET}  {context_window:,} tokens")
    print(
        f"{_DIM}Type  {_RESET}{_BOLD}exit{_RESET}{_DIM}  or press  {_RESET}{_BOLD}Ctrl-C{_RESET}{_DIM}  to quit{_RESET}\n"
    )
    print(hr())
    print()


def print_thinking(text: str) -> None:
    if not text or not text.strip():
        return
    w = _term_width()
    inner = w - 4
    label = " thinking "
    bar_left = (inner - len(label)) // 2
    bar_right = inner - len(label) - bar_left
    print(f"{STYLE_THINK}┌─{label}{'─' * bar_right}─┐")
    for line in text.strip().splitlines():
        for chunk in textwrap.wrap(line, width=inner) or [""]:
            print(f"│ {chunk:<{inner}} │")
    print(f"└{'─' * (w - 2)}┘{_RESET}")
    print()


# --- Streaming thinking helpers ---

_thinking_col = 0  # current column inside the box (tracks position for soft-wrap)


def print_thinking_start() -> None:
    global _thinking_col
    w = _term_width()
    inner = w - 4
    label = " thinking "
    bar_left = (inner - len(label)) // 2
    bar_right = inner - len(label) - bar_left
    sys.stdout.write(f"{STYLE_THINK}┌─{label}{'─' * bar_right}─┐\n│ ")
    sys.stdout.flush()
    _thinking_col = 0


def print_thinking_token(text: str) -> None:
    global _thinking_col
    w = _term_width()
    inner = w - 4
    for ch in text:
        if ch == "\n":
            sys.stdout.write("\n│ ")
            _thinking_col = 0
        else:
            if _thinking_col >= inner:
                sys.stdout.write("\n│ ")
                _thinking_col = 0
            sys.stdout.write(ch)
            _thinking_col += 1
    sys.stdout.flush()


def print_thinking_end() -> None:
    w = _term_width()
    sys.stdout.write(f"\n└{'─' * (w - 2)}┘{_RESET}\n\n")
    sys.stdout.flush()


def print_tool_call(name: str, args: dict) -> None:
    args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
    print(f"  {STYLE_TOOL_CALL}⚙ {name}({args_str}){_RESET}")


def print_tool_result(result: str) -> None:
    display = (
        result[:_TOOL_RESULT_MAX] + f"… ({len(result)} chars)"
        if len(result) > _TOOL_RESULT_MAX
        else result
    )
    # Indent every line of the result
    indented = "\n".join(f"    {line}" for line in display.splitlines())
    print(f"{STYLE_TOOL_RES}{indented}{_RESET}\n")


def print_response(text: str) -> None:
    w = _term_width()
    print(f"\n{STYLE_ASSISTANT}╭─ assistant {'─' * (w - 14)}╮{_RESET}")
    for line in (text or "").splitlines():
        for chunk in textwrap.wrap(line, width=w - 4) or [""]:
            print(f"  {chunk}")
    print(f"{STYLE_ASSISTANT}╰{'─' * (w - 2)}╯{_RESET}\n")


# --- Streaming response helpers ---

_response_col = 0  # current column inside the response box


def print_response_start() -> None:
    global _response_col
    w = _term_width()
    sys.stdout.write(f"\n{STYLE_ASSISTANT}╭─ assistant {'─' * (w - 14)}╮{_RESET}\n  ")
    sys.stdout.flush()
    _response_col = 0


def print_response_token(text: str) -> None:
    global _response_col
    w = _term_width()
    inner = w - 4
    for ch in text:
        if ch == "\n":
            sys.stdout.write("\n  ")
            _response_col = 0
        else:
            if _response_col >= inner:
                sys.stdout.write("\n  ")
                _response_col = 0
            sys.stdout.write(ch)
            _response_col += 1
    sys.stdout.flush()


def print_response_end() -> None:
    w = _term_width()
    sys.stdout.write(f"\n{STYLE_ASSISTANT}╰{'─' * (w - 2)}╯{_RESET}\n\n")
    sys.stdout.flush()


def print_stats(
    elapsed: float,
    prompt_tok: int,
    eval_tok: int,
    context_used: int,
    context_window: int,
) -> None:
    rate = eval_tok / elapsed if elapsed > 0 else 0
    bar = context_bar(context_used, context_window)
    print(
        f"{STYLE_STATS}  {elapsed:.1f}s  ·  {prompt_tok:,} prompt tok  ·  "
        f"{eval_tok:,} generated tok  ·  {rate:.0f} tok/s{_RESET}"
    )
    print(f"{STYLE_STATS}  context  {_RESET}{bar}\n")
