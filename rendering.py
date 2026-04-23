"""
Rendering helpers for the LLM tool calling interface.
"""

import sys
import shutil
import textwrap
from tools import schemas

from rich.console import Console as _RichConsole
from rich.markdown import Markdown as _Markdown

from context_window import (
    _ansi,
    _IS_TTY,
    _RESET,
    _BOLD,
    _DIM,
    _ITALIC,
    _FG_RED,
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
        f"{_DIM}Type  {_RESET}{_BOLD}exit{_RESET}{_DIM}  to quit  ·  {_RESET}{_BOLD}Ctrl-C{_RESET}{_DIM}  during generation to interrupt{_RESET}\n"
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


# Orchestrator event styling — distinct from tool calls/results so the user
# visibly sees harness-driven scrutiny (triage, verifier injection, critic).
_STYLE_ORCH = _FG_RED

_ORCH_ICONS = {
    "triage": "◇",
    "verifier-queued": "⧗",
    "verifier": "⟳",
    "plan-nudge": "↻",
    "snapshot-nudge": "⤓",
    "critic": "⟳",
    "critic-revise": "⟲",
    "interrupted": "✕",
}


def print_orchestrator_event(kind: str, detail: str = "") -> None:
    """One-line notice for a harness-driven event (verifier/critic/nudge/triage)."""
    icon = _ORCH_ICONS.get(kind, "•")
    tail = f"  {detail}" if detail else ""
    print(f"  {_STYLE_ORCH}{icon} {kind}{tail}{_RESET}")


class CLIRenderer:
    """Adapter: satisfies the Orchestrator's duck-typed Renderer protocol by
    delegating to the print_* module-level functions in this file."""

    def thinking_start(self):
        print_thinking_start()

    def thinking_token(self, text: str):
        print_thinking_token(text)

    def thinking_end(self):
        print_thinking_end()

    def response_start(self):
        print_response_start()

    def response_token(self, text: str):
        print_response_token(text)

    def response_end(self):
        print_response_end()

    def tool_call(self, name: str, args: dict):
        print_tool_call(name, args)

    def tool_result(self, result: str):
        print_tool_result(result)

    def stats(self, elapsed, pt, et, cu, cw):
        print_stats(elapsed, pt, et, cu, cw)

    def orchestrator_event(self, kind: str, detail: str = ""):
        print_orchestrator_event(kind, detail)

    def blank_line(self):
        print()


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
    console = _RichConsole(highlight=False, width=w - 4)
    console.print(_Markdown(text or ""))
    print(f"{STYLE_ASSISTANT}╰{'─' * (w - 2)}╯{_RESET}\n")


# --- Streaming response helpers ---
# Buffer tokens during streaming (single-line progress, no scroll-buffer pollution),
# then render complete markdown once at the end.

_response_buffer: str = ""


def print_response_start() -> None:
    global _response_buffer
    _response_buffer = ""
    sys.stdout.write("\n")
    sys.stdout.flush()


def print_response_token(text: str) -> None:
    global _response_buffer
    _response_buffer += text
    sys.stdout.write(f"\r  {_DIM}▸ {len(_response_buffer)} chars…{_RESET}")
    sys.stdout.flush()


def print_response_end() -> None:
    global _response_buffer
    w = _term_width()
    sys.stdout.write(f"\r\033[K")  # erase progress line
    sys.stdout.flush()
    print(f"{STYLE_ASSISTANT}╭─ assistant {'─' * (w - 14)}╮{_RESET}")
    console = _RichConsole(highlight=False, width=w - 4)
    console.print(_Markdown(_response_buffer))
    sys.stdout.write(f"{STYLE_ASSISTANT}╰{'─' * (w - 2)}╯{_RESET}\n\n")
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
