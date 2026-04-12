"""
Tool-calling chat loop for a local Ollama model.

Run:
    micromamba run -n internet python chat.py
"""

import shutil
import sys
import textwrap
import time

import ollama
import tools.builtins  # noqa: F401  — registers built-in tools
import tools.files  # noqa: F401  — registers write_file and edit_file
import tools.memory  # noqa: F401  — registers memory_save, memory_search, memory_list, memory_delete
import tools.notebooklm  # noqa: F401  — registers notebooklm_* tools
import tools.web  # noqa: F401  — registers web_search and read_file
from tools import call, schemas

MODEL = "qwen3.5:35b"

_SYSTEM_PROMPT = (
    "You are a helpful assistant with access to tools. "
    "You have a persistent semantic memory — use it proactively:\n"
    "- Call memory_search at the start of any conversation that may involve "
    "previously stored facts, user preferences, or past decisions.\n"
    "- Call memory_save whenever you learn something worth remembering across sessions "
    "(facts, preferences, decisions, observations).\n"
    "- When answering questions that involve time-sensitive or current information "
    "(e.g. 'right now', 'today', 'currently', 'latest', 'recent'), "
    "always call get_current_datetime FIRST before any web search, "
    "and include the current date in your query."
)

# ---------------------------------------------------------------------------
# ANSI helpers
# ---------------------------------------------------------------------------

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
_STYLE_USER = _BOLD + _FG_GREEN
_STYLE_ASSISTANT = _BOLD + _FG_CYAN
_STYLE_TOOL_CALL = _FG_YELLOW
_STYLE_TOOL_RES = _FG_GRAY
_STYLE_THINK = _DIM
_STYLE_STATS = _DIM
_STYLE_HEADER = _BOLD + _FG_BLUE

_TOOL_RESULT_MAX = 300  # chars shown in tool-result lines


def _term_width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _hr(char: str = "─", style: str = _DIM) -> str:
    return f"{style}{char * _term_width()}{_RESET}"


def _print_header() -> None:
    w = _term_width()
    title = f"  {MODEL}  "
    pad = (w - len(title)) // 2
    print()
    print(_hr("━", _STYLE_HEADER))
    print(f"{_STYLE_HEADER}{' ' * pad}{title}{_RESET}")
    print(_hr("━", _STYLE_HEADER))
    tools_list = "  ".join(
        f"{_FG_YELLOW}⬡{_RESET} {s['function']['name']}" for s in schemas()
    )
    print(f"\n{_DIM}Tools:{_RESET}  {tools_list}")
    print(
        f"{_DIM}Type  {_RESET}{_BOLD}exit{_RESET}{_DIM}  or press  {_RESET}{_BOLD}Ctrl-C{_RESET}{_DIM}  to quit{_RESET}\n"
    )
    print(_hr())
    print()


def _print_thinking(text: str) -> None:
    if not text or not text.strip():
        return
    w = _term_width()
    inner = w - 4
    label = " thinking "
    bar_left = (inner - len(label)) // 2
    bar_right = inner - len(label) - bar_left
    print(f"{_STYLE_THINK}┌─{label}{'─' * bar_right}─┐")
    for line in text.strip().splitlines():
        for chunk in textwrap.wrap(line, width=inner) or [""]:
            print(f"│ {chunk:<{inner}} │")
    print(f"└{'─' * (w - 2)}┘{_RESET}")
    print()


def _print_tool_call(name: str, args: dict) -> None:
    args_str = ", ".join(f"{k}={repr(v)[:60]}" for k, v in args.items())
    print(f"  {_STYLE_TOOL_CALL}⚙ {name}({args_str}){_RESET}")


def _print_tool_result(result: str) -> None:
    display = (
        result[:_TOOL_RESULT_MAX] + f"… ({len(result)} chars)"
        if len(result) > _TOOL_RESULT_MAX
        else result
    )
    # Indent every line of the result
    indented = "\n".join(f"    {line}" for line in display.splitlines())
    print(f"{_STYLE_TOOL_RES}{indented}{_RESET}\n")


def _print_response(text: str) -> None:
    w = _term_width()
    print(f"\n{_STYLE_ASSISTANT}╭─ assistant {'─' * (w - 14)}╮{_RESET}")
    for line in (text or "").splitlines():
        for chunk in textwrap.wrap(line, width=w - 4) or [""]:
            print(f"  {chunk}")
    print(f"{_STYLE_ASSISTANT}╰{'─' * (w - 2)}╯{_RESET}\n")


def _print_stats(elapsed: float, prompt_tok: int, eval_tok: int) -> None:
    rate = eval_tok / elapsed if elapsed > 0 else 0
    print(
        f"{_STYLE_STATS}  {elapsed:.1f}s  ·  {prompt_tok} prompt tok  ·  "
        f"{eval_tok} generated tok  ·  {rate:.0f} tok/s{_RESET}\n"
    )


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _warmup() -> None:
    msg = f"  Loading {MODEL}…"
    print(f"{_DIM}{msg}{_RESET}", end="", flush=True)
    try:
        ollama.chat(
            model=MODEL,
            messages=[{"role": "user", "content": "hi"}],
            think=False,
            options={"num_predict": 1},
        )
    except Exception:
        pass
    print(f"\r{' ' * len(msg)}\r", end="", flush=True)


# ---------------------------------------------------------------------------
# Main chat loop
# ---------------------------------------------------------------------------


def chat() -> None:
    _warmup()
    _print_header()

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    while True:
        try:
            prompt = f"{_STYLE_USER}You:{_RESET} "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Bye.{_RESET}")
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{_DIM}Bye.{_RESET}")
            break

        messages.append({"role": "user", "content": user_input})
        print()

        prompt_tokens = 0
        eval_tokens = 0
        t_start = time.perf_counter()

        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=schemas(),
                think=True,
            )
            msg = response.message
            prompt_tokens += getattr(response, "prompt_eval_count", 0) or 0
            eval_tokens += getattr(response, "eval_count", 0) or 0

            _print_thinking(getattr(msg, "thinking", None))

            messages.append(
                {
                    "role": "assistant",
                    "content": msg.content or "",
                    "tool_calls": msg.tool_calls or [],
                }
            )

            if not msg.tool_calls:
                elapsed = time.perf_counter() - t_start
                _print_response(msg.content)
                _print_stats(elapsed, prompt_tokens, eval_tokens)
                break

            for tc in msg.tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments
                _print_tool_call(fn_name, fn_args)

                result = call(fn_name, fn_args)
                result_str = str(result)
                _print_tool_result(result_str)

                messages.append({"role": "tool", "content": result_str})


if __name__ == "__main__":
    chat()
