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
import tools.files  # noqa: F401  — registers write_file, edit_file, read_json, write_json
import tools.filesystem  # noqa: F401  — registers list_directory, make_directory, remove_file, get_working_context
import tools.memory  # noqa: F401  — registers memory_save, memory_search, memory_list, memory_delete
import tools.notebooklm  # noqa: F401  — registers notebooklm_* tools
import tools.web  # noqa: F401  — registers web_search, read_file, file_info, search_file, read_pdf, fetch_url, read_url
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
    "and include the current date in your query.\n"
    "- If write_file, edit_file, make_directory, or remove_file returns an error "
    "containing 'User denied', stop immediately — do not retry, do not attempt "
    "alternative paths, do not make any further file changes. Instead, explain in "
    "plain language: what you were trying to do, why, and what would need to change "
    "for you to proceed.\n"
    "- Before calling make_directory or remove_file, always tell the user what you "
    "are about to do and why. These tools will show a confirmation prompt, but you "
    "should explain your intent in chat first so the user can make an informed choice.\n"
    "- When working with a file whose size you don't already know, ALWAYS call "
    "file_info first. Then follow the returned 'fits_in_one_read' field:\n"
    "  • fits_in_one_read = true  → call read_file with no line range.\n"
    "  • fits_in_one_read = false → follow the 'read_strategy' advice in the response: "
    "either call search_file to locate the relevant section and read only those lines, "
    "or read the file in sequential chunks using 'suggested_chunk_lines' as the chunk "
    "size. Never attempt to read an entire large file in one call.\n"
    "- When you need to read a web page by URL, ALWAYS call fetch_url first. "
    "Then follow the same 'fits_in_one_read' logic:\n"
    "  • fits_in_one_read = true  → call read_url with chunk=1.\n"
    "  • fits_in_one_read = false → call read_url repeatedly with chunk=1, 2, … "
    "up to 'chunks_needed'. Each response includes 'next_chunk' as a reminder."
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
_CTX_BAR_WIDTH = 24  # characters in the context fill bar


def _term_width() -> int:
    return shutil.get_terminal_size((100, 24)).columns


# ---------------------------------------------------------------------------
# Context window helpers
# ---------------------------------------------------------------------------


def _get_context_window() -> int:
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


def _context_bar(used: int, total: int) -> str:
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


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def _hr(char: str = "─", style: str = _DIM) -> str:
    return f"{style}{char * _term_width()}{_RESET}"


def _print_header(context_window: int) -> None:
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
    print(f"{_DIM}Context window:{_RESET}  {context_window:,} tokens")
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


def _print_stats(
    elapsed: float,
    prompt_tok: int,
    eval_tok: int,
    context_used: int,
    context_window: int,
) -> None:
    rate = eval_tok / elapsed if elapsed > 0 else 0
    bar = _context_bar(context_used, context_window)
    print(
        f"{_STYLE_STATS}  {elapsed:.1f}s  ·  {prompt_tok:,} prompt tok  ·  "
        f"{eval_tok:,} generated tok  ·  {rate:.0f} tok/s{_RESET}"
    )
    print(f"{_STYLE_STATS}  context  {_RESET}{bar}\n")


# ---------------------------------------------------------------------------
# Warmup
# ---------------------------------------------------------------------------


def _offload() -> None:
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
    context_window = _get_context_window()
    _print_header(context_window)

    messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]

    while True:
        try:
            prompt = f"{_STYLE_USER}You:{_RESET} "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{_DIM}Bye.{_RESET}")
            _offload()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{_DIM}Bye.{_RESET}")
            _offload()
            break

        messages.append({"role": "user", "content": user_input})
        print()

        prompt_tokens = 0  # accumulated evaluation cost for this turn
        eval_tokens = 0
        context_used = 0  # last prompt_eval_count = actual context fill
        t_start = time.perf_counter()

        while True:
            response = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=schemas(),
                think=True,
            )
            msg = response.message
            last_prompt = getattr(response, "prompt_eval_count", 0) or 0
            prompt_tokens += last_prompt
            eval_tokens += getattr(response, "eval_count", 0) or 0
            context_used = last_prompt  # most recent value = current fill level

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
                _print_stats(
                    elapsed, prompt_tokens, eval_tokens, context_used, context_window
                )
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
