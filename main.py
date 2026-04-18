#!/usr/bin/env python3
"""
Main chat loop for the LLM tool calling interface.
"""

import os
import sys
import time
import shutil
import textwrap


# Load environment variables
def _load_dotenv(path: str = ".env") -> None:
    """Load KEY=VALUE pairs from *path* into os.environ (existing vars are not overwritten)."""
    try:
        with open(path, encoding="utf-8") as fh:
            for raw in fh:
                line = raw.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, _, value = line.partition("=")
                # Strip optional surrounding quotes
                value = value.strip().strip("\"'")
                os.environ.setdefault(key.strip(), value)
    except FileNotFoundError:
        pass


_load_dotenv()

# Import tools
import ollama
import tools.builtins  # noqa: F401  — registers built-in tools
import tools.file_tools  # noqa: F401  — registers read_file, file_info, search_file, write_file, write_json, edit_file
import tools.filesystem  # noqa: F401  — registers list_directory, make_directory, remove_file, get_working_context
import tools.memory  # noqa: F401  — registers memory_save, memory_search, memory_list, memory_delete
import tools.notebooklm  # noqa: F401  — registers notebooklm_* tools
import tools.session  # noqa: F401  — registers session_save, session_recall, session_clear
import tools.web  # noqa: F401  — registers web_search, fetch_url, read_url
import tools.tool_index  # noqa: F401  — registers search_tools, load_tools
from tools import call, schemas
from tools.tool_index import build_index

build_index()

from system_prompt import SYSTEM_PROMPT, MODEL
from rendering import (
    print_header,
    print_thinking_start,
    print_thinking_token,
    print_thinking_end,
    print_response_start,
    print_response_token,
    print_response_end,
    print_tool_call,
    print_tool_result,
    print_stats,
    hr,
    _term_width,
    STYLE_USER,
    STYLE_ASSISTANT,
    STYLE_TOOL_CALL,
    STYLE_TOOL_RES,
    STYLE_THINK,
    STYLE_STATS,
    STYLE_HEADER,
    _RESET,
)
from context_window import (
    get_context_window,
    context_bar,
    context_prefix,
    trim_messages,
    offload,
    warmup,
    init_scratch_dir,
    cleanup_scratch,
    maybe_offload_result,
    is_scratch_path,
    compact_messages,
)


def chat() -> None:
    # Initialize session
    tools.session._clear_session_file()
    warmup()
    init_scratch_dir()
    context_window = get_context_window()
    print_header(context_window)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    context_used = 0  # updated after each Ollama response; persists across turns

    while True:
        try:
            prompt = f"{STYLE_USER}You:{_RESET} "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{STYLE_STATS}Bye.{_RESET}")
            offload()
            cleanup_scratch()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{STYLE_STATS}Bye.{_RESET}")
            offload()
            cleanup_scratch()
            break

        # Prepend context usage so the LLM knows how full the window is
        messages.append(
            {
                "role": "user",
                "content": context_prefix(context_used, context_window) + user_input,
            }
        )
        print()

        prompt_tokens = 0  # accumulated evaluation cost for this turn
        eval_tokens = 0
        t_start = time.perf_counter()

        while True:
            compact_messages(messages, context_used, context_window)
            stream = ollama.chat(
                model=MODEL,
                messages=messages,
                tools=schemas(),
                think=True,
                stream=True,
                keep_alive="15m",
            )

            thinking_open = False
            response_open = False
            thinking_parts: list[str] = []
            content_parts: list[str] = []
            msg_tool_calls: list = []

            for chunk in stream:
                cmsg = chunk.message
                if cmsg.thinking:
                    if not thinking_open:
                        print_thinking_start()
                        thinking_open = True
                    print_thinking_token(cmsg.thinking)
                    thinking_parts.append(cmsg.thinking)
                if cmsg.content:
                    if thinking_open:
                        print_thinking_end()
                        thinking_open = False
                    if not response_open:
                        print_response_start()
                        response_open = True
                    print_response_token(cmsg.content)
                    content_parts.append(cmsg.content)
                if getattr(cmsg, "tool_calls", None):
                    msg_tool_calls = cmsg.tool_calls
                if chunk.done:
                    last_prompt = getattr(chunk, "prompt_eval_count", 0) or 0
                    prompt_tokens += last_prompt
                    eval_tokens += getattr(chunk, "eval_count", 0) or 0
                    context_used = last_prompt

            if thinking_open:
                print_thinking_end()
            if response_open:
                print_response_end()

            full_content = "".join(content_parts)

            messages.append(
                {
                    "role": "assistant",
                    "content": full_content,
                    "tool_calls": msg_tool_calls or [],
                }
            )

            if not msg_tool_calls:
                elapsed = time.perf_counter() - t_start
                print_stats(
                    elapsed, prompt_tokens, eval_tokens, context_used, context_window
                )
                break

            for tc in msg_tool_calls:
                fn_name = tc.function.name
                fn_args = tc.function.arguments
                print_tool_call(fn_name, fn_args)

                result = call(fn_name, fn_args)
                result_str = str(result)
                # Never offload reads of scratch files — they ARE the offloaded content.
                # Offloading them again creates an infinite redirect cycle.
                if not (
                    fn_name == "read_file" and is_scratch_path(fn_args.get("path", ""))
                ):
                    result_str = maybe_offload_result(fn_name, result_str)
                print_tool_result(result_str)

                messages.append({"role": "tool", "content": result_str})

                # Auto-trim context after a successful session snapshot
                if fn_name == "session_save":
                    try:
                        import json as _json

                        if _json.loads(result_str).get("saved"):
                            trim_messages(messages)
                            context_used = (
                                0  # window shrank; reset until next Ollama response
                            )
                    except Exception:
                        pass


if __name__ == "__main__":
    chat()
