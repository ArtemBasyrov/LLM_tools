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
import tools.plan_tools  # noqa: F401  — registers plan_create, plan_add_step, plan_start_step, plan_complete_step, plan_status, plan_log, plan_abandon
import tools.session  # noqa: F401  — registers session_save, session_recall, session_clear
import tools.verify_tools  # noqa: F401  — registers verify_report
import tools.web  # noqa: F401  — registers web_search, fetch_url, read_url
import tools.tool_index  # noqa: F401  — registers search_tools, load_tools
from tools import call, schemas
from tools.tool_index import build_index

build_index()

from system_prompt import SYSTEM_PROMPT, MODEL
from rendering import (
    print_header,
    CLIRenderer,
    STYLE_USER,
    STYLE_STATS,
    _RESET,
)
from context_window import (
    get_context_window,
    offload,
    warmup,
)
from agent.orchestrator import Orchestrator


def chat() -> None:
    # Initialize session
    tools.session._clear_session_file()
    warmup(SYSTEM_PROMPT)
    context_window = get_context_window()
    print_header(context_window)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    renderer = CLIRenderer()
    orchestrator = Orchestrator(
        messages=messages,
        context_window=context_window,
        model=MODEL,
        renderer=renderer,
    )

    while True:
        try:
            prompt = f"{STYLE_USER}You:{_RESET} "
            user_input = input(prompt).strip()
        except (EOFError, KeyboardInterrupt):
            print(f"\n{STYLE_STATS}Bye.{_RESET}")
            offload()
            break

        if not user_input:
            continue
        if user_input.lower() in ("exit", "quit"):
            print(f"{STYLE_STATS}Bye.{_RESET}")
            offload()
            break

        try:
            orchestrator.turn(user_input)
        except KeyboardInterrupt:
            print(f"\n{STYLE_STATS}[interrupted]{_RESET}")
            continue


if __name__ == "__main__":
    chat()
