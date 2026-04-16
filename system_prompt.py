"""
System prompt and constants for the LLM tool calling interface.
"""

import datetime
import os
from pathlib import Path
import ollama

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.5:35b")

# Context files to scan for project/agent instructions
_CONTEXT_FILES = [
    "CLAUDE.md",
    "AGENT.md",
    "AGENTS.md",
    ".cursorrules",
    ".windsurfrules",
    ".clinerules",
]


def _load_cwd_context() -> str:
    """
    Walk from CWD up to filesystem root, collect any agent context files, and
    return them formatted as a Markdown block ready to append to the system
    prompt.  Returns an empty string if nothing is found.
    """
    start = Path.cwd()
    dirs: list[Path] = []
    current = start
    while True:
        dirs.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    dirs.reverse()  # outermost first

    sections: list[str] = []
    for d in dirs:
        for name in _CONTEXT_FILES:
            path = d / name
            if path.is_file():
                try:
                    content = path.read_text(encoding="utf-8").strip()
                    if content:
                        sections.append(f"### {path}\n\n{content}")
                except Exception:
                    pass

    if not sections:
        return ""

    joined = "\n\n---\n\n".join(sections)
    return (
        "\n\n## Project / Directory Context\n\n"
        "The following files were found in the current working directory (or its "
        "parents) and contain project-specific instructions. Follow them.\n\n" + joined
    )


_SESSION_START = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

SYSTEM_PROMPT = (
    f"Current date and time (session start): {_SESSION_START}\n\n"
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
    "- Before writing any file whose name or content will include a timestamp, date, "
    "or time value, ALWAYS call get_current_datetime first (unless the session-start "
    "time above is recent enough).\n"
    "- CONTEXT-WINDOW RULE (highest priority — check this every turn):\n"
    "  • Every user message is prefixed with a <context_window> tag showing token usage. "
    "This is system metadata — never repeat or quote it back to the user.\n"
    "  • When <context_window> shows usage ≥70%: call session_save() BEFORE responding. "
    "session_save is the ONLY correct tool for this — NOT memory_save, NOT write_file. "
    "Write a self-contained Markdown summary: ## Current Task, ## Key Facts & Decisions, "
    "## Pending Work, ## Outcomes. Then immediately call session_recall() — "
    "old messages will have been trimmed and the snapshot is now your only record.\n"
    "  • The system auto-compacts at 80/85/92% as a safety net. Do not wait for it.\n"
    "  • Call session_clear() when starting a completely new, unrelated task.\n"
    "  • memory_save is for durable facts across future sessions, NOT for context management.\n"
    "- When a tool result begins with '[Output offloaded:', the full content was "
    "automatically saved to a scratch file to protect the context window. "
    "The result includes the file path and a short preview. "
    "Do NOT re-run the original tool — use read_file(path) to retrieve specific "
    "sections, or search_file(path, pattern) to locate content by regex. "
    "The offloaded file persists for the entire session.\n"
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
    "up to 'chunks_needed'. Each response includes 'next_chunk' as a reminder.\n"
    "- When you start working inside a directory you haven't visited before "
    "(e.g. after cd-ing via bash), call read_context_files(directory=<path>) "
    "to load any CLAUDE.md / AGENT.md / .cursorrules etc. before taking action. "
    "Honour any project-specific instructions found there."
)

# Append any context files found in the current working directory at startup
SYSTEM_PROMPT += _load_cwd_context()
