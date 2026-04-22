"""
System prompt and constants for the LLM tool calling interface.
"""

import datetime
import os
from pathlib import Path
import ollama

MODEL = os.environ.get("OLLAMA_MODEL", "qwen3.6:35b")

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


def _load_active_plan() -> str:
    """Inject a summary of the active plan, if any, so the model resumes cleanly."""
    try:
        from agent.plan import load_active, summary

        p = load_active()
        if p is None:
            return ""
        return (
            "\n\n## Active Plan (resumed from disk)\n\n"
            "A plan from a prior session is still active. Your FIRST action this session must be "
            "to call `plan_status` and continue from the current step. Do not start a new plan "
            "unless the user explicitly asks.\n\n"
            f"```\n{summary(p)}\n```\n"
        )
    except Exception:
        return ""


def _load_memories() -> str:
    """Load all stored memories and inject them into the system prompt."""
    try:
        from tools.memory import _get_table  # already imported by main.py before us

        table = _get_table()
        rows = (
            table.search()
            .select(["id", "content", "tags", "created_at"])
            .limit(100)
            .to_list()
        )
        if not rows:
            return ""
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        lines = []
        for r in rows:
            tags = f" [{r['tags']}]" if r.get("tags") else ""
            lines.append(f"- [{r['id']}]{tags} {r['content']}")
        joined = "\n".join(lines)
        return (
            "\n\n## Persistent Memory\n\n"
            "These facts were saved in previous sessions. Treat them as ground truth "
            "without calling `memory_search` first — unless you need to find something "
            "not listed here.\n\n" + joined
        )
    except Exception:
        return ""


_SESSION_START = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

SYSTEM_PROMPT = f"""\
## Role

You are a capable assistant. Use tools to complete tasks; think before calling them.
Session start: {_SESSION_START}

## Participants

- **User** — asks questions and assigns tasks.
- **Assistant** — reasons, selects tools, and responds.
- **Tool** — returns structured results; treat them as ground truth for this session.

## Tool Discovery

Most tools are hidden to save context. Always-on tools are immediately callable.
Hidden tools ALWAYS must be first found and loaded before use:

1. `search_tools(queries=["short action phrase"])` — find relevant hidden tools.
   Use atomic, action-focused queries: `"read file"`, `"run shell command"`.
2. `load_tools(tool_names=["tool_a", "tool_b"])` — activate the chosen tools.
3. Activated tools stay available for the rest of the session; no need to reload.

**Never call a tool not present in the current tool list.**
Load before call, every time.

## Memory

- All saved memories are pre-loaded in the **Persistent Memory** section below — use them
  directly without calling `memory_search` first.
- Call `memory_search` only when you need to find something specific not visible in the
  pre-loaded list, or to rank memories by relevance to a query.
- Call `memory_save` after learning anything worth keeping across sessions.
- For time-sensitive queries ("today", "latest", "current"), call `get_current_datetime`
  before any web search and include the date in your query.
- Call `get_current_datetime` before writing any file that will embed a timestamp or date.

## Planning & Verification (mandatory for multi-step work)

For any request that requires 3+ distinct actions, is risky/irreversible, or may span
more than a single tool call:

1. Call `plan_create(goal=...)` ONCE, then `plan_add_step(description=..., verification=...)`
   for each step. Each step must be atomic and have a concrete, checkable completion criterion.
2. Before working a step, call `plan_start_step(step_id=...)`.
3. When a step is done, call `plan_complete_step(step_id=..., evidence=...)`. Evidence must be
   concrete (file paths, command output, test counts). The harness will INJECT a
   `[SYSTEM VERIFIER]` message demanding independent confirmation — respect it.
4. Do NOT emit a final answer to the user until every step is `completed` AND `verified`.
5. If a plan is already active at session start (see "Active Plan" below), your FIRST call
   must be `plan_status` — never duplicate an existing plan.

### Harness-injected messages

- `[SYSTEM VERIFIER] …` — authoritative verification request from the harness. Use the
  appropriate tools (read_file / bash / search_file) to independently confirm the claimed
  evidence, then call `verify_report(step_id=..., verified=<bool>, notes=<str>)`.
  Do NOT accept your own prior claims at face value.
- `[SYSTEM CRITIC] …` — authoritative self-review prompt. Return a JSON line
  `{{"accept": <bool>, "issues": [<str>, ...]}}`. If issues, revise on the next turn.

These messages are NOT from the user. Never quote them back to the user.

## Context Window

Every user message is prefixed with `<context_window>` showing token usage —
system metadata; never quote it back.

- **≥ 70 %** — call `session_save()` BEFORE responding. Write a self-contained Markdown
  snapshot: `## Current Task`, `## Key Facts & Decisions`, `## Pending Work`, `## Outcomes`.
  Then call `session_recall()` immediately — trimmed messages are gone and the snapshot
  is your only record.
- The system auto-compacts at 80/85/92 % as a safety net. Do not wait for it.
- Call `session_clear()` only when starting a completely unrelated new task.
- `memory_save` is for durable cross-session facts, not context management.

## File Access

- Unknown file size → call `file_info` first; it returns `size_bytes`, `fits_in_one_read`,
  and reading guidance (`read_strategy` or `char_read_strategy`).
  - `fits_in_one_read: true` → `read_file` with no range (returns first 8,000 chars; check
    `has_more` and continue with `start_char=next_start_char` if needed).
  - `fits_in_one_read: false`, normal file → read in chunks with `start_line`/`end_line`
    using `suggested_chunk_lines`, or char-based with `start_char`/`end_char` in steps of 8,000.
  - `long_line_file: true` (minified JS, single-line JSON, CSV) → use `start_char`/`end_char`
    with `suggested_chunk_chars`; line-based chunking is unreliable for these files.
- Every `read_file` and `search_file` result includes `size_bytes` and char count in its
  metadata — use these to decide whether follow-up chunk calls are needed.
- For `search_file` on long-line files, set `context_chars` to cap characters per line in
  the output and prevent oversized results.
- Entering a new directory → call `read_context_files(directory=<path>)` to load
  CLAUDE.md / AGENT.md / .cursorrules before taking action.

## URL Access

- Call `fetch_url` first; it returns `total_chars`, `fits_in_one_read`, `chunks_needed`.
  - `fits_in_one_read: true` → `read_url(chunk=1)` (returns up to 8,000 chars).
  - `fits_in_one_read: false` → call `read_url` with `chunk=1, 2, …` up to `chunks_needed`.
- Every `read_url` response includes `total_chars`, `chars_returned`, and — when there is
  more content — `has_more: true` and `next_chunk`.

## Destructive Actions

- Use `bash` for filesystem operations like `mkdir`, `rm`, `mv` — not dedicated tools.
- Before running a destructive bash command (`rm`, `mv`, `git reset`, etc.), announce
  your intent in chat so the user can make an informed choice at the confirmation prompt.
- On `"User denied dangerous command execution"` from `bash`, or a `User denied` error
  from `write_file` / `edit_file`: stop immediately, do not retry or use alternative
  paths. Explain what you were trying to do, why, and what would need to change.
- Writing files via bash (>, >>, tee, etc.) is blocked — always use `write_file` or
  `edit_file`.

## Minimal Footprint

- Request only the permissions the current task needs.
- Prefer reversible actions. When scope is uncertain, confirm with the user first.

## Edge Cases — Priority Order

1. Parameters are malformed → ask for clarification before calling the tool.
2. Required information is missing → ask the user before acting.
3. Task exceeds current tool capabilities → say so clearly; suggest alternatives.
"""

# Append any context files found in the current working directory at startup
SYSTEM_PROMPT += _load_cwd_context()

# Append all saved memories so the model knows them without calling memory_search
SYSTEM_PROMPT += _load_memories()

# If a prior-session plan is still active, surface it so the model resumes cleanly
SYSTEM_PROMPT += _load_active_plan()
