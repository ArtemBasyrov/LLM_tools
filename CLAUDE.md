# LLM Tools

This project builds tools for a locally running LLM to use via function/tool calling.

## Target Model

- **Model**: configured via `OLLAMA_MODEL` env var (default: `qwen3.5:35b`, running locally via [Ollama](https://ollama.com))
- **Interface**: Ollama API, host configured via `OLLAMA_HOST` env var (default: `http://localhost:11434`)
- **Config**: copy `.env.example` → `.env` to set model and host

## Project Goal

Implement a suite of tools that the local LLM can invoke. Tools should be well-defined with clear schemas so the model can reliably select and call them.

## Design Principles

- Tools must have precise JSON schemas (name, description, parameters with types and descriptions)
- Each tool should do one thing well — avoid multi-purpose tools that confuse the model
- Keep descriptions concise and action-oriented; the model uses them to decide when to call a tool
- Prefer returning structured data over plain text where possible
- Tools should be fast — local LLMs are sensitive to latency in the tool-call loop

## Stack

- Python via micromamba environment `internet` — activate with:
  ```bash
  micromamba run -n internet python ...
  # or
  micromamba activate internet
  ```
- Ollama Python SDK or direct HTTP calls to `http://localhost:11434/api/chat`
- Tools implemented as plain Python functions with schema definitions alongside them

## Project Structure

```
.
├── main.py               # Main chat loop and Ollama integration
├── system_prompt.py      # System prompt and model configuration
├── rendering.py          # Display and formatting helpers
├── context_window.py     # Context window management + sticky-message compaction
├── agent/
│   ├── __init__.py       # Agentic package root
│   ├── orchestrator.py   # State machine wrapping each user turn
│   ├── plan.py           # Plan/Step dataclasses + disk I/O (JSON)
│   ├── triage.py         # Complexity heuristic (simple vs. complex)
│   ├── verifier.py       # [SYSTEM VERIFIER] injection builder
│   ├── critic.py         # [SYSTEM CRITIC] injection + JSON verdict parser
│   └── prompts.py        # Templated strings for all synthetic injections
├── tools/
│   ├── __init__.py       # Tool registry: register(), activate(), schemas()
│   ├── tool_index.py     # search_tools, load_tools — tool discovery meta-tools
│   ├── builtins.py       # get_current_datetime, calculate
│   ├── file_tools/       # split package — see File Tools section below
│   │   ├── _state.py     # session ledger: read mtimes, backup ring, open-files snapshot
│   │   ├── _helpers.py   # diffs, line-number gutter, near-miss scorer, confirm()
│   │   ├── read.py       # read_file (line-numbered, multi-file, outline), file_info, search_file
│   │   ├── edit.py       # edit_file (replace_all, anchor_line, batch atomic, near-miss hint)
│   │   ├── write.py      # write_file, write_json (probe-required, autoformat, syntax check)
│   │   ├── check.py      # check_syntax + shared _check_content used by edit/write
│   │   ├── format.py     # format_file (ruff/prettier/rustfmt/gofmt)
│   │   ├── patch.py      # apply_patch (unified-diff multi-file with rollback)
│   │   └── undo.py       # undo_last_edit, undo_list
│   ├── code_intel.py     # go_to_definition, find_references (Python AST)
│   ├── filesystem.py     # bash, read_context_files, list_directory, …
│   ├── find.py           # find_files (typed glob with hidden-dir guard)
│   ├── git_tools.py      # git_status, git_diff, git_blame
│   ├── memory.py         # memory_save, memory_search, memory_list, memory_delete
│   ├── notebooklm.py     # notebooklm_* tools
│   ├── plan_tools.py     # plan_create, plan_add_step, plan_start_step,
│   │                     #   plan_complete_step, plan_status, plan_log, plan_abandon
│   ├── session.py        # session_save, session_recall, session_clear
│   ├── test_runner.py    # run_tests (pytest with capped output)
│   ├── verify_tools.py   # verify_report (called during VERIFYING phase)
│   └── web.py            # web_search, fetch_url, read_url
├── tests/                # pytest suite (run: micromamba run -n internet python -m pytest tests/ -v)
├── CLAUDE.md
└── README.md
```

## Tool Modules

Tools are split into **always-on** (visible every turn) and **hidden** (discovered via `search_tools` / activated via `load_tools`).

### Always-on tools

| Tool | Module | Description |
|------|--------|-------------|
| `search_tools` | `tool_index.py` | Semantic search over all available tools |
| `load_tools` | `tool_index.py` | Activate tools returned by search_tools |
| `get_current_datetime` | `builtins.py` | Current local date and time |
| `calculate` | `builtins.py` | Safe arithmetic expression evaluator |
| `web_search` | `web.py` | DuckDuckGo search |
| `memory_save` | `memory.py` | Save a fact to persistent semantic memory |
| `memory_search` | `memory.py` | Retrieve relevant memories by query |
| `session_save` | `session.py` | Snapshot context for window management |
| `session_recall` | `session.py` | Reload snapshot after trim |
| `session_clear` | `session.py` | Erase the current snapshot |
| `plan_create` | `plan_tools.py` | Start a new plan with a goal (errors if one is active unless `replace=True`) |
| `plan_add_step` | `plan_tools.py` | Append an atomic step with `description` + `verification` criterion |
| `plan_start_step` | `plan_tools.py` | Transition a pending step to `in_progress` |
| `plan_complete_step` | `plan_tools.py` | Mark a step complete with `evidence`; triggers `[SYSTEM VERIFIER]` injection |
| `plan_status` | `plan_tools.py` | Inspect the active plan's full state |
| `plan_log` | `plan_tools.py` | Append a note to a step |
| `plan_abandon` | `plan_tools.py` | Archive the active plan with a reason |
| `verify_report` | `verify_tools.py` | Report the outcome of a verification; `verified=false` rolls the step back |

### Hidden tools (load on demand)

#### File I/O & navigation

| Tool | Module | Description |
|------|--------|-------------|
| `read_file` | `file_tools/read.py` | Read with **line-numbered output** (`   42 | …`), records on-disk mtime in the session ledger. Supports `paths=[…]` multi-file, `outline=true` for AST-based symbol-only output, `start_char/end_char` and `start_line/end_line` chunking, and `key_path` for JSON. |
| `file_info` | `file_tools/read.py` | Metadata (`size_bytes`, `line_count`, `fits_in_one_read`) + chunking strategy. Call before `read_file` on unknown files. |
| `search_file` | `file_tools/read.py` | Regex search within one file with line numbers; use `context_chars` for minified/long-line files. |
| `write_file` | `file_tools/write.py` | Create or overwrite. **Refuses** to overwrite an existing file that hasn't been read this session unless `overwrite=true`. Honors the staleness ledger. Optional `autoformat=true`. Returns a syntax-check verdict. Backup captured for `undo_last_edit`. |
| `write_json` | `file_tools/write.py` | Same safety rules as `write_file`; validates and reformats JSON before writing. |
| `edit_file` | `file_tools/edit.py` | Replace `old_string` with `new_string`. Supports `replace_all`, `anchor_line` (disambiguator when `old_string` is non-unique), and atomic `edits=[{old, new}]` batch with rollback. Returns a **near-miss hint** with line numbers when `old_string` is not found. Stale-read guard against external changes. Auto syntax check + ±5-line post-edit preview. |
| `apply_patch` | `file_tools/patch.py` | Apply a unified diff across one or more files atomically (uses system `patch`). Per-file stale check; rolls back on failure. |
| `undo_last_edit` | `file_tools/undo.py` | Revert the most recent `write_file`/`edit_file`/`format_file`/`apply_patch`. New-file creations are undone by deletion. |
| `undo_list` | `file_tools/undo.py` | List in-session backups (newest first). |
| `format_file` | `file_tools/format.py` | Run `ruff format` / `prettier` / `rustfmt` / `gofmt` based on extension. Backup captured. |
| `check_syntax` | `file_tools/check.py` | Per-extension static check: `py_compile` + `ruff` (.py), strict parser (.json/.yaml/.toml). Auto-invoked by `write_file`/`edit_file`. |
| `find_files` | `find.py` | Typed glob with hard cap (500); skips `node_modules`, `.venv`, `.git`, caches, etc. |

#### Shell / context / code intelligence

| Tool | Module | Description |
|------|--------|-------------|
| `bash` | `filesystem.py` | Run a shell command |
| `read_context_files` | `filesystem.py` | Load CLAUDE.md / AGENT.md from a directory and its parents |
| `go_to_definition` | `code_intel.py` | Locate a Python symbol's `def`/`class` line in a file or project (AST). |
| `find_references` | `code_intel.py` | List `Name`+`Attribute` usages of a Python symbol across a file or project (AST). |
| `run_tests` | `test_runner.py` | Run pytest scoped to a file/dir/nodeid. Output trimmed to last ~6K chars; uses `-x --tb=short`. |
| `git_status` | `git_tools.py` | Branch + porcelain v2 status as JSON. |
| `git_diff` | `git_tools.py` | Working tree, staged, single-file, or two-ref unified diff (capped at ~12K chars). |
| `git_blame` | `git_tools.py` | Author/sha/summary for a line range. |

#### Web

| Tool | Module | Description |
|------|--------|-------------|
| `fetch_url` | `web.py` | URL metadata (`total_chars`, `fits_in_one_read`, `chunks_needed`) + preview; call before `read_url` on unknown pages |
| `read_url` | `web.py` | Fetch a specific 8,000-char chunk of a URL; response includes `total_chars`, `has_more`, `next_chunk` |

#### Memory

| Tool | Module | Description |
|------|--------|-------------|
| `memory_list` | `memory.py` | List recent memories |
| `memory_delete` | `memory.py` | Delete a memory by ID |

#### NotebookLM

| Tool | Module | Description |
|------|--------|-------------|
| `notebooklm_create_notebook` | `notebooklm.py` | Create a NotebookLM notebook |
| `notebooklm_list_notebooks` | `notebooklm.py` | List notebooks |
| `notebooklm_add_source` | `notebooklm.py` | Add a source to a notebook |
| `notebooklm_list_sources` | `notebooklm.py` | List sources in a notebook |
| `notebooklm_ask` | `notebooklm.py` | Ask a question to a notebook |
| `notebooklm_generate` | `notebooklm.py` | Generate an artifact (summary, FAQ, etc.) |
| `notebooklm_list_artifacts` | `notebooklm.py` | List generated artifacts |
| `notebooklm_download` | `notebooklm.py` | Download an artifact |

## File Tools: feedback loop & safety

The file-tool stack is built around a per-session ledger (`tools/file_tools/_state.py`) that backs four guarantees the local model can rely on:

1. **Stale-read guard.** Every successful `read_file` records the on-disk
   mtime. `edit_file` / `write_file` / `apply_patch` refuse to act when the
   file has changed since the last read (tolerance ≈ 50 ms). Override with
   `skip_stale_check=true` (`edit_file`) or `overwrite=true` (`write_file`)
   only when explicitly intended.
2. **Probe-first overwrite.** `write_file` refuses to clobber an existing
   file that has never been read this session. Forces the model to ground
   its understanding before destroying content.
3. **Generate-and-check.** `write_file` and `edit_file` automatically run
   `_check_content` (py_compile + ruff for `.py`; strict parser for
   `.json` / `.yaml` / `.toml`) and return the verdict in their tool
   response. Lets a weak model self-correct without the user pointing out
   the syntax error.
4. **Backup ring + undo.** Every mutating op pushes a snapshot into a
   per-path ring (max 10 per file) and a global 64-deep order. `undo_last_edit`
   walks back through the global order; new-file creations are undone by
   deletion. Backup memory cap is 2 MB per file.

The orchestrator additionally maintains a single sticky `[SYSTEM FILES]`
message at the top of the inference window, refreshed every cycle from the
ledger. The model sees its current working set, ages, and any external-change
warnings before generating its next tool call.

When `_surgical_clear` evicts an old `read_file` tool result, it leaves a
typed pointer keyed by path (`Re-run read_file(path='…') to reload.`) so the
model knows exactly which call to repeat instead of re-deriving the path.

## Editing patterns the model is expected to follow

- **Read before editing.** `edit_file` enforces this — call `read_file` first
  so the staleness ledger has a reading to compare against.
- **Use `outline=true` to skim.** For modules longer than ~200 lines, prefer
  `read_file(path=…, outline=true)` to get a symbol map, then `start_line`/
  `end_line` to read the chunk you actually need.
- **Disambiguate, don't truncate context.** When `old_string` is non-unique,
  pick `anchor_line` over inventing fake surrounding context. For renames,
  use `replace_all=true`.
- **Batch atomically.** When multiple edits in one file go together, pass
  `edits=[{old, new}, …]` so a single failure rolls back the whole change.
- **React to checks.** If the response's `checks` field has a non-`ok`
  value, fix it in the next turn — the model should not "ship" a syntactically
  broken file.
- **Use `undo_last_edit` not `bash rm`** to revert mistakes.
- **Use `find_files` not `bash find`** so result sets stay capped and heavy
  dirs are skipped automatically.

## Agentic Mode

When `AGENTIC_MODE=true` (default), each turn runs through the
`Orchestrator` state machine in `agent/orchestrator.py`:

1. **Triage** — `agent/triage.py` classifies the request. Complex requests
   get a `[SYSTEM TRIAGE]` injection demanding the model draft a plan
   before answering.
2. **Plan** — plans are dataclasses (`agent/plan.py`) persisted as JSON at
   `~/.llm_plans/active.json` (override with `LLM_PLAN_DIR`). The active
   plan is loaded at startup so work can resume across sessions.
3. **Verify** — every `plan_complete_step` call queues a
   `[SYSTEM VERIFIER]` injection asking the model to independently confirm
   the claimed evidence (via `read_file` / `bash` / `search_file`) then
   call `verify_report(...)`. Steps with `verified=False` roll back to
   `in_progress`.
4. **Critic** — final responses run through up to `CRITIC_MAX_ROUNDS`
   rounds of self-review (`agent/critic.py`). Verdict is a JSON object on
   its own line: `{"accept": bool, "issues": [...]}`.
5. **Plan-nudge** — if the model emits a final response while the plan
   still has unverified steps, orchestrator injects
   `[SYSTEM ORCHESTRATOR]` nudging it back to work (capped at 2 per turn).
6. **Snapshot-nudge** — when context is ≥70% full AND a plan is active,
   orchestrator asks the model to call `session_save` so the plan
   context survives the next trim (fires once per turn).
7. **Files-refresh** — at the top of every inference cycle the
   orchestrator updates a single sticky `[SYSTEM FILES]` message with the
   current open-files registry from the file-tool ledger (path, size,
   age, staleness). Lets the model reason about its working set without
   re-reading.

### Conventions

- All harness-injected messages use the prefix `[SYSTEM <ROLE>]` — the
  system prompt teaches the model these are authoritative, not user
  speech.
- These messages are sticky: `context_window._is_sticky` preserves them
  across `_fast_prune` compaction.
- `Orchestrator` accepts injected `chat_fn`, `tool_call_fn`, `renderer`,
  etc., for testability. Tests use a scripted `FakeChat` (see
  `tests/test_orchestrator.py`).

### Relevant env vars

| Variable | Default | Effect |
|----------|---------|--------|
| `AGENTIC_MODE` | `true` | Master switch for the orchestrator |
| `ENFORCE_PLANNING` | `true` | Block final answer while plan incomplete |
| `CRITIC_MAX_ROUNDS` | `2` | Max critic revise rounds per final response |
| `LLM_PLAN_DIR` | `~/.llm_plans` | Where plan JSON lives |
| `CONSTRAIN_THINK` | `true` | Constrain `<think>` to GOAL/APPROACH/EDGE (3 lines); ~92% fewer thinking tokens, ~91% lower TTFC. Set `false` for unconstrained reasoning. |

### Tests

Run the suite in the `internet` micromamba env:
```bash
micromamba run -n internet python -m pytest tests/ -v
```

## Semantic Memory

`tools/memory.py` provides persistent cross-session memory via LanceDB + `all-MiniLM-L6-v2` embeddings.

- **Data dir**: `~/.llm_memory/` (override with `LLM_MEMORY_DIR` env var)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~22 MB, downloaded on first use
- **Search**: cosine similarity, returns `relevance` score (0–1)
- The system prompt instructs the model to search memory at session start and save notable facts

## Tool Description Best Practices

Writing effective descriptions is critical — the model uses them to decide when and how to call a tool.

### Description template
```
"Tool to <do X>. Use when <Y scenario happens>."
```
Example: `"Book flight tickets after confirming the user's requirements (time, departure, destination, party size, airline, etc.)"`

### Rules

- **Atomic scope** — one tool, one operation. `copy_file` + `move_file` + `delete_file` beats a monolithic `manage_files`.
- **"Goldilocks" specificity** — state preconditions and impactful limits (e.g., `"max 750 lines"`); skip minor ones. Keep under 1,024 characters.
- **Action-oriented names** — use verbs: `search_documents`, `get_weather_data`. Avoid abbreviations (`api_get_user`, not `u_g_a`). Use consistent casing (`snake_case`) — inconsistency implies hierarchy to the model.
- **Document hidden parameter rules** — if mutually optional params have a required-at-least-one constraint, say so explicitly: `"At least one of agent_id | user_id required"`.
- **Inline examples in param descriptions** — e.g., `'from:user@example.com is:unread'` for a query filter. For complex types, include a minimal valid example of the required structure.
- **Use enums for finite values** — don't leave categorical options as ambiguous prose.
- **Don't over-document** — avoid listing every edge case; context rot confuses the model. Focus on the most impactful constraints only.

### Examples in tool descriptions

Provide examples at two levels:

**1. Inline in parameter descriptions** — tiny, actionable strings showing exact expected format. Keep them immediately after the param description so the model sees them at decision time. Example:
```python
"description": "Dot-separated path to a nested JSON value. e.g. 'users.0.email' or 'config.timeout'"
```

**2. Few-shot examples in the tool description** — 2–5 canonical examples using a consistent **Input → Output** or **Use case → Call** pattern. Cover:
- A simple happy-path case
- A hard or ambiguous request  
- An edge case or "refrain" scenario (where the tool should NOT be called)

Keep few-shot examples inside the description string, or in the system prompt if they are too long for the schema. Prefer real-world phrasing over synthetic toy cases.

## System Prompt Best Practices for Tool Use

The system prompt is a functional manual — it defines the model's role, available tools, and the logic for operating them.

### Required components

- **Role + task definition** — state the model's identity and high-level goal
- **Tool specs** — names, descriptions, parameter schemas, output types
- **Strict output format** — define exact syntax for tool calls (e.g., `[ApiName(key='value')]`) so downstream parsing is reliable
- **Role definitions** — explicitly name User / Assistant / Tool and what each does
- **Parameter rules** — when to use original user text vs. transformed values; which params are optional
- **"Refrain" strategy** — what the model should do when the request is unclear, info is missing, or the task exceeds tool capabilities

### Authoring rules

- **Goldilocks detail** — use heuristics, not hardcoded if-else chains; keep tokens minimal and high-signal to preserve attention budget
- **Modular structure** — separate sections with Markdown headers (`## Tool guidance`) or XML tags (`<instructions>`); in multi-step flows add an explicit reminder like *"load a tool before calling it"*
- **Priority order for edge cases:**
  1. Clarify if parameter values are malformed
  2. Ask for missing required information
  3. Refuse only if the task is completely outside current tool capabilities
- **Grounding** — hint at preconditions in tool descriptions; for stateful systems instruct the model to reason about world state before calling; use 2–5 few-shot examples to map messy user input to structured params
- **Minimal footprint** — instruct the agent to request only necessary permissions, prefer reversible actions, and confirm with the user when scope is uncertain

## Adding a New Tool

1. Create or edit a module in `tools/`
2. Decorate each function with `@register(description=..., parameters=..., always_on=False)`
   - Set `always_on=True` only for tools the model needs on every turn (e.g. core utilities, context management)
   - Leave `always_on=False` (default) for everything else — it will be discoverable via `search_tools`
3. Import the module in `main.py` with a `# noqa: F401` comment
4. `build_index()` runs automatically after all imports and will include the new tool

<!-- code-review-graph MCP tools -->
## MCP Tools: code-review-graph

**IMPORTANT: This project has a knowledge graph. ALWAYS use the
code-review-graph MCP tools BEFORE using Grep/Glob/Read to explore
the codebase.** The graph is faster, cheaper (fewer tokens), and gives
you structural context (callers, dependents, test coverage) that file
scanning cannot.

### When to use graph tools FIRST

- **Exploring code**: `semantic_search_nodes` or `query_graph` instead of Grep
- **Understanding impact**: `get_impact_radius` instead of manually tracing imports
- **Code review**: `detect_changes` + `get_review_context` instead of reading entire files
- **Finding relationships**: `query_graph` with callers_of/callees_of/imports_of/tests_for
- **Architecture questions**: `get_architecture_overview` + `list_communities`

Fall back to Grep/Glob/Read **only** when the graph doesn't cover what you need.

### Key Tools

| Tool | Use when |
|------|----------|
| `detect_changes` | Reviewing code changes — gives risk-scored analysis |
| `get_review_context` | Need source snippets for review — token-efficient |
| `get_impact_radius` | Understanding blast radius of a change |
| `get_affected_flows` | Finding which execution paths are impacted |
| `query_graph` | Tracing callers, callees, imports, tests, dependencies |
| `semantic_search_nodes` | Finding functions/classes by name or keyword |
| `get_architecture_overview` | Understanding high-level codebase structure |
| `refactor_tool` | Planning renames, finding dead code |

### Workflow

1. The graph auto-updates on file changes (via hooks).
2. Use `detect_changes` for code review.
3. Use `get_affected_flows` to understand impact.
4. Use `query_graph` pattern="tests_for" to check coverage.
