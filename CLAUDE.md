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
├── context_window.py     # Context window management
├── tools/
│   ├── __init__.py       # Tool registry: register(), activate(), schemas()
│   ├── tool_index.py     # search_tools, load_tools — tool discovery meta-tools
│   ├── builtins.py       # get_current_datetime, calculate
│   ├── file_tools/       # read_file, file_info, search_file, write_file, edit_file, write_json
│   ├── filesystem.py     # bash, read_context_files
│   ├── memory.py         # memory_save, memory_search, memory_list, memory_delete
│   ├── web.py            # web_search, fetch_url, read_url
│   ├── session.py        # session_save, session_recall, session_clear
│   └── notebooklm.py     # notebooklm_* tools
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

### Hidden tools (load on demand)

| Tool | Module | Description |
|------|--------|-------------|
| `read_file` | `file_tools/` | Read file content; returns first 8,000 chars by default with `has_more` + `next_start_char`; use `start_char`/`end_char` or `start_line`/`end_line` for chunks |
| `file_info` | `file_tools/` | File metadata (`size_bytes`, `line_count`, `fits_in_one_read`, read strategy); call before `read_file` on unknown files |
| `search_file` | `file_tools/` | Regex search within a file; result includes `size_bytes`, `total_chars`, line numbers; use `context_chars` for minified/long-line files |
| `write_file` | `file_tools/` | Create or overwrite a file |
| `edit_file` | `file_tools/` | Replace a specific string in a file |
| `write_json` | `file_tools/` | Write structured data as JSON |
| `bash` | `filesystem.py` | Run a shell command |
| `read_context_files` | `filesystem.py` | Load CLAUDE.md / AGENT.md from a directory and its parents |
| `fetch_url` | `web.py` | URL metadata (`total_chars`, `fits_in_one_read`, `chunks_needed`) + preview; call before `read_url` on unknown pages |
| `read_url` | `web.py` | Fetch a specific 8,000-char chunk of a URL; response includes `total_chars`, `has_more`, `next_chunk` |
| `memory_list` | `memory.py` | List recent memories |
| `memory_delete` | `memory.py` | Delete a memory by ID |
| `notebooklm_create_notebook` | `notebooklm.py` | Create a NotebookLM notebook |
| `notebooklm_list_notebooks` | `notebooklm.py` | List notebooks |
| `notebooklm_add_source` | `notebooklm.py` | Add a source to a notebook |
| `notebooklm_list_sources` | `notebooklm.py` | List sources in a notebook |
| `notebooklm_ask` | `notebooklm.py` | Ask a question to a notebook |
| `notebooklm_generate` | `notebooklm.py` | Generate an artifact (summary, FAQ, etc.) |
| `notebooklm_list_artifacts` | `notebooklm.py` | List generated artifacts |
| `notebooklm_download` | `notebooklm.py` | Download an artifact |

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
