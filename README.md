# LLM Tools

A tool-calling interface for locally running LLMs, enabling safe, interactive agent execution through a rich set of capabilities.

## Overview

This project provides a suite of tools that a local LLM can invoke via function/tool calling. Built on top of [Ollama](https://ollama.com) and the [Ollama Python SDK](https://github.com/ollama/ollama-python), it supports:

- **File operations** with a generate-and-check feedback loop — line-numbered reads, near-miss hints on failed edits, atomic batch edits, mtime-based staleness guard, undo ring, and automatic syntax checks (py_compile / ruff / JSON / YAML / TOML)
- **Code intelligence** — Python AST `go_to_definition` / `find_references`, multi-file `apply_patch`, typed `find_files`
- **Git tools** — `git_status` / `git_diff` / `git_blame` as first-class JSON tools
- **Test runner** — focused `pytest` invocations with capped output
- **Web capabilities** — DuckDuckGo search, URL fetching with chunking
- **Semantic memory** — persistent cross-session facts via LanceDB + vector search
- **NotebookLM integration** — Google's LLM-based document analysis
- **Agentic orchestration** — per-turn state machine with plan/verify/critic loops and a sticky `[SYSTEM FILES]` view of the current working set

## Quick Start

### Prerequisites

- **Python 3.10+**
- **[micromamba](https://mamba.readthedocs.io/)** (recommended) or conda
- **[Ollama](https://ollama.com)** with `qwen3.5:35b` model

### Installation

```bash
# Clone and navigate to the repository
git clone https://github.com/yourusername/llm_tools.git
cd llm_tools

# Create and activate micromamba environment
micromamba create -n llm_tools python=3.11 -y
micromamba activate llm_tools

# Install dependencies
micromamba install -n llm_tools ollama lancedb sentence-transformers pyarrow -y

# NotebookLM integration (optional)
pip install notebooklm-py
notebooklm login   # one-time Google auth
```

### Running

```bash
micromamba run -n llm_tools python main.py
```

The model will automatically download if not present. You'll see a prompt-ready interface with:
- Colored tool call indicators
- LLM "thinking" display
- Context window usage stats
- Tool availability listing

## Supported Tools

| Category | Tools | Description |
|----------|-------|-------------|
| **Builtins** | `get_current_datetime`, `calculate` | Time queries and safe arithmetic |
| **File I/O** | `read_file`, `file_info`, `search_file`, `write_file`, `write_json`, `edit_file`, `apply_patch`, `format_file`, `check_syntax`, `undo_last_edit`, `undo_list` | Line-numbered reads, multi-file batch, AST `outline` mode, near-miss hints on failed edits, replace_all / anchor_line / batch atomic edits, stale-read guard, auto syntax check, undo ring |
| **Code intelligence** | `go_to_definition`, `find_references` | Python AST-based symbol navigation |
| **Search & shell** | `find_files`, `bash`, `read_context_files`, `list_directory`, `make_directory`, `remove_file`, `get_working_context` | Typed glob with hidden-dir guard, shell, context-file loader |
| **Git** | `git_status`, `git_diff`, `git_blame` | Porcelain v2 status, capped diffs, line-range blame |
| **Tests** | `run_tests` | `pytest -x --tb=short` with output trimming |
| **Web** | `web_search`, `fetch_url`, `read_url` | DuckDuckGo search, URL fetching with chunking |
| **Memory** | `memory_save`, `memory_search`, `memory_list`, `memory_delete` | Persistent semantic memory with vector embeddings |
| **Plan / verify / session** | `plan_*`, `verify_report`, `session_save`, `session_recall`, `session_clear`, `set_mode` | Agentic plan execution, verification, snapshot/recall, mode switching |
| **NotebookLM** | `notebooklm_create_notebook`, `notebooklm_add_source`, `notebooklm_ask`, `notebooklm_generate`, `notebooklm_list_artifacts`, `notebooklm_download` | Google NotebookLM integration via [notebooklm-py](https://github.com/teng-lin/notebooklm-py) |

## How It Works

### Tool Registration Pattern

```python
from tools import register

@register(
    description="What this tool does and when to use it.",
    parameters={
        "type": "object",
        "properties": {
            "param": {"type": "string", "description": "Parameter description"},
        },
        "required": ["param"],
    },
)
def my_tool(param: str) -> str:
    return f"Processed: {param}"
```

Register by importing the module, then include it in `main.py`:
```python
import tools.my_tools  # noqa: F401
```

### Semantic Memory

Persistent memory uses **LanceDB** + **all-MiniLM-L6-v2** embeddings:
- Data stored in `~/.llm_memory/` (override with `LLM_MEMORY_DIR` env var)
- 384-dimensional vectors, cosine similarity search
- Automatically retrieved based on conversation context

### Safety Features

- **Diff previews** before any file modification, with user confirmation on destructive ops
- **Stale-read guard** — `edit_file` / `write_file` / `apply_patch` refuse to act when the on-disk mtime has moved since the last `read_file` (~50 ms tolerance)
- **Probe-first overwrite** — `write_file` refuses to clobber an existing file that hasn't been read in this session unless `overwrite=true`
- **Atomic batch edits** — `edit_file(edits=[…])` and `apply_patch` roll back on any per-step failure
- **Undo ring** — every mutating op snapshots the file (10-deep per path, 64-deep global); `undo_last_edit` walks back through the ring
- **Generate-and-check** — `write_file` / `edit_file` / `apply_patch` automatically run a syntax check (py_compile + ruff for `.py`; strict parsers for JSON/YAML/TOML) and return the verdict
- **Near-miss hints** — when `edit_file`'s `old_string` is missing, the response lists the closest existing lines with line numbers so the model can correct whitespace/typo errors
- **Context-aware chunking** for large files; old `read_file` tool results are evicted with a path-keyed re-read pointer
- **`[SYSTEM FILES]` sticky view** — orchestrator maintains a single up-to-date message listing the model's working set + staleness flags
- **Error recovery** with structured JSON responses
- **Time-aware prompting** (reminds model to get datetime first)

## Project Structure

```
llm_tools/
├── main.py               # Main chat loop and Ollama integration
├── system_prompt.py      # System prompt and model configuration
├── rendering.py          # Display and formatting helpers
├── context_window.py     # Context window management + sticky-message compaction
├── agent/
│   ├── orchestrator.py   # State machine wrapping each user turn
│   ├── plan.py / triage / verifier / critic / prompts / modes
├── tools/
│   ├── __init__.py       # Tool registry: register(), activate(), schemas()
│   ├── tool_index.py     # search_tools, load_tools meta-tools
│   ├── builtins.py       # get_current_datetime, calculate
│   ├── file_tools/       # read / write / edit / patch / undo / format / check
│   │   └── _state.py     # session ledger: mtimes, backup ring, open-files snapshot
│   ├── code_intel.py     # go_to_definition, find_references (Python AST)
│   ├── filesystem.py     # bash, list_directory, …
│   ├── find.py           # find_files (typed glob)
│   ├── git_tools.py      # git_status, git_diff, git_blame
│   ├── memory.py         # semantic memory (LanceDB + embeddings)
│   ├── notebooklm.py     # Google NotebookLM integration
│   ├── plan_tools.py     # plan_create/add_step/start/complete/log/abandon
│   ├── session.py        # session_save / recall / clear
│   ├── test_runner.py    # run_tests
│   ├── verify_tools.py   # verify_report
│   └── web.py            # web_search, fetch_url, read_url
├── tests/                # pytest suite (91 tests)
├── CLAUDE.md             # Internal documentation for LLM
└── README.md             # This file
```

## Configuration

Copy `.env.example` to `.env` and edit as needed — the file is gitignored and never committed:

```bash
cp .env.example .env
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `OLLAMA_MODEL` | `qwen3.5:35b` | Ollama model to use (must be pulled locally) |
| `OLLAMA_HOST` | `http://localhost:11434` | Ollama server URL |
| `LLM_MEMORY_DIR` | `~/.llm_memory` | Custom location for semantic memory database |
| `LLM_PLAN_DIR` | `~/.llm_plans` | Custom location for agentic plan JSON files |
| `AGENTIC_MODE` | `true` | Wrap each turn in the orchestrator state machine |
| `ENFORCE_PLANNING` | `true` | Block final answer while an active plan has unverified steps |
| `CRITIC_MAX_ROUNDS` | `2` | Max critic revise rounds per final response (0 disables) |

The system supports models with 32K+ context windows.

## Agentic Mode

When `AGENTIC_MODE=true`, each turn is driven by a state machine
(`agent/orchestrator.py`) that enforces rigorous execution:

1. **Triage** — complex requests get a `[SYSTEM TRIAGE]` nudge asking the
   model to draft a plan via `plan_create` + `plan_add_step`.
2. **Plan** — plans persist to `~/.llm_plans/active.json` so work can span
   context windows and process restarts. On startup the system prompt notes
   any active plan so the model can resume.
3. **Verify** — every `plan_complete_step` call triggers a `[SYSTEM VERIFIER]`
   injection demanding the model independently confirm the claimed evidence
   (via `read_file` / `bash` / `search_file`) and call `verify_report(...)`.
   Unverified steps roll back to `in_progress`.
4. **Critic** — final responses run through up to `CRITIC_MAX_ROUNDS` of
   self-review. The model emits `{"accept": bool, "issues": [...]}` JSON;
   unresolved issues trigger a revise loop.
5. **Snapshot nudge** — when context ≥70% full and a plan is active, the
   orchestrator asks the model to call `session_save` so the plan survives
   trimming.

All synthetic injections use an authoritative `[SYSTEM <ROLE>]` prefix and
are kept sticky across context compaction.

## Adding New Tools

1. Create a new module in `tools/` (e.g., `tools/my_tools.py`)
2. Decorate functions with `@register(description=..., parameters=...)`
3. Import the module in `main.py`:
   ```python
   import tools.my_tools  # noqa: F401
   ```
4. Tool is automatically available to the LLM

### Best Practices

- **One purpose per tool** — avoid multi-purpose functions
- **Clear descriptions** — the model uses these to decide when to call tools
- **Structured return values** — prefer JSON for parsing
- **Consistent patterns** — follow existing tool conventions
- **Safe defaults** — validate inputs, handle errors gracefully

## Development Notes

### Code Style

- Python type hints throughout
- Consistent docstrings
- TTY-aware ANSI colors (respects terminal capabilities)
- Modular, single-responsibility design

### Testing

Automated pytest suite (91 tests) — file-tool behavior, orchestrator state machine, plan/verify/critic logic, triage:

```bash
micromamba run -n internet python -m pytest tests/ -v
```

Manual scenarios still worth exercising via the chat interface:
- Time-sensitive queries (verify `get_current_datetime` called first)
- File edits (verify diff + confirmation, stale-read refusal, near-miss hints)
- Large file handling (verify chunking logic and `outline=true`)
- Memory search (verify semantic retrieval)

## Known Limitations

1. **Ollama dependency** — requires local Ollama instance running
2. **Model size** — optimized for 35B+ parameter models (smaller models may struggle with tool selection)
3. **Network calls** — `fetch_url` requires internet access
4. **NotebookLM** — requires [notebooklm-py](https://github.com/teng-lin/notebooklm-py) and a one-time `notebooklm login`; Google-specific rate limits may apply
5. **Embedding model** — downloads `all-MiniLM-L6-v2` on first memory use (~22 MB)

## Security Considerations

- No API keys are required — all tools use local resources or key-free APIs (DuckDuckGo)
- Sensitive config (model name, host) lives in `.env`, which is gitignored
- File write/delete operations require explicit user confirmation at runtime
- The `calculate` tool only allows basic arithmetic characters

## License

MIT License (or add your preferred license)

## Contributing

Contributions welcome! Please:
1. Follow existing tool patterns
2. Add clear descriptions for new tools
3. Include tests if adding complex logic
4. Update documentation as needed

---

**Built with ❤️ for local LLM agents**