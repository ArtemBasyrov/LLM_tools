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

## Tool Modules

| Module | Tools | Description |
|--------|-------|-------------|
| `tools/builtins.py` | `get_current_datetime`, `calculate` | Time and arithmetic |
| `tools/files.py` | `write_file`, `edit_file`, `read_json`, `write_json` | Create, edit, and handle JSON files |
| `tools/web.py` | `web_search`, `read_file`, `file_info`, `search_file`, `fetch_url`, `read_url` | DuckDuckGo search, file reading, URL fetching with chunking |
| `tools/filesystem.py` | `list_directory`, `get_working_context`, `make_directory`, `remove_file` | Directory listing, context, folder creation, deletion |
| `tools/memory.py` | `memory_save`, `memory_search`, `memory_list`, `memory_delete` | Persistent semantic memory |
| `tools/notebooklm.py` | `notebooklm_*` | Google NotebookLM integration |

## Semantic Memory

`tools/memory.py` provides persistent cross-session memory via LanceDB + `all-MiniLM-L6-v2` embeddings.

- **Data dir**: `~/.llm_memory/` (override with `LLM_MEMORY_DIR` env var)
- **Model**: `sentence-transformers/all-MiniLM-L6-v2` — 384-dim, ~22 MB, downloaded on first use
- **Search**: cosine similarity, returns `relevance` score (0–1)
- The system prompt instructs the model to search memory at session start and save notable facts

## Adding a New Tool

1. Create or edit a module in `tools/`
2. Decorate each function with `@register(description=..., parameters=...)`
3. Import the module in `chat.py` with a `# noqa: F401` comment
