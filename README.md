# LLM Tools

A tool-calling interface for locally running LLMs, enabling safe, interactive agent execution through a rich set of capabilities.

## Overview

This project provides a suite of tools that a local LLM can invoke via function/tool calling. Built on top of [Ollama](https://ollama.com) and the [Ollama Python SDK](https://github.com/ollama/ollama-python), it supports:

- **File operations** (read, write, edit, JSON parsing)
- **Web capabilities** (DuckDuckGo search, URL fetching)
- **Semantic memory** (persistent cross-session facts via vector search)
- **NotebookLM integration** (Google's LLM-based document analysis)
- **Filesystem exploration** (directory listing, context queries)

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
```

### Running

```bash
micromamba run -n llm_tools python chat.py
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
| **Files** | `write_file`, `edit_file`, `read_json`, `write_json` | File creation/editing with diff previews |
| **Web** | `web_search`, `read_file`, `file_info`, `search_file`, `read_pdf`, `fetch_url`, `read_url` | Search, file reading with chunking, URL fetching |
| **Filesystem** | `list_directory`, `get_working_context`, `make_directory`, `remove_file` | Directory navigation and manipulation |
| **Memory** | `memory_save`, `memory_search`, `memory_list`, `memory_delete` | Persistent semantic memory with vector embeddings |
| **NotebookLM** | `notebooklm_create_notebook`, `notebooklm_add_source`, `notebooklm_ask`, `notebooklm_generate`, `notebooklm_list_artifacts`, `notebooklm_download` | Google NotebookLM integration |

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

Register by importing the module, then include it in `chat.py`:
```python
import tools.my_tools  # noqa: F401
```

### Semantic Memory

Persistent memory uses **LanceDB** + **all-MiniLM-L6-v2** embeddings:
- Data stored in `~/.llm_memory/` (override with `LLM_MEMORY_DIR` env var)
- 384-dimensional vectors, cosine similarity search
- Automatically retrieved based on conversation context

### Safety Features

- **Diff previews** before any file modification
- **User confirmation** required for destructive operations
- **Context-aware chunking** for large files (prevents overflow)
- **Error recovery** with structured JSON responses
- **Time-aware prompting** (reminds model to get datetime first)

## Project Structure

```
llm_tools/
├── chat.py              # Main chat loop and Ollama integration
├── tools/
│   ├── __init__.py    # Tool registry system
│   ├── builtins.py    # Basic utilities
│   ├── files.py       # File manipulation tools
│   ├── filesystem.py  # Directory operations
│   ├── memory.py      # Semantic memory (LanceDB + embeddings)
│   ├── web.py         # Web search + file reading
│   └── notebooklm.py  # Google NotebookLM integration
├── CLAUDE.md          # Internal documentation for LLM
└── README.md          # This file
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

The system supports models with 32K+ context windows.

## Adding New Tools

1. Create a new module in `tools/` (e.g., `tools/my_tools.py`)
2. Decorate functions with `@register(description=..., parameters=...)`
3. Import the module in `chat.py`:
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

Manual testing via the chat interface. Key scenarios:
- Time-sensitive queries (verify `get_current_datetime` called first)
- File edits (verify diff + confirmation)
- Large file handling (verify chunking logic)
- Memory search (verify semantic retrieval)

## Known Limitations

1. **Ollama dependency** — requires local Ollama instance running
2. **Model size** — optimized for 35B+ parameter models (smaller models may struggle with tool selection)
3. **Network calls** — `fetch_url` requires internet access
4. **NotebookLM API** — Google-specific rate limits may apply
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
