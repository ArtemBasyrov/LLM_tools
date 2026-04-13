"""
Semantic memory tools backed by LanceDB.

Stores text memories as vector embeddings so the LLM can save and
retrieve relevant facts across sessions.

Data lives in ~/.llm_memory/ (override with LLM_MEMORY_DIR env var).
Embedding model: all-MiniLM-L6-v2 (~22 MB, fast, good quality).

Tools registered:
    memory_save   — embed and persist a memory
    memory_search — retrieve the most relevant memories for a query
    memory_list   — show recent memories (for inspection / housekeeping)
    memory_delete — remove a memory by its ID
"""

import datetime
import json
import os
import uuid

import logging

import lancedb
import pyarrow as pa
from sentence_transformers import SentenceTransformer
from tools import register

# Suppress noisy HuggingFace startup output (unauthenticated warning, load
# report, and weight-loading progress bar) — all harmless for local use.
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("transformers").setLevel(logging.ERROR)
try:
    import transformers as _transformers

    _transformers.logging.set_verbosity_error()
    _transformers.logging.disable_progress_bar()
except Exception:
    pass

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

_DATA_DIR = os.path.expanduser(os.environ.get("LLM_MEMORY_DIR", "~/.llm_memory"))
_TABLE_NAME = "memories"
_MODEL_NAME = "all-MiniLM-L6-v2"
_EMBED_DIM = 384  # dimension for all-MiniLM-L6-v2

# ---------------------------------------------------------------------------
# Lazy singletons — loaded on first use to keep import time fast
# ---------------------------------------------------------------------------

_model: SentenceTransformer | None = None
_db: lancedb.DBConnection | None = None
_table = None  # lancedb.Table


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(
            _MODEL_NAME,
            cache_folder=os.path.join(_DATA_DIR, "models"),
        )
    return _model


def _get_table():
    global _db, _table
    if _table is not None:
        return _table

    os.makedirs(_DATA_DIR, exist_ok=True)
    _db = lancedb.connect(_DATA_DIR)

    schema = pa.schema(
        [
            pa.field("id", pa.string()),
            pa.field("content", pa.string()),
            pa.field("tags", pa.string()),  # comma-separated
            pa.field("created_at", pa.string()),  # ISO timestamp
            pa.field("vector", pa.list_(pa.float32(), _EMBED_DIM)),
        ]
    )

    if _TABLE_NAME in _db.table_names():
        _table = _db.open_table(_TABLE_NAME)
    else:
        _table = _db.create_table(_TABLE_NAME, schema=schema, mode="create")

    return _table


def _embed(text: str) -> list[float]:
    return _get_model().encode(text, normalize_embeddings=True).tolist()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@register(
    description=(
        "Save a memory so it can be retrieved later by semantic search. "
        "Use this to persist facts, user preferences, decisions, observations, "
        "or any information worth remembering across sessions. "
        "Optionally tag the memory for easier filtering. "
        "Returns the ID of the saved memory."
    ),
    parameters={
        "type": "object",
        "properties": {
            "content": {
                "type": "string",
                "description": "The text to remember. Be specific and self-contained — this will be read without context later.",
            },
            "tags": {
                "type": "string",
                "description": "Optional comma-separated tags, e.g. 'preference,ui' or 'fact,python'.",
            },
        },
        "required": ["content"],
    },
)
def memory_save(content: str, tags: str = "") -> str:
    try:
        table = _get_table()
        memory_id = str(uuid.uuid4())[:8]
        row = {
            "id": memory_id,
            "content": content,
            "tags": tags.strip(),
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "vector": _embed(content),
        }
        table.add([row])
        return json.dumps({"saved": True, "id": memory_id})
    except Exception as e:
        return json.dumps({"error": str(e)})


@register(
    description=(
        "Search saved memories by semantic similarity to a query. "
        "Returns the most relevant memories ranked by relevance. "
        "Call this before answering questions that may involve previously stored "
        "facts, preferences, or context."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "Natural-language description of what you are looking for.",
            },
            "top_k": {
                "type": "integer",
                "description": "Maximum number of results to return (1–20). Defaults to 5.",
            },
        },
        "required": ["query"],
    },
)
def memory_search(query: str, top_k: int = 5) -> str:
    try:
        top_k = max(1, min(top_k, 20))
        table = _get_table()
        results = (
            table.search(_embed(query), vector_column_name="vector")
            .metric("cosine")
            .select(["id", "content", "tags", "created_at", "_distance"])
            .limit(top_k)
            .to_list()
        )
        if not results:
            return json.dumps({"results": [], "message": "No memories found."})
        memories = [
            {
                "id": r["id"],
                "content": r["content"],
                "tags": r["tags"],
                "created_at": r["created_at"],
                # cosine distance: 0 = identical, 2 = opposite; convert to similarity score
                "relevance": round(1 - float(r.get("_distance", 1)), 4),
            }
            for r in results
        ]
        return json.dumps({"results": memories}, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)})


@register(
    description=(
        "List recently saved memories, newest first. "
        "Useful for reviewing what has been stored or finding IDs for deletion."
    ),
    parameters={
        "type": "object",
        "properties": {
            "limit": {
                "type": "integer",
                "description": "Maximum number of memories to return (1–50). Defaults to 10.",
            },
        },
        "required": [],
    },
)
def memory_list(limit: int = 10) -> str:
    try:
        limit = max(1, min(limit, 50))
        table = _get_table()
        rows = (
            table.search()
            .select(["id", "content", "tags", "created_at"])
            .limit(limit)
            .to_list()
        )
        # Sort newest-first by created_at string (ISO format sorts correctly)
        rows.sort(key=lambda r: r.get("created_at", ""), reverse=True)
        memories = [
            {
                "id": r["id"],
                "content": r["content"],
                "tags": r["tags"],
                "created_at": r["created_at"],
            }
            for r in rows
        ]
        return json.dumps(
            {"count": len(memories), "memories": memories}, ensure_ascii=False
        )
    except Exception as e:
        return json.dumps({"error": str(e)})


@register(
    description=(
        "Delete a memory by its ID. "
        "Use this to remove outdated, incorrect, or no-longer-relevant memories. "
        "Get IDs from memory_search or memory_list."
    ),
    parameters={
        "type": "object",
        "properties": {
            "id": {
                "type": "string",
                "description": "The ID of the memory to delete (8-character string).",
            },
        },
        "required": ["id"],
    },
)
def memory_delete(id: str) -> str:
    try:
        table = _get_table()
        table.delete(f"id = '{id}'")
        return json.dumps({"deleted": True, "id": id})
    except Exception as e:
        return json.dumps({"error": str(e)})
