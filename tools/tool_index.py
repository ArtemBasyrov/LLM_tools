"""
Tool discovery meta-tools: search_tools and load_tools.

These two tools are always visible to the LLM. All other non-always-on tools
are hidden until the LLM explicitly loads them:

  1. search_tools(queries=["read a file"])  → names + short descriptions
  2. load_tools(tool_names=["read_file"])   → activates schemas for next call

build_index() must be called once after all tool modules are imported.
"""

import json
import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from tools import _registry, activate, register

logging.getLogger("sentence_transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_MODEL_NAME = "all-MiniLM-L6-v2"
_model: SentenceTransformer | None = None
_index: list[tuple[str, np.ndarray]] = []  # (tool_name, embedding)


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        _model = SentenceTransformer(_MODEL_NAME)
    return _model


def build_index() -> None:
    """Embed all registered tool descriptions. Call once after all imports."""
    global _index
    model = _get_model()
    entries = []
    for name, entry in _registry.items():
        desc = entry["schema"]["function"]["description"]
        text = f"{name}: {desc}"
        vec = model.encode(text, normalize_embeddings=True)
        entries.append((name, vec))
    _index = entries


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b))


@register(
    description=(
        "Discover available tools by semantic search. "
        "Call this when you need a capability that is not in your current tool list. "
        "Pass one or more short, action-oriented queries (e.g. ['read file', 'list directory']). "
        "Returns tool names and descriptions. "
        "You MUST then call load_tools with the chosen names before invoking them."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "queries": {
                "type": "array",
                "items": {"type": "string"},
                "description": "One or more short search queries, e.g. ['write file', 'search file content'].",
            },
            "top_k": {
                "type": "integer",
                "description": "Max results per query (1–10). Defaults to 5.",
            },
        },
        "required": ["queries"],
    },
)
def search_tools(queries: list[str], top_k: int = 5) -> str:
    if not _index:
        return json.dumps(
            {"error": "Tool index not built yet. This is a bug — report it."}
        )
    top_k = max(1, min(top_k, 10))
    model = _get_model()
    seen: dict[str, float] = {}
    for query in queries:
        qvec = model.encode(query, normalize_embeddings=True)
        scores = [(_cosine(qvec, vec), name) for name, vec in _index]
        scores.sort(reverse=True)
        for score, name in scores[:top_k]:
            if name not in seen or score > seen[name]:
                seen[name] = score

    ranked = sorted(seen.items(), key=lambda x: -x[1])
    results = []
    for name, score in ranked[: top_k * len(queries)]:
        desc = _registry[name]["schema"]["function"]["description"]
        short_desc = desc.split(".")[0].strip()
        results.append(
            {"name": name, "description": short_desc, "score": round(score, 3)}
        )

    return json.dumps(
        {
            "results": results,
            "reminder": "Call load_tools with chosen tool names before invoking them.",
        },
        ensure_ascii=False,
    )


@register(
    description=(
        "Activate one or more tools so you can call them. "
        "Pass the tool names returned by search_tools. "
        "Returns the full parameter schemas for each activated tool. "
        "The activated tools are available immediately — you can call them in the same response."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "tool_names": {
                "type": "array",
                "items": {"type": "string"},
                "description": "List of tool names to activate, e.g. ['read_file', 'write_file'].",
            },
        },
        "required": ["tool_names"],
    },
)
def load_tools(tool_names: list[str]) -> str:
    activated = []
    not_found = []
    schemas = []
    for name in tool_names:
        if activate(name):
            activated.append(name)
            schemas.append(_registry[name]["schema"]["function"])
        else:
            not_found.append(name)

    return json.dumps(
        {
            "activated": activated,
            "not_found": not_found,
            "schemas": schemas,
            "note": f"Tools {activated} are now active and can be called directly.",
        },
        ensure_ascii=False,
    )
