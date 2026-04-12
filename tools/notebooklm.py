"""
NotebookLM tools — wraps the `notebooklm` CLI for the local LLM.

Requires:
    pip install notebooklm-py
    notebooklm login   (one-time Google auth)

Each tool maps to one CLI operation and returns JSON-serialised structured data.
"""

import json
import shlex
import subprocess
from typing import Any

from tools import register

# ---------------------------------------------------------------------------
# Internal helper
# ---------------------------------------------------------------------------

_TIMEOUT = 60  # seconds; long-running ops (generate/wait) should use dedicated tools


def _run(args: list[str], timeout: int = _TIMEOUT) -> dict[str, Any]:
    """Run a notebooklm CLI command and return parsed output."""
    try:
        result = subprocess.run(
            ["notebooklm"] + args,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except FileNotFoundError:
        return {
            "error": "notebooklm CLI not found. Install with: pip install notebooklm-py"
        }
    except subprocess.TimeoutExpired:
        return {"error": f"Command timed out after {timeout}s"}

    stdout = result.stdout.strip()
    stderr = result.stderr.strip()

    if result.returncode != 0:
        return {"error": stderr or stdout or f"Exit code {result.returncode}"}

    if not stdout:
        return {"ok": True}

    # Try JSON first (most commands support --json)
    try:
        return json.loads(stdout)
    except json.JSONDecodeError:
        return {"output": stdout}


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@register(
    description=(
        "List all NotebookLM notebooks in the authenticated account. "
        "Returns notebook IDs and titles. Call this first to find a notebook_id."
    ),
)
def notebooklm_list_notebooks() -> str:
    return json.dumps(_run(["list", "--json"]))


@register(
    description=(
        "Create a new NotebookLM notebook with the given title. "
        "Returns the new notebook's id and title."
    ),
    parameters={
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Title for the new notebook.",
            },
        },
        "required": ["title"],
    },
)
def notebooklm_create_notebook(title: str) -> str:
    return json.dumps(_run(["create", title, "--json"]))


@register(
    description=(
        "Add a source to a NotebookLM notebook. "
        "Accepts a URL (web page, YouTube video) or a local file path (PDF, text, Markdown, etc.). "
        "Returns source_id and processing status. "
        "Sources must finish processing before they can be used for chat or generation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "notebook_id": {
                "type": "string",
                "description": "ID of the target notebook (from notebooklm_list_notebooks).",
            },
            "source": {
                "type": "string",
                "description": "URL or absolute/relative file path to add as a source.",
            },
        },
        "required": ["notebook_id", "source"],
    },
)
def notebooklm_add_source(notebook_id: str, source: str) -> str:
    return json.dumps(
        _run(["source", "add", source, "--notebook", notebook_id, "--json"])
    )


@register(
    description=(
        "List all sources in a NotebookLM notebook. "
        "Returns each source's id, title, and processing status (processing/ready/error). "
        "Check this to confirm sources are ready before asking questions or generating content."
    ),
    parameters={
        "type": "object",
        "properties": {
            "notebook_id": {
                "type": "string",
                "description": "ID of the notebook.",
            },
        },
        "required": ["notebook_id"],
    },
)
def notebooklm_list_sources(notebook_id: str) -> str:
    return json.dumps(_run(["source", "list", "--notebook", notebook_id, "--json"]))


@register(
    description=(
        "Ask a question about the sources in a NotebookLM notebook (chat). "
        "Returns the answer and citation references. "
        "All sources must have status=ready before calling this. "
        "Use follow_up=true and pass conversation_id to continue an existing conversation."
    ),
    parameters={
        "type": "object",
        "properties": {
            "notebook_id": {
                "type": "string",
                "description": "ID of the notebook to query.",
            },
            "question": {
                "type": "string",
                "description": "The question to ask about the notebook's sources.",
            },
            "conversation_id": {
                "type": "string",
                "description": "Conversation ID to continue a prior chat thread. Omit to start a new conversation.",
            },
        },
        "required": ["notebook_id", "question"],
    },
)
def notebooklm_ask(
    notebook_id: str, question: str, conversation_id: str | None = None
) -> str:
    args = ["ask", question, "--notebook", notebook_id, "--json"]
    if conversation_id:
        args += ["-c", conversation_id]
    return json.dumps(_run(args))


@register(
    description=(
        "Generate an artifact (report, quiz, flashcards, mind-map, or podcast) from a NotebookLM notebook. "
        "Mind-map is instant. All others are long-running — this call starts generation and returns a task_id. "
        "Use notebooklm_list_artifacts to check when status becomes 'completed'. "
        "artifact_type options: 'report', 'quiz', 'flashcards', 'mind-map', 'audio' (podcast). "
        "format options for report: 'briefing-doc' (default), 'study-guide', 'blog-post'. "
        "format options for audio: 'deep-dive' (default), 'brief', 'critique', 'debate'."
    ),
    parameters={
        "type": "object",
        "properties": {
            "notebook_id": {
                "type": "string",
                "description": "ID of the notebook.",
            },
            "artifact_type": {
                "type": "string",
                "description": "Type of artifact to generate: 'report', 'quiz', 'flashcards', 'mind-map', or 'audio'.",
                "enum": ["report", "quiz", "flashcards", "mind-map", "audio"],
            },
            "format": {
                "type": "string",
                "description": "Optional sub-format. For report: 'briefing-doc', 'study-guide', 'blog-post'. For audio: 'deep-dive', 'brief', 'critique', 'debate'.",
            },
            "instructions": {
                "type": "string",
                "description": "Optional natural-language instructions to guide the generation.",
            },
        },
        "required": ["notebook_id", "artifact_type"],
    },
)
def notebooklm_generate(
    notebook_id: str,
    artifact_type: str,
    format: str | None = None,
    instructions: str | None = None,
) -> str:
    args = ["generate", artifact_type, "--notebook", notebook_id, "--json"]
    if format:
        args += ["--format", format]
    if instructions:
        args.append(instructions)
    # Generation can take many minutes; use a longer timeout
    return json.dumps(_run(args, timeout=120))


@register(
    description=(
        "List all generated artifacts in a NotebookLM notebook. "
        "Returns each artifact's id, type, and status (pending/in_progress/completed/unknown). "
        "Poll this after calling notebooklm_generate to check when an artifact is ready to download."
    ),
    parameters={
        "type": "object",
        "properties": {
            "notebook_id": {
                "type": "string",
                "description": "ID of the notebook.",
            },
        },
        "required": ["notebook_id"],
    },
)
def notebooklm_list_artifacts(notebook_id: str) -> str:
    return json.dumps(_run(["artifact", "list", "--notebook", notebook_id, "--json"]))


@register(
    description=(
        "Download a completed artifact from a NotebookLM notebook to a local file. "
        "Only call this after notebooklm_list_artifacts shows status='completed'. "
        "artifact_type options: 'audio' (.mp3), 'report' (.md), 'quiz' (.json or .md), "
        "'flashcards' (.json or .md), 'mind-map' (.json). "
        "output_path must be a writable absolute or relative path including the file extension."
    ),
    parameters={
        "type": "object",
        "properties": {
            "notebook_id": {
                "type": "string",
                "description": "ID of the notebook.",
            },
            "artifact_type": {
                "type": "string",
                "description": "Type of artifact to download: 'audio', 'report', 'quiz', 'flashcards', 'mind-map'.",
                "enum": ["audio", "report", "quiz", "flashcards", "mind-map"],
            },
            "output_path": {
                "type": "string",
                "description": "Local path to save the file, including extension (e.g. './podcast.mp3', './quiz.md').",
            },
            "artifact_id": {
                "type": "string",
                "description": "Specific artifact ID to download. Use when multiple artifacts of the same type exist.",
            },
            "format": {
                "type": "string",
                "description": "Output format: 'json' or 'markdown' for quiz/flashcards; 'pptx' for slide-deck.",
            },
        },
        "required": ["notebook_id", "artifact_type", "output_path"],
    },
)
def notebooklm_download(
    notebook_id: str,
    artifact_type: str,
    output_path: str,
    artifact_id: str | None = None,
    format: str | None = None,
) -> str:
    args = ["download", artifact_type, output_path, "-n", notebook_id]
    if artifact_id:
        args += ["-a", artifact_id]
    if format:
        args += ["--format", format]
    return json.dumps(_run(args, timeout=120))
