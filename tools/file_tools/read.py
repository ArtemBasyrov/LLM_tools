"""
read_file  — read any local file (text, PDF, JSON); auto-detects format by extension
file_info  — metadata + chunking guidance without reading the whole file
search_file — regex search within a file with context lines
"""

import json
import math
import os
import re

from tools import register
from tools.file_tools._helpers import _MAX_BYTES

_PDF_MAX_CHARS = 100_000
_MAX_SEARCH_BYTES = 50_000_000


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------


@register(
    description=(
        "Read a local file and return its contents. "
        "Automatically selects the right mode based on file extension: "
        "• .pdf — extracts text page by page; use start_page/end_page for large PDFs. "
        "• .json — parses and returns structured data; use key_path to extract a nested value "
        "  (dot-separated, e.g. 'users.0.name'). "
        "• all other files — returns raw text; use start_line/end_line or start_char/end_char "
        "  for large files. "
        "Call file_info first when you don't know a file's size."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            # Text params
            "start_line": {
                "type": "integer",
                "description": "First line to return (1-indexed, inclusive). Text files only. Ignored if start_char/end_char provided.",
            },
            "end_line": {
                "type": "integer",
                "description": "Last line to return (1-indexed, inclusive). Text files only. Ignored if start_char/end_char provided.",
            },
            "start_char": {
                "type": "integer",
                "description": "Character offset to start reading from (0-indexed). Text files only. Useful for files with very long lines.",
            },
            "end_char": {
                "type": "integer",
                "description": "Character offset to stop reading at (0-indexed, exclusive). Text files only.",
            },
            # PDF params
            "start_page": {
                "type": "integer",
                "description": "First page to read (1-indexed, inclusive). PDF files only.",
            },
            "end_page": {
                "type": "integer",
                "description": "Last page to read (1-indexed, inclusive). PDF files only.",
            },
            # JSON params
            "key_path": {
                "type": "string",
                "description": "Dot-separated path to extract a nested value, e.g. 'a.b.0'. JSON files only.",
            },
        },
        "required": ["path"],
    },
)
def read_file(
    path: str,
    # text
    start_line: int | None = None,
    end_line: int | None = None,
    start_char: int | None = None,
    end_char: int | None = None,
    # pdf
    start_page: int | None = None,
    end_page: int | None = None,
    # json
    key_path: str | None = None,
) -> str:
    if start_line is not None:
        start_line = int(start_line)
    if end_line is not None:
        end_line = int(end_line)
    if start_char is not None:
        start_char = int(start_char)
    if end_char is not None:
        end_char = int(end_char)
    if start_page is not None:
        start_page = int(start_page)
    if end_page is not None:
        end_page = int(end_page)

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    ext = os.path.splitext(path)[1].lower()

    if ext == ".pdf":
        return _read_pdf(path, start_page, end_page)
    if ext == ".json":
        return _read_json(path, key_path)
    return _read_text(path, start_line, end_line, start_char, end_char)


# ---------------------------------------------------------------------------
# Internal implementations
# ---------------------------------------------------------------------------


def _read_text(
    path: str,
    start_line: int | None,
    end_line: int | None,
    start_char: int | None,
    end_char: int | None,
) -> str:
    if start_char is not None or end_char is not None:
        sc = start_char if start_char is not None else 0
        ec = end_char
        try:
            with open(path, "r", encoding="utf-8", errors="strict") as fh:
                if sc > 0:
                    fh.read(sc)
                chunk_limit = min(
                    (ec - sc) if ec is not None else _MAX_BYTES, _MAX_BYTES
                )
                content = fh.read(chunk_limit)
                has_more = fh.read(1) != ""
        except UnicodeDecodeError:
            return json.dumps(
                {"error": "File appears to be binary and cannot be read as text."}
            )
        except OSError as e:
            return json.dumps({"error": str(e)})

        actual_end = sc + len(content)
        meta: dict = {
            "path": path,
            "chars_returned": f"{sc}–{actual_end}",
            "chars_in_chunk": len(content),
        }
        if has_more:
            meta["has_more"] = True
            meta["next_start_char"] = actual_end
        return json.dumps({"meta": meta, "content": content}, ensure_ascii=False)

    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            raw = fh.read(_MAX_BYTES)
            truncated = fh.read(1) != ""
    except UnicodeDecodeError:
        return json.dumps(
            {"error": "File appears to be binary and cannot be read as text."}
        )
    except OSError as e:
        return json.dumps({"error": str(e)})

    lines = raw.splitlines(keepends=True)
    total_lines = len(lines)

    if start_line is not None or end_line is not None:
        sl = (start_line - 1) if start_line is not None else 0
        el = end_line if end_line is not None else total_lines
        lines = lines[sl:el]

    content = "".join(lines)
    meta = {"path": path, "total_lines": total_lines}
    if truncated:
        meta["warning"] = f"File truncated at {_MAX_BYTES} bytes."
    if start_line or end_line:
        meta["lines_returned"] = f"{start_line or 1}–{end_line or total_lines}"

    return json.dumps({"meta": meta, "content": content}, ensure_ascii=False)


def _read_json(path: str, key_path: str | None) -> str:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            raw = fh.read(_MAX_BYTES)
            truncated = fh.read(1) != ""
    except OSError as e:
        return json.dumps({"error": str(e)})

    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        return json.dumps({"error": f"Invalid JSON: {e}"})

    if key_path:
        try:
            for key in key_path.split("."):
                data = data[int(key)] if isinstance(data, list) else data[key]
        except (KeyError, IndexError, ValueError, TypeError) as e:
            return json.dumps({"error": f"Key path '{key_path}' not found: {e}"})

    result: dict = {"path": path, "data": data}
    if truncated:
        result["warning"] = (
            f"File truncated at {_MAX_BYTES} bytes — JSON may be incomplete."
        )
    return json.dumps(result, ensure_ascii=False)


def _read_pdf(path: str, start_page: int | None, end_page: int | None) -> str:
    try:
        import pypdf
    except ImportError:
        return json.dumps({"error": "pypdf is not installed. Run: pip install pypdf"})

    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        return json.dumps({"error": f"Failed to open PDF: {e}"})

    total_pages = len(reader.pages)
    sp = max(0, (start_page - 1) if start_page is not None else 0)
    ep = min(total_pages, end_page if end_page is not None else total_pages)

    if sp >= total_pages:
        return json.dumps(
            {"error": f"start_page {start_page} exceeds total pages ({total_pages})."}
        )

    pages_out = []
    total_chars = 0
    truncated = False

    for i in range(sp, ep):
        try:
            text = reader.pages[i].extract_text() or ""
        except Exception as e:
            text = f"[extraction error: {e}]"

        if total_chars + len(text) > _PDF_MAX_CHARS:
            remaining = _PDF_MAX_CHARS - total_chars
            pages_out.append({"page": i + 1, "text": text[:remaining]})
            truncated = True
            break

        pages_out.append({"page": i + 1, "text": text})
        total_chars += len(text)

    result: dict = {
        "path": path,
        "total_pages": total_pages,
        "pages_returned": f"{sp + 1}–{sp + len(pages_out)}",
        "pages": pages_out,
    }
    if truncated:
        result["warning"] = (
            f"Output truncated at {_PDF_MAX_CHARS:,} characters. "
            "Call again with a later start_page to continue."
        )
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# file_info
# ---------------------------------------------------------------------------


@register(
    description=(
        "Return metadata about a file — size, line count, a short preview, and "
        "chunking guidance — without reading the entire file. "
        "Always call this before read_file when you don't know a file's size. "
        "The response includes 'fits_in_one_read' (bool) and, when false, "
        "'chunks_needed' and 'suggested_chunk_lines' so you know exactly how to "
        "split subsequent read_file calls."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "preview_lines": {
                "type": "integer",
                "description": "Number of lines to include as a preview (default 20, max 100).",
            },
        },
        "required": ["path"],
    },
)
def file_info(path: str, preview_lines: int = 20) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    preview_lines = max(1, min(preview_lines, 100))
    size_bytes = os.path.getsize(path)

    try:
        line_count = 0
        preview: list[str] = []
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                line_count += 1
                if line_count <= preview_lines:
                    preview.append(line)
    except OSError as e:
        return json.dumps({"error": str(e)})

    fits = size_bytes <= _MAX_BYTES
    chunks_needed = math.ceil(size_bytes / _MAX_BYTES) if not fits else 1
    suggested_chunk_lines = math.ceil(line_count / chunks_needed) if line_count else 0
    avg_line_chars = (size_bytes / line_count) if line_count else 0
    long_lines = line_count <= 5 and avg_line_chars > _MAX_BYTES / 4

    result: dict = {
        "path": path,
        "size_bytes": size_bytes,
        "line_count": line_count,
        "fits_in_one_read": fits,
        "preview_lines_shown": len(preview),
        "preview": "".join(preview),
    }
    if long_lines:
        suggested_chunk_chars = _MAX_BYTES // 2
        result["long_line_file"] = True
        result["suggested_chunk_chars"] = suggested_chunk_chars
        result["char_read_strategy"] = (
            f"File has very long lines (avg {avg_line_chars:,.0f} chars/line) — "
            "line-based chunking is impractical. "
            f"Use read_file with start_char/end_char in steps of {suggested_chunk_chars:,} chars."
        )
    if not fits:
        result["chunks_needed"] = chunks_needed
        if not long_lines:
            result["suggested_chunk_lines"] = suggested_chunk_lines
            result["read_strategy"] = (
                f"File is too large for a single read_file call ({size_bytes:,} bytes > "
                f"{_MAX_BYTES:,} byte limit). "
                f"Either search_file to locate the relevant section, or read in "
                f"{chunks_needed} sequential chunks of ~{suggested_chunk_lines} lines each."
            )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# search_file
# ---------------------------------------------------------------------------


@register(
    description=(
        "Search a file for lines matching a regex pattern and return matching lines "
        "with their line numbers and optional surrounding context. "
        "Use this to locate relevant sections in a file that is too large to read "
        "entirely, then use read_file with start_line/end_line to read those sections."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "pattern": {
                "type": "string",
                "description": "Regular expression pattern to search for (case-insensitive by default).",
            },
            "context_lines": {
                "type": "integer",
                "description": "Lines to include before and after each match (default 2, max 10).",
            },
            "max_matches": {
                "type": "integer",
                "description": "Maximum number of matches to return (default 20, max 100).",
            },
            "case_sensitive": {
                "type": "boolean",
                "description": "Whether the search is case-sensitive (default false).",
            },
        },
        "required": ["path", "pattern"],
    },
)
def search_file(
    path: str,
    pattern: str,
    context_lines: int = 2,
    max_matches: int = 20,
    case_sensitive: bool = False,
) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    context_lines = max(0, min(context_lines, 10))
    max_matches = max(1, min(max_matches, 100))
    flags = 0 if case_sensitive else re.IGNORECASE

    try:
        regex = re.compile(pattern, flags)
    except re.error as e:
        return json.dumps({"error": f"Invalid regex pattern: {e}"})

    try:
        bytes_read = 0
        all_lines: list[str] = []
        truncated = False
        with open(path, "r", encoding="utf-8", errors="replace") as fh:
            for line in fh:
                bytes_read += len(line.encode("utf-8"))
                if bytes_read > _MAX_SEARCH_BYTES:
                    truncated = True
                    break
                all_lines.append(line)
    except OSError as e:
        return json.dumps({"error": str(e)})

    total_lines = len(all_lines)
    match_indices: list[int] = []
    for i, line in enumerate(all_lines):
        if regex.search(line):
            match_indices.append(i)
            if len(match_indices) >= max_matches:
                break

    if not match_indices:
        return json.dumps({"path": path, "total_lines": total_lines, "matches": []})

    blocks: list[tuple[int, int]] = []
    for idx in match_indices:
        start = max(0, idx - context_lines)
        end = min(total_lines - 1, idx + context_lines)
        if blocks and start <= blocks[-1][1] + 1:
            blocks[-1] = (blocks[-1][0], max(blocks[-1][1], end))
        else:
            blocks.append((start, end))

    matches = []
    for start, end in blocks:
        lines_out = []
        for i in range(start, end + 1):
            lines_out.append(
                {
                    "line_number": i + 1,
                    "text": all_lines[i].rstrip("\n"),
                    "match": bool(regex.search(all_lines[i])),
                }
            )
        matches.append(
            {"start_line": start + 1, "end_line": end + 1, "lines": lines_out}
        )

    result: dict = {
        "path": path,
        "total_lines": total_lines,
        "pattern": pattern,
        "match_count": len(match_indices),
        "blocks_returned": len(matches),
        "matches": matches,
    }
    if truncated:
        result["warning"] = (
            f"File truncated at {_MAX_SEARCH_BYTES // 1_000_000} MB — search may be incomplete."
        )
    if len(match_indices) >= max_matches:
        result["note"] = (
            f"Reached max_matches={max_matches}; there may be more matches."
        )
    return json.dumps(result, ensure_ascii=False)
