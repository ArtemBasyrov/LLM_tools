"""
Web search and file reading tools.

web_search   — query DuckDuckGo and return top results (no API key needed)
read_file    — read a local file and return its contents (supports line ranges)
file_info    — return metadata (size, line count, head preview) without reading the whole file
search_file  — regex search within a file, return matching lines with context and line numbers
read_pdf     — extract text from a local PDF file (supports page ranges)
fetch_url    — fetch a URL and return metadata + preview (fits_in_one_read, chunks_needed)
read_url     — fetch a URL and return a specific chunk of its text content
"""

import json
import os
import re

import requests
from bs4 import BeautifulSoup

from tools import register

# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------

_DDGO_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


@register(
    description=(
        "Search the web using DuckDuckGo and return the top results. "
        "Each result contains a title, URL, and short snippet. "
        "Use this to find current information or URLs you don't already know."
    ),
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1–10). Defaults to 5.",
            },
        },
        "required": ["query"],
    },
)
def web_search(query: str, max_results: int = 5) -> str:
    max_results = max(1, min(max_results, 10))
    try:
        resp = requests.post(
            _DDGO_URL,
            data={"q": query, "b": "", "kl": ""},
            headers=_HEADERS,
            timeout=10,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return json.dumps({"error": str(e)})

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []
    for result in soup.select(".result__body")[:max_results]:
        title_tag = result.select_one(".result__title a")
        snippet_tag = result.select_one(".result__snippet")
        if title_tag is None:
            continue
        href = title_tag.get("href", "")
        # DuckDuckGo wraps URLs — extract the real one from the `uddg` param
        if "uddg=" in href:
            from urllib.parse import parse_qs, urlparse

            qs = parse_qs(urlparse(href).query)
            href = qs.get("uddg", [href])[0]
        results.append(
            {
                "title": title_tag.get_text(strip=True),
                "url": href,
                "snippet": snippet_tag.get_text(strip=True) if snippet_tag else "",
            }
        )

    if not results:
        return json.dumps({"error": "No results found."})
    return json.dumps(results, ensure_ascii=False)


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------

_MAX_BYTES = 100_000  # ~100 KB — keeps context manageable


@register(
    description=(
        "Read a local file and return its text contents. "
        "Supports plain text, source code, JSON, CSV, and similar text-based formats. "
        "Returns an error if the file does not exist or is binary."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the file.",
            },
            "start_line": {
                "type": "integer",
                "description": (
                    "First line to return (1-indexed, inclusive). "
                    "Omit to start from the beginning."
                ),
            },
            "end_line": {
                "type": "integer",
                "description": (
                    "Last line to return (1-indexed, inclusive). "
                    "Omit to read to the end of the file."
                ),
            },
        },
        "required": ["path"],
    },
)
def read_file(
    path: str, start_line: int | None = None, end_line: int | None = None
) -> str:
    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    try:
        with open(path, "r", encoding="utf-8", errors="strict") as fh:
            raw = fh.read(_MAX_BYTES)
            truncated = fh.read(1) != ""  # check if there's more
    except UnicodeDecodeError:
        return json.dumps(
            {"error": "File appears to be binary and cannot be read as text."}
        )
    except OSError as e:
        return json.dumps({"error": str(e)})

    lines = raw.splitlines(keepends=True)
    total_lines = len(lines)

    # Apply line range if requested
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

    import math

    fits = size_bytes <= _MAX_BYTES
    chunks_needed = math.ceil(size_bytes / _MAX_BYTES) if not fits else 1
    # Distribute lines evenly across chunks; add a small overlap buffer
    suggested_chunk_lines = math.ceil(line_count / chunks_needed) if line_count else 0

    result: dict = {
        "path": path,
        "size_bytes": size_bytes,
        "line_count": line_count,
        "fits_in_one_read": fits,
        "preview_lines_shown": len(preview),
        "preview": "".join(preview),
    }
    if not fits:
        result["chunks_needed"] = chunks_needed
        result["suggested_chunk_lines"] = suggested_chunk_lines
        result["read_strategy"] = (
            f"File is too large for a single read_file call ({size_bytes:,} bytes > "
            f"{_MAX_BYTES:,} byte limit). "
            f"Either (a) call search_file to locate the relevant section then read only "
            f"those lines, or (b) read in {chunks_needed} sequential chunks of "
            f"~{suggested_chunk_lines} lines each."
        )

    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# search_file
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# read_pdf
# ---------------------------------------------------------------------------

_PDF_MAX_CHARS = 100_000  # ~100 KB of extracted text


@register(
    description=(
        "Extract and return the text content of a local PDF file. "
        "Supports page ranges so you can read a large document in chunks. "
        "Returns extracted text page by page with page numbers. "
        "Use this for PDFs; use read_file for plain text files."
    ),
    parameters={
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Absolute or relative path to the PDF file.",
            },
            "start_page": {
                "type": "integer",
                "description": (
                    "First page to read (1-indexed, inclusive). "
                    "Omit to start from the first page."
                ),
            },
            "end_page": {
                "type": "integer",
                "description": (
                    "Last page to read (1-indexed, inclusive). "
                    "Omit to read to the last page."
                ),
            },
        },
        "required": ["path"],
    },
)
def read_pdf(
    path: str,
    start_page: int | None = None,
    end_page: int | None = None,
) -> str:
    try:
        import pypdf
    except ImportError:
        return json.dumps({"error": "pypdf is not installed. Run: pip install pypdf"})

    path = os.path.expanduser(path)
    if not os.path.exists(path):
        return json.dumps({"error": f"File not found: {path}"})
    if not os.path.isfile(path):
        return json.dumps({"error": f"Path is not a file: {path}"})

    try:
        reader = pypdf.PdfReader(path)
    except Exception as e:
        return json.dumps({"error": f"Failed to open PDF: {e}"})

    total_pages = len(reader.pages)

    # Clamp and convert to 0-based indices
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
            text = text[:remaining]
            pages_out.append({"page": i + 1, "text": text})
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
# search_file
# ---------------------------------------------------------------------------

_MAX_SEARCH_BYTES = 50_000_000  # 50 MB — stop scanning past this


@register(
    description=(
        "Search a file for lines matching a regex pattern and return the matching lines "
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
                "description": (
                    "Number of lines to include before and after each match for context "
                    "(default 2, max 10)."
                ),
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

    # Read all lines (up to size limit) to support context windows
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

    # Find matching line indices (0-based)
    match_indices: list[int] = []
    for i, line in enumerate(all_lines):
        if regex.search(line):
            match_indices.append(i)
            if len(match_indices) >= max_matches:
                break

    if not match_indices:
        return json.dumps({"path": path, "total_lines": total_lines, "matches": []})

    # Build match blocks, merging overlapping context windows
    blocks: list[tuple[int, int]] = []  # (start, end) inclusive, 0-based
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
            line_text = all_lines[i].rstrip("\n")
            is_match = bool(regex.search(all_lines[i]))
            lines_out.append(
                {"line_number": i + 1, "text": line_text, "match": is_match}
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


# ---------------------------------------------------------------------------
# fetch_url / read_url
# ---------------------------------------------------------------------------

_WEB_MAX_CHARS = 50_000  # chars per read_url chunk (~50 KB of text)
_WEB_TIMEOUT = 15  # seconds


def _fetch_and_extract(url: str) -> tuple[str, str]:
    """
    Fetch *url* and return (title, clean_text).
    Strips HTML tags; raises requests.RequestException on network errors.
    """
    resp = requests.get(url, headers=_HEADERS, timeout=_WEB_TIMEOUT)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type!r}")

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove noise tags before extracting text
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""

    # Collapse runs of whitespace / blank lines
    raw_text = soup.get_text(separator="\n")
    lines = [ln.strip() for ln in raw_text.splitlines()]
    clean = "\n".join(ln for ln in lines if ln)

    return title, clean


@register(
    description=(
        "Fetch a URL and return its metadata and a short text preview — without "
        "returning the full content. Mirrors file_info for web pages. "
        "Always call this before read_url when you don't know how long the page is. "
        "The response includes 'fits_in_one_read' (bool) and, when false, "
        "'chunks_needed' and 'suggested_chunk_size' so you know how many "
        "read_url calls are required."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "preview_chars": {
                "type": "integer",
                "description": (
                    "Number of characters to include as a preview of the page text "
                    "(default 500, max 2000)."
                ),
            },
        },
        "required": ["url"],
    },
)
def fetch_url(url: str, preview_chars: int = 500) -> str:
    import math

    preview_chars = max(100, min(preview_chars, 2000))

    try:
        title, text = _fetch_and_extract(url)
    except requests.RequestException as e:
        return json.dumps({"error": str(e)})
    except ValueError as e:
        return json.dumps({"error": str(e)})

    total_chars = len(text)
    fits = total_chars <= _WEB_MAX_CHARS
    chunks_needed = math.ceil(total_chars / _WEB_MAX_CHARS) if not fits else 1

    result: dict = {
        "url": url,
        "title": title,
        "total_chars": total_chars,
        "fits_in_one_read": fits,
        "preview": text[:preview_chars],
    }
    if not fits:
        result["chunks_needed"] = chunks_needed
        result["suggested_chunk_size"] = _WEB_MAX_CHARS
        result["read_strategy"] = (
            f"Page is too long for a single read_url call ({total_chars:,} chars > "
            f"{_WEB_MAX_CHARS:,} char limit). "
            f"Call read_url {chunks_needed} time(s) with chunk=1 through chunk={chunks_needed}."
        )

    return json.dumps(result, ensure_ascii=False)


@register(
    description=(
        "Fetch a URL and return a specific chunk of its text content. "
        "Use fetch_url first to learn how many chunks the page has. "
        "If fits_in_one_read is true, call with chunk=1 (the default). "
        "If false, call repeatedly with chunk=1, chunk=2, … up to chunks_needed."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch.",
            },
            "chunk": {
                "type": "integer",
                "description": (
                    "Which chunk to return (1-indexed). "
                    "Each chunk is up to 50,000 characters. Defaults to 1."
                ),
            },
        },
        "required": ["url"],
    },
)
def read_url(url: str, chunk: int = 1) -> str:
    import math

    chunk = max(1, chunk)

    try:
        title, text = _fetch_and_extract(url)
    except requests.RequestException as e:
        return json.dumps({"error": str(e)})
    except ValueError as e:
        return json.dumps({"error": str(e)})

    total_chars = len(text)
    total_chunks = math.ceil(total_chars / _WEB_MAX_CHARS) if total_chars else 1

    if chunk > total_chunks:
        return json.dumps(
            {
                "error": (
                    f"chunk={chunk} is out of range — page has {total_chunks} chunk(s) "
                    f"({total_chars:,} chars total)."
                )
            }
        )

    start = (chunk - 1) * _WEB_MAX_CHARS
    end = start + _WEB_MAX_CHARS
    content = text[start:end]

    result: dict = {
        "url": url,
        "title": title,
        "chunk": chunk,
        "total_chunks": total_chunks,
        "total_chars": total_chars,
        "chars_returned": len(content),
        "content": content,
    }
    if chunk < total_chunks:
        result["next_chunk"] = chunk + 1

    return json.dumps(result, ensure_ascii=False)
