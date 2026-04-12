"""
Web search and file reading tools.

web_search  — query DuckDuckGo and return top results (no API key needed)
read_file   — read a local file and return its contents
"""

import json
import os

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
