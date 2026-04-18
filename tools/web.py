"""
Web search and URL fetching tools.

web_search — query DuckDuckGo and return top results (no API key needed)
fetch_url  — fetch a URL and return metadata + preview (fits_in_one_read, chunks_needed)
read_url   — fetch a URL and return a specific chunk of its text content
"""

import json
import math

import requests
from bs4 import BeautifulSoup

from tools import register

_DDGO_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}
_WEB_MAX_CHARS = 50_000
_WEB_TIMEOUT = 15


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


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
# fetch_url / read_url
# ---------------------------------------------------------------------------


def _fetch_and_extract(url: str) -> tuple[str, str]:
    resp = requests.get(url, headers=_HEADERS, timeout=_WEB_TIMEOUT)
    resp.raise_for_status()

    content_type = resp.headers.get("content-type", "")
    if "html" not in content_type and "text" not in content_type:
        raise ValueError(f"Unsupported content type: {content_type!r}")

    soup = BeautifulSoup(resp.text, "html.parser")
    for tag in soup(
        ["script", "style", "nav", "footer", "header", "aside", "noscript"]
    ):
        tag.decompose()

    title = soup.title.string.strip() if soup.title and soup.title.string else ""
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
                "description": "Number of characters to include as a preview (default 500, max 2000).",
            },
        },
        "required": ["url"],
    },
)
def fetch_url(url: str, preview_chars: int = 500) -> str:
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
                "description": "Which chunk to return (1-indexed). Each chunk is up to 50,000 characters. Defaults to 1.",
            },
        },
        "required": ["url"],
    },
)
def read_url(url: str, chunk: int = 1) -> str:
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
    content = text[start : start + _WEB_MAX_CHARS]

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
