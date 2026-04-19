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
_WEB_DEFAULT_CHARS = 8_000  # chars returned per chunk by default
_WEB_TIMEOUT = 15


# ---------------------------------------------------------------------------
# web_search
# ---------------------------------------------------------------------------


@register(
    description=(
        "Search the web using DuckDuckGo and return the top results. "
        "Each result contains a title, URL, and short snippet. "
        "Use this to find current information or URLs you don't already know. "
        "If you already have the URL, skip this and call fetch_url or read_url directly. "
        "Examples: "
        "user asks 'latest Python release' → query='Python latest release 2024'; "
        "user asks for a library's homepage → query='requests library python homepage', then use the URL from results with read_url; "
        "NOT for: fetching or reading the content of a page you already have the URL for (use read_url instead)."
    ),
    always_on=True,
    parameters={
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific — include version years or context when relevant. e.g. 'FastAPI tutorial async 2024' or 'pandas merge on multiple columns'.",
            },
            "max_results": {
                "type": "integer",
                "description": "Maximum number of results to return (1–10). Defaults to 5. Use 1–2 when you just need a URL; use 5–10 for research.",
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
        "returning the full content. Analogous to file_info for web pages. "
        "Always call this before read_url when you don't know how long the page is. "
        "Returns total_chars, fits_in_one_read, chunks_needed, and suggested_chunk_size "
        "so you know exactly how many read_url calls are required. "
        "Examples: "
        "unknown page → fetch_url(url='https://docs.python.org/3/library/asyncio.html') → fits_in_one_read=false, chunks_needed=3 → then call read_url 3 times; "
        "want just the title/topic before reading → fetch_url with preview_chars=200; "
        "NOT for: reading page content — use read_url for that."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch. Must be an http:// or https:// URL.",
            },
            "preview_chars": {
                "type": "integer",
                "description": "Characters to include as preview (default 500, max 2000). Increase to 2000 if the preview alone may answer the question.",
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
    fits = total_chars <= _WEB_DEFAULT_CHARS
    chunks_needed = math.ceil(total_chars / _WEB_DEFAULT_CHARS) if not fits else 1

    result: dict = {
        "url": url,
        "title": title,
        "total_chars": total_chars,
        "fits_in_one_read": fits,
        "chunks_needed": chunks_needed,
        "suggested_chunk_size": _WEB_DEFAULT_CHARS,
        "preview": text[:preview_chars],
    }
    if not fits:
        result["read_strategy"] = (
            f"Page has {total_chars:,} chars. "
            f"Call read_url {chunks_needed} time(s) with chunk=1 through chunk={chunks_needed}."
        )

    return json.dumps(result, ensure_ascii=False)


@register(
    description=(
        "Fetch a URL and return a specific 8,000-character chunk of its text content. "
        "Use fetch_url first to get total_chars and chunks_needed. "
        "Check has_more in the response and continue with next_chunk if needed. "
        "Call with chunk=1 (default) for the first read; use next_chunk value to continue. "
        "Examples: "
        "first read → read_url(url='https://example.com/docs') → has_more=true, next_chunk=2 → read_url(url=..., chunk=2); "
        "fetch_url said chunks_needed=1 → read_url(url=...) with no chunk param is sufficient; "
        "NOT for: discovering page size or topic — call fetch_url first for that."
    ),
    parameters={
        "type": "object",
        "properties": {
            "url": {
                "type": "string",
                "description": "The URL to fetch. Must be an http:// or https:// URL.",
            },
            "chunk": {
                "type": "integer",
                "description": "Which 8,000-char chunk to return (1-indexed). Defaults to 1. Use next_chunk from a prior response to continue paginating.",
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
    total_chunks = math.ceil(total_chars / _WEB_DEFAULT_CHARS) if total_chars else 1

    if chunk > total_chunks:
        return json.dumps(
            {
                "error": (
                    f"chunk={chunk} is out of range — page has {total_chunks} chunk(s) "
                    f"({total_chars:,} chars total)."
                )
            }
        )

    start = (chunk - 1) * _WEB_DEFAULT_CHARS
    content = text[start : start + _WEB_DEFAULT_CHARS]

    result: dict = {
        "url": url,
        "title": title,
        "total_chars": total_chars,
        "chunk": chunk,
        "total_chunks": total_chunks,
        "chars_returned": len(content),
        "content": content,
    }
    if chunk < total_chunks:
        result["has_more"] = True
        result["next_chunk"] = chunk + 1

    return json.dumps(result, ensure_ascii=False)
