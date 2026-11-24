"""Safe Microsoft Edge desktop launcher and HTTP webpage text extractor."""

from __future__ import annotations

import html
import json
import re
import subprocess
from collections.abc import Callable
from typing import Any
from urllib.parse import urlparse

import httpx

EDGE_APP_NAME = "Microsoft Edge"

_ALLOWED_SCHEMES = frozenset({"http", "https"})
_SHELL_META_CHARS = frozenset(";|`$<>(){}\\\'\"\n\r\t ")
_HOSTNAME_RE = re.compile(r"^[A-Za-z0-9.\-:\[\]]+$")

_NOISE_BLOCK_RE = re.compile(
    r"<(script|style|noscript|svg|title)\b[^>]*>[\s\S]*?</\1>|<!--[\s\S]*?-->",
    re.IGNORECASE,
)
_TITLE_RE = re.compile(r"<title\b[^>]*>([\s\S]*?)</title>", re.IGNORECASE)
_BLOCK_TAG_RE = re.compile(
    r"</?(?:p|div|br|li|ul|ol|h[1-6]|tr|td|th|section|article|header|footer|blockquote|pre)\b[^>]*>",
    re.IGNORECASE,
)
_ANY_TAG_RE = re.compile(r"<[^>]+>")


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0:
        return ""
    if len(text) <= max_chars:
        return text
    if max_chars <= 3:
        return text[:max_chars]
    return text[: max_chars - 3].rstrip() + "..."


def extract_html_summary(raw_html: str, max_chars: int = 2000) -> tuple[str | None, str]:
    """Strip scripts/styles/tags from HTML and return (title, clean_text_excerpt)."""
    title: str | None = None
    title_match = _TITLE_RE.search(raw_html)
    if title_match:
        raw_title = _ANY_TAG_RE.sub(" ", title_match.group(1))
        cleaned_title = re.sub(r"\s+", " ", html.unescape(raw_title)).strip()
        if cleaned_title:
            title = cleaned_title

    cleaned = _NOISE_BLOCK_RE.sub(" ", raw_html)
    cleaned = _BLOCK_TAG_RE.sub(" ", cleaned)
    cleaned = _ANY_TAG_RE.sub("", cleaned)
    cleaned = html.unescape(cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return title, truncate_text(cleaned, max_chars)


def validate_http_url(url: str) -> str:
    """Validate that url is an http:// or https:// URL free of shell metacharacters."""
    if not isinstance(url, str):
        raise ValueError("URL must be a string.")
    cleaned = url.strip()
    if not cleaned:
        raise ValueError("A non-empty http:// or https:// URL is required.")
    if any(ord(ch) < 32 or ord(ch) == 127 or ch in _SHELL_META_CHARS for ch in cleaned):
        raise ValueError("URL contains disallowed whitespace, control characters, or shell metacharacters.")
    if "&&" in cleaned or "||" in cleaned:
        raise ValueError("URL contains disallowed shell command chaining operators.")

    parsed = urlparse(cleaned)
    scheme = parsed.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES or not cleaned.lower().startswith(f"{scheme}://"):
        raise ValueError("Only http:// and https:// URLs are allowed.")
    if not parsed.netloc or not parsed.hostname:
        raise ValueError("URL must include a valid host.")
    if not _HOSTNAME_RE.match(parsed.hostname):
        raise ValueError("URL hostname contains disallowed characters.")
    # Accessing parsed.port validates that any port number is numeric and in 0..65535.
    _ = parsed.port
    return cleaned


def build_edge_open_command(url: str) -> list[str]:
    """Build the macOS `open -a \"Microsoft Edge\" <url>` argv list without shell expansion."""
    validated_url = validate_http_url(url)
    return ["open", "-a", EDGE_APP_NAME, validated_url]


def open_in_edge(
    url: str,
    *,
    dry_run: bool = False,
    runner: Callable[[list[str]], Any] | None = None,
) -> str:
    """Validate an http/https URL and open it in Microsoft Edge, returning status JSON."""
    cmd = build_edge_open_command(url)
    validated_url = cmd[-1]
    if dry_run:
        return json.dumps(
            {
                "status": "dry_run",
                "browser": EDGE_APP_NAME,
                "url": validated_url,
                "command": cmd,
            },
            ensure_ascii=False,
        )

    if runner is not None:
        runner(cmd)
    else:
        subprocess.run(cmd, check=True, capture_output=True, text=True)

    return json.dumps(
        {
            "status": "opened",
            "browser": EDGE_APP_NAME,
            "url": validated_url,
            "command": cmd,
        },
        ensure_ascii=False,
    )


def _format_response_text(resp: httpx.Response, max_chars: int) -> tuple[str | None, str]:
    content_type = resp.headers.get("content-type", "").lower()
    body = resp.text
    if "html" in content_type or (not content_type and "<html" in body.lower()):
        return extract_html_summary(body, max_chars=max_chars)
    return None, truncate_text(re.sub(r"\s+", " ", body).strip(), max_chars)


def fetch_webpage_text(
    url: str,
    max_chars: int = 2000,
    *,
    client: httpx.Client | None = None,
) -> str:
    """Validate URL, fetch via httpx, and return clean readable text."""
    validated_url = validate_http_url(url)
    limit = max(0, int(max_chars))
    owns_client = client is None
    http = client or httpx.Client(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "xiao-speaker-assistant/0.1"},
    )
    try:
        resp = http.get(validated_url)
        resp.raise_for_status()
    finally:
        if owns_client:
            http.close()

    title, excerpt = _format_response_text(resp, max_chars=limit)
    if title and excerpt and not excerpt.startswith(title):
        return truncate_text(f"{title}\n{excerpt}", limit)
    return excerpt or (truncate_text(title, limit) if title else "")


async def async_fetch_webpage_summary(
    url: str,
    max_chars: int = 2000,
    *,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Async helper returning structured summary metadata for MCP server tool calls."""
    validated_url = validate_http_url(url)
    limit = max(0, int(max_chars))
    owns_client = client is None
    http = client or httpx.AsyncClient(
        timeout=15.0,
        follow_redirects=True,
        headers={"User-Agent": "xiao-speaker-assistant/0.1"},
    )
    try:
        resp = await http.get(validated_url)
        resp.raise_for_status()
    finally:
        if owns_client:
            await http.aclose()

    title, excerpt = _format_response_text(resp, max_chars=limit)
    return {
        "source": "http_fetch",
        "url": str(resp.url),
        "status_code": resp.status_code,
        "title": title,
        "excerpt": excerpt,
    }
