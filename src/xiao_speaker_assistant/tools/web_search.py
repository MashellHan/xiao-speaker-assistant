"""Pluggable web search providers (Tavily API, SearXNG, and Bing HTML fallback)."""

from __future__ import annotations

import html
import logging
import re
from typing import Any

import httpx

from xiao_speaker_assistant.config import SearchToolConfig, read_env_secret
from xiao_speaker_assistant.tools.browser import validate_http_url

logger = logging.getLogger(__name__)

_BING_CAPTCHA_MARKERS = (
    "captcha",
    "cf-challenge",
    "challenge-platform",
    "unusual traffic",
    "验证您不是机器人",
    "人机验证",
)


def resolve_search_provider(config: SearchToolConfig) -> str:
    """Resolve effective search backend (`tavily`, `searxng`, or `bing`)."""
    pref = (config.provider or "auto").strip().lower()
    if pref == "tavily":
        return "tavily"
    if pref == "searxng":
        return "searxng"
    if pref == "bing":
        return "bing"
    # Auto mode: prefer structured JSON APIs when configured, else zero-key Bing fallback.
    if read_env_secret(config.tavily_api_key_env):
        return "tavily"
    if (config.searxng_url or "").strip() or read_env_secret(config.searxng_url_env):
        return "searxng"
    return "bing"


async def _search_tavily(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    api_key: str,
) -> dict[str, Any]:
    resp = await client.post(
        "https://api.tavily.com/search",
        json={
            "api_key": api_key,
            "query": query,
            "max_results": limit,
            "search_depth": "basic",
            "include_answer": False,
        },
    )
    resp.raise_for_status()
    data = resp.json() if isinstance(resp.json(), dict) else {}
    raw_results = data.get("results") or []
    items: list[dict[str, str]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        snippet = str(entry.get("content") or entry.get("snippet") or "").strip()
        if title or snippet:
            items.append(
                {
                    "title": title or url,
                    "snippet": snippet[:280],
                    "url": url,
                }
            )
        if len(items) >= limit:
            break
    return {"query": query, "provider": "tavily", "results": items, "count": len(items)}


async def _search_searxng(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
    base_url: str,
) -> dict[str, Any]:
    validated = validate_http_url(base_url, block_private_network=False)
    endpoint = validated.rstrip("/")
    if not endpoint.endswith("/search"):
        endpoint = f"{endpoint}/search"
    resp = await client.get(
        endpoint,
        params={"q": query, "format": "json", "language": "zh-CN"},
        headers={"Accept": "application/json"},
    )
    resp.raise_for_status()
    data = resp.json() if isinstance(resp.json(), dict) else {}
    raw_results = data.get("results") or []
    items: list[dict[str, str]] = []
    for entry in raw_results:
        if not isinstance(entry, dict):
            continue
        title = str(entry.get("title") or "").strip()
        url = str(entry.get("url") or "").strip()
        snippet = str(entry.get("content") or entry.get("snippet") or "").strip()
        if title or snippet:
            items.append(
                {
                    "title": title or url,
                    "snippet": snippet[:280],
                    "url": url,
                }
            )
        if len(items) >= limit:
            break
    return {"query": query, "provider": "searxng", "results": items, "count": len(items)}


async def _search_bing(
    client: httpx.AsyncClient,
    query: str,
    *,
    limit: int,
) -> dict[str, Any]:
    resp = await client.get(
        "https://www.bing.com/search",
        params={"q": query, "mkt": "zh-CN"},
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36",
            "Accept-Language": "zh-CN,zh;q=0.9",
        },
    )
    resp.raise_for_status()
    body = resp.text
    items: list[dict[str, str]] = []
    for block in body.split('<li class="b_algo"')[1:]:
        m_link = re.search(
            r'<h2[^>]*>[\s\S]*?<a[^>]+href="([^"]+)"[^>]*>([\s\S]*?)</a>', block
        )
        if not m_link:
            continue
        url = html.unescape(m_link.group(1))
        title = html.unescape(re.sub(r"<[^>]+>", "", m_link.group(2))).strip()
        m_snip = re.search(r"<p[^>]*>([\s\S]*?)</p>", block)
        snippet = (
            html.unescape(re.sub(r"<[^>]+>", "", m_snip.group(1))).strip()
            if m_snip
            else ""
        )
        if title:
            items.append({"title": title, "snippet": snippet[:240], "url": url})
        if len(items) >= limit:
            break

    result: dict[str, Any] = {
        "query": query,
        "provider": "bing",
        "results": items,
        "count": len(items),
    }
    if not items:
        lower_body = body.lower()
        if any(marker in lower_body for marker in _BING_CAPTCHA_MARKERS):
            result["warning"] = (
                "bing_captcha_or_blocked: Bing returned an anti-bot challenge; "
                "set TAVILY_API_KEY or SEARXNG_URL for reliable search."
            )
        else:
            result["warning"] = (
                "bing_empty_results: Bing HTML parsing returned 0 results; "
                "consider configuring TAVILY_API_KEY or SEARXNG_URL."
            )
    return result


async def run_web_search(
    query: str,
    *,
    limit: int = 5,
    config: SearchToolConfig | None = None,
    client: httpx.AsyncClient | None = None,
) -> dict[str, Any]:
    """Execute a web search using Tavily, SearXNG, or Bing HTML fallback."""
    cfg = config or SearchToolConfig()
    provider = resolve_search_provider(cfg)
    owns_client = client is None
    http = client or httpx.AsyncClient(timeout=8.0, follow_redirects=True)
    try:
        if provider == "tavily":
            api_key = read_env_secret(cfg.tavily_api_key_env)
            if not api_key:
                if cfg.provider == "tavily":
                    raise RuntimeError(
                        f"Missing Tavily API key in env {cfg.tavily_api_key_env}"
                    )
                provider = "bing"
            else:
                try:
                    return await _search_tavily(http, query, limit=limit, api_key=api_key)
                except Exception as exc:
                    if cfg.provider == "tavily":
                        raise
                    logger.warning("Tavily search failed (%s); falling back to Bing", exc)
                    provider = "bing"

        if provider == "searxng":
            searxng_url = (cfg.searxng_url or "").strip() or (
                read_env_secret(cfg.searxng_url_env) or ""
            )
            if not searxng_url:
                if cfg.provider == "searxng":
                    raise RuntimeError(
                        f"Missing SearXNG URL in tools.search.searxng_url or env {cfg.searxng_url_env}"
                    )
                provider = "bing"
            else:
                try:
                    return await _search_searxng(http, query, limit=limit, base_url=searxng_url)
                except Exception as exc:
                    if cfg.provider == "searxng":
                        raise
                    logger.warning("SearXNG search failed (%s); falling back to Bing", exc)
                    provider = "bing"

        return await _search_bing(http, query, limit=limit)
    finally:
        if owns_client:
            await http.aclose()
