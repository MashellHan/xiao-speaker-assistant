"""Stdio JSON-RPC 2.0 MCP server exposing Outlook, Xiaohongshu, and Browser tools."""

from __future__ import annotations

import asyncio
import html
import json
import logging
import re
import sys
from typing import Any, TextIO
from urllib.parse import urlparse

import httpx

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.tools.confirmation import ConfirmationStore
from xiao_speaker_assistant.tools.mcp_client import HttpMcpClient
from xiao_speaker_assistant.tools.outlook import OutlookClient, PendingEmailDraft

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "xiao-speaker-assistant", "version": "0.1.0"}

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


TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "outlook_list_emails",
        "description": "List recent Outlook emails with subject, sender, timestamp, and preview.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "minimum": 1, "maximum": 50, "default": 5},
                "unread_only": {"type": "boolean", "default": False},
            },
        },
    },
    {
        "name": "outlook_send_email",
        "description": (
            "Send an email via Outlook. Guarded by require_voice_confirmation_for_send: "
            "unless confirmed=true (or confirmation is disabled), queues the draft and returns "
            "a confirmation_required status."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "to": {"type": "string", "description": "Recipient email address"},
                "subject": {"type": "string", "description": "Email subject"},
                "body": {"type": "string", "description": "Plain-text email body"},
                "confirmed": {
                    "type": "boolean",
                    "default": False,
                    "description": "True only after explicit user voice confirmation",
                },
            },
            "required": ["to", "subject", "body"],
        },
    },
    {
        "name": "xiaohongshu_search",
        "description": "Search Xiaohongshu notes by keyword via Xiaohongshu MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "keyword": {"type": "string", "description": "Search keyword"},
            },
            "required": ["keyword"],
        },
    },
    {
        "name": "xiaohongshu_detail",
        "description": "Fetch full detail of a Xiaohongshu note via Xiaohongshu MCP.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "feed_id": {"type": "string", "description": "Xiaohongshu feed ID"},
                "xsec_token": {"type": "string", "description": "xsec_token from search results"},
            },
            "required": ["feed_id"],
        },
    },
    {
        "name": "browser_fetch_summary",
        "description": (
            "Fetch a web page summary: delegates to Browser MCP when enabled, "
            "otherwise fetches via HTTP and strips HTML tags into a clean text excerpt."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL to fetch"},
                "max_chars": {"type": "integer", "minimum": 100, "maximum": 20000, "default": 2000},
            },
            "required": ["url"],
        },
    },
]


class McpServer:
    """Stdio JSON-RPC 2.0 MCP server for Xiao Speaker Assistant tools."""

    def __init__(
        self,
        config: AppConfig | None = None,
        *,
        outlook_client: OutlookClient | None = None,
        xiaohongshu_client: HttpMcpClient | None = None,
        browser_client: HttpMcpClient | None = None,
        confirmation_store: ConfirmationStore | None = None,
        http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self._config = config or AppConfig()
        self._outlook = outlook_client or OutlookClient(self._config.tools.outlook)
        self._xiaohongshu = xiaohongshu_client or HttpMcpClient(
            self._config.tools.xiaohongshu.mcp_url
        )
        self._browser = browser_client or HttpMcpClient(self._config.tools.browser.mcp_url)
        self.confirmation_store = confirmation_store or ConfirmationStore()
        self._http_client = http_client

    @staticmethod
    def _ok(req_id: Any, result: Any) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "result": result}

    @staticmethod
    def _error(req_id: Any, code: int, message: str) -> dict[str, Any]:
        return {"jsonrpc": "2.0", "id": req_id, "error": {"code": code, "message": message}}

    @staticmethod
    def _tool_result(payload: Any, *, is_error: bool = False) -> dict[str, Any]:
        text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
        return {
            "content": [{"type": "text", "text": text}],
            "isError": is_error,
        }

    async def handle_line(self, line: str) -> str | None:
        stripped = line.strip()
        if not stripped:
            return None
        try:
            message = json.loads(stripped)
        except json.JSONDecodeError:
            return json.dumps(self._error(None, -32700, "Parse error"), ensure_ascii=False)
        if not isinstance(message, dict):
            return json.dumps(self._error(None, -32600, "Invalid Request"), ensure_ascii=False)
        response = await self.handle_message(message)
        if response is None:
            return None
        return json.dumps(response, ensure_ascii=False)

    async def handle_message(self, message: dict[str, Any]) -> dict[str, Any] | None:
        has_id = "id" in message
        req_id = message.get("id")
        method = message.get("method")
        params = message.get("params") or {}

        if not isinstance(method, str):
            return self._error(req_id, -32600, "Invalid Request") if has_id else None

        # JSON-RPC notifications omit `id` and must not receive a response.
        if not has_id:
            logger.debug("Received MCP notification: %s", method)
            return None

        if method == "initialize":
            return self._ok(
                req_id,
                {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {"listChanged": False}},
                    "serverInfo": SERVER_INFO,
                },
            )
        if method == "ping":
            return self._ok(req_id, {})
        if method == "tools/list":
            return self._ok(req_id, {"tools": TOOL_DEFINITIONS})
        if method == "tools/call":
            if not isinstance(params, dict):
                return self._error(req_id, -32602, "Invalid params")
            name = params.get("name")
            arguments = params.get("arguments") or {}
            if not isinstance(name, str) or not isinstance(arguments, dict):
                return self._error(req_id, -32602, "Invalid params: expected tool name and arguments")
            return await self._dispatch_tool_call(req_id, name, arguments)

        return self._error(req_id, -32601, f"Method not found: {method}")

    async def _dispatch_tool_call(
        self, req_id: Any, name: str, args: dict[str, Any]
    ) -> dict[str, Any]:
        if name == "outlook_list_emails":
            return self._ok(req_id, await self._call_outlook_list_emails(args))
        if name == "outlook_send_email":
            return self._ok(req_id, await self._call_outlook_send_email(args))
        if name == "xiaohongshu_search":
            return self._ok(req_id, await self._call_xiaohongshu_search(args))
        if name == "xiaohongshu_detail":
            return self._ok(req_id, await self._call_xiaohongshu_detail(args))
        if name == "browser_fetch_summary":
            return self._ok(req_id, await self._call_browser_fetch_summary(args))
        return self._error(req_id, -32602, f"Unknown tool: {name}")

    async def _call_outlook_list_emails(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.outlook.enabled:
            return self._tool_result(
                {
                    "error": "outlook_disabled",
                    "message": "Outlook tool is disabled. Set tools.outlook.enabled: true in config.yaml.",
                },
                is_error=True,
            )
        limit = max(1, min(50, int(args.get("limit", 5))))
        unread_only = bool(args.get("unread_only", False))
        try:
            emails = await self._outlook.list_recent_emails(limit=limit, unread_only=unread_only)
        except Exception as exc:
            return self._tool_result(
                {"error": "outlook_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result({"emails": emails, "count": len(emails)})

    async def _call_outlook_send_email(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.outlook.enabled:
            return self._tool_result(
                {
                    "error": "outlook_disabled",
                    "message": "Outlook tool is disabled. Set tools.outlook.enabled: true in config.yaml.",
                },
                is_error=True,
            )
        to = str(args.get("to") or "").strip()
        subject = str(args.get("subject") or "").strip()
        body = str(args.get("body") or "").strip()
        if not to or not subject or not body:
            return self._tool_result(
                {
                    "error": "invalid_arguments",
                    "message": "Arguments 'to', 'subject', and 'body' are required.",
                },
                is_error=True,
            )

        confirmed = bool(args.get("confirmed", False))
        draft = PendingEmailDraft(to=to, subject=subject, body=body, confirmed=confirmed)
        if self._config.tools.outlook.require_voice_confirmation_for_send and not confirmed:
            self.confirmation_store.set_pending(draft)
            result_json = await self._outlook.send_email(draft)
            return self._tool_result(result_json, is_error=False)

        try:
            result_json = await self._outlook.send_email(draft)
        except Exception as exc:
            return self._tool_result(
                {"error": "outlook_error", "message": str(exc)},
                is_error=True,
            )
        self.confirmation_store.clear()
        return self._tool_result(result_json, is_error=False)

    def _xiaohongshu_disabled_result(self) -> dict[str, Any]:
        mcp_url = self._config.tools.xiaohongshu.mcp_url
        return self._tool_result(
            {
                "error": "xiaohongshu_mcp_not_enabled",
                "message": (
                    "Xiaohongshu MCP is not enabled. Set tools.xiaohongshu.enabled: true "
                    f"in config.yaml and start the Xiaohongshu MCP server at {mcp_url}."
                ),
                "mcp_url": mcp_url,
            },
            is_error=True,
        )

    async def _call_xiaohongshu_search(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.xiaohongshu.enabled:
            return self._xiaohongshu_disabled_result()
        keyword = str(args.get("keyword") or "").strip()
        if not keyword:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'keyword' is required."},
                is_error=True,
            )
        try:
            res = await self._xiaohongshu.call_tool("search_feeds", {"keyword": keyword})
        except Exception as exc:
            return self._tool_result(
                {"error": "xiaohongshu_mcp_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(res)

    async def _call_xiaohongshu_detail(self, args: dict[str, Any]) -> dict[str, Any]:
        if not self._config.tools.xiaohongshu.enabled:
            return self._xiaohongshu_disabled_result()
        feed_id = str(args.get("feed_id") or "").strip()
        if not feed_id:
            return self._tool_result(
                {"error": "invalid_arguments", "message": "Argument 'feed_id' is required."},
                is_error=True,
            )
        payload: dict[str, Any] = {"feed_id": feed_id}
        xsec_token = str(args.get("xsec_token") or "").strip()
        if xsec_token:
            payload["xsec_token"] = xsec_token
        try:
            res = await self._xiaohongshu.call_tool("get_feed_detail", payload)
        except Exception as exc:
            return self._tool_result(
                {"error": "xiaohongshu_mcp_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result(res)

    async def _call_browser_fetch_summary(self, args: dict[str, Any]) -> dict[str, Any]:
        url = str(args.get("url") or "").strip()
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            return self._tool_result(
                {"error": "invalid_url", "message": "A valid http:// or https:// URL is required."},
                is_error=True,
            )
        max_chars = max(100, min(20000, int(args.get("max_chars", 2000))))

        if self._config.tools.browser.enabled:
            try:
                res = await self._browser.call_tool("browser_navigate", {"url": url})
            except Exception as exc:
                return self._tool_result(
                    {"error": "browser_mcp_error", "message": str(exc)},
                    is_error=True,
                )
            return self._tool_result({"source": "browser_mcp", "url": url, "result": res})

        owns_client = self._http_client is None
        client = self._http_client or httpx.AsyncClient(
            timeout=15.0,
            follow_redirects=True,
            headers={"User-Agent": "xiao-speaker-assistant/0.1"},
        )
        try:
            resp = await client.get(url)
            resp.raise_for_status()
        except Exception as exc:
            return self._tool_result(
                {"error": "http_fetch_error", "message": str(exc), "url": url},
                is_error=True,
            )
        finally:
            if owns_client:
                await client.aclose()

        content_type = resp.headers.get("content-type", "").lower()
        body = resp.text
        if "html" in content_type or (not content_type and "<html" in body.lower()):
            title, excerpt = extract_html_summary(body, max_chars=max_chars)
        else:
            title = None
            excerpt = truncate_text(re.sub(r"\s+", " ", body).strip(), max_chars)

        return self._tool_result(
            {
                "source": "http_fetch",
                "url": str(resp.url),
                "status_code": resp.status_code,
                "title": title,
                "excerpt": excerpt,
            }
        )

    async def serve_stdio(
        self,
        stdin: TextIO | None = None,
        stdout: TextIO | None = None,
    ) -> None:
        in_stream = stdin or sys.stdin
        out_stream = stdout or sys.stdout
        while True:
            line = await asyncio.to_thread(in_stream.readline)
            if not line:
                break
            response = await self.handle_line(line)
            if response is not None:
                out_stream.write(response + "\n")
                out_stream.flush()
