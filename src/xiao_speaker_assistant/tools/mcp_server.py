"""Stdio JSON-RPC 2.0 MCP server exposing Outlook, Xiaohongshu, and Browser tools."""

from __future__ import annotations

import asyncio
import json
import logging
import subprocess
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TextIO

import httpx

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.tools.browser import (
    async_fetch_webpage_summary,
    extract_html_summary,
    open_in_edge,
    truncate_text,
    validate_http_url,
)
from xiao_speaker_assistant.tools.confirmation import ConfirmationStore
from xiao_speaker_assistant.tools.mac_local import (
    MacLocalAccessError,
    list_recent_files,
    read_local_text_file,
)
from xiao_speaker_assistant.tools.mcp_client import HttpMcpClient
from xiao_speaker_assistant.tools.outlook import OutlookClient, PendingEmailDraft

logger = logging.getLogger(__name__)

MCP_PROTOCOL_VERSION = "2024-11-05"
SERVER_INFO = {"name": "xiao-speaker-assistant", "version": "0.1.0"}


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

LOCAL_TOOL_DEFINITIONS: list[dict[str, Any]] = [
    {
        "name": "browser_open_edge",
        "description": "Validate an HTTP/HTTPS URL and open it in Microsoft Edge on macOS.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "HTTP/HTTPS URL to open in Microsoft Edge"},
                "dry_run": {
                    "type": "boolean",
                    "default": False,
                    "description": "Validate and return command without launching Edge",
                },
            },
            "required": ["url"],
        },
    },
    {
        "name": "mac_list_recent_files",
        "description": (
            "List recent non-sensitive files inside allowed macOS user directories "
            "(~/Desktop, ~/Downloads, ~/Documents)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "directory": {
                    "type": "string",
                    "default": "~/Desktop",
                    "description": "Target directory inside allowed user roots",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            },
        },
    },
    {
        "name": "mac_read_text_file",
        "description": (
            "Read a UTF-8 text file inside allowed macOS user directories, "
            "blocking traversal and sensitive files or keychains."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "File path inside allowed user roots"},
                "max_chars": {"type": "integer", "minimum": 1, "maximum": 20000, "default": 3000},
            },
            "required": ["path"],
        },
    },
]

ALL_TOOL_DEFINITIONS: list[dict[str, Any]] = [*TOOL_DEFINITIONS, *LOCAL_TOOL_DEFINITIONS]


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
        include_local_tools: bool = False,
        mac_allowed_roots: Sequence[str | Path] | None = None,
        edge_dry_run: bool = False,
    ) -> None:
        self._config = config or AppConfig()
        self._outlook = outlook_client or OutlookClient(self._config.tools.outlook)
        self._xiaohongshu = xiaohongshu_client or HttpMcpClient(
            self._config.tools.xiaohongshu.mcp_url
        )
        self._browser = browser_client or HttpMcpClient(self._config.tools.browser.mcp_url)
        self.confirmation_store = confirmation_store or ConfirmationStore()
        self._http_client = http_client
        self._include_local_tools = include_local_tools
        self._mac_allowed_roots = mac_allowed_roots
        self._edge_dry_run = edge_dry_run

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
            tools = (
                ALL_TOOL_DEFINITIONS
                if (
                    self._include_local_tools
                    or self._config.tools.browser.enabled
                    or self._mac_allowed_roots is not None
                )
                else TOOL_DEFINITIONS
            )
            return self._ok(req_id, {"tools": tools})
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
        if name == "browser_open_edge":
            return self._ok(req_id, await self._call_browser_open_edge(args))
        if name == "mac_list_recent_files":
            return self._ok(req_id, await self._call_mac_list_recent_files(args))
        if name == "mac_read_text_file":
            return self._ok(req_id, await self._call_mac_read_text_file(args))
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
        raw_url = str(args.get("url") or "").strip()
        try:
            url = validate_http_url(raw_url)
        except ValueError as exc:
            return self._tool_result(
                {"error": "invalid_url", "message": str(exc)},
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

        try:
            summary = await async_fetch_webpage_summary(
                url,
                max_chars=max_chars,
                client=self._http_client,
            )
        except Exception as exc:
            return self._tool_result(
                {"error": "http_fetch_error", "message": str(exc), "url": url},
                is_error=True,
            )
        return self._tool_result(summary)

    async def _call_browser_open_edge(self, args: dict[str, Any]) -> dict[str, Any]:
        raw_url = str(args.get("url") or "").strip()
        dry_run = bool(args.get("dry_run", self._edge_dry_run))
        try:
            result_json = await asyncio.to_thread(open_in_edge, raw_url, dry_run=dry_run)
        except ValueError as exc:
            return self._tool_result(
                {"error": "invalid_url", "message": str(exc)},
                is_error=True,
            )
        except (subprocess.SubprocessError, OSError) as exc:
            return self._tool_result(
                {"error": "edge_launch_error", "message": str(exc), "url": raw_url},
                is_error=True,
            )
        return self._tool_result(result_json)

    async def _call_mac_list_recent_files(self, args: dict[str, Any]) -> dict[str, Any]:
        directory = str(args.get("directory") or "~/Desktop").strip()
        limit = max(1, min(100, int(args.get("limit", 10))))
        try:
            files = await asyncio.to_thread(
                list_recent_files,
                directory,
                limit,
                allowed_roots=self._mac_allowed_roots,
            )
        except MacLocalAccessError as exc:
            return self._tool_result(
                {"error": "access_denied", "message": str(exc)},
                is_error=True,
            )
        except (FileNotFoundError, NotADirectoryError, ValueError) as exc:
            return self._tool_result(
                {"error": "mac_local_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result({"directory": directory, "files": files, "count": len(files)})

    async def _call_mac_read_text_file(self, args: dict[str, Any]) -> dict[str, Any]:
        path = str(args.get("path") or "").strip()
        max_chars = max(1, min(20000, int(args.get("max_chars", 3000))))
        try:
            content = await asyncio.to_thread(
                read_local_text_file,
                path,
                max_chars,
                allowed_roots=self._mac_allowed_roots,
            )
        except MacLocalAccessError as exc:
            return self._tool_result(
                {"error": "access_denied", "message": str(exc)},
                is_error=True,
            )
        except (FileNotFoundError, IsADirectoryError, ValueError) as exc:
            return self._tool_result(
                {"error": "mac_local_error", "message": str(exc)},
                is_error=True,
            )
        return self._tool_result({"path": path, "content": content, "chars": len(content)})

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
