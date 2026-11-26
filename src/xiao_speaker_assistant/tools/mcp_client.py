"""Minimal Streamable-HTTP MCP client for Xiaohongshu and Browser MCP servers."""

from __future__ import annotations

import itertools
import json
from typing import Any

import httpx


def parse_mcp_http_response(
    content_type: str, text: str, *, expected_id: int | str | None = None
) -> dict[str, Any]:
    """Parse Streamable-HTTP MCP responses in either JSON or SSE (`text/event-stream`) format."""
    ct = (content_type or "").lower()
    if "text/event-stream" in ct or text.lstrip().startswith("event:") or text.lstrip().startswith("data:"):
        data_lines: list[str] = []
        payloads: list[dict[str, Any]] = []

        def _flush() -> None:
            if not data_lines:
                return
            candidate = "\n".join(data_lines)
            data_lines.clear()
            try:
                parsed = json.loads(candidate)
                if isinstance(parsed, dict):
                    payloads.append(parsed)
            except json.JSONDecodeError:
                pass

        for raw_line in text.splitlines():
            line = raw_line.rstrip("\r\n")
            if not line.strip():
                _flush()
                continue
            if line.startswith("data:"):
                data_lines.append(line[5:].removeprefix(" "))
        _flush()

        if expected_id is not None:
            for item in payloads:
                if item.get("id") == expected_id and ("result" in item or "error" in item):
                    return item
        for item in reversed(payloads):
            if "result" in item or "error" in item:
                return item
        raise ValueError(f"Empty or invalid SSE MCP response payload: {text[:200]!r}")

    parsed = json.loads(text)
    if not isinstance(parsed, dict):
        raise ValueError(f"Expected JSON object from MCP endpoint, got {type(parsed).__name__}")
    return parsed


class HttpMcpClient:
    def __init__(
        self,
        endpoint_url: str,
        auth_token: str | None = None,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> None:
        self._url = endpoint_url
        self._auth_token = auth_token
        self._transport = transport
        self._ids = itertools.count(1)
        self._session_id: str | None = None

    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "Accept": "application/json, text/event-stream",
        }
        if self._auth_token:
            headers["Authorization"] = f"Bearer {self._auth_token}"
        if self._session_id:
            headers["Mcp-Session-Id"] = self._session_id
        return headers

    async def _rpc(self, method: str, params: dict[str, Any] | None = None) -> dict[str, Any]:
        req_id = next(self._ids)
        payload = {
            "jsonrpc": "2.0",
            "id": req_id,
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=30.0, transport=self._transport) as client:
            resp = await client.post(self._url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            session = resp.headers.get("Mcp-Session-Id")
            if session:
                self._session_id = session
            data = parse_mcp_http_response(
                resp.headers.get("content-type", ""), resp.text, expected_id=req_id
            )
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def initialize(self) -> dict[str, Any]:
        res = await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "xiao-speaker-assistant", "version": "0.1.0"},
            },
        )
        # Send notifications/initialized per MCP Streamable-HTTP spec (best-effort)
        async with httpx.AsyncClient(timeout=10.0, transport=self._transport) as client:
            try:
                await client.post(
                    self._url,
                    headers=self._headers(),
                    json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                )
            except httpx.HTTPError:
                pass
        return res

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._session_id:
            await self.initialize()
        res = await self._rpc("tools/list")
        return res.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._session_id:
            await self.initialize()
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})
