"""Minimal Streamable-HTTP MCP client for Xiaohongshu and Browser MCP servers."""

from __future__ import annotations

import itertools
from typing import Any

import httpx


class HttpMcpClient:
    def __init__(self, endpoint_url: str, auth_token: str | None = None) -> None:
        self._url = endpoint_url
        self._auth_token = auth_token
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
        payload = {
            "jsonrpc": "2.0",
            "id": next(self._ids),
            "method": method,
            "params": params or {},
        }
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(self._url, headers=self._headers(), json=payload)
            resp.raise_for_status()
            session = resp.headers.get("Mcp-Session-Id")
            if session:
                self._session_id = session
            data = resp.json()
        if "error" in data:
            raise RuntimeError(f"MCP error: {data['error']}")
        return data.get("result", {})

    async def initialize(self) -> dict[str, Any]:
        return await self._rpc(
            "initialize",
            {
                "protocolVersion": "2024-11-05",
                "capabilities": {},
                "clientInfo": {"name": "xiao-speaker-assistant", "version": "0.1.0"},
            },
        )

    async def list_tools(self) -> list[dict[str, Any]]:
        if not self._session_id:
            await self.initialize()
        res = await self._rpc("tools/list")
        return res.get("tools", [])

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any:
        if not self._session_id:
            await self.initialize()
        return await self._rpc("tools/call", {"name": name, "arguments": arguments})
