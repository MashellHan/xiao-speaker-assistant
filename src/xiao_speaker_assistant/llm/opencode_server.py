"""Managed OpenCode V2 HTTP server lifecycle and SSE streaming client."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import secrets
from collections.abc import AsyncIterator
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

import httpx

from xiao_speaker_assistant.config import OpencodeCliConfig, read_env_secret

logger = logging.getLogger(__name__)


class SessionNotFoundError(RuntimeError):
    """Raised when an OpenCode session ID is no longer present on the server."""


def parse_model_ref(model_str: str) -> dict[str, str]:
    """Convert `provider/model[#variant]` into OpenCode V2 `Model.Ref` JSON."""
    raw = model_str.strip()
    if "/" not in raw:
        raise ValueError(f"OpenCode model must be in 'provider/model' format, got: {model_str!r}")
    provider_id, rest = raw.split("/", 1)
    if "#" in rest:
        model_id, variant = rest.split("#", 1)
        return {"providerID": provider_id, "id": model_id, "variant": variant}
    return {"providerID": provider_id, "id": rest}


def build_opencode_v2_config(
    opencode_cfg: OpencodeCliConfig,
    *,
    mcp_url: str,
    system_prompt: str,
    mcp_auth_token: str | None = None,
) -> dict[str, Any]:
    """Generate isolated OpenCode V2 `opencode.jsonc` config for the `nomi` voice agent."""
    mcp_entry: dict[str, Any] = {
        "type": "remote",
        "url": mcp_url,
        "oauth": False,
        "codemode": False,
        "protocol": "legacy",
    }
    if mcp_auth_token:
        mcp_entry["headers"] = {"Authorization": f"Bearer {mcp_auth_token}"}

    return {
        "$schema": "https://opencode.ai/config.json",
        "model": opencode_cfg.model,
        "default_agent": "nomi",
        "snapshots": False,
        "update": "disable",
        "agents": {
            "nomi": {
                "description": "Home voice assistant Nomi for Xiaomi Smart Speaker Pro",
                "mode": "primary",
                "model": opencode_cfg.model,
                "system": system_prompt,
                "permissions": [
                    {"action": "xsa_tools_*", "resource": "*", "effect": "allow"},
                    {"action": "websearch", "resource": "*", "effect": "allow"},
                    {"action": "webfetch", "resource": "*", "effect": "allow"},
                    {"action": "edit", "resource": "*", "effect": "deny"},
                    {"action": "write", "resource": "*", "effect": "deny"},
                    {"action": "shell", "resource": "*", "effect": "deny"},
                ],
            }
        },
        "mcp": {
            "servers": {
                "xsa_tools": mcp_entry,
            }
        },
    }


def ensure_agent_home(
    opencode_cfg: OpencodeCliConfig,
    *,
    mcp_url: str,
    system_prompt: str,
    mcp_auth_token: str | None = None,
) -> Path:
    """Create the isolated agent-home directory and write its `opencode.jsonc`."""
    home = Path(opencode_cfg.agent_home).expanduser().resolve()
    home.mkdir(parents=True, exist_ok=True)
    cfg_payload = build_opencode_v2_config(
        opencode_cfg,
        mcp_url=mcp_url,
        system_prompt=system_prompt,
        mcp_auth_token=mcp_auth_token,
    )
    (home / "opencode.jsonc").write_text(
        json.dumps(cfg_payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return home


def load_or_create_server_password(agent_home: Path, password_env: str) -> str:
    """Read password from env, or persist/reuse a `0600` token file inside `agent_home`."""
    env_pw = read_env_secret(password_env)
    if env_pw:
        return env_pw
    return _load_or_create_token_file(agent_home, "server.token")


def load_or_create_mcp_token(agent_home: Path) -> str:
    """Persist/reuse a `0600` Bearer token file for the local HTTP MCP endpoint."""
    return _load_or_create_token_file(agent_home, "mcp.token")


def _load_or_create_token_file(agent_home: Path, filename: str) -> str:
    agent_home.mkdir(parents=True, exist_ok=True, mode=0o700)
    os.chmod(agent_home, 0o700)
    token_path = agent_home / filename
    if token_path.exists():
        existing = token_path.read_text(encoding="utf-8").strip()
        if existing:
            os.chmod(token_path, 0o600)
            return existing

    token = secrets.token_urlsafe(24)
    fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    try:
        os.write(fd, (token + "\n").encode("utf-8"))
    finally:
        os.close(fd)
    os.chmod(token_path, 0o600)
    return token


class OpenCodeV2Client:
    """Async HTTP + SSE client for OpenCode V2 (`opencode serve`)."""

    def __init__(
        self,
        base_url: str,
        *,
        password: str | None = None,
        client: httpx.AsyncClient | None = None,
        timeout_sec: float = 120.0,
    ) -> None:
        self._base_url = base_url.rstrip("/")
        self._auth = ("opencode", password) if password else None
        self._injected_client = client
        self._timeout_sec = timeout_sec

    def _make_client(self, *, timeout: float | None = None) -> tuple[httpx.AsyncClient, bool]:
        if self._injected_client is not None:
            return self._injected_client, False
        t = self._timeout_sec if timeout is None else timeout
        return (
            httpx.AsyncClient(
                base_url=self._base_url,
                auth=self._auth,
                timeout=httpx.Timeout(t, connect=10.0),
            ),
            True,
        )

    async def create_session(
        self,
        *,
        model: str,
        directory: str,
        agent: str = "nomi",
        title: str = "xsa-voice",
    ) -> str:
        payload: dict[str, Any] = {
            "title": title,
            "agent": agent,
            "model": parse_model_ref(model),
            "location": {"directory": directory},
        }
        client, owns = self._make_client(timeout=15.0)
        try:
            resp = await client.post("/api/session", json=payload)
            resp.raise_for_status()
            data = resp.json().get("data") or {}
            sid = data.get("id")
            if not isinstance(sid, str) or not sid:
                raise RuntimeError(f"OpenCode V2 create_session returned invalid payload: {resp.text[:200]}")
            return sid
        finally:
            if owns:
                await client.aclose()

    async def interrupt(self, session_id: str) -> bool:
        client, owns = self._make_client(timeout=3.0)
        try:
            resp = await client.post(f"/api/session/{session_id}/interrupt")
            if resp.status_code == 404:
                return False
            resp.raise_for_status()
            return bool(resp.json().get("interrupted", False))
        except Exception as exc:
            logger.debug("OpenCode V2 interrupt failed for %s: %s", session_id, exc)
            return False
        finally:
            if owns:
                await client.aclose()

    async def _reply_permission(
        self, client: httpx.AsyncClient, session_id: str, request_id: str, action: str
    ) -> None:
        reply = "once" if (action.startswith("xsa_tools_") or action in {"websearch", "webfetch"}) else "reject"
        try:
            await client.post(
                f"/api/session/{session_id}/permission/{request_id}/reply",
                json={"reply": reply},
            )
        except Exception as exc:
            logger.debug("Failed to reply permission %s: %s", request_id, exc)

    async def stream_prompt(self, session_id: str, text: str) -> AsyncIterator[str]:
        """Subscribe to `/api/event` before sending `/prompt` and yield text deltas."""
        params = [
            ("types", t)
            for t in (
                "session.text.delta",
                "session.execution.succeeded",
                "session.execution.failed",
                "session.execution.interrupted",
                "session.idle",
                "permission.asked",
            )
        ]
        client, owns = self._make_client(timeout=self._timeout_sec)
        try:
            async with client.stream("GET", "/api/event", params=params) as sse:
                sse.raise_for_status()
                prompt_sent = False

                async def send_prompt() -> None:
                    nonlocal prompt_sent
                    if prompt_sent:
                        return
                    prompt_sent = True
                    pr = await client.post(
                        f"/api/session/{session_id}/prompt",
                        json={"text": text},
                    )
                    if pr.status_code == 404:
                        raise SessionNotFoundError(f"OpenCode session not found: {session_id}")
                    pr.raise_for_status()

                async for raw_line in sse.aiter_lines():
                    if not raw_line.startswith("data:"):
                        continue
                    body = raw_line[5:].strip()
                    if not body:
                        continue
                    try:
                        evt = json.loads(body)
                    except json.JSONDecodeError:
                        continue

                    etype = evt.get("type")
                    data = evt.get("data") or {}

                    if etype == "server.connected":
                        await send_prompt()
                        continue

                    if not prompt_sent:
                        await send_prompt()

                    if data.get("sessionID") != session_id:
                        continue

                    if etype == "session.text.delta":
                        delta = data.get("delta")
                        if isinstance(delta, str) and delta:
                            yield delta
                    elif etype == "permission.asked":
                        req_id = data.get("id")
                        action = str(data.get("action") or "")
                        if isinstance(req_id, str) and req_id:
                            await self._reply_permission(client, session_id, req_id, action)
                    elif etype in {"session.execution.succeeded", "session.idle", "session.execution.interrupted"}:
                        break
                    elif etype == "session.execution.failed":
                        err = data.get("error") or {}
                        msg = err.get("message") if isinstance(err, dict) else str(err)
                        raise RuntimeError(f"OpenCode execution failed: {msg or 'unknown error'}")
        finally:
            if owns:
                await client.aclose()


class OpenCodeServerManager:
    """Starts and monitors an isolated `opencode serve` child process for voice sessions."""

    def __init__(self, config: OpencodeCliConfig) -> None:
        self._config = config
        self._proc: asyncio.subprocess.Process | None = None
        self._password: str | None = read_env_secret(config.password_env)
        self._agent_home: Path | None = None

    @property
    def agent_home(self) -> Path:
        if self._agent_home is not None:
            return self._agent_home
        return Path(self._config.agent_home).expanduser().resolve()

    def has_dead_process(self) -> bool:
        return self._proc is not None and self._proc.returncode is not None

    async def is_healthy(self, password: str | None = None) -> bool:
        base_url = self._config.server_url.rstrip("/")
        pw = password if password is not None else self._get_or_load_password()
        return await self._is_healthy(base_url, pw)

    def _get_or_load_password(self) -> str:
        if self._password:
            return self._password
        home = self.agent_home
        self._password = load_or_create_server_password(home, self._config.password_env)
        return self._password

    async def _is_healthy(self, base_url: str, password: str | None) -> bool:
        auth = ("opencode", password) if password else None
        try:
            async with httpx.AsyncClient(base_url=base_url, auth=auth, timeout=2.0) as client:
                r = await client.get("/api/info")
                return r.status_code == 200
        except Exception:
            return False

    async def ensure_running(
        self,
        *,
        mcp_url: str,
        system_prompt: str,
        mcp_auth_token: str | None = None,
        binary_resolver: Any = None,
    ) -> tuple[str, str | None]:
        base_url = self._config.server_url.rstrip("/")
        password = self._get_or_load_password()

        if self._proc is not None and self._proc.returncode is None:
            if await self._is_healthy(base_url, password):
                return base_url, password

        # Reuse an externally started or already-running server (e.g. `xsa serve` while running `xsa ask`)
        # without clobbering its active `opencode.jsonc` / MCP auth token.
        if await self._is_healthy(base_url, password):
            return base_url, password

        if not self._config.auto_start_server:
            raise RuntimeError(f"OpenCode V2 server is not reachable at {base_url}")

        self._agent_home = ensure_agent_home(
            self._config,
            mcp_url=mcp_url,
            system_prompt=system_prompt,
            mcp_auth_token=mcp_auth_token,
        )

        parsed = urlparse(base_url)
        host = parsed.hostname or "127.0.0.1"
        port = parsed.port or 4396

        if binary_resolver is not None:
            binary = binary_resolver(self._config.binary)
        else:
            from xiao_speaker_assistant.llm.router import resolve_binary

            binary = resolve_binary(self._config.binary)

        env = {
            **os.environ,
            "OPENCODE_DB": str(self._agent_home / "opencode.db"),
            "OPENCODE_PASSWORD": password,
            "OPENCODE_SERVER_PASSWORD": password,
        }
        log_path = self._agent_home / "opencode-serve.log"
        logger.info("Starting isolated OpenCode V2 server on %s (home=%s)", base_url, self._agent_home)

        # Redirect to a log file rather than an unread PIPE to avoid 64 KB buffer deadlocks.
        log_fh = open(log_path, "ab")
        try:
            self._proc = await asyncio.create_subprocess_exec(
                binary,
                "serve",
                "--hostname",
                host,
                "--port",
                str(port),
                cwd=str(self._agent_home),
                env=env,
                stdout=log_fh,
                stderr=log_fh,
            )
        finally:
            log_fh.close()

        deadline = asyncio.get_running_loop().time() + 15.0
        while asyncio.get_running_loop().time() < deadline:
            if self._proc.returncode is not None:
                err_tail = ""
                if log_path.exists():
                    err_tail = log_path.read_text(encoding="utf-8", errors="replace")[-400:]
                raise RuntimeError(
                    f"OpenCode V2 server exited ({self._proc.returncode}): {err_tail.strip()}"
                )
            if await self._is_healthy(base_url, password):
                return base_url, password
            await asyncio.sleep(0.15)

        await self.stop()
        raise RuntimeError(f"Timed out waiting for OpenCode V2 server at {base_url}")

    async def stop(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None or proc.returncode is not None:
            return
        proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=5.0)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
