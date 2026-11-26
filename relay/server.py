"""Stateless WebSocket relay bridging `/ws/speaker` and `/ws/host`."""

from __future__ import annotations

import argparse
import asyncio
import base64
import hmac
import json
import logging
import os
import posixpath
import sys
import time
import uuid
from collections import deque
from http import HTTPStatus
from typing import Any, Literal
from urllib.parse import parse_qs, unquote, urlsplit

import websockets
from websockets.asyncio.server import Request, Response, Server, ServerConnection

try:
    from aiohttp import web as aiohttp_web
except ImportError:  # pragma: no cover - optional only if console HTTP proxy disabled
    aiohttp_web = None  # type: ignore[assignment]

logger = logging.getLogger("xsa.relay")

Role = Literal["speaker", "host", "console"]

CLOSE_REPLACED = 4000
CLOSE_UNAUTHORIZED = 4401
CLOSE_NOT_FOUND = 4404
CLOSE_RATE_LIMITED = 4429

_ALLOWED_CONSOLE_METHODS = frozenset({"GET", "POST", "DELETE", "HEAD", "OPTIONS"})
_MAX_CONSOLE_REQ_BYTES = 1 * 1024 * 1024
# Keep base64-encoded JSON frames safely below the 4 MiB WebSocket max_size limit.
_MAX_CONSOLE_RESP_BYTES = 2 * 1024 * 1024
_MAX_PENDING_CONSOLE_REQS = 64
_HOP_BY_HOP_HEADERS = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "content-length",
    }
)


def _read_token_from_env(env_name: str = "XSA_RELAY_TOKEN") -> str | None:
    value = os.environ.get(env_name, "").strip()
    return value or None


def _read_bool_env(env_name: str, *, default: bool = False) -> bool:
    raw = os.environ.get(env_name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _extract_bearer_token(authorization_header: str) -> str | None:
    scheme, sep, token = authorization_header.strip().partition(" ")
    if not sep or scheme.lower() != "bearer":
        return None
    cleaned = token.strip()
    return cleaned or None


async def _safe_close(conn: ServerConnection, code: int, reason: str) -> None:
    try:
        await conn.close(code=code, reason=reason)
    except Exception:
        pass


def _parse_trusted_proxies(raw: str | None) -> frozenset[str]:
    if not raw:
        return frozenset()
    return frozenset(part.strip() for part in raw.split(",") if part.strip())


def _peer_ip(conn: ServerConnection) -> str:
    remote = conn.remote_address
    if isinstance(remote, tuple) and remote:
        return str(remote[0])
    return str(remote or "unknown")


def _client_ip_from_headers(
    peer: str, headers: Any, trusted_proxies: frozenset[str]
) -> str:
    if peer not in trusted_proxies or headers is None:
        return peer
    if hasattr(headers, "get_all"):
        header_values = headers.get_all("X-Forwarded-For")
    elif hasattr(headers, "getall"):
        header_values = headers.getall("X-Forwarded-For", [])
    else:
        raw = headers.get("X-Forwarded-For")
        header_values = [raw] if raw else []
    if not header_values:
        return peer
    # Walk right-to-left: a trusted reverse proxy appends the real peer on the right,
    # so the rightmost hop not in trusted_proxies cannot be forged by the client.
    hops = [
        part.strip()
        for raw_xff in header_values
        for part in raw_xff.split(",")
        if part.strip()
    ]
    for hop in reversed(hops):
        if hop not in trusted_proxies:
            return hop
    return peer


def _client_ip(conn: ServerConnection, trusted_proxies: frozenset[str]) -> str:
    peer = _peer_ip(conn)
    headers = conn.request.headers if conn.request is not None else None
    return _client_ip_from_headers(peer, headers, trusted_proxies)


class RelayServer:
    """Stateless bidirectional relay between one active speaker and one active host."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        *,
        console_port: int | None = None,
        token: str | None = None,
        host_token: str | None = None,
        speaker_token: str | None = None,
        allow_insecure_no_auth: bool = False,
        max_size: int | None = 4 * 1024 * 1024,
        auth_fail_limit: int = 10,
        auth_fail_window_sec: float = 60.0,
        auth_ban_sec: float = 300.0,
        trusted_proxies: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> None:
        self.host = host
        self.port = port
        self.console_port = console_port
        legacy_token = _read_token_from_env("XSA_RELAY_TOKEN") if token is None else (token.strip() or None)
        resolved_host_token = (
            _read_token_from_env("XSA_RELAY_HOST_TOKEN")
            if host_token is None
            else (host_token.strip() or None)
        )
        resolved_speaker_token = (
            _read_token_from_env("XSA_RELAY_SPEAKER_TOKEN")
            if speaker_token is None
            else (speaker_token.strip() or None)
        )
        # Fail closed on half-configured split tokens instead of silently mixing with legacy token.
        if bool(resolved_host_token) ^ bool(resolved_speaker_token):
            raise ValueError(
                "Both XSA_RELAY_HOST_TOKEN and XSA_RELAY_SPEAKER_TOKEN must be set together "
                "when using split role tokens."
            )
        self._host_token = resolved_host_token or legacy_token
        self._speaker_token = resolved_speaker_token or legacy_token
        self._allow_insecure_no_auth = allow_insecure_no_auth
        if not self._allow_insecure_no_auth and (not self._host_token or not self._speaker_token):
            raise ValueError(
                "Relay authentication token is required. Set XSA_RELAY_HOST_TOKEN + "
                "XSA_RELAY_SPEAKER_TOKEN (or XSA_RELAY_TOKEN), or explicitly pass allow_insecure_no_auth=True."
            )
        self._max_size = max_size
        self._auth_fail_limit = max(1, int(auth_fail_limit))
        self._auth_fail_window_sec = max(1.0, float(auth_fail_window_sec))
        self._auth_ban_sec = max(1.0, float(auth_ban_sec))
        self._max_tracked_ips = 4096
        self._trusted_proxies = (
            _parse_trusted_proxies(os.environ.get("XSA_RELAY_TRUSTED_PROXIES"))
            if trusted_proxies is None
            else frozenset(p.strip() for p in trusted_proxies if p.strip())
        )
        self._auth_failures: dict[str, deque[float]] = {}
        self._banned_until: dict[str, float] = {}
        self._speaker: ServerConnection | None = None
        self._host_conn: ServerConnection | None = None
        self._console_conn: ServerConnection | None = None
        self._pending_console_reqs: dict[str, tuple[ServerConnection, asyncio.Future[dict[str, Any]]]] = {}
        self._bg_tasks: set[asyncio.Task[Any]] = set()
        self._server: Server | None = None
        self._console_runner: Any = None
        self._bound_console_port: int | None = None

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.port
        return int(self._server.sockets[0].getsockname()[1])

    @property
    def bound_console_port(self) -> int | None:
        return self._bound_console_port

    @property
    def separate_role_tokens(self) -> bool:
        return bool(
            self._host_token
            and self._speaker_token
            and not hmac.compare_digest(self._host_token, self._speaker_token)
        )

    def _prune_rate_tables(self, now: float) -> None:
        cutoff = now - self._auth_fail_window_sec
        expired_bans = [ip for ip, until in self._banned_until.items() if now >= until]
        for ip in expired_bans:
            self._banned_until.pop(ip, None)
        if len(self._banned_until) >= self._max_tracked_ips:
            # Evict the soonest-to-expire entries to bound memory without dropping fresh bans.
            overflow = len(self._banned_until) - self._max_tracked_ips + 1
            for ip, _ in sorted(self._banned_until.items(), key=lambda kv: kv[1])[:overflow]:
                self._banned_until.pop(ip, None)

        stale_failures = [
            ip for ip, q in self._auth_failures.items() if not q or q[-1] < cutoff
        ]
        for ip in stale_failures:
            self._auth_failures.pop(ip, None)
        if len(self._auth_failures) >= self._max_tracked_ips:
            overflow = len(self._auth_failures) - self._max_tracked_ips + 1
            for ip, _ in sorted(
                self._auth_failures.items(), key=lambda kv: kv[1][-1] if kv[1] else 0.0
            )[:overflow]:
                self._auth_failures.pop(ip, None)

    def _is_banned(self, ip: str, now: float) -> bool:
        until = self._banned_until.get(ip)
        if until is None:
            return False
        if now >= until:
            self._banned_until.pop(ip, None)
            return False
        return True

    def _record_auth_failure(self, ip: str, now: float) -> bool:
        if len(self._auth_failures) >= self._max_tracked_ips or len(self._banned_until) >= self._max_tracked_ips:
            self._prune_rate_tables(now)
        q = self._auth_failures.get(ip)
        if q is None:
            q = deque()
            self._auth_failures[ip] = q
        cutoff = now - self._auth_fail_window_sec
        while q and q[0] < cutoff:
            q.popleft()
        q.append(now)
        if len(q) >= self._auth_fail_limit:
            self._banned_until[ip] = now + self._auth_ban_sec
            q.clear()
            return True
        return False

    def _clear_auth_failures(self, ip: str) -> None:
        self._auth_failures.pop(ip, None)

    def _expected_token_for_role(self, role: Role) -> str | None:
        return self._speaker_token if role == "speaker" else self._host_token

    def _is_request_authorized(
        self,
        request: Request | None,
        *,
        role: Role,
        allow_query_token: bool = False,
    ) -> bool:
        expected = self._expected_token_for_role(role)
        if expected is None:
            return self._allow_insecure_no_auth
        if request is None:
            return False
        auth_values = request.headers.get_all("Authorization")
        if len(auth_values) > 1:
            return False
        raw_header = auth_values[0] if auth_values else ""
        presented = _extract_bearer_token(raw_header)
        if presented is not None:
            return hmac.compare_digest(presented, expected)
        if allow_query_token:
            query_params = parse_qs(urlsplit(request.path).query, keep_blank_values=False)
            tokens = query_params.get("token")
            if tokens and len(tokens) == 1 and tokens[0].strip():
                return hmac.compare_digest(tokens[0].strip(), expected)
        return False

    def _is_authorized(
        self,
        conn: ServerConnection,
        *,
        role: Role,
        allow_query_token: bool = False,
    ) -> bool:
        return self._is_request_authorized(
            conn.request, role=role, allow_query_token=allow_query_token
        )

    def _get_conn(self, role: Role) -> ServerConnection | None:
        if role == "speaker":
            return self._speaker
        if role == "console":
            return self._console_conn
        return self._host_conn

    def _set_conn(self, role: Role, conn: ServerConnection | None) -> None:
        if role == "speaker":
            self._speaker = conn
        elif role == "console":
            self._console_conn = conn
        else:
            self._host_conn = conn

    def _get_peer(self, role: Role) -> ServerConnection | None:
        if role == "console":
            return None
        return self._host_conn if role == "speaker" else self._speaker

    @staticmethod
    def _resolve_role(path: str) -> Role | None:
        if path in {"/ws/speaker", "/xsa/ws/speaker"}:
            return "speaker"
        if path in {"/ws/host", "/xsa/ws/host"}:
            return "host"
        if path in {"/ws/console", "/xsa/ws/console"}:
            return "console"
        return None

    def _process_request(
        self, conn: ServerConnection, request: Request
    ) -> Response | None:
        # Allow HTTP health checks (for example Azure Container Apps probes) without a WS upgrade.
        upgrade_tokens = [
            tok.strip().lower()
            for raw_val in request.headers.get_all("Upgrade")
            for tok in raw_val.split(",")
            if tok.strip()
        ]
        is_ws = len(upgrade_tokens) == 1 and upgrade_tokens[0] == "websocket"
        if not is_ws:
            path = urlsplit(request.path).path
            if path in {"/", "/health", "/healthz"}:
                return conn.respond(HTTPStatus.OK, "ok\n")
            return conn.respond(HTTPStatus.NOT_FOUND, "not found\n")

        # Reject banned, unauthorized, or unknown WebSocket handshakes at the HTTP layer
        # (401/404/429) BEFORE completing a 101 Switching Protocols upgrade.
        ip = _client_ip_from_headers(_peer_ip(conn), request.headers, self._trusted_proxies)
        now = time.monotonic()
        if self._is_banned(ip, now):
            logger.warning("Rejected rate-limited handshake from %s", ip)
            return conn.respond(HTTPStatus.TOO_MANY_REQUESTS, "too many requests\n")

        path = urlsplit(request.path).path
        role = self._resolve_role(path)
        if role is None:
            authed = self._is_request_authorized(
                request, role="host", allow_query_token=False
            ) or self._is_request_authorized(
                request, role="speaker", allow_query_token=False
            )
            if not authed:
                banned = self._record_auth_failure(ip, now)
                logger.warning(
                    "Rejected unauthorized handshake from %s%s",
                    ip,
                    " (IP rate-limited)" if banned else "",
                )
                return conn.respond(
                    HTTPStatus.TOO_MANY_REQUESTS if banned else HTTPStatus.UNAUTHORIZED,
                    "too many requests\n" if banned else "unauthorized\n",
                )
            logger.warning("Rejected unknown endpoint %r from %s", path, ip)
            return conn.respond(HTTPStatus.NOT_FOUND, "not found\n")

        if not self._is_request_authorized(
            request, role=role, allow_query_token=(role == "speaker")
        ):
            banned = self._record_auth_failure(ip, now)
            logger.warning(
                "Rejected unauthorized %s handshake from %s%s",
                role,
                ip,
                " (IP rate-limited)" if banned else "",
            )
            return conn.respond(
                HTTPStatus.TOO_MANY_REQUESTS if banned else HTTPStatus.UNAUTHORIZED,
                "too many requests\n" if banned else "unauthorized\n",
            )

        self._clear_auth_failures(ip)
        return None

    async def _handle_connection(self, conn: ServerConnection) -> None:
        ip = _client_ip(conn, self._trusted_proxies)
        now = time.monotonic()
        if self._is_banned(ip, now):
            logger.warning("Rejected rate-limited connection from %s", ip)
            await conn.close(code=CLOSE_RATE_LIMITED, reason="Too Many Requests")
            return

        path = urlsplit(conn.request.path).path if conn.request else ""
        role = self._resolve_role(path)
        if role is None:
            # Defense-in-depth if _handle_connection is ever invoked without _process_request.
            authed = self._is_authorized(conn, role="host", allow_query_token=False) or self._is_authorized(
                conn, role="speaker", allow_query_token=False
            )
            if not authed:
                banned = self._record_auth_failure(ip, now)
                logger.warning(
                    "Rejected unauthorized connection from %s%s",
                    ip,
                    " (IP rate-limited)" if banned else "",
                )
                await conn.close(
                    code=CLOSE_RATE_LIMITED if banned else CLOSE_UNAUTHORIZED,
                    reason="Too Many Requests" if banned else "Unauthorized",
                )
                return
            logger.warning("Rejected unknown endpoint %r from %s", path, ip)
            await conn.close(code=CLOSE_NOT_FOUND, reason="Not Found")
            return

        if not self._is_authorized(conn, role=role, allow_query_token=(role == "speaker")):
            banned = self._record_auth_failure(ip, now)
            logger.warning(
                "Rejected unauthorized %s connection from %s%s",
                role,
                ip,
                " (IP rate-limited)" if banned else "",
            )
            await conn.close(
                code=CLOSE_RATE_LIMITED if banned else CLOSE_UNAUTHORIZED,
                reason="Too Many Requests" if banned else "Unauthorized",
            )
            return

        self._clear_auth_failures(ip)

        previous = self._get_conn(role)
        self._set_conn(role, conn)
        if previous is not None and previous is not conn:
            logger.info("Replacing active %s connection", role)
            # Close asynchronously so a half-open peer cannot stall the replacement up to close_timeout.
            close_task = asyncio.create_task(
                _safe_close(previous, code=CLOSE_REPLACED, reason=f"Replaced by new {role}")
            )
            self._bg_tasks.add(close_task)
            close_task.add_done_callback(self._bg_tasks.discard)

        logger.info("Connected %s from %s", role, ip)
        try:
            if role == "console":
                async for message in conn:
                    if self._get_conn("console") is not conn:
                        break
                    if not isinstance(message, str):
                        continue
                    try:
                        payload = json.loads(message)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(payload, dict) or payload.get("type") != "http_response":
                        continue
                    req_id = payload.get("id")
                    entry = (
                        self._pending_console_reqs.get(req_id)
                        if isinstance(req_id, str)
                        else None
                    )
                    if entry is not None:
                        owner_conn, fut = entry
                        if owner_conn is conn and not fut.done():
                            fut.set_result(payload)
            else:
                async for message in conn:
                    if self._get_conn(role) is not conn:
                        break
                    peer = self._get_peer(role)
                    if peer is None:
                        continue
                    try:
                        await peer.send(message)
                    except websockets.ConnectionClosed:
                        logger.debug("Peer dropped while forwarding %s message", role)
        except websockets.ConnectionClosed:
            pass
        finally:
            if self._get_conn(role) is conn:
                self._set_conn(role, None)
            if role == "console":
                for owner_conn, fut in list(self._pending_console_reqs.values()):
                    if owner_conn is conn and not fut.done():
                        fut.set_exception(ConnectionError("Console host disconnected"))
            logger.info("Disconnected %s", role)

    @staticmethod
    def _is_console_path_allowed(raw_path_or_qs: str) -> bool:
        # Never proxy internal /mcp endpoint or traversal variants over the public console tunnel.
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in raw_path_or_qs):
            return False
        parts = urlsplit(raw_path_or_qs)
        if parts.scheme or parts.netloc or parts.fragment:
            return False
        raw_path = parts.path
        if not raw_path.startswith("/") or "\\" in raw_path:
            return False
        decoded = raw_path
        for _ in range(3):
            if "\\" in decoded or any(ord(c) < 0x20 or ord(c) == 0x7F for c in decoded):
                return False
            if any(seg == ".." for seg in decoded.split("/")):
                return False
            nxt = unquote(decoded)
            if nxt == decoded:
                break
            decoded = nxt
        if "\\" in decoded or any(ord(c) < 0x20 or ord(c) == 0x7F for c in decoded):
            return False
        if any(seg == ".." for seg in decoded.split("/")):
            return False
        normalized = posixpath.normpath(decoded)
        if not normalized.startswith("/"):
            return False
        return normalized == "/" or normalized == "/health" or normalized.startswith("/api/")

    async def _handle_console_http(self, request: Any) -> Any:
        assert aiohttp_web is not None
        path = request.path
        if path == "/healthz":
            return aiohttp_web.json_response(
                {"ok": True, "console_connected": self._console_conn is not None}
            )
        if not (
            self._is_console_path_allowed(path)
            and self._is_console_path_allowed(request.raw_path)
        ):
            return aiohttp_web.json_response(
                {"error": "forbidden", "message": "Path is not exposed via remote console"},
                status=403,
            )
        if request.method not in _ALLOWED_CONSOLE_METHODS:
            return aiohttp_web.json_response(
                {"error": "method_not_allowed", "message": "Method not allowed"},
                status=405,
            )

        conn = self._console_conn
        if conn is None:
            return aiohttp_web.json_response(
                {"error": "host_offline", "message": "Mac 控制台主机当前未在线，请稍后再试"},
                status=503,
            )
        if len(self._pending_console_reqs) >= _MAX_PENDING_CONSOLE_REQS:
            return aiohttp_web.json_response(
                {"error": "service_busy", "message": "控制台并发请求过多，请稍后再试"},
                status=503,
            )

        raw_body = await request.read()
        if len(raw_body) > _MAX_CONSOLE_REQ_BYTES:
            return aiohttp_web.json_response(
                {"error": "payload_too_large", "message": "Request body too large"},
                status=413,
            )

        peer = str(request.remote or "unknown")
        from_trusted_proxy = peer in self._trusted_proxies
        client_ip = _client_ip_from_headers(peer, request.headers, self._trusted_proxies)

        fwd_headers: dict[str, str] = {}
        for k in ("Host", "Origin", "Content-Type", "Accept", "Authorization"):
            val = request.headers.get(k)
            if val is not None:
                fwd_headers[k] = val
        cookie_vals = request.headers.getall("Cookie", [])
        if cookie_vals:
            fwd_headers["Cookie"] = "; ".join(cookie_vals)

        # Always inject X-Forwarded-For & X-Forwarded-Proto so Mac McpServer treats
        # tunneled traffic as remote (enforcing console.token login & Secure cookies).
        # Derive client IP right-to-left via trusted_proxies so spoofed XFF cannot bypass login rate limits.
        fwd_headers["X-Forwarded-For"] = client_ip or "remote"
        raw_proto = request.headers.get("X-Forwarded-Proto", "").strip().lower()
        if from_trusted_proxy and raw_proto in {"http", "https"}:
            fwd_headers["X-Forwarded-Proto"] = raw_proto
        else:
            fwd_headers["X-Forwarded-Proto"] = "https"
        if from_trusted_proxy and request.headers.get("X-Forwarded-Host"):
            fwd_headers["X-Forwarded-Host"] = request.headers["X-Forwarded-Host"]
        elif "Host" in fwd_headers:
            fwd_headers["X-Forwarded-Host"] = fwd_headers["Host"]

        req_id = uuid.uuid4().hex
        fut: asyncio.Future[dict[str, Any]] = asyncio.get_running_loop().create_future()
        self._pending_console_reqs[req_id] = (conn, fut)
        try:
            frame = {
                "type": "http_request",
                "id": req_id,
                "method": request.method,
                "path_qs": request.raw_path,
                "headers": fwd_headers,
                "body_b64": base64.b64encode(raw_body).decode("ascii") if raw_body else "",
            }
            await conn.send(json.dumps(frame, ensure_ascii=False))
            resp_payload = await asyncio.wait_for(fut, timeout=25.0)
        except TimeoutError:
            return aiohttp_web.json_response(
                {"error": "gateway_timeout", "message": "Mac 控制台响应超时"},
                status=504,
            )
        except Exception:
            return aiohttp_web.json_response(
                {"error": "bad_gateway", "message": "Mac 控制台连接中断"},
                status=502,
            )
        finally:
            self._pending_console_reqs.pop(req_id, None)

        try:
            status = int(resp_payload.get("status", 502))
            if not (100 <= status <= 599):
                raise ValueError("invalid status")
            body_b64 = str(resp_payload.get("body_b64") or "")
            body = base64.b64decode(body_b64, validate=True) if body_b64 else b""
        except Exception:
            return aiohttp_web.json_response(
                {"error": "bad_gateway", "message": "Invalid response from console host"},
                status=502,
            )
        if len(body) > _MAX_CONSOLE_RESP_BYTES:
            return aiohttp_web.json_response(
                {"error": "bad_gateway", "message": "Response too large"},
                status=502,
            )
        out_headers: list[tuple[str, str]] = []
        for pair in resp_payload.get("headers") or []:
            if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                continue
            hk, hv = str(pair[0]), str(pair[1])
            if hk.lower() in _HOP_BY_HOP_HEADERS:
                continue
            if any(ch in hk or ch in hv for ch in ("\r", "\n", "\x00")):
                return aiohttp_web.json_response(
                    {"error": "bad_gateway", "message": "Invalid response header from console host"},
                    status=502,
                )
            out_headers.append((hk, hv))
        return aiohttp_web.Response(status=status, body=body, headers=out_headers)

    async def start(self) -> RelayServer:
        if self._server is not None:
            return self
        self._server = await websockets.serve(
            self._handle_connection,
            self.host,
            self.port,
            process_request=self._process_request,
            max_size=self._max_size,
        )
        if self.console_port is not None:
            if aiohttp_web is None:
                raise RuntimeError("aiohttp is required when console_port is enabled")
            app = aiohttp_web.Application(client_max_size=_MAX_CONSOLE_REQ_BYTES + 4096)
            app.router.add_route("*", "/{tail:.*}", self._handle_console_http)
            runner = aiohttp_web.AppRunner(app, access_log=None)
            await runner.setup()
            site = aiohttp_web.TCPSite(runner, self.host, self.console_port)
            await site.start()
            self._console_runner = runner
            if site._server and site._server.sockets:  # type: ignore[attr-defined]
                self._bound_console_port = int(site._server.sockets[0].getsockname()[1])  # type: ignore[attr-defined]
            else:
                self._bound_console_port = self.console_port
            logger.info(
                "Console HTTP relay listening on http://%s:%d",
                self.host,
                self._bound_console_port,
            )
        auth_mode = (
            "split-role-tokens"
            if self.separate_role_tokens
            else ("shared-token" if self._host_token else "disabled")
        )
        logger.info(
            "Relay listening on ws://%s:%d (auth=%s)",
            self.host,
            self.bound_port,
            auth_mode,
        )
        return self

    async def close(self) -> None:
        for _owner_conn, fut in list(self._pending_console_reqs.values()):
            if not fut.done():
                fut.set_exception(ConnectionError("Relay server shutting down"))
        self._pending_console_reqs.clear()
        if self._console_runner is not None:
            await self._console_runner.cleanup()
            self._console_runner = None
            self._bound_console_port = None
        for role in ("speaker", "host", "console"):
            conn = self._get_conn(role)  # type: ignore[arg-type]
            self._set_conn(role, None)  # type: ignore[arg-type]
            if conn is not None:
                await _safe_close(conn, code=1001, reason="Server shutting down")
        if self._server is not None:
            self._server.close()
            await self._server.wait_closed()
            self._server = None

    async def __aenter__(self) -> RelayServer:
        return await self.start()

    async def __aexit__(self, *_: object) -> None:
        await self.close()

    async def serve_forever(self) -> None:
        await self.start()
        try:
            await asyncio.Future()
        finally:
            await self.close()


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Xiao Speaker Assistant stateless WebSocket relay")
    parser.add_argument("--host", default=os.environ.get("RELAY_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("RELAY_PORT", "8765")))
    raw_console_port = os.environ.get("RELAY_CONSOLE_PORT", "").strip()
    default_console_port = (
        int(raw_console_port) if raw_console_port.isdigit() and int(raw_console_port) > 0 else None
    )
    parser.add_argument(
        "--console-port",
        type=int,
        default=default_console_port,
        help="Optional HTTP port proxying remote Web Console traffic to `/ws/console` (0 disables)",
    )
    parser.add_argument(
        "--log-level",
        default=os.environ.get("RELAY_LOG_LEVEL", "INFO"),
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    # Prevent websockets handshake logs from recording ?token= or Authorization headers at DEBUG.
    logging.getLogger("websockets").setLevel(logging.INFO)

    relay = RelayServer(
        host=args.host,
        port=args.port,
        console_port=args.console_port if args.console_port and args.console_port > 0 else None,
        allow_insecure_no_auth=_read_bool_env("XSA_RELAY_ALLOW_INSECURE_NO_AUTH", default=False),
    )
    try:
        asyncio.run(relay.serve_forever())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
