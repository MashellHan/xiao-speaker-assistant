"""Outbound WebSocket client connecting Mac SpeakerGateway to the Azure relay."""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import posixpath
from typing import Any, Callable
from urllib.parse import unquote, urlsplit, urlunsplit

import httpx
import websockets
from websockets.asyncio.client import ClientConnection
from websockets.exceptions import InvalidStatus

from xiao_speaker_assistant.config import AppConfig, read_env_secret
from xiao_speaker_assistant.voice.ws_server import (
    QueryHandler,
    SpeakerGateway,
    StreamQueryHandler,
)

logger = logging.getLogger(__name__)


def normalize_host_ws_url(relay_url: str) -> str:
    parts = urlsplit(relay_url.strip())
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError(f"Relay URL must use ws:// or wss:// with a host: {relay_url!r}")
    path = parts.path.rstrip("/")
    if not path:
        path = "/ws/host"
    elif not path.endswith("/ws/host"):
        path = f"{path}/ws/host"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


class RelayHostClient:
    """Dials outbound to `/ws/host` on the relay and dispatches frames to SpeakerGateway."""

    def __init__(
        self,
        gateway: SpeakerGateway,
        relay_url: str,
        *,
        auth_token: str | None = None,
        token_env: str = "XSA_RELAY_TOKEN",
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_factor: float = 2.0,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        healthy_reset_sec: float = 5.0,
    ) -> None:
        if initial_backoff <= 0:
            raise ValueError("initial_backoff must be positive")
        if max_backoff < initial_backoff:
            raise ValueError("max_backoff must be >= initial_backoff")
        if backoff_factor < 1.0:
            raise ValueError("backoff_factor must be >= 1.0")

        self._gateway = gateway
        self.relay_url = normalize_host_ws_url(relay_url)
        if auth_token is not None:
            self._auth_token: str | None = auth_token.strip() or None
        else:
            self._auth_token = (
                read_env_secret("XSA_RELAY_HOST_TOKEN")
                if token_env == "XSA_RELAY_TOKEN"
                else None
            ) or read_env_secret(token_env)

        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_factor = backoff_factor
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._healthy_reset_sec = healthy_reset_sec

        self._stopped = False
        self._stop_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._active_ws: ClientConnection | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    async def wait_connected(self, timeout: float | None = None) -> None:
        if timeout is None:
            await self._connected_event.wait()
            return
        await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)

    def _build_headers(self) -> dict[str, str] | None:
        if not self._auth_token:
            return None
        return {"Authorization": f"Bearer {self._auth_token}"}

    def _ensure_gateway_pipeline(self) -> None:
        self._gateway.ensure_pipeline()

    async def _serve_connection(self, ws: ClientConnection) -> bool:
        conn: Any = ws
        self._active_ws = ws
        self._gateway._connections.add(conn)
        self._gateway._pending_rpc.setdefault(conn, {})
        self._gateway._get_session(conn)
        self._gateway._get_playback(conn)
        self._connected_event.set()
        received_any = False

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=200)
        audio_worker_task = asyncio.create_task(self._gateway._audio_worker(conn, audio_queue))
        watchdog_task: asyncio.Task[None] | None = None

        try:
            if self._gateway._pipeline is not None:
                await self._gateway.start_recording(conn)
                watchdog_task = asyncio.create_task(self._gateway._audio_watchdog(conn))

            async for raw in ws:
                received_any = True
                if isinstance(raw, bytes):
                    self._gateway.enqueue_audio_frame(audio_queue, raw)
                else:
                    await self._gateway._handle_text_message(conn, raw)
        finally:
            self._connected_event.clear()
            if watchdog_task is not None:
                watchdog_task.cancel()
            audio_worker_task.cancel()
            if self._active_ws is ws:
                self._active_ws = None
            self._gateway._cleanup_connection(conn)

        return received_any

    async def serve_forever(self) -> None:
        self._stopped = False
        self._stop_event.clear()
        self._ensure_gateway_pipeline()
        delay = self._initial_backoff
        loop = asyncio.get_running_loop()

        while not self._stopped:
            connected_at: float | None = None
            received_any = False
            try:
                async with websockets.connect(
                    self.relay_url,
                    additional_headers=self._build_headers(),
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                ) as ws:
                    connected_at = loop.time()
                    logger.info("Connected to relay host endpoint: %s", self.relay_url)
                    received_any = await self._serve_connection(ws)
            except asyncio.CancelledError:
                raise
            except InvalidStatus as exc:
                status = exc.response.status_code
                if status == 401:
                    logger.error("Relay rejected host authentication (HTTP 401 Unauthorized)")
                elif status == 429:
                    logger.error("Relay rate-limited host connection (HTTP 429 Too Many Requests)")
                else:
                    logger.warning("Relay rejected host handshake (HTTP %s)", status)
            except websockets.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                if code == 4401:
                    logger.error("Relay rejected host authentication (4401 Unauthorized)")
                else:
                    logger.warning("Relay host connection closed (code=%s): %s", code, exc)
            except Exception as exc:
                logger.warning("Relay host connection error: %s", exc)

            if self._stopped:
                break

            healthy = connected_at is not None and (
                received_any or (loop.time() - connected_at) >= self._healthy_reset_sec
            )
            if healthy:
                delay = self._initial_backoff

            wait_sec = delay
            delay = min(self._max_backoff, delay * self._backoff_factor)
            logger.info("Reconnecting to relay in %.2fs", wait_sec)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_sec)
                break
            except TimeoutError:
                continue

    async def stop(self) -> None:
        self._stopped = True
        self._stop_event.set()
        ws = self._active_ws
        if ws is not None:
            try:
                await ws.close(code=1000, reason="Client shutting down")
            except Exception:
                pass


def normalize_console_ws_url(relay_url: str) -> str:
    """Derive `/ws/console` (preserving any prefix such as `/xsa`) from a relay URL."""
    parts = urlsplit(relay_url.strip())
    if parts.scheme not in {"ws", "wss"} or not parts.netloc:
        raise ValueError(f"Relay URL must use ws:// or wss:// with a host: {relay_url!r}")
    path = parts.path.rstrip("/")
    if path.endswith("/ws/host"):
        path = path[: -len("/ws/host")] + "/ws/console"
    elif path.endswith("/ws/console"):
        pass
    elif not path:
        path = "/ws/console"
    else:
        path = f"{path}/ws/console"
    return urlunsplit((parts.scheme, parts.netloc, path, parts.query, ""))


class RelayConsoleClient:
    """Dials outbound to `/ws/console` on the Azure relay and forwards console HTTP to 127.0.0.1."""

    _ALLOWED_METHODS = frozenset({"GET", "POST", "DELETE", "HEAD", "OPTIONS"})
    _ALLOWED_FWD_HEADERS = frozenset(
        {
            "host",
            "origin",
            "content-type",
            "accept",
            "cookie",
            "authorization",
            "x-forwarded-for",
            "x-forwarded-proto",
            "x-forwarded-host",
        }
    )
    _MAX_REQ_BYTES = 1 * 1024 * 1024
    _MAX_RESP_BYTES = 2 * 1024 * 1024
    _MAX_CONCURRENT_REQUESTS = 64
    _HOP_BY_HOP = frozenset(
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

    def __init__(
        self,
        relay_url: str,
        local_console_url: str,
        *,
        auth_token: str | None = None,
        token_env: str = "XSA_RELAY_TOKEN",
        initial_backoff: float = 1.0,
        max_backoff: float = 60.0,
        backoff_factor: float = 2.0,
        ping_interval: float | None = 20.0,
        ping_timeout: float | None = 20.0,
        healthy_reset_sec: float = 5.0,
    ) -> None:
        self.relay_url = normalize_console_ws_url(relay_url)
        self.local_console_url = local_console_url.rstrip("/")
        if auth_token is not None:
            self._auth_token: str | None = auth_token.strip() or None
        else:
            self._auth_token = (
                read_env_secret("XSA_RELAY_HOST_TOKEN")
                if token_env == "XSA_RELAY_TOKEN"
                else None
            ) or read_env_secret(token_env)
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._backoff_factor = backoff_factor
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._healthy_reset_sec = healthy_reset_sec
        self._stopped = False
        self._stop_event = asyncio.Event()
        self._connected_event = asyncio.Event()
        self._active_ws: ClientConnection | None = None

    @property
    def is_connected(self) -> bool:
        return self._connected_event.is_set()

    async def wait_connected(self, timeout: float | None = None) -> None:
        if timeout is None:
            await self._connected_event.wait()
            return
        await asyncio.wait_for(self._connected_event.wait(), timeout=timeout)

    def _build_headers(self) -> dict[str, str] | None:
        if not self._auth_token:
            return None
        return {"Authorization": f"Bearer {self._auth_token}"}

    @staticmethod
    def _is_path_allowed(path_qs: str) -> bool:
        parts = urlsplit(path_qs)
        if parts.scheme or parts.netloc:
            return False
        raw_path = parts.path
        if not raw_path.startswith("/") or "\\" in raw_path or "\x00" in raw_path:
            return False
        decoded = unquote(raw_path)
        if "\\" in decoded or "\x00" in decoded:
            return False
        if any(seg == ".." for seg in raw_path.split("/")) or any(
            seg == ".." for seg in decoded.split("/")
        ):
            return False
        normalized = posixpath.normpath(decoded)
        if not normalized.startswith("/"):
            return False
        # Defense-in-depth on Mac side too: never forward `/mcp` over the remote console tunnel.
        return normalized == "/" or normalized == "/health" or normalized.startswith("/api/")

    async def _handle_http_request(
        self,
        ws: ClientConnection,
        http_client: httpx.AsyncClient,
        frame: dict[str, Any],
    ) -> None:
        req_id = frame.get("id")
        if not isinstance(req_id, str) or not req_id:
            return
        method = str(frame.get("method") or "GET").upper()
        path_qs = str(frame.get("path_qs") or "/")

        if not self._is_path_allowed(path_qs):
            await self._send_response(
                ws,
                req_id,
                403,
                [("Content-Type", "application/json; charset=utf-8")],
                b'{"error":"forbidden","message":"Path is not exposed via remote console"}',
            )
            return
        if method not in self._ALLOWED_METHODS:
            await self._send_response(
                ws,
                req_id,
                405,
                [("Content-Type", "application/json; charset=utf-8")],
                b'{"error":"method_not_allowed"}',
            )
            return

        raw_headers = frame.get("headers") if isinstance(frame.get("headers"), dict) else {}
        fwd_headers: dict[str, str] = {}
        for k, v in raw_headers.items():
            if not isinstance(k, str) or not isinstance(v, str):
                continue
            lk = k.lower()
            if lk in self._HOP_BY_HOP or lk not in self._ALLOWED_FWD_HEADERS:
                continue
            fwd_headers[k] = v
        # Ensure Mac McpServer ALWAYS recognizes tunneled requests as remote (never loopback-bypass).
        existing_xff = next((v.strip() for k, v in fwd_headers.items() if k.lower() == "x-forwarded-for"), "")
        if not existing_xff:
            fwd_headers["X-Forwarded-For"] = "remote"
        existing_proto = next(
            (v.strip().lower() for k, v in fwd_headers.items() if k.lower() == "x-forwarded-proto"),
            "",
        )
        if existing_proto not in {"http", "https"}:
            fwd_headers["X-Forwarded-Proto"] = "https"

        body_b64 = str(frame.get("body_b64") or "")
        try:
            body = base64.b64decode(body_b64, validate=True) if body_b64 else b""
        except Exception:
            await self._send_response(
                ws,
                req_id,
                400,
                [("Content-Type", "application/json; charset=utf-8")],
                b'{"error":"bad_request","message":"Invalid base64 body"}',
            )
            return
        if len(body) > self._MAX_REQ_BYTES:
            await self._send_response(
                ws,
                req_id,
                413,
                [("Content-Type", "application/json; charset=utf-8")],
                b'{"error":"payload_too_large","message":"Request body too large"}',
            )
            return

        try:
            resp = await http_client.request(
                method,
                f"{self.local_console_url}{path_qs}",
                headers=fwd_headers,
                content=body if body else None,
            )
            if len(resp.content) > self._MAX_RESP_BYTES:
                await self._send_response(
                    ws,
                    req_id,
                    502,
                    [("Content-Type", "application/json; charset=utf-8")],
                    b'{"error":"bad_gateway","message":"Console response too large"}',
                )
                return
            out_headers = [
                (k, v) for k, v in resp.headers.multi_items() if k.lower() not in self._HOP_BY_HOP
            ]
            await self._send_response(ws, req_id, resp.status_code, out_headers, resp.content)
        except Exception as exc:
            logger.warning("Console relay local forward failed for %s %s: %s", method, path_qs, exc)
            await self._send_response(
                ws,
                req_id,
                502,
                [("Content-Type", "application/json; charset=utf-8")],
                b'{"error":"bad_gateway","message":"Local console unavailable"}',
            )

    @staticmethod
    async def _send_response(
        ws: ClientConnection,
        req_id: str,
        status: int,
        headers: list[tuple[str, str]],
        body: bytes,
    ) -> None:
        payload = {
            "type": "http_response",
            "id": req_id,
            "status": int(status),
            "headers": [[k, v] for k, v in headers],
            "body_b64": base64.b64encode(body).decode("ascii") if body else "",
        }
        try:
            await ws.send(json.dumps(payload, ensure_ascii=False))
        except Exception:
            pass

    async def _serve_connection(self, ws: ClientConnection) -> bool:
        self._active_ws = ws
        self._connected_event.set()
        received_any = False
        tasks: set[asyncio.Task[None]] = set()
        async with httpx.AsyncClient(timeout=20.0, follow_redirects=False) as http_client:
            try:
                async for raw in ws:
                    received_any = True
                    if not isinstance(raw, str):
                        continue
                    try:
                        frame = json.loads(raw)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(frame, dict) or frame.get("type") != "http_request":
                        continue
                    if len(tasks) >= self._MAX_CONCURRENT_REQUESTS:
                        req_id = frame.get("id")
                        if isinstance(req_id, str) and req_id:
                            await self._send_response(
                                ws,
                                req_id,
                                503,
                                [("Content-Type", "application/json; charset=utf-8")],
                                b'{"error":"service_busy","message":"Too many concurrent console requests"}',
                            )
                        continue
                    task = asyncio.create_task(self._handle_http_request(ws, http_client, frame))
                    tasks.add(task)
                    task.add_done_callback(tasks.discard)
            finally:
                self._connected_event.clear()
                for t in list(tasks):
                    t.cancel()
                if self._active_ws is ws:
                    self._active_ws = None
        return received_any

    async def serve_forever(self) -> None:
        self._stopped = False
        self._stop_event.clear()
        delay = self._initial_backoff
        loop = asyncio.get_running_loop()

        while not self._stopped:
            connected_at: float | None = None
            received_any = False
            try:
                async with websockets.connect(
                    self.relay_url,
                    additional_headers=self._build_headers(),
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    max_size=4 * 1024 * 1024,
                ) as ws:
                    connected_at = loop.time()
                    logger.info("Connected to relay console endpoint: %s", self.relay_url)
                    received_any = await self._serve_connection(ws)
            except asyncio.CancelledError:
                raise
            except InvalidStatus as exc:
                logger.warning("Relay rejected console handshake (HTTP %s)", exc.response.status_code)
            except websockets.ConnectionClosed as exc:
                code = exc.rcvd.code if exc.rcvd else None
                logger.warning("Relay console connection closed (code=%s)", code)
            except Exception as exc:
                logger.warning("Relay console connection error: %s", exc)

            if self._stopped:
                break

            healthy = connected_at is not None and (
                received_any or (loop.time() - connected_at) >= self._healthy_reset_sec
            )
            if healthy:
                delay = self._initial_backoff

            wait_sec = delay
            delay = min(self._max_backoff, delay * self._backoff_factor)
            try:
                await asyncio.wait_for(self._stop_event.wait(), timeout=wait_sec)
                break
            except TimeoutError:
                continue

    async def stop(self) -> None:
        self._stopped = True
        self._stop_event.set()
        ws = self._active_ws
        if ws is not None:
            try:
                await ws.close(code=1000, reason="Client shutting down")
            except Exception:
                pass


async def run_with_relay(
    config: AppConfig,
    on_query: QueryHandler | None = None,
    relay_url: str = "",
    *,
    gateway: SpeakerGateway | None = None,
    on_stream_query: StreamQueryHandler | None = None,
    on_session_reset: Callable[[], None] | None = None,
    auth_token: str | None = None,
    token_env: str = "XSA_RELAY_TOKEN",
    also_serve_local: bool = False,
    console_local_url: str | None = None,
) -> None:
    """Run SpeakerGateway connected outbound to the Azure relay (optionally also local WS & console tunnel)."""
    gw = gateway or SpeakerGateway(
        config,
        on_query=on_query,
        on_stream_query=on_stream_query,
        on_session_reset=on_session_reset,
    )
    gw.ensure_pipeline()
    client = RelayHostClient(
        gw,
        relay_url=relay_url,
        auth_token=auth_token,
        token_env=token_env,
    )
    coros: list[Any] = [client.serve_forever()]
    if also_serve_local:
        coros.insert(0, gw.serve_forever())
    if console_local_url:
        console_client = RelayConsoleClient(
            relay_url=relay_url,
            local_console_url=console_local_url,
            auth_token=auth_token,
            token_env=token_env,
        )
        coros.append(console_client.serve_forever())
    if len(coros) == 1:
        await coros[0]
    else:
        await asyncio.gather(*coros)
