"""Outbound WebSocket client connecting Mac SpeakerGateway to the Azure relay."""

from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable
from urllib.parse import urlsplit, urlunsplit

import websockets
from websockets.asyncio.client import ClientConnection

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
            self._auth_token = read_env_secret(token_env)

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

        audio_queue: asyncio.Queue[bytes] = asyncio.Queue(maxsize=48)
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


async def run_with_relay(
    config: AppConfig,
    on_query: QueryHandler | None = None,
    relay_url: str = "",
    *,
    on_stream_query: StreamQueryHandler | None = None,
    on_session_reset: Callable[[], None] | None = None,
    auth_token: str | None = None,
    token_env: str = "XSA_RELAY_TOKEN",
    also_serve_local: bool = False,
) -> None:
    """Run SpeakerGateway connected outbound to the Azure relay (optionally also local WS)."""
    gateway = SpeakerGateway(
        config,
        on_query=on_query,
        on_stream_query=on_stream_query,
        on_session_reset=on_session_reset,
    )
    gateway.ensure_pipeline()
    client = RelayHostClient(
        gateway,
        relay_url=relay_url,
        auth_token=auth_token,
        token_env=token_env,
    )
    if also_serve_local:
        await asyncio.gather(gateway.serve_forever(), client.serve_forever())
    else:
        await client.serve_forever()
