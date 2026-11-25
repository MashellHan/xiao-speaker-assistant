"""Stateless WebSocket relay bridging `/ws/speaker` and `/ws/host`."""

from __future__ import annotations

import argparse
import asyncio
import hmac
import logging
import os
import sys
from http import HTTPStatus
from typing import Literal
from urllib.parse import parse_qs, urlsplit

import websockets
from websockets.asyncio.server import Request, Response, Server, ServerConnection

logger = logging.getLogger("xsa.relay")

Role = Literal["speaker", "host"]

CLOSE_REPLACED = 4000
CLOSE_UNAUTHORIZED = 4401
CLOSE_NOT_FOUND = 4404


def _read_token_from_env(env_name: str = "XSA_RELAY_TOKEN") -> str | None:
    value = os.environ.get(env_name, "").strip()
    return value or None


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


class RelayServer:
    """Stateless bidirectional relay between one active speaker and one active host."""

    def __init__(
        self,
        host: str = "0.0.0.0",
        port: int = 8765,
        *,
        token: str | None = None,
        max_size: int | None = 1048576,
    ) -> None:
        self.host = host
        self.port = port
        self._token = _read_token_from_env() if token is None else (token.strip() or None)
        self._max_size = max_size
        self._speaker: ServerConnection | None = None
        self._host_conn: ServerConnection | None = None
        self._server: Server | None = None

    @property
    def bound_port(self) -> int:
        if self._server is None or not self._server.sockets:
            return self.port
        return int(self._server.sockets[0].getsockname()[1])

    def _is_authorized(self, conn: ServerConnection, *, allow_query_token: bool = False) -> bool:
        if self._token is None:
            return True
        if conn.request is None:
            return False
        raw_header = conn.request.headers.get("Authorization", "")
        presented = _extract_bearer_token(raw_header)
        if presented is not None:
            return hmac.compare_digest(presented, self._token)
        if allow_query_token:
            query_params = parse_qs(urlsplit(conn.request.path).query, keep_blank_values=False)
            tokens = query_params.get("token")
            if tokens and len(tokens) == 1 and tokens[0].strip():
                return hmac.compare_digest(tokens[0].strip(), self._token)
        return False

    def _get_conn(self, role: Role) -> ServerConnection | None:
        return self._speaker if role == "speaker" else self._host_conn

    def _set_conn(self, role: Role, conn: ServerConnection | None) -> None:
        if role == "speaker":
            self._speaker = conn
        else:
            self._host_conn = conn

    def _get_peer(self, role: Role) -> ServerConnection | None:
        return self._host_conn if role == "speaker" else self._speaker

    def _process_request(
        self, conn: ServerConnection, request: Request
    ) -> Response | None:
        # Allow HTTP health checks (for example Azure Container Apps probes) without a WS upgrade.
        if request.headers.get("Upgrade", "").lower() != "websocket":
            path = urlsplit(request.path).path
            if path in {"/", "/health", "/healthz"}:
                return conn.respond(HTTPStatus.OK, "ok\n")
            return conn.respond(HTTPStatus.NOT_FOUND, "not found\n")
        return None

    async def _handle_connection(self, conn: ServerConnection) -> None:
        path = urlsplit(conn.request.path).path if conn.request else ""
        if not self._is_authorized(conn, allow_query_token=(path == "/ws/speaker")):
            logger.warning("Rejected unauthorized connection from %s", conn.remote_address)
            await conn.close(code=CLOSE_UNAUTHORIZED, reason="Unauthorized")
            return

        if path == "/ws/speaker":
            role: Role = "speaker"
        elif path == "/ws/host":
            role = "host"
        else:
            logger.warning("Rejected unknown endpoint %r from %s", path, conn.remote_address)
            await conn.close(code=CLOSE_NOT_FOUND, reason="Not Found")
            return

        previous = self._get_conn(role)
        self._set_conn(role, conn)
        if previous is not None and previous is not conn:
            logger.info("Replacing active %s connection", role)
            # Close asynchronously so a half-open peer cannot stall the replacement up to close_timeout.
            asyncio.create_task(
                _safe_close(previous, code=CLOSE_REPLACED, reason=f"Replaced by new {role}")
            )

        logger.info("Connected %s from %s", role, conn.remote_address)
        try:
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
            logger.info("Disconnected %s", role)

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
        logger.info(
            "Relay listening on ws://%s:%d (auth=%s)",
            self.host,
            self.bound_port,
            "enabled" if self._token else "disabled",
        )
        return self

    async def close(self) -> None:
        for role in ("speaker", "host"):
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

    relay = RelayServer(host=args.host, port=args.port)
    try:
        asyncio.run(relay.serve_forever())
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
