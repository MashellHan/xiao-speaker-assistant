#!/usr/bin/env python3
"""Lightweight TCP bridge for /usr/bin/python3 (pre-permitted in macOS Firewall).

Listens on 0.0.0.0:4399 and forwards connections to 127.0.0.1:4398 where `xsa serve` runs.
"""

from __future__ import annotations

import argparse
import asyncio
import sys


async def _close_writer(writer: asyncio.StreamWriter) -> None:
    try:
        if not writer.is_closing():
            writer.close()
        await writer.wait_closed()
    except Exception:
        pass


async def pipe(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        while True:
            data = await reader.read(65536)
            if not data:
                break
            writer.write(data)
            await writer.drain()
    except (ConnectionResetError, BrokenPipeError, asyncio.CancelledError):
        pass


async def handle_client(
    client_reader: asyncio.StreamReader,
    client_writer: asyncio.StreamWriter,
    target_host: str,
    target_port: int,
) -> None:
    peer = client_writer.get_extra_info("peername")
    print(f"[bridge] accepted {peer} -> {target_host}:{target_port}", flush=True)
    try:
        up_reader, up_writer = await asyncio.open_connection(target_host, target_port)
    except Exception as exc:
        print(f"[bridge] upstream connect failed: {exc}", flush=True)
        await _close_writer(client_writer)
        return

    t1 = asyncio.create_task(pipe(client_reader, up_writer))
    t2 = asyncio.create_task(pipe(up_reader, client_writer))
    try:
        done, pending = await asyncio.wait({t1, t2}, return_when=asyncio.FIRST_COMPLETED)
        for task in pending:
            task.cancel()
        if pending:
            await asyncio.gather(*pending, return_exceptions=True)
        for task in done:
            task.result()
    except Exception:
        pass
    finally:
        await asyncio.gather(_close_writer(client_writer), _close_writer(up_writer), return_exceptions=True)


async def run_bridge(listen_host: str, listen_port: int, target_host: str, target_port: int) -> None:
    server = await asyncio.start_server(
        lambda r, w: handle_client(r, w, target_host, target_port),
        host=listen_host,
        port=listen_port,
        reuse_address=True,
    )
    print(
        f"[bridge] listening on {listen_host}:{listen_port} -> {target_host}:{target_port}",
        flush=True,
    )
    async with server:
        await server.serve_forever()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--listen-host", default="0.0.0.0")
    parser.add_argument("--listen-port", type=int, default=4399)
    parser.add_argument("--target-host", default="127.0.0.1")
    parser.add_argument("--target-port", type=int, default=4398)
    args = parser.parse_args()
    try:
        asyncio.run(run_bridge(args.listen_host, args.listen_port, args.target_host, args.target_port))
    except KeyboardInterrupt:
        return 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
