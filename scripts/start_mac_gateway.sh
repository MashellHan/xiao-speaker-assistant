#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

uv run xsa serve --host 127.0.0.1 --port 4398 "$@" &
SRV_PID=$!

cleanup() {
  kill "$SRV_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

/usr/bin/python3 scripts/tcp_bridge.py --listen-port 4399 --target-port 4398
