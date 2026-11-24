#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

ENV_FILE="${XSA_ENV_FILE:-$HOME/.config/xiao-speaker-assistant/env}"
if [ -f "$ENV_FILE" ]; then
  set -a
  # shellcheck disable=SC1090
  . "$ENV_FILE"
  set +a
fi

uv run xsa serve --host 127.0.0.1 --port 4398 "$@" &
SRV_PID=$!

cleanup() {
  kill "$SRV_PID" >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

/usr/bin/python3 scripts/tcp_bridge.py --listen-port 4399 --target-port 4398
