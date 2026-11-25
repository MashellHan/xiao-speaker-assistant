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

WITH_MCP="false"
PASS_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--with-mcp" ]; then
    WITH_MCP="true"
  else
    PASS_ARGS+=("$arg")
  fi
done

MCP_PID=""
if [ "$WITH_MCP" = "true" ]; then
  ./scripts/start_mcp_services.sh &
  MCP_PID=$!
fi

uv run xsa serve --host 127.0.0.1 --port 4398 "${PASS_ARGS[@]}" &
SRV_PID=$!

cleanup() {
  kill "$SRV_PID" ${MCP_PID:+"$MCP_PID"} >/dev/null 2>&1 || true
}
trap cleanup EXIT INT TERM

/usr/bin/python3 scripts/tcp_bridge.py --listen-port 4399 --target-port 4398
