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
HAS_RELAY="false"
PASS_ARGS=()
for arg in "$@"; do
  if [ "$arg" = "--with-mcp" ]; then
    WITH_MCP="true"
  else
    if [ "$arg" = "--relay" ] || [[ "$arg" == --relay=* ]]; then
      HAS_RELAY="true"
    fi
    PASS_ARGS+=("$arg")
  fi
done

MCP_PID=""
if [ "$WITH_MCP" = "true" ]; then
  ./scripts/start_mcp_services.sh &
  MCP_PID=$!
fi

# When using an outbound relay, do not expose an unauthenticated LAN port (0.0.0.0:4399)
# unless explicitly opted in with XSA_LOCAL_LAN_BRIDGE=1.
LOCAL_BRIDGE="${XSA_LOCAL_LAN_BRIDGE:-}"
if [ -z "$LOCAL_BRIDGE" ]; then
  if [ "$HAS_RELAY" = "true" ]; then
    LOCAL_BRIDGE="0"
  else
    LOCAL_BRIDGE="1"
  fi
fi

if [ "$LOCAL_BRIDGE" = "1" ]; then
  uv run --no-sync xsa serve --host 127.0.0.1 --port 4398 ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} &
  SRV_PID=$!
  cleanup() {
    kill "$SRV_PID" ${MCP_PID:+"$MCP_PID"} >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM
  if [ "$(uname -s)" = "Darwin" ] && [ -x /usr/bin/python3 ]; then
    /usr/bin/python3 scripts/tcp_bridge.py --listen-port 4399 --target-port 4398
  else
    python3 scripts/tcp_bridge.py --listen-port 4399 --target-port 4398
  fi
else
  uv run --no-sync xsa serve --host 127.0.0.1 --port 4398 ${PASS_ARGS[@]+"${PASS_ARGS[@]}"} &
  SRV_PID=$!
  cleanup() {
    kill "$SRV_PID" ${MCP_PID:+"$MCP_PID"} >/dev/null 2>&1 || true
  }
  trap cleanup EXIT INT TERM
  wait "$SRV_PID"
fi
