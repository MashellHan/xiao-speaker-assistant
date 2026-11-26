#!/usr/bin/env bash
set -euo pipefail

# Starts optional local MCP servers (Playwright Microsoft Edge MCP & Xiaohongshu MCP)
# for Xiao Speaker Assistant. Press Ctrl+C to stop both cleanly.

EDGE_PORT="${XSA_BROWSER_MCP_PORT:-8931}"
XHS_BIN="${XSA_XHS_BIN:-$HOME/.local/bin/xiaohongshu-mcp-darwin-arm64}"
PIDS=()

cleanup() {
  for pid in "${PIDS[@]:-}"; do
    kill "$pid" >/dev/null 2>&1 || true
  done
}
trap cleanup EXIT INT TERM

if command -v npx >/dev/null 2>&1; then
  echo "[mcp] Starting Playwright Edge MCP on http://127.0.0.1:${EDGE_PORT}/mcp ..."
  npx -y @playwright/mcp --port "$EDGE_PORT" --host 127.0.0.1 --browser msedge &
  PIDS+=($!)
else
  echo "[mcp] WARN: npx not found; skipping Playwright Edge MCP" >&2
fi

if [ -x "$XHS_BIN" ]; then
  XHS_BIN_ABS="$(cd "$(dirname "$XHS_BIN")" && pwd)/$(basename "$XHS_BIN")"
  XHS_WORKDIR="${XSA_XHS_WORKDIR:-$HOME/.local/share/xiao-speaker-assistant/xiaohongshu}"
  XHS_PORT="${XSA_XHS_PORT:-18060}"
  mkdir -m 700 -p "$XHS_WORKDIR"
  chmod 700 "$XHS_WORKDIR"
  echo "[mcp] Starting Xiaohongshu MCP on http://127.0.0.1:${XHS_PORT}/mcp (dir=$XHS_WORKDIR) ..."
  (umask 077 && cd "$XHS_WORKDIR" && exec "$XHS_BIN_ABS" -port "127.0.0.1:${XHS_PORT}") &
  PIDS+=($!)
else
  echo "[mcp] INFO: $XHS_BIN not found; skipping Xiaohongshu MCP"
fi

if [ ${#PIDS[@]} -eq 0 ]; then
  echo "[mcp] No optional MCP services started."
  exit 1
fi

wait
