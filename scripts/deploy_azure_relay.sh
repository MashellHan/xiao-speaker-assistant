#!/usr/bin/env bash
set -euo pipefail

# Helper to deploy relay/ to an Azure VM via SSH+Docker, or print Azure Container Apps CLI commands.
# Never reads or unlocks macOS login Keychain.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  1) Deploy to an Azure Ubuntu/Debian VM with Docker via SSH:
       XSA_RELAY_TOKEN="<token>" ./scripts/deploy_azure_relay.sh --vm <user@azure-vm-ip-or-host> [--port 8765] [--ssh-key ~/.ssh/id_ed25519]

  2) Print Azure Container Apps deployment commands (run manually in your authenticated shell):
       ./scripts/deploy_azure_relay.sh --print-aca --rg <resource-group> --location <region> [--name xsa-relay]
EOF
}

VM_TARGET=""
SSH_KEY=""
RELAY_PORT="8765"
PRINT_ACA="false"
RG="xsa-relay-rg"
LOCATION="eastasia"
APP_NAME="xsa-relay"

while [ $# -gt 0 ]; do
  case "$1" in
    --vm) VM_TARGET="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --port) RELAY_PORT="$2"; shift 2 ;;
    --print-aca) PRINT_ACA="true"; shift ;;
    --rg) RG="$2"; shift 2 ;;
    --location) LOCATION="$2"; shift 2 ;;
    --name) APP_NAME="$2"; shift 2 ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2 ;;
  esac
done

if [ "$PRINT_ACA" = "true" ]; then
  cat <<EOF
# 1. Generate a strong relay token (save in ~/.config/xiao-speaker-assistant/env with chmod 600):
export XSA_RELAY_TOKEN="\$(openssl rand -hex 32)"

# 2. Deploy to Azure Container Apps (managed wss:// ingress, pinned to 1 replica):
az group create --name "$RG" --location "$LOCATION"
az containerapp up \\
  --name "$APP_NAME" \\
  --resource-group "$RG" \\
  --location "$LOCATION" \\
  --source "$REPO_ROOT/relay" \\
  --ingress external \\
  --target-port 8765 \\
  --env-vars XSA_RELAY_TOKEN="\$XSA_RELAY_TOKEN"
az containerapp update --name "$APP_NAME" --resource-group "$RG" --min-replicas 1 --max-replicas 1
EOF
  exit 0
fi

if [ -z "$VM_TARGET" ]; then
  usage
  exit 2
fi

if [ -z "${XSA_RELAY_TOKEN:-}" ]; then
  echo "ERROR: Set XSA_RELAY_TOKEN in your environment first (e.g. openssl rand -hex 32)." >&2
  exit 2
fi

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [ -n "$SSH_KEY" ]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi

tar -C "$REPO_ROOT/relay" -cf - Dockerfile server.py | ssh "${SSH_ARGS[@]}" "$VM_TARGET" "
  set -euo pipefail
  mkdir -p ~/xsa-relay && cd ~/xsa-relay
  tar -xf -
  docker build -t xsa-relay:latest .
  docker rm -f xsa-relay >/dev/null 2>&1 || true
  docker run -d \\
    --name xsa-relay \\
    --restart unless-stopped \\
    -p ${RELAY_PORT}:8765 \\
    -e XSA_RELAY_TOKEN=$(printf '%q' "$XSA_RELAY_TOKEN") \\
    xsa-relay:latest
  echo 'Relay container running on port ${RELAY_PORT}.'
"
