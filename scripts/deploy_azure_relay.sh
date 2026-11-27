#!/usr/bin/env bash
set -euo pipefail

# Helper to deploy hardened relay/ to an Azure VM via SSH+Docker, or print Azure Container Apps CLI commands.
# Never reads or unlocks macOS login Keychain.

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"

usage() {
  cat <<'EOF'
Usage:
  1) Deploy to an Azure Ubuntu/Debian VM with Docker via SSH (supports split host/speaker tokens):
       XSA_RELAY_HOST_TOKEN="<host-tok>" XSA_RELAY_SPEAKER_TOKEN="<spk-tok>" \
         ./scripts/deploy_azure_relay.sh --vm <user@azure-vm-ip-or-host> \
           [--bind 127.0.0.1] [--port 8765] [--console-port 8766] \
           [--trusted-proxies "127.0.0.1,::1,172.17.0.1"] [--ssh-key ~/.ssh/id_ed25519]

     Set --console-port 0 to disable the remote Web Console HTTP relay listener.
     (Legacy single token XSA_RELAY_TOKEN="<tok>" is still accepted if split tokens are unset.)

  2) Print Azure Container Apps deployment commands (run manually in your authenticated shell):
       ./scripts/deploy_azure_relay.sh --print-aca --rg <resource-group> --location <region> [--name xsa-relay]
EOF
}

VM_TARGET=""
SSH_KEY=""
RELAY_BIND="127.0.0.1"
RELAY_PORT="8765"
CONSOLE_PORT="8766"
TRUSTED_PROXIES=""
PRINT_ACA="false"
RG="xsa-relay-rg"
LOCATION="eastasia"
APP_NAME="xsa-relay"

while [ $# -gt 0 ]; do
  case "$1" in
    --vm) VM_TARGET="$2"; shift 2 ;;
    --ssh-key) SSH_KEY="$2"; shift 2 ;;
    --bind) RELAY_BIND="$2"; shift 2 ;;
    --port) RELAY_PORT="$2"; shift 2 ;;
    --console-port) CONSOLE_PORT="$2"; shift 2 ;;
    --trusted-proxies) TRUSTED_PROXIES="$2"; shift 2 ;;
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
# 1. Generate separate strong relay tokens (save in ~/.config/xiao-speaker-assistant/env with chmod 600):
export XSA_RELAY_HOST_TOKEN="\$(openssl rand -hex 32)"
export XSA_RELAY_SPEAKER_TOKEN="\$(openssl rand -hex 32)"

# 2. Deploy to Azure Container Apps (managed wss:// ingress, pinned to 1 replica):
az group create --name "$RG" --location "$LOCATION"
az containerapp up \\
  --name "$APP_NAME" \\
  --resource-group "$RG" \\
  --location "$LOCATION" \\
  --source "$REPO_ROOT/relay" \\
  --ingress external \\
  --target-port 8765 \\
  --env-vars RELAY_CONSOLE_PORT=0 XSA_RELAY_HOST_TOKEN="\$XSA_RELAY_HOST_TOKEN" XSA_RELAY_SPEAKER_TOKEN="\$XSA_RELAY_SPEAKER_TOKEN"
az containerapp update --name "$APP_NAME" --resource-group "$RG" --min-replicas 1 --max-replicas 1
EOF
  exit 0
fi

if [ -z "$VM_TARGET" ]; then
  usage
  exit 2
fi

HOST_TOK="${XSA_RELAY_HOST_TOKEN:-${XSA_RELAY_TOKEN:-}}"
SPK_TOK="${XSA_RELAY_SPEAKER_TOKEN:-${XSA_RELAY_TOKEN:-}}"
if [ -z "$HOST_TOK" ] || [ -z "$SPK_TOK" ]; then
  echo "ERROR: Set XSA_RELAY_HOST_TOKEN and XSA_RELAY_SPEAKER_TOKEN (or legacy XSA_RELAY_TOKEN) first." >&2
  exit 2
fi

# Validate inputs so tokens and bind/port cannot corrupt relay.env or inject shell syntax.
case "$HOST_TOK$SPK_TOK" in
  *$'\n'*|*$'\r'*)
    echo "ERROR: Relay tokens must not contain CR/LF characters." >&2
    exit 2
    ;;
esac

if ! [[ "$RELAY_BIND" =~ ^[A-Za-z0-9._:-]+$ ]]; then
  echo "ERROR: Invalid --bind address: $RELAY_BIND" >&2
  exit 2
fi
if ! [[ "$RELAY_PORT" =~ ^[0-9]+$ ]] || [ "$RELAY_PORT" -lt 1 ] || [ "$RELAY_PORT" -gt 65535 ]; then
  echo "ERROR: Invalid --port: $RELAY_PORT" >&2
  exit 2
fi
if ! [[ "$CONSOLE_PORT" =~ ^[0-9]+$ ]] || [ "$CONSOLE_PORT" -gt 65535 ]; then
  echo "ERROR: Invalid --console-port: $CONSOLE_PORT (expected 0 to disable, or 1..65535)" >&2
  exit 2
fi
if [ -n "$TRUSTED_PROXIES" ] && ! [[ "$TRUSTED_PROXIES" =~ ^[A-Za-z0-9.,:_-]+$ ]]; then
  echo "ERROR: Invalid --trusted-proxies list: $TRUSTED_PROXIES" >&2
  exit 2
fi

SSH_ARGS=(-o StrictHostKeyChecking=accept-new)
if [ -n "$SSH_KEY" ]; then
  SSH_ARGS+=(-i "$SSH_KEY")
fi

# Package Dockerfile, server.py, and 0600 relay.env in a private temp dir and stream over stdin
# so secrets NEVER appear in local/remote SSH argv or `ps` output.
STAGE_DIR="$(mktemp -d)"
chmod 700 "$STAGE_DIR"
cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT INT TERM

cp "$REPO_ROOT/relay/Dockerfile" "$REPO_ROOT/relay/server.py" "$STAGE_DIR/"
(
  umask 077
  {
    printf 'RELAY_HOST=0.0.0.0\n'
    printf 'RELAY_PORT=8765\n'
    if [ "$CONSOLE_PORT" -gt 0 ]; then
      printf 'RELAY_CONSOLE_PORT=8766\n'
    else
      printf 'RELAY_CONSOLE_PORT=0\n'
    fi
    printf 'XSA_RELAY_HOST_TOKEN=%s\n' "$HOST_TOK"
    printf 'XSA_RELAY_SPEAKER_TOKEN=%s\n' "$SPK_TOK"
    if [ -n "$TRUSTED_PROXIES" ]; then
      printf 'XSA_RELAY_TRUSTED_PROXIES=%s\n' "$TRUSTED_PROXIES"
    fi
  } > "$STAGE_DIR/relay.env"
)
chmod 600 "$STAGE_DIR/relay.env"

REMOTE_PUBLISH="$(printf '%s:%s:8765' "$RELAY_BIND" "$RELAY_PORT")"
CONSOLE_PUBLISH_FLAG=""
CONSOLE_STATUS_LABEL="disabled"
if [ "$CONSOLE_PORT" -gt 0 ]; then
  REMOTE_CONSOLE_PUBLISH="$(printf '127.0.0.1:%s:8766' "$CONSOLE_PORT")"
  CONSOLE_PUBLISH_FLAG="-p '${REMOTE_CONSOLE_PUBLISH}'"
  CONSOLE_STATUS_LABEL="${REMOTE_CONSOLE_PUBLISH}"
fi
# Remote command contains no secrets (relay.env rides inside the tar stream on stdin);
# REMOTE_PUBLISH / CONSOLE_PUBLISH_FLAG are already regex-validated above.
REMOTE_CMD="set -euo pipefail
sudo mkdir -p /opt/xsa-relay
sudo rm -f /opt/xsa-relay/._Dockerfile /opt/xsa-relay/._server.py /opt/xsa-relay/._relay.env
sudo tar --no-same-owner -C /opt/xsa-relay -xf -
sudo chown root:root /opt/xsa-relay/Dockerfile /opt/xsa-relay/server.py /opt/xsa-relay/relay.env
sudo chmod 644 /opt/xsa-relay/Dockerfile /opt/xsa-relay/server.py
sudo chmod 600 /opt/xsa-relay/relay.env
sudo docker build -t xsa-relay:latest /opt/xsa-relay
sudo docker rm -f xsa-relay >/dev/null 2>&1 || true
sudo docker run -d \\
  --name xsa-relay \\
  --restart unless-stopped \\
  --user 65532:65532 \\
  --read-only \\
  --cap-drop=ALL \\
  --security-opt=no-new-privileges:true \\
  --memory=256m \\
  --pids-limit=128 \\
  --env-file /opt/xsa-relay/relay.env \\
  -p '${REMOTE_PUBLISH}' \\
  ${CONSOLE_PUBLISH_FLAG} \\
  xsa-relay:latest
echo 'Relay container running (ws=${REMOTE_PUBLISH}, console=${CONSOLE_STATUS_LABEL}, uid=65532, read-only).'"

COPYFILE_DISABLE=1 tar -C "$STAGE_DIR" -cf - Dockerfile server.py relay.env | \
  ssh "${SSH_ARGS[@]}" "$VM_TARGET" "$REMOTE_CMD"
