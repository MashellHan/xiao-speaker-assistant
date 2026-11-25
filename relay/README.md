# Xiao Speaker Assistant — Azure WebSocket Relay

Stateless WebSocket relay that bridges the speaker (`/ws/speaker`) and the Mac host (`/ws/host`) over TLS (`wss://`) so the Mac never needs a public IP or inbound port forwarding.

## Protocol & Security

- **Endpoints**:
  - `ws://<your-domain>/ws/speaker?token=<XSA_RELAY_TOKEN>` (or `wss://...` if client supports TLS): Xiaomi Smart Speaker client (`open-xiaoai`). Note: stock `open-xiaoai-client` v1.0.0 (`connect_async(url)`) does not compile TLS or send custom `Authorization` headers, so `/ws/speaker` accepts `?token=<XSA_RELAY_TOKEN>` (validated with `hmac.compare_digest`).
  - `wss://<your-domain>/ws/host`: Mac host dialing outbound via `RelayHostClient` (`Authorization: Bearer <XSA_RELAY_TOKEN>`)
  - `GET /healthz`: Plain HTTP `200 ok` health check for load balancers / Container Apps
- **Authentication**: Set `XSA_RELAY_TOKEN` to a strong random secret. `/ws/host` requires `Authorization: Bearer <XSA_RELAY_TOKEN>`, while `/ws/speaker` accepts either `Authorization: Bearer` or `?token=<XSA_RELAY_TOKEN>`. Invalid/missing tokens close with `4401`.
- **Reverse proxy examples**: See `relay/Caddyfile.example` and `relay/nginx.conf.example`, or run `./scripts/deploy_azure_relay.sh --help`.
- **Reconnection**: If a speaker or host reconnects, the relay cleanly replaces the older connection for that role without disturbing the peer.

Generate a token:

```bash
openssl rand -hex 32
```

## Option 1: Azure VM + Docker + Caddy (Recommended)

Caddy automatically obtains and renews Let's Encrypt TLS certificates and proxies WebSocket upgrades out of the box.

1. Build and run the relay container bound to localhost (`127.0.0.1:8765`):

```bash
cd relay
docker build -t xsa-relay:latest .
docker run -d \
  --name xsa-relay \
  --restart unless-stopped \
  -p 127.0.0.1:8765:8765 \
  -e XSA_RELAY_TOKEN="your-secret-token" \
  xsa-relay:latest
```

2. Configure `/etc/caddy/Caddyfile` (replace `relay.example.com` with your Azure VM DNS name):

```caddyfile
relay.example.com {
    reverse_proxy 127.0.0.1:8765
}
```

3. Open ports `80` and `443` in your Azure VM Network Security Group (NSG) and reload Caddy:

```bash
sudo systemctl reload caddy
```

## Option 2: Azure VM + Docker + Nginx (`wss://`)

If you terminate TLS with Nginx + Certbot, forward `Upgrade`, `Connection`, and `Authorization` headers and raise the read/send timeouts for long-lived WebSocket sessions:

```nginx
server {
    listen 443 ssl http2;
    server_name relay.example.com;

    ssl_certificate     /etc/letsencrypt/live/relay.example.com/fullchain.pem;
    ssl_certificate_key /etc/letsencrypt/live/relay.example.com/privkey.pem;

    location /ws/ {
        proxy_pass http://127.0.0.1:8765;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header Authorization $http_authorization;
        proxy_read_timeout 3600s;
        proxy_send_timeout 3600s;
    }

    location = /healthz {
        proxy_pass http://127.0.0.1:8765/healthz;
    }
}
```

## Option 3: Azure Container Apps (Managed `wss://`)

Azure Container Apps provides managed TLS (`wss://`) ingress automatically:

1. Push the image from `relay/` to Azure Container Registry (`az acr build`).
2. Deploy with external ingress targeting port `8765` and **`min-replicas 1`** (so the active speaker and host share the same in-memory relay instance):

```bash
az containerapp up \
  --name xsa-relay \
  --resource-group <rg> \
  --location <region> \
  --source ./relay \
  --ingress external \
  --target-port 8765 \
  --env-vars XSA_RELAY_TOKEN="<your-secret-token>"

az containerapp update \
  --name xsa-relay \
  --resource-group <rg> \
  --min-replicas 1 \
  --max-replicas 1
```

## Connecting the Mac Host

On your Mac, export `XSA_RELAY_TOKEN` and connect `RelayHostClient` (or `run_with_relay`) to `wss://relay.example.com/ws/host`:

```python
import asyncio
from xiao_speaker_assistant.config import load_config
from xiao_speaker_assistant.voice.relay_client import run_with_relay

config = load_config()

async def on_query(query: str) -> str:
    return f"收到：{query}"

asyncio.run(run_with_relay(config, on_query, "wss://relay.example.com/ws/host"))
```
