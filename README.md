# Xiao Speaker Assistant

将 **Xiaomi 智能音箱 Pro (`OH2P`, Type-C)** 接入 **Mac 常驻 AI 语音中枢**，支持 **`Hi Nomi`** 本地唤醒与 **说话中途随时打断（退出/停止/闭嘴）**、**多轮上下文对话**、**LLM 边生成边分段语音播报**，并对接本机 **`opencode` CLI（Gemini / Raven Proxy）**、**`codex` CLI**、**Outlook 邮件**、**Microsoft Edge 浏览器**、**小红书 (`xiaohongshu-mcp`)** 与 **Azure 跨网加密中继**。

## 整体架构

```text
[OH2P 小米智能音箱 Pro]
  ├─ 原厂超级小爱（保留，日常控电视/闹钟不变）
  └─ 最小化 SSH 补丁 + /data/xiaoai-agent/client
         │
         ├── 局域网直连 ws://<Mac_IP>:4399
         └── 或跨网经 Azure 中继 wss://<Relay>/ws/speaker ↔ /ws/host
         ▼
[Mac 主机 (Apple Silicon, 常驻不休眠)]
  ├─ 音频前端：80Hz 高通滤波 + 动态 RMS AGC (tanh 防削波)
  ├─ 唤醒与打断：KWS (你好管家/你好诺米) + 播报期轻量打断 KWS (退出/停止/闭嘴) + Silero VAD
  ├─ 语音识别 (4 选 1)：
  │    ├─ sense_voice（本地离线，默认 zh 防幻觉）
  │    ├─ paraformer（本地三语模型）
  │    ├─ doubao_flash / openai_whisper（本地仅 VAD+唤醒，语音段送云端高精度听写）
  │    └─ xiaoai_asr（直接复用原厂云端高精识别）
  ├─ 对话与流式播报：opencode CLI (--session 多轮续接) / codex CLI → 分句流式 TTS
  └─ 本地工具 & MCP Server (`xsa mcp-server`)：
       ├─ Outlook 邮件（Graph API + 0600 私有缓存，发邮件前强制语音确认）
       ├─ Microsoft Edge 打开网页 / 网页正文提取
       ├─ 小红书 MCP（xiaohongshu-mcp 搜笔记、看详情）
       └─ Mac 本地文件（严格限定 Desktop/Downloads/Documents，隔离凭据）
```

## 快速开始

### 1. 准备依赖与模型

```bash
brew install uv squashfs libusb-compat
uv sync --extra dev --default-index https://mirrors.aliyun.com/pypi/simple
uv run xsa download-models
uv run xsa doctor
```

### 2. 启动语音网关

- **默认 `Hi Nomi` 本地唤醒 + SenseVoice + 流式语音**：
  ```bash
  ./scripts/start_mac_gateway.sh --asr sense_voice
  ```
- **低本地算力 + 云端高精度听写（本地仅跑 VAD/唤醒）**：
  ```bash
  export DOUBAO_ASR_APP_KEY="..."
  export DOUBAO_ASR_ACCESS_KEY="..."
  ./scripts/start_mac_gateway.sh --asr sense_voice --asr-model doubao_flash
  ```
- **直接使用原厂小米云端听写**：
  ```bash
  ./scripts/start_mac_gateway.sh
  ```

### 3. 配置 Outlook、Edge、小红书与本地 MCP Server

将本地 MCP Server 接入 `opencode` / `codex`：

```bash
# Outlook 设备码登录（凭据存在 ~/.config/xiao-speaker-assistant/ 0600 文件中，不碰登录钥匙串）
export OUTLOOK_CLIENT_ID="你的Azure应用ClientID"
uv run xsa outlook-login

# 作为 stdio MCP Server 运行（供 opencode / codex 调用）
uv run xsa mcp-server
```

### 4. Azure 跨网中继与常驻部署

- **部署 Azure 中继**：见 [`relay/README.md`](relay/README.md)（无状态转发，不存任何凭据）。
- **音箱切到 Azure / 局域网地址**：
  ```bash
  python3 scripts/deploy_speaker_client.py --host <音箱IP> --server-url wss://<你的Azure域名>/ws/speaker --restart --status
  ```
- **Mac 连接 Azure + 开机自启防休眠**：
  ```bash
  export XSA_RELAY_TOKEN="你的高强度Token"
  ./scripts/start_mac_gateway.sh --asr sense_voice --relay wss://<你的Azure域名>

  # 安装 LaunchAgent（--keep-awake 使用 caffeinate -s 防休眠）
  python3 scripts/install_launchagent.py --asr sense_voice --relay wss://<你的Azure域名> --keep-awake --install
  ```

## 测试

```bash
uv run pytest
```
