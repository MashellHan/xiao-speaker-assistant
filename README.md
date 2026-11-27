# Xiao Speaker Assistant

将 **Xiaomi 智能音箱 Pro (`OH2P`, Type-C)** 接入 **常驻 AI 语音中枢**，支持 **`Hi Nomi` / `糯米糯米` 本地唤醒**、**全双工硬件 AEC 回声消除与随时插话打断**、**家庭声纹识别**、**多轮上下文对话**、**LLM + 预热 Edge-TTS + 跨网流式播报**、**可视化 Web 控制台与延迟 Trace 拆解**，并内置标准 MCP 工具集（天气、联网搜索、长期记忆、定时提醒、Outlook 邮件、浏览器、小红书 MCP 与本地文件）。

## 整体架构

```text
[OH2P 小米智能音箱 Pro]
  ├─ 原厂超级小爱（完整保留，日常控家电/闹钟互不干扰）
  └─ 最小化 SSH 补丁 + /data/xiaoai-agent/boot.sh + client
         │
         ├── 局域网直连 ws://<Host_IP>:4399
         └── 或跨网经 Dropbear SSH + 中继 wss://<Relay>/xsa/ws/speaker ↔ /xsa/ws/host
         ▼
[主机端 (macOS / Linux)]
  ├─ 音频前端：2ch 硬件回采 FDAF AEC + 80Hz 高通滤波 + 可选动态 AGC
  ├─ 唤醒与打断：中英文 KWS + Silero VAD + 3D-Speaker 声纹识别
  ├─ 语音识别 (4 选 1)：sense_voice / paraformer（本地离线）| doubao_flash / openai_whisper | xiaoai_asr
  ├─ 对话与低延迟播报：OpenAI-Compatible / Gemini API / OpenCode V2 / Codex → 预热 Edge-TTS WSS + 流式 PCM
  ├─ Web 控制台 (`http://127.0.0.1:4397/`，支持远端安全口令门禁)：
  │    ├─ 音箱实时在线状态、14 日用量/花费/延迟趋势、最近 10 次 Trace 阶段耗时拆解
  │    └─ 模型/音色/语速/唤醒词热更新、能力开关、记忆/声纹/定时提醒管理
  └─ 统一 MCP Server (`xsa mcp-server` / `/mcp`)：25+ 工具，OpenAI-Compatible 与 OpenCode 共享全量能力
```

## 快速开始

### 1. 安装依赖与初始化配置

```bash
# macOS 安装基础工具（Linux 安装 ffmpeg 即可）
brew install uv ffmpeg

# 安装 Python 依赖（中国大陆用户可追加 --default-index https://mirrors.aliyun.com/pypi/simple）
uv sync --extra dev

# 生成 config.yaml 模板并下载本地唤醒/语音模型
uv run xsa init
uv run xsa download-models
```

### 2. 配置大模型（任选其一）并自检

- **最简单（推荐外人上手）：任意 OpenAI 兼容接口（OpenAI / DeepSeek / SiliconFlow / Ollama）**
  ```bash
  export OPENAI_API_KEY="sk-..."
  # 在 config.yaml 中设置 llm.default_provider: openai_compatible（默认模板已配置）
  uv run xsa doctor
  ```
- **Google Gemini API**：设置 `llm.default_provider: gemini_api` 并 `export GEMINI_API_KEY="..."`
- **本机 OpenCode V2 / Codex CLI**：设置 `llm.default_provider: opencode` 或 `codex`

### 3. 启动网关与打开 Web 控制台

```bash
# 本地启动（同时启动语音网关 :4399 与 Web 控制台 http://127.0.0.1:4397/）
./scripts/start_mac_gateway.sh --asr sense_voice

# 或直接运行 CLI
uv run xsa serve --asr sense_voice
```

打开浏览器访问 **`http://127.0.0.1:4397/`** 即可查看实时音箱在线状态、近 14 日延迟与用量趋势、最近 10 次对话 Trace 耗时拆解，并热更新模型、音色、唤醒词与能力开关。

### 4. OH2P 音箱安全刷机与跨网中继部署

- **首次给 OH2P 开 SSH（安全双读校验）**：见 [`flash/README.md`](flash/README.md)。
- **部署跨网中继（免公网 IP，含远端控制台安全隧道）**：见 [`relay/README.md`](relay/README.md)。
- **音箱端推送配置**：
  ```bash
  python3 scripts/deploy_speaker_client.py --host <音箱IP> --server-url wss://<你的中继域名>/ws/speaker --restart --status
  ```
- **开机自启防休眠（macOS LaunchAgent）**：
  ```bash
  python3 scripts/install_launchagent.py --asr sense_voice --relay wss://<你的中继域名> --keep-awake --install
  ```

### 5. Docker 一键部署（Linux / NAS / 服务器）

```bash
mkdir -p data models
cp config.example.yaml data/config.yaml
# 在 data/config.yaml 中配置 llm.default_provider（如 openai_compatible / gemini_api）
# 并通过环境变量提供 OPENAI_API_KEY（可选 TAVILY_API_KEY / SEARXNG_URL 增强搜索）
docker compose run --rm xiao-speaker-assistant download-models
OPENAI_API_KEY="sk-..." docker compose up -d
```

### 6. 可选能力（Outlook / MCP Server / 搜索源）

```bash
# 可选：启用高质量联网搜索（默认自动回退 Bing）
export TAVILY_API_KEY="tvly-..."
# 或自建 SearXNG：export SEARXNG_URL="http://127.0.0.1:8080"

# Outlook 设备码登录（凭据 0600 私有文件存储，不碰登录钥匙串）
export OUTLOOK_CLIENT_ID="你的Azure应用ClientID"
uv run xsa outlook-login

# 作为 stdio MCP Server 供外部客户端调用
uv run xsa mcp-server
```

## 测试

```bash
uv run pytest
```
