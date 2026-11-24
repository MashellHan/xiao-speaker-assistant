# Xiao Speaker Assistant

将 **Xiaomi 智能音箱 Pro (`OH2P`, Type-C)** 接入 **Mac 本地 AI 语音中枢**，连接 **Codex / Gemini** 大模型，并通过 Tool / MCP 访问你的 **Outlook 邮件、Microsoft Edge / 浏览器网页、小红书 (`xiaohongshu-mcp`)** 与本地资源。

## 系统架构

```text
[OH2P 小米智能音箱 Pro]
  ├─ 原厂超级小爱（保留，日常控灯/闹钟不变）
  └─ 最小化 SSH 补丁 + open-xiaoai Rust Client (WebSocket :4399)
         │
         ▼ 局域网双向音频流 / 控制
[Mac 主机 (Apple Silicon)]
  ├─ 语音层：VAD + 自定义唤醒词 + 本地离线 ASR + TTS (原生 / MLX-Audio / OpenAI)
  ├─ 大模型路由：Codex (OpenAI Responses/Chat) | Gemini
  ├─ 异步语音交互：慢任务先语音播报提示，完成后自动口语总结；高危写操作强制语音确认
  └─ 本地工具与 MCP：
       ├─ Outlook 邮件（Microsoft Graph + 私有文件 Token 缓存，不触碰 macOS 登录钥匙串）
       ├─ 浏览器网页访问（Playwright / Edge MCP）
       └─ 小红书查询（xiaohongshu-mcp）
```

## 快速开始

### 1. 准备 Mac 依赖

```bash
brew install uv squashfs libusb-compat
uv sync --extra dev
```

检查本机刷机与运行环境：

```bash
uv run xsa doctor
python3 flash/check_mac.py
```

### 2. 音箱刷机前置（仅做只读探测与备份）

> ⚠️ 2025/2026 年生产的 `OH2P` 出厂固件多为 `1.62.x` 及以上版本。**切勿直接刷网上老版本 `1.58.6` 的固件包**。详见 [`flash/README.md`](flash/README.md)。

1. 先用米家 App 为音箱配网，并记录 App 内显示的**固件版本号**。
2. 用**支持数据传输的 Type-C 线**连接音箱底部接口和 Mac，先做只读 USB 识别：

```bash
python3 flash/usb_probe.py
```

### 3. 配置与启动

```bash
cp config.example.yaml config.yaml
export OPENAI_API_KEY="sk-..."
export GEMINI_API_KEY="..."
export OUTLOOK_CLIENT_ID="..."
```

完成 Outlook 设备码授权（凭据保存在 `0600` 本地缓存文件中）：

```bash
uv run xsa outlook-login
```

启动语音网关：

```bash
uv run xsa serve
```

## 测试

```bash
uv run pytest
```
