"""Configuration models and loader."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field


class SpeakerConfig(BaseModel):
    model: str = "OH2P"
    ws_host: str = "0.0.0.0"
    ws_port: int = 4399
    auth_token_env: str = "XSA_SPEAKER_TOKEN"
    tts_provider: Literal["edge", "say", "xiaoai", "openai", "mlx_audio"] = "edge"
    edge_voice: str = "zh-CN-XiaoxiaoNeural"
    say_voice: str = "Tingting"
    tts_sample_rate: int = 24000


class SpeechConfig(BaseModel):
    wakeup_keywords: list[str] = Field(
        default_factory=lambda: [
            "帮我",
            "请",
            "nomi",
            "你好管家",
            "小助手",
            "看一下",
            "查一下",
            "我电脑",
            "电脑",
            "邮件",
            "邮箱",
            "小红书",
            "网页",
        ]
    )
    local_keywords: list[str] = Field(
        default_factory=lambda: ["你好管家", "你好诺米", "你好糯米", "嗨糯米"]
    )
    en_kws_enabled: bool = True
    en_keywords: list[str] = Field(
        default_factory=lambda: ["HI NOMI", "HI NAOMI", "HI KNOW ME", "HI NO ME", "HEY NOMI", "HI NOAMI"]
    )
    en_kws_score: float = 2.0
    en_kws_threshold: float = 0.25
    interrupt_keywords: list[str] = Field(
        default_factory=lambda: ["退出", "停止", "闭嘴", "退下", "别说了", "再见"]
    )
    asr_provider: Literal["sense_voice", "xiaoai_asr"] = "sense_voice"
    asr_model: Literal["sense_voice", "paraformer", "doubao_flash", "openai_whisper"] = "sense_voice"
    asr_language: str = "auto"
    whisper_base_url: str = "https://api.openai.com/v1"
    whisper_api_key_env: str | None = "OPENAI_API_KEY"
    whisper_model: str = "whisper-1"
    models_dir: str = "models"
    capture_pcm: str = "noop"
    capture_channels: Literal[1, 2] = 1
    capture_bits_per_sample: Literal[16, 32] = 16
    s32_shift: int = 16
    kws_score: float = 2.0
    kws_threshold: float = 0.20
    vad_threshold: float = 0.12
    vad_min_silence_ms: int = 550
    input_gain: float = 16.0
    agc_enabled: bool = False
    agc_target_rms: float = 0.10
    agc_max_gain: float = 48.0
    aec_enabled: bool = False
    conversation_timeout_sec: float = 25.0


class OpencodeCliConfig(BaseModel):
    binary: str = "opencode"
    model: str = "raven/gemini-4-hydrogen"
    timeout_sec: float = 45.0
    auto_approve: bool = True
    server_url: str = "http://127.0.0.1:4396"
    agent_home: str = "~/.local/share/xiao-speaker-assistant/agent-home"
    auto_start_server: bool = True
    password_env: str = "XSA_OPENCODE_PASSWORD"


class CodexCliConfig(BaseModel):
    binary: str = "codex"
    model: str | None = None
    workdir: str = "."
    timeout_sec: float = 120.0


class CodexApiConfig(BaseModel):
    base_url: str = "https://api.openai.com/v1"
    api_key_env: str = "OPENAI_API_KEY"
    model: str = "gpt-5.1-codex"


class GeminiApiConfig(BaseModel):
    api_key_env: str = "GEMINI_API_KEY"
    model: str = "gemini-2.5-flash"


class LLMProvidersConfig(BaseModel):
    opencode: OpencodeCliConfig = Field(default_factory=OpencodeCliConfig)
    codex: CodexCliConfig = Field(default_factory=CodexCliConfig)
    codex_api: CodexApiConfig = Field(default_factory=CodexApiConfig)
    gemini_api: GeminiApiConfig = Field(default_factory=GeminiApiConfig)


class LLMConfig(BaseModel):
    default_provider: Literal["opencode", "codex", "codex_api", "gemini_api"] = "opencode"
    spoken_max_chars: int = 220
    providers: LLMProvidersConfig = Field(default_factory=LLMProvidersConfig)


class OutlookToolConfig(BaseModel):
    enabled: bool = True
    client_id_env: str = "OUTLOOK_CLIENT_ID"
    tenant_id: str = "consumers"
    token_cache_file: str = "~/.config/xiao-speaker-assistant/outlook_token_cache.json"
    require_voice_confirmation_for_send: bool = True


class McpEndpointConfig(BaseModel):
    enabled: bool = False
    mcp_url: str
    preferred_browser: str | None = None


class ToolsConfig(BaseModel):
    outlook: OutlookToolConfig = Field(default_factory=OutlookToolConfig)
    xiaohongshu: McpEndpointConfig = Field(
        default_factory=lambda: McpEndpointConfig(enabled=False, mcp_url="http://127.0.0.1:18060/mcp")
    )
    browser: McpEndpointConfig = Field(
        default_factory=lambda: McpEndpointConfig(
            enabled=False, mcp_url="http://127.0.0.1:8931/mcp", preferred_browser="msedge"
        )
    )
    mcp_http_port: int = 4397


class AppConfig(BaseModel):
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    target = Path(path) if path else Path("config.yaml")
    if not target.exists():
        return AppConfig()
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)


def read_env_secret(env_name: str) -> str | None:
    value = os.environ.get(env_name, "").strip()
    return value or None
