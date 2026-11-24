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
    tts_provider: Literal["xiaoai", "openai", "mlx_audio"] = "xiaoai"


class SpeechConfig(BaseModel):
    wakeup_keywords: list[str] = Field(
        default_factory=lambda: ["帮我", "请", "nomi", "你好管家", "小助手"]
    )
    local_keywords: list[str] = Field(default_factory=lambda: ["hi nomi", "你好管家"])
    asr_provider: Literal["sense_voice", "xiaoai_asr"] = "xiaoai_asr"
    models_dir: str = "models"
    capture_bits_per_sample: Literal[16, 32] = 32
    kws_score: float = 2.5
    kws_threshold: float = 0.15
    vad_threshold: float = 0.2
    vad_min_silence_ms: int = 700
    input_gain: float = 1.0
    conversation_timeout_sec: float = 25.0


class OpencodeCliConfig(BaseModel):
    binary: str = "opencode"
    model: str = "raven/gemini-4-hydrogen"
    timeout_sec: float = 120.0
    auto_approve: bool = False


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


class AppConfig(BaseModel):
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    target = Path(path) if path else Path("config.yaml")
    if not target.exists():
        example = Path("config.example.yaml")
        if path is None and example.exists():
            data = yaml.safe_load(example.read_text(encoding="utf-8")) or {}
            return AppConfig.model_validate(data)
        return AppConfig()
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)


def read_env_secret(env_name: str) -> str | None:
    value = os.environ.get(env_name, "").strip()
    return value or None
