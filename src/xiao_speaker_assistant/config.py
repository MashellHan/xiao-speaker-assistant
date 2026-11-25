"""Configuration models and loader."""

from __future__ import annotations

import os
import shutil
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
    edge_rate: str = "+8%"
    voice_routes: dict[str, str] = Field(default_factory=dict)
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
        default_factory=lambda: [
            "糯米糯米",
            "诺米诺米",
        ]
    )
    en_kws_enabled: bool = True
    en_keywords: list[str] = Field(
        default_factory=lambda: ["HEY NUOMI", "HEY NOMI", "HEY NOUMI", "HEY NAOMI"]
    )
    en_kws_score: float = 1.5
    en_kws_threshold: float = 0.20
    speaker_id_enabled: bool = True
    speaker_id_model: str = "3dspeaker_speech_campplus_sv_zh-cn_16k-common.onnx"
    speaker_id_threshold: float = 0.45
    speaker_profiles_file: str = "~/.local/share/xiao-speaker-assistant/speakers.json"
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
    conversation_timeout_sec: float = 45.0


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


class PersonaConfig(BaseModel):
    assistant_name: str = "Nomi"
    address_user_as: str = ""
    style: str = "亲切自然、简短口语化、直接给出核心结论"
    extra_instructions: str = ""


class MemoryConfig(BaseModel):
    enabled: bool = True
    file_path: str = "~/.local/share/xiao-speaker-assistant/memory.json"
    max_prompt_items: int = 12


class ReminderConfig(BaseModel):
    enabled: bool = True
    file_path: str = "~/.local/share/xiao-speaker-assistant/reminders.json"
    quiet_start_hhmm: str = "23:00"
    quiet_end_hhmm: str = "07:00"


class AppConfig(BaseModel):
    speaker: SpeakerConfig = Field(default_factory=SpeakerConfig)
    speech: SpeechConfig = Field(default_factory=SpeechConfig)
    llm: LLMConfig = Field(default_factory=LLMConfig)
    tools: ToolsConfig = Field(default_factory=ToolsConfig)
    persona: PersonaConfig = Field(default_factory=PersonaConfig)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    reminders: ReminderConfig = Field(default_factory=ReminderConfig)


def load_config(path: str | Path | None = None) -> AppConfig:
    target = Path(path) if path else Path("config.yaml")
    if not target.exists():
        return AppConfig()
    data = yaml.safe_load(target.read_text(encoding="utf-8")) or {}
    return AppConfig.model_validate(data)


def persist_runtime_settings(
    config_path: str | Path | None,
    *,
    model: str | None = None,
    edge_voice: str | None = None,
    edge_rate: str | None = None,
) -> Path:
    """Atomically update selected runtime fields in `config.yaml` while preserving other sections."""
    target = Path(config_path) if config_path else Path("config.yaml")
    data: dict = {}
    if target.exists():
        loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
        if isinstance(loaded, dict):
            data = loaded

    if edge_voice is not None or edge_rate is not None:
        speaker = data.setdefault("speaker", {})
        if not isinstance(speaker, dict):
            speaker = {}
            data["speaker"] = speaker
        if edge_voice is not None:
            speaker["edge_voice"] = edge_voice
        if edge_rate is not None:
            speaker["edge_rate"] = edge_rate

    if model is not None:
        llm = data.setdefault("llm", {})
        if not isinstance(llm, dict):
            llm = {}
            data["llm"] = llm
        providers = llm.setdefault("providers", {})
        if not isinstance(providers, dict):
            providers = {}
            llm["providers"] = providers
        opencode = providers.setdefault("opencode", {})
        if not isinstance(opencode, dict):
            opencode = {}
            providers["opencode"] = opencode
        opencode["model"] = model

    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(".yaml.tmp")
    tmp.write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
    os.replace(tmp, target)
    return target


def read_env_secret(env_name: str) -> str | None:
    value = os.environ.get(env_name, "").strip()
    return value or None


def resolve_binary(name_or_path: str) -> str:
    found = shutil.which(name_or_path)
    if found:
        return found
    local_bin = Path("~/.local/bin").expanduser() / name_or_path
    if local_bin.is_file():
        return str(local_bin)
    raise RuntimeError(f"CLI binary not found: {name_or_path}")
