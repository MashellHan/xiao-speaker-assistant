"""Configuration models and loader."""

from __future__ import annotations

import os
import shutil
import tempfile
import threading
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field

_CONFIG_PERSIST_LOCK = threading.Lock()


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
        default_factory=lambda: [
            "HEY NUOMI",
            "HEY NOMI",
            "HEY NOUMI",
            "HEY NAOMI",
            "HI NOMI",
            "HI NUOMI",
            "HI NAOMI",
            "NI NAOMI",
        ]
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
    vad_min_silence_ms: int = 450
    input_gain: float = 16.0
    agc_enabled: bool = False
    agc_target_rms: float = 0.10
    agc_max_gain: float = 48.0
    aec_enabled: bool = False
    conversation_timeout_sec: float = 45.0


class OpencodeCliConfig(BaseModel):
    binary: str = "opencode"
    model: str = "raven/gemini-4-hydrogen#low"
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
    require_host_voiceprint_for_sensitive_tools: bool = False


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
    local_keywords: list[str] | None = None,
    en_keywords: list[str] | None = None,
    en_kws_enabled: bool | None = None,
    reminders_enabled: bool | None = None,
    quiet_start_hhmm: str | None = None,
    quiet_end_hhmm: str | None = None,
    memory_enabled: bool | None = None,
    xiaohongshu_enabled: bool | None = None,
    browser_enabled: bool | None = None,
    outlook_enabled: bool | None = None,
    require_host_voiceprint: bool | None = None,
) -> Path:
    """Atomically update selected runtime fields in `config.yaml` while preserving other sections."""
    target = Path(config_path) if config_path else Path("config.yaml")
    with _CONFIG_PERSIST_LOCK:
        data: dict = {}
        if target.exists():
            loaded = yaml.safe_load(target.read_text(encoding="utf-8"))
            if isinstance(loaded, dict):
                data = loaded

        def _ensure_dict(parent: dict, key: str) -> dict:
            val = parent.get(key)
            if not isinstance(val, dict):
                val = {}
                parent[key] = val
            return val

        if edge_voice is not None or edge_rate is not None:
            speaker = _ensure_dict(data, "speaker")
            if edge_voice is not None:
                speaker["edge_voice"] = edge_voice
            if edge_rate is not None:
                speaker["edge_rate"] = edge_rate

        if model is not None:
            llm = _ensure_dict(data, "llm")
            providers = _ensure_dict(llm, "providers")
            opencode = _ensure_dict(providers, "opencode")
            opencode["model"] = model

        if local_keywords is not None or en_keywords is not None or en_kws_enabled is not None:
            speech = _ensure_dict(data, "speech")
            if local_keywords is not None:
                speech["local_keywords"] = list(local_keywords)
            if en_keywords is not None:
                speech["en_keywords"] = list(en_keywords)
            if en_kws_enabled is not None:
                speech["en_kws_enabled"] = bool(en_kws_enabled)

        if reminders_enabled is not None or quiet_start_hhmm is not None or quiet_end_hhmm is not None:
            reminders = _ensure_dict(data, "reminders")
            if reminders_enabled is not None:
                reminders["enabled"] = bool(reminders_enabled)
            if quiet_start_hhmm is not None:
                reminders["quiet_start_hhmm"] = quiet_start_hhmm
            if quiet_end_hhmm is not None:
                reminders["quiet_end_hhmm"] = quiet_end_hhmm

        if memory_enabled is not None:
            memory = _ensure_dict(data, "memory")
            memory["enabled"] = bool(memory_enabled)

        if (
            xiaohongshu_enabled is not None
            or browser_enabled is not None
            or outlook_enabled is not None
            or require_host_voiceprint is not None
        ):
            tools = _ensure_dict(data, "tools")
            if xiaohongshu_enabled is not None:
                xhs = _ensure_dict(tools, "xiaohongshu")
                xhs.setdefault("mcp_url", "http://127.0.0.1:18060/mcp")
                xhs["enabled"] = bool(xiaohongshu_enabled)
            if browser_enabled is not None:
                br = _ensure_dict(tools, "browser")
                br.setdefault("mcp_url", "http://127.0.0.1:8931/mcp")
                br.setdefault("preferred_browser", "msedge")
                br["enabled"] = bool(browser_enabled)
            if outlook_enabled is not None:
                _ensure_dict(tools, "outlook")["enabled"] = bool(outlook_enabled)
            if require_host_voiceprint is not None:
                tools["require_host_voiceprint_for_sensitive_tools"] = bool(require_host_voiceprint)

        target.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(data, allow_unicode=True, sort_keys=False)
        fd, tmp_name = tempfile.mkstemp(prefix=".config-", suffix=".yaml.tmp", dir=str(target.parent))
        tmp_path = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as fh:
                fh.write(payload)
            os.replace(tmp_path, target)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
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
