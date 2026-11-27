"""Edge-TTS Chinese voice catalog, alias resolver, and safe config.yaml persistence."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class VoiceProfile:
    id: str
    name: str
    gender: str
    style: str


EDGE_ZH_VOICES: tuple[VoiceProfile, ...] = (
    VoiceProfile("zh-CN-XiaoxiaoNeural", "晓晓", "female", "温暖柔和、标准自然"),
    VoiceProfile("zh-CN-XiaoyiNeural", "晓伊", "female", "轻快活泼、口语自然"),
    VoiceProfile("zh-CN-YunxiNeural", "云希", "male", "阳光自然、像日常聊天"),
    VoiceProfile("zh-CN-YunyangNeural", "云扬", "male", "沉稳专业、适合管家"),
    VoiceProfile("zh-CN-YunjianNeural", "云健", "male", "饱满有力"),
    VoiceProfile("zh-CN-YunxiaNeural", "云夏", "male", "清亮少年感"),
    VoiceProfile("zh-CN-liaoning-XiaobeiNeural", "小贝", "female", "东北话、幽默风趣"),
    VoiceProfile("zh-CN-shaanxi-XiaoniNeural", "小妮", "female", "陕西话、爽朗"),
)

_VOICE_BY_ID: dict[str, VoiceProfile] = {v.id: v for v in EDGE_ZH_VOICES}

_VOICE_ALIASES: dict[str, str] = {
    "晓晓": "zh-CN-XiaoxiaoNeural",
    "小小": "zh-CN-XiaoxiaoNeural",
    "xiaoxiao": "zh-CN-XiaoxiaoNeural",
    "默认女声": "zh-CN-XiaoxiaoNeural",
    "晓伊": "zh-CN-XiaoyiNeural",
    "xiaoyi": "zh-CN-XiaoyiNeural",
    "活泼女声": "zh-CN-XiaoyiNeural",
    "女声": "zh-CN-XiaoyiNeural",
    "云希": "zh-CN-YunxiNeural",
    "yunxi": "zh-CN-YunxiNeural",
    "男声": "zh-CN-YunxiNeural",
    "阳光男声": "zh-CN-YunxiNeural",
    "云扬": "zh-CN-YunyangNeural",
    "yunyang": "zh-CN-YunyangNeural",
    "管家": "zh-CN-YunyangNeural",
    "沉稳男声": "zh-CN-YunyangNeural",
    "云健": "zh-CN-YunjianNeural",
    "yunjian": "zh-CN-YunjianNeural",
    "云夏": "zh-CN-YunxiaNeural",
    "yunxia": "zh-CN-YunxiaNeural",
    "少年": "zh-CN-YunxiaNeural",
    "小贝": "zh-CN-liaoning-XiaobeiNeural",
    "xiaobei": "zh-CN-liaoning-XiaobeiNeural",
    "东北话": "zh-CN-liaoning-XiaobeiNeural",
    "小妮": "zh-CN-shaanxi-XiaoniNeural",
    "xiaoni": "zh-CN-shaanxi-XiaoniNeural",
    "陕西话": "zh-CN-shaanxi-XiaoniNeural",
}


def resolve_edge_voice(query: str) -> VoiceProfile:
    """Resolve an Edge-TTS voice ID or friendly alias against the curated catalog."""
    raw = query.strip()
    if not raw:
        raise ValueError("Voice name must not be empty.")
    if raw in _VOICE_BY_ID:
        return _VOICE_BY_ID[raw]
    lower = raw.lower()
    for vid, profile in _VOICE_BY_ID.items():
        if vid.lower() == lower:
            return profile
    if lower in _VOICE_ALIASES:
        return _VOICE_BY_ID[_VOICE_ALIASES[lower]]
    for alias, vid in sorted(_VOICE_ALIASES.items(), key=lambda kv: len(kv[0]), reverse=True):
        if alias in lower:
            return _VOICE_BY_ID[vid]
    available = ", ".join(f"{v.name}({v.id})" for v in EDGE_ZH_VOICES)
    raise ValueError(f"Unsupported voice {query!r}. Available voices: {available}")


def normalize_edge_rate(rate_str: str) -> str:
    """Validate and clamp a speech rate string to `[-50%, +100%]`."""
    raw = rate_str.strip()
    m = re.fullmatch(r"([+-]?)(\d{1,3})%?", raw)
    if not m:
        raise ValueError(f"Invalid rate {rate_str!r}; expected percentage like '+10%' or '-10%'.")
    sign = m.group(1) or "+"
    val = int(m.group(2))
    signed = -val if sign == "-" else val
    clamped = max(-50, min(100, signed))
    return f"+{clamped}%" if clamped >= 0 else f"{clamped}%"


def persist_speaker_voice_config(
    config_path: str | Path | None,
    *,
    edge_voice: str | None = None,
    edge_rate: str | None = None,
) -> Path:
    """Update `speaker.edge_voice` / `speaker.edge_rate` atomically via `persist_runtime_settings`."""
    from xiao_speaker_assistant.config import persist_runtime_settings

    return persist_runtime_settings(config_path, edge_voice=edge_voice, edge_rate=edge_rate)


def list_voice_profiles_dicts() -> list[dict[str, str]]:
    return [asdict(v) for v in EDGE_ZH_VOICES]
