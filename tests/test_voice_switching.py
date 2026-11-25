from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from xiao_speaker_assistant.config import AppConfig, load_config
from xiao_speaker_assistant.tools import McpServer
from xiao_speaker_assistant.voice.voice_catalog import (
    normalize_edge_rate,
    persist_speaker_voice_config,
    resolve_edge_voice,
)
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_resolve_edge_voice_and_normalize_rate() -> None:
    assert resolve_edge_voice("晓伊").id == "zh-CN-XiaoyiNeural"
    assert resolve_edge_voice("小小").id == "zh-CN-XiaoxiaoNeural"
    assert resolve_edge_voice("yunxia").id == "zh-CN-YunxiaNeural"
    assert resolve_edge_voice("xiaoni").id == "zh-CN-shaanxi-XiaoniNeural"
    assert resolve_edge_voice("男声").id == "zh-CN-YunxiNeural"
    assert resolve_edge_voice("zh-CN-YunyangNeural").id == "zh-CN-YunyangNeural"

    with pytest.raises(ValueError):
        resolve_edge_voice("nonexistent-voice")

    assert normalize_edge_rate("15%") == "+15%"
    assert normalize_edge_rate("-10%") == "-10%"
    assert normalize_edge_rate("+200%") == "+100%"
    assert normalize_edge_rate("-90%") == "-50%"
    with pytest.raises(ValueError):
        normalize_edge_rate("fast")


def test_persist_speaker_voice_config_preserves_speech_section(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text(
        "speech:\n  capture_pcm: xsa_aec\n  capture_channels: 2\n  aec_enabled: true\n",
        encoding="utf-8",
    )
    persist_speaker_voice_config(cfg_path, edge_voice="zh-CN-XiaoyiNeural", edge_rate="+15%")
    loaded = load_config(cfg_path)
    assert loaded.speech.capture_pcm == "xsa_aec"
    assert loaded.speech.aec_enabled is True
    assert loaded.speaker.edge_voice == "zh-CN-XiaoyiNeural"
    assert loaded.speaker.edge_rate == "+15%"


def test_mcp_speaker_voice_tools_update_gateway_and_persist(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("speech:\n  aec_enabled: true\n", encoding="utf-8")
    cfg = load_config(cfg_path)

    async def dummy(q: str) -> str:
        return q

    gw = SpeakerGateway(cfg, on_query=dummy)

    class FakeConn:
        async def send(self, _raw) -> None:
            pass

    conn = FakeConn()
    sess = gw._get_session(conn)
    sess.active_voice = "zh-CN-YunyangNeural"

    srv = McpServer(
        cfg,
        config_path=cfg_path,
        on_voice_changed=gw.apply_voice_change,
        include_local_tools=True,
    )

    list_resp = asyncio.run(
        srv.handle_message(
            {"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "speaker_list_voices", "arguments": {}}}
        )
    )
    assert list_resp is not None and list_resp["result"]["isError"] is False
    voices_data = json.loads(list_resp["result"]["content"][0]["text"])
    assert len(voices_data["voices"]) == 8

    set_resp = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {
                    "name": "speaker_set_voice",
                    "arguments": {"voice": "小妮", "rate": "+12%"},
                },
            }
        )
    )
    assert set_resp is not None and set_resp["result"]["isError"] is False
    set_data = json.loads(set_resp["result"]["content"][0]["text"])
    assert set_data["voice"] == "zh-CN-shaanxi-XiaoniNeural"
    assert set_data["rate"] == "+12%"
    assert gw._config.speaker.edge_voice == "zh-CN-shaanxi-XiaoniNeural"
    assert gw._config.speaker.edge_rate == "+12%"
    # Immediate reply must clear previous wake-word route override so new voice takes effect now
    assert sess.active_voice is None
    raw_yaml = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
    assert raw_yaml["speech"]["aec_enabled"] is True
    assert raw_yaml["speaker"]["edge_voice"] == "zh-CN-shaanxi-XiaoniNeural"
