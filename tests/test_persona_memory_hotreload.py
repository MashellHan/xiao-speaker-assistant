from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from xiao_speaker_assistant.config import AppConfig, LLMConfig, PersonaConfig, SpeakerConfig
from xiao_speaker_assistant.config_watcher import ConfigWatcher
from xiao_speaker_assistant.llm.router import LLMRouter
from xiao_speaker_assistant.tools import (
    McpServer,
    MemorySecurityError,
    MemoryStore,
)
from xiao_speaker_assistant.voice.playback import PlaybackController
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_memory_store_persists_0600_and_rejects_secrets(tmp_path: Path) -> None:
    mem_path = tmp_path / "data" / "memory.json"
    store = MemoryStore(mem_path)

    e1 = store.save("常住城市", "北京延庆", category="location")
    store.save("老张邮箱", "zhang@example.com", category="contact")
    # Update existing key case-insensitively
    e1_upd = store.save("常住城市", "北京市延庆区", category="location")
    assert e1_upd.id == e1.id

    assert (mem_path.stat().st_mode & 0o777) == 0o600
    assert (mem_path.parent.stat().st_mode & 0o777) == 0o700

    # Reload from disk
    reloaded = MemoryStore(mem_path)
    hits = reloaded.search("延庆")
    assert len(hits) == 1
    assert hits[0].value == "北京市延庆区"
    assert "老张邮箱: zhang@example.com" in reloaded.format_for_prompt()

    assert reloaded.forget("常住城市") is True
    assert reloaded.search("延庆") == []

    # Secret rejection
    for k, v in (
        ("github_token", "ghp_123"),
        ("邮箱密码", "123456"),
        ("api_key", "sk-abc"),
        ("笔记", "sk-abcdef1234567890abcdef"),
        ("备份", "ghp_abcdef1234567890abcdef"),
    ):
        with pytest.raises(MemorySecurityError):
            reloaded.save(k, v)


def test_mcp_server_memory_tools_and_change_callback(tmp_path: Path) -> None:
    changes: list[bool] = []
    store = MemoryStore(tmp_path / "mem.json")
    srv = McpServer(
        AppConfig(),
        memory_store=store,
        on_memory_changed=lambda: changes.append(True),
        include_local_tools=True,
    )

    save_resp = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "memory_save",
                    "arguments": {"key": "默认城市", "value": "延庆"},
                },
            }
        )
    )
    assert save_resp is not None and save_resp["result"]["isError"] is False
    assert changes == [True]

    search_resp = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "tools/call",
                "params": {"name": "memory_search", "arguments": {"query": "延庆"}},
            }
        )
    )
    assert search_resp is not None
    data = json.loads(search_resp["result"]["content"][0]["text"])
    assert data["count"] == 1
    assert data["memories"][0]["key"] == "默认城市"

    forget_resp = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "tools/call",
                "params": {"name": "memory_forget", "arguments": {"key_or_id": "默认城市"}},
            }
        )
    )
    assert forget_resp is not None
    assert json.loads(forget_resp["result"]["content"][0]["text"])["removed"] is True
    assert changes == [True, True]


def test_persona_and_memory_injected_into_router_prompt(tmp_path: Path) -> None:
    store = MemoryStore(tmp_path / "mem.json")
    store.save("喜好", "喜欢简短直接的天气播报")
    persona = PersonaConfig(
        assistant_name="小管家",
        address_user_as="韩先生",
        style="幽默风趣、干脆利落",
        extra_instructions="优先报最高温和最低温。",
    )
    router = LLMRouter(LLMConfig(), persona=persona, memory_store=store)
    prompt = router._build_system_prompt()
    assert "小管家" in prompt
    assert "韩先生" in prompt
    assert "幽默风趣、干脆利落" in prompt
    assert "优先报最高温和最低温。" in prompt
    assert "喜好: 喜欢简短直接的天气播报" in prompt


def test_config_watcher_hot_reloads_on_mtime_change_and_ignores_invalid_yaml(tmp_path: Path) -> None:
    cfg_path = tmp_path / "config.yaml"
    cfg_path.write_text("speaker:\n  edge_voice: zh-CN-XiaoxiaoNeural\n", encoding="utf-8")

    reloaded: list[AppConfig] = []
    watcher = ConfigWatcher(cfg_path, reloaded.append, poll_interval=0.05)

    async def run() -> None:
        assert await watcher.check_once() is False
        cfg_path.write_text(
            "speaker:\n  edge_voice: zh-CN-XiaoyiNeural\n  edge_rate: '+15%'\npersona:\n  assistant_name: Nomi2\n",
            encoding="utf-8",
        )
        os.utime(cfg_path, (os.stat(cfg_path).st_atime + 2, os.stat(cfg_path).st_mtime + 2))
        assert await watcher.check_once() is True
        assert reloaded[-1].speaker.edge_voice == "zh-CN-XiaoyiNeural"
        assert reloaded[-1].speaker.edge_rate == "+15%"
        assert reloaded[-1].persona.assistant_name == "Nomi2"

        # Invalid YAML must not crash watcher
        cfg_path.write_text("speaker: [unclosed", encoding="utf-8")
        os.utime(cfg_path, (os.stat(cfg_path).st_atime + 4, os.stat(cfg_path).st_mtime + 4))
        assert await watcher.check_once() is False

    asyncio.run(run())


def test_multi_voice_routing_passes_voice_override_and_edge_rate() -> None:
    cfg = AppConfig(
        speaker=SpeakerConfig(
            edge_voice="zh-CN-XiaoxiaoNeural",
            edge_rate="+12%",
            voice_routes={"": "zh-CN-BadBlank", "你好管家": "zh-CN-YunxiNeural"},
        )
    )
    called_voices: list[tuple[str, str]] = []

    async def fake_stream(text: str, *, voice: str, rate: str, sample_rate: int, chunk_bytes: int):
        called_voices.append((voice, rate))
        yield b"\x01\x00" * 1200

    async def dummy(q: str) -> str:
        return f"回答：{q}"

    gw = SpeakerGateway(cfg, on_query=dummy)

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            if req.get("command") in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(
                        gw._handle_text_message(self, json.dumps({"Response": {"id": rid, "code": 0}}))
                    )
                )

    async def run() -> None:
        conn = FakeConn()
        with patch("xiao_speaker_assistant.voice.tts_stream.stream_edge_tts_pcm", fake_stream):
            await gw._process_local_utterance(conn, "你好管家，你好")
            # Switching to an unrouted wake word must reset back to default edge_voice (and blank route key must be ignored)
            await gw._process_local_utterance(conn, "Hi Nomi，你好")

    asyncio.run(run())
    assert called_voices == [("zh-CN-YunxiNeural", "+12%"), ("zh-CN-XiaoxiaoNeural", "+12%")]
