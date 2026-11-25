from __future__ import annotations

import asyncio
import json
from pathlib import Path

import numpy as np

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.tools.mcp_server import McpServer
from xiao_speaker_assistant.voice.audio_pipeline import match_wakeup_phrase
from xiao_speaker_assistant.voice.speaker_id import SpeakerProfileStore
from xiao_speaker_assistant.voice.ws_server import CURRENT_SPEAKER, SpeakerGateway


def _unit_vec(idx: int, dim: int = 192) -> list[float]:
    v = np.zeros(dim, dtype=np.float32)
    v[idx] = 1.0
    return v.tolist()


def test_reduplicated_wakewords_matching() -> None:
    kws = ["糯米糯米", "诺米诺米"]
    assert match_wakeup_phrase("糯米糯米", kws) == ("糯米糯米", "")
    assert match_wakeup_phrase("糯米、糯米，今天天气怎么样", kws) == ("糯米糯米", "今天天气怎么样")
    assert match_wakeup_phrase("诺米诺米，帮我讲个故事", kws) == ("诺米诺米", "帮我讲个故事")
    assert match_wakeup_phrase("糯米诺米，明天北京天气", kws) == ("糯米糯米", "明天北京天气")
    assert match_wakeup_phrase("Hey Nuomi, what time is it", kws) == ("hi nomi", "what time is it")
    assert match_wakeup_phrase("Hey Nomi", kws) == ("hi nomi", "")


def test_speaker_profile_store_enroll_previous_update_and_identify(tmp_path: Path) -> None:
    spk_file = tmp_path / "speakers.json"
    store = SpeakerProfileStore(spk_file)

    child_vec = _unit_vec(1)
    host_vec = _unit_vec(2)

    # Simulate child ("拳拳") asking a question, then host ("大雄") saying "记下来刚才和你说话的是拳拳"
    store.record_utterance("讲个恐龙故事", child_vec)
    store.record_utterance("记下来刚才和你说话的是拳拳，他是小孩", host_vec)

    profile, rec = store.enroll_from_recent(
        "拳拳",
        role="child",
        capabilities=["storytelling", "dinosaur_facts", "kid_safe_only"],
        notes="喜欢恐龙",
        target="previous",
    )
    assert profile.name == "拳拳"
    assert profile.role == "child"
    assert rec.text == "讲个恐龙故事"
    assert "kid_safe_only" in profile.capabilities
    assert (spk_file.stat().st_mode & 0o777) == 0o600

    # Enroll host with target="current"
    host_profile, host_rec = store.enroll_from_recent(
        "大雄",
        role="host",
        target="current",
    )
    assert host_profile.name == "大雄"
    assert host_rec.text.startswith("记下来刚才")

    # Reload from disk and verify identification + extensible update
    reloaded = SpeakerProfileStore(spk_file)
    m_child = reloaded.identify(child_vec, threshold=0.45)
    assert m_child is not None
    assert m_child.name == "拳拳"
    assert "dinosaur_facts" in m_child.capabilities
    prefix = m_child.format_context_prefix()
    assert "拳拳" in prefix and "喜欢恐龙" in prefix

    m_host = reloaded.identify(host_vec, threshold=0.45)
    assert m_host is not None and m_host.name == "大雄"

    updated = reloaded.update_profile(
        "拳拳",
        capabilities=["storytelling", "english_words"],
        preferred_voice="晓伊",
    )
    assert updated is not None
    assert updated.preferred_voice == "晓伊"
    assert "english_words" in updated.capabilities

    # target="previous" must refuse to enroll caller's own voice when only 1 utterance exists.
    single_store = SpeakerProfileStore(tmp_path / "single.json")
    single_store.record_utterance("记住刚才说话的是拳拳", host_vec)
    try:
        single_store.enroll_from_recent("拳拳", role="child", target="previous")
        raise AssertionError("Expected ValueError when fewer than 2 embeddings exist")
    except ValueError:
        pass

    assert reloaded.delete_profile("拳拳") is True
    assert reloaded.identify(child_vec, threshold=0.45) is None


def test_mcp_voiceprint_tools_and_gateway_speaker_context(tmp_path: Path) -> None:
    store = SpeakerProfileStore(tmp_path / "speakers.json")
    child_vec = _unit_vec(5)
    host_vec = _unit_vec(9)

    store.record_utterance("为什么霸王龙手很短", child_vec)
    store.record_utterance("记住刚才和你说话的是拳拳，他是小孩", host_vec)

    srv = McpServer(AppConfig(), speaker_store=store, include_local_tools=True)
    res = asyncio.run(
        srv.handle_message(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "tools/call",
                "params": {
                    "name": "speaker_enroll_voiceprint",
                    "arguments": {
                        "name": "拳拳",
                        "role": "child",
                        "capabilities": ["storytelling", "kid_safe_only"],
                        "notes": "4岁男孩",
                        "preferred_voice": "晓伊",
                        "target": "previous",
                    },
                },
            }
        )
    )
    assert res is not None and res["result"]["isError"] is False
    enrolled = json.loads(res["result"]["content"][0]["text"])
    assert enrolled["status"] == "enrolled"
    assert enrolled["captured_utterance"] == "为什么霸王龙手很短"
    assert enrolled["preferred_voice"] == "zh-CN-XiaoyiNeural"

    # Now simulate SpeakerGateway receiving a follow-up utterance from 拳拳
    seen_speakers: list[str] = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9001)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(
                        gw._handle_text_message(self, json.dumps({"Response": {"id": rid, "code": 0}}))
                    )
                )

    async def on_query(q: str) -> str:
        spk = CURRENT_SPEAKER.get()
        seen_speakers.append(f"{spk.name}:{spk.role}" if spk else "unknown")
        return f"收到：{q}"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "say"
    cfg.speaker.edge_voice = "zh-CN-YunxiNeural"
    gw = SpeakerGateway(cfg, on_query=on_query, speaker_store=store)

    # Enroll host with default voice (preferred_voice=None) and verify voice does NOT stick across turns.
    store.enroll("大雄", host_vec, role="host")
    spoken_voices: list[str | None] = []
    orig_speak = gw._get_playback

    async def run() -> None:
        from unittest.mock import patch

        conn = FakeConn()
        gw._connections.add(conn)  # type: ignore[arg-type]
        pb = orig_speak(conn)

        async def fake_pb_speak(txt: str, *, interrupt_native: bool = True, voice_override: str | None = None) -> None:
            spoken_voices.append(voice_override)

        with patch.object(pb, "speak", side_effect=fake_pb_speak):
            await gw._process_local_utterance(conn, "再讲一个恐龙故事", embedding=child_vec)
            await gw._process_local_utterance(conn, "我问个工作问题", embedding=host_vec)
            # Bare wake word "糯米糯米" must reach match_wakeup_phrase and trigger chime instead of fragment drop.
            await gw._process_local_utterance(conn, "糯米糯米")

    asyncio.run(run())
    assert seen_speakers == ["拳拳:child", "大雄:host"]
    assert spoken_voices == ["zh-CN-XiaoyiNeural", None]
