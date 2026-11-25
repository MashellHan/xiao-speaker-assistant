import asyncio
import json

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway, is_self_echo


def test_is_self_echo_filters_tts_fragments_but_allows_followups() -> None:
    recent = ["这周五延庆天气晴朗，最高气温约4.5度，最低气温零下3.6度左右。"]

    # Residual speaker echo fragments must be rejected
    assert is_self_echo("延庆天气晴朗。", recent) is True
    assert is_self_echo("最高气温约4.5度", recent) is True
    assert is_self_echo("零下3.6度左右", recent) is True

    # Real user follow-ups while Nomi is speaking must pass through
    assert is_self_echo("北京呢？", recent) is False
    assert is_self_echo("延庆呢？", recent) is False
    assert is_self_echo("那周六呢？", recent) is False
    assert is_self_echo("查一下我电脑磁盘空间", recent) is False

    # Email confirmations/cancellations must NEVER be suppressed as self-echo even right after TTS
    confirm_prompt = ["即将发送邮件给 bob@example.com，请说确认发送或者取消。"]
    assert is_self_echo("确认", confirm_prompt) is False
    assert is_self_echo("确认发送", confirm_prompt) is False
    assert is_self_echo("取消", confirm_prompt) is False


def test_bare_wake_word_during_2ch_playback_interrupts_instead_of_blocking() -> None:
    sent_cmds: list[str] = []
    first_playing = asyncio.Event()

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            sent_cmds.append(cmd)
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(
                        gw._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": rid, "code": 0}}),
                        )
                    )
                )

    async def slow_stream(query: str):
        yield "第一句很长的话，正在播报中。"
        first_playing.set()
        await asyncio.sleep(10.0)

    cfg = AppConfig()
    cfg.speaker.tts_provider = "say"
    cfg.speech.aec_enabled = True
    cfg.speech.capture_channels = 2
    gw = SpeakerGateway(cfg, on_stream_query=slow_stream)

    async def run() -> None:
        conn = FakeConn()
        t1 = asyncio.create_task(gw._process_local_utterance(conn, "你好管家，讲个长故事"))
        await first_playing.wait()
        await asyncio.sleep(0.05)
        await asyncio.wait_for(gw._process_local_utterance(conn, "你好管家。"), timeout=2.0)
        await asyncio.wait_for(t1, timeout=2.0)

    asyncio.run(run())
    assert "stop_play" in sent_cmds


def test_echo_window_expires_after_playback_for_short_answers() -> None:
    import time

    seen: list[str] = []

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
                        gw._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": rid, "code": 0}}),
                        )
                    )
                )

    async def fake_stream(query: str):
        seen.append(query)
        yield "今天还是明天？"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "say"
    cfg.speech.aec_enabled = True
    cfg.speech.capture_channels = 2
    gw = SpeakerGateway(cfg, on_stream_query=fake_stream)

    async def run() -> None:
        conn = FakeConn()
        await gw._process_local_utterance(conn, "你好管家，订个日程")
        sess = gw._get_session(conn)
        # Simulate >1.5s elapsed since playback finished -> user answering "明天" must pass!
        sess.last_playback_ended_at = time.monotonic() - 2.0
        await gw._process_local_utterance(conn, "明天")

    asyncio.run(run())
    assert seen == ["订个日程", "明天"]


def test_full_duplex_followup_interrupts_active_reply_without_resetting_session() -> None:
    seen_queries: list[str] = []
    resets: list[bool] = []
    first_playing = asyncio.Event()

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = req.get("command")
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda: asyncio.create_task(
                        gw._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": rid, "code": 0}}),
                        )
                    )
                )

    async def fake_stream(query: str):
        seen_queries.append(query)
        if "延庆" in query:
            yield "这周五延庆天气晴朗，最高气温约4度。"
            first_playing.set()
            await asyncio.sleep(10.0)
            yield "永远不该说完的第二句。"
        else:
            yield "北京这周五也是晴天，最高气温6度。"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "say"
    cfg.speech.aec_enabled = True
    cfg.speech.capture_channels = 2
    gw = SpeakerGateway(
        cfg,
        on_stream_query=fake_stream,
        on_session_reset=lambda: resets.append(True),
    )

    async def run() -> None:
        conn = FakeConn()
        t1 = asyncio.create_task(gw._process_local_utterance(conn, "你好管家，这周五延庆天气怎么样？"))
        await first_playing.wait()
        await asyncio.sleep(0.05)

        # Residual self-echo from first sentence must be ignored without interrupting t1
        await gw._process_local_utterance(conn, "延庆天气晴朗")
        assert not t1.done()

        # Direct follow-up "北京呢？" mid-speech must interrupt t1, keep session (resets==[]), and answer Beijing
        await gw._process_local_utterance(conn, "北京呢？")
        await asyncio.wait_for(t1, timeout=2.0)

    asyncio.run(run())
    assert seen_queries == ["这周五延庆天气怎么样", "北京呢？"]
    assert resets == []
