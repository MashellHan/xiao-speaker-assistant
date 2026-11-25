import json

from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_extract_final_asr_parses_open_xiaoai_newline_event() -> None:
    line = json.dumps(
        {
            "header": {
                "namespace": "SpeechRecognizer",
                "name": "RecognizeResult",
                "dialog_id": "dlg-123",
                "id": "msg-1",
            },
            "payload": {
                "is_final": True,
                "is_vad_begin": False,
                "results": [{"text": "帮我介绍一下量子计算", "confidence": 0.99}],
            },
        },
        ensure_ascii=False,
    )
    event = {
        "Event": {
            "id": "evt-1",
            "event": "instruction",
            "data": {"NewLine": line},
        }
    }
    assert SpeakerGateway._extract_final_asr(event) == ("dlg-123", "帮我介绍一下量子计算")


def test_extract_final_asr_ignores_partial_results() -> None:
    line = json.dumps(
        {
            "header": {
                "namespace": "SpeechRecognizer",
                "name": "RecognizeResult",
                "dialog_id": "dlg-123",
                "id": "msg-1",
            },
            "payload": {
                "is_final": False,
                "is_vad_begin": False,
                "results": [{"text": "帮我"}],
            },
        },
        ensure_ascii=False,
    )
    event = {"Event": {"id": "evt-1", "event": "instruction", "data": {"NewLine": line}}}
    assert SpeakerGateway._extract_final_asr(event) is None


def test_should_ignore_bare_wake_word() -> None:
    assert SpeakerGateway.should_ignore_utterance("小爱同学") is True
    assert SpeakerGateway.should_ignore_utterance("小爱同学。") is True
    assert SpeakerGateway.should_ignore_utterance("介绍一下自己") is False


def test_detects_native_speak_event() -> None:
    line = json.dumps({"header": {"namespace": "SpeechSynthesizer", "name": "Speak"}})
    event = {"Event": {"id": "evt-2", "event": "instruction", "data": {"NewLine": line}}}
    assert SpeakerGateway._is_native_speak_event(event) is True


def test_is_exit_command_matches_short_and_phrase_exits() -> None:
    assert SpeakerGateway.is_exit_command("退出") is True
    assert SpeakerGateway.is_exit_command("退出。") is True
    assert SpeakerGateway.is_exit_command("退一下") is True
    assert SpeakerGateway.is_exit_command("好了退出") is True
    assert SpeakerGateway.is_exit_command("再见") is True
    assert SpeakerGateway.is_exit_command("我要退出对话了") is True
    assert SpeakerGateway.is_exit_command("帮我退出登录") is False


def test_does_not_suppress_own_tts_while_speaking() -> None:
    import asyncio, time
    from xiao_speaker_assistant.config import AppConfig

    sent = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, str):
                sent.append(json.loads(raw))

    async def dummy(q: str) -> str:
        return q

    cfg = AppConfig()
    cfg.speech.asr_provider = "xiaoai_asr"
    gw = SpeakerGateway(cfg, on_query=dummy)
    conn = FakeConn()
    gw._mute_native_until = time.monotonic() + 10.0
    pb = gw._get_playback(conn)
    line = json.dumps({"header": {"namespace": "SpeechSynthesizer", "name": "Speak"}})
    event = json.dumps({"Event": {"id": "evt-3", "event": "instruction", "data": {"NewLine": line}}})

    async def run() -> None:
        async with pb._lock:
            await gw._handle_text_message(conn, event)
            await asyncio.sleep(0.01)

    asyncio.run(run())
    assert sent == []


def test_ignores_short_filler_noise_in_local_utterance() -> None:
    import asyncio
    from xiao_speaker_assistant.config import AppConfig

    called = []

    async def fake_query(q: str) -> str:
        called.append(q)
        return "ok"

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw: str) -> None:
            pass

    gw = SpeakerGateway(AppConfig(), on_query=fake_query)

    async def run() -> None:
        await gw._process_local_utterance(FakeConn(), "嗯。")
        await gw._process_local_utterance(FakeConn(), "嗯嗯。")
        await gw._process_local_utterance(FakeConn(), "关家。")
        await gw._process_local_utterance(FakeConn(), "我有未读邮件吗？")

    asyncio.run(run())
    assert called == ["我有未读邮件吗？"]


def test_yields_to_native_xiaoai_without_killing_playback() -> None:
    import asyncio
    from unittest.mock import MagicMock

    from xiao_speaker_assistant.config import AppConfig

    sent = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, str):
                sent.append(json.loads(raw))

    async def dummy(q: str) -> str:
        return q

    reset_called = []
    gw = SpeakerGateway(AppConfig(), on_query=dummy, on_session_reset=lambda: reset_called.append(True))
    fake_pipeline = MagicMock()
    fake_pipeline.awake = True
    gw._pipeline = fake_pipeline

    # Non-XiaoAI RecognizeResult ("你好管家") or StopCapture must NOT yield/kill Nomi!
    non_xiaoai_line = json.dumps(
        {
            "header": {"namespace": "SpeechRecognizer", "name": "RecognizeResult"},
            "payload": {"results": [{"text": "你好管家"}]},
        }
    )
    non_xiaoai_evt = json.dumps(
        {"Event": {"id": "evt-4a", "event": "instruction", "data": {"NewLine": non_xiaoai_line}}}
    )
    # Explicit SpeechWakeup.Wakeup or "小爱..." DOES yield to native XiaoAI.
    xiaoai_line = json.dumps({"header": {"namespace": "SpeechWakeup", "name": "Wakeup"}})
    xiaoai_evt = json.dumps(
        {"Event": {"id": "evt-4b", "event": "instruction", "data": {"NewLine": xiaoai_line}}}
    )

    async def run() -> None:
        await gw._handle_text_message(FakeConn(), non_xiaoai_evt)
        assert reset_called == []
        fake_pipeline.set_awake.assert_not_called()

        await gw._handle_text_message(FakeConn(), xiaoai_evt)

    asyncio.run(run())
    fake_pipeline.set_awake.assert_called_once_with(False)
    fake_pipeline.mute_for.assert_called_once_with(6.0)
    assert reset_called == [True]
    assert sent == []


def test_start_recording_uses_larger_oh2p_alsa_buffer() -> None:
    import asyncio
    from xiao_speaker_assistant.config import AppConfig

    sent = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw: str) -> None:
            sent.append(json.loads(raw))

    async def dummy_query(q: str) -> str:
        return q

    gw = SpeakerGateway(AppConfig(), on_query=dummy_query)

    async def run() -> None:
        await gw.send_command(
            FakeConn(),
            "start_recording",
            {
                "pcm": "noop",
                "channels": 1,
                "bits_per_sample": 16,
                "sample_rate": 16000,
                "period_size": 360,
                "buffer_size": 1440,
            },
        )

    asyncio.run(run())
    assert sent[0]["Request"]["payload"]["buffer_size"] == 1440
    assert sent[0]["Request"]["payload"]["period_size"] == 360
