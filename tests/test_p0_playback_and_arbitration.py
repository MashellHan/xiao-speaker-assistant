import asyncio
import json
from unittest.mock import MagicMock, patch

from xiao_speaker_assistant.config import AppConfig, SpeakerConfig
from xiao_speaker_assistant.voice.audio_pipeline import match_wakeup_phrase
from xiao_speaker_assistant.voice.playback import PlaybackController
from xiao_speaker_assistant.voice.tts_stream import generate_chime_pcm
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_match_wakeup_phrase_tolerates_punctuation_in_chinese_keywords() -> None:
    assert match_wakeup_phrase("你好，管家。", ["你好管家"]) == ("你好管家", "")
    assert match_wakeup_phrase("你好，管家！帮我查一下", ["你好管家"]) == ("你好管家", "帮我查一下")
    assert match_wakeup_phrase("你好，小助手，今天星期几", ["你好小助手"]) == ("你好小助手", "今天星期几")


def test_generate_chime_pcm_produces_16bit_mono_audio() -> None:
    pcm = generate_chime_pcm(sample_rate=24000)
    assert len(pcm) == 24000 * 0.15 * 2
    assert any(b != 0 for b in pcm)


def test_playback_controller_streams_pcm_frames_via_start_play() -> None:
    sent_text: list[dict] = []
    sent_binary: list[dict] = []
    states: list[bool] = []

    async def fake_send_text(t: str) -> None:
        sent_text.append(json.loads(t))

    async def fake_send_binary(b: bytes) -> None:
        sent_binary.append(json.loads(b.decode("utf-8")))

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def fake_stream(text: str, cfg: SpeakerConfig):
        yield b"\x01\x00" * 1200

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        on_playback_state=states.append,
    )

    async def run() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_stream):
            await pb.speak("你好世界", interrupt_native=True)

    asyncio.run(run())
    assert sent_text[0]["Request"]["command"] == "start_play"
    assert sent_binary[0]["tag"] == "play"
    assert len(sent_binary[0]["bytes"]) == 2400
    assert states == [True, False]


def test_sense_voice_mode_suppresses_native_speak_while_awake() -> None:
    sent: list[dict] = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, str):
                sent.append(json.loads(raw))

    async def dummy(q: str) -> str:
        return q

    gw = SpeakerGateway(AppConfig(), on_query=dummy)
    fake_pipeline = MagicMock()
    fake_pipeline.awake = True
    gw._pipeline = fake_pipeline

    speak_line = json.dumps({"header": {"namespace": "SpeechSynthesizer", "name": "Speak"}})
    evt = json.dumps({"Event": {"id": "evt-s1", "event": "instruction", "data": {"NewLine": speak_line}}})

    async def run() -> None:
        await gw._handle_text_message(FakeConn(), evt)
        await asyncio.sleep(0.02)

    asyncio.run(run())
    assert len(sent) == 1
    assert "player_wakeup" in sent[0]["Request"]["payload"]
    assert "killall miplayer" in sent[0]["Request"]["payload"]
