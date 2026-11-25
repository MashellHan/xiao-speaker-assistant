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


def test_playback_controller_cancel_epoch_aborts_queued_waiters_and_chime_skips_mute() -> None:
    sent_text: list[dict] = []
    sent_binary: list[dict] = []
    run_shells: list[str] = []
    states: list[bool] = []
    spoken_texts: list[str] = []
    first_started = asyncio.Event()

    async def fake_send_text(t: str) -> None:
        sent_text.append(json.loads(t))

    async def fake_send_binary(b: bytes) -> None:
        sent_binary.append(json.loads(b.decode("utf-8")))

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        run_shells.append(script)
        return {"code": 0}

    async def slow_stream(text: str, cfg: SpeakerConfig):
        spoken_texts.append(text)
        first_started.set()
        await asyncio.sleep(0.2)
        yield b"\x01\x00" * 1200

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        on_playback_state=states.append,
    )

    async def run() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", slow_stream):
            await pb.play_chime()
            # Chime must NOT call on_playback_state (which would wipe VAD during one-shot wakeup)
            assert states == []

            t1 = asyncio.create_task(pb.speak("第一句", interrupt_native=True))
            await first_started.wait()
            t2 = asyncio.create_task(pb.speak("不该播放的第二句", interrupt_native=False))
            await asyncio.sleep(0.01)
            await pb.stop_playback(stop_native=True)
            await asyncio.gather(t1, t2)

    asyncio.run(run())
    assert spoken_texts == ["第一句"]
    assert any(m["Request"]["command"] == "stop_play" for m in sent_text)
    assert any("player_wakeup" in s for s in run_shells)


def test_playback_controller_awaits_start_play_ack_and_pads_trailing_silence() -> None:
    events: list[str] = []
    sent_binary: list[dict] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        sent_binary.append(json.loads(b.decode("utf-8")))
        events.append("binary")

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def fake_send_rpc(cmd: str, payload, wait: bool, timeout: float):
        events.append(f"rpc:{cmd}:wait={wait}:buf={payload.get('buffer_size') if isinstance(payload, dict) else None}")
        await asyncio.sleep(0.01)
        return {"code": 0}

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=fake_send_rpc,
    )

    async def run() -> None:
        await pb.play_chime()

    asyncio.run(run())
    assert events[0] == "rpc:start_play:wait=True:buf=4800"
    assert events[1] == "binary"
    # Total PCM bytes sent must exceed ALSA 4800-frame (9600-byte) buffer_size so chime plays immediately.
    total_bytes = sum(len(frame["bytes"]) for frame in sent_binary)
    assert total_bytes >= 15000
    assert events[-1].startswith("rpc:stop_play")
