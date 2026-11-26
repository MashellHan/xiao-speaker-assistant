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
    assert match_wakeup_phrase("Hno米，今天北京天气", ["你好管家"]) == ("hi nomi", "今天北京天气")
    assert match_wakeup_phrase("你好，老米查一下磁盘", ["你好管家"]) == ("hi nomi", "查一下磁盘")


def test_sense_voice_transcribe_filters_hangul_and_kana_hallucinations() -> None:
    from xiao_speaker_assistant.voice.audio_pipeline import LocalSpeechPipeline

    fake_pipe = LocalSpeechPipeline.__new__(LocalSpeechPipeline)
    fake_pipe._cloud_asr = None

    class FakeResult:
        def __init__(self, text: str) -> None:
            self.text = text

    class FakeStream:
        def __init__(self, text: str) -> None:
            self.result = FakeResult(text)

        def accept_waveform(self, sr: int, samples) -> None:
            pass

    class FakeASR:
        def __init__(self, text: str) -> None:
            self._text = text

        def create_stream(self):
            return FakeStream(self._text)

        def decode_stream(self, stream) -> None:
            pass

    for bad in ("안녕하세요", "ありがとう", "はい。"):
        fake_pipe._asr = FakeASR(bad)
        assert fake_pipe._transcribe([0.1, 0.2]) == ""

    fake_pipe._asr = FakeASR("Hi Nomi，今天星期三。")
    assert fake_pipe._transcribe([0.1, 0.2]) == "Hi Nomi，今天星期三。"


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
    # Aborted speak() must NOT call on_playback_state(False), which would re-mute/wipe VAD during barge-in.
    assert states == [True]
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


def test_playback_controller_wan_relay_uses_jitter_buffer_and_continuous_stream() -> None:
    rpcs: list[tuple[str, object]] = []
    sent_binary_raw: list[bytes] = []
    states: list[bool] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        sent_binary_raw.append(b)

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def fake_send_rpc(cmd: str, payload, wait: bool, timeout: float):
        rpcs.append((cmd, payload))
        return {"code": 0}

    async def fake_stream(text: str, cfg: SpeakerConfig, **kwargs):
        assert kwargs.get("chunk_bytes") == 6400
        assert kwargs.get("sample_rate_override") == 16000
        yield b"\x02\x00" * 3200

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=fake_send_rpc,
        on_playback_state=states.append,
        wan_relay=True,
    )

    async def run() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_stream):
            await pb.speak("第一句。", interrupt_native=True, keep_stream_open=True)
            # Between back-to-back sentences while ALSA stream is still open, is_playing stays True
            assert pb.stream_open is True
            assert pb.is_playing is True
            await pb.speak("第二句。", interrupt_native=False, keep_stream_open=True)
            await pb.finish_stream(pad_tail_sec=0.05)

    asyncio.run(run())
    start_rpcs = [p for cmd, p in rpcs if cmd == "start_play"]
    stop_rpcs = [p for cmd, p in rpcs if cmd == "stop_play"]
    # Continuous multi-sentence stream opens ALSA `aplay` ONCE and closes ONCE at the end.
    assert len(start_rpcs) == 1
    assert start_rpcs[0] == {
        "pcm": "default",
        "channels": 1,
        "bits_per_sample": 16,
        "sample_rate": 16000,
        "period_size": 1600,
        "buffer_size": 8000,
    }
    assert len(stop_rpcs) == 1
    # Compact JSON encoding omits spaces after commas/colons (~22% smaller over wire)
    assert b", " not in sent_binary_raw[0]
    # Balanced 1-to-1 playback state transition across multi-sentence stream
    assert states == [True, False]
    assert pb.is_playing is False


def test_playback_controller_wan_short_utterance_exceeds_alsa_start_threshold() -> None:
    sent_binary: list[dict] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        sent_binary.append(json.loads(b.decode("utf-8")))

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def fake_send_rpc(cmd: str, payload, wait: bool, timeout: float):
        return {"code": 0}

    async def short_stream(text: str, cfg: SpeakerConfig, **kwargs):
        # 200ms short reply ("好的") = 9600 bytes at 24kHz S16_LE
        yield b"\x05\x00" * 4800

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=fake_send_rpc,
        wan_relay=True,
    )

    async def run() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", short_stream):
            await pb.speak("好的", interrupt_native=True)

    asyncio.run(run())
    # Must exceed ALSA 8000-frame (16000-byte) start_threshold at 16kHz WAN so short utterances never drop!
    total_bytes = sum(len(frame["bytes"]) for frame in sent_binary)
    assert total_bytes >= 8000 * 2


def test_playback_controller_keep_open_failure_still_unmutes_vad() -> None:
    states: list[bool] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        pass

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def failing_rpc(cmd: str, payload, wait: bool, timeout: float):
        # Simulate start_play RPC timeout -> returns None
        return None

    async def fake_stream(text: str, cfg: SpeakerConfig, **kwargs):
        yield b"\x01\x00" * 4800

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=failing_rpc,
        on_playback_state=states.append,
        wan_relay=True,
    )

    async def run() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_stream):
            await pb.speak("第一句", interrupt_native=True, keep_stream_open=True)

    asyncio.run(run())
    assert pb.stream_open is False
    # 1ch VAD MUST be unmuted cleanly when start_play fails under keep_stream_open=True
    assert states == [True, False]


def test_stop_playback_does_not_block_on_missing_stop_play_ack() -> None:
    import time

    rpcs: list[tuple[str, bool]] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        pass

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def slow_if_waiting_rpc(cmd: str, payload, wait: bool, timeout: float):
        rpcs.append((cmd, wait))
        if cmd == "stop_play" and wait:
            await asyncio.sleep(timeout)
        return {"code": 0}

    pb = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=slow_if_waiting_rpc,
        wan_relay=True,
    )

    async def run() -> None:
        assert await pb.ensure_stream_started()
        t0 = time.monotonic()
        await pb.stop_playback(stop_native=True)
        assert (time.monotonic() - t0) < 0.3

    asyncio.run(run())
    assert ("stop_play", False) in rpcs


def test_pipelined_tts_stream_closes_cleanly_on_start_play_failure_and_barge_in() -> None:
    import time

    closed_count = 0
    sent_binary: list[dict] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        sent_binary.append(json.loads(b.decode("utf-8")))

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    # Case 1: start_play fails (returns None) -> tts_stream MUST be closed via aclose()
    async def failing_rpc(cmd: str, payload, wait: bool, timeout: float):
        return None

    async def tracked_stream(text: str, cfg: SpeakerConfig, **kwargs):
        nonlocal closed_count
        try:
            yield b""
            yield b"\x04\x00" * 4800
        finally:
            closed_count += 1

    pb1 = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=failing_rpc,
        wan_relay=True,
    )

    async def run_fail() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", tracked_stream):
            await pb1.speak("测试关闭", interrupt_native=True)

    asyncio.run(run_fail())
    assert closed_count == 1
    assert sent_binary == []

    # Case 2: stop_playback() while anext(tts_stream) is blocked on slow Edge-TTS unblocks immediately
    slow_started = asyncio.Event()

    async def ok_rpc(cmd: str, payload, wait: bool, timeout: float):
        return {"code": 0}

    async def slow_tts_stream(text: str, cfg: SpeakerConfig, **kwargs):
        nonlocal closed_count
        slow_started.set()
        try:
            await asyncio.sleep(10.0)
            yield b"\x01\x00" * 4800
        finally:
            closed_count += 1

    pb2 = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=ok_rpc,
        wan_relay=True,
    )

    async def run_barge_in() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", slow_tts_stream):
            speak_task = asyncio.create_task(pb2.speak("慢速TTS", interrupt_native=True))
            await slow_started.wait()
            await asyncio.sleep(0.02)
            t0 = time.monotonic()
            await pb2.stop_playback(stop_native=True)
            await asyncio.wait_for(speak_task, timeout=1.0)
            assert (time.monotonic() - t0) < 0.4

    asyncio.run(run_barge_in())
    assert closed_count == 2

    # Case 3: leading empty b"" chunk followed by real PCM still streams cleanly
    pb3 = PlaybackController(
        SpeakerConfig(tts_provider="edge"),
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=ok_rpc,
        wan_relay=True,
    )

    async def run_empty_first() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", tracked_stream):
            await pb3.speak("空首块测试", interrupt_native=True)

    asyncio.run(run_empty_first())
    assert sum(len(f["bytes"]) for f in sent_binary) >= 8000 * 2
    assert closed_count == 3


def test_streaming_reply_prefetches_subsequent_sentences_in_parallel() -> None:
    import time

    sent_binary: list[dict] = []
    synth_starts: dict[str, float] = {}

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                sent_binary.append(json.loads(raw.decode("utf-8")))
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda r=rid: asyncio.create_task(
                        gw._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": r, "code": 0}}),
                        )
                    )
                )

    async def fake_tts(text: str, cfg: SpeakerConfig, **kwargs):
        synth_starts[text] = time.monotonic()
        await asyncio.sleep(0.15)
        yield b"\x02\x00" * 3200

    async def two_sentences(query: str):
        yield "第一句比较长的话用来测试并行预合成效果。"
        yield "第二句应该在第一句合成或播放时就提前开始合成。"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "edge"
    gw = SpeakerGateway(cfg, on_stream_query=two_sentences)

    async def run() -> None:
        conn = FakeConn()
        gw._relay_connections.add(conn)
        with (
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw._run_query_and_speak(conn, "测试预合成")

    asyncio.run(run())
    assert len(synth_starts) == 2
    t1 = synth_starts["第一句比较长的话用来测试并行预合成效果。"]
    t2 = synth_starts["第二句应该在第一句合成或播放时就提前开始合成。"]
    # Second sentence must start synthesizing BEFORE the first sentence finishes its 0.15s TTS!
    assert abs(t2 - t1) < 0.10
    assert len(sent_binary) >= 2


def test_barge_in_during_continuous_stream_does_not_drain_or_close_new_chime() -> None:
    import time

    sent_rpcs: list[str] = []
    states: list[bool] = []
    first_done = asyncio.Event()

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            if cmd:
                sent_rpcs.append(cmd)
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda r=rid: asyncio.create_task(
                        gw._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": r, "code": 0}}),
                        )
                    )
                )

    async def fake_tts(text: str, cfg: SpeakerConfig, **kwargs):
        yield b"\x03\x00" * 4800

    async def multi_sentence_stream(query: str):
        yield "第一句话正在播报。"
        first_done.set()
        await asyncio.sleep(10.0)
        yield "不该播出的第二句话。"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "edge"
    cfg.speech.aec_enabled = False
    cfg.speech.capture_channels = 1
    gw = SpeakerGateway(cfg, on_stream_query=multi_sentence_stream)

    async def run() -> None:
        conn = FakeConn()
        gw._relay_connections.add(conn)
        pb = gw._get_playback(conn)
        pb._on_playback_state = states.append
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts):
            reply_task = asyncio.create_task(gw._run_query_and_speak(conn, "讲个故事"))
            await first_done.wait()
            await asyncio.sleep(0.05)
            t0 = time.monotonic()
            await gw._handle_interrupt(conn)
            await asyncio.wait_for(reply_task, timeout=1.0)
            elapsed = time.monotonic() - t0
            assert elapsed < 0.6

    asyncio.run(run())
    # Aborted continuous stream must NOT emit on_playback_state(False) or hang draining tail
    assert states == [True]
    assert sent_rpcs.count("stop_play") == 1


def test_empty_prefetched_pcm_falls_back_to_live_tts_and_chime_tracks_sample_rate() -> None:
    sent_binary: list[dict] = []
    live_calls: list[str] = []

    async def fake_send_text(t: str) -> None:
        pass

    async def fake_send_binary(b: bytes) -> None:
        sent_binary.append(json.loads(b.decode("utf-8")))

    async def fake_run_shell(script: str, wait: bool, timeout: float):
        return {"code": 0}

    async def ok_rpc(cmd: str, payload, wait: bool, timeout: float):
        return {"code": 0}

    async def fake_tts(text: str, cfg: SpeakerConfig, **kwargs):
        live_calls.append(text)
        yield b"\x05\x00" * 3200

    cfg = SpeakerConfig(tts_provider="edge", tts_sample_rate=24000)
    pb = PlaybackController(
        cfg,
        send_text=fake_send_text,
        send_binary=fake_send_binary,
        run_shell=fake_run_shell,
        send_rpc=ok_rpc,
        wan_relay=False,
    )

    async def run() -> None:
        with patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts):
            await pb.speak("空预取回退测试", interrupt_native=False, prefetched_pcm=b"")
        assert pb._chime_sample_rate == 24000
        old_len = len(pb._chime_pcm)
        pb._config = SpeakerConfig(tts_provider="edge", tts_sample_rate=16000)
        await pb.play_chime()
        assert pb._chime_sample_rate == 16000
        assert len(pb._chime_pcm) < old_len

    asyncio.run(run())
    assert live_calls == ["空预取回退测试"]
    assert len(sent_binary) >= 2


def test_conversational_ack_speaks_immediately_for_slow_weather_and_skips_fast_reply() -> None:
    from xiao_speaker_assistant.voice.ws_server import build_conversational_ack

    phrase, delay = build_conversational_ack("北京明天天气怎么样")
    assert phrase == "好的，我这就帮你查一下北京天气。"
    assert delay is not None and delay <= 1.5

    phrase_b, _ = build_conversational_ack("帮我查一下杭州明天天气怎么样")
    assert phrase_b == "好的，我这就帮你查一下杭州天气。"

    # Edge cases: no garbled city names, indoor AC temp doesn't trigger weather, case-insensitive Outlook/Bug
    assert build_conversational_ack("天气怎么样")[0] == "好的，我这就帮你查一下天气。"
    assert build_conversational_ack("明天不会下雨吧")[0] == "好的，我这就帮你查一下天气。"
    assert build_conversational_ack("北京明天会下雨吗")[0] == "好的，我这就帮你查一下北京天气。"
    assert build_conversational_ack("空调温度调到26度")[0] == "稍等一下，我这就帮你查一下。"
    assert build_conversational_ack("帮我看看Outlook")[0] == "好的，我这就帮你查一下邮件。"
    assert build_conversational_ack("帮我查一下小红书上海美食")[0] == "好的，我这就去小红书帮你搜一下。"
    assert build_conversational_ack("修复一个Bug")[0] == "好的，我这就帮你分析一下。"
    assert build_conversational_ack("确认发送") == (None, None)
    assert build_conversational_ack("你好") == (None, None)
    assert build_conversational_ack("帮我取消明天的闹钟")[0] is not None

    spoken_texts: list[str] = []
    sent_rpcs: list[str] = []

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        def __init__(self, target_gw: SpeakerGateway) -> None:
            self._gw = target_gw

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            if cmd:
                sent_rpcs.append(cmd)
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda r=rid, g=self._gw: asyncio.create_task(
                        g._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": r, "code": 0}}),
                        )
                    )
                )

    async def fake_tts(text: str, cfg: SpeakerConfig, **kwargs):
        yield b"\x02\x00" * 3200

    # Slow weather tool query -> speaks ack first, then multi-sentence reply, without re-running run_shell
    async def slow_weather_stream(query: str):
        await asyncio.sleep(0.55)
        yield "上海明天晴，18到24度。"
        yield "适合穿长袖衬衫出门。"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "edge"
    cfg.speaker.tool_ack_enabled = True
    cfg.speaker.tool_ack_delay_sec = 0.35
    gw = SpeakerGateway(cfg, on_stream_query=slow_weather_stream)

    async def run_slow() -> None:
        conn = FakeConn(gw)
        gw._relay_connections.add(conn)
        with (
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw._run_query_and_speak(conn, "上海明天天气怎么样")
        spoken_texts.extend(t for _, t in gw._get_session(conn).recent_spoken_texts)

    asyncio.run(run_slow())
    assert spoken_texts == [
        "好的，我这就帮你查一下上海天气。",
        "上海明天晴，18到24度。",
        "适合穿长袖衬衫出门。",
    ]
    # STOP_NATIVE_SCRIPT run_shell is sent once for ack, NOT repeated on the follow-up reply stream.
    assert sent_rpcs.count("run_shell") == 1
    assert sent_rpcs.count("stop_play") == 2

    # Fast reply (<0.1s) -> ack must NOT fire
    spoken_texts.clear()
    sent_rpcs.clear()

    async def fast_stream(query: str):
        await asyncio.sleep(0.02)
        yield "你好呀，我在呢。"

    gw_fast = SpeakerGateway(cfg, on_stream_query=fast_stream)

    async def run_fast() -> None:
        conn = FakeConn(gw_fast)
        gw_fast._relay_connections.add(conn)
        with (
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw_fast._run_query_and_speak(conn, "上海天气怎么样")
        spoken_texts.extend(t for _, t in gw_fast._get_session(conn).recent_spoken_texts)

    asyncio.run(run_fast())
    assert spoken_texts == ["你好呀，我在呢。"]

    # Disabled tool_ack_enabled -> never speaks ack even on slow query
    spoken_texts.clear()
    cfg_off = AppConfig()
    cfg_off.speaker.tts_provider = "edge"
    cfg_off.speaker.tool_ack_enabled = False
    gw_off = SpeakerGateway(cfg_off, on_stream_query=slow_weather_stream)

    async def run_off() -> None:
        conn = FakeConn(gw_off)
        gw_off._relay_connections.add(conn)
        with (
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw_off._run_query_and_speak(conn, "上海明天天气怎么样")
        spoken_texts.extend(t for _, t in gw_off._get_session(conn).recent_spoken_texts)

    asyncio.run(run_off())
    assert spoken_texts == ["上海明天晴，18到24度。", "适合穿长袖衬衫出门。"]


def test_prewarmed_edge_tts_and_speculative_start_play_overlap_with_llm() -> None:
    import time
    from xiao_speaker_assistant.voice.tts_stream import PrewarmedEdgeTTS

    events: list[tuple[str, float]] = []
    prewarm_closed = False

    class FakePrewarmed(PrewarmedEdgeTTS):
        def __init__(self) -> None:
            super().__init__(object(), type("W", (), {"closed": False})(), created_at=time.monotonic())

        async def close(self) -> None:
            nonlocal prewarm_closed
            prewarm_closed = True
            self._closed = True

    fake_prewarm = FakePrewarmed()

    async def fake_open(*, connect_timeout: float = 8.0):
        events.append(("prewarm_open", time.monotonic()))
        return fake_prewarm

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        def __init__(self, target_gw: SpeakerGateway) -> None:
            self._gw = target_gw

        async def send(self, raw) -> None:
            if isinstance(raw, (bytes, bytearray)):
                return
            msg = json.loads(raw)
            req = msg.get("Request") or {}
            cmd = str(req.get("command") or "")
            if cmd == "start_play":
                events.append(("start_play", time.monotonic()))
            if cmd in {"start_play", "stop_play"} and isinstance(req.get("id"), str):
                rid = req["id"]
                asyncio.get_running_loop().call_soon(
                    lambda r=rid, g=self._gw: asyncio.create_task(
                        g._handle_text_message(
                            self,
                            json.dumps({"Response": {"id": r, "code": 0}}),
                        )
                    )
                )

    received_prewarm: list[object] = []

    async def fake_tts(text: str, cfg: SpeakerConfig, **kwargs):
        received_prewarm.append(kwargs.get("prewarmed"))
        yield b"\x02\x00" * 3200

    async def slow_llm(query: str):
        await asyncio.sleep(0.15)
        events.append(("llm_first", time.monotonic()))
        yield "我在呢大熊，有什么可以帮你的？"

    cfg = AppConfig()
    cfg.speaker.tts_provider = "edge"
    cfg.speaker.tool_ack_enabled = False
    gw = SpeakerGateway(cfg, on_stream_query=slow_llm)

    async def run() -> None:
        conn = FakeConn(gw)
        gw._relay_connections.add(conn)
        with (
            patch.object(PrewarmedEdgeTTS, "open", side_effect=fake_open),
            patch("xiao_speaker_assistant.voice.playback.synthesize_tts_pcm_stream", fake_tts),
            patch("xiao_speaker_assistant.voice.ws_server.synthesize_tts_pcm_stream", fake_tts),
        ):
            await gw._run_query_and_speak(conn, "你在吗")

    asyncio.run(run())
    ev_map = dict(events)
    assert "prewarm_open" in ev_map and "start_play" in ev_map and "llm_first" in ev_map
    # Both Edge-TTS WS prewarm and speculative start_play MUST happen BEFORE LLM emits first sentence!
    assert ev_map["prewarm_open"] < ev_map["llm_first"]
    assert ev_map["start_play"] < ev_map["llm_first"]
    assert received_prewarm == [fake_prewarm]
    assert prewarm_closed is True


def test_prewarmed_edge_tts_stream_mp3_escapes_xml_and_parses_audio_frames() -> None:
    import aiohttp
    import time
    from xiao_speaker_assistant.voice.tts_stream import PrewarmedEdgeTTS

    sent_payloads: list[str] = []
    ws_closed = False
    sess_closed = False

    class FakeMsg:
        def __init__(self, msg_type, data):
            self.type = msg_type
            self.data = data

    audio_hdr = b"X-RequestId:1\r\nContent-Type:audio/mpeg\r\nPath:audio\r\n\r\n"
    audio_frame = len(audio_hdr).to_bytes(2, "big") + audio_hdr + b"\xff\xfb\x90\x00MP3DATA"
    non_audio_hdr = b"Path:audio.metadata\r\n\r\n"
    meta_frame = len(non_audio_hdr).to_bytes(2, "big") + non_audio_hdr + b"{}"

    class FakeWS:
        closed = False

        async def send_str(self, s: str) -> None:
            sent_payloads.append(s)

        async def close(self) -> None:
            nonlocal ws_closed
            ws_closed = True
            self.closed = True

        def __aiter__(self):
            async def _gen():
                yield FakeMsg(aiohttp.WSMsgType.BINARY, meta_frame)
                yield FakeMsg(aiohttp.WSMsgType.BINARY, audio_frame)
                yield FakeMsg(aiohttp.WSMsgType.TEXT, "Path:turn.end\r\n\r\n")
            return _gen()

    class FakeSession:
        async def close(self) -> None:
            nonlocal sess_closed
            sess_closed = True

    pre = PrewarmedEdgeTTS(FakeSession(), FakeWS(), created_at=time.monotonic())

    async def run() -> list[bytes]:
        assert pre.claim() is True
        assert pre.claim() is False  # cannot claim twice
        return [chunk async for chunk in pre.stream_mp3("A & B < C > D", voice="zh-CN-XiaoyiNeural", rate="+8%")]

    chunks = asyncio.run(run())
    assert chunks == [b"\xff\xfb\x90\x00MP3DATA"]
    assert len(sent_payloads) == 1
    assert "A &amp; B &lt; C &gt; D" in sent_payloads[0]
    assert ws_closed is True
    assert sess_closed is True
