"""Local KWS (`Hi Nomi`) + Silero VAD + SenseVoice offline ASR pipeline."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Callable

import numpy as np

from xiao_speaker_assistant.config import SpeechConfig
from xiao_speaker_assistant.voice.models_setup import find_sense_voice_dir, generate_keywords_file

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512


def extract_record_pcm_from_binary(raw_bytes: bytes) -> bytes | None:
    """Decode open-xiaoai-client binary Stream JSON frame (`tag == 'record'`) to int16 PCM."""
    try:
        obj = json.loads(raw_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None
    if not isinstance(obj, dict) or obj.get("tag") != "record":
        return None
    data = obj.get("bytes")
    if not isinstance(data, list) or not data:
        return None
    return bytes(data)


class LocalSpeechPipeline:
    """Runs KWS (`Hi Nomi`) -> Silero VAD -> SenseVoice ASR on 16kHz int16 mono PCM."""

    def __init__(self, config: SpeechConfig) -> None:
        import sherpa_onnx

        self._config = config
        models_dir = Path(config.models_dir).expanduser()
        kw_file = generate_keywords_file(models_dir, config.local_keywords)

        self._kws = sherpa_onnx.KeywordSpotter(
            provider="cpu",
            num_threads=1,
            max_active_paths=8,
            keywords_score=config.kws_score,
            keywords_threshold=config.kws_threshold,
            num_trailing_blanks=0,
            keywords_file=str(kw_file),
            tokens=str(models_dir / "tokens.txt"),
            encoder=str(models_dir / "encoder.onnx"),
            decoder=str(models_dir / "decoder.onnx"),
            joiner=str(models_dir / "joiner.onnx"),
        )
        self._kws_stream = self._kws.create_stream()

        silero = sherpa_onnx.SileroVadModelConfig(
            model=str(models_dir / "silero_vad.onnx"),
            threshold=config.vad_threshold,
            min_silence_duration=config.vad_min_silence_ms / 1000.0,
            min_speech_duration=0.25,
            window_size=VAD_WINDOW_SAMPLES,
            max_speech_duration=20.0,
        )
        vad_cfg = sherpa_onnx.VadModelConfig(
            silero_vad=silero,
            sample_rate=SAMPLE_RATE,
            num_threads=1,
            provider="cpu",
        )
        self._vad = sherpa_onnx.VoiceActivityDetector(vad_cfg, buffer_size_in_seconds=30.0)

        sv_dir = find_sense_voice_dir(models_dir)
        self._asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
            model=str(sv_dir / "model.int8.onnx"),
            tokens=str(sv_dir / "tokens.txt"),
            num_threads=2,
            language="auto",
            use_itn=True,
            provider="cpu",
        )

        self.awake: bool = False
        self._last_active_at: float = 0.0
        self._muted_until: float = 0.0
        self._vad_buffer = np.zeros(0, dtype=np.float32)

    def mute_for(self, seconds: float) -> None:
        self._muted_until = max(self._muted_until, time.monotonic() + seconds)

    def set_awake(self, awake: bool) -> None:
        self.awake = awake
        self._last_active_at = time.monotonic() if awake else 0.0
        self._vad.reset()
        self._vad_buffer = np.zeros(0, dtype=np.float32)
        self._kws.reset_stream(self._kws_stream)

    def feed_pcm(
        self,
        pcm_bytes: bytes,
        *,
        on_wakeup: Callable[[str], None] | None = None,
        on_utterance: Callable[[str], None] | None = None,
        on_sleep: Callable[[], None] | None = None,
    ) -> None:
        now = time.monotonic()
        if now < self._muted_until:
            return

        if self.awake and (now - self._last_active_at) > self._config.conversation_timeout_sec:
            logger.info("Local conversation timed out; returning to KWS idle")
            self.set_awake(False)
            if on_sleep:
                on_sleep()
            return

        samples = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        if self._config.input_gain != 1.0:
            samples = np.clip(samples * self._config.input_gain, -1.0, 1.0)

        if not self.awake:
            self._kws_stream.accept_waveform(SAMPLE_RATE, samples)
            while self._kws.is_ready(self._kws_stream):
                self._kws.decode_stream(self._kws_stream)
                keyword = self._kws.get_result(self._kws_stream)
                if keyword:
                    self.set_awake(True)
                    logger.info("KWS triggered: %s", keyword)
                    if on_wakeup:
                        on_wakeup(keyword)
                    return
            return

        # Awake: feed VAD in 512-sample windows and transcribe completed segments with SenseVoice.
        self._vad_buffer = np.concatenate([self._vad_buffer, samples])
        while len(self._vad_buffer) >= VAD_WINDOW_SAMPLES:
            chunk = self._vad_buffer[:VAD_WINDOW_SAMPLES]
            self._vad_buffer = self._vad_buffer[VAD_WINDOW_SAMPLES:]
            self._vad.accept_waveform(chunk)
            if self._vad.is_speech_detected():
                self._last_active_at = time.monotonic()

        while not self._vad.empty():
            segment = self._vad.front.samples
            self._vad.pop()
            self._last_active_at = time.monotonic()
            text = self._transcribe(segment)
            if text and on_utterance:
                on_utterance(text)

    def _transcribe(self, samples: list[float] | np.ndarray) -> str:
        stream = self._asr.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._asr.decode_stream(stream)
        return stream.result.text.strip()
