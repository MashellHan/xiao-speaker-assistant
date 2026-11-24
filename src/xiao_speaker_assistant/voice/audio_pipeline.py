"""Local KWS + Silero VAD + SenseVoice offline ASR pipeline (supports `Hi Nomi`)."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from xiao_speaker_assistant.config import SpeechConfig
from xiao_speaker_assistant.voice.models_setup import (
    find_paraformer_dir,
    find_sense_voice_dir,
    generate_keywords_file,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512

HI_NOMI_REGEX = re.compile(
    r"^\s*(?:hi|hey|hello|can|h|i|嗨|哈喽|你好)?\s*[,，.。!！?\s]*"
    r"(?:nomi|normi|nami|hnomi|know\s*mi|know\s*me|no\s*mi|no\s*me|naomi|noumi|诺米|糯米|闹米)"
    r"(?:[,，.。!！?\s:]*(.*))?$",
    re.IGNORECASE | re.DOTALL,
)


def extract_record_pcm_from_binary(raw_bytes: bytes) -> bytes | None:
    """Decode open-xiaoai-client binary Stream JSON frame (`tag == 'record'`) to PCM bytes."""
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


def pcm_to_float32(pcm_bytes: bytes, bits_per_sample: int = 32) -> np.ndarray:
    """Convert raw PCM bytes to normalized [-1, 1] float32 samples.

    OH2P (Amlogic A113) PDM capture puts 24-bit audio in the lower 24 bits of S32_LE,
    so shifting right by 8 recovers full-scale 16-bit amplitude.
    """
    if bits_per_sample == 32:
        usable = len(pcm_bytes) - (len(pcm_bytes) % 4)
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        s32 = np.frombuffer(pcm_bytes[:usable], dtype="<i4")
        s16 = np.clip(s32 >> 8, -32768, 32767)
        return s16.astype(np.float32) / 32768.0
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    s16 = np.frombuffer(pcm_bytes[:usable], dtype="<i2")
    return s16.astype(np.float32) / 32768.0


def match_wakeup_phrase(
    text: str, extra_keywords: list[str] | None = None
) -> tuple[str, str] | None:
    """Match `Hi Nomi` (with phonetic variants) or configured wake keywords.

    Returns (wake_label, remaining_query) or None if not matched.
    """
    stripped = text.strip()
    if not stripped:
        return None

    m = HI_NOMI_REGEX.match(stripped)
    if m:
        rest = (m.group(1) or "").strip(" ，,。.!！？:：")
        return "hi nomi", rest

    lowered = stripped.lower()
    for kw in extra_keywords or []:
        clean_kw = kw.strip().lower()
        if not clean_kw:
            continue
        if lowered.startswith(clean_kw):
            rest = stripped[len(clean_kw) :].strip(" ，,。.!！？:：")
            return clean_kw, rest
    return None


class LocalSpeechPipeline:
    """Runs KWS + Silero VAD + SenseVoice ASR on 16kHz mono PCM."""

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
            min_speech_duration=0.1,
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

        if config.asr_model == "paraformer":
            pf_dir = find_paraformer_dir(models_dir)
            self._asr = sherpa_onnx.OfflineRecognizer.from_paraformer(
                paraformer=str(pf_dir / "model.int8.onnx"),
                tokens=str(pf_dir / "tokens.txt"),
                num_threads=2,
                provider="cpu",
            )
        else:
            sv_dir = find_sense_voice_dir(models_dir)
            self._asr = sherpa_onnx.OfflineRecognizer.from_sense_voice(
                model=str(sv_dir / "model.int8.onnx"),
                tokens=str(sv_dir / "tokens.txt"),
                num_threads=2,
                language=config.asr_language,
                use_itn=True,
                provider="cpu",
            )

        self.awake: bool = False
        self._last_active_at: float = 0.0
        self._muted_until: float = 0.0
        self._explicit_muted: bool = False
        self._vad_buffer = np.zeros(0, dtype=np.float32)
        self._logged_first_audio = False

    def set_muted(self, muted: bool) -> None:
        self._explicit_muted = muted
        if not muted:
            self._vad.reset()
            self._vad_buffer = np.zeros(0, dtype=np.float32)

    def mute_for(self, seconds: float) -> None:
        self._muted_until = max(self._muted_until, time.monotonic() + seconds)

    def refresh_activity(self) -> None:
        if self.awake:
            self._last_active_at = time.monotonic()

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
        if self._explicit_muted or now < self._muted_until:
            return

        if self.awake and (now - self._last_active_at) > self._config.conversation_timeout_sec:
            logger.info("Local conversation timed out; returning to idle")
            self.set_awake(False)
            if on_sleep:
                on_sleep()
            return

        samples = pcm_to_float32(pcm_bytes, bits_per_sample=self._config.capture_bits_per_sample)
        if len(samples) == 0:
            return
        if len(samples) >= 160:
            samples = samples - np.mean(samples)
        if self._config.input_gain != 1.0:
            samples = np.clip(samples * self._config.input_gain, -1.0, 1.0)
        if not self._logged_first_audio:
            self._logged_first_audio = True
            rms = float(np.sqrt(np.mean(samples * samples)))
            peak = float(np.max(np.abs(samples)))
            logger.info(
                "First audio frame received (rms=%.4f, peak=%.4f, bits=%d)",
                rms,
                peak,
                self._config.capture_bits_per_sample,
            )

        # Fast Zipformer KWS path when idle (instant trigger for Chinese wake words like 你好管家).
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

        # Feed Silero VAD continuously so both `Hi Nomi` wakeup and full utterances work smoothly.
        self._vad_buffer = np.concatenate([self._vad_buffer, samples])
        while len(self._vad_buffer) >= VAD_WINDOW_SAMPLES:
            chunk = self._vad_buffer[:VAD_WINDOW_SAMPLES]
            self._vad_buffer = self._vad_buffer[VAD_WINDOW_SAMPLES:]
            self._vad.accept_waveform(chunk)
            if self.awake and self._vad.is_speech_detected():
                self._last_active_at = time.monotonic()

        while not self._vad.empty():
            segment = self._vad.front.samples
            self._vad.pop()
            text = self._transcribe(segment)
            if not text:
                continue

            if not self.awake:
                matched = match_wakeup_phrase(text, self._config.local_keywords)
                if not matched:
                    logger.info("Idle speech (ignored): %s", text)
                    continue
                wake_label, rest = matched
                logger.info("SenseVoice wake matched (%s): %s", wake_label, text)
                self.set_awake(True)
                if rest:
                    if on_utterance:
                        on_utterance(rest)
                elif on_wakeup:
                    on_wakeup(wake_label)
            else:
                self._last_active_at = time.monotonic()
                if on_utterance:
                    on_utterance(text)

    def _transcribe(self, samples: list[float] | np.ndarray) -> str:
        stream = self._asr.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._asr.decode_stream(stream)
        return stream.result.text.strip()
