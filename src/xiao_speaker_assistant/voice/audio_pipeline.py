"""Local KWS + Silero VAD + SenseVoice offline ASR pipeline (supports `Hi Nomi`)."""

from __future__ import annotations

import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

import numpy as np

from pypinyin import lazy_pinyin

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

_PINYIN_FIRST = {"hai", "cai", "kai", "ai", "tai", "sai", "hei", "ha", "a", "chen", "han", "an"}
_PINYIN_SECOND = {"nuo", "na", "luo", "no", "nao", "lao"}
_PINYIN_THIRD = {"mi", "min", "mei", "ming"}


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


def pcm_to_float32(pcm_bytes: bytes, bits_per_sample: int = 16, s32_shift: int = 16) -> np.ndarray:
    """Convert raw PCM bytes to normalized [-1, 1] float32 samples.

    OH2P firmware 1.60.3 places 16-bit audio in the upper 16 bits of S32_LE (`s32_shift=16`),
    whereas legacy A113 firmware used lower 24 bits (`s32_shift=8`).
    """
    if bits_per_sample == 32:
        usable = len(pcm_bytes) - (len(pcm_bytes) % 4)
        if usable <= 0:
            return np.zeros(0, dtype=np.float32)
        s32 = np.frombuffer(pcm_bytes[:usable], dtype="<i4")
        s16 = np.clip(s32 >> s32_shift, -32768, 32767)
        return s16.astype(np.float32) / 32768.0
    usable = len(pcm_bytes) - (len(pcm_bytes) % 2)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32)
    s16 = np.frombuffer(pcm_bytes[:usable], dtype="<i2")
    return s16.astype(np.float32) / 32768.0


def apply_highpass_filter(
    samples: np.ndarray,
    *,
    prev_x: float = 0.0,
    prev_y: float = 0.0,
    alpha: float = 0.968,
) -> tuple[np.ndarray, float, float]:
    """First-order DC/low-rumble high-pass filter (~80 Hz at 16 kHz) preserving filter state."""
    if len(samples) == 0:
        return samples, prev_x, prev_y
    out = np.empty_like(samples, dtype=np.float32)
    x_prev = prev_x
    y_prev = prev_y
    for i in range(len(samples)):
        x = float(samples[i])
        y = alpha * (y_prev + x - x_prev)
        out[i] = y
        x_prev = x
        y_prev = y
    return out, x_prev, y_prev


class DynamicAGC:
    """Lightweight RMS-based AGC with noise floor gating and tanh soft limiting."""

    def __init__(
        self,
        *,
        target_rms: float = 0.10,
        min_gain: float = 2.0,
        max_gain: float = 48.0,
        noise_floor_rms: float = 0.0012,
    ) -> None:
        self.target_rms = target_rms
        self.min_gain = min_gain
        self.max_gain = max_gain
        self.noise_floor_rms = noise_floor_rms
        self.gain = min_gain

    def process(self, samples: np.ndarray) -> np.ndarray:
        if len(samples) == 0:
            return samples
        frame_rms = float(np.sqrt(np.mean(samples * samples)))
        if frame_rms > self.noise_floor_rms:
            desired = min(self.max_gain, max(self.min_gain, self.target_rms / frame_rms))
            alpha = 0.35 if desired < self.gain else 0.20
            self.gain = (1.0 - alpha) * self.gain + alpha * desired
        else:
            # Decay gently toward min_gain in silence so room noise stays well below VAD threshold.
            self.gain = 0.92 * self.gain + 0.08 * self.min_gain
        return np.tanh(samples * self.gain).astype(np.float32)


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

    # Check 3-char CJK phonetic prefix (e.g. "蔡纳米", "海诺米", "嗨，糯米") via pypinyin.
    cjk_chars: list[tuple[int, str]] = []
    for idx, ch in enumerate(stripped):
        if "\u4e00" <= ch <= "\u9fff":
            cjk_chars.append((idx, ch))
            if len(cjk_chars) == 3:
                break
        elif ch not in " ,，.。!！?？:：":
            break
    if len(cjk_chars) == 3:
        py = lazy_pinyin("".join(ch for _, ch in cjk_chars))
        if len(py) == 3 and py[0] in _PINYIN_FIRST and py[1] in _PINYIN_SECOND and py[2] in _PINYIN_THIRD:
            cut = cjk_chars[-1][0] + 1
            rest = stripped[cut:].strip(" ，,。.!！？:：")
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
        self._hp_prev_x: float = 0.0
        self._hp_prev_y: float = 0.0
        self._agc = (
            DynamicAGC(
                target_rms=config.agc_target_rms,
                max_gain=config.agc_max_gain,
            )
            if config.agc_enabled
            else None
        )

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

        samples = pcm_to_float32(
            pcm_bytes,
            bits_per_sample=self._config.capture_bits_per_sample,
            s32_shift=self._config.s32_shift,
        )
        if len(samples) == 0:
            return
        samples, self._hp_prev_x, self._hp_prev_y = apply_highpass_filter(
            samples, prev_x=self._hp_prev_x, prev_y=self._hp_prev_y
        )
        if self._agc is not None:
            samples = self._agc.process(samples)
        elif self._config.input_gain != 1.0:
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
