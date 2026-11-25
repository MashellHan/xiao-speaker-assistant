"""Low-CPU local KWS + Silero VAD + SenseVoice/Paraformer/Cloud ASR pipeline."""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Callable

import numpy as np
from pypinyin import lazy_pinyin
from scipy.signal import lfilter

from xiao_speaker_assistant.config import SpeechConfig
from xiao_speaker_assistant.voice.aec import LinearEchoCanceller
from xiao_speaker_assistant.voice.cloud_asr import DoubaoFlashASR, OpenAIWhisperASR
from xiao_speaker_assistant.voice.models_setup import (
    find_english_kws_dir,
    find_paraformer_dir,
    find_sense_voice_dir,
    generate_english_keywords_file,
    generate_keywords_file,
)

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000
VAD_WINDOW_SAMPLES = 512

HI_NOMI_REGEX = re.compile(
    r"^\s*(?:hi|hey|hello|can|h|i|嗨|哈喽|你好)?\s*[,，.。!！?\s]*"
    r"(?:nomi|normi|nami|hnomi|hno\s*米|h\s*no\s*米|know\s*mi|know\s*me|no\s*mi|no\s*me|naomi|noumi|诺米|糯米|闹米|老米|纳米)"
    r"(?:[,，.。!！?\s:]*(.*))?$",
    re.IGNORECASE | re.DOTALL,
)

_PINYIN_FIRST = {"hai", "cai", "kai", "ai", "tai", "sai", "hei", "ha", "a", "chen", "han", "an"}
_PINYIN_SECOND = {"nuo", "na", "luo", "no", "nao", "lao"}
_PINYIN_THIRD = {"mi", "min", "mei", "ming"}
_NON_ZH_EN_HALLUCINATION_RE = re.compile(r"[\u3040-\u30ff\uac00-\ud7af\u1100-\u11ff]")


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
    """Convert raw PCM bytes to normalized [-1, 1] float32 samples."""
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


def deinterleave_mic_and_ref(
    samples: np.ndarray, *, channels: int
) -> tuple[np.ndarray, np.ndarray | None]:
    """Split interleaved `[mic, ref, ...]` float32 samples into `(mic, ref)`."""
    if channels <= 1:
        return samples, None
    usable = len(samples) - (len(samples) % channels)
    if usable <= 0:
        return np.zeros(0, dtype=np.float32), None
    frames = samples[:usable].reshape(-1, channels)
    return frames[:, 0].copy(), frames[:, 1].copy()


def apply_highpass_filter(
    samples: np.ndarray,
    *,
    prev_x: float = 0.0,
    prev_y: float = 0.0,
    alpha: float = 0.968,
) -> tuple[np.ndarray, float, float]:
    """Vectorized 1st-order DC/low-rumble high-pass filter (~80 Hz at 16 kHz)."""
    if len(samples) == 0:
        return samples, prev_x, prev_y
    b = np.array([alpha, -alpha], dtype=np.float64)
    a = np.array([1.0, -alpha], dtype=np.float64)
    zi = np.array([alpha * (prev_y - prev_x)], dtype=np.float64)
    y, _ = lfilter(b, a, samples.astype(np.float64), zi=zi)
    out = y.astype(np.float32)
    return out, float(samples[-1]), float(out[-1])


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

    def process(self, samples: np.ndarray, *, freeze_boost: bool = False) -> np.ndarray:
        if len(samples) == 0:
            return samples
        frame_rms = float(np.sqrt(np.mean(samples * samples)))
        if frame_rms > self.noise_floor_rms:
            desired = min(self.max_gain, max(self.min_gain, self.target_rms / frame_rms))
            if freeze_boost and desired > self.gain:
                desired = self.gain
            alpha = 0.35 if desired < self.gain else 0.20
            self.gain = (1.0 - alpha) * self.gain + alpha * desired
        else:
            self.gain = 0.92 * self.gain + 0.08 * self.min_gain
        return np.tanh(samples * self.gain).astype(np.float32)


def match_wakeup_phrase(
    text: str, extra_keywords: list[str] | None = None
) -> tuple[str, str] | None:
    """Match `Hi Nomi` or punctuation-tolerant Chinese wake keywords (e.g. 你好，管家。)."""
    stripped = text.strip()
    if not stripped:
        return None

    m = HI_NOMI_REGEX.match(stripped)
    if m:
        rest = (m.group(1) or "").strip(" ，,。.!！？:：")
        return "hi nomi", rest

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

    # Punctuation-tolerant Chinese keyword prefix match (handles "你好，管家。" from SenseVoice ITN).
    norm_chars: list[tuple[int, str]] = []
    for idx, ch in enumerate(stripped):
        if ch not in " ,，.。!！?？:：":
            norm_chars.append((idx, ch.lower()))
    norm_str = "".join(c for _, c in norm_chars)

    for kw in extra_keywords or []:
        clean_kw = re.sub(r"[ ,，.。!！?？:：]+", "", kw.strip().lower())
        if not clean_kw:
            continue
        if norm_str.startswith(clean_kw):
            last_orig_idx = norm_chars[len(clean_kw) - 1][0]
            rest = stripped[last_orig_idx + 1 :].strip(" ，,。.!！？:：")
            return clean_kw, rest
    return None


def _normalize_kws_tag(raw_kw: str) -> str:
    stripped = raw_kw.strip()
    if any("\u4e00" <= ch <= "\u9fff" for ch in stripped):
        return re.sub(r"\s+", "", stripped)
    return re.sub(r"\s+", "_", stripped)


class LocalSpeechPipeline:
    """Runs unified KWS (wakeup + interrupt) + Silero VAD + local/cloud ASR."""

    def __init__(self, config: SpeechConfig) -> None:
        import sherpa_onnx

        self._config = config
        self._lock = threading.RLock()
        models_dir = Path(config.models_dir).expanduser()

        self._wake_set = {_normalize_kws_tag(kw) for kw in config.local_keywords if kw.strip()}
        self._interrupt_set = {
            _normalize_kws_tag(kw) for kw in config.interrupt_keywords if kw.strip()
        }
        combined_keywords = list(dict.fromkeys([*config.local_keywords, *config.interrupt_keywords]))
        kw_file = generate_keywords_file(models_dir, combined_keywords, filename="keywords.txt")

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

        self._en_kws = None
        self._en_kws_stream = None
        if config.en_kws_enabled:
            en_dir = find_english_kws_dir(models_dir)
            if en_dir is not None:
                en_kw_file = generate_english_keywords_file(en_dir, config.en_keywords)
                self._en_kws = sherpa_onnx.KeywordSpotter(
                    provider="cpu",
                    num_threads=1,
                    max_active_paths=8,
                    keywords_score=config.en_kws_score,
                    keywords_threshold=config.en_kws_threshold,
                    num_trailing_blanks=0,
                    keywords_file=str(en_kw_file),
                    tokens=str(en_dir / "tokens.txt"),
                    encoder=str(en_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                    decoder=str(en_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                    joiner=str(en_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                )
                self._en_kws_stream = self._en_kws.create_stream()
                logger.info("English Hi Nomi KWS enabled (%s)", en_dir.name)

        silero = sherpa_onnx.SileroVadModelConfig(
            model=str(models_dir / "silero_vad.onnx"),
            threshold=config.vad_threshold,
            min_silence_duration=config.vad_min_silence_ms / 1000.0,
            min_speech_duration=0.10,
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

        self._cloud_asr: DoubaoFlashASR | OpenAIWhisperASR | None = None
        self._asr = None
        if config.asr_model == "doubao_flash":
            asr = DoubaoFlashASR()
            asr.validate_credentials()
            self._cloud_asr = asr
        elif config.asr_model == "openai_whisper":
            asr_w = OpenAIWhisperASR(
                base_url=config.whisper_base_url,
                api_key_env=config.whisper_api_key_env,
                model=config.whisper_model,
                language=config.asr_language,
            )
            asr_w.validate_credentials()
            self._cloud_asr = asr_w
        elif config.asr_model == "paraformer":
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
        self._hp_ref_prev_x: float = 0.0
        self._hp_ref_prev_y: float = 0.0
        self._aec = (
            LinearEchoCanceller()
            if (config.aec_enabled and config.capture_channels >= 2)
            else None
        )
        self._agc = (
            DynamicAGC(
                target_rms=config.agc_target_rms,
                max_gain=config.agc_max_gain,
            )
            if config.agc_enabled
            else None
        )

    def set_muted(self, muted: bool) -> None:
        with self._lock:
            self._explicit_muted = muted
            if not muted:
                self._vad.reset()
                self._vad_buffer = np.zeros(0, dtype=np.float32)
            else:
                self._kws.reset_stream(self._kws_stream)
                if self._en_kws is not None and self._en_kws_stream is not None:
                    self._en_kws.reset_stream(self._en_kws_stream)

    def mute_for(self, seconds: float) -> None:
        with self._lock:
            self._muted_until = max(self._muted_until, time.monotonic() + seconds)

    def refresh_activity(self) -> None:
        with self._lock:
            if self.awake:
                self._last_active_at = time.monotonic()

    def set_awake(self, awake: bool, *, preserve_vad: bool = False) -> None:
        with self._lock:
            self.awake = awake
            self._last_active_at = time.monotonic() if awake else 0.0
            if not preserve_vad:
                self._vad.reset()
                self._vad_buffer = np.zeros(0, dtype=np.float32)
            self._kws.reset_stream(self._kws_stream)
            if self._en_kws is not None and self._en_kws_stream is not None:
                self._en_kws.reset_stream(self._en_kws_stream)

    def update_config(self, new_config: SpeechConfig) -> None:
        """Hot-reload speech thresholds, gain, AEC/AGC, and keywords safely under `_lock`."""
        import sherpa_onnx

        with self._lock:
            old_cfg = self._config
            self._config = new_config
            models_dir = Path(new_config.models_dir).expanduser()

            if (
                new_config.local_keywords != old_cfg.local_keywords
                or new_config.interrupt_keywords != old_cfg.interrupt_keywords
                or new_config.kws_score != old_cfg.kws_score
                or new_config.kws_threshold != old_cfg.kws_threshold
            ):
                self._wake_set = {
                    _normalize_kws_tag(kw) for kw in new_config.local_keywords if kw.strip()
                }
                self._interrupt_set = {
                    _normalize_kws_tag(kw) for kw in new_config.interrupt_keywords if kw.strip()
                }
                combined_keywords = list(
                    dict.fromkeys([*new_config.local_keywords, *new_config.interrupt_keywords])
                )
                kw_file = generate_keywords_file(
                    models_dir, combined_keywords, filename="keywords.txt"
                )
                self._kws = sherpa_onnx.KeywordSpotter(
                    provider="cpu",
                    num_threads=1,
                    max_active_paths=8,
                    keywords_score=new_config.kws_score,
                    keywords_threshold=new_config.kws_threshold,
                    num_trailing_blanks=0,
                    keywords_file=str(kw_file),
                    tokens=str(models_dir / "tokens.txt"),
                    encoder=str(models_dir / "encoder.onnx"),
                    decoder=str(models_dir / "decoder.onnx"),
                    joiner=str(models_dir / "joiner.onnx"),
                )
                self._kws_stream = self._kws.create_stream()

            if (
                new_config.en_kws_enabled != old_cfg.en_kws_enabled
                or new_config.en_keywords != old_cfg.en_keywords
                or new_config.en_kws_score != old_cfg.en_kws_score
                or new_config.en_kws_threshold != old_cfg.en_kws_threshold
            ):
                self._en_kws = None
                self._en_kws_stream = None
                if new_config.en_kws_enabled:
                    en_dir = find_english_kws_dir(models_dir)
                    if en_dir is not None:
                        en_kw_file = generate_english_keywords_file(en_dir, new_config.en_keywords)
                        self._en_kws = sherpa_onnx.KeywordSpotter(
                            provider="cpu",
                            num_threads=1,
                            max_active_paths=8,
                            keywords_score=new_config.en_kws_score,
                            keywords_threshold=new_config.en_kws_threshold,
                            num_trailing_blanks=0,
                            keywords_file=str(en_kw_file),
                            tokens=str(en_dir / "tokens.txt"),
                            encoder=str(en_dir / "encoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                            decoder=str(en_dir / "decoder-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                            joiner=str(en_dir / "joiner-epoch-12-avg-2-chunk-16-left-64.int8.onnx"),
                        )
                        self._en_kws_stream = self._en_kws.create_stream()

            if (
                new_config.vad_threshold != old_cfg.vad_threshold
                or new_config.vad_min_silence_ms != old_cfg.vad_min_silence_ms
            ):
                silero = sherpa_onnx.SileroVadModelConfig(
                    model=str(models_dir / "silero_vad.onnx"),
                    threshold=new_config.vad_threshold,
                    min_silence_duration=new_config.vad_min_silence_ms / 1000.0,
                    min_speech_duration=0.10,
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
                self._vad_buffer = np.zeros(0, dtype=np.float32)

            if new_config.agc_enabled:
                self._agc = DynamicAGC(
                    target_rms=new_config.agc_target_rms,
                    max_gain=new_config.agc_max_gain,
                )
            else:
                self._agc = None

            if new_config.aec_enabled and new_config.capture_channels >= 2:
                if self._aec is None:
                    self._aec = LinearEchoCanceller()
            else:
                self._aec = None

    def feed_pcm(
        self,
        pcm_bytes: bytes,
        *,
        on_wakeup: Callable[[str], None] | None = None,
        on_utterance: Callable[[str], None] | None = None,
        on_sleep: Callable[[], None] | None = None,
        on_interrupt: Callable[[str], None] | None = None,
        on_barge_in_wakeup: Callable[[str], None] | None = None,
    ) -> None:
        with self._lock:
            self._feed_pcm_locked(
                pcm_bytes,
                on_wakeup=on_wakeup,
                on_utterance=on_utterance,
                on_sleep=on_sleep,
                on_interrupt=on_interrupt,
                on_barge_in_wakeup=on_barge_in_wakeup,
            )

    def _feed_pcm_locked(
        self,
        pcm_bytes: bytes,
        *,
        on_wakeup: Callable[[str], None] | None = None,
        on_utterance: Callable[[str], None] | None = None,
        on_sleep: Callable[[], None] | None = None,
        on_interrupt: Callable[[str], None] | None = None,
        on_barge_in_wakeup: Callable[[str], None] | None = None,
    ) -> None:
        now = time.monotonic()
        is_muted = self._explicit_muted or now < self._muted_until

        if self.awake and not is_muted and (now - self._last_active_at) > self._config.conversation_timeout_sec:
            logger.info("Local conversation timed out; returning to idle")
            self.set_awake(False)
            if on_sleep:
                on_sleep()
            return

        raw_samples = pcm_to_float32(
            pcm_bytes,
            bits_per_sample=self._config.capture_bits_per_sample,
            s32_shift=self._config.s32_shift,
        )
        if len(raw_samples) == 0:
            return
        mic_samples, ref_samples = deinterleave_mic_and_ref(
            raw_samples, channels=self._config.capture_channels
        )
        if len(mic_samples) == 0:
            return
        mic_samples, self._hp_prev_x, self._hp_prev_y = apply_highpass_filter(
            mic_samples, prev_x=self._hp_prev_x, prev_y=self._hp_prev_y
        )
        # Apply linear pre-gain identically to both mic and ref before AEC so quiet OH2P hardware
        # (PDM Gain=0) stays in the well-conditioned adaptation regime while preserving linearity.
        if self._agc is None and self._config.input_gain != 1.0:
            mic_samples = mic_samples * self._config.input_gain
            if ref_samples is not None:
                ref_samples = ref_samples * self._config.input_gain

        far_end_active = False
        if self._aec is not None and ref_samples is not None:
            ref_samples, self._hp_ref_prev_x, self._hp_ref_prev_y = apply_highpass_filter(
                ref_samples, prev_x=self._hp_ref_prev_x, prev_y=self._hp_ref_prev_y
            )
            far_end_active = float(np.mean(ref_samples * ref_samples)) > 1e-6
            samples = self._aec.process(mic_samples, ref_samples)
            if len(samples) == 0:
                return
        else:
            samples = mic_samples

        if self._agc is not None:
            samples = self._agc.process(samples, freeze_boost=far_end_active)
        else:
            samples = np.clip(samples, -1.0, 1.0)
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

        # Unified single KeywordSpotter handles both idle wakeup and awake interrupt.
        self._kws_stream.accept_waveform(SAMPLE_RATE, samples)
        while self._kws.is_ready(self._kws_stream):
            self._kws.decode_stream(self._kws_stream)
            hit = self._kws.get_result(self._kws_stream)
            if not hit:
                continue
            if self.awake and hit in self._interrupt_set:
                logger.info("Barge-in interrupt keyword detected: %s", hit)
                self._explicit_muted = False
                self._muted_until = 0.0
                self.set_awake(False)
                if on_interrupt:
                    on_interrupt(hit)
                return
            if not self.awake and hit in self._wake_set:
                # Preserve VAD buffer so one-shot phrases right after the wake word aren't clipped.
                self.set_awake(True, preserve_vad=True)
                logger.info("KWS triggered: %s", hit)
                if on_wakeup:
                    on_wakeup(hit)
                break
            if self.awake and is_muted and hit in self._wake_set:
                logger.info("Barge-in wake word detected while speaking: %s", hit)
                self._explicit_muted = False
                self._muted_until = 0.0
                self.set_awake(True)
                if on_barge_in_wakeup:
                    on_barge_in_wakeup(hit)
                return

        # Dedicated lightweight English GigaSpeech KWS for `Hi Nomi` when available.
        if self._en_kws is not None and self._en_kws_stream is not None:
            self._en_kws_stream.accept_waveform(SAMPLE_RATE, samples)
            while self._en_kws.is_ready(self._en_kws_stream):
                self._en_kws.decode_stream(self._en_kws_stream)
                en_hit = self._en_kws.get_result(self._en_kws_stream)
                if not en_hit:
                    continue
                self._en_kws.reset_stream(self._en_kws_stream)
                if not self.awake:
                    self.set_awake(True, preserve_vad=True)
                    logger.info("English KWS triggered: %s", en_hit)
                    if on_wakeup:
                        on_wakeup(en_hit)
                    break
                if self.awake and is_muted:
                    logger.info("English barge-in wake word detected: %s", en_hit)
                    self._explicit_muted = False
                    self._muted_until = 0.0
                    self.set_awake(True)
                    if on_barge_in_wakeup:
                        on_barge_in_wakeup(en_hit)
                    return

        if is_muted:
            return

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
                logger.info("ASR wake matched (%s): %s", wake_label, text)
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
        if self._cloud_asr is not None:
            return self._cloud_asr.transcribe(samples)
        assert self._asr is not None
        stream = self._asr.create_stream()
        stream.accept_waveform(SAMPLE_RATE, samples)
        self._asr.decode_stream(stream)
        text = stream.result.text.strip()
        if text and _NON_ZH_EN_HALLUCINATION_RE.search(text):
            logger.debug("Dropping non-zh/en SenseVoice hallucination: %s", text)
            return ""
        return text
