import asyncio
import struct
from unittest.mock import MagicMock

import numpy as np

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.voice.aec import LinearEchoCanceller
from xiao_speaker_assistant.voice.audio_pipeline import deinterleave_mic_and_ref
from xiao_speaker_assistant.voice.speaker_deployer import (
    build_asound_xsa_conf,
    build_boot_script,
)
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_linear_aec_attenuates_speaker_echo_and_preserves_near_end() -> None:
    rng = np.random.default_rng(42)
    aec = LinearEchoCanceller(hop_size=256, num_blocks=8, step_size=0.45)

    # 3 seconds of synthetic far-end speaker signal
    n = 16000 * 3
    ref = 0.25 * rng.standard_normal(n).astype(np.float32)

    # Simulated acoustic echo path (multi-tap delayed reflection)
    echo = np.zeros(n, dtype=np.float32)
    echo[40:] += 0.55 * ref[:-40]
    echo[120:] += -0.25 * ref[:-120]
    echo[310:] += 0.12 * ref[:-310]

    # Far-end-only portion first (2.0s) to adapt, then 1.0s double-talk with near-end speech tone
    t_near = np.arange(16000, dtype=np.float32) / 16000.0
    near_speech = 0.12 * np.sin(2 * np.pi * 440.0 * t_near).astype(np.float32)
    mic = echo.copy()
    mic[32000:] += near_speech

    cleaned = aec.process(mic, ref)
    assert len(cleaned) == (n // 256) * 256

    # Measure ERLE on converged echo-only window [24000:32000]
    in_pow = float(np.mean(echo[24000:32000] ** 2))
    out_pow = float(np.mean(cleaned[24000:32000] ** 2))
    erle_db = 10.0 * np.log10(in_pow / max(out_pow, 1e-12))
    assert erle_db > 15.0

    # Near-end speech tone remains clearly recoverable in double-talk window
    dt_seg = cleaned[34000:46000]
    corr = float(np.corrcoef(dt_seg, near_speech[2000:14000])[0, 1])
    assert corr > 0.75


def test_deinterleave_mic_and_ref_splits_two_channels() -> None:
    interleaved = np.array([0.1, -0.5, 0.2, -0.4, 0.3, -0.3], dtype=np.float32)
    mic, ref = deinterleave_mic_and_ref(interleaved, channels=2)
    assert np.allclose(mic, [0.1, 0.2, 0.3])
    assert np.allclose(ref, [-0.5, -0.4, -0.3])


def test_asound_xsa_conf_routes_ch0_and_ch3_loopback() -> None:
    conf = build_asound_xsa_conf()
    assert "</etc/asound.conf>" in conf
    assert "pcm.xsa_aec" in conf
    assert "ttable.0.0 1" in conf
    assert "ttable.1.3 1" in conf
    boot = build_boot_script()
    assert "ALSA_CONFIG_PATH=" in boot


def test_aec_enabled_2ch_does_not_mute_pipeline_during_playback() -> None:
    cfg = AppConfig()
    cfg.speech.aec_enabled = True
    cfg.speech.capture_channels = 2
    cfg.speech.capture_pcm = "xsa_aec"

    async def dummy(q: str) -> str:
        return q

    gw = SpeakerGateway(cfg, on_query=dummy)
    fake_pipeline = MagicMock()
    gw._pipeline = fake_pipeline

    class FakeConn:
        remote_address = ("127.0.0.1", 9999)

        async def send(self, raw) -> None:
            pass

    pb = gw._get_playback(FakeConn())
    assert pb._on_playback_state is not None
    pb._on_playback_state(True)
    pb._on_playback_state(False)
    fake_pipeline.set_muted.assert_not_called()
