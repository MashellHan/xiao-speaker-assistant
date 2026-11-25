from unittest.mock import MagicMock

import numpy as np

from xiao_speaker_assistant.config import AppConfig
from xiao_speaker_assistant.voice.aec import LinearEchoCanceller
from xiao_speaker_assistant.voice.audio_pipeline import DynamicAGC, deinterleave_mic_and_ref
from xiao_speaker_assistant.voice.speaker_deployer import (
    build_asound_xsa_conf,
    build_boot_script,
)
from xiao_speaker_assistant.voice.ws_server import SpeakerGateway


def test_linear_aec_attenuates_harmonic_tts_echo_and_preserves_double_talk() -> None:
    aec = LinearEchoCanceller(hop_size=256, num_blocks=8, step_size=0.35)

    # Realistic sparse harmonic + amplitude-modulated TTS speech (3 seconds at 16 kHz)
    n = 16000 * 3
    t = np.arange(n, dtype=np.float32) / 16000.0
    env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    f0 = 165.0 + 25.0 * np.sin(2 * np.pi * 1.8 * t)
    ref = env * (
        0.18 * np.sin(2 * np.pi * f0 * t)
        + 0.10 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.05 * np.sin(2 * np.pi * 3 * f0 * t)
    )
    ref = ref.astype(np.float32)

    echo = np.zeros(n, dtype=np.float32)
    echo[40:] += 0.55 * ref[:-40]
    echo[120:] += -0.25 * ref[:-120]
    echo[310:] += 0.12 * ref[:-310]

    # 2.0s far-end adaptation, 0.5s double-talk with unvoiced/high-band near-end, 0.5s post-DT
    rng = np.random.default_rng(7)
    near_speech = np.zeros(8000, dtype=np.float32)
    t_near = np.arange(8000, dtype=np.float32) / 16000.0
    near_speech += 0.10 * np.sin(2 * np.pi * 1900.0 * t_near).astype(np.float32)
    near_speech += 0.04 * rng.standard_normal(8000).astype(np.float32)

    mic = echo.copy()
    mic[32000:40000] += near_speech

    cleaned = aec.process(mic, ref)
    assert len(cleaned) == (n // 256) * 256

    # Post-double-talk echo cancellation must stay converged (no divergence!)
    in_pow = float(np.mean(echo[40000:47000] ** 2))
    out_pow = float(np.mean(cleaned[40000:47000] ** 2))
    post_dt_erle = 10.0 * np.log10(in_pow / max(out_pow, 1e-12))
    assert post_dt_erle > 12.0

    # Near-end speech remains strongly correlated during double-talk
    dt_seg = cleaned[33000:39500]
    corr = float(np.corrcoef(dt_seg, near_speech[1000:7500])[0, 1])
    assert corr > 0.80


def test_dynamic_agc_freezes_boost_during_active_far_end_reference() -> None:
    agc = DynamicAGC(target_rms=0.10, min_gain=2.0, max_gain=48.0, noise_floor_rms=0.0012)
    agc.gain = 6.0
    t = np.linspace(0, 1, 480, dtype=np.float32)
    residual_echo = 0.006 * np.sin(2 * np.pi * 10 * t)
    for _ in range(10):
        agc.process(residual_echo, freeze_boost=True)
    # Gain must NOT climb toward 16x while far-end speaker is active
    assert agc.gain <= 6.01


def test_deinterleave_mic_and_ref_splits_two_channels() -> None:
    interleaved = np.array([0.1, -0.5, 0.2, -0.4, 0.3, -0.3], dtype=np.float32)
    mic, ref = deinterleave_mic_and_ref(interleaved, channels=2)
    assert np.allclose(mic, [0.1, 0.2, 0.3])
    assert np.allclose(ref, [-0.5, -0.4, -0.3])


def test_asound_xsa_conf_includes_system_alsa_and_routes_ch0_and_ch3_loopback() -> None:
    conf = build_asound_xsa_conf()
    assert "</usr/share/alsa/alsa.conf>" in conf
    assert "pcm.xsa_aec" in conf
    assert "ttable.0.0 1" in conf
    assert "ttable.1.3 1" in conf
    boot = build_boot_script()
    assert "ALSA_CONFIG_PATH=" in boot
    assert "unset ALSA_CONFIG_PATH" in boot


def test_aec_enabled_2ch_keeps_vad_open_for_full_duplex_barge_in() -> None:
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


def test_linear_aec_converges_on_quiet_oh2p_amplitude_signal() -> None:
    rng = np.random.default_rng(123)
    aec = LinearEchoCanceller(hop_size=256, num_blocks=8, step_size=0.40, reg=1e-5)
    n = 16000 * 2
    ref = 0.015 * rng.standard_normal(n).astype(np.float32)
    mic = np.zeros(n, dtype=np.float32)
    mic[24:] += 0.45 * ref[:-24]
    mic[90:] += -0.20 * ref[:-90]

    cleaned = aec.process(mic, ref)
    in_pow = float(np.mean(mic[16000:30000] ** 2))
    out_pow = float(np.mean(cleaned[16000:30000] ** 2))
    erle_db = 10.0 * np.log10(in_pow / max(out_pow, 1e-12))
    assert erle_db > 15.0
