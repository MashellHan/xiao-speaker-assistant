"""Real-time Frequency-Domain Adaptive Filter (FDAF) linear Acoustic Echo Canceller."""

from __future__ import annotations

import numpy as np


class LinearEchoCanceller:
    """Overlap-save frequency-domain NLMS echo canceller (16 kHz, 256-sample hop / 512 FFT).

    Uses SpeexDSP/Valin per-bin error-power step regularization so sparse harmonic TTS speech
    and double-talk do not cause weight divergence.
    """

    def __init__(
        self,
        *,
        hop_size: int = 256,
        num_blocks: int = 8,
        step_size: float = 0.35,
        reg: float = 1e-2,
    ) -> None:
        self.hop_size = hop_size
        self.fft_size = hop_size * 2
        self.num_blocks = num_blocks
        self.step_size = step_size
        self.reg = reg
        self.num_bins = self.fft_size // 2 + 1

        self._mic_in_buf = np.zeros(0, dtype=np.float32)
        self._ref_in_buf = np.zeros(0, dtype=np.float32)
        self._ref_prev = np.zeros(self.hop_size, dtype=np.float32)

        self._X_hist = np.zeros((self.num_blocks, self.num_bins), dtype=np.complex64)
        self._W = np.zeros((self.num_blocks, self.num_bins), dtype=np.complex64)

    def reset(self) -> None:
        self._mic_in_buf = np.zeros(0, dtype=np.float32)
        self._ref_in_buf = np.zeros(0, dtype=np.float32)
        self._ref_prev.fill(0.0)
        self._X_hist.fill(0.0)
        self._W.fill(0.0)

    def _process_block(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        ref_frame = np.concatenate([self._ref_prev, ref_hop])
        self._ref_prev = ref_hop.copy()

        self._X_hist[1:] = self._X_hist[:-1]
        X_new = np.fft.rfft(ref_frame).astype(np.complex64)
        self._X_hist[0] = X_new

        Y_spec = np.sum(self._W * self._X_hist, axis=0)
        y_time = np.fft.irfft(Y_spec, n=self.fft_size).astype(np.float32)[self.hop_size :]

        err = mic_hop - y_time
        ref_energy = float(np.mean(ref_hop * ref_hop))

        if ref_energy > 1e-8:
            e_padded = np.concatenate([np.zeros(self.hop_size, dtype=np.float32), err])
            E_spec = np.fft.rfft(e_padded).astype(np.complex64)
            tot_ref = (
                np.sum(self._X_hist.real * self._X_hist.real + self._X_hist.imag * self._X_hist.imag, axis=0)
            ).astype(np.float32)
            err_pow = (E_spec.real * E_spec.real + E_spec.imag * E_spec.imag).astype(np.float32)
            floor = max(self.reg, 0.02 * float(np.mean(tot_ref)))
            mu = self.step_size / (tot_ref + self.num_blocks * err_pow + floor)
            self._W += (np.conj(self._X_hist) * E_spec[None, :]) * mu[None, :]
            # Constrain filter weights to valid causal half in time domain
            w_time = np.fft.irfft(self._W, n=self.fft_size, axis=1)
            w_time[:, self.hop_size :] = 0.0
            self._W = np.fft.rfft(w_time, axis=1).astype(np.complex64)

        return np.clip(err, -1.0, 1.0).astype(np.float32)

    def process(self, mic: np.ndarray, ref: np.ndarray) -> np.ndarray:
        """Process 1D float32 mic and ref arrays of equal length and return echo-cancelled mic."""
        if len(mic) == 0:
            return mic
        n = min(len(mic), len(ref))
        self._mic_in_buf = np.concatenate([self._mic_in_buf, mic[:n]])
        self._ref_in_buf = np.concatenate([self._ref_in_buf, ref[:n]])

        outputs: list[np.ndarray] = []
        while len(self._mic_in_buf) >= self.hop_size:
            m_hop = self._mic_in_buf[: self.hop_size]
            r_hop = self._ref_in_buf[: self.hop_size]
            self._mic_in_buf = self._mic_in_buf[self.hop_size :]
            self._ref_in_buf = self._ref_in_buf[self.hop_size :]
            outputs.append(self._process_block(m_hop, r_hop))

        if not outputs:
            return np.zeros(0, dtype=np.float32)
        return np.concatenate(outputs)
