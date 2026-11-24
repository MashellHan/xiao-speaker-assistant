"""Real-time Frequency-Domain Adaptive Filter (FDAF) linear Acoustic Echo Canceller."""

from __future__ import annotations

import numpy as np


class LinearEchoCanceller:
    """Overlap-save frequency-domain NLMS echo canceller (16 kHz, 256-sample hop / 512 FFT).

    Cancels speaker loopback reference (`ref`, OH2P hw:0,3 ch3) from `mic` (ch0) in the linear
    domain BEFORE any nonlinear AGC/tanh or VAD/KWS runs.
    """

    def __init__(
        self,
        *,
        hop_size: int = 256,
        num_blocks: int = 8,
        step_size: float = 0.35,
        reg: float = 1e-4,
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

        # Frequency-domain tapped delay line of reference frames and adaptive filter weights
        self._X_hist = np.zeros((self.num_blocks, self.num_bins), dtype=np.complex64)
        self._W = np.zeros((self.num_blocks, self.num_bins), dtype=np.complex64)
        self._power_est = np.full(self.num_bins, reg, dtype=np.float32)

    def reset(self) -> None:
        self._mic_in_buf = np.zeros(0, dtype=np.float32)
        self._ref_in_buf = np.zeros(0, dtype=np.float32)
        self._ref_prev.fill(0.0)
        self._X_hist.fill(0.0)
        self._W.fill(0.0)
        self._power_est.fill(self.reg)

    def _process_block(self, mic_hop: np.ndarray, ref_hop: np.ndarray) -> np.ndarray:
        ref_frame = np.concatenate([self._ref_prev, ref_hop])
        self._ref_prev = ref_hop.copy()

        # Shift reference FFT history and insert newest block
        self._X_hist[1:] = self._X_hist[:-1]
        X_new = np.fft.rfft(ref_frame).astype(np.complex64)
        self._X_hist[0] = X_new

        # Predicted echo in time domain (overlap-save: keep last hop_size samples)
        Y_spec = np.sum(self._W * self._X_hist, axis=0)
        y_time = np.fft.irfft(Y_spec, n=self.fft_size).astype(np.float32)[self.hop_size :]

        err = mic_hop - y_time

        # Smooth far-end spectral power + double-talk step attenuation
        ref_pow = (X_new.real * X_new.real + X_new.imag * X_new.imag).astype(np.float32)
        self._power_est = 0.85 * self._power_est + 0.15 * ref_pow
        total_ref_pow = np.sum(np.abs(self._X_hist) ** 2, axis=0) + self.reg

        mic_energy = float(np.mean(mic_hop * mic_hop))
        err_energy = float(np.mean(err * err))
        ref_energy = float(np.mean(ref_hop * ref_hop))

        # Only adapt significantly when far-end reference is active; slow adaptation on near-end double-talk.
        if ref_energy > 1e-6:
            dt_scale = 0.15 if err_energy > mic_energy * 1.1 else 1.0
            mu = (self.step_size * dt_scale) / total_ref_pow
            e_padded = np.concatenate([np.zeros(self.hop_size, dtype=np.float32), err])
            E_spec = np.fft.rfft(e_padded).astype(np.complex64)
            grad = (np.conj(self._X_hist) * E_spec[None, :]) * mu[None, :]
            self._W += grad
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
