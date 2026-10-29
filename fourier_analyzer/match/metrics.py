"""Objective similarity metrics between the original and a reconstruction.

Every number here has a standard, textbook definition (no hand-tuned weights):

waveform_similarity   Pearson correlation coefficient r between the two
                      waveforms at the best time alignment (lag search
                      within +-max_lag), reported as 100 * max(r, 0).
                      This is the literal "how alike are the two waveforms".
                      It is phase sensitive: two sounds that are audibly
                      identical but start their cycles at different points
                      score low.
spectral_similarity   Cosine similarity of the two magnitude spectrograms
                      (STFT, 2048-point Hann, hop 512), 100 * cos(theta).
                      Phase independent: compares which frequencies are
                      present, how strongly, and when.
si_sdr_db             Scale-invariant signal-to-distortion ratio (Le Roux et
                      al. 2019) at the best alignment, in dB. Standard in
                      source separation; 0 dB = error as loud as the signal.
lsd_db                Log-spectral distance: RMS difference of the two power
                      spectrograms in dB (floored at -80 dB re. peak).
envelope_correlation  Pearson r of the two RMS loudness envelopes (23 ms
                      windows) * 100: does the volume move the same way.

The reconstruction is gain-matched to the original first (least squares),
because output level is a trivial knob.
"""
from __future__ import annotations

import numpy as np

N_FFT = 2048
HOP = 512


def _stft_mag(y: np.ndarray) -> np.ndarray:
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad))
    nfr = max(1, 1 + (len(yp) - N_FFT) // HOP)
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(nfr)[:, None]
    return np.abs(np.fft.rfft(yp[idx] * np.hanning(N_FFT)[None, :], axis=1))


def best_lag(a: np.ndarray, b: np.ndarray, max_lag: int) -> tuple[int, float]:
    """Shift for b (as applied by ``_shift``) that maximises the Pearson r with a,
    and that r. Negative when b is late."""
    n = len(a)
    a0 = a - a.mean()
    b0 = b - b.mean()
    m = 1 << int(np.ceil(np.log2(2 * n)))
    xc = np.fft.irfft(np.fft.rfft(a0, m) * np.conj(np.fft.rfft(b0, m)), m)
    lags = np.concatenate([np.arange(0, max_lag + 1), np.arange(-max_lag, 0)])
    vals = xc[lags % m]
    i = int(np.argmax(vals))
    denom = np.sqrt((a0 ** 2).sum() * (b0 ** 2).sum()) + 1e-20
    return int(lags[i]), float(vals[i] / denom)


def _shift(b: np.ndarray, lag: int) -> np.ndarray:
    if lag > 0:
        return np.concatenate([np.zeros(lag), b[:-lag]])
    if lag < 0:
        return np.concatenate([b[-lag:], np.zeros(-lag)])
    return b


def _pearson(a, b):
    a = a - a.mean()
    b = b - b.mean()
    d = np.sqrt((a ** 2).sum() * (b ** 2).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


class Reference:
    """Pre-computes the original's features so each candidate is cheap."""

    def __init__(self, y: np.ndarray, sr: int, max_lag_s: float = 0.02):
        self.y = np.asarray(y, dtype=np.float64)
        self.sr = sr
        self.n = len(self.y)
        self.max_lag = int(max_lag_s * sr)
        self.S = _stft_mag(self.y)
        self.S_norm = np.sqrt((self.S ** 2).sum()) + 1e-20
        self.P_db = 20 * np.log10(self.S + 1e-12)
        self.floor = self.P_db.max() - 80
        self.env = self._env(self.y)

    @staticmethod
    def _env(y):
        n = len(y) // 512
        if n == 0:
            return np.zeros(1)
        return np.sqrt(np.mean(y[: n * 512].reshape(n, 512) ** 2, axis=1))

    def _fit(self, c):
        c = np.asarray(c, dtype=np.float64)
        if len(c) < self.n:
            c = np.pad(c, (0, self.n - len(c)))
        return c[: self.n]

    def spectral(self, c) -> float:
        c = self._fit(c)
        Sc = _stft_mag(c)
        k = min(len(Sc), len(self.S))
        num = (self.S[:k] * Sc[:k]).sum()
        den = self.S_norm * (np.sqrt((Sc[:k] ** 2).sum()) + 1e-20)
        return float(100.0 * max(num / den, 0.0))

    def waveform(self, c) -> float:
        c = self._fit(c)
        if not np.any(c):
            return 0.0
        _, r = best_lag(self.y, c, self.max_lag)
        return float(100.0 * max(r, 0.0))

    def all(self, c) -> dict:
        c = self._fit(c)
        if not np.any(c):
            return {"waveform_similarity": 0.0, "spectral_similarity": 0.0, "si_sdr_db": -60.0,
                    "lsd_db": 80.0, "envelope_correlation": 0.0, "lag_ms": 0.0}
        lag, r = best_lag(self.y, c, self.max_lag)
        cs = _shift(c, lag)
        # SI-SDR at best alignment
        t = self.y - self.y.mean()
        e_ = cs - cs.mean()
        s_t = e_ * ((t * e_).sum() / ((e_ ** 2).sum() + 1e-20))   # least-squares gain match
        proj = ((e_ * t).sum() / ((t ** 2).sum() + 1e-20)) * t    # SI-SDR target projection
        noise = e_ - proj
        si_sdr = 10 * np.log10(((proj ** 2).sum() + 1e-20) / ((noise ** 2).sum() + 1e-20))
        # LSD on gain-matched spectrogram
        Sc = _stft_mag(s_t)
        k = min(len(Sc), len(self.S))
        Pc = np.maximum(20 * np.log10(Sc[:k] + 1e-12), self.floor)
        Pt = np.maximum(self.P_db[:k], self.floor)
        lsd = float(np.mean(np.sqrt(np.mean((Pt - Pc) ** 2, axis=1))))
        return {
            "waveform_similarity": round(100.0 * max(r, 0.0), 2),
            "spectral_similarity": round(self.spectral(c), 2),
            "si_sdr_db": round(float(si_sdr), 2),
            "lsd_db": round(lsd, 2),
            "envelope_correlation": round(100.0 * max(_pearson(self.env, self._env(cs)), 0.0), 2),
            "lag_ms": round(1000.0 * lag / self.sr, 2),
        }


METRIC_INFO = {
    "waveform_similarity": "Pearson correlation of the two waveforms at best alignment (x100). Phase sensitive.",
    "spectral_similarity": "Cosine similarity of the magnitude spectrograms (x100). Phase independent.",
    "si_sdr_db": "Scale-invariant signal-to-distortion ratio in dB (higher is better; >10 dB is close).",
    "lsd_db": "Log-spectral distance in dB (lower is better; <3 dB is close).",
    "envelope_correlation": "Pearson correlation of the loudness envelopes (x100).",
}
