"""Perceptual similarity score between a target sound and a candidate render.

The score is 0..100 and combines:

* multi-resolution log-mel spectrogram error (timbre over time)   40 %
* high-resolution harmonic spectrogram error below 5 kHz          25 %
  (catches wrong waveforms: odd vs even harmonics, partial ratios)
* loudness-envelope error (articulation, timing, LFO on volume)    15 %
* long-term average spectrum error (overall tone balance)          10 %
* spectral-centroid trajectory error (filter/LFO movement)          10 %

Each component is a mean absolute dB error mapped linearly so that 0 dB of
error is 100 % and ``SCALE_DB`` dB of error is 0 %. Both signals are
loudness-matched first, because output gain is a trivial knob.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import uniform_filter1d

RESOLUTIONS = (512, 1024, 2048)
HOP = 256
N_MELS = 64
FLOOR_DB = -70.0
SCALE_DB = 12.0
HARM_FFT = 4096
HARM_MAX_HZ = 5000.0
SMOOTH_HARM = 8
SMOOTH_MEL = 8
DEADZONE = 2.0         # dB of per-cell error ignored (phase/beating noise)
WEIGHTS = {"spectral": 0.40, "harmonic": 0.25, "envelope": 0.15, "tone": 0.10, "movement": 0.10}

_mel_cache: dict = {}


def _mel_fb(sr, n_fft):
    key = (sr, n_fft)
    if key not in _mel_cache:
        import librosa
        _mel_cache[key] = librosa.filters.mel(sr=sr, n_fft=n_fft, n_mels=N_MELS, fmin=25.0,
                                              fmax=sr / 2 * 0.95)
    return _mel_cache[key]


def _stft_mag(y, n_fft):
    pad = n_fft // 2
    y = np.pad(y, (pad, pad))
    nfr = 1 + (len(y) - n_fft) // HOP
    if nfr <= 0:
        return np.zeros((n_fft // 2 + 1, 1))
    idx = np.arange(n_fft)[None, :] + HOP * np.arange(nfr)[:, None]
    frames = y[idx] * np.hanning(n_fft)[None, :]
    return np.abs(np.fft.rfft(frames, axis=1)).T


@dataclass
class Features:
    sr: int
    mels: list            # per resolution: (N_MELS, frames) in dB
    env: np.ndarray       # frame loudness dB
    ltas: np.ndarray      # mel dB, normalised
    centroid: np.ndarray  # per frame, in mel-bin units
    harm: np.ndarray      # (bins < 5 kHz, frames) linear-frequency dB
    weight: np.ndarray    # per frame weights (audible frames)
    gain: float           # rms


def features(y: np.ndarray, sr: int, ref_gain: float | None = None) -> Features:
    y = np.asarray(y, dtype=np.float64)
    rms = float(np.sqrt(np.mean(y ** 2)) + 1e-12)
    if ref_gain is not None:
        y = y * (ref_gain / rms)
    mels = []
    for n_fft in RESOLUTIONS:
        S = _stft_mag(y, n_fft)
        M = np.maximum(uniform_filter1d(_mel_fb(sr, n_fft) @ (S ** 2), size=SMOOTH_MEL, axis=1,
                                        mode="nearest"), 0)
        mels.append(10 * np.log10(M + 1e-10))
    ref = max(float(m.max()) for m in mels) if ref_gain is None else None
    mid = mels[1]
    e = 10 * np.log10(np.mean(10 ** (mid / 10), axis=0) + 1e-10)
    lin = 10 ** (mid / 10)
    cent = (np.arange(N_MELS)[:, None] * lin).sum(0) / (lin.sum(0) + 1e-12)
    ltas = 10 * np.log10(lin.mean(1) + 1e-12)
    H = _stft_mag(y, HARM_FFT)
    nb = int(HARM_MAX_HZ / (sr / HARM_FFT))
    # smooth ~90 ms in time: unison beating and random phases are not
    # reproducible exactly and should not count as timbre errors
    P2 = uniform_filter1d(H[1:nb] ** 2, size=SMOOTH_HARM, axis=1, mode="nearest")
    harm = 10 * np.log10(np.maximum(P2, 0) + 1e-18)
    f = Features(sr=sr, mels=mels, env=e, ltas=ltas - ltas.max(), centroid=cent,
                 harm=harm, weight=np.ones_like(e), gain=rms)
    f._ref = ref
    return f


def _clipdb(m, ref):
    return np.maximum(m, ref + FLOOR_DB)


def _dz(d):
    return np.maximum(d - DEADZONE, 0.0) if DEADZONE > 0 else d


def _sim(err_db):
    if not np.isfinite(err_db):
        return 0.0
    return float(np.clip(1.0 - err_db / SCALE_DB, 0.0, 1.0))


class Target:
    """Pre-computed target features so each candidate costs one analysis."""

    def __init__(self, y: np.ndarray, sr: int):
        self.y = np.asarray(y, dtype=np.float64)
        self.sr = sr
        self.f = features(self.y, sr)
        self.ref = self.f._ref
        self.n = len(self.y)
        # weight frames by audibility so silence does not dominate
        e = self.f.env
        self.w = np.clip((e - (e.max() - 50)) / 50, 0.05, 1.0)
        self.href = float(self.f.harm.max())
        he = self.f.harm.max(0)
        self.hw = np.clip((he - (he.max() - 50)) / 50, 0.05, 1.0)
        tl = 10 * np.log10((10 ** (self.f.harm / 10) * self.hw).sum(1) / self.hw.sum() + 1e-18)
        self.hl = np.maximum(tl - tl.max(), FLOOR_DB)

    def score(self, cand: np.ndarray, detail: bool = False):
        c = np.asarray(cand, dtype=np.float64)
        if len(c) < self.n:
            c = np.pad(c, (0, self.n - len(c)))
        c = c[: self.n]
        if not np.any(c):
            return (0.0, {k: 0.0 for k in WEIGHTS}) if detail else 0.0
        cf = features(c, self.sr, ref_gain=self.f.gain)
        ref = self.ref
        spec_errs = []
        for mt, mc in zip(self.f.mels, cf.mels):
            nfr = min(mt.shape[1], mc.shape[1])
            d = _dz(np.abs(_clipdb(mt[:, :nfr], ref) - _clipdb(mc[:, :nfr], ref)))
            w = self.w[:nfr] if len(self.w) >= nfr else np.ones(nfr)
            spec_errs.append(float((d.mean(0) * w).sum() / w.sum()))
        nfr = min(len(self.f.env), len(cf.env))
        w = self.w[:nfr]
        env_err = float((np.abs(np.maximum(self.f.env[:nfr], ref + FLOOR_DB)
                                - np.maximum(cf.env[:nfr], ref + FLOOR_DB)) * w).sum() / w.sum())
        # weight bands by how loud they are in the target: errors 60 dB down
        # are inaudible and dominated by interpolation/phase noise
        lt = np.maximum(self.f.ltas, -60.0)
        lc = np.maximum(cf.ltas, -60.0)
        tw = np.clip((lt + 60) / 60, 0.02, 1.0) ** 2
        tone_err = float((_dz(np.abs(lt - lc)) * tw).sum() / tw.sum())
        mov_err = float((np.abs(self.f.centroid[:nfr] - cf.centroid[:nfr]) * w).sum() / w.sum())
        # harmonic structure: high-res spectrum averaged over audible frames
        # (insensitive to unison beating / random phase, sensitive to waveform)
        hn = min(self.f.harm.shape[1], cf.harm.shape[1])
        fw = self.hw[:hn]
        cl = 10 * np.log10((10 ** (cf.harm[:, :hn] / 10) * fw).sum(1) / fw.sum() + 1e-18)
        hc = np.maximum(cl - cl.max(), FLOOR_DB)
        bw = np.clip((self.hl + 60) / 60, 0.02, 1.0) ** 2
        harm_err = float((_dz(np.abs(self.hl - hc)) * bw).sum() / bw.sum())
        parts = {
            "spectral": _sim(float(np.mean(spec_errs))),
            "harmonic": _sim(harm_err),
            "envelope": _sim(env_err),
            "tone": _sim(tone_err),
            # centroid is in mel bins; 1 bin ~ 1/64 of the range -> scale to dB-like
            "movement": _sim(mov_err * 1.5),
        }
        total = 100.0 * sum(WEIGHTS[k] * parts[k] for k in WEIGHTS)
        if not np.isfinite(total):
            total = 0.0
        if detail:
            return total, {k: round(100 * v, 1) for k, v in parts.items()}
        return total


def align(target: np.ndarray, cand: np.ndarray, sr: int, max_shift: float = 0.05) -> int:
    """Best integer-sample lag (cand should be delayed by the result)."""
    from scipy import signal
    m = int(max_shift * sr)
    a = target[: min(len(target), sr * 4)]
    b = cand[: len(a)]
    xc = signal.correlate(a, b, mode="full", method="fft")
    mid = len(b) - 1
    lo, hi = max(0, mid - m), min(len(xc), mid + m + 1)
    return int(np.argmax(xc[lo:hi]) + lo - mid)
