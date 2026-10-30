"""Keeping LFO wobble out of embedded samples.

A note cut from a wobble bass carries its LFO movement (loudness and filter)
inside the audio. Played back by a sample oscillator, that movement speeds up
and slows down with the played pitch, so every other note wobbles at the
wrong rate. Before a note is embedded, its loudness and spectral-envelope
movement are flattened; the wobble then lives in an LFO routing (synced to
the tempo), and candidates whose remaining wobble rate follows the pitch are
rejected.
"""
from __future__ import annotations

import numpy as np
from scipy import signal
from scipy.ndimage import uniform_filter1d


def _smooth_rms(y: np.ndarray, sr: int, win_s: float) -> np.ndarray:
    """Per-sample RMS over a Hann window (long enough to span several pitch
    periods, so the waveform itself does not ripple the measurement)."""
    w = np.hanning(max(8, int(win_s * sr)))
    w /= w.sum()
    return np.sqrt(np.maximum(signal.fftconvolve(y ** 2, w, mode="same"), 0.0)) + 1e-9


def flatten_movement(y: np.ndarray, sr: int, f0_hz: float, attack_s: float = 0.03) -> np.ndarray:
    """Remove time-varying loudness and filter (spectral-envelope) movement
    after the attack, keeping the harmonic content and the attack itself."""
    y = np.asarray(y, dtype=np.float64)
    if len(y) < 4096:
        return y
    nper = 2048
    hop = 256
    _, _, Z = signal.stft(y, sr, nperseg=nper, noverlap=nper - hop, boundary="even")
    mag = np.abs(Z) + 1e-12
    bin_hz = sr / nper
    # spectral envelope: magnitude smoothed across a few harmonics, so the
    # harmonic peaks stay and only the overall filter shape is measured
    width = int(max(3, min(nper // 8, 2.5 * max(f0_hz, 30.0) / bin_hz)))
    env = uniform_filter1d(mag, size=width, axis=0, mode="nearest")
    env = uniform_filter1d(env, size=3, axis=1, mode="nearest")     # a little in time too
    a0 = min(env.shape[1] - 1, int(attack_s * sr / hop) + 1)
    ref = np.median(env[:, a0:], axis=1, keepdims=True) if env.shape[1] > a0 + 1 else env.mean(axis=1, keepdims=True)
    gain = np.clip(ref / env, 0.003, 300.0)       # up to +-50 dB: a closed filter is deep
    ramp = np.clip((np.arange(env.shape[1]) - a0 + 2) / 3.0, 0.0, 1.0)  # attack untouched, then fade in
    Z2 = Z * (1.0 + (gain - 1.0) * ramp[None, :])
    _, y2 = signal.istft(Z2, sr, nperseg=nper, noverlap=nper - hop, boundary=True)
    y2 = np.asarray(y2[: len(y)], dtype=np.float64)
    if len(y2) < len(y):
        y2 = np.pad(y2, (0, len(y) - len(y2)))
    # loudness: whatever level movement is left after the attack
    rms = _smooth_rms(y2, sr, max(0.03, 2.5 / max(f0_hz, 30.0)))
    i0 = int(attack_s * sr)
    if len(y2) > i0 + sr // 20:
        target = float(np.median(rms[i0:]))
        g = np.clip(target / rms, 0.1, 10.0)
        ramp = np.clip((np.arange(len(y2)) - i0) / (0.01 * sr), 0.0, 1.0)
        y2 = y2 * (1.0 + (g - 1.0) * ramp)
    return y2


def short_loop(n_samples: int, sr: int, f0_hz: float, lfo_period_s: float | None,
               start_frac: float = 0.45) -> tuple[float, float]:
    """A loop shorter than half an LFO cycle (and at most 0.25 s), a whole
    number of pitch periods long, in the sustain. Returns (start, end) as
    fractions of the sample."""
    period = sr / max(f0_hz, 20.0)
    max_len = 0.25 * sr
    if lfo_period_s:
        max_len = min(max_len, 0.45 * lfo_period_s * sr)
    k = max(4, int(max_len // period))
    length = k * period
    start = start_frac * n_samples
    if start + length > n_samples * 0.97:
        start = max(0.0, n_samples * 0.97 - length)
    return start / n_samples, min(1.0, (start + length) / n_samples)


def modulation(y: np.ndarray, sr: int, t0: float = 0.08, min_hz: float = 0.8) -> dict | None:
    """The strongest periodic movement in a held note: rate (Hz) and depth
    (dB) of its loudness or brightness, whichever moves more."""
    y = np.asarray(y, dtype=np.float64)[int(t0 * sr):]
    hop = max(1, int(0.005 * sr))
    n = len(y) // hop
    if n < 60:
        return None
    loud = 20 * np.log10(_smooth_rms(y, sr, 0.04)[hop // 2:: hop][:n])
    if loud.max() < -80:
        return None
    nf = 2048
    _, _, Z = signal.stft(y, sr, nperseg=nf, noverlap=nf - hop * 2, boundary=None)
    mag = np.abs(Z)
    fr = np.fft.rfftfreq(nf, 1 / sr)[:, None]
    cent = (fr * mag).sum(axis=0) / (mag.sum(axis=0) + 1e-12)
    cent_db = 20 * np.log10(np.maximum(cent, 20.0))
    best = None
    for name, x, dt in (("loudness", loud, hop / sr), ("brightness", cent_db, 2 * hop / sr)):
        if len(x) < 40:
            continue
        x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        sp = np.abs(np.fft.rfft(x * np.hanning(len(x)), 8 * len(x)))
        f = np.fft.rfftfreq(8 * len(x), dt)
        m = (f >= min_hz) & (f < 25)
        if not m.any():
            continue
        i = int(np.argmax(np.where(m, sp, 0)))
        depth = float(np.percentile(x, 95) - np.percentile(x, 5))
        if best is None or depth > best["depth_db"]:
            best = {"feature": name, "rate_hz": float(f[i]), "depth_db": depth}
    return best


def pitch_tracking(render, pitches, duration: float, lfo_rate_hz: float | None,
                   min_depth_db: float = 1.5, tol: float = 0.15, wobble_hz: float | None = None) -> dict:
    """Render one held note at several pitches and compare the wobble rate.
    ``render(pitch, duration) -> mono array``. A candidate fails when the
    wobble rate changes with the pitch (by more than ``tol``) or, with a
    tempo-synced LFO, strays from the synced rate."""
    rows = []
    # look for wobble around the analysed LFO rate (a fifth down still shows a
    # baked wobble at 2/3 of it), not at slow drifts of the whole note
    hint = lfo_rate_hz or wobble_hz
    min_hz = 0.5 * hint if hint else 0.8
    for pt in pitches:
        y, sr = render(pt, duration)
        m = modulation(y, sr, min_hz=min_hz)
        rows.append({"pitch": float(pt), **(m or {"rate_hz": None, "depth_db": 0.0})})
    moving = [r for r in rows if r["rate_hz"] and r["depth_db"] >= min_depth_db]
    ok, why = True, "no wobble left in the sample"
    if len(moving) >= 2:
        rates = np.array([r["rate_hz"] for r in moving])
        spread = float((rates.max() - rates.min()) / max(rates.mean(), 1e-6))
        if spread > tol:
            ok, why = False, f"wobble rate follows the pitch ({', '.join(f'{r:.2f}' for r in rates)} Hz)"
        elif lfo_rate_hz:
            # harmonics of the LFO rate are fine (a square wobble has 2x, 3x ...)
            ratio = rates.mean() / lfo_rate_hz
            if abs(ratio - round(ratio)) > tol * max(1.0, round(ratio)) or round(ratio) < 1:
                ok, why = False, f"wobble at {rates.mean():.2f} Hz, not the synced {lfo_rate_hz:.2f} Hz"
            else:
                why = f"wobble stays at the synced rate ({rates.mean():.2f} Hz)"
        else:
            why = f"wobble rate steady across pitches ({rates.mean():.2f} Hz)"
    return {"ok": ok, "why": why, "pitches": rows}
