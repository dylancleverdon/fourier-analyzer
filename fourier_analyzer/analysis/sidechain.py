"""Sidechain (ducking) detection on a synth stem.

In bass music the synth bus is usually compressed by the kick: its level dips
right after every kick and recovers over ~50-300 ms. That dip is not part of
the Serum patch (it happens in the DAW), so it has to be found and removed
before the patch is fitted, otherwise the optimizer would try to explain it
with amp envelopes or LFOs.

Method (event-locked averaging):
1. Kick times come from the separated kick stem when available, otherwise
   from the beat grid (BPM + best phase).
2. For every kick that falls *inside* a sustained note (so a new note's attack
   cannot be mistaken for ducking), take the synth's loudness (dB, 2.9 ms hop)
   relative to the 30 ms before the kick.
3. The median of those curves is the ducking shape. It is fitted with a
   compressor gain-reduction model: linear attack to -depth dB, hold, then
   exponential release.
4. Detection requires the dip to be consistent across kicks (>= 2 dB, the
   per-kick dips agree in sign for 70 % of kicks) and at least 3 usable kicks.
"""
from __future__ import annotations

import numpy as np

HOP = 64


def kick_times_from_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Onsets in a kick stem (low band energy rise)."""
    import librosa
    from scipy import signal
    sos = signal.butter(4, 180, "low", fs=sr, output="sos")
    low = signal.sosfiltfilt(sos, y)
    env = librosa.onset.onset_strength(y=low, sr=sr, hop_length=128)
    on = librosa.onset.onset_detect(onset_envelope=env, sr=sr, hop_length=128, units="time",
                                    backtrack=True, delta=0.15)
    return np.asarray(on, dtype=float)


def kick_times_from_grid(y: np.ndarray, sr: int, bpm: float) -> np.ndarray:
    """Beat positions with the phase that best matches the onsets."""
    import librosa
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=128)
    fps = sr / 128
    per = fps * 60 / bpm
    best, best_ph = -1, 0.0
    for ph in np.linspace(0, per, 64, endpoint=False):
        pos = np.arange(ph, len(env) - 1, per)
        v = np.interp(pos, np.arange(len(env)), env).mean()
        if v > best:
            best, best_ph = v, ph
    return np.arange(best_ph, len(env) - 1, per) / fps


def _loud_db(y, sr):
    n = len(y) // HOP
    fr = y[: n * HOP].reshape(n, HOP)
    e = np.mean(fr ** 2, axis=1)
    # 3-frame smoothing (~9 ms) keeps attacks sharp but removes cycle ripple
    k = max(3, int(0.012 * sr / HOP))
    e = np.convolve(e, np.ones(k) / k, mode="same")
    return 10 * np.log10(e + 1e-12)


def duck_shape(tau: np.ndarray, depth_db, attack, hold, release) -> np.ndarray:
    """Gain reduction in dB (<= 0) at time tau after a kick."""
    g = np.zeros_like(tau)
    a = np.maximum(attack, 1e-4)
    on = tau >= 0
    ramp = np.clip(tau / a, 0, 1)
    rel = np.where(tau > a + hold, np.exp(-(tau - a - hold) / max(release, 1e-3)), 1.0)
    g[on] = -depth_db * ramp[on] * rel[on]
    return g


def estimate(y: np.ndarray, sr: int, notes, kicks: np.ndarray, source: str = "kick stem") -> dict:
    """Return the sidechain model (``detected`` False if there is none)."""
    out = {"detected": False, "source": source, "kicks": [round(float(k), 4) for k in kicks[:500]]}
    if len(kicks) < 3:
        out["reason"] = "fewer than 3 kicks in the selection"
        return out
    db = _loud_db(np.asarray(y, dtype=np.float64), sr)
    fps = sr / HOP
    win = 0.5
    taus = np.arange(0, int(win * fps)) / fps
    curves = []
    for i, tk in enumerate(kicks):
        nxt = kicks[i + 1] if i + 1 < len(kicks) else tk + win
        span = min(win, nxt - tk)
        if span < 0.08:
            continue
        # must sit inside one sustained note: note started >= 40 ms before the
        # kick and lasts >= 100 ms after it
        inside = any(n.start <= tk - 0.04 and n.start + n.duration >= tk + min(span, 0.1) for n in notes)
        if not inside:
            continue
        i0 = int((tk - 0.03) * fps)
        i1 = int(tk * fps)
        if i0 < 0 or i1 + len(taus) >= len(db):
            continue
        ref = np.median(db[i0:i1])
        c = db[i1:i1 + len(taus)] - ref
        c[taus > span] = np.nan
        curves.append(c)
    out["kicks_used"] = len(curves)
    if len(curves) < 3:
        out["reason"] = "not enough kicks land inside sustained notes to measure ducking"
        return out
    C = np.array(curves)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            med = np.nanmedian(C, axis=0)
    valid = np.isfinite(med)
    early = med[valid][: int(0.15 * fps)]
    dip = -float(np.nanmin(early)) if len(early) else 0.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_kick_dip = np.nanmin(C[:, : int(0.15 * fps)], axis=1)
    consistency = float(np.mean(per_kick_dip < -1.0))
    out.update({"dip_db": round(dip, 2), "consistency": round(consistency, 2),
                "curve_db": [None if not np.isfinite(v) else round(float(v), 2) for v in med[::4]],
                "curve_step_ms": round(4000 / fps, 2)})
    if dip < 2.0 or consistency < 0.7:
        out["reason"] = f"no consistent dip after kicks (median dip {dip:.1f} dB, {consistency:.0%} of kicks)"
        return out
    # fit the compressor model by grid search (robust, no local minima issues)
    t = taus[valid]
    m = med[valid]
    best = (1e18, None)
    for attack in (0.001, 0.003, 0.006, 0.012, 0.025):
        for hold in (0.0, 0.01, 0.025, 0.05, 0.08, 0.12, 0.18):
            for release in np.geomspace(0.02, 0.6, 18):
                shape = duck_shape(t, 1.0, attack, hold, release)
                denom = (shape ** 2).sum()
                if denom <= 0:
                    continue
                depth = max(0.0, float((shape * m).sum() / denom))
                err = ((m - depth * shape) ** 2).sum()
                if err < best[0]:
                    best = (err, (depth, attack, hold, release))
    depth, attack, hold, release = best[1]
    resid = float(np.sqrt(best[0] / len(t)))
    out.update({"detected": True, "depth_db": round(depth, 2), "attack_ms": round(attack * 1000, 1),
                "hold_ms": round(hold * 1000, 1), "release_ms": round(float(release) * 1000, 1),
                "fit_rms_db": round(resid, 2)})
    return out


def gain_curve(model: dict, n: int, sr: int, offset: float = 0.0) -> np.ndarray:
    """Per-sample linear gain implied by the model (1.0 where no ducking).
    ``offset`` shifts kick times (seconds) when rendering a sub-range."""
    g_db = np.zeros(n)
    if not model or not model.get("detected"):
        return np.ones(n)
    t = np.arange(n) / sr
    d, a = model["depth_db"], model["attack_ms"] / 1000
    h, r = model["hold_ms"] / 1000, model["release_ms"] / 1000
    for k in model["kicks"]:
        k = k - offset
        if k > n / sr or k + a + h + 6 * r < 0:
            continue
        i0 = max(0, int(k * sr))
        i1 = min(n, int((k + a + h + 6 * r) * sr))
        if i1 <= i0:
            continue
        seg = duck_shape(t[i0:i1] - k, d, a, h, r)
        g_db[i0:i1] = np.minimum(g_db[i0:i1], seg)
    return 10 ** (g_db / 20)


def logic_settings(model: dict) -> str:
    if not model or not model.get("detected"):
        return ""
    return (f"Logic Compressor on the synth track, Side Chain = kick track: "
            f"Attack {model['attack_ms']:.0f} ms, Release {model['hold_ms'] + model['release_ms']:.0f} ms, "
            f"ratio 8:1 or higher, lower Threshold until gain reduction peaks near "
            f"{model['depth_db']:.1f} dB on each kick.")


def add_to_patch(patch, model: dict, bpm: float, enabled: bool = False) -> str | None:
    """Put the measured ducking into the preset as a tempo-synced LFO on
    Serum's Voice Amp (the classic "LFO sidechain"). The mod slot is written
    bypassed unless ``enabled``: Serum users can switch it on instead of a
    sidechain compressor in the DAW, without ducking twice. Returns a short
    description, or None when there is no regular kick pattern."""
    from ..synth.patch import LFO, Mod, SYNC_DIVISIONS, sync_to_hz
    if not model or not model.get("detected") or len(model.get("kicks", [])) < 3:
        return None
    kicks = np.asarray(model["kicks"], dtype=float)
    period = float(np.median(np.diff(kicks)))
    best, err = None, 1e9
    for d in SYNC_DIVISIONS:
        p = 1.0 / sync_to_hz(d, bpm)
        e = abs(p - period) / period
        if e < err:
            best, err = d, e
    if best is None or err > 0.08:
        return None                    # kicks are not on a regular grid: leave it to the DAW
    period = 1.0 / sync_to_hz(best, bpm)
    xs = np.linspace(0, 1, 25)
    g = 10 ** (duck_shape(xs * period, model["depth_db"], model["attack_ms"] / 1000,
                          model["hold_ms"] / 1000, model["release_ms"] / 1000) / 20)
    duck = 1 - g
    depth = float(duck.max())
    if depth <= 0.01:
        return None
    pts = [[float(x), float(v / depth), 0.5] for x, v in zip(xs, duck)]
    used = {m.source for m in patch.mods} | {m.aux for m in patch.mods if m.aux}
    slot = next((k for k in range(len(patch.lfos) - 1, -1, -1) if f"lfo{k + 1}" not in used), None)
    if slot is None:
        return None
    phase = float((-kicks[0] / period) % 1.0)
    patch.lfos[slot] = LFO(shape="custom", sync=best, mode="free", anchored=True, points=pts, phase=phase)
    patch.mods.append(Mod(f"lfo{slot + 1}", "voiceamp", -0.5 * depth, bypass=not enabled))
    state = "on" if enabled else "bypassed (enable the mod slot in Serum's MATRIX)"
    return f"sidechain as LFO{slot + 1} {best} -> Voice Amp, -{model['depth_db']:.1f} dB, {state}"
