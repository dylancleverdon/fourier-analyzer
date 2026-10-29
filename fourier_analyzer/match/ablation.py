"""Verification checks ("check your work").

Two complementary ideas:

* Decompose the *original*: split it into a tonal part (STFT bins on the
  harmonic series) and a noise residual, then compare those against the patch
  rendered with only its oscillators / only its noise. If the patch's tonal
  core matches the de-noised original, the noise was attributed correctly.
* Ablate the *patch*: remove one component at a time (noise, distortion, an
  LFO, unison, ...) and measure how much the score drops. Components whose
  removal costs nothing are flagged as unjustified.

Plus timing (onset error in ms) and pitch (cents) checks per note.
"""
from __future__ import annotations

import numpy as np

from ..synth.engine import Renderer
from ..synth.patch import Patch
from .score import Target

N_FFT = 4096
HOP = 256


def harmonic_split(y: np.ndarray, sr: int, f0: np.ndarray, width_bins: int = 2):
    """Return (tonal, noise) parts of y using a mask on the f0 harmonic series."""
    import librosa
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP)
    nb, nfr = S.shape
    mask = np.zeros(S.shape, dtype=np.float32)
    binhz = sr / N_FFT
    for i in range(min(nfr, len(f0))):
        f = f0[i]
        if not np.isfinite(f) or f <= 0:
            continue
        ks = np.arange(1, int((sr / 2) / f))
        bins = np.round(ks * f / binhz).astype(int)
        for d in range(-width_bins, width_bins + 1):
            b = np.clip(bins + d, 0, nb - 1)
            mask[b, i] = 1.0
    tonal = librosa.istft(S * mask, hop_length=HOP, length=len(y))
    return tonal.astype(np.float32), (y - tonal).astype(np.float32)


def _energy(y):
    return float(np.mean(np.square(y)) + 1e-12)


def run_checks(patch: Patch, y_target: np.ndarray, sr: int, notes, f0: np.ndarray,
               renderer: Renderer, target: Target | None = None, gain=None) -> list[dict]:
    """``f0`` must be on the same hop (256) as y_target at ``sr``. ``gain`` is
    the measured sidechain gain applied to every render (DAW context)."""
    dur = len(y_target) / sr
    target = target or Target(y_target, sr)
    _render = renderer.render

    class _R:
        sr = renderer.sr

        @staticmethod
        def render(p, nts, d, solo=None):
            out = _render(p, nts, d, solo=solo)
            if gain is not None:
                g = gain[: len(out)]
                out = out[: len(g)] * g
            return out
    renderer = _R
    full = renderer.render(patch, notes, dur)
    base = target.score(full)
    checks = []

    # 1. noise attribution ----------------------------------------------------
    tonal, noise = harmonic_split(y_target, sr, f0)
    t_ratio = _energy(noise) / (_energy(noise) + _energy(tonal))
    osc_only = renderer.render(patch, notes, dur, solo="oscs")
    p_tonal, p_noise = harmonic_split(full, sr, f0)
    p_ratio = _energy(p_noise) / (_energy(p_noise) + _energy(p_tonal))
    tonal_score = Target(tonal, sr).score(osc_only) if np.any(osc_only) else 0.0
    ok = abs(t_ratio - p_ratio) < 0.12
    checks.append({
        "name": "Noise attribution",
        "detail": (f"original is {t_ratio*100:.0f}% non-harmonic energy, patch is {p_ratio*100:.0f}%. "
                   f"Patch oscillators alone vs original with noise removed: {tonal_score:.1f}"),
        "status": "pass" if ok else "warn",
    })

    # 2. component ablations ---------------------------------------------------
    def ablate(label, fn):
        q = patch.copy()
        fn(q)
        s = target.score(renderer.render(q, notes, dur))
        drop = base - s
        checks.append({
            "name": f"Remove {label}",
            "detail": f"score {base:.1f} -> {s:.1f} ({-drop:+.1f})",
            "status": "pass" if drop > 0.3 else "warn",
            "note": "needed" if drop > 0.3 else "contributes little; consider removing",
        })

    if patch.noise.enabled:
        ablate("noise", lambda q: setattr(q.noise, "enabled", False))
    if patch.distortion.enabled:
        ablate(f"distortion ({patch.distortion.mode})", lambda q: setattr(q.distortion, "enabled", False))
    if patch.ott.enabled:
        ablate("OTT", lambda q: setattr(q.ott, "enabled", False))
    if patch.oscs[0].unison > 1:
        ablate("unison", lambda q: setattr(q.oscs[0], "unison", 1))
    for i in (1, 2):
        if patch.oscs[i].enabled:
            ablate(f"oscillator {'ABC'[i]}", lambda q, i=i: setattr(q.oscs[i], "enabled", False))
    if patch.sub.enabled:
        ablate("sub", lambda q: setattr(q.sub, "enabled", False))
    for j, m in enumerate(patch.mods):
        if m.amount:
            ablate(f"{m.source} -> {m.dest}", lambda q, j=j: setattr(q.mods[j], "amount", 0.0))

    # 3. distortion evidence in the original ----------------------------------
    if patch.filter.enabled:
        import librosa
        S = np.abs(librosa.stft(y_target, n_fft=2048, hop_length=512)) ** 2
        fr = librosa.fft_frequencies(sr=sr, n_fft=2048)
        hi = S[fr > patch.filter.cutoff * 4].sum() / (S.sum() + 1e-12)
        checks.append({
            "name": "Energy above the filter",
            "detail": (f"{hi*100:.1f}% of the original's energy sits >2 octaves above the cutoff "
                       f"({patch.filter.cutoff:.0f} Hz); a filter alone removes most of that, so a "
                       f"large value means distortion/FX after the filter"),
            "status": "pass" if (hi < 0.05) or patch.distortion.enabled or patch.ott.enabled else "warn",
        })

    # 4. timing: same note detector on both signals ----------------------------
    from ..analysis.features import analyze
    try:
        rn = analyze(full, sr, bpm=patch.bpm).notes
        ot = np.array([n.start for n in notes])
        oc = np.array([n.start for n in rn])
        if len(ot) and len(oc):
            errs = [float(np.min(np.abs(oc - t))) for t in ot]
            ms = 1000 * float(np.median(errs))
            checks.append({"name": "Timing", "detail": f"median note-start error {ms:.1f} ms over {len(ot)} notes "
                                                        f"({len(rn)} detected in the patch render)",
                           "status": "pass" if ms < 12 and abs(len(rn) - len(ot)) <= max(1, len(ot) // 5) else "warn"})
    except Exception as e:      # analysis of a degenerate render
        checks.append({"name": "Timing", "detail": f"could not analyse render: {e}", "status": "warn"})

    # 5. pitch ----------------------------------------------------------------
    from ..analysis.features import _frames_stft, track_f0
    f_r, _ = track_f0(_frames_stft(full, sr), sr)
    n = min(len(f_r), len(f0))
    both = np.isfinite(f_r[:n]) & np.isfinite(f0[:n])
    if both.sum() > 5:
        cents = 1200 * np.abs(np.log2(f_r[:n][both] / f0[:n][both]))
        cents = np.minimum(cents, np.abs(cents - 1200))     # ignore octave errors of the tracker
        med = float(np.median(cents))
        checks.append({"name": "Pitch", "detail": f"median pitch error {med:.1f} cents",
                       "status": "pass" if med < 15 else "warn"})
    return checks
