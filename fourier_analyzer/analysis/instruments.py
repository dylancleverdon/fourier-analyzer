"""Describe what is playing in each separated stem.

Labels come from the separation itself (each stem/drum part is one instrument
family) plus signal features: register, sustain, polyphony and noisiness,
which distinguish e.g. sub bass vs. growl bass, lead vs. pad vs. pluck.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import audio
from ..separation.stems import DRUM_LABELS

HOP = 512


def _activity(y, sr):
    n = len(y) // HOP
    if n == 0:
        return np.zeros(0), -120.0
    fr = y[: n * HOP].reshape(n, HOP)
    db = 10 * np.log10(np.mean(fr ** 2, axis=1) + 1e-12)
    return db, float(db.max())


def _segments(active, sr):
    segs, start = [], None
    for i, a in enumerate(active):
        if a and start is None:
            start = i
        if not a and start is not None:
            segs.append([round(start * HOP / sr, 3), round(i * HOP / sr, 3)])
            start = None
    if start is not None:
        segs.append([round(start * HOP / sr, 3), round(len(active) * HOP / sr, 3)])
    # merge gaps < 150 ms
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 0.15:
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    return merged


def describe_stem(name: str, path: Path, mix_peak_db: float) -> dict:
    import librosa
    y, sr = audio.load(path, sr=22050, mono=True)
    db, peak = _activity(y, sr)
    present = peak > mix_peak_db - 45 and len(db) > 0
    info = {"stem": name, "present": bool(present), "level_db": round(peak - mix_peak_db, 1)}
    if not present:
        info["label"] = _base_label(name)
        info["segments"] = []
        return info
    active = db > peak - 30
    info["segments"] = _segments(active, sr)
    info["active_ratio"] = round(float(active.mean()), 3)
    if name.startswith("drum_"):
        part = name[5:]
        onsets = librosa.onset.onset_detect(y=y, sr=sr, units="time", backtrack=True)
        info["label"] = DRUM_LABELS.get(part, part)
        info["hits"] = int(len(onsets))
        info["hit_times"] = [round(float(t), 3) for t in onsets[:400]]
        return info
    S = np.abs(librosa.stft(y, n_fft=4096, hop_length=HOP))
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    loud = np.zeros(S.shape[1], dtype=bool)
    m = min(len(db), S.shape[1])
    loud[:m] = db[:m] > peak - 25
    if not loud.any():
        loud = np.ones(S.shape[1], bool)
    spec = S[:, loud].mean(1)
    cent = float((spec * freqs).sum() / (spec.sum() + 1e-12))
    flat = float(np.mean(librosa.feature.spectral_flatness(S=S[:, loud])))
    # polyphony: average number of strong spectral peaks below 2 kHz that are
    # not harmonics of the lowest one
    peaks_per_frame = []
    for j in np.nonzero(loud)[0][::8]:
        col = S[: int(2000 / (sr / 4096)), j]
        thr = col.max() * 0.25
        pk = np.nonzero((col[1:-1] > col[:-2]) & (col[1:-1] > col[2:]) & (col[1:-1] > thr))[0] + 1
        if len(pk) == 0:
            continue
        f = freqs[pk]
        f0 = f[0]
        nonharm = [x for x in f if abs(x / f0 - round(x / f0)) > 0.08]
        peaks_per_frame.append(1 + len(nonharm) / 3)
    poly = float(np.median(peaks_per_frame)) if peaks_per_frame else 1.0
    seg_lens = [b - a for a, b in info["segments"]] or [0]
    sustain = float(np.median(seg_lens))
    info.update({"centroid_hz": round(cent, 1), "noisiness": round(flat, 3), "polyphony": round(poly, 2),
                 "median_note_len": round(sustain, 3)})
    info["label"] = _classify(name, cent, flat, poly, sustain)
    return info


def _base_label(name):
    return {"bass": "Bass", "other": "Synths / other", "vocals": "Vocals", "drums": "Drums"}.get(
        name, name.replace("_", " ").title())


def _classify(name, cent, flat, poly, sustain):
    if name == "vocals":
        return "Vocals / vocal chops"
    if name == "bass" or name.startswith("bass_layer"):
        if cent < 120 and flat < 0.02:
            return "Sub bass"
        if cent > 600 or flat > 0.05:
            return "Synth bass (growl / mid bass)"
        return "Synth bass"
    if name == "drums":
        return "Drums (full kit)"
    if name.startswith("other") or name.endswith("_layer1") or "layer" in name:
        if flat > 0.2:
            return "Noise / FX / riser"
        if poly >= 2.0 and sustain > 0.5:
            return "Pad / chords"
        if sustain < 0.25:
            return "Pluck / stab synth"
        return "Lead synth"
    return _base_label(name)


def describe_all(stems: dict[str, Path], mix_path: Path) -> list[dict]:
    y, sr = audio.load(mix_path, sr=22050, mono=True)
    _, mix_peak = _activity(y, sr)
    out = []
    for name, p in stems.items():
        try:
            out.append(describe_stem(name, Path(p), mix_peak))
        except Exception as e:     # never fail the whole project on one stem
            out.append({"stem": name, "present": False, "label": _base_label(name), "error": str(e)})
    return out
