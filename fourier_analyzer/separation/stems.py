"""Source separation.

Stage A: full mix -> drums / bass / other / vocals (Demucs v4 htdemucs_ft)
Stage B: drums -> kick / snare / toms / hh / ride / crash (MDX23C DrumSep)
Both models run through the ``audio-separator`` package (PyTorch, uses the
Apple-silicon GPU via MPS when available). If the ML stack is missing a DSP
fallback keeps the app usable (lower quality).
"""
from __future__ import annotations

import logging
import shutil
from pathlib import Path
from typing import Callable

import numpy as np

from .. import audio, config

log = logging.getLogger(__name__)

STEM_MODEL = "htdemucs_ft.yaml"
DRUM_MODEL = "MDX23C-DrumSep-aufr33-jarredou.ckpt"
STEM_NAMES = ("drums", "bass", "other", "vocals")
DRUM_NAMES = ("kick", "snare", "toms", "hh", "ride", "crash")
DRUM_LABELS = {"kick": "Kick", "snare": "Snare", "toms": "Toms", "hh": "Hi-hat", "ride": "Ride",
               "crash": "Crash"}


def ml_available() -> bool:
    import os
    if os.environ.get("FA_FORCE_DSP"):
        return False
    try:
        import audio_separator  # noqa: F401
        return True
    except Exception:
        return False


def _separator(out_dir: Path):
    from audio_separator.separator import Separator
    return Separator(output_dir=str(out_dir), model_file_dir=str(config.models_dir()),
                     output_format="WAV", sample_rate=44100, log_level=logging.WARNING)


def _collect(out_dir: Path, files, names) -> dict[str, Path]:
    found = {}
    paths = [Path(f) if Path(f).is_absolute() else out_dir / f for f in files]
    for p in paths:
        low = p.name.lower()
        for n in names:
            if f"({n})" in low or f"_{n}_" in low or low.endswith(f"_{n}.wav"):
                found[n] = p
    return found


def separate_ml(src: Path, out_dir: Path, model: str, names, progress=None) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    sep = _separator(out_dir)
    if progress:
        progress(f"loading model {model}")
    sep.load_model(model_filename=model)
    if progress:
        progress(f"separating with {model}")
    files = sep.separate(str(src))
    found = _collect(out_dir, files, names)
    result = {}
    for n, p in found.items():
        dst = out_dir / f"{n}.wav"
        if p != dst:
            shutil.move(str(p), dst)
        result[n] = dst
    return result


# ----------------------------------------------------------------------------
# DSP fallbacks


def _bandsplit(y, sr, lo, hi):
    from scipy import signal
    sos = []
    if lo:
        sos.append(signal.butter(4, lo, "high", fs=sr, output="sos"))
    if hi and hi < sr / 2:
        sos.append(signal.butter(4, hi, "low", fs=sr, output="sos"))
    for s in sos:
        y = signal.sosfiltfilt(s, y)
    return y


def separate_dsp(src: Path, out_dir: Path, progress=None) -> dict[str, Path]:
    """HPSS-based fallback: percussive -> drums, harmonic low band -> bass."""
    import librosa
    out_dir.mkdir(parents=True, exist_ok=True)
    y, sr = audio.load(src, sr=44100, mono=True)
    if progress:
        progress("DSP fallback separation (install ML models for better results)")
    H, P = librosa.effects.hpss(y, margin=(1.0, 3.0))
    bass = _bandsplit(H, sr, None, 250)
    other = H - bass
    res = {}
    for n, s in (("drums", P), ("bass", bass), ("other", other)):
        res[n] = audio.save(out_dir / f"{n}.wav", s, sr)
    return res


def separate_drums_dsp(src: Path, out_dir: Path) -> dict[str, Path]:
    y, sr = audio.load(src, sr=44100, mono=True)
    parts = {"kick": _bandsplit(y, sr, None, 150), "snare": _bandsplit(y, sr, 150, 5000),
             "hh": _bandsplit(y, sr, 6000, None)}
    return {n: audio.save(out_dir / f"{n}.wav", s, sr) for n, s in parts.items()}


# ----------------------------------------------------------------------------


def separate_all(src: Path, out_dir: Path, progress: Callable[[str], None] | None = None,
                 drums: bool = True) -> dict[str, Path]:
    """Full pipeline. Returns {stem_name: wav_path} including drum parts."""
    out_dir = Path(out_dir)
    stems: dict[str, Path] = {}
    use_ml = ml_available()
    if use_ml:
        try:
            stems = separate_ml(src, out_dir, STEM_MODEL, STEM_NAMES, progress)
        except Exception as e:
            log.exception("ML separation failed, falling back to DSP")
            if progress:
                progress(f"ML separation failed ({e}); using DSP fallback")
            use_ml = False
    if not stems:
        stems = separate_dsp(src, out_dir, progress)
    if drums and "drums" in stems:
        ddir = out_dir / "drums_parts"
        parts = {}
        if use_ml:
            try:
                parts = separate_ml(stems["drums"], ddir, DRUM_MODEL, DRUM_NAMES, progress)
            except Exception as e:
                log.exception("drum separation failed")
                if progress:
                    progress(f"drum model failed ({e}); using band-split fallback")
        if not parts:
            parts = separate_drums_dsp(stems["drums"], ddir)
        for n, p in parts.items():
            stems[f"drum_{n}"] = p
    return stems


def prefetch_models(progress=None) -> list[str]:
    """Download both models into the models dir (used by the installer)."""
    done = []
    if not ml_available():
        return done
    tmp = config.cache_dir() / "prefetch"
    tmp.mkdir(parents=True, exist_ok=True)
    for m in (STEM_MODEL, DRUM_MODEL):
        sep = _separator(tmp)
        if progress:
            progress(f"downloading {m}")
        sep.load_model(model_filename=m)
        done.append(m)
    return done


def split_layers(src: Path, out_dir: Path, n_layers: int = 2, start: float | None = None,
                 end: float | None = None) -> dict[str, Path]:
    """Split one stem (e.g. bass) into N layers with NMF.

    Components are clustered by when they are active and by register, then
    each layer is rebuilt with a soft (Wiener) mask. Works when the layers
    differ in timing, pitch range or timbre; heavily overlapping layers of the
    same register cannot be fully separated by any current method."""
    import librosa
    from sklearn.decomposition import NMF  # type: ignore
    y, sr = audio.load(src, sr=44100, mono=True)
    if start is not None or end is not None:
        a = int((start or 0) * sr)
        b = int((end or len(y) / sr) * sr)
        y = y[a:b]
    S = librosa.stft(y, n_fft=4096, hop_length=512)
    M = np.abs(S)
    k = max(4, n_layers * 6)
    model = NMF(n_components=k, init="nndsvda", max_iter=400, beta_loss="kullback-leibler",
                solver="mu", random_state=0)
    W = model.fit_transform(M)
    Hm = model.components_
    freqs = librosa.fft_frequencies(sr=sr, n_fft=4096)
    cent = (W * freqs[:, None]).sum(0) / (W.sum(0) + 1e-12)
    act = Hm / (Hm.max(1, keepdims=True) + 1e-12)
    feats = np.hstack([np.log2(cent + 1)[:, None] * 2.0, act[:, :: max(1, act.shape[1] // 200)]])
    from sklearn.cluster import KMeans  # type: ignore
    labels = KMeans(n_clusters=n_layers, n_init=10, random_state=0).fit_predict(feats)
    # order layers low -> high register
    order = np.argsort([cent[labels == c].mean() if np.any(labels == c) else 1e9 for c in range(n_layers)])
    V = W @ Hm + 1e-10
    out = {}
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = Path(src).stem
    for rank, c in enumerate(order):
        sel = labels == c
        if not sel.any():
            continue
        mask = (W[:, sel] @ Hm[sel]) / V
        yc = librosa.istft(S * mask, hop_length=512, length=len(y))
        out[f"{stem}_layer{rank + 1}"] = audio.save(out_dir / f"{stem}_layer{rank + 1}.wav", yc, sr)
    return out
