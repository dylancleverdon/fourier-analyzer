"""Audio file I/O helpers."""
from __future__ import annotations

from pathlib import Path

import numpy as np


def load(path: str | Path, sr: int | None = 44100, mono: bool = True) -> tuple[np.ndarray, int]:
    """Load any common audio file. Returns (samples, sr); mono -> (n,), else (ch, n)."""
    import soundfile as sf
    try:
        data, file_sr = sf.read(str(path), dtype="float32", always_2d=True)
        data = data.T
    except Exception:
        import librosa  # falls back to audioread/ffmpeg for formats libsndfile lacks
        data, file_sr = librosa.load(str(path), sr=None, mono=False)
        data = np.atleast_2d(data).astype(np.float32)
    if mono:
        data = data.mean(axis=0)
    if sr is not None and file_sr != sr:
        import librosa
        data = librosa.resample(data, orig_sr=file_sr, target_sr=sr, res_type="soxr_hq")
        file_sr = sr
    return np.ascontiguousarray(data, dtype=np.float32), file_sr


def save(path: str | Path, y: np.ndarray, sr: int) -> Path:
    import soundfile as sf
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    y = np.asarray(y, dtype=np.float32)
    if y.ndim == 2 and y.shape[0] < y.shape[1]:
        y = y.T
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    if peak > 1.0:
        y = y / peak * 0.99
    sf.write(str(path), y, sr, subtype="PCM_24")
    return path


def resample(y: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return y
    import librosa
    return librosa.resample(y, orig_sr=sr_from, target_sr=sr_to, res_type="soxr_hq").astype(np.float32)


def rms_db(y: np.ndarray) -> float:
    return float(20 * np.log10(np.sqrt(np.mean(np.square(y))) + 1e-12))


def normalize(y: np.ndarray, peak: float = 0.9) -> np.ndarray:
    m = float(np.max(np.abs(y))) if y.size else 0.0
    return y if m == 0 else (y / m * peak).astype(np.float32)
