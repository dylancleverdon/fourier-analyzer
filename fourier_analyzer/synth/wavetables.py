"""Wavetables: built-in basic shapes, Serum wavetable .wav loading, a searchable
index of the user's stock Serum 2 tables, and band-limited mipmaps for the
renderer."""
from __future__ import annotations

import logging
import os
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config

log = logging.getLogger(__name__)

FRAME = 2048
N_HARM_FEAT = 64          # harmonics used for table matching
N_MIP = 11                # mip level l keeps harmonics <= 1024 >> l


def _from_harmonics(mags, phases=None, normalize: bool = True) -> np.ndarray:
    spec = np.zeros(FRAME // 2 + 1, dtype=np.complex128)
    mags = np.asarray(mags, dtype=float)
    n = min(len(mags), FRAME // 2)
    ph = np.zeros(n) if phases is None else np.asarray(phases[:n], dtype=float)
    spec[1:n + 1] = mags[:n] * np.exp(1j * ph)
    w = np.fft.irfft(spec, FRAME)
    if not normalize:
        return w.astype(np.float32)
    peak = np.max(np.abs(w))
    return (w / peak if peak > 0 else w).astype(np.float32)


def _basic(name: str) -> np.ndarray:
    k = np.arange(1, FRAME // 2 + 1)
    if name == "sine":
        frames = [_from_harmonics([1.0])]
    elif name == "saw":
        frames = [_from_harmonics(1.0 / k, phases=np.full(k.size, -np.pi / 2))]
    elif name == "square":
        m = np.where(k % 2 == 1, 1.0 / k, 0.0)
        frames = [_from_harmonics(m, phases=np.full(k.size, -np.pi / 2))]
    elif name == "triangle":
        m = np.where(k % 2 == 1, 1.0 / k ** 2, 0.0)
        ph = np.where((k // 2) % 2 == 0, -np.pi / 2, np.pi / 2)
        frames = [_from_harmonics(m, phases=ph)]
    elif name == "shapes":   # sine -> triangle -> saw -> square morph
        frames = [_basic(n)[0] for n in ("sine", "triangle", "saw", "square")]
    elif name == "pwm":      # pulse width sweep 50% -> 5%
        frames = []
        for w in np.linspace(0.5, 0.05, 16):
            m = 2.0 / (np.pi * k) * np.sin(np.pi * k * w)
            frames.append(_from_harmonics(np.abs(m), phases=np.where(m >= 0, -np.pi / 2, np.pi / 2)))
    else:
        raise KeyError(name)
    return np.stack(frames).astype(np.float32)


BASIC_TABLES = ("sine", "saw", "square", "triangle", "shapes", "pwm")


def read_serum_wav(path: str | Path) -> np.ndarray:
    """Load a Serum-style wavetable .wav into (frames, 2048)."""
    import soundfile as sf
    data, sr = sf.read(str(path), dtype="float32", always_2d=True)
    data = data[:, 0]
    size = _clm_frame_size(path) or FRAME
    n = len(data) // size
    if n == 0:
        # single-cycle shorter than a frame: resample to 2048
        x = np.interp(np.linspace(0, len(data), FRAME, endpoint=False), np.arange(len(data)), data)
        return x[None, :].astype(np.float32)
    frames = data[: n * size].reshape(n, size)
    if size != FRAME:
        xs = np.linspace(0, size, FRAME, endpoint=False)
        frames = np.stack([np.interp(xs, np.arange(size), f, period=size) for f in frames])
    return frames.astype(np.float32)


def _clm_frame_size(path) -> int | None:
    try:
        with open(path, "rb") as fh:
            head = fh.read(12)
            if head[:4] != b"RIFF":
                return None
            while True:
                hdr = fh.read(8)
                if len(hdr) < 8:
                    return None
                cid, size = hdr[:4], struct.unpack("<I", hdr[4:])[0]
                if cid == b"clm ":
                    txt = fh.read(size).decode("latin-1")
                    if txt.startswith("<!>"):
                        return int(txt[3:7])
                    return None
                fh.seek(size + (size & 1), 1)
    except (OSError, ValueError):
        return None


def write_serum_wav(path: str | Path, frames: np.ndarray, sr: int = 44100) -> None:
    """Write frames as a 32-bit float wav with Serum's clm chunk."""
    frames = np.asarray(frames, dtype="<f4")
    data = frames.ravel().tobytes()
    clm = b"<!>2048 20000000 wavetable (www.xferrecords.com)"
    if len(clm) % 2:
        clm += b"\0"
    fmt = struct.pack("<HHIIHH", 3, 1, sr, sr * 4, 4, 32)
    body = (b"WAVE" + b"fmt " + struct.pack("<I", len(fmt)) + fmt
            + b"clm " + struct.pack("<I", len(clm)) + clm
            + b"data" + struct.pack("<I", len(data)) + data)
    Path(path).write_bytes(b"RIFF" + struct.pack("<I", len(body)) + body)


class TableBank:
    """Resolves table ids to frame arrays and caches mipmaps."""

    def __init__(self, serum_root: str | Path | None = None):
        self.serum_root = Path(serum_root) if serum_root else config.serum_tables_dir()
        self.custom: dict[str, np.ndarray] = {}
        self._mip_cache: dict[tuple, np.ndarray] = {}

    def frames(self, table_id: str) -> np.ndarray:
        kind, _, name = table_id.partition(":")
        if kind == "basic":
            return _basic_cached(name)
        if kind == "custom":
            return self.custom[table_id]
        if kind == "serum":
            if not self.serum_root:
                raise FileNotFoundError("Serum tables folder not found")
            return _serum_cached(str(self.serum_root / name))
        raise KeyError(table_id)

    def add_custom(self, frames: np.ndarray) -> str:
        tid = f"custom:{len(self.custom)}"
        self.custom[tid] = np.asarray(frames, dtype=np.float32)
        return tid

    def mip(self, table_id: str, level: int) -> np.ndarray:
        """Band-limited copy of the table keeping harmonics <= 1024 >> level."""
        key = (table_id, level)
        m = self._mip_cache.get(key)
        if m is None:
            frames = self.frames(table_id)
            spec = np.fft.rfft(frames, axis=1)
            keep = max(1, (FRAME // 2) >> level)
            spec[:, keep + 1:] = 0
            spec[:, 0] = 0      # remove DC
            m = np.fft.irfft(spec, FRAME, axis=1).astype(np.float32)
            self._mip_cache[key] = m
        return m


@lru_cache(maxsize=None)
def _basic_cached(name):
    return _basic(name)


@lru_cache(maxsize=64)
def _serum_cached(path):
    return read_serum_wav(path)


def harmonic_features(frames: np.ndarray, n: int = N_HARM_FEAT) -> np.ndarray:
    """Per-frame harmonic magnitudes in dB, normalised to the loudest harmonic."""
    spec = np.abs(np.fft.rfft(frames, axis=1))[:, 1:n + 1]
    db = 20 * np.log10(spec + 1e-9)
    return db - db.max(axis=1, keepdims=True)


class StockIndex:
    """Harmonic-feature index over every wavetable in the Serum tables folder."""

    def __init__(self, root: Path | None = None):
        self.root = Path(root) if root else config.serum_tables_dir()
        self.ids: list[str] = []
        self.frame_owner: np.ndarray = np.zeros(0, int)
        self.frame_pos: np.ndarray = np.zeros(0)
        self.feats: np.ndarray = np.zeros((0, N_HARM_FEAT))

    def build(self, progress=None) -> "StockIndex":
        cache = config.cache_dir() / "stock_index.npz"
        files = sorted(self.root.rglob("*.wav")) if self.root and self.root.exists() else []
        sig = f"{self.root}|{len(files)}|{sum(int(f.stat().st_mtime) for f in files)}"
        if cache.exists():
            z = np.load(cache, allow_pickle=True)
            if str(z["sig"]) == sig:
                self.ids = list(z["ids"])
                self.frame_owner, self.frame_pos, self.feats = z["owner"], z["pos"], z["feats"]
                return self
        ids, owner, pos, feats = [], [], [], []
        for i, f in enumerate(files):
            try:
                fr = read_serum_wav(f)
            except Exception as e:      # unreadable / not a wavetable
                log.debug("skip %s: %s", f, e)
                continue
            # subsample long tables to at most 32 frames for the index
            sel = np.unique(np.linspace(0, len(fr) - 1, min(len(fr), 32)).round().astype(int))
            tid = "serum:" + str(f.relative_to(self.root))
            ids.append(tid)
            owner += [len(ids) - 1] * len(sel)
            pos += list(sel / max(1, len(fr) - 1))
            feats.append(harmonic_features(fr[sel]))
            if progress and i % 25 == 0:
                progress(i / max(1, len(files)))
        self.ids = ids
        self.frame_owner = np.array(owner, int)
        self.frame_pos = np.array(pos, float)
        self.feats = np.concatenate(feats) if feats else np.zeros((0, N_HARM_FEAT))
        try:
            np.savez(cache, sig=sig, ids=np.array(ids, dtype=object), owner=self.frame_owner,
                     pos=self.frame_pos, feats=self.feats)
        except OSError:
            pass
        return self

    def search(self, target_db: np.ndarray, k: int = 8, extra: dict | None = None):
        """Rank tables by how well any frame matches ``target_db`` (harmonic dB
        profile) after allowing a low-pass tilt. Returns [(table_id, pos, err)]."""
        cand_feats = [self.feats]
        cand_ids = [(self.ids[o], p) for o, p in zip(self.frame_owner, self.frame_pos)]
        if extra:
            for tid, fr in extra.items():
                hf = harmonic_features(fr)
                cand_feats.append(hf)
                cand_ids += [(tid, j / max(1, len(fr) - 1)) for j in range(len(fr))]
        F = np.concatenate(cand_feats) if cand_feats else np.zeros((0, N_HARM_FEAT))
        if len(F) == 0:
            return []
        err = lowpass_fit_error(F, target_db)
        best: dict[str, tuple] = {}
        for (tid, p), e in zip(cand_ids, err):
            if tid not in best or e < best[tid][1]:
                best[tid] = (p, e)
        ranked = sorted(((t, p, e) for t, (p, e) in best.items()), key=lambda x: x[2])
        return ranked[:k]


def lowpass_fit_error(F: np.ndarray, target_db: np.ndarray) -> np.ndarray:
    """Distance between candidate harmonic profiles F (n, H) and the target,
    minimised over a family of low-pass slopes (the patch filter can tilt the
    oscillator's spectrum, so raw distance would be unfair)."""
    H = min(F.shape[1], len(target_db))
    F = F[:, :H]
    t = np.asarray(target_db[:H])
    h = np.arange(1, H + 1)
    cut_h = np.array([1.5, 2.5, 4, 6, 10, 16, 25, 40, 1e6])
    best = np.full(len(F), np.inf)
    w = np.clip(t + 60, 0, 60) / 60 + 0.05        # weight audible harmonics
    for c in cut_h:
        for slope in (12, 24):
            lp = -slope * np.log2(np.maximum(1.0, h / c))
            cand = np.maximum(F + lp, -80)
            cand = cand - cand.max(axis=1, keepdims=True)
            e = (np.abs(cand - np.maximum(t, -80)) * w).sum(1) / w.sum()
            best = np.minimum(best, e)
    return best


def custom_table_from_harmonics(frames_mag: np.ndarray, frames_phase: np.ndarray | None = None,
                                max_frames: int = 256) -> np.ndarray:
    """Build a wavetable (n, 2048) from per-frame harmonic magnitudes (linear)."""
    frames_mag = np.atleast_2d(frames_mag)[:max_frames]
    out = []
    for i, m in enumerate(frames_mag):
        ph = None if frames_phase is None else frames_phase[i]
        if ph is None:
            ph = np.full(len(m), -np.pi / 2)
        out.append(_from_harmonics(m, ph, normalize=False))
    tab = np.stack(out)
    peak = float(np.max(np.abs(tab)))      # one gain for the whole table
    return (tab / peak if peak > 0 else tab).astype(np.float32)


def resolve_serum_ref(table_id: str) -> str | None:
    """Relative path Serum stores in relativePathToWT for a stock table."""
    if table_id.startswith("serum:"):
        return table_id[len("serum:"):].replace(os.sep, "/")
    return None
