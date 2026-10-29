"""Audio sources for the Sample, Granular, Spectral, Multisample and Noise
oscillators and the Convolve effect.

Sources are addressed by id:

* ``embedded:<n>``   audio stored inside a preset (or created by the optimizer)
* ``serum:<rel>``    a file from Serum's content folders (Samples, Multisamples,
                     Noises, IRs ...). Serum stores paths relative to the
                     folder of the content type, so files are found by their
                     relative path under the "Serum 2 Presets" folder.
* ``file:<path>``    any audio file on disk

Factory content only exists where Serum 2 is installed (the Mac); missing
files render as silence and are listed in ``SampleBank.missing``.
"""
from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import numpy as np

from .. import config

AUDIO_EXT = (".wav", ".flac", ".aif", ".aiff", ".ogg", ".mp3")


def serum_content_root() -> Path | None:
    """The "Serum 2 Presets" folder (parent of Tables)."""
    custom = config.settings().get("serum_content_dir")
    if custom and Path(custom).expanduser().exists():
        return Path(custom).expanduser()
    t = config.serum_tables_dir()
    return t.parent if t else None


@lru_cache(maxsize=4)
def _index(root: str) -> dict:
    """lower-case relative path (with / separators) -> absolute path, for every
    audio/sfz file under the content root."""
    out = {}
    for dirpath, _dirs, files in os.walk(root):
        for fn in files:
            if fn.lower().endswith(AUDIO_EXT + (".sfz",)):
                full = os.path.join(dirpath, fn)
                rel = os.path.relpath(full, root).replace(os.sep, "/").lower()
                out[rel] = full
    return out


def find_content(rel: str, root: Path | None = None) -> Path | None:
    """Locate a file Serum references by a relative path (any content folder,
    either slash style)."""
    root = root or serum_content_root()
    if not rel:
        return None
    rel_n = rel.replace("\\", "/").lstrip("/").lower()
    p = Path(rel)
    if p.is_absolute() and p.exists():
        return p
    if root is None:
        return None
    idx = _index(str(root))
    if rel_n in idx:
        return Path(idx[rel_n])
    hits = [v for k, v in idx.items() if k.endswith("/" + rel_n)]
    if hits:
        return Path(min(hits, key=len))
    base = rel_n.rsplit("/", 1)[-1]
    hits = [v for k, v in idx.items() if k.endswith("/" + base) or k == base]
    return Path(hits[0]) if hits else None


@lru_cache(maxsize=64)
def _load_file(path: str) -> tuple[np.ndarray, int]:
    from ..audio import load
    y, sr = load(path, sr=None, mono=False)
    y = np.atleast_2d(y).astype(np.float32)
    return y, sr


@dataclass
class Sample:
    data: np.ndarray            # (channels, n) float32
    sr: int
    root: float = 60.0          # MIDI note of the original pitch
    name: str = ""

    @property
    def mono(self) -> np.ndarray:
        return self.data.mean(axis=0)


NOTE_RE = re.compile(r"[ _]([A-Ga-g])([#b]?)(-?\d)$")


def root_from_name(name: str, default: float = 60.0) -> float:
    """Serum reads a root note from the end of the file name ("Bass F2.flac")."""
    stem = Path(name).stem
    m = NOTE_RE.search(stem)
    if not m:
        return default
    pc = "C D EF G A B".index(m.group(1).upper())
    acc = {"#": 1, "b": -1}.get(m.group(2), 0)
    return float(12 * (int(m.group(3)) + 2) + pc + acc)     # Serum: C3 = 60


@dataclass
class SampleBank:
    root: Path | None = None
    embedded: dict = field(default_factory=dict)
    missing: list = field(default_factory=list)

    def __post_init__(self):
        if self.root is None:
            self.root = serum_content_root()

    def add(self, data, sr: int, root: float = 60.0, name: str = "") -> str:
        arr = np.atleast_2d(np.asarray(data, dtype=np.float32))
        sid = f"embedded:{len(self.embedded)}"
        self.embedded[sid] = Sample(arr, int(sr), float(root), name)
        return sid

    def get(self, sid: str) -> Sample | None:
        if not sid:
            return None
        if sid in self.embedded:
            return self.embedded[sid]
        kind, _, ref = sid.partition(":")
        path = None
        if kind == "serum":
            path = find_content(ref, self.root)
        elif kind == "file":
            path = Path(ref) if Path(ref).exists() else None
        if path is None:
            if sid not in self.missing:
                self.missing.append(sid)
            return None
        data, sr = _load_file(str(path))
        return Sample(data, sr, root_from_name(path.name), path.name)


# ---------------------------------------------------------------------------
# SFZ (multisample instruments)

@dataclass
class Region:
    sample: str
    lokey: int = 0
    hikey: int = 127
    lovel: int = 1
    hivel: int = 127
    keycenter: float = 60.0
    tune: float = 0.0          # cents
    transpose: int = 0
    volume: float = 0.0        # dB
    loop_mode: str = "no_loop"
    loop_start: int = 0
    loop_end: int = 0
    offset: int = 0
    amp_veltrack: float = 100.0
    ampeg: dict = field(default_factory=dict)


def _note_num(v: str) -> int:
    v = v.strip()
    if re.fullmatch(r"-?\d+", v):
        return int(v)
    m = re.fullmatch(r"([A-Ga-g])([#b]?)(-?\d)", v)
    if not m:
        return 60
    pc = "C D EF G A B".index(m.group(1).upper())
    acc = {"#": 1, "b": -1}.get(m.group(2), 0)
    return 12 * (int(m.group(3)) + 1) + pc + acc      # SFZ: C4 = 60


def parse_sfz(text: str) -> list[Region]:
    """Minimal SFZ reader: <control>/<global>/<master>/<group>/<region>
    headers with the opcodes Serum's multisamples use."""
    text = re.sub(r"//[^\n]*", "", text or "")
    scopes = {"control": {}, "global": {}, "master": {}, "group": {}}
    regions = []
    target = scopes["group"]
    for tok in re.split(r"(<\w+>)", text):
        tok = tok.strip()
        if not tok:
            continue
        m = re.fullmatch(r"<(\w+)>", tok)
        if m:
            hdr = m.group(1)
            if hdr == "region":
                inherited = {}
                for sc in ("control", "global", "master", "group"):
                    inherited.update(scopes[sc])
                regions.append(inherited)
                target = regions[-1]
            elif hdr in scopes:
                order = ["control", "global", "master", "group"]
                for lower in order[order.index(hdr):]:
                    scopes[lower] = {}
                target = scopes[hdr]
            continue
        # opcodes; sample= may contain spaces, so values run to the next "key="
        for k, v in re.findall(r"(\w+)=(.*?)(?=\s+\w+=|$)", tok, flags=re.S):
            target[k] = v.strip()
    out = []
    for op in regions:
        if "sample" not in op:
            continue
        key = op.get("key")
        lo = _note_num(op.get("lokey", key or "0"))
        hi = _note_num(op.get("hikey", key or "127"))
        kc = _note_num(op.get("pitch_keycenter", key or str(lo)))
        eg = {k[6:]: float(v) for k, v in op.items() if k.startswith("ampeg_") and _isnum(v)}
        out.append(Region(sample=op["sample"].replace("\\", "/"), lokey=lo, hikey=hi,
                          lovel=int(float(op.get("lovel", 1))), hivel=int(float(op.get("hivel", 127))),
                          keycenter=float(kc), tune=_f(op.get("tune")), transpose=int(_f(op.get("transpose"))),
                          volume=_f(op.get("volume")), loop_mode=op.get("loop_mode", "no_loop"),
                          loop_start=int(_f(op.get("loop_start"))), loop_end=int(_f(op.get("loop_end"))),
                          offset=int(_f(op.get("offset"))), amp_veltrack=_f(op.get("amp_veltrack"), 100.0),
                          ampeg=eg))
    return out


def _isnum(v) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _f(v, default=0.0) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def pick_region(regions: list[Region], note: float, vel: float) -> Region | None:
    v = int(round(np.clip(vel, 0, 1) * 127))
    n = int(round(note))
    cands = [r for r in regions if r.lokey <= n <= r.hikey and r.lovel <= max(v, 1) <= r.hivel]
    if not cands:
        cands = [r for r in regions if r.lokey <= n <= r.hikey] or regions
    if not cands:
        return None
    return min(cands, key=lambda r: abs(r.keycenter - n))
