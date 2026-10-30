"""Correcting our engine during a run, from real Serum 2.

At every checkpoint the same patch exists twice: our engine's render and real
Serum's render. A few quick corrections are fitted so our engine sounds more
like Serum for the rest of the run (overall level, brightness tilt, filter
cutoff scale and amp-envelope timing). They change how our engine renders
during the search, not the preset. Each correction is kept only while it
makes our engine match Serum better; corrections that held up are saved and
start the next run.
"""
from __future__ import annotations

import json
import logging
import math
import time

import numpy as np

from .metrics import Reference

log = logging.getLogger(__name__)

# parameter ratios: patch path -> applied as value * ratio when rendering
RATIO_PATHS = {
    "cutoff": ("filter.cutoff", "filter2.cutoff"),
    "attack": ("envs.0.attack",),
    "decay": ("envs.0.decay",),
    "release": ("envs.0.release",),
}
GRIDS = {"cutoff": (0.6, 0.75, 0.87, 1.0, 1.15, 1.33, 1.6), "attack": (0.6, 0.8, 1.0, 1.25, 1.6),
         "decay": (0.7, 0.85, 1.0, 1.2, 1.4), "release": (0.7, 0.85, 1.0, 1.2, 1.4)}
LIMITS = {"cutoff": (0.4, 2.5), "attack": (0.3, 3.0), "decay": (0.4, 2.5), "release": (0.4, 2.5),
          "tilt": (-3.0, 3.0), "level_db": (-12.0, 12.0)}


class EngineCorrection:
    def __init__(self, values: dict | None = None):
        self.v = {"level_db": 0.0, "tilt": 0.0, "cutoff": 1.0, "attack": 1.0, "decay": 1.0, "release": 1.0}
        if values:
            for k, x in values.items():
                if k in self.v:
                    lo, hi = LIMITS[k]
                    self.v[k] = float(np.clip(x, lo, hi))

    def is_identity(self) -> bool:
        return all(abs(self.v[k] - (0.0 if k in ("level_db", "tilt") else 1.0)) < 1e-6 for k in self.v)

    def as_dict(self) -> dict:
        return {k: round(x, 4) for k, x in self.v.items()}

    def copy(self) -> "EngineCorrection":
        return EngineCorrection(dict(self.v))

    # ------------------------------------------------------------------
    def adjust_patch(self, patch):
        """The patch with the parameter ratios applied (or the patch itself)."""
        from . import params as P
        if all(abs(self.v[k] - 1.0) < 1e-6 for k in RATIO_PATHS):
            return patch
        q = patch.copy()
        for k, paths in RATIO_PATHS.items():
            r = self.v[k]
            if abs(r - 1.0) < 1e-6:
                continue
            for path in paths:
                try:
                    P.set_attr(q, path, float(P.get_attr(q, path)) * r)
                except (AttributeError, IndexError, KeyError, TypeError):
                    pass
        q.filter.cutoff = float(np.clip(q.filter.cutoff, 20.0, 20000.0))
        q.filter2.cutoff = float(np.clip(q.filter2.cutoff, 20.0, 20000.0))
        return q

    def post(self, y: np.ndarray, sr: int) -> np.ndarray:
        """Level and brightness tilt (dB per octave around 1 kHz)."""
        lv, tilt = self.v["level_db"], self.v["tilt"]
        if abs(tilt) > 1e-4:
            Y = np.fft.rfft(y)
            f = np.maximum(np.fft.rfftfreq(len(y), 1 / sr), 20.0)
            Y *= 10 ** (tilt * np.log2(f / 1000.0) / 20)
            y = np.fft.irfft(Y, n=len(y))
        if abs(lv) > 1e-4:
            y = y * 10 ** (lv / 20)
        return y


def match(y_engine: np.ndarray, y_serum: np.ndarray, sr: int) -> float:
    """How alike our engine's and Serum's renders of one patch are (0..100):
    spectral similarity and loudness-envelope correlation, minus a little for
    level and spectral (dB) distance."""
    n = min(len(y_engine), len(y_serum))
    if n < 1024 or not np.any(y_serum[:n]) or not np.any(y_engine[:n]):
        return 0.0
    d = Reference(np.asarray(y_serum[:n], dtype=np.float64), sr).all(np.asarray(y_engine[:n], dtype=np.float64))
    re, rs = np.sqrt(np.mean(y_engine[:n] ** 2)) + 1e-12, np.sqrt(np.mean(y_serum[:n] ** 2)) + 1e-12
    level = abs(20 * math.log10(re / rs))
    return 0.6 * d["spectral_similarity"] + 0.4 * d["envelope_correlation"] - 0.5 * d["lsd_db"] - 0.3 * level


def fit(corr: EngineCorrection, render_raw, patch, y_serum: np.ndarray, sr: int,
        time_limit: float = 20.0) -> tuple[EngineCorrection, float, float]:
    """Coordinate search for corrections that make ``render_raw(patch) ->
    engine audio`` match ``y_serum``. Returns (correction, match before,
    match after); the input correction is returned unchanged if nothing helps."""
    t0 = time.time()
    raw_cache: dict = {}

    def render_with(c: EngineCorrection) -> np.ndarray:
        key = tuple(round(c.v[k], 4) for k in RATIO_PATHS)
        if key not in raw_cache:
            raw_cache[key] = render_raw(c.adjust_patch(patch))
        return c.post(raw_cache[key], sr)

    def score(c):
        return match(render_with(c), y_serum, sr)

    base = score(corr)
    best, best_s = corr.copy(), base
    # level: closed form, then tilt: from the slope of the band difference
    ye = render_with(best)
    n = min(len(ye), len(y_serum))
    re, rs = np.sqrt(np.mean(ye[:n] ** 2)) + 1e-12, np.sqrt(np.mean(y_serum[:n] ** 2)) + 1e-12
    c = best.copy()
    c.v["level_db"] = float(np.clip(c.v["level_db"] + 20 * math.log10(rs / re), *LIMITS["level_db"]))
    s = score(c)
    if s > best_s:
        best, best_s = c, s
    for tilt_step in (0.5, 0.2):
        for sign in (1, -1):
            while time.time() - t0 < time_limit:
                c = best.copy()
                c.v["tilt"] = float(np.clip(c.v["tilt"] + sign * tilt_step, *LIMITS["tilt"]))
                s = score(c)
                if s > best_s + 0.02:
                    best, best_s = c, s
                else:
                    break
    uses_filter = patch.filter.enabled or patch.filter2.enabled
    for _ in range(2):
        for k in ("cutoff", "attack", "decay", "release"):
            if k == "cutoff" and not uses_filter:
                continue
            if time.time() - t0 > time_limit:
                break
            cur = best.v[k]
            for g in GRIDS[k]:
                c = best.copy()
                c.v[k] = float(np.clip(cur * g, *LIMITS[k]))
                if abs(c.v[k] - cur) < 1e-6:
                    continue
                s = score(c)
                if s > best_s + 0.02:
                    best, best_s = c, s
    if best_s <= base + 0.2:
        return corr, base, base
    return best, base, best_s


# ----------------------------------------------------------------------------
# corrections that held up across runs start the next run

def _path():
    from .. import calibration
    return calibration.cal_dir() / "engine_correction.json"


def load_saved() -> EngineCorrection:
    try:
        d = json.loads(_path().read_text())
        return EngineCorrection(d.get("values"))
    except (OSError, ValueError):
        return EngineCorrection()


def save_run(final: EngineCorrection, helped: bool) -> dict:
    """Fold one run's validated correction into the saved one (a running
    average in log space for ratios, plain for dB values)."""
    try:
        d = json.loads(_path().read_text())
    except (OSError, ValueError):
        d = {"values": {}, "runs": 0}
    if not helped:
        return d
    n = int(d.get("runs", 0))
    old = EngineCorrection(d.get("values")).v
    new = {}
    for k, x in final.v.items():
        if k in ("level_db", "tilt"):
            new[k] = (old[k] * n + x) / (n + 1)
        else:
            new[k] = math.exp((math.log(old[k]) * n + math.log(x)) / (n + 1))
    d = {"values": {k: round(v, 4) for k, v in new.items()}, "runs": n + 1, "updated": time.time()}
    try:
        _path().write_text(json.dumps(d, indent=1))
    except OSError as e:
        log.warning("could not save the engine correction: %s", e)
    return d


def reset_saved() -> None:
    try:
        _path().unlink()
    except OSError:
        pass
