"""Knob curves: how a parameter value maps to Serum's 0..1 knob travel.

Serum applies modulation in knob space: a mod depth of 100 % moves the knob
through its whole range, whatever the unit underneath (Hz, seconds, dB ...).
Generic modulation destinations ("@<patch path>") use these curves to turn a
modulation signal into parameter values over time.

Ranges follow the Serum 2 manual and the value ranges found in factory
presets; "Calibrate with Serum 2" can refine them (settings key
``knob_overrides``: {"<suffix>": [kind, lo, hi]}).
"""
from __future__ import annotations

import math
import re
from functools import lru_cache

import numpy as np

from .. import config

# (regex on the path, kind, lo, hi); kind: lin | log | pow4 (envelope times)
RULES = [
    (r"(^|\.)(cutoff|freq|freq1|freq2|filter_freq|xover_lo|xover_hi|lpf|hpf|lf_xover|freq_lo|freq_hi)$",
     "log", 8.0, 22050.0),
    (r"envs\.\d+\.(attack|decay|release|hold)$", "pow4", 0.0, 32.0),
    (r"envs\.\d+\.(sustain|start|end)$", "lin", 0.0, 1.0),
    (r"envs\.\d+\.\w+_curve$", "lin", 0.0, 1.0),
    (r"lfos\.\d+\.rate_hz$", "log", 0.01, 100.0),
    (r"lfos\.\d+\.(rise|delay)$", "pow4", 0.0, 32.0),
    (r"\.(fine)$", "lin", -100.0, 100.0),
    (r"\.(semi)$", "lin", -12.0, 12.0),
    (r"\.(octave)$", "lin", -4.0, 4.0),
    (r"\.(coarse)$", "lin", -48.0, 48.0),
    (r"^master_tune$", "lin", -24.0, 24.0),
    (r"\.(hz_offset)$", "lin", -1000.0, 1000.0),
    (r"\.(ratio)$", "lin", 0.0, 32.0),
    (r"\.(pan|balance|unison_wtpos|unison_warp|unison_warp2|window_skew|skew)$", "lin", -1.0, 1.0),
    (r"\.(unison_range)$", "lin", 0.0, 48.0),
    (r"\.(time|time2|predelay|delay_time)$", "log", 0.001, 4.0),
    (r"\.(attack_ms|release_ms)$", "log", 0.1, 1000.0),
    (r"\.(decay)$", "log", 0.05, 20.0),
    (r"\.(rate_hz|rate)$", "log", 0.01, 20.0),
    (r"\.(shift_hz)$", "lin", -2000.0, 2000.0),
    (r"\.(gain_db|gain1|gain2|makeup_db|gain_lo|gain_mid|gain_hi|ir_gain)$", "lin", -24.0, 24.0),
    (r"\.(threshold_db)$", "lin", -60.0, 0.0),
    (r"\.(ratio_comp)$", "log", 1.0, 100.0),
    (r"\.(delay_ms|delay2_ms)$", "lin", 0.0, 20.0),
    (r"\.(voices|stages|poles)$", "lin", 0.0, 12.0),
    (r"\.(density)$", "log", 0.1, 1000.0),
    (r"\.(length)$", "log", 0.001, 4.0),
    (r"\.(scan)$", "lin", -2.0, 2.0),
    (r"\.(rand_pitch)$", "lin", 0.0, 24.0),
    (r"\.(filter_shift)$", "lin", -100.0, 100.0),
    (r"\.(pitch)$", "lin", -48.0, 48.0),
    (r"^(fx_chain|fx_buses|filter|filter2)\b.*\.level$", "lin", 0.0, 2.0),
    (r"\.(level)$", "lin", 0.0, 1.0),
    (r"tuning_a4$", "lin", 400.0, 480.0),
    (r"transpose$", "lin", -24.0, 24.0),
    (r"glide$", "pow4", 0.0, 10.0),
]
_COMPILED = [(re.compile(p), k, lo, hi) for p, k, lo, hi in RULES]


@lru_cache(maxsize=1)
def _overrides() -> dict:
    try:
        return dict(config.settings().get("knob_overrides", {}) or {})
    except Exception:
        return {}


@lru_cache(maxsize=4096)
def spec(path: str) -> tuple[str, float, float]:
    for suffix, val in _overrides().items():
        if path.endswith(suffix):
            return tuple(val)
    for rx, kind, lo, hi in _COMPILED:
        if rx.search(path):
            return kind, lo, hi
    return "lin", 0.0, 1.0


def to_knob(path: str, value):
    kind, lo, hi = spec(path)
    v = np.asarray(value, dtype=float)
    if kind == "log":
        return np.log(np.clip(v, lo, hi) / lo) / math.log(hi / lo)
    if kind == "pow4":
        return (np.clip(v, lo, hi) - lo) ** 0.25 / (hi - lo) ** 0.25
    return (v - lo) / (hi - lo)


def from_knob(path: str, knob):
    kind, lo, hi = spec(path)
    k = np.clip(np.asarray(knob, dtype=float), 0.0, 1.0)
    if kind == "log":
        return lo * (hi / lo) ** k
    if kind == "pow4":
        return lo + (hi - lo) * k ** 4
    return lo + (hi - lo) * k


def modulated(path: str, base, knob_offset):
    """Parameter value over time: base value moved by ``knob_offset`` (a
    fraction of the knob travel, scalar or array)."""
    if knob_offset is None or (np.ndim(knob_offset) == 0 and float(knob_offset) == 0.0):
        return base
    return from_knob(path, to_knob(path, base) + knob_offset)
