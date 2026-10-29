"""Convert between :class:`Patch` and a Serum 2 preset state (both ways).

Parameter names, module layout, enum values and modulation IDs come from the
structure of Serum 2.0.x presets (701 presets were surveyed, see
param_ids.json) and the Serum 2 user guide. A few *unit scales* (filter
cutoff curve, LFO rate units, pitch-mod depth ...) are not stored in presets;
they live in ``SCALES`` and are replaced by Serum's own values when
"Calibrate with Serum 2" has been run.

Everything the patch model covers is converted both ways. Preset parts it
does not interpret (clip banks, arp patterns, MIDI maps, mod slots with
unknown targets) are carried in ``Patch.serum_extra`` / ``Patch.raw_mods`` so
an imported preset exports unchanged.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .. import config
from ..synth.patch import (FX_DEFAULTS, LFO, N_ENVS, N_LFOS, N_MACROS, Arp, Clip, Env, Filter, GranularCfg,
                           Mod, MultiCfg, NoiseOsc, Osc, Patch, SampleCfg, SpectralCfg, SubOsc, VoiceControl,
                           fx, sync_to_hz)
from ..synth.wavetables import FRAME, TableBank, resolve_serum_ref

SKELETON_PATH = Path(__file__).with_name("init_skeleton.json")
PARAM_IDS_PATH = Path(__file__).with_name("param_ids.json")

SCALES = {
    # VoiceFilter kParamFreq is 0..1; exponential from f_lo to f_hi (measured
    # in Serum 2: 8 Hz at 0, ~425 Hz at 0.5, 22.05 kHz at 1)
    "cutoff_lo_hz": 8.176,
    "cutoff_hi_hz": 22050.0,
    # semitones reached by a +100 % modulation of Oscillator kParamCoarsePit
    "pitch_semis_per_100": 64.0,
    # semitones reached by +100 % modulation of Global > Main Tuning
    "master_tune_semis_per_100": 48.0,
    "lfo_rate_is_hz": True,
    "tablepos_lo": 1.0,
    "tablepos_hi": 256.0,
}

LEGACY_FILTERS = {"lp12": "L12", "lp24": "L24", "hp12": "H12", "hp24": "H24", "bp12": "B12"}
LEGACY_DIST = {"soft": "SoftClip", "hard": "HardClip", "tube": "Tube", "sinefold": "SinFold",
               "linfold": "LinFold", "diode": "Diode1"}
LEGACY_WARPS = {"fm_b": "FM_OSC", "fm_sub": "FM_SUB", "am_b": "AM_OSC", "rm_b": "RM_OSC", "rm_sub": "RM_SUB"}
SUB_NAMES = {"sine": None, "triangle": "kTriangle", "saw": "kSaw", "square": "kSquare", "pulse": "kPulse",
             "roundrect": "kRoundRect"}
LFO_TYPE_NAMES = {"random": "RandomSH", "lorenz": "Lorenz", "rossler": "Rossler", "path": "Path"}
OSC_TYPES = {"sample": "kOsc_Sample", "granular": "kOsc_Granular", "spectral": "kOsc_Spectral",
             "multisample": "kOsc_MultiSample"}
OSC_SUB = {"wavetable": "WTOsc", "sample": "SampleOsc", "granular": "GranularOsc", "spectral": "SpectralOsc",
           "multisample": "MultiSampleOsc"}
LOOP_MODES = {"forward": "kForward", "pingpong": "kPingPong", "reverse": "kReverse", "tailed": "kTailed"}
DETUNE_MODES = {"super": "kDetuneSuper", "exp": "kDetuneExp", "inv": "kDetuneInv", "random": "kDetuneRandom"}
PHASE_MEMORY = {"contiguous": "kContiguous", "per_voice": "kPerVoice"}
SPAWN = {"even": "kEven", "exp": "kExponential", "random": "kRandom"}
ROUTE_SLOT = {"osc1": 0, "osc2": 1, "osc3": 2, "noise": 3, "sub": 4, "filter1": 5, "filter2": 6}
ROUTE_DEST = {"filter": "kRoutingDestFilter", "main": "kRoutingDestMaster", "direct": "kRoutingDestDirect",
              "none": "kRoutingDestNone"}
ROUTE_DEST_INV = {v: k for k, v in ROUTE_DEST.items()}
BUS_DEST = {0.0: "main", 1.0: "direct", 2.0: "bus"}

# FX rack entry "type" numbers and their module keys
FX_TYPE_IDS = {"distortion": 0, "flanger": 1, "phaser": 2, "chorus": 3, "delay": 4, "comp": 5, "reverb": 6,
               "eq": 7, "filter": 8, "hyperdim": 9, "bode": 10, "convolve": 11, "utility": 12, "split": 13,
               "split3": 14, "splitms": 15}
FX_KEYS = {"distortion": "FXDistortion", "flanger": "FXFlanger", "phaser": "FXPhaser", "chorus": "FXChorus",
           "delay": "FXDelay", "comp": "FXComp", "reverb": "FXReverb", "eq": "FXEQ", "filter": "FXFilter",
           "hyperdim": "FXHyperD", "bode": "FXBode", "convolve": "FXConv", "utility": "FXUtils",
           "split": "FXSplit", "split3": "FXSplit3", "splitms": "FXSplitMS"}
FX_BY_ID = {v: k for k, v in FX_TYPE_IDS.items()}
FX_BY_KEY = {v: k for k, v in FX_KEYS.items()}

# modulation sources (ModSlot "source": [id, aux_id]). Envelopes, LFOs,
# velocity, note, mod wheel and macros were verified against factory
# presets; the others are inferred from how presets use them. Unknown ids
# import as "src<N>" (rendered as 0) and export unchanged.
SOURCE_IDS = {"fixed": 1}
SOURCE_IDS.update({f"env{i}": 1 + i for i in range(1, N_ENVS + 1)})
SOURCE_IDS.update({f"lfo{i}": 5 + i for i in range(1, N_LFOS + 1)})
SOURCE_IDS.update({"velocity": 16, "note": 17, "modwheel": 18, "aftertouch": 19, "polyat": 20,
                   "rand1": 21, "rand2": 22, "alt1": 23, "alt2": 24})
SOURCE_IDS.update({f"macro{i}": 24 + i for i in range(1, N_MACROS + 1)})
SOURCE_IDS.update({"pitchbend": 38})
SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}


def _src_id(name: str | None) -> int:
    if not name:
        return 0
    if name in SOURCE_IDS:
        return SOURCE_IDS[name]
    if name.startswith("src") and name[3:].isdigit():
        return int(name[3:])
    return 0


def _src_name(i: int) -> str | None:
    if not i:
        return None
    return SOURCE_NAMES.get(i, f"src{i}")


def _load_param_ids() -> dict:
    try:
        return json.loads(PARAM_IDS_PATH.read_text())
    except OSError:
        return {}


PARAM_IDS = _load_param_ids()
PARAM_IDS.update({"WTOsc.kParamTablePos": 6, "Oscillator.kParamDetune": 26, "Oscillator.kParamDetuneWid": 27})


def scales() -> dict:
    s = dict(SCALES)
    s.update(config.settings().get("serum_scales", {}))
    return s


def cutoff_to_norm(hz: float, sc: dict | None = None) -> float:
    sc = sc or scales()
    table = sc.get("cutoff_table")
    if table and len(table) >= 5:          # measured in Serum by "Calibrate"
        xs = np.array([r[0] for r in table], dtype=float)
        hzs = np.log(np.array([r[1] for r in table], dtype=float))
        order = np.argsort(hzs)
        return float(np.clip(np.interp(np.log(max(hz, 1e-3)), hzs[order], xs[order]), 0.0, 1.0))
    lo, hi = sc["cutoff_lo_hz"], sc["cutoff_hi_hz"]
    return float(np.clip(math.log(max(hz, lo) / lo) / math.log(hi / lo), 0.0, 1.0))


def norm_to_cutoff(x: float, sc: dict | None = None) -> float:
    sc = sc or scales()
    table = sc.get("cutoff_table")
    if table and len(table) >= 5:
        xs = np.array([r[0] for r in table], dtype=float)
        hzs = np.log(np.array([r[1] for r in table], dtype=float))
        return float(np.exp(np.interp(float(x), xs, hzs)))
    lo, hi = sc["cutoff_lo_hz"], sc["cutoff_hi_hz"]
    return float(lo * (hi / lo) ** float(np.clip(x, 0, 1)))


def cutoff_octaves_span(sc: dict | None = None) -> float:
    sc = sc or scales()
    return math.log2(sc["cutoff_hi_hz"] / sc["cutoff_lo_hz"])


def load_skeleton() -> dict:
    return json.loads(SKELETON_PATH.read_text())


def _pp(module: dict) -> dict:
    """Get a module's plainParams dict, materialising 'default'."""
    if not isinstance(module.get("plainParams"), dict):
        module["plainParams"] = {}
    return module["plainParams"]


def _pget(module, key, default=None):
    pp = module.get("plainParams") if isinstance(module, dict) else None
    return pp.get(key, default) if isinstance(pp, dict) else default


def _k(name: str | None) -> str | None:
    """Serum enum spelling: 'Sync' -> 'kSync'."""
    if not name or name == "none":
        return None
    return name if name.startswith("k") and name[1:2].isupper() else "k" + name


def _unk(name: str | None) -> str:
    if not name:
        return "none"
    return name[1:] if name.startswith("k") and name[1:2].isupper() else name


def _f(v) -> float:
    return float(v)


# ---------------------------------------------------------------------------
# drawn curves (LFO graphs, remap graphs, mod curves, X-Shaper, spectral mask)

# Serum stores every drawn curve with y measured from the top of the graph
# (factory "saw up" is y 1 -> 0, VELO->Amp curves fall from ~0.9 to 0), so
# y is flipped on the way in and out; patch curves use y = 0 at the bottom.

def curve_from_serum(cd: dict | None) -> list | None:
    if not isinstance(cd, dict) or not cd.get("xVals"):
        return None
    xs, ys = cd["xVals"], cd["yVals"]
    cv = cd.get("curveVals") or [0.5] * len(xs)
    return [[float(x), 1.0 - float(y), float(c)] for x, y, c in zip(xs, ys, cv)]


def curve_to_serum(points) -> dict:
    pts = [list(p) + [0.5] * (3 - len(p)) for p in points]
    return {"curveVals": [float(p[2]) for p in pts], "numPoints": max(len(pts) - 1, 1),
            "xVals": [float(p[0]) for p in pts], "yVals": [1.0 - float(p[1]) for p in pts]}


def _lfo_points(shape: str, points):
    if shape == "custom" and points:
        pts = [tuple(float(v) for v in p) for p in points]
    elif shape == "triangle":
        pts = [(0.0, 0.0), (0.5, 1.0), (1.0, 0.0)]
    elif shape == "saw_up":
        pts = [(0.0, 0.0), (0.999, 1.0), (1.0, 0.0)]
    elif shape == "saw_down":
        pts = [(0.0, 1.0), (0.999, 0.0), (1.0, 1.0)]
    elif shape == "square":
        pts = [(0.0, 1.0), (0.499, 1.0), (0.5, 0.0), (0.999, 0.0), (1.0, 1.0)]
    else:  # sine sampled with 16 linear segments, starting at the bottom
        xs = np.linspace(0, 1, 17)
        pts = [(float(x), float(0.5 - 0.5 * math.cos(2 * math.pi * x))) for x in xs]
    if pts[0][0] > 0:
        pts.insert(0, (0.0, pts[0][1]))
    if pts[-1][0] < 1:
        pts.append((1.0, pts[0][1]))
    return pts


# ---------------------------------------------------------------------------
# FX modules: our dict <-> Serum plainParams

# simple 1:1 parameters: (our key, serum key, scale) with serum = ours * scale
FX_SIMPLE = {
    "flanger": [("rate_hz", "kParamRate", 1), ("depth", "kParamDepth", 100), ("feedback", "kParamFeedback", 100),
                ("phase", "kParamWidth", 360), ("mix", "kParamWet", 100)],
    "phaser": [("rate_hz", "kParamRate", 1), ("depth", "kParamDepth", 100), ("depth2", "kParamDepth2", 1),
               ("freq", "kParamFreq", 1), ("feedback", "kParamFeedback", 100), ("stages", "kParamNumPoles", 1),
               ("phase", "kParamWidth", 360), ("mix", "kParamWet", 100)],
    "chorus": [("rate_hz", "kParamRate", 1), ("depth", "kParamDepth", 25), ("delay_ms", "kParamDelay", 1),
               ("delay2_ms", "kParamDelay2", 1), ("feedback", "kParamFeedback", 100), ("filt_hz", "kParamFilt", 1),
               ("mix", "kParamWet", 100)],
    "delay": [("offset_l", "kParamOffsetL", 1), ("offset_r", "kParamOffsetR", 1),
              ("feedback", "kParamFeedback", 100), ("freq", "kParamFreq", 1), ("bw", "kParamBW", 1),
              ("mix", "kParamWet", 100)],
    "hyperdim": [("rate", "kParamRate", 100), ("detune", "kParamDetune", 100), ("voices", "kParamUnison", 1),
                 ("mix", "kParamWet", 100), ("dim_size", "kParamDimESize", 100), ("dim_mix", "kParamDimEWet", 100)],
    "bode": [("range_hz", "kParamRange", 1), ("direction", "kParamOutputMix", 100), ("width", "kParamOutputWidth", 100),
             ("delay_time", "kParamDelayTime", 1), ("feedback", "kParamFeedback", 100),
             ("balance", "kParamDelayBalance", 100), ("blur", "kParamBlur", 100), ("mix", "kParamWet", 100)],
    "convolve": [("decay", "kParamDecay", 1), ("predelay", "kParamPredelay", 1), ("attack", "kParamAttack", 1),
                 ("damp", "kParamDamping", 100), ("tone", "kParamTone", 100), ("ir_gain_db", "kParamIpTrim", 1),
                 ("mix", "kParamWet", 100)],
    "utility": [("hpf", "kParamHPF", 1), ("lpf", "kParamLPF", 1), ("mono_freq", "kParamLFXover", 1),
                ("width", "kParamWidth", 100), ("balance", "kParamBalance", 100), ("mix", "kParamWet", 100)],
    "eq": [("freq1", "kParamFreq1", 1), ("gain1", "kParamGain1", 1), ("freq2", "kParamFreq2", 1),
           ("gain2", "kParamGain2", 1)],
    "split": [("freq", "kParamFreq", 1)],
    "split3": [("freq", "kParamFreq", 1), ("freq2", "kParamFreq2", 1)],
    "distortion": [("drive", "kParamDrive", 100), ("mix", "kParamWet", 100), ("filter_freq", "kParamFreq", 1),
                   ("filter_type", "kParamLPHP", 100), ("stages", "kParamNumStages", 1)],
    "filter": [("reso", "kParamReso", 100), ("drive", "kParamDrive", 100), ("var", "kParamVar", 100),
               ("mix", "kParamWet", 100), ("x", "kParamX", 1), ("y", "kParamY", 1)],
    "reverb": [("size", "kParamSize", 100), ("width", "kParamWidth", 100), ("lo_cut", "kParamFreq", 100),
               ("hi_cut", "kParamFreqB", 100), ("damping", "kParamFreqC", 100), ("mode", "kParamMode", 1),
               ("mix", "kParamWet", 100)],
    "comp": [("attack_ms", "kParamAttack", 1), ("release_ms", "kParamRelease", 1), ("xover_lo", "kParamXoverLow", 1),
             ("xover_hi", "kParamXoverHi", 1)],
}
# serum defaults when a key is absent (stored presets only keep changed values)
FX_SERUM_DEFAULTS = {
    "flanger": {"kParamRate": 0.3, "kParamDepth": 50, "kParamFeedback": 50, "kParamWidth": 0, "kParamWet": 50},
    "phaser": {"kParamRate": 0.3, "kParamDepth": 50, "kParamDepth2": 0.5, "kParamFreq": 800, "kParamFeedback": 40,
               "kParamNumPoles": 4, "kParamWidth": 0, "kParamWet": 50},
    "chorus": {"kParamRate": 0.8, "kParamDepth": 7.5, "kParamDelay": 7.0, "kParamDelay2": 0.0, "kParamFeedback": 0,
               "kParamFilt": 20000.0, "kParamWet": 30},
    "delay": {"kParamOffsetL": 1.0, "kParamOffsetR": 1.0, "kParamFeedback": 35, "kParamFreq": 2000.0,
              "kParamBW": 3.0, "kParamWet": 25},
    "hyperdim": {"kParamRate": 40, "kParamDetune": 30, "kParamUnison": 4, "kParamWet": 40, "kParamDimESize": 50,
                 "kParamDimEWet": 0},
    "bode": {"kParamRange": 100.0, "kParamOutputMix": 0, "kParamOutputWidth": 50, "kParamDelayTime": 0.0,
             "kParamFeedback": 0, "kParamDelayBalance": 0, "kParamBlur": 0, "kParamWet": 50},
    "convolve": {"kParamDecay": 1.0, "kParamPredelay": 0.0, "kParamAttack": 0.0, "kParamDamping": 0,
                 "kParamTone": 0, "kParamIpTrim": 0.0, "kParamWet": 20},
    "utility": {"kParamHPF": 0.0, "kParamLPF": 0.0, "kParamLFXover": 120.0, "kParamWidth": 100,
                "kParamBalance": 0, "kParamWet": 100},
    "eq": {"kParamFreq1": 120.0, "kParamGain1": 0.0, "kParamFreq2": 6000.0, "kParamGain2": 0.0},
    "split": {"kParamFreq": 300.0}, "split3": {"kParamFreq": 200.0, "kParamFreq2": 2000.0},
    "distortion": {"kParamDrive": 30, "kParamWet": 100, "kParamFreq": 0.5, "kParamLPHP": 0, "kParamNumStages": 2},
    "filter": {"kParamReso": 20, "kParamDrive": 0, "kParamVar": 50, "kParamWet": 100, "kParamX": 0.0, "kParamY": 0.0},
    "reverb": {"kParamSize": 50, "kParamWidth": 100, "kParamFreq": 0, "kParamFreqB": 0, "kParamFreqC": 50,
               "kParamMode": 0, "kParamWet": 20},
    "comp": {"kParamAttack": 5.0, "kParamRelease": 80.0, "kParamXoverLow": 200.0, "kParamXoverHi": 3000.0},
}
EQ_TYPES = {0.0: ("lowshelf", "highshelf"), 1.0: ("peak", "peak"), 2.0: ("hp", "lp")}


def _comp_thresh_db(x: float) -> float:
    return 20 * math.log10(max((1 - float(np.clip(x, 0, 1))) ** 2, 1e-6))


def _comp_thresh_norm(db: float) -> float:
    return float(np.clip(1 - math.sqrt(10 ** (db / 20)), 0, 1))


def _fx_to_serum(m: dict, sc: dict, bpm: float) -> tuple[dict, dict]:
    """(plainParams, extra entry/module keys) for one module."""
    kind = m["type"]
    d = copy.deepcopy(FX_DEFAULTS.get(kind, {}))
    d.update(m)
    pp, extra = {}, {}
    if not m.get("enabled", True):
        pp["kParamEnable"] = 0.0
    defaults = FX_SERUM_DEFAULTS.get(kind, {})
    for ours, key, sc_ in FX_SIMPLE.get(kind, []):
        v = d.get(ours)
        if v is None or isinstance(v, bool):
            continue
        val = float(np.mean(v)) * sc_
        if abs(val - float(defaults.get(key, val + 1))) > 1e-9:
            pp[key] = val
    lvl = d.get("level", 1.0)
    if abs(float(lvl) - 1.0) > 1e-6:
        pp["kParamLevelOut"] = float(np.clip(lvl / 2, 0, 1))
    if kind == "distortion":
        mode = LEGACY_DIST.get(d["mode"], d["mode"])
        if mode != "Tube":
            pp["kParamMode"] = _k(mode)
        if d.get("filter") in ("pre", "post"):
            pp["kParamPrePost"] = 1.0 if d["filter"] == "pre" else 2.0
            pp["kParamBW"] = float(0.075 + 7.5 * d.get("filter_bw", 0.5))
        flex = [curve_to_serum(d["curve_a"]) if d.get("curve_a") else {},
                curve_to_serum(d["curve_b"]) if d.get("curve_b") else {}]
        extra["flex"] = flex
    elif kind in ("flanger", "phaser", "chorus"):
        if d.get("sync"):
            pp["kParamBeatSync"] = 1.0
            pp["kParamRate"] = float(1.0 / (4 * 60.0 / max(bpm, 1) * _div_beats(d["sync"]) / 4))
        if kind == "chorus" and d.get("filt_mode") == "hp":
            pp["kParamFiltMode"] = 1.0
    elif kind == "delay":
        pp["kParamBeatSync"] = 0.0           # times are written in seconds
        pp["kParamTimeL"] = float(d["time"])
        pp["kParamTimeR"] = float(d["time2"])
        mode = {"pingpong": 1.0, "tap": 2.0}.get(d.get("mode", "normal"))
        if mode:
            pp["kParamMode"] = mode
        if d.get("link"):
            pp["kParamLink"] = 1.0
    elif kind == "comp":
        pp["kParamMultiband"] = 1.0 if d["multiband"] else 0.0
        pp["kParamWet"] = float(d["depth"] * 100)
        if d["multiband"]:
            pp["kParamThresh"] = float(np.clip(1.0 - d["amount"] * 0.7, 0.05, 1.0))
            for bi, bc in enumerate(d.get("band_cfg") or []):
                raw = bc.get("raw") or {}
                for kk, vv in raw.items():
                    pp[f"{kk}{bi}"] = float(vv)
                if "gain_db" in bc:
                    pp[f"kParamGain{bi}"] = float(bc["gain_db"])
        else:
            pp.update({"kParamThresh": _comp_thresh_norm(d["threshold_db"]),
                       "kParamRatio": float(1e6 if d["ratio"] >= 100 else d["ratio"]),
                       "kParamMakeup": float(10 ** (d["makeup_db"] / 20))})
    elif kind == "reverb":
        kind_name = str(d["kind"])
        if kind_name not in ("Plate", "kPlate"):
            pp["kParamType"] = _k({"Basin": "Abyss", "Nitrous": "Space"}.get(kind_name, kind_name))
        if d.get("predelay"):
            pp["kParamPreDelay"] = float(d["predelay"])
        rt = float(d.get("decay", 1.5))
        if _unk(pp.get("kParamType", "kPlate")) in ("Space", "Abyss"):
            pp["kParamFeedback"] = float(np.clip(100 * (1 - math.exp(-rt / 3)), 0, 100))
        else:
            pp["kParamDelay"] = float(np.clip(100 * math.sqrt(max(rt - 0.3, 0) / 10), 0, 100))
    elif kind == "eq":
        t1 = {"lowshelf": 0.0, "peak": 1.0, "hp": 2.0}.get(d["type1"], 0.0)
        t2 = {"highshelf": 0.0, "peak": 1.0, "lp": 2.0}.get(d["type2"], 0.0)
        if t1:
            pp["kParamType1"] = t1
        if t2:
            pp["kParamType2"] = t2
        pp["kParamReso1"] = float(100 * math.sqrt(max(d["q1"] - 0.3, 0) / 3))
        pp["kParamReso2"] = float(100 * math.sqrt(max(d["q2"] - 0.3, 0) / 3))
    elif kind == "filter":
        pp["kParamType"] = LEGACY_FILTERS.get(d["ftype"], d["ftype"])
        pp["kParamFreq"] = cutoff_to_norm(float(np.mean(d["cutoff"])), sc)
        if d.get("pan"):
            pp["kParamStereo"] = float(50 + 50 * d["pan"])
        if d.get("clean"):
            pp["kParamPad"] = 1.0
        if d.get("pz"):
            extra["PZs"] = d["pz"]
    elif kind == "hyperdim":
        if d.get("retrig"):
            pp["kParamRetrig"] = 1.0
    elif kind == "bode":
        rng_ = max(float(d.get("range_hz", 100.0)), 1.0)
        if abs(d["shift_hz"]) > rng_:
            rng_ = abs(d["shift_hz"])
            pp["kParamRange"] = rng_
        pp["kParamShift"] = float(100.0 * d["shift_hz"] / rng_)
        for ours, key in (("mono_input", "kParamMonoInput"), ("retrig", "kParamRetrig")):
            if d.get(ours):
                pp[key] = 1.0
    elif kind == "convolve":
        pp["kParamSize"] = float(np.clip(d["size"], 0.01, 4) * 200)
        if d.get("min_phase"):
            pp["kParamMinPhase"] = 1.0
        if d.get("ir"):
            if str(d["ir"]).startswith("serum:"):
                extra["relativePathToIR"] = d["ir"][6:]
    elif kind == "utility":
        if d.get("gain_db"):
            lvl = float(d.get("level", 1.0)) * 10 ** (d["gain_db"] / 20)
            pp["kParamLevelOut"] = float(np.clip(lvl / 2, 0, 1))
        if d.get("invert"):
            pp["kParamPolarityL"] = 1.0
        if d.get("invert_r"):
            pp["kParamPolarityR"] = 1.0
        if d.get("mono_bass"):
            pp["kParamLFMono"] = 1.0
    clean = {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
             for k, v in pp.items()}
    return clean, extra


def _div_beats(sync: str) -> float:
    s = sync.strip()
    mult = 1.0
    if s.endswith("T"):
        mult, s = 2.0 / 3.0, s[:-1]
    elif s.endswith("D"):
        mult, s = 1.5, s[:-1]
    num, den = s.split("/")
    return 4.0 * float(num) / float(den) * mult


def _fx_entry(m: dict, sc: dict, bpm: float) -> list[dict]:
    """One module (plus, for splitters, its band modules) as rack entries."""
    kind = m["type"]
    pp, extra = _fx_to_serum(m, sc, bpm)
    head = {FX_KEYS[kind]: {"plainParams": pp or "default"}, "type": FX_TYPE_IDS[kind]}
    for k, v in (m.get("_serum") or {}).items():           # verbatim extras from an import
        head.setdefault(k, copy.deepcopy(v))
    mod_keys = {k: v for k, v in extra.items() if k in ("PZs", "relativePathToIR")}
    head[FX_KEYS[kind]].update(mod_keys)
    if "flex" in extra:
        head["flex"] = extra["flex"]
    if kind in ("split", "split3", "splitms"):
        bands = m.get("bands") or []
        nb = 3 if kind == "split3" else 2
        sub = []
        if not isinstance(head[FX_KEYS[kind]]["plainParams"], dict):
            head[FX_KEYS[kind]]["plainParams"] = {}
        for b in range(nb):
            chain = bands[b] if b < len(bands) else []
            flat = []
            for e in chain:
                flat += _fx_entry(e, sc, bpm)
            head[FX_KEYS[kind]]["plainParams"][f"kParamModuleCount{b + 1}"] = float(len(flat))
            sub += flat
        return [head] + sub
    if kind == "distortion":
        head.setdefault("flex", [{}, {}])
    if kind in ("chorus", "flanger", "phaser"):
        head[FX_KEYS[kind]].setdefault("lfophasor", 0.0)
    if kind not in ("chorus", "eq", "split", "split3", "splitms"):
        head.setdefault("kUIParamMixOrGain", 0.0)
    return [head]


def _serum_to_fx(entry: dict, sc: dict, bpm: float, samples=None) -> dict | None:
    kind = FX_BY_ID.get(entry.get("type"))
    if kind is None:
        return None
    mod = entry.get(FX_KEYS[kind]) or {}
    pp = mod.get("plainParams")
    pp = pp if isinstance(pp, dict) else {}
    defaults = FX_SERUM_DEFAULTS.get(kind, {})
    g = lambda k, dflt=None: pp.get(k, defaults.get(k, dflt))
    m = fx(kind)
    if pp.get("kParamEnable", 1.0) == 0.0:
        m["enabled"] = False
    for ours, key, sc_ in FX_SIMPLE.get(kind, []):
        v = g(key)
        if v is not None:
            m[ours] = float(v) / sc_
    if "kParamLevelOut" in pp:
        m["level"] = 2 * float(pp["kParamLevelOut"])
    if kind == "distortion":
        m["mode"] = _unk(pp.get("kParamMode")) if pp.get("kParamMode") else "Tube"
        pre = pp.get("kParamPrePost", 0.0)
        m["filter"] = {1.0: "pre", 2.0: "post"}.get(pre, "off")
        m["filter_bw"] = (pp.get("kParamBW", 3.8) - 0.075) / 7.5
        flex = entry.get("flex") or []
        m["curve_a"] = curve_from_serum(flex[0]) if len(flex) > 0 else None
        m["curve_b"] = curve_from_serum(flex[1]) if len(flex) > 1 else None
    elif kind in ("flanger", "phaser", "chorus"):
        if pp.get("kParamBeatSync") == 1.0:
            m["sync"] = None              # synced rate: keep the Hz Serum shows at 1/x of the bar
        if kind == "chorus":
            m["filt_mode"] = "hp" if pp.get("kParamFiltMode") == 1.0 else "lp"
    elif kind == "delay":
        synced = pp.get("kParamBeatSync", 1.0) != 0.0
        beat = 60.0 / max(bpm, 1)
        t1 = pp.get("kParamTimeL", 0.25)
        t2 = pp.get("kParamTimeR", t1)
        if synced:            # synced times are in whole notes
            t1, t2 = t1 * 4 * beat, t2 * 4 * beat
        m.update(time=float(t1), time2=float(t2), link=pp.get("kParamLink") == 1.0,
                 mode={1.0: "pingpong", 2.0: "tap"}.get(pp.get("kParamMode", 0.0), "normal"))
    elif kind == "comp":
        mb = pp.get("kParamMultiband", 0.0) == 1.0
        m.update(multiband=mb, depth=pp.get("kParamWet", 100) / 100)
        if mb:
            m["amount"] = float(np.clip((1 - pp.get("kParamThresh", 0.6)) / 0.7, 0, 1))
            bands = []
            for bi in range(3):
                raw = {k[:-1]: pp[k] for k in pp if k[-1:] == str(bi) and k[:-1] in
                       ("kParamThreshUD", "kParamRatio", "kParamRatioBelow", "kParamDeadband")}
                bc = {"raw": raw}
                if f"kParamGain{bi}" in pp:
                    bc["gain_db"] = float(pp[f"kParamGain{bi}"])
                if "kParamRatio" in raw:
                    bc["ratio"] = float(1 / max(1 - raw["kParamRatio"], 0.01))
                if "kParamRatioBelow" in raw:
                    bc["ratio_below"] = float(1 / max(1 - raw["kParamRatioBelow"], 0.01))
                if "kParamThreshUD" in raw:
                    bc["thresh_down_db"] = -float(raw["kParamThreshUD"]) * 60 / 165
                    bc["thresh_up_db"] = bc["thresh_down_db"] - 20
                bands.append(bc)
            if any(b != {"raw": {}} for b in bands):
                m["band_cfg"] = bands
        else:
            m.update(threshold_db=_comp_thresh_db(pp.get("kParamThresh", 0.3)),
                     ratio=min(float(pp.get("kParamRatio", 4.0)), 100.0),
                     makeup_db=20 * math.log10(max(pp.get("kParamMakeup", 1.0), 1e-4)))
    elif kind == "reverb":
        t = _unk(pp.get("kParamType")) if pp.get("kParamType") else "Plate"
        m["kind"] = t
        m["predelay"] = float(pp.get("kParamPreDelay", 0.0))
        if t in ("Space", "Abyss"):
            fb = pp.get("kParamFeedback", 60) / 100
            m["decay"] = float(-3 * math.log(max(1 - fb, 0.02)))
        elif t == "Plate":
            m["decay"] = 0.3 + 4 * m["size"] ** 2
        else:
            m["decay"] = 0.3 + 10 * (pp.get("kParamDelay", 40) / 100) ** 2
    elif kind == "eq":
        t1 = EQ_TYPES.get(pp.get("kParamType1", 0.0), EQ_TYPES[0.0])[0]
        t2 = EQ_TYPES.get(pp.get("kParamType2", 0.0), EQ_TYPES[0.0])[1]
        m.update(type1=t1, type2=t2, q1=0.3 + 3 * (pp.get("kParamReso1", 40) / 100) ** 2,
                 q2=0.3 + 3 * (pp.get("kParamReso2", 40) / 100) ** 2)
    elif kind == "filter":
        m.update(ftype=pp.get("kParamType", "MgL6"), cutoff=norm_to_cutoff(pp.get("kParamFreq", 0.5), sc),
                 pan=(pp.get("kParamStereo", 50) - 50) / 50, clean=pp.get("kParamPad") == 1.0,
                 pz=mod.get("PZs"))
    elif kind == "hyperdim":
        m["retrig"] = pp.get("kParamRetrig") == 1.0
    elif kind == "bode":
        m["shift_hz"] = pp.get("kParamShift", 0.0) / 100 * m["range_hz"]
        m["mono_input"] = pp.get("kParamMonoInput") == 1.0
        m["retrig"] = pp.get("kParamRetrig") == 1.0
    elif kind == "convolve":
        m["size"] = pp.get("kParamSize", 100.0) / 200
        m["min_phase"] = pp.get("kParamMinPhase") == 1.0
        if mod.get("relativePathToIR"):
            m["ir"] = "serum:" + mod["relativePathToIR"]
        elif mod.get("embeddedIR") and samples is not None:
            data = np.asarray(mod["embeddedIR"], dtype=np.float32)
            m["ir"] = samples.add(np.atleast_2d(data), int(mod.get("sampleRate", 44100)))
    elif kind == "utility":
        m.update(gain_db=0.0, invert=pp.get("kParamPolarityL") == 1.0, invert_r=pp.get("kParamPolarityR") == 1.0,
                 mono_bass=pp.get("kParamLFMono") == 1.0)
    if kind in ("split", "split3", "splitms"):
        nb = 3 if kind == "split3" else 2
        m["_counts"] = [int(pp.get(f"kParamModuleCount{b + 1}", 1)) for b in range(nb)]
    keep = {k: v for k, v in entry.items() if k in ("kUIParamMixOrGain", "kUIParamMixOrGainDimE",
                                                    "kUIParamMixOrGainHyper")}
    if keep:
        m["_serum"] = keep
    return m


def rack_to_chain(entries: list, sc: dict, bpm: float, samples=None) -> list:
    """Serum's flat rack list (splitters followed by their band modules)."""
    out = []
    i = 0
    entries = entries or []

    def take(n):
        nonlocal i
        chain = []
        while n > 0 and i < len(entries):
            m = _serum_to_fx(entries[i], sc, bpm, samples)
            i += 1
            if m is None:
                n -= 1
                continue
            counts = m.pop("_counts", None)
            if counts:
                m["bands"] = [take(c) for c in counts]
            chain.append(m)
            n -= 1
        return chain
    while i < len(entries):
        out += take(1)
    return out


def chain_to_rack(chain: list, sc: dict, bpm: float) -> list:
    rack = []
    for m in chain or []:
        rack += _fx_entry(m, sc, bpm)
    return rack


def _flat_paths(chain: list, prefix: str) -> list[str]:
    """Patch paths of every module in Serum's flattened rack order."""
    out = []
    for j, m in enumerate(chain or []):
        out.append(f"{prefix}.{j}")
        for b, band in enumerate(m.get("bands") or []):
            out += _flat_paths(band, f"{prefix}.{j}.bands.{b}")
    return out


def _count_modules(m: dict) -> int:
    return 1 + sum(_count_modules(e) for band in (m.get("bands") or []) for e in band)


def _top_level_rack_index(chain: list) -> list[int]:
    out, pos = [], 0
    for m in chain or []:
        out.append(pos)
        pos += _count_modules(m)
    return out


def _legacy_chain(patch: Patch) -> list:
    """The original fixed FX slots (distortion, OTT, EQ, chorus) as modules."""
    out = []
    d = patch.distortion
    if d.enabled:
        out.append(fx("distortion", mode=LEGACY_DIST.get(d.mode, d.mode), drive=d.drive, mix=d.mix))
    if patch.ott.enabled:
        out.append(fx("comp", multiband=True, depth=patch.ott.depth, amount=patch.ott.amount))
    if patch.eq.enabled:
        e = patch.eq
        out.append(fx("eq", freq1=e.low_freq, gain1=e.low_gain, type1="lowshelf", freq2=e.high_freq,
                      gain2=e.high_gain, type2="highshelf"))
    if patch.chorus.enabled:
        c = patch.chorus
        out.append(fx("chorus", rate_hz=c.rate_hz, depth=c.depth, mix=c.mix))
    return out


# ---------------------------------------------------------------------------
# generic modulation destinations: "@<patch path>" <-> (module type, id, param)

def _osc_bindings():
    rows = []
    osc = [("level", "kParamVolume"), ("pan", "kParamPan"), ("octave", "kParamOctave"), ("semi", "kParamPitch"),
           ("coarse", "kParamCoarsePit"), ("fine", "kParamFine"), ("detune", "kParamDetune"),
           ("blend", "kParamDetuneWid"), ("unison_width", "kParamUnisonStereo"),
           ("unison_warp", "kParamUnisonWarp"), ("unison_warp2", "kParamUnisonWarp2"),
           ("hz_offset", "kParamHzOffset"), ("ratio", "kParamPitchRatio"),
           ("smp.start", "kParamStart"), ("smp.end", "kParamEnd"), ("smp.loop_start", "kParamLoopStart"),
           ("smp.loop_end", "kParamLoopEnd"), ("smp.scan", "kParamScanRate"), ("smp.position", "kParamPosition"),
           ("smp.crossfade", "kParamLoopCrossfade"), ("smp.span", "kParamUnisonSpan"),
           ("smp.random_start", "kParamRandomStart")]
    for f, k in osc:
        rows.append(("oscs.{i}." + f, "Oscillator", k))
    for f, k in (("position", "kParamTablePos"), ("phase", "kParamInitialPhase"), ("random_phase", "kParamRandomPhase"),
                 ("unison_wtpos", "kParamUnisonWTPos")):
        rows.append(("oscs.{i}." + f, "WTOsc", k))
    for f, k in (("warp_amount", "kParamWarp"), ("warp2_amount", "kParamWarp2"), ("warp_var", "kParamWarpVar"),
                 ("warp2_var", "kParamWarpVar2")):
        rows.append(("oscs.{i}." + f, "<sub>", k))
    for f, k in (("density", "kParamDensity"), ("length", "kParamGrainLength"), ("rand_dir", "kParamRandomDir"),
                 ("rand_gain", "kParamRandomGain"), ("rand_length", "kParamRandomGrainLength"),
                 ("rand_offset", "kParamRandomOffset"), ("rand_pan", "kParamRandomPan"),
                 ("rand_pitch", "kParamRandomPitch"), ("rand_warp", "kParamRandomWarp"),
                 ("rand_warp2", "kParamRandomWarp2"), ("window_amount", "kParamWindowParam"),
                 ("window_skew", "kParamWindowSkew"), ("rand_window", "kParamRandomWindowAmount"),
                 ("rand_skew", "kParamRandomWindowSkew")):
        rows.append(("oscs.{i}.gran." + f, "GranularOsc", k))
    for f, k in (("freq_lo", "kParamFreqLo"), ("freq_hi", "kParamFreqHi"), ("filter_shift", "kParamSpecFltShift"),
                 ("filter_mix", "kParamSpecFltWetDry")):
        rows.append(("oscs.{i}.spec." + f, "SpectralOsc", k))
    rows.append(("oscs.{i}.multi.timbre", "MultiSampleOsc", "kParamTimbreShift"))
    return rows


# (path template, module type, param, value scale: serum = ours * scale (+ offset))
VALUE_SCALE = {
    "kParamVolume": 1, "kParamPan": 50, "kParamOctave": 1, "kParamPitch": 1, "kParamCoarsePit": 1, "kParamFine": 1,
    "kParamDetune": 1, "kParamDetuneWid": 100, "kParamUnisonStereo": 100, "kParamUnisonWarp": 100,
    "kParamUnisonWarp2": 100, "kParamHzOffset": 1, "kParamPitchRatio": 1, "kParamStart": 100, "kParamEnd": 100,
    "kParamLoopStart": 100, "kParamLoopEnd": 100, "kParamScanRate": 100, "kParamPosition": 100,
    "kParamLoopCrossfade": 100, "kParamUnisonSpan": 100, "kParamRandomStart": 100,
    "kParamInitialPhase": 360, "kParamRandomPhase": 100, "kParamUnisonWTPos": 100,
    "kParamWarp": 1, "kParamWarp2": 1, "kParamWarpVar": 1, "kParamWarpVar2": 1,
}


def _bindings() -> list[tuple]:
    rows = _osc_bindings()
    rows += [("noise.level", "Oscillator:3", "kParamVolume"), ("noise.pan", "Oscillator:3", "kParamPan"),
             ("noise.color", "NoiseOsc:3", "kParamColor"), ("noise.fine", "NoiseOsc:3", "kParamFine"),
             ("noise.start", "NoiseOsc:3", "kParamInitialPhase"),
             ("noise.random_start", "NoiseOsc:3", "kParamRandomPhase"),
             ("sub.level", "Oscillator:4", "kParamVolume"), ("sub.pan", "Oscillator:4", "kParamPan"),
             ("sub.octave", "Oscillator:4", "kParamOctave"), ("sub.coarse", "Oscillator:4", "kParamCoarsePit")]
    for fi, key in ((0, "filter"), (1, "filter2")):
        for f, k in (("cutoff", "kParamFreq"), ("reso", "kParamReso"), ("drive", "kParamDrive"), ("var", "kParamVar"),
                     ("mix", "kParamWet"), ("level", "kParamLevelOut"), ("pan", "kParamStereo"), ("x", "kParamX"),
                     ("y", "kParamY")):
            rows.append((f"{key}.{f}", f"VoiceFilter:{fi}", k))
    for f, k in (("attack", "kParamAttack"), ("hold", "kParamHold"), ("decay", "kParamDecay"),
                 ("sustain", "kParamSustain"), ("release", "kParamRelease"), ("attack_curve", "kParamCurve1"),
                 ("decay_curve", "kParamCurve2"), ("release_curve", "kParamCurve3")):
        rows.append(("envs.{i}." + f, "Env", k))
    for f, k in (("rate_hz", "kParamRate"), ("rise", "kParamRise"), ("delay", "kParamDelay"),
                 ("smooth", "kParamSmooth"), ("phase", "kParamPhase")):
        rows.append(("lfos.{i}." + f, "LFO", k))
    rows += [("macros.{i}", "Macro", "kParamValue"), ("lfo_buses.{i}", "LFOPointModBus", "kParamValue"),
             ("master_tune", "Global:0", "kParamMasterTuning"), ("glide", "Global:0", "kParamPortamentoTime"),
             ("transpose", "Global:0", "kParamTranspose"), ("swing", "Global:0", "kParamSwing"),
             ("voice.env_scale", "VoicePanel:0", "kParamGlobalScalingEnvTime"),
             ("voice.lfo_scale", "VoicePanel:0", "kParamGlobalScalingLfoTime")]
    for name, slot in ROUTE_SLOT.items():
        for f, k in (("balance", "kParamFilterBalance"), ("send1", "kParamFXBus1Level"), ("send2", "kParamFXBus2Level")):
            rows.append((f"routing.{name}.{f}", f"RoutingSlot:{slot}", k))
    return rows


BINDINGS = _bindings()


def _osc_sub_type(patch: Patch, i: int) -> str:
    o = patch.oscs[i] if i < len(patch.oscs) else None
    return OSC_SUB.get(o.type if o else "wavetable", "WTOsc")


def dest_to_serum(patch: Patch, path: str, fx_paths: dict) -> tuple | None:
    """'@'-destination path -> (module type, module id, param name)."""
    if path.startswith(("fx_chain.", "fx_buses.")):
        mpath, _, key = path.rpartition(".")
        if mpath not in fx_paths:
            return None
        mid, kind = fx_paths[mpath]
        for ours, skey, _sc in FX_SIMPLE.get(kind, []):
            if ours == key:
                return FX_KEYS[kind], mid, skey
        special = {"cutoff": "kParamFreq", "shift_hz": "kParamShift", "time": "kParamTimeL", "time2": "kParamTimeR",
                   "threshold_db": "kParamThresh", "makeup_db": "kParamMakeup", "ratio": "kParamRatio",
                   "level": "kParamLevelOut", "gain_db": "kParamLevelOut", "decay": "kParamDecay"}
        if key in special:
            return FX_KEYS[kind], mid, special[key]
        return None
    for tpl, mod, param in BINDINGS:
        if "{i}" in tpl:
            pre, post = tpl.split("{i}")
            if path.startswith(pre) and path.endswith(post) and path[len(pre):len(path) - len(post)].isdigit():
                i = int(path[len(pre):len(path) - len(post)])
                mtype = _osc_sub_type(patch, i) if mod == "<sub>" else mod
                return mtype, i, param
        elif tpl == path:
            mtype, _, mid = mod.partition(":")
            return mtype, int(mid or 0), param
    return None


def serum_to_dest(mtype: str, mid: int, pname: str, fx_flat: dict) -> str | None:
    """(module type, id, param) -> '@'-destination path (without '@')."""
    if mtype in FX_BY_KEY and mid in fx_flat:
        path, kind = fx_flat[mid]
        for ours, skey, _sc in FX_SIMPLE.get(kind, []):
            if skey == pname:
                return f"{path}.{ours}"
        inv = {"kParamFreq": "cutoff" if kind == "filter" else "freq", "kParamShift": "shift_hz",
               "kParamTimeL": "time", "kParamTimeR": "time2", "kParamThresh": "threshold_db",
               "kParamMakeup": "makeup_db", "kParamRatio": "ratio", "kParamLevelOut": "level",
               "kParamDecay": "decay"}
        if pname in inv:
            return f"{path}.{inv[pname]}"
        return None
    for tpl, mod, param in BINDINGS:
        if param != pname:
            continue
        mt, _, m_id = mod.partition(":")
        if "{i}" in tpl:
            if mt == "<sub>" and mtype in OSC_SUB.values():
                return tpl.replace("{i}", str(mid))
            if mt == mtype:
                return tpl.replace("{i}", str(mid))
        elif mt == mtype and int(m_id or 0) == mid:
            return tpl
    return None


# ---------------------------------------------------------------------------
# Patch -> Serum


def _embed_sample(bank_samples, sid: str) -> dict:
    """Serum entries for a sample source (embedded audio or a content path)."""
    if not sid:
        return {}
    if sid.startswith("serum:"):
        return {"samplePathRelative": sid[6:]}
    smp = bank_samples.get(sid) if bank_samples is not None else None
    if smp is None:
        return {}
    data = np.atleast_2d(smp.data).astype(np.float32)
    return {"embedded_audio": {"data": [ch for ch in data], "sampleRate": float(smp.sr),
                               "table_name": smp.name or "Fourier Analyzer"},
            "numChannels": int(data.shape[0]), "numFrames": int(data.shape[1]), "sampleRate": int(smp.sr),
            "samplePath": ""}


def _write_osc(st, i, o, sc, bank, samples):
    mod = st[f"Oscillator{i}"]
    pp = _pp(mod)
    pp["kParamEnable"] = 1.0 if o.enabled else 0.0
    if not o.enabled and o == Osc():
        return
    if o.type in OSC_TYPES:
        pp["kParamType"] = OSC_TYPES[o.type]
    pp["kParamVolume"] = float(np.clip(o.level, 0, 1))
    for attr, key, scale, dflt in (("octave", "kParamOctave", 1, 0), ("semi", "kParamPitch", 1, 0),
                                   ("coarse", "kParamCoarsePit", 1, 0), ("fine", "kParamFine", 1, 0),
                                   ("pan", "kParamPan", 50, 0), ("hz_offset", "kParamHzOffset", 1, 0)):
        v = getattr(o, attr)
        if v != dflt:
            pp[key] = float(v * scale)
    if o.pitch_mode != "semitones":
        pp["kParamPitchMode"] = "Harmonics" if o.pitch_mode == "harmonics" else "Ratio"
        pp["kParamPitchRatio"] = float(o.ratio)
        if o.ratio_source >= 0:
            pp["kParamPitchSource"] = float(o.ratio_source)
    if not o.pitch_track:
        pp["kParamPitchTrack"] = 0.0
    if not o.bend_track:
        pp["kParamPitchBendTrack"] = 0.0
    if o.unison > 1 or o.unison_mode != "linear" or o.unison_width != 1.0 or o.unison_stack != "none":
        if o.unison > 1:
            pp["kParamUnison"] = float(o.unison)
        pp["kParamDetune"] = float(o.detune)
        pp["kParamDetuneWid"] = float(o.blend * 100)
        if o.unison_range != 2.0:
            pp["kParamUnisonRange"] = float(o.unison_range)
        if o.unison_stack and o.unison_stack != "none":
            pp["kParamUnisonStack"] = _k(o.unison_stack)
        if o.unison_mode in DETUNE_MODES:
            pp["kParamDetuneMode"] = DETUNE_MODES[o.unison_mode]
        if o.unison_width != 1.0:
            pp["kParamUnisonStereo"] = float(o.unison_width * 100)
        if o.unison_warp:
            pp["kParamUnisonWarp"] = float(o.unison_warp * 100)
        if o.unison_warp2:
            pp["kParamUnisonWarp2"] = float(o.unison_warp2 * 100)
    if tuple(o.key_range) != (0, 127):
        pp["kParamKeyZoneMin"], pp["kParamKeyZoneMax"] = float(o.key_range[0]), float(min(o.key_range[1], 126))
    if tuple(o.vel_range) != (0, 127):
        pp["kParamVelocityZoneMin"], pp["kParamVelocityZoneMax"] = float(o.vel_range[0]), float(o.vel_range[1])
    if o.key_warp:
        pp["kParamKeyZoneWarp"] = 1.0
    subkey = OSC_SUB.get(o.type, "WTOsc")
    sub = mod.setdefault(f"{subkey}{i}", {"plainParams": "default"})
    spp = _pp(sub)
    for slot, (w, amt, var) in enumerate(((o.warp, o.warp_amount, o.warp_var), (o.warp2, o.warp2_amount, o.warp2_var))):
        name = _k(LEGACY_WARPS.get(w, w))
        if name and w != "none":
            spp["kParamWarpMenu" if slot == 0 else "kParamWarpMenu2"] = name
            spp["kParamWarp" if slot == 0 else "kParamWarp2"] = float(np.clip(amt, 0, 1))
            if abs(var - 0.5) > 1e-6:
                spp["kParamWarpVar" if slot == 0 else "kParamWarpVar2"] = float(var)
    if o.type == "wavetable":
        wt = sub
        wpp = spp
        wpp["kParamTablePos"] = float(sc["tablepos_lo"] + o.position * (sc["tablepos_hi"] - sc["tablepos_lo"]))
        wpp["kParamRandomPhase"] = float(o.random_phase * 100)
        if o.phase:
            wpp["kParamInitialPhase"] = float((o.phase % 1.0) * 360)
        if o.phase_memory in PHASE_MEMORY:
            wpp["kParamPhaseMemory"] = PHASE_MEMORY[o.phase_memory]
        if o.smooth_interp:
            wpp["kParamXfadeMode"] = 1.0
        if o.unison_wtpos:
            wpp["kParamUnisonWTPos"] = float(o.unison_wtpos * 100)
        if o.remap_curve:
            wt["flex"] = curve_to_serum(o.remap_curve)
        ref = resolve_serum_ref(o.table)
        if ref:
            wt["relativePathToWT"] = ref
            try:
                wt["numFrames"] = int(bank.frames(o.table).shape[0] * FRAME)
            except (OSError, KeyError, RuntimeError):
                pass        # Serum reads the file itself; the count is informational
            for k in ("embeddedWTData", "tableDisplayName", "interpolateAfterLoad"):
                wt.pop(k, None)
        else:
            frames = bank.frames(o.table)
            wt.pop("relativePathToWT", None)
            wt["embeddedWTData"] = np.asarray(frames, dtype=np.float32).ravel()
            wt["numFrames"] = int(frames.size)
            wt["numChannels"] = 1
            wt["sampleRate"] = 44100
            wt["tableDisplayName"] = _display_name(o.table)
        return
    cfg = o.smp
    if o.type == "multisample":
        sub["embedded_sfz"] = o.multi.sfz
        if o.multi.folder:
            sub["sfzPathRelative"] = o.multi.folder
        if o.multi.timbre:
            spp["kParamTimbreShift"] = float(o.multi.timbre)
        if o.multi.env_override and o.multi.env:
            spp["kParamEnvOverride"] = 1.0
            for k, v in o.multi.env.items():
                spp["kParamEnv" + k.capitalize()] = float(v)
        return
    sub.update(_embed_sample(samples, cfg.source))
    if cfg.root is not None:
        sub["baseNote"] = int(round(cfg.root))
    if cfg.detune:
        sub["baseDetune"] = int(round(cfg.detune))
    for attr, key, scale, dflt in (("start", "kParamStart", 100, 0.0), ("end", "kParamEnd", 100, 1.0),
                                   ("loop_start", "kParamLoopStart", 100, 0.0), ("loop_end", "kParamLoopEnd", 100, 1.0),
                                   ("crossfade", "kParamLoopCrossfade", 100, 0.0), ("scan", "kParamScanRate", 100, 1.0),
                                   ("position", "kParamPosition", 100, 0.0), ("span", "kParamUnisonSpan", 100, 0.0),
                                   ("random_start", "kParamRandomStart", 100, 0.0), ("base_tempo", "kParamBaseTempo", 1, 0.0)):
        v = getattr(cfg, attr)
        if abs(v - dflt) > 1e-9:
            pp[key] = float(v * scale)
    if cfg.loop in LOOP_MODES:
        pp["kParamLoopMode"] = LOOP_MODES[cfg.loop]
    for attr, key in (("reverse", "kParamReverse"), ("link_loop", "kParamLinkLoopLength"),
                      ("relative_loop", "kParamLoopStartLink"), ("exit_on_release", "kParamLoopEndsAtRelease"),
                      ("manual_position", "kParamManualPositionMode"), ("scan_tempo_lock", "kParamScanTempoLock")):
        if getattr(cfg, attr):
            pp[key] = 1.0
    if cfg.scan_range:
        pp["kParamScanRange"] = float(cfg.scan_range)
    if cfg.slicing != "off":
        pp["kParamSlicingEnabled"] = cfg.slicing.capitalize()
        if cfg.slices:
            mod["sliceMarkers"] = [float(x) for x in cfg.slices]
    if o.type == "granular":
        gc = o.gran
        spp.update({"kParamDensity": float(gc.density), "kParamGrainLength": float(gc.length)})
        if gc.density_mode != "free":
            spp["kParamDensityMode"] = {"bpm": "kDensityBPM", "grains": "kDensityGrains"}[gc.density_mode]
        if gc.length_mode != "free":
            spp["kParamLengthMode"] = {"bpm": "kLengthBPM", "percent": "kLengthPercent"}[gc.length_mode]
        if gc.window != "Hann":
            spp["kParamWindowShape"] = "kWindow" + gc.window
        spp["kParamWindowParam"] = float(gc.window_amount * 100)
        if gc.window_skew:
            spp["kParamWindowSkew"] = float(gc.window_skew * 100)
        for attr, key, scale in (("rand_offset", "kParamRandomOffset", 100), ("rand_dir", "kParamRandomDir", 100),
                                 ("rand_pitch", "kParamRandomPitch", 1 / 12), ("rand_length", "kParamRandomGrainLength", 100),
                                 ("rand_pan", "kParamRandomPan", 100), ("rand_gain", "kParamRandomGain", 100),
                                 ("rand_window", "kParamRandomWindowAmount", 100),
                                 ("rand_skew", "kParamRandomWindowSkew", 100), ("rand_warp", "kParamRandomWarp", 100),
                                 ("rand_warp2", "kParamRandomWarp2", 100)):
            v = getattr(gc, attr)
            if v:
                spp[key] = float(v * scale)
        if gc.spawn in SPAWN:
            spp["kParamUnisonTrigPattern"] = SPAWN[gc.spawn]
    elif o.type == "spectral":
        s2 = o.spec
        spp.update({"kParamFreqLo": float(s2.freq_lo), "kParamFreqHi": float(s2.freq_hi)})
        for attr, key in (("smooth_edges", "kParamLoHiIsSmooth"), ("phase_lock", "kParamPhaseLock"),
                          ("transients", "kParamTransients"), ("post_warp", "kParamPostWarp")):
            if getattr(s2, attr):
                spp[key] = 1.0
        if s2.filter_shift:
            spp["kParamSpecFltShift"] = float(s2.filter_shift)
        if s2.filter_mix:
            spp["kParamSpecFltWetDry"] = float(s2.filter_mix * 100)
        if s2.filter_curve:
            sub["flex"] = curve_to_serum(s2.filter_curve)


def patch_to_state(patch: Patch, bank: TableBank | None = None, samples=None) -> dict:
    sc = scales()
    bank = bank or TableBank()
    st = load_skeleton()
    for k, v in (patch.serum_extra or {}).items():       # untouched parts of an imported preset
        st[k] = copy.deepcopy(v)

    g = _pp(st["Global0"])
    g["kParamMasterVolume"] = float(np.clip(patch.master * 0.5, 0.0, 1.0))
    if patch.mono:
        g["kParamMonoToggle"] = 1.0
    if patch.legato and patch.mono:
        g["kParamLegato"] = 1.0
    if not patch.mono and patch.poly != 16:
        g["kParamPolyCount"] = float(patch.poly)
    if patch.same_note_limit:
        g["kParamLimitSameNotePolyphony"] = 1.0
    g["kParamBendRangeUp"] = float(patch.bend_range[0])     # matches export.BEND_RANGE
    g["kParamBendRangeDn"] = float(patch.bend_range[1])
    if patch.glide > 0:
        g["kParamPortamentoTime"] = float(patch.glide)
        if patch.glide_always:
            g["kParamPortaAlways"] = 1.0
        if patch.glide_scaled:
            g["kParamPortaScaled"] = 1.0
        if abs(patch.glide_curve - 0.5) > 1e-6:
            g["kParamPortamentoCurve"] = float(patch.glide_curve * 100)
    if patch.transpose:
        g["kParamTranspose"] = float(patch.transpose)
    if patch.tuning_a4 != 440.0:
        g["kParamGlobalTuning"] = float(patch.tuning_a4)
    if patch.master_tune:
        g["kParamMasterTuning"] = float(patch.master_tune)
    if patch.direct_level != 1.0:
        g["kParamDirectVol"] = float(patch.direct_level)
    if patch.modwheel:
        g["kParamModWheel"] = float(patch.modwheel * 100)
    if patch.swing:
        g["kParamSwing"] = float(patch.swing * 100)
    if abs(patch.voice_amp - 0.5) > 1e-6:
        g["kParamVoiceAmp"] = float(patch.voice_amp)
    for b in range(2):
        if b < len(patch.fx_buses) and patch.fx_buses[b]:
            g[f"kParamFXBus{b + 1}Vol"] = float(patch.bus_levels[b] if b < len(patch.bus_levels) else 1.0)
        dest = patch.bus_dest[b] if b < len(patch.bus_dest) else "main"
        if dest != "main":
            g[f"kParamFXBus{b + 1}Dest"] = 1.0 if dest == "direct" else 2.0
    if patch.tuning_table and len(patch.tuning_table) >= 128:
        st["tuningData"] = [float(x) for x in patch.tuning_table]
        st["tuningName"] = "(special)"

    # keyboard: key / scale quantizer
    if patch.scale_notes:
        pq = st.setdefault("PitchQuantizer0", {"plainParams": "default"})
        _pp(pq)["kParamKey"] = float(patch.scale_key)
        _pp(pq)["kParamScale"] = 1.0
        pq["scale"] = [2 if a else 0 for a in patch.scale_notes]
        pq["scaleName"] = "Custom"

    # --- oscillators ----------------------------------------------------------
    for i, o in enumerate(patch.oscs[:3]):
        _write_osc(st, i, o, sc, bank, samples)

    # --- noise (Oscillator3) and sub (Oscillator4) ----------------------------
    npp = _pp(st["Oscillator3"])
    npp["kParamEnable"] = 1.0 if patch.noise.enabled else 0.0
    nz = patch.noise
    if nz.enabled:
        npp["kParamVolume"] = float(np.clip(nz.level, 0, 1))
        if nz.pan:
            npp["kParamPan"] = float(nz.pan * 50)
        if nz.keytrack:
            npp["kParamPitchTrack"] = 1.0
        nmod = st["Oscillator3"]["NoiseOsc3"]
        nzp = _pp(nmod)
        if nz.kind == "sample" and nz.sample:
            if nz.sample.startswith("serum:"):
                nmod["relativePathToNoiseSample"] = nz.sample[6:]
        else:
            nzp["kParamNoiseType"] = nz.kind if nz.kind in ("White", "Pink", "Brown", "Geiger") else \
                ("Pink" if nz.color < 0.35 else "White")
        nzp["kParamColor"] = float(np.clip(nz.color, 0, 1))
        if nz.one_shot:
            nzp["kParamOneShot"] = 1.0
        if nz.start:
            nzp["kParamInitialPhase"] = float(nz.start * 100)
        if nz.random_start != 1.0:
            nzp["kParamRandomPhase"] = float(nz.random_start * 100)
        if nz.fine:
            nzp["kParamFine"] = float(nz.fine / 100)
        if nz.pitch:
            npp["kParamPitch"] = float(nz.pitch)
    spp = _pp(st["Oscillator4"])
    s = patch.sub
    spp["kParamEnable"] = 1.0 if s.enabled else 0.0
    if s.enabled:
        spp["kParamVolume"] = float(np.clip(s.level, 0, 1))
        spp["kParamOctave"] = float(s.octave)
        if s.coarse:
            spp["kParamCoarsePit"] = float(s.coarse)
        if s.pan:
            spp["kParamPan"] = float(s.pan * 50)
        sub = st["Oscillator4"].setdefault("SubOsc4", {"plainParams": "default"})
        shape = SUB_NAMES.get(s.shape)
        if shape:
            _pp(sub)["kParamShape"] = shape
        if s.phase:
            _pp(sub)["kParamInitialPhase"] = float(s.phase * 360)
        if s.contiguous:
            _pp(sub)["kParamContiguousPhase"] = 1.0

    # --- filters + routing ------------------------------------------------------
    for idx, f in enumerate((patch.filter, patch.filter2)):
        fmod = st[f"VoiceFilter{idx}"]
        fpp = _pp(fmod)
        if f.enabled:
            fpp["kParamEnable"] = 1.0
            fpp["kParamType"] = LEGACY_FILTERS.get(f.type, f.type)
            fpp["kParamFreq"] = cutoff_to_norm(f.cutoff, sc)
            fpp["kParamReso"] = float(f.reso * 100)
            if f.drive:
                fpp["kParamDrive"] = float(f.drive * 100)
            if f.keytrack:
                fpp["kParamKeyTrack"] = 1.0
            if f.var != 0.5:
                fpp["kParamVar"] = float(f.var * 100)
            if f.mix < 1:
                fpp["kParamWet"] = float(f.mix * 100)
            if f.level != 1.0:
                fpp["kParamLevelOut"] = float(np.clip(f.level / 2, 0, 1))
            if f.pan:
                fpp["kParamStereo"] = float(50 + 50 * f.pan)
            if f.clean:
                fpp["kParamPad"] = 1.0
            if f.x:
                fpp["kParamX"] = float(f.x)
            if f.y:
                fpp["kParamY"] = float(f.y)
            if f.pz:
                fmod["PZs"] = f.pz
        else:
            fpp["kParamEnable"] = 0.0
    any_filter = patch.filter.enabled or patch.filter2.enabled
    sources = [f"osc{i + 1}" for i, o in enumerate(patch.oscs[:3]) if o.enabled]
    if patch.noise.enabled:
        sources.append("noise")
    if patch.sub.enabled:
        sources.append("sub")
    sources += [f"filter{k}" for k, f in ((1, patch.filter), (2, patch.filter2)) if f.enabled]
    for s_ in sources:
        r = patch.route(s_)
        rp = _pp(st[f"RoutingSlot{ROUTE_SLOT[s_]}"])
        dest = r["dest"]
        if dest == "filter" and not any_filter and not s_.startswith("filter"):
            dest = "main"
        rp["kParamRoutingDest"] = ROUTE_DEST.get(dest, "kRoutingDestFilter")
        if dest == "filter" and not s_.startswith("filter"):
            bal = r["balance"] if (patch.filter.enabled and patch.filter2.enabled) else \
                (0.0 if patch.filter.enabled else 1.0)
            if bal:
                rp["kParamFilterBalance"] = float(-100 + 200 * bal)
        if r.get("send1"):
            rp["kParamFXBus1Level"] = float(r["send1"] * 100)
        if r.get("send2"):
            rp["kParamFXBus2Level"] = float(r["send2"] * 100)
        if not r.get("env1", True):
            rp["kParamViaEnv1"] = 0.0

    # --- envelopes ------------------------------------------------------------
    for i, e in enumerate(patch.envs[:N_ENVS]):
        epp = _pp(st[f"Env{i}"])
        epp.update({"kParamAttack": float(e.attack), "kParamHold": float(e.hold),
                    "kParamDecay": float(e.decay), "kParamSustain": float(e.sustain),
                    "kParamRelease": float(e.release),
                    "kParamCurve1": float(e.attack_curve * 100), "kParamCurve2": float(e.decay_curve * 100),
                    "kParamCurve3": float(e.release_curve * 100)})
        if e.start:
            epp["kParamStart"] = float(e.start)
        if e.end:
            epp["kParamEnd"] = float(e.end)
        if e.legato_inverted:
            epp["kParamLegatoInverted"] = 1.0

    # --- LFOs -----------------------------------------------------------------
    used = {m.source for m in patch.mods if m.amount} | {m.aux for m in patch.mods if m.amount and m.aux}
    for k, l in enumerate(patch.lfos[:N_LFOS]):
        if (f"lfo{k+1}" not in used and not l.point_mods and l == LFO()) or f"LFO{k}" not in st:
            continue
        mod = st[f"LFO{k}"]
        lpp = _pp(mod)
        rate = sync_to_hz(l.sync, patch.bpm) if l.sync else l.rate_hz
        lpp["kParamBeatSync"] = 0.0      # write exact Hz derived from the tempo
        lpp["kParamRate"] = float(rate)
        lpp["kParamMode"] = {"free": "Free", "env": "Envelope"}.get(l.mode, "Retrig")
        if lpp["kParamMode"] == "Retrig":
            lpp.pop("kParamMode")
        if l.phase:
            lpp["kParamPhase"] = float(l.phase * 360)
        if l.type in LFO_TYPE_NAMES:
            lpp["kParamType"] = LFO_TYPE_NAMES[l.type]
        for attr, key, scale in (("rise", "kParamRise", 1), ("delay", "kParamDelay", 1), ("smooth", "kParamSmooth", 100)):
            if getattr(l, attr):
                lpp[key] = float(getattr(l, attr) * scale)
        if l.direction != "forward":
            lpp["kParamDirection"] = 1.0 if l.direction == "reverse" else 2.0
        if l.mono:
            lpp["kParamMono"] = 1.0
        if l.anchored:
            lpp["kParamAnchored"] = 1.0
        if l.swing:
            lpp["kParamSwing"] = 1.0
        pts = _lfo_points(l.shape, l.points)
        cd = curve_to_serum([[p[0], p[1], p[2] if len(p) > 2 else 0.5] for p in pts])
        if l.loopback is not None:
            xs = [p[0] for p in pts]
            cd["loopbackPointNum"] = int(np.argmin([abs(x - l.loopback) for x in xs]))
        mod["curveData"] = cd
        mod["curveDisplayName"] = "Custom"
        if l.type == "path" and l.path:
            pd = curve_to_serum(l.path)
            pd["isOpen"] = bool(l.path_open)
            mod["pathData"] = pd
            mod["pathDisplayName"] = "Custom"
    pm = []
    for k, l in enumerate(patch.lfos[:N_LFOS]):
        for a in l.point_mods:
            pm.append({"busID": int(a.get("bus", 0)), "lfoID": k, "lfoType": 1 if a.get("path") else 0,
                       "pointID": int(a.get("point", 0)), "target": {"x": 0, "y": 1, "curve": 2}.get(a.get("axis"), 1)})
    if pm:
        st["lfoPointModAssignments"] = pm
    for b, v in enumerate(patch.lfo_buses):
        if v and f"LFOPointModBus{b}" in st:
            _pp(st[f"LFOPointModBus{b}"])["kParamValue"] = float(v)

    # --- velocity / note curves -----------------------------------------------
    scal = st.setdefault("scalars", {"note": {}, "velo": {}})
    if patch.velo_curve:
        scal["velo"] = dict(curve_to_serum(patch.velo_curve), legato=True)
    if patch.note_curve:
        scal["note"] = dict(curve_to_serum(patch.note_curve), legato=True)

    # --- macros -----------------------------------------------------------------
    for i, v in enumerate(patch.macros[:N_MACROS]):
        if v:
            _pp(st[f"Macro{i}"])["kParamValue"] = float(v * 100)

    # --- voice control ----------------------------------------------------------
    vc = patch.voice
    vpp = None
    if vc.rand_pan or vc.rand_detune or vc.rand_cutoff or vc.rand_env or vc.env_scale != 1 or vc.lfo_scale != 1 or vc.steps:
        vpp = _pp(st.setdefault("VoicePanel0", {"plainParams": "default"}))
        for attr, key, scale in (("rand_pan", "kParamGlobalRandomOscPan", 100), ("rand_detune", "kParamGlobalRandomOscDetune", 1),
                                 ("rand_cutoff", "kParamGlobalRandomFilterCutoff", 100),
                                 ("rand_env", "kParamGlobalRandomEnvTime", 100)):
            if getattr(vc, attr):
                vpp[key] = float(getattr(vc, attr) * scale)
        if vc.env_scale != 1:
            vpp["kParamGlobalScalingEnvTime"] = float(vc.env_scale * 100)
        if vc.lfo_scale != 1:
            vpp["kParamGlobalScalingLfoTime"] = float(vc.lfo_scale * 100)
        if vc.steps:
            vpp["kParamVoiceCount"] = float(len(vc.steps))
            for j, stp in enumerate(vc.steps[:8]):
                for key, name, scale in (("detune", "Detune", 1), ("pan", "Pan", 100), ("cutoff", "FilterCutoff", 25),
                                         ("env", "EnvTime", 100), ("mod1", "Mod1", 100), ("mod2", "Mod2", 100)):
                    if stp.get(key):
                        vpp[f"kParamVoice{j + 1}{name}"] = float(stp[key] * scale)

    # --- arpeggiator ----------------------------------------------------------
    if patch.arp.enabled and not any(k.startswith("Arp") for k in (patch.serum_extra or {})):
        a = patch.arp
        _pp(st["Arp0"])["kParamEnabled"] = 1.0
        apc = _pp(st["ArpClip0"])
        apc["kParamShape"] = a.shape
        divs = ["1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64"]
        base = a.rate.rstrip("TD")
        apc["kParamRate"] = float(divs.index(base) / 8 if base in divs else 0.5)
        if a.rate.endswith("T"):
            apc["kParamTriplets"] = 1.0
        if a.rate.endswith("D"):
            apc["kParamDotted"] = 1.0
        apc.update({"kParamGate": float(a.gate * 100), "kParamTransposeShift": float(a.shift),
                    "kParamTransposeRange": float(a.range), "kParamRepeats": float(a.repeats),
                    "kParamChance": float(a.chance * 100)})
        if a.offset:
            apc["kParamOffset"] = float(a.offset)
        if a.transpose_shape != "Up":
            apc["kParamTransposeShape"] = a.transpose_shape

    # --- clips (the reconstructed riff becomes the preview clip) ----------------
    for ci, c in enumerate(patch.clips[:12]):
        if f"MidiClip{ci}" in (patch.serum_extra or {}):
            continue
        notes = [{"attributes": [float(np.clip(n[3], 0, 1)), 1.0, 0.0, 0.5, 0.0, 0.5, 0.0, 0.5039370078740157],
                  "channel": 0, "expressionEvents": [None] * 5, "length": float(n[1]),
                  "noteNum": int(round(n[2])), "timeStamp": float(n[0])} for n in c.notes]
        st[f"MidiClip{ci}"] = {"clip": {"notes": notes, "regionEndBeats": float(c.length_beats)},
                               "displayLength_Beats": float(c.length_beats), "gridWidth_Beats": float(c.length_beats),
                               "name": c.name or f"Clip {ci + 1}", "plainParams": "default"}
    if patch.clips:
        cp = st.setdefault("ClipPlayer", {})
        cp["kUIParamPreviewClip"] = float((patch.preview_clip + 1) / 12)
        cp["kUIParamSelectedClip"] = float(patch.preview_clip / 11)

    # --- FX racks -------------------------------------------------------------
    legacy = _legacy_chain(patch)
    main_chain = legacy + list(patch.fx_chain)
    st["FXRack0"]["FX"] = chain_to_rack(main_chain, sc, patch.bpm)
    for b in range(2):
        chain = patch.fx_buses[b] if b < len(patch.fx_buses) else []
        st[f"FXRack{b + 1}"]["FX"] = chain_to_rack(chain, sc, patch.bpm)
    # flat ids of every FX module for modulation targets
    fx_paths = {}
    n_legacy = len(legacy)
    for j, p_ in enumerate(_flat_paths(patch.fx_chain, "fx_chain")):
        fx_paths[p_] = (n_legacy + j, _get_kind(patch, p_))
    for b in range(2):
        chain = patch.fx_buses[b] if b < len(patch.fx_buses) else []
        for j, p_ in enumerate(_flat_paths(chain, f"fx_buses.{b}")):
            fx_paths[p_] = (100 * (b + 1) + j, _get_kind(patch, p_))

    # --- mod matrix -----------------------------------------------------------
    _write_mod_matrix(st, patch, sc, fx_paths, legacy)

    # macro labels make the important knobs easy to find
    if not any(k.startswith("Macro") for k in (patch.serum_extra or {})):
        labels = ["CUTOFF", "RESO", "DRIVE", "WT POS"]
        for i, lab in enumerate(labels):
            st[f"Macro{i}"].setdefault("name", lab)
    return st


def _get_kind(patch: Patch, path: str) -> str:
    obj = patch
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else (obj.get(part) if isinstance(obj, dict) else getattr(obj, part))
    return obj.get("type")


def _write_mod_matrix(st, patch, sc, fx_paths, legacy):
    slot = 0

    def add(src, aux, mtype, mid, pname, amount, m):
        nonlocal slot
        while slot < 64 and f"ModSlot{slot}" in st and "source" in st[f"ModSlot{slot}"]:
            slot += 1
        if slot >= 64:
            return
        pp = {"kParamAmount": float(np.clip(amount, -100, 100))}
        if m.bipolar:
            pp["kParamBipolar"] = 1.0
        if m.curve:
            pp["kParamCurveIn"] = float(m.curve * 100)
        if m.aux_curve:
            pp["kParamAuxCurve"] = float(m.aux_curve * 100)
        if m.aux_invert:
            pp["kParamAuxInverted"] = 1.0
        if m.smooth_rise:
            pp["kParamSmoothRise"] = float(min(m.smooth_rise * 100, 100))
        if m.smooth_fall:
            pp["kParamSmoothFall"] = float(min(m.smooth_fall * 100, 100))
        if m.output != 1.0:
            pp["kParamOut"] = float(m.output * 100)
        if m.bypass:
            pp["kParamBypass"] = 1.0
        entry = {"destModuleID": int(mid), "destModuleParamID": PARAM_IDS.get(f"{mtype}.{pname}", 0),
                 "destModuleParamName": pname, "destModuleTypeString": mtype,
                 "plainParams": pp, "source": [_src_id(src), _src_id(aux)]}
        if m.curve_points or m.aux_curve_points:
            entry["flex"] = [curve_to_serum(m.curve_points) if m.curve_points else {},
                             curve_to_serum(m.aux_curve_points) if m.aux_curve_points else {}]
            if m.curve_points:
                pp["kParamMainCurveData"] = 1.0
            if m.aux_curve_points:
                pp["kParamAuxCurveData"] = 1.0
        st[f"ModSlot{slot}"] = entry
        slot += 1

    span = cutoff_octaves_span(sc)
    active = [i for i, o in enumerate(patch.oscs[:3]) if o.enabled]
    n_legacy = len(legacy)
    top_index = [n_legacy + k for k in _top_level_rack_index(patch.fx_chain)]
    for m in patch.mods:
        if not _src_id(m.source):
            continue
        a, d, aux = m.amount, m.dest, m.aux
        if d.startswith("@"):
            t = dest_to_serum(patch, d[1:], fx_paths)
            if t:
                add(m.source, aux, t[0], t[1], t[2], a * 100, m)
            continue
        if d in ("filter.cutoff", "filter1.cutoff", "filter2.cutoff"):
            add(m.source, aux, "VoiceFilter", 1 if d.startswith("filter2") else 0, "kParamFreq",
                a * 10.0 / span * 100, m)
        elif d.startswith(("filter.", "filter1.", "filter2.")):
            what = d.split(".")[1]
            pname = {"reso": "kParamReso", "drive": "kParamDrive", "var": "kParamVar"}.get(what)
            if pname:
                add(m.source, aux, "VoiceFilter", 1 if d.startswith("filter2") else 0, pname, a * 100, m)
        elif d.startswith("osc") and "." in d:
            i = int(d[3]) - 1
            what = d.split(".")[1]
            sub = OSC_SUB.get(patch.oscs[i].type if i < len(patch.oscs) else "wavetable", "WTOsc")
            row = {"pos": ("WTOsc", "kParamTablePos"), "level": ("Oscillator", "kParamVolume"),
                   "pan": ("Oscillator", "kParamPan"), "fine": ("Oscillator", "kParamFine"),
                   "warp": (sub, "kParamWarp"), "warp2": (sub, "kParamWarp2")}.get(what)
            if row:
                add(m.source, aux, row[0], i, row[1], a * 100, m)
        elif d == "amp":
            for i in active + ([3] if patch.noise.enabled else []) + ([4] if patch.sub.enabled else []):
                add(m.source, aux, "Oscillator", i, "kParamVolume", a * 100, m)
        elif d == "voiceamp":
            add(m.source, aux, "Global", 0, "kParamVoiceAmp", a * 100, m)
        elif d == "pitch":
            semis = a * 48.0
            for i in active + ([4] if patch.sub.enabled else []):
                add(m.source, aux, "Oscillator", i, "kParamCoarsePit", semis / sc["pitch_semis_per_100"] * 100, m)
        elif d == "noise.level":
            add(m.source, aux, "Oscillator", 3, "kParamVolume", a * 100, m)
        elif d == "sub.level":
            add(m.source, aux, "Oscillator", 4, "kParamVolume", a * 100, m)
        elif d.startswith("fx."):
            parts = d.split(".")
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) < len(patch.fx_chain):
                j = int(parts[1])
                mdl = patch.fx_chain[j]
                mtype = FX_KEYS.get(mdl["type"])
                pname = {"mix": "kParamWet", "drive": "kParamDrive"}.get(parts[2])
                if mtype and pname:
                    add(m.source, aux, mtype, top_index[j], pname, a * 100, m)
    for raw in patch.raw_mods:                          # imported slots we don't interpret
        while slot < 64 and f"ModSlot{slot}" in st and "source" in st[f"ModSlot{slot}"]:
            slot += 1
        if slot < 64:
            st[f"ModSlot{slot}"] = copy.deepcopy(raw)
            slot += 1


def _display_name(tid: str) -> str:
    kind, _, name = tid.partition(":")
    if kind == "basic":
        return f"Basic {name.title()}"
    return f"FA Custom {name}"


def build_preset_bytes(patch: Patch, bank: TableBank | None = None, name: str | None = None,
                       description: str = "", samples=None) -> bytes:
    from .preset import write_preset_bytes
    st = patch_to_state(patch, bank, samples)
    types = {o.type for o in patch.oscs if o.enabled}
    tags = ["Bass", "Mono" if patch.mono else "Poly"]
    tags += [{"wavetable": "Wavetable", "sample": "Sample", "granular": "Granular", "spectral": "Spectral",
              "multisample": "Multisample"}[t] for t in sorted(types)]
    if any(o.enabled and o.table.startswith(("custom:", "basic:")) for o in patch.oscs) or \
            any(o.enabled and o.smp.source.startswith("embedded:") for o in patch.oscs):
        tags.append("Embedded-Data")
    return write_preset_bytes(st, name or patch.name, author="Fourier Analyzer",
                              description=description, tags=tags)


# ---------------------------------------------------------------------------
# Serum -> Patch

PASSTHROUGH = ("ArpClip", "MidiClip", "ClipPlayer", "Arp0", "midiMap", "RetriggerState", "clipBankDisplayName",
               "arpBankDisplayName", "mpeConfig", "mpeEnabled", "mpePitchBendRange", "Macro")


def _read_osc(st, i, sc, bank, samples) -> tuple[Osc, str | None]:
    mod = st.get(f"Oscillator{i}", {})
    en = _pget(mod, "kParamEnable", 1.0 if i == 0 else 0.0)
    kind = _pget(mod, "kParamType", "kOsc_WT")
    otype = {v: k for k, v in OSC_TYPES.items()}.get(kind, "wavetable")
    g = lambda k, d=None: _pget(mod, k, d)
    o = Osc(enabled=bool(en), type=otype)
    o.octave = int(g("kParamOctave", 0))
    o.semi = int(round(g("kParamPitch", 0)))
    o.coarse = float(g("kParamCoarsePit", 0.0))
    o.fine = float(g("kParamFine", 0.0))
    o.level = float(g("kParamVolume", 0.75))
    o.pan = float(g("kParamPan", 0.0)) / 50
    pm = g("kParamPitchMode")
    if pm in ("Harmonics", "Ratio"):
        o.pitch_mode = pm.lower()
        o.ratio = float(g("kParamPitchRatio", 1.0)) or 1.0
        if g("kParamPitchSource") is not None:
            o.ratio_source = int(g("kParamPitchSource"))
    o.hz_offset = float(g("kParamHzOffset", 0.0))
    o.pitch_track = g("kParamPitchTrack", 1.0) != 0.0
    o.bend_track = g("kParamPitchBendTrack", 1.0) != 0.0
    o.unison = int(g("kParamUnison", 1))
    o.detune = float(g("kParamDetune", 0.25))
    o.blend = float(g("kParamDetuneWid", 75)) / 100
    o.unison_mode = {v: k for k, v in DETUNE_MODES.items()}.get(g("kParamDetuneMode"), "linear")
    o.unison_width = float(g("kParamUnisonStereo", 100)) / 100
    o.unison_warp = float(g("kParamUnisonWarp", 0)) / 100
    o.unison_warp2 = float(g("kParamUnisonWarp2", 0)) / 100
    o.unison_stack = _unk(g("kParamUnisonStack")) if g("kParamUnisonStack") else "none"
    o.unison_range = float(g("kParamUnisonRange", 2.0))
    o.key_range = (int(g("kParamKeyZoneMin", 0)), int(g("kParamKeyZoneMax", 127)) if g("kParamKeyZoneMax", 127) < 126 else 127)
    o.vel_range = (int(g("kParamVelocityZoneMin", 0)), int(g("kParamVelocityZoneMax", 127)))
    o.key_warp = g("kParamKeyZoneWarp") == 1.0
    sub = mod.get(f"{OSC_SUB[otype]}{i}", {})
    spp = sub.get("plainParams") if isinstance(sub.get("plainParams"), dict) else {}
    o.warp = _unk(spp.get("kParamWarpMenu")) if spp.get("kParamWarpMenu") else "none"
    o.warp_amount = float(spp.get("kParamWarp", 0.0)) if spp.get("kParamWarpMenu") else 0.0
    o.warp_var = float(spp.get("kParamWarpVar", 0.5))
    o.warp2 = _unk(spp.get("kParamWarpMenu2")) if spp.get("kParamWarpMenu2") else "none"
    o.warp2_amount = float(spp.get("kParamWarp2", 0.0)) if spp.get("kParamWarpMenu2") else 0.0
    o.warp2_var = float(spp.get("kParamWarpVar2", 0.5))
    note = None
    if otype == "wavetable":
        wt = sub
        wpp = spp
        pos = (wpp.get("kParamTablePos", sc["tablepos_lo"]) - sc["tablepos_lo"]) / (sc["tablepos_hi"] - sc["tablepos_lo"])
        o.position = float(np.clip(pos, 0, 1))
        o.random_phase = float(wpp.get("kParamRandomPhase", 100.0)) / 100
        o.phase = float(wpp.get("kParamInitialPhase", 0.0)) / 360
        o.phase_memory = {v: k for k, v in PHASE_MEMORY.items()}.get(wpp.get("kParamPhaseMemory"), "all")
        o.smooth_interp = wpp.get("kParamXfadeMode") == 1.0
        o.unison_wtpos = float(wpp.get("kParamUnisonWTPos", 0.0)) / 100
        rc = curve_from_serum(wt.get("flex"))
        if rc and len(rc) > 2:
            o.remap_curve = rc
        if "embeddedWTData" in wt and bank is not None:
            data = np.asarray(wt["embeddedWTData"], dtype=np.float32)
            nfr = max(1, len(data) // FRAME)
            o.table = bank.add_custom(data[: nfr * FRAME].reshape(nfr, FRAME))
        elif "relativePathToWT" in wt:
            o.table = "serum:" + wt["relativePathToWT"].lstrip("/")
        return o, None
    cfg = SampleCfg()
    if otype == "multisample":
        o.multi = MultiCfg(sfz=sub.get("embedded_sfz", ""), folder=str(Path(sub.get("sfzPathRelative", "")).parent)
                           if sub.get("sfzPathRelative") else "",
                           timbre=float(spp.get("kParamTimbreShift", 0.0)))
        env = {k[9:].lower(): float(v) for k, v in spp.items() if k.startswith("kParamEnv") and k != "kParamEnvOverride"}
        if env:
            o.multi.env, o.multi.env_override = env, True
        return o, None
    if "embedded_audio" in sub and samples is not None:
        ea = sub["embedded_audio"]
        data = np.asarray(ea.get("data") or [[0.0]], dtype=np.float32)
        cfg.source = samples.add(data, int(ea.get("sampleRate", 44100)), name=ea.get("table_name", ""))
    elif sub.get("samplePathRelative"):
        cfg.source = "serum:" + sub["samplePathRelative"]
    if "baseNote" in sub:
        cfg.root = float(sub["baseNote"])
    cfg.detune = float(sub.get("baseDetune", 0.0))
    cfg.start = float(g("kParamStart", 0.0)) / 100
    cfg.end = float(g("kParamEnd", 100.0)) / 100
    cfg.loop = {v: k for k, v in LOOP_MODES.items()}.get(g("kParamLoopMode"), "off")
    cfg.loop_start = float(g("kParamLoopStart", 0.0)) / 100
    cfg.loop_end = float(g("kParamLoopEnd", 100.0)) / 100
    cfg.crossfade = float(g("kParamLoopCrossfade", 0.0)) / 100
    cfg.scan = float(g("kParamScanRate", 100.0)) / 100
    cfg.scan_range = int(g("kParamScanRange", 0))
    cfg.reverse = g("kParamReverse") == 1.0
    cfg.position = float(g("kParamPosition", 0.0)) / 100
    cfg.span = float(g("kParamUnisonSpan", 0.0)) / 100
    cfg.random_start = float(g("kParamRandomStart", 0.0)) / 100
    cfg.link_loop = g("kParamLinkLoopLength") == 1.0
    cfg.relative_loop = g("kParamLoopStartLink") == 1.0
    cfg.exit_on_release = g("kParamLoopEndsAtRelease") == 1.0
    cfg.manual_position = g("kParamManualPositionMode") == 1.0
    cfg.scan_tempo_lock = g("kParamScanTempoLock") == 1.0
    cfg.base_tempo = float(g("kParamBaseTempo", 0.0))
    cfg.slicing = str(g("kParamSlicingEnabled", "off")).lower()
    cfg.slices = [float(x) for x in mod.get("sliceMarkers", [])] if isinstance(mod.get("sliceMarkers"), list) else []
    o.smp = cfg
    if otype == "granular":
        gc = GranularCfg()
        gc.density = float(spp.get("kParamDensity", 20.0))
        gc.density_mode = {"kDensityBPM": "bpm", "kDensityGrains": "grains"}.get(spp.get("kParamDensityMode"), "free")
        gc.length = float(spp.get("kParamGrainLength", 0.08))
        gc.length_mode = {"kLengthBPM": "bpm", "kLengthPercent": "percent"}.get(spp.get("kParamLengthMode"), "free")
        gc.window = str(spp.get("kParamWindowShape", "kWindowHann")).replace("kWindow", "")
        gc.window_amount = float(spp.get("kParamWindowParam", 100.0)) / 100
        gc.window_skew = float(spp.get("kParamWindowSkew", 0.0)) / 100
        for attr, key, scale in (("rand_offset", "kParamRandomOffset", 100), ("rand_dir", "kParamRandomDir", 100),
                                 ("rand_pitch", "kParamRandomPitch", 1 / 12), ("rand_length", "kParamRandomGrainLength", 100),
                                 ("rand_pan", "kParamRandomPan", 100), ("rand_gain", "kParamRandomGain", 100),
                                 ("rand_window", "kParamRandomWindowAmount", 100),
                                 ("rand_skew", "kParamRandomWindowSkew", 100), ("rand_warp", "kParamRandomWarp", 100),
                                 ("rand_warp2", "kParamRandomWarp2", 100)):
            if key in spp:
                setattr(gc, attr, float(spp[key]) / scale)
        gc.spawn = {v: k for k, v in SPAWN.items()}.get(spp.get("kParamUnisonTrigPattern"), "together")
        o.gran = gc
    elif otype == "spectral":
        s2 = SpectralCfg(freq_lo=float(spp.get("kParamFreqLo", 20.0)), freq_hi=float(spp.get("kParamFreqHi", 20000.0)),
                         smooth_edges=spp.get("kParamLoHiIsSmooth") == 1.0, post_warp=spp.get("kParamPostWarp") == 1.0,
                         phase_lock=spp.get("kParamPhaseLock") == 1.0, transients=spp.get("kParamTransients") == 1.0,
                         filter_shift=float(spp.get("kParamSpecFltShift", 0.0)),
                         filter_mix=float(spp.get("kParamSpecFltWetDry", 0.0)) / 100,
                         filter_curve=curve_from_serum(sub.get("flex")))
        o.spec = s2
    if en and not cfg.source:
        note = f"osc{i + 1}:{otype} sample missing"
    return o, note


def state_to_patch(st: dict, bank: TableBank | None = None, bpm: float = 140.0, samples=None) -> Patch:
    """Import a Serum 2 preset state. Embedded wavetables / samples are added
    to ``bank`` / ``samples`` (when given); referenced factory content
    becomes serum:<path> ids that render where Serum 2 is installed."""
    sc = scales()
    p = Patch(bpm=bpm)
    notes = []
    g = st.get("Global0", {})
    gp = lambda k, d=None: _pget(g, k, d)
    p.mono = gp("kParamMonoToggle", 0.0) == 1.0
    p.legato = gp("kParamLegato", 0.0) == 1.0
    p.poly = int(gp("kParamPolyCount", 16))
    p.same_note_limit = gp("kParamLimitSameNotePolyphony", 0.0) == 1.0
    p.glide = float(gp("kParamPortamentoTime", 0.0))
    p.glide_curve = float(gp("kParamPortamentoCurve", 50.0)) / 100
    p.glide_always = gp("kParamPortaAlways", 0.0) == 1.0
    p.glide_scaled = gp("kParamPortaScaled", 0.0) == 1.0
    p.master = float(gp("kParamMasterVolume", 0.35)) * 2
    p.direct_level = float(gp("kParamDirectVol", 1.0))
    p.transpose = int(gp("kParamTranspose", 0))
    p.tuning_a4 = float(gp("kParamGlobalTuning", 440.0))
    p.master_tune = float(gp("kParamMasterTuning", 0.0))
    p.modwheel = float(gp("kParamModWheel", 0.0)) / 100
    p.swing = float(gp("kParamSwing", 0.0)) / 100
    p.bend_range = (int(gp("kParamBendRangeUp", 2)), int(gp("kParamBendRangeDn", -2)))
    p.bus_dest = [{0.0: "main", 1.0: "direct", 2.0: "bus2" if b == 0 else "bus1"}.get(gp(f"kParamFXBus{b + 1}Dest", 0.0), "main")
                  for b in range(2)]
    if isinstance(st.get("tuningData"), list) and len(st["tuningData"]) >= 128:
        p.tuning_table = [float(x) for x in st["tuningData"][:128]]
    pq = st.get("PitchQuantizer0", {})
    if isinstance(pq.get("plainParams"), dict) and pq["plainParams"].get("kParamScale") and pq.get("scale"):
        p.scale_key = int(pq["plainParams"].get("kParamKey", 0))
        p.scale_notes = [bool(x) for x in pq["scale"]][:12]
    for i in range(3):
        o, note = _read_osc(st, i, sc, bank, samples)
        p.oscs[i] = o
        if note:
            notes.append(note)
    nmod = st.get("Oscillator3", {})
    if _pget(nmod, "kParamEnable", 0.0):
        nz = nmod.get("NoiseOsc3", {})
        ng = lambda k, d=None: _pget(nz, k, d)
        kind = ng("kParamNoiseType")
        rel = nz.get("relativePathToNoiseSample")
        p.noise = NoiseOsc(enabled=True, color=float(ng("kParamColor", 0.5)),
                           level=float(_pget(nmod, "kParamVolume", 0.75)),
                           kind=kind if kind else ("sample" if rel else "White"),
                           sample="serum:" + rel if (rel and not kind) else "",
                           one_shot=ng("kParamOneShot") == 1.0, start=float(ng("kParamInitialPhase", 0.0)) / 100,
                           random_start=float(ng("kParamRandomPhase", 100.0)) / 100,
                           fine=float(ng("kParamFine", 0.0)) * 100, pitch=float(_pget(nmod, "kParamPitch", 0.0)),
                           keytrack=_pget(nmod, "kParamPitchTrack", 0.0) == 1.0,
                           pan=float(_pget(nmod, "kParamPan", 0.0)) / 50)
    smod = st.get("Oscillator4", {})
    if _pget(smod, "kParamEnable", 0.0):
        so = smod.get("SubOsc4", {})
        shape = _pget(so, "kParamShape")
        inv = {v: k for k, v in SUB_NAMES.items() if v}
        p.sub = SubOsc(enabled=True, shape=inv.get(shape, "sine"), octave=int(_pget(smod, "kParamOctave", -1)),
                       coarse=float(_pget(smod, "kParamCoarsePit", 0.0)),
                       level=float(_pget(smod, "kParamVolume", 0.75)), pan=float(_pget(smod, "kParamPan", 0.0)) / 50,
                       phase=float(_pget(so, "kParamInitialPhase", 0.0)) / 360,
                       contiguous=_pget(so, "kParamContiguousPhase") == 1.0)
    for idx, attr in enumerate(("filter", "filter2")):
        f = st.get(f"VoiceFilter{idx}", {})
        fg = lambda k, d=None: _pget(f, k, d)
        if fg("kParamEnable", 1.0 if idx == 0 else 0.0) == 1.0 and isinstance(f.get("plainParams"), dict):
            setattr(p, attr, Filter(enabled=True, type=fg("kParamType", "MgL6"),
                                    cutoff=norm_to_cutoff(fg("kParamFreq", 1.0), sc),
                                    reso=float(fg("kParamReso", 0)) / 100,
                                    drive=float(fg("kParamDrive", 0)) / 100,
                                    keytrack=1.0 if fg("kParamKeyTrack") else 0.0,
                                    var=float(fg("kParamVar", 50)) / 100, mix=float(fg("kParamWet", 100)) / 100,
                                    level=2 * float(fg("kParamLevelOut", 0.5)), pan=(float(fg("kParamStereo", 50)) - 50) / 50,
                                    clean=fg("kParamPad") == 1.0, pz=f.get("PZs"),
                                    x=float(fg("kParamX", 0.0)), y=float(fg("kParamY", 0.0))))
    routing = {}
    for name, slot in ROUTE_SLOT.items():
        r = st.get(f"RoutingSlot{slot}", {})
        dest = _pget(r, "kParamRoutingDest")
        entry = {}
        if dest in ROUTE_DEST_INV:
            entry["dest"] = ROUTE_DEST_INV[dest]
        elif name.startswith("filter"):
            entry["dest"] = "main"
        bal = _pget(r, "kParamFilterBalance")
        if bal is not None:
            entry["balance"] = (float(bal) + 100) / 200
        for b in (1, 2):
            v = _pget(r, f"kParamFXBus{b}Level")
            if v:
                entry[f"send{b}"] = float(v) / 100
        if _pget(r, "kParamViaEnv1") == 0.0:
            entry["env1"] = False
        if entry:
            routing[name] = entry
    # Serum's defaults: osc A -> filter 1, everything else -> main
    for name in ("osc2", "osc3", "noise", "sub"):
        routing.setdefault(name, {}).setdefault("dest", "main")
    routing.setdefault("osc1", {}).setdefault("dest", "filter")
    p.routing = routing
    for i in range(N_ENVS):
        e = st.get(f"Env{i}", {})
        p.envs[i] = Env(attack=_pget(e, "kParamAttack", 0.001), hold=_pget(e, "kParamHold", 0.0),
                        decay=_pget(e, "kParamDecay", 1.0), sustain=_pget(e, "kParamSustain", 1.0 if i == 0 else 0.0),
                        release=_pget(e, "kParamRelease", 0.01),
                        attack_curve=_pget(e, "kParamCurve1", 50.0) / 100,
                        decay_curve=_pget(e, "kParamCurve2", 66.6) / 100,
                        release_curve=_pget(e, "kParamCurve3", 66.6) / 100,
                        start=float(_pget(e, "kParamStart", 0.0)), end=float(_pget(e, "kParamEnd", 0.0)),
                        legato_inverted=_pget(e, "kParamLegatoInverted") == 1.0)
    for k in range(N_LFOS):
        l = st.get(f"LFO{k}", {})
        cd = l.get("curveData") or {}
        pts = curve_from_serum(cd)
        loop = None
        if isinstance(cd.get("loopbackPointNum"), int) and pts and 0 <= cd["loopbackPointNum"] < len(pts):
            loop = pts[cd["loopbackPointNum"]][0]
        mode = {"Free": "free", "Envelope": "env"}.get(_pget(l, "kParamMode"), "trig")
        typ = {v: k2 for k2, v in LFO_TYPE_NAMES.items()}.get(_pget(l, "kParamType"), "shape")
        rate = float(_pget(l, "kParamRate", 2.0))
        path = curve_from_serum(l.get("pathData"))
        p.lfos[k] = LFO(shape="custom" if pts else "triangle", rate_hz=max(rate, 0.01), mode=mode, points=pts,
                        phase=float(_pget(l, "kParamPhase", 0.0)) / 360, type=typ,
                        rise=float(_pget(l, "kParamRise", 0.0)), delay=float(_pget(l, "kParamDelay", 0.0)),
                        smooth=float(_pget(l, "kParamSmooth", 0.0)) / 100,
                        direction={1.0: "reverse", 2.0: "pingpong"}.get(_pget(l, "kParamDirection", 0.0), "forward"),
                        loopback=loop if mode == "env" else None, mono=_pget(l, "kParamMono") == 1.0,
                        anchored=_pget(l, "kParamAnchored") == 1.0, swing=_pget(l, "kParamSwing") == 1.0,
                        path=path, path_open=bool((l.get("pathData") or {}).get("isOpen", False)))
    for a in st.get("lfoPointModAssignments") or []:
        k = int(a.get("lfoID", 0))
        if k < N_LFOS:
            p.lfos[k].point_mods.append({"point": int(a.get("pointID", 0)), "bus": int(a.get("busID", 0)),
                                         "axis": {0: "x", 1: "y", 2: "curve"}.get(a.get("target", 1), "y"),
                                         "path": a.get("lfoType") == 1})
    p.lfo_buses = [float(_pget(st.get(f"LFOPointModBus{b}", {}), "kParamValue", 0.0)) for b in range(16)]
    scal = st.get("scalars") or {}
    p.velo_curve = curve_from_serum(scal.get("velo"))
    p.note_curve = curve_from_serum(scal.get("note"))
    p.macros = [float(_pget(st.get(f"Macro{i}", {}), "kParamValue", 0.0)) / 100 for i in range(N_MACROS)]
    vp = st.get("VoicePanel0", {})
    vg = lambda k, d=0.0: float(_pget(vp, k, d))
    vc = VoiceControl(rand_pan=vg("kParamGlobalRandomOscPan") / 100,
                      rand_detune=vg("kParamGlobalRandomOscDetune") * (10 if _pget(vp, "kParamGlobalRandomOscDetune10x") else 1),
                      rand_cutoff=vg("kParamGlobalRandomFilterCutoff") / 100, rand_env=vg("kParamGlobalRandomEnvTime") / 100,
                      env_scale=vg("kParamGlobalScalingEnvTime", 100.0) / 100,
                      lfo_scale=vg("kParamGlobalScalingLfoTime", 100.0) / 100)
    nsteps = int(vg("kParamVoiceCount", 0))
    steps = []
    for j in range(8):
        stp = {}
        for key, name, scale in (("detune", "Detune", 1), ("pan", "Pan", 100), ("cutoff", "FilterCutoff", 25),
                                 ("env", "EnvTime", 100), ("mod1", "Mod1", 100), ("mod2", "Mod2", 100)):
            v = _pget(vp, f"kParamVoice{j + 1}{name}")
            if v:
                stp[key] = float(v) / scale
        steps.append(stp)
    if any(steps):
        last = max(j for j, s_ in enumerate(steps) if s_) + 1
        vc.steps = steps[:max(nsteps, last)]
    p.voice = vc
    # arpeggiator
    arp0 = st.get("Arp0", {})
    if _pget(arp0, "kParamEnabled") == 1.0:
        cid = int(_pget(arp0, "kParamActiveClipID", 0))
        ac = st.get(f"ArpClip{cid}", {})
        ag = lambda k, d=None: _pget(ac, k, d)
        divs = ["1/1", "1/2", "1/4", "1/8", "1/16", "1/32", "1/64"]
        rate = divs[int(np.clip(round(float(ag("kParamRate", 0.5)) * 8), 0, len(divs) - 1))]
        rate += "T" if ag("kParamTriplets") == 1.0 else "D" if ag("kParamDotted") == 1.0 else ""
        p.arp = Arp(enabled=True, shape=str(ag("kParamShape", "Up")), rate=rate,
                    gate=float(ag("kParamGate", 75.0)) / 100, shift=int(ag("kParamTransposeShift", 12)),
                    range=int(ag("kParamTransposeRange", 1)), transpose_shape=str(ag("kParamTransposeShape", "Up")),
                    repeats=int(ag("kParamRepeats", 1)), offset=int(ag("kParamOffset", 0)),
                    chance=float(ag("kParamChance", 100.0)) / 100)
        pat = ac.get("clip", {}).get("notes") if isinstance(ac.get("clip"), dict) else None
        if pat:
            p.arp.pattern = [[n.get("timeStamp", 0.0), n.get("length", 0.25), n.get("noteNum", 0),
                              (n.get("attributes") or [1.0])[0]] for n in pat]
    for ci in range(12):
        mc = st.get(f"MidiClip{ci}", {})
        cl = mc.get("clip") if isinstance(mc.get("clip"), dict) else None
        if cl and cl.get("notes"):
            p.clips.append(Clip(notes=[[n.get("timeStamp", 0.0), n.get("length", 0.25), n.get("noteNum", 60),
                                        (n.get("attributes") or [1.0])[0]] for n in cl["notes"]],
                                length_beats=float(mc.get("displayLength_Beats", cl.get("regionEndBeats", 4.0))),
                                name=mc.get("name", "")))
    # keep the parts we don't interpret for a faithful re-export
    p.serum_extra = {k: copy.deepcopy(v) for k, v in st.items() if k.startswith(PASSTHROUGH)}

    # FX racks
    p.fx_chain = rack_to_chain(st.get("FXRack0", {}).get("FX", []), sc, bpm, samples)
    p.fx_buses = [rack_to_chain(st.get(f"FXRack{b}", {}).get("FX", []), sc, bpm, samples) for b in (1, 2)]
    p.bus_levels = [float(gp("kParamFXBus1Vol", 1.0)), float(gp("kParamFXBus2Vol", 1.0))]
    fx_flat = {}
    for j, path in enumerate(_flat_paths(p.fx_chain, "fx_chain")):
        fx_flat[j] = (path, _get_kind(p, path))
    for b in range(2):
        for j, path in enumerate(_flat_paths(p.fx_buses[b], f"fx_buses.{b}")):
            fx_flat[100 * (b + 1) + j] = (path, _get_kind(p, path))
    top = _top_level_rack_index(p.fx_chain)

    # mod matrix
    span = cutoff_octaves_span(sc)
    mods, raw = [], []
    for k in range(64):
        ms = st.get(f"ModSlot{k}", {})
        if "source" not in ms:
            continue
        src_ids = ms["source"]
        src = _src_name(src_ids[0])
        aux = _src_name(src_ids[1]) if len(src_ids) > 1 else None
        if src is None:
            continue
        pp = ms.get("plainParams") if isinstance(ms.get("plainParams"), dict) else {}
        amt = float(pp.get("kParamAmount", 0.0)) / 100
        t, pn, mid = ms.get("destModuleTypeString"), ms.get("destModuleParamName"), ms.get("destModuleID", 0)
        flex = ms.get("flex") or [{}, {}]
        common = dict(bipolar=pp.get("kParamBipolar") == 1.0, aux=aux, aux_invert=pp.get("kParamAuxInverted") == 1.0,
                      curve=float(pp.get("kParamCurveIn", 0.0)) / 100, aux_curve=float(pp.get("kParamAuxCurve", 0.0)) / 100,
                      curve_points=curve_from_serum(flex[0]) if pp.get("kParamMainCurveData") and len(flex) > 0 else None,
                      aux_curve_points=curve_from_serum(flex[1]) if pp.get("kParamAuxCurveData") and len(flex) > 1 else None,
                      smooth_rise=float(pp.get("kParamSmoothRise", 0.0)) / 100,
                      smooth_fall=float(pp.get("kParamSmoothFall", 0.0)) / 100,
                      output=float(pp.get("kParamOut", 100.0)) / 100, bypass=pp.get("kParamBypass") == 1.0)
        dest = None
        if t == "VoiceFilter" and pn == "kParamFreq":
            dest, amt = ("filter2" if mid == 1 else "filter") + ".cutoff", amt * span / 10
        elif t == "VoiceFilter" and pn in ("kParamReso", "kParamDrive", "kParamVar"):
            dest = ("filter2" if mid == 1 else "filter") + "." + {"kParamReso": "reso", "kParamDrive": "drive",
                                                                  "kParamVar": "var"}[pn]
        elif (t == "WTOsc" and mid < 3 and pn == "kParamTablePos") or \
                (t in OSC_SUB.values() and mid < 3 and pn in ("kParamWarp", "kParamWarp2")):
            dest = f"osc{mid + 1}." + {"kParamTablePos": "pos", "kParamWarp": "warp", "kParamWarp2": "warp2"}[pn]
        elif t == "Oscillator" and mid < 3 and pn in ("kParamVolume", "kParamPan", "kParamFine"):
            dest = f"osc{mid + 1}." + {"kParamVolume": "level", "kParamPan": "pan", "kParamFine": "fine"}[pn]
        elif t == "Oscillator" and mid < 5 and pn == "kParamCoarsePit" and mid != 3:
            dest, amt = "pitch", amt * sc["pitch_semis_per_100"] / 48
            if any(x.dest == "pitch" and x.source == src and x.aux == aux for x in mods):
                continue        # one pitch mod is written per oscillator; keep a single entry
        elif t == "Oscillator" and mid == 3 and pn == "kParamVolume":
            dest = "noise.level"
        elif t == "Oscillator" and mid == 4 and pn == "kParamVolume":
            dest = "sub.level"
        elif t == "Global" and pn == "kParamVoiceAmp":
            dest = "voiceamp"
        elif t and t.startswith("FX") and mid in top and pn in ("kParamWet", "kParamDrive") and \
                FX_BY_KEY.get(t) == (p.fx_chain[top.index(mid)]["type"] if top.index(mid) < len(p.fx_chain) else None):
            dest = f"fx.{top.index(mid)}." + ("mix" if pn == "kParamWet" else "drive")
        else:
            path = serum_to_dest(t, mid, pn, fx_flat) if t else None
            if path:
                dest = "@" + path
        if dest:
            mods.append(Mod(source=src, dest=dest, amount=amt, **common))
        else:
            raw.append(copy.deepcopy(ms))
    p.mods = mods
    p.raw_mods = raw
    p.voice_amp = float(gp("kParamVoiceAmp", 0.5))
    p.name = "Imported" + (" (" + "; ".join(notes) + ")" if notes else "")
    return p
