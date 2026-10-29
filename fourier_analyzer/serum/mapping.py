"""Convert between :class:`Patch` and a Serum 2 preset state (both ways).

Parameter names, module layout, enum values and modulation IDs come from the
structure of Serum 2.0.x presets (see param_ids.json). A few *unit scales*
(filter cutoff curve, LFO rate units, pitch-mod depth) are not stored in
presets; they live in ``SCALES`` and are replaced by Serum's own values when
"Calibrate with Serum 2" has been run.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .. import config
from ..synth.patch import (FX_DEFAULTS, LFO, N_ENVS, N_LFOS, N_MACROS, Env, Filter, Mod, NoiseOsc, Osc,
                           Patch, SubOsc, fx, sync_to_hz)
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
    "lfo_rate_is_hz": True,
    "lfo_y_inverted": False,
    "tablepos_lo": 1.0,
    "tablepos_hi": 256.0,
}

LEGACY_FILTERS = {"lp12": "L12", "lp24": "L24", "hp12": "H12", "hp24": "H24", "bp12": "BP12"}
LEGACY_DIST = {"soft": "SoftClip", "hard": "HardClip", "tube": "Tube", "sinefold": "SinFold",
               "linfold": "LinFold", "diode": "Diode1"}
LEGACY_WARPS = {"fm_b": "FM_OSC", "fm_sub": "FM_SUB", "am_b": "AM_OSC", "rm_b": "RM_OSC", "rm_sub": "RM_SUB"}
SUB_NAMES = {"sine": None, "triangle": "kTriangle", "saw": "kSaw", "square": "kSquare", "pulse": "kPulse",
             "roundrect": "kRoundRect"}
LFO_TYPE_NAMES = {"random": "RandomSH", "lorenz": "Lorenz", "rossler": "Rossler", "path": "Path"}
ROUTE_SLOT = {"osc1": 0, "osc2": 1, "osc3": 2, "noise": 3, "sub": 4}

# FX rack entry "type" numbers and their module keys
FX_TYPE_IDS = {"distortion": 0, "flanger": 1, "phaser": 2, "chorus": 3, "delay": 4, "comp": 5, "reverb": 6,
               "eq": 7, "filter": 8, "hyperdim": 9, "bode": 10, "convolve": 11, "utility": 12, "split": 13,
               "split3": 14, "splitms": 15}
FX_KEYS = {"distortion": "FXDistortion", "flanger": "FXFlanger", "phaser": "FXPhaser", "chorus": "FXChorus",
           "delay": "FXDelay", "comp": "FXComp", "reverb": "FXReverb", "eq": "FXEQ", "filter": "FXFilter",
           "hyperdim": "FXHyperD", "bode": "FXBode", "convolve": "FXConv", "utility": "FXUtils",
           "split": "FXSplit", "split3": "FXSplit3", "splitms": "FXSplitMS"}
FX_BY_ID = {v: k for k, v in FX_TYPE_IDS.items()}

# modulation sources (ModSlot "source": [id, aux_id])
SOURCE_IDS = {f"env{i}": 1 + i for i in range(1, N_ENVS + 1)}
SOURCE_IDS.update({f"lfo{i}": 5 + i for i in range(1, N_LFOS + 1)})
SOURCE_IDS.update({"velocity": 16, "note": 17, "modwheel": 18})
SOURCE_IDS.update({f"macro{i}": 24 + i for i in range(1, N_MACROS + 1)})
SOURCE_NAMES = {v: k for k, v in SOURCE_IDS.items()}


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

def _fx_to_serum(m: dict, sc: dict, bpm: float) -> dict:
    kind = m["type"]
    d = dict(FX_DEFAULTS.get(kind, {}))
    d.update(m)
    pp = {}
    if not m.get("enabled", True):
        pp["kParamEnable"] = 0.0
    if kind == "distortion":
        mode = LEGACY_DIST.get(d["mode"], d["mode"])
        if mode != "Tube":
            pp["kParamMode"] = _k(mode)
        pp.update({"kParamDrive": d["drive"] * 100, "kParamWet": d["mix"] * 100})
        if d.get("filter") in ("pre", "post"):
            pp.update({"kParamPrePost": 1.0 if d["filter"] == "pre" else 2.0, "kParamFreq": d["filter_freq"],
                       "kParamBW": 0.075 + 7.5 * d["filter_bw"]})
    elif kind == "flanger":
        pp.update({"kParamRate": d["rate_hz"], "kParamDepth": d["depth"] * 100,
                   "kParamFeedback": d["feedback"] * 100, "kParamWet": d["mix"] * 100})
    elif kind == "phaser":
        pp.update({"kParamRate": d["rate_hz"], "kParamDepth": d["depth"] * 100, "kParamFreq": d["freq"],
                   "kParamFeedback": d["feedback"] * 100, "kParamNumPoles": float(d["stages"]),
                   "kParamWet": d["mix"] * 100})
    elif kind == "chorus":
        pp.update({"kParamRate": d["rate_hz"], "kParamDepth": d["depth"] * 25, "kParamDelay": d["delay_ms"],
                   "kParamFeedback": d["feedback"] * 100, "kParamWet": d["mix"] * 100})
    elif kind == "delay":
        pp.update({"kParamBeatSync": 0.0, "kParamTimeL": d["time"], "kParamTimeR": d["time2"],
                   "kParamFeedback": d["feedback"] * 100, "kParamFreq": d["freq"], "kParamBW": d["bw"],
                   "kParamWet": d["mix"] * 100})
    elif kind == "comp":
        pp.update({"kParamMultiband": 1.0 if d["multiband"] else 0.0, "kParamWet": d["depth"] * 100})
        if d["multiband"]:
            pp["kParamThresh"] = float(np.clip(1.0 - d["amount"] * 0.7, 0.05, 1.0))
        else:
            pp.update({"kParamThresh": float(np.clip(10 ** (d["threshold_db"] / 20), 0, 1)),
                       "kParamRatio": d["ratio"], "kParamAttack": d["attack_ms"], "kParamRelease": d["release_ms"],
                       "kParamMakeup": 10 ** (d["makeup_db"] / 20)})
    elif kind == "reverb":
        pp.update({"kParamType": _k(str(d["kind"])), "kParamSize": d["size"] * 100,
                   "kParamFeedback": float(np.clip(100 * (1 - math.exp(-d["decay"] / 3)), 0, 100)),
                   "kParamFreq": (1 - d["damping"]) * 100, "kParamPreDelay": d["predelay"],
                   "kParamWet": d["mix"] * 100})
    elif kind == "eq":
        types = {"lowshelf": 1.0, "peak": 0.0, "highshelf": 1.0, "lp": 2.0, "hp": 2.0}
        pp.update({"kParamFreq1": d["freq1"], "kParamGain1": d["gain1"], "kParamReso1": d["q1"] * 20,
                   "kParamType1": types.get(d["type1"], 0.0),
                   "kParamFreq2": d["freq2"], "kParamGain2": d["gain2"], "kParamReso2": d["q2"] * 20,
                   "kParamType2": types.get(d["type2"], 0.0)})
    elif kind == "filter":
        pp.update({"kParamType": LEGACY_FILTERS.get(d["ftype"], d["ftype"]),
                   "kParamFreq": cutoff_to_norm(d["cutoff"], sc), "kParamReso": d["reso"] * 100,
                   "kParamDrive": d["drive"] * 100, "kParamVar": d["var"] * 100, "kParamWet": d["mix"] * 100})
    elif kind == "hyperdim":
        pp.update({"kParamRate": d["rate"] * 100, "kParamDetune": d["detune"] * 100,
                   "kParamUnison": float(d["voices"]), "kParamDimESize": d["dim_size"] * 100,
                   "kParamWet": d["mix"] * 100})
    elif kind == "bode":
        rng = max(abs(d["shift_hz"]), 1.0)
        pp.update({"kParamRange": rng, "kParamShift": 100.0 * d["shift_hz"] / rng,
                   "kParamFeedback": d["feedback"] * 100, "kParamWet": d["mix"] * 100})
    elif kind == "convolve":
        pp.update({"kParamSize": 10 + 990 * d["size"], "kParamDecay": d["decay"],
                   "kParamTone": d["tone"] * 100, "kParamPredelay": d["predelay"], "kParamWet": d["mix"] * 100})
    elif kind == "utility":
        pp.update({"kParamLevelOut": float(np.clip(10 ** (d["gain_db"] / 20) / 2, 0, 1))})
        if d["hpf"]:
            pp["kParamHPF"] = d["hpf"]
        if d["lpf"]:
            pp["kParamLPF"] = d["lpf"]
        if d["invert"]:
            pp["kParamPolarityL"] = pp["kParamPolarityR"] = 1.0
    elif kind == "split":
        pp["kParamFreq"] = d["freq"]
    elif kind == "split3":
        pp.update({"kParamFreq": d["freq"], "kParamFreq2": d["freq2"]})
    return {k: (float(v) if isinstance(v, (int, float, np.floating)) and not isinstance(v, bool) else v)
            for k, v in pp.items()}


def _fx_entry(m: dict, sc: dict, bpm: float) -> list[dict]:
    """One module (plus, for splitters, its band modules) as rack entries."""
    kind = m["type"]
    pp = _fx_to_serum(m, sc, bpm)
    entries = []
    head = {FX_KEYS[kind]: {"plainParams": pp}, "type": FX_TYPE_IDS[kind]}
    if kind in ("split", "split3", "splitms"):
        bands = m.get("bands") or []
        nb = 3 if kind == "split3" else 2
        sub = []
        for b in range(nb):
            chain = bands[b] if b < len(bands) else []
            flat = []
            for e in chain:
                flat += _fx_entry(e, sc, bpm)
            pp[f"kParamModuleCount{b + 1}"] = float(len(flat))
            sub += flat
        entries.append(head)
        entries += sub
        return entries
    if kind == "distortion":
        head["flex"] = [{}, {}]
    if kind in ("chorus", "flanger", "phaser"):
        head[FX_KEYS[kind]]["lfophasor"] = 0.0
    if kind not in ("chorus", "eq", "split", "split3", "splitms"):
        head["kUIParamMixOrGain"] = 0.0
    return [head]


def _serum_to_fx(entry: dict, sc: dict, bpm: float) -> dict | None:
    kind = FX_BY_ID.get(entry.get("type"))
    if kind is None:
        return None
    pp = (entry.get(FX_KEYS[kind]) or {}).get("plainParams")
    pp = pp if isinstance(pp, dict) else {}
    g = pp.get
    m = fx(kind)
    if g("kParamEnable", 1.0) == 0.0:
        m["enabled"] = False
    if kind == "distortion":
        m.update(mode=_unk(g("kParamMode")) if g("kParamMode") else "Tube", drive=g("kParamDrive", 30) / 100,
                 mix=g("kParamWet", 100) / 100)
        if g("kParamPrePost"):
            m.update(filter="pre" if g("kParamPrePost") == 1.0 else "post", filter_freq=g("kParamFreq", 0.5),
                     filter_bw=(g("kParamBW", 3.8) - 0.075) / 7.5)
    elif kind == "flanger":
        m.update(rate_hz=g("kParamRate", 0.3), depth=g("kParamDepth", 50) / 100,
                 feedback=g("kParamFeedback", 50) / 100, mix=g("kParamWet", 50) / 100)
    elif kind == "phaser":
        m.update(rate_hz=g("kParamRate", 0.3), depth=g("kParamDepth", 50) / 100, freq=g("kParamFreq", 800.0),
                 feedback=g("kParamFeedback", 40) / 100, stages=int(g("kParamNumPoles", 4)),
                 mix=g("kParamWet", 50) / 100)
    elif kind == "chorus":
        m.update(rate_hz=g("kParamRate", 0.8), depth=g("kParamDepth", 7.5) / 25, delay_ms=g("kParamDelay", 7.0),
                 feedback=g("kParamFeedback", 0) / 100, mix=g("kParamWet", 30) / 100)
    elif kind == "delay":
        synced = g("kParamBeatSync", 1.0) != 0.0
        beat = 60.0 / max(bpm, 1)
        t1, t2 = g("kParamTimeL", 0.25), g("kParamTimeR", g("kParamTimeL", 0.25))
        if synced:            # synced times are in whole notes
            t1, t2 = t1 * 4 * beat, t2 * 4 * beat
        m.update(time=t1, time2=t2, feedback=g("kParamFeedback", 35) / 100, freq=g("kParamFreq", 2000.0),
                 bw=g("kParamBW", 3.0), mix=g("kParamWet", 25) / 100)
    elif kind == "comp":
        mb = g("kParamMultiband", 0.0) == 1.0
        m.update(multiband=mb, depth=g("kParamWet", 100) / 100)
        if mb:
            m["amount"] = float(np.clip((1 - g("kParamThresh", 0.6)) / 0.7, 0, 1))
        else:
            m.update(threshold_db=20 * math.log10(max(g("kParamThresh", 0.3), 1e-4)),
                     ratio=min(g("kParamRatio", 4.0), 100.0), attack_ms=g("kParamAttack", 5.0),
                     release_ms=g("kParamRelease", 80.0), makeup_db=20 * math.log10(max(g("kParamMakeup", 1.0), 1e-4)))
    elif kind == "reverb":
        fb = g("kParamFeedback", 60) / 100
        m.update(kind=_unk(g("kParamType")) if g("kParamType") else "Hall", size=g("kParamSize", 50) / 100,
                 decay=float(-3 * math.log(max(1 - fb, 0.02))), damping=1 - g("kParamFreq", 50) / 100,
                 predelay=g("kParamPreDelay", 0.0) / (1000.0 if g("kParamPreDelay", 0.0) > 5 else 1.0),
                 mix=g("kParamWet", 20) / 100)
    elif kind == "eq":
        tmap = {1.0: "lowshelf", 0.0: "peak", 2.0: "lp"}
        t1 = tmap.get(g("kParamType1", 0.0), "peak")
        t2 = tmap.get(g("kParamType2", 0.0), "peak")
        if t2 == "lowshelf":
            t2 = "highshelf"
        if t2 == "lp":
            t2 = "lp"
        if t1 == "lp":
            t1 = "hp"
        m.update(freq1=g("kParamFreq1", 120.0), gain1=g("kParamGain1", 0.0), q1=max(g("kParamReso1", 14) / 20, 0.3),
                 type1=t1, freq2=g("kParamFreq2", 6000.0), gain2=g("kParamGain2", 0.0),
                 q2=max(g("kParamReso2", 14) / 20, 0.3), type2=t2)
    elif kind == "filter":
        m.update(ftype=g("kParamType", "L24"), cutoff=norm_to_cutoff(g("kParamFreq", 0.5), sc),
                 reso=g("kParamReso", 20) / 100, drive=g("kParamDrive", 0) / 100, var=g("kParamVar", 50) / 100,
                 mix=g("kParamWet", 100) / 100)
    elif kind == "hyperdim":
        m.update(rate=g("kParamRate", 40) / 100, detune=g("kParamDetune", 30) / 100,
                 voices=int(g("kParamUnison", 4)), dim_size=g("kParamDimESize", 50) / 100,
                 mix=g("kParamWet", 40) / 100)
    elif kind == "bode":
        m.update(shift_hz=g("kParamShift", 0.0) / 100 * g("kParamRange", 100.0),
                 feedback=g("kParamFeedback", 0) / 100, mix=g("kParamWet", 50) / 100)
    elif kind == "convolve":
        m.update(size=(g("kParamSize", 500) - 10) / 990, decay=g("kParamDecay", 1.0), tone=g("kParamTone", 0) / 100,
                 predelay=g("kParamPredelay", 0.0), mix=g("kParamWet", 20) / 100)
    elif kind == "utility":
        m.update(gain_db=20 * math.log10(max(2 * g("kParamLevelOut", 0.5), 1e-4)), hpf=g("kParamHPF", 0.0),
                 lpf=g("kParamLPF", 0.0), invert=bool(g("kParamPolarityL", 0.0)))
    elif kind == "split":
        m.update(freq=g("kParamFreq", 300.0))
    elif kind == "split3":
        m.update(freq=g("kParamFreq", 200.0), freq2=g("kParamFreq2", 2000.0))
    if kind in ("split", "split3", "splitms"):
        nb = 3 if kind == "split3" else 2
        m["_counts"] = [int(g(f"kParamModuleCount{b + 1}", 1)) for b in range(nb)]
    return m


def rack_to_chain(entries: list, sc: dict, bpm: float) -> list:
    """Serum's flat rack list (splitters followed by their band modules)."""
    out = []
    i = 0
    entries = entries or []

    def take(n):
        nonlocal i
        chain = []
        while n > 0 and i < len(entries):
            m = _serum_to_fx(entries[i], sc, bpm)
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


# ---------------------------------------------------------------------------
# Patch -> Serum


def patch_to_state(patch: Patch, bank: TableBank | None = None) -> dict:
    sc = scales()
    bank = bank or TableBank()
    st = load_skeleton()

    g = _pp(st["Global0"])
    g["kParamMasterVolume"] = float(np.clip(patch.master * 0.5, 0.05, 0.9))
    if patch.mono:
        g["kParamMonoToggle"] = 1.0
        g["kParamLegato"] = 1.0
    g["kParamBendRangeUp"] = 12.0     # matches export.BEND_RANGE
    g["kParamBendRangeDn"] = -12.0
    if patch.glide > 0:
        g["kParamPortamentoTime"] = float(patch.glide)
        g["kParamPortaAlways"] = 1.0
    for b in range(2):
        if b < len(patch.fx_buses) and patch.fx_buses[b]:
            g[f"kParamFXBus{b + 1}Vol"] = float(patch.bus_levels[b] if b < len(patch.bus_levels) else 1.0)

    # --- oscillators A/B/C ---------------------------------------------------
    for i, o in enumerate(patch.oscs[:3]):
        mod = st[f"Oscillator{i}"]
        pp = _pp(mod)
        pp["kParamEnable"] = 1.0 if o.enabled else 0.0
        if not o.enabled:
            continue
        pp["kParamVolume"] = float(np.clip(o.level, 0, 1))
        if o.octave:
            pp["kParamOctave"] = float(o.octave)
        if o.semi:
            pp["kParamCoarsePit"] = float(o.semi)
        if o.fine:
            pp["kParamFine"] = float(o.fine)
        if o.pan:
            pp["kParamPan"] = float(o.pan * 50)
        if o.unison > 1:
            pp["kParamUnison"] = float(o.unison)
            pp["kParamDetune"] = float(o.detune)
            pp["kParamDetuneWid"] = float(o.blend * 100)
            if o.unison_range != 2.0:
                pp["kParamUnisonRange"] = float(o.unison_range)
            if o.unison_stack and o.unison_stack != "none":
                pp["kParamUnisonStack"] = _k(o.unison_stack)
        wt = mod[f"WTOsc{i}"]
        wpp = _pp(wt)
        wpp["kParamTablePos"] = float(sc["tablepos_lo"] + o.position * (sc["tablepos_hi"] - sc["tablepos_lo"]))
        wpp["kParamRandomPhase"] = float(o.random_phase * 100)
        if o.phase:
            wpp["kParamInitialPhase"] = float((o.phase % 1.0) * 360)
        for slot, (w, amt) in enumerate(((o.warp, o.warp_amount), (o.warp2, o.warp2_amount))):
            name = _k(LEGACY_WARPS.get(w, w))
            if name and w != "none":
                wpp["kParamWarpMenu" if slot == 0 else "kParamWarpMenu2"] = name
                wpp["kParamWarp" if slot == 0 else "kParamWarp2"] = float(np.clip(amt, 0, 1))
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

    # --- noise (Oscillator3) and sub (Oscillator4) ----------------------------
    npp = _pp(st["Oscillator3"])
    npp["kParamEnable"] = 1.0 if patch.noise.enabled else 0.0
    if patch.noise.enabled:
        npp["kParamVolume"] = float(np.clip(patch.noise.level, 0, 1))
        nz = _pp(st["Oscillator3"]["NoiseOsc3"])
        nz["kParamNoiseType"] = "Pink" if patch.noise.color < 0.35 else "White"
        nz["kParamColor"] = float(np.clip(patch.noise.color, 0, 1))
    spp = _pp(st["Oscillator4"])
    spp["kParamEnable"] = 1.0 if patch.sub.enabled else 0.0
    if patch.sub.enabled:
        spp["kParamVolume"] = float(np.clip(patch.sub.level, 0, 1))
        spp["kParamOctave"] = float(patch.sub.octave)
        shape = SUB_NAMES.get(patch.sub.shape)
        if shape:
            _pp(st["Oscillator4"]["SubOsc4"])["kParamShape"] = shape

    # --- filters + routing ------------------------------------------------------
    for idx, f in enumerate((patch.filter, patch.filter2)):
        fpp = _pp(st[f"VoiceFilter{idx}"])
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
        else:
            fpp["kParamEnable"] = 0.0
    any_filter = patch.filter.enabled or patch.filter2.enabled
    sources = [f"osc{i + 1}" for i, o in enumerate(patch.oscs[:3]) if o.enabled]
    if patch.noise.enabled:
        sources.append("noise")
    if patch.sub.enabled:
        sources.append("sub")
    for s in sources:
        r = patch.route(s)
        rp = _pp(st[f"RoutingSlot{ROUTE_SLOT[s]}"])
        if r["dest"] == "none":
            rp["kParamRoutingDest"] = "kRoutingDestNone"
        elif r["dest"] == "direct" or not any_filter:
            rp["kParamRoutingDest"] = "kRoutingDestDirect"
        else:
            rp["kParamRoutingDest"] = "kRoutingDestFilter"
            bal = r["balance"] if (patch.filter.enabled and patch.filter2.enabled) else \
                (0.0 if patch.filter.enabled else 1.0)
            if bal:
                rp["kParamFilterBalance"] = float(-100 + 200 * bal)
        if r.get("send1"):
            rp["kParamFXBus1Level"] = float(r["send1"] * 100)
        if r.get("send2"):
            rp["kParamFXBus2Level"] = float(r["send2"] * 100)

    # --- envelopes ------------------------------------------------------------
    for i, e in enumerate(patch.envs[:N_ENVS]):
        epp = _pp(st[f"Env{i}"])
        epp.update({"kParamAttack": float(e.attack), "kParamHold": float(e.hold),
                    "kParamDecay": float(e.decay), "kParamSustain": float(e.sustain),
                    "kParamRelease": float(e.release),
                    "kParamCurve1": float(e.attack_curve * 100), "kParamCurve2": float(e.decay_curve * 100),
                    "kParamCurve3": float(e.release_curve * 100)})

    # --- LFOs -----------------------------------------------------------------
    used = {m.source for m in patch.mods if m.amount} | {m.aux for m in patch.mods if m.amount and m.aux}
    for k, l in enumerate(patch.lfos[:N_LFOS]):
        if f"lfo{k+1}" not in used or f"LFO{k}" not in st:
            continue
        mod = st[f"LFO{k}"]
        lpp = _pp(mod)
        rate = sync_to_hz(l.sync, patch.bpm) if l.sync else l.rate_hz
        lpp["kParamBeatSync"] = 0.0      # write exact Hz derived from the tempo
        lpp["kParamRate"] = float(rate)
        if l.mode == "free":
            lpp["kParamMode"] = "Free"
        elif l.mode == "env":
            lpp["kParamMode"] = "Envelope"
        if l.phase:
            lpp["kParamPhase"] = float(l.phase * 360)
        if l.type in LFO_TYPE_NAMES and l.type != "path":
            lpp["kParamType"] = LFO_TYPE_NAMES[l.type]
        if l.rise:
            lpp["kParamRise"] = float(l.rise)
        if l.delay:
            lpp["kParamDelay"] = float(l.delay)
        if l.smooth:
            lpp["kParamSmooth"] = float(l.smooth * 100)
        pts = _lfo_points(l.shape, l.points)
        ys = [1 - p[1] if sc["lfo_y_inverted"] else p[1] for p in pts]
        mod["curveData"] = {"curveVals": [float(p[2]) if len(p) > 2 else 0.5 for p in pts],
                            "numPoints": len(pts) - 1, "xVals": [p[0] for p in pts], "yVals": ys}
        mod["curveDisplayName"] = "Custom"

    # --- macros -----------------------------------------------------------------
    for i, v in enumerate(patch.macros[:N_MACROS]):
        if v:
            _pp(st[f"Macro{i}"])["kParamValue"] = float(v * 100)

    # --- mod matrix -----------------------------------------------------------
    slot = 0

    def add(src, aux, mtype, mid, pname, amount, bipolar=False):
        nonlocal slot
        key = f"{mtype}.{pname}"
        if slot >= 64 or src not in SOURCE_IDS or key not in PARAM_IDS:
            return
        pp = {"kParamAmount": float(np.clip(amount, -100, 100))}
        if bipolar:
            pp["kParamBipolar"] = 1.0
        st[f"ModSlot{slot}"] = {"destModuleID": mid, "destModuleParamID": PARAM_IDS[key],
                                "destModuleParamName": pname, "destModuleTypeString": mtype,
                                "plainParams": pp, "source": [SOURCE_IDS[src], SOURCE_IDS.get(aux, 0) if aux else 0]}
        slot += 1

    span = cutoff_octaves_span(sc)
    active = [i for i, o in enumerate(patch.oscs[:3]) if o.enabled]
    n_legacy = sum(_count_modules(m) for m in _legacy_chain(patch))
    top_index = [n_legacy + k for k in _top_level_rack_index(patch.fx_chain)]
    for m in patch.mods:
        if not m.amount:
            continue
        a, d, bp, aux = m.amount, m.dest, m.bipolar, m.aux
        if d in ("filter.cutoff", "filter1.cutoff", "filter2.cutoff"):
            add(m.source, aux, "VoiceFilter", 1 if d.startswith("filter2") else 0, "kParamFreq",
                a * 10.0 / span * 100, bp)
        elif d.startswith(("filter.", "filter1.", "filter2.")):
            what = d.split(".")[1]
            pname = {"reso": "kParamReso", "drive": "kParamDrive", "var": "kParamVar"}.get(what)
            if pname:
                add(m.source, aux, "VoiceFilter", 1 if d.startswith("filter2") else 0, pname, a * 100, bp)
        elif d.startswith("osc") and "." in d:
            i = int(d[3]) - 1
            what = d.split(".")[1]
            if what == "pos":
                add(m.source, aux, "WTOsc", i, "kParamTablePos", a * 100, bp)
            elif what == "level":
                add(m.source, aux, "Oscillator", i, "kParamVolume", a * 100, bp)
            elif what == "pan":
                add(m.source, aux, "Oscillator", i, "kParamPan", a * 100, bp)
            elif what == "fine":
                add(m.source, aux, "Oscillator", i, "kParamFine", a * 100, bp)
            elif what == "warp":
                add(m.source, aux, "WTOsc", i, "kParamWarp", a * 100, bp)
            elif what == "warp2":
                add(m.source, aux, "WTOsc", i, "kParamWarp2", a * 100, bp)
        elif d == "amp":
            for i in active + ([3] if patch.noise.enabled else []) + ([4] if patch.sub.enabled else []):
                add(m.source, aux, "Oscillator", i, "kParamVolume", a * 100, bp)
        elif d == "pitch":
            semis = a * 48.0
            for i in active + ([4] if patch.sub.enabled else []):
                add(m.source, aux, "Oscillator", i, "kParamCoarsePit", semis / sc["pitch_semis_per_100"] * 100, bp)
        elif d == "noise.level":
            add(m.source, aux, "Oscillator", 3, "kParamVolume", a * 100, bp)
        elif d == "sub.level":
            add(m.source, aux, "Oscillator", 4, "kParamVolume", a * 100, bp)
        elif d.startswith("fx."):
            parts = d.split(".")
            if len(parts) == 3 and parts[1].isdigit() and int(parts[1]) < len(patch.fx_chain):
                j = int(parts[1])
                mdl = patch.fx_chain[j]
                mtype = FX_KEYS.get(mdl["type"])
                pname = {"mix": "kParamWet", "drive": "kParamDrive"}.get(parts[2])
                if mtype and pname:
                    add(m.source, aux, mtype, top_index[j], pname, a * 100, bp)

    # --- FX racks -------------------------------------------------------------
    legacy = _legacy_chain(patch)
    st["FXRack0"]["FX"] = chain_to_rack(legacy + list(patch.fx_chain), sc, patch.bpm)
    for b in range(2):
        chain = patch.fx_buses[b] if b < len(patch.fx_buses) else []
        st[f"FXRack{b + 1}"]["FX"] = chain_to_rack(chain, sc, patch.bpm)

    # macro labels make the important knobs easy to find
    labels = ["CUTOFF", "RESO", "DRIVE", "WT POS"]
    for i, lab in enumerate(labels):
        st[f"Macro{i}"]["name"] = lab
    return st


def _top_level_rack_index(chain: list) -> list[int]:
    """Position of each top-level module in Serum's flattened rack, after the
    legacy modules (which are written first)."""
    out, pos = [], 0
    for m in chain or []:
        out.append(pos)
        pos += _count_modules(m)
    return out


def _count_modules(m: dict) -> int:
    return 1 + sum(_count_modules(e) for band in (m.get("bands") or []) for e in band)


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


def _display_name(tid: str) -> str:
    kind, _, name = tid.partition(":")
    if kind == "basic":
        return f"Basic {name.title()}"
    return f"FA Custom {name}"


def build_preset_bytes(patch: Patch, bank: TableBank | None = None, name: str | None = None,
                       description: str = "") -> bytes:
    from .preset import write_preset_bytes
    st = patch_to_state(patch, bank)
    tags = ["Wavetable", "Bass", "Mono" if patch.mono else "Poly"]
    if any(o.enabled and o.table.startswith(("custom:", "basic:")) for o in patch.oscs):
        tags.append("Embedded-Data")
    return write_preset_bytes(st, name or patch.name, author="Fourier Analyzer",
                              description=description, tags=tags)


# ---------------------------------------------------------------------------
# Serum -> Patch


def state_to_patch(st: dict, bank: TableBank | None = None, bpm: float = 140.0) -> Patch:
    """Import a Serum 2 preset state. Embedded wavetables are added to ``bank``
    (when given); referenced stock tables become serum:<path> table ids.
    Sample, granular, spectral and multisample oscillators are reported in
    ``patch.name`` notes (unsupported_*), not rendered."""
    sc = scales()
    p = Patch(bpm=bpm)
    unsupported = []
    g = st.get("Global0", {})
    p.mono = _pget(g, "kParamMonoToggle", 0.0) == 1.0
    p.glide = float(_pget(g, "kParamPortamentoTime", 0.0))
    p.master = float(_pget(g, "kParamMasterVolume", 0.35)) * 2
    for i in range(3):
        mod = st.get(f"Oscillator{i}", {})
        en = _pget(mod, "kParamEnable", 1.0 if i == 0 else 0.0)
        kind = _pget(mod, "kParamType", "kOsc_WT")
        if kind != "kOsc_WT" and en:
            unsupported.append(f"osc{i + 1}:{kind}")
        wt = mod.get(f"WTOsc{i}", {})
        wpp = wt.get("plainParams") if isinstance(wt.get("plainParams"), dict) else {}
        pos = (wpp.get("kParamTablePos", sc["tablepos_lo"]) - sc["tablepos_lo"]) / (sc["tablepos_hi"] - sc["tablepos_lo"])
        if "embeddedWTData" in wt and bank is not None:
            data = np.asarray(wt["embeddedWTData"], dtype=np.float32)
            nfr = max(1, len(data) // FRAME)
            table = bank.add_custom(data[: nfr * FRAME].reshape(nfr, FRAME))
        elif "relativePathToWT" in wt:
            table = "serum:" + wt["relativePathToWT"].lstrip("/")
        else:
            table = "basic:saw"
        p.oscs[i] = Osc(enabled=bool(en), table=table, position=float(np.clip(pos, 0, 1)),
                        octave=int(_pget(mod, "kParamOctave", 0)), semi=int(_pget(mod, "kParamCoarsePit", 0)),
                        fine=float(_pget(mod, "kParamFine", 0)), level=float(_pget(mod, "kParamVolume", 0.75)),
                        pan=float(_pget(mod, "kParamPan", 0.0)) / 50,
                        unison=int(_pget(mod, "kParamUnison", 1)), detune=float(_pget(mod, "kParamDetune", 0.25)),
                        blend=float(_pget(mod, "kParamDetuneWid", 75)) / 100,
                        unison_stack=_unk(_pget(mod, "kParamUnisonStack")) if _pget(mod, "kParamUnisonStack") else "none",
                        unison_range=float(_pget(mod, "kParamUnisonRange", 2.0)),
                        random_phase=float(wpp.get("kParamRandomPhase", 100.0)) / 100,
                        phase=float(wpp.get("kParamInitialPhase", 0.0)) / 360,
                        warp=_unk(wpp.get("kParamWarpMenu")) if wpp.get("kParamWarpMenu") else "none",
                        warp_amount=float(wpp.get("kParamWarp", 0.0)) if wpp.get("kParamWarpMenu") else 0.0,
                        warp2=_unk(wpp.get("kParamWarpMenu2")) if wpp.get("kParamWarpMenu2") else "none",
                        warp2_amount=float(wpp.get("kParamWarp2", 0.0)) if wpp.get("kParamWarpMenu2") else 0.0)
    nmod = st.get("Oscillator3", {})
    if _pget(nmod, "kParamEnable", 0.0):
        p.noise = NoiseOsc(enabled=True, color=float(_pget(nmod.get("NoiseOsc3", {}), "kParamColor", 0.5)),
                           level=float(_pget(nmod, "kParamVolume", 0.75)))
    smod = st.get("Oscillator4", {})
    if _pget(smod, "kParamEnable", 0.0):
        shape = _pget(smod.get("SubOsc4", {}), "kParamShape")
        inv = {v: k for k, v in SUB_NAMES.items() if v}
        p.sub = SubOsc(enabled=True, shape=inv.get(shape, "sine"), octave=int(_pget(smod, "kParamOctave", -1)),
                       level=float(_pget(smod, "kParamVolume", 0.75)))
    for idx, attr in enumerate(("filter", "filter2")):
        f = st.get(f"VoiceFilter{idx}", {})
        if _pget(f, "kParamEnable", 1.0 if idx == 0 else 0.0) == 1.0 and isinstance(f.get("plainParams"), dict):
            setattr(p, attr, Filter(enabled=True, type=_pget(f, "kParamType", "MgL24"),
                                    cutoff=norm_to_cutoff(_pget(f, "kParamFreq", 1.0), sc),
                                    reso=float(_pget(f, "kParamReso", 0)) / 100,
                                    drive=float(_pget(f, "kParamDrive", 0)) / 100,
                                    keytrack=1.0 if _pget(f, "kParamKeyTrack") else 0.0,
                                    var=float(_pget(f, "kParamVar", 50)) / 100,
                                    mix=float(_pget(f, "kParamWet", 100)) / 100))
    routing = {}
    for name, slot in ROUTE_SLOT.items():
        r = st.get(f"RoutingSlot{slot}", {})
        dest = _pget(r, "kParamRoutingDest")
        entry = {}
        if dest in ("kRoutingDestDirect", "kRoutingDestMaster"):
            entry["dest"] = "direct"
        elif dest == "kRoutingDestNone":
            entry["dest"] = "none"
        bal = _pget(r, "kParamFilterBalance")
        if bal is not None:
            entry["balance"] = (float(bal) + 100) / 200
        for b in (1, 2):
            v = _pget(r, f"kParamFXBus{b}Level")
            if v:
                entry[f"send{b}"] = float(v) / 100
        if entry:
            routing[name] = entry
    p.routing = routing
    for i in range(N_ENVS):
        e = st.get(f"Env{i}", {})
        p.envs[i] = Env(attack=_pget(e, "kParamAttack", 0.001), hold=_pget(e, "kParamHold", 0.0),
                        decay=_pget(e, "kParamDecay", 1.0), sustain=_pget(e, "kParamSustain", 1.0 if i == 0 else 0.0),
                        release=_pget(e, "kParamRelease", 0.01),
                        attack_curve=_pget(e, "kParamCurve1", 50.0) / 100,
                        decay_curve=_pget(e, "kParamCurve2", 66.6) / 100,
                        release_curve=_pget(e, "kParamCurve3", 66.6) / 100)
    for k in range(N_LFOS):
        l = st.get(f"LFO{k}", {})
        cd = l.get("curveData") or {}
        pts = None
        if cd.get("xVals"):
            ys = cd["yVals"]
            if sc["lfo_y_inverted"]:
                ys = [1 - y for y in ys]
            cv = cd.get("curveVals") or [0.5] * len(ys)
            pts = [[float(x), float(y), float(c)] for x, y, c in zip(cd["xVals"], ys, cv)]
        mode = {"Free": "free", "Envelope": "env"}.get(_pget(l, "kParamMode"), "trig")
        typ = {v: k2 for k2, v in LFO_TYPE_NAMES.items()}.get(_pget(l, "kParamType"), "shape")
        rate = float(_pget(l, "kParamRate", 2.0))
        p.lfos[k] = LFO(shape="custom" if pts else "triangle", rate_hz=max(rate, 0.01), mode=mode, points=pts,
                        phase=float(_pget(l, "kParamPhase", 0.0)) / 360, type=typ,
                        rise=float(_pget(l, "kParamRise", 0.0)), delay=float(_pget(l, "kParamDelay", 0.0)),
                        smooth=float(_pget(l, "kParamSmooth", 0.0)) / 100)
    p.macros = [float(_pget(st.get(f"Macro{i}", {}), "kParamValue", 0.0)) / 100 for i in range(N_MACROS)]
    span = cutoff_octaves_span(sc)
    mods = []
    for k in range(64):
        ms = st.get(f"ModSlot{k}", {})
        if "source" not in ms:
            continue
        src = SOURCE_NAMES.get(ms["source"][0])
        aux = SOURCE_NAMES.get(ms["source"][1]) if len(ms["source"]) > 1 and ms["source"][1] else None
        if src is None:
            continue
        amt = float(_pget(ms, "kParamAmount", 0.0)) / 100
        bp = bool(_pget(ms, "kParamBipolar", 0.0))
        t, pn, mid = ms.get("destModuleTypeString"), ms.get("destModuleParamName"), ms.get("destModuleID", 0)
        dest = None
        if t == "VoiceFilter":
            key = "filter2" if mid == 1 else "filter"
            if pn == "kParamFreq":
                dest, amt = f"{key}.cutoff", amt * span / 10
            else:
                dest = {"kParamReso": f"{key}.reso", "kParamDrive": f"{key}.drive", "kParamVar": f"{key}.var"}.get(pn)
        elif t == "WTOsc" and mid < 3:
            dest = {"kParamTablePos": f"osc{mid + 1}.pos", "kParamWarp": f"osc{mid + 1}.warp",
                    "kParamWarp2": f"osc{mid + 1}.warp2"}.get(pn)
        elif t == "Oscillator":
            if mid < 3:
                dest = {"kParamVolume": f"osc{mid + 1}.level", "kParamPan": f"osc{mid + 1}.pan",
                        "kParamFine": f"osc{mid + 1}.fine"}.get(pn)
                if pn == "kParamCoarsePit":
                    dest, amt = "pitch", amt * sc["pitch_semis_per_100"] / 48
            elif mid == 3 and pn == "kParamVolume":
                dest = "noise.level"
            elif mid == 4 and pn == "kParamVolume":
                dest = "sub.level"
        elif t and t.startswith("FX") and mid < 100:
            dest = None    # resolved after the rack is read (needs top-level index)
            mods.append(("fx", src, aux, t, pn, mid, amt, bp))
            continue
        if dest == "pitch" and any(isinstance(x, Mod) and x.dest == "pitch" and x.source == src
                                   and x.aux == aux for x in mods):
            continue        # one pitch mod is written per oscillator; keep a single entry
        if dest:
            mods.append(Mod(source=src, dest=dest, amount=amt, bipolar=bp, aux=aux))
    p.fx_chain = rack_to_chain(st.get("FXRack0", {}).get("FX", []), sc, bpm)
    p.fx_buses = [rack_to_chain(st.get(f"FXRack{b}", {}).get("FX", []), sc, bpm) for b in (1, 2)]
    p.bus_levels = [float(_pget(g, "kParamFXBus1Vol", 1.0)), float(_pget(g, "kParamFXBus2Vol", 1.0))]
    flat_pos = _top_level_rack_index(p.fx_chain)
    resolved = []
    for m in mods:
        if isinstance(m, tuple):
            _, src, aux, t, pn, mid, amt, bp = m
            if mid in flat_pos:
                j = flat_pos.index(mid)
                what = {"kParamWet": "mix", "kParamDrive": "drive"}.get(pn)
                if what:
                    resolved.append(Mod(source=src, dest=f"fx.{j}.{what}", amount=amt, bipolar=bp, aux=aux))
        else:
            resolved.append(m)
    p.mods = resolved
    for b in range(2):
        if p.fx_buses[b] and not any(r.get(f"send{b + 1}") for r in routing.values()):
            p.fx_buses[b] = []           # bus with nothing sent to it is silent
    if unsupported:
        p.name = "Imported (not rendered: " + ", ".join(unsupported) + ")"
    else:
        p.name = "Imported"
    return p
