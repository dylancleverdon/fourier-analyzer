"""Convert a :class:`Patch` into a Serum 2 preset state.

Parameter names, module layout and modulation IDs come from the structure of
Serum 2.0.x factory presets. A few *unit scales* (filter cutoff curve, LFO rate
units, pitch-mod depth) are not stored in presets and are assumptions; they
live in ``SCALES`` and can be overridden from settings.json (key
``serum_scales``) or measured with ``fourier_analyzer.serum.host.calibrate``
on a Mac that has Serum 2 installed.
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

import numpy as np

from .. import config
from ..synth.patch import Patch, sync_to_hz
from ..synth.wavetables import FRAME, TableBank, resolve_serum_ref

SKELETON_PATH = Path(__file__).with_name("init_skeleton.json")

SCALES = {
    # VoiceFilter kParamFreq is 0..1; assumed exponential from f_lo to f_hi
    "cutoff_lo_hz": 8.176,
    "cutoff_hi_hz": 22050.0,
    # semitones reached by a +100 % modulation of Oscillator kParamCoarsePit
    "pitch_semis_per_100": 64.0,
    # LFO kParamRate in free mode is Hz
    "lfo_rate_is_hz": True,
    # Serum stores LFO y with 1 = top of the graph
    "lfo_y_inverted": False,
    # kParamTablePos runs 1..256 regardless of frame count
    "tablepos_lo": 1.0,
    "tablepos_hi": 256.0,
}

FILTER_NAMES = {"lp12": "L12", "lp24": "L24", "hp12": "H12", "hp24": "H24", "bp12": "BP12"}
DIST_NAMES = {"soft": "kSoftClip", "hard": "kHardClip", "tube": None, "sinefold": "kSinFold",
              "linfold": "kLinFold", "diode": "kDiode1"}
# Serum 2 WT warp menu values (osc A: "_OSC" = from osc B)
WARP_NAMES = {"fm_b": "kFM_OSC", "fm_sub": "kFM_SUB", "am_b": "kAM_OSC", "rm_b": "kRM_OSC",
              "rm_sub": "kRM_SUB"}
SUB_NAMES = {"sine": None, "triangle": "kTriangle", "saw": "kSaw", "square": "kSquare"}

# modulation source ids (ModSlot "source": [id, 0])
SOURCE_IDS = {"env1": 2, "env2": 3, "env3": 4, "lfo1": 6, "lfo2": 7, "lfo3": 8, "lfo4": 9}
# (module type, param name) -> param id
PARAM_IDS = {
    ("Oscillator", "kParamVolume"): 1, ("Oscillator", "kParamPan"): 2,
    ("Oscillator", "kParamOctave"): 3, ("Oscillator", "kParamFine"): 5,
    ("Oscillator", "kParamCoarsePit"): 6,
    ("WTOsc", "kParamTablePos"): 6, ("WTOsc", "kParamWarp"): 0,
    ("VoiceFilter", "kParamFreq"): 3, ("VoiceFilter", "kParamReso"): 4,
    ("VoiceFilter", "kParamDrive"): 5, ("NoiseOsc", "kParamColor"): 0,
    ("FXDistortion", "kParamDrive"): 2, ("FXDistortion", "kParamWet"): 1,
}


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


def _lfo_points(shape: str, points):
    if shape == "custom" and points:
        pts = [(float(x), float(y)) for x, y in points]
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
        wt = mod[f"WTOsc{i}"]
        wpp = _pp(wt)
        wpp["kParamTablePos"] = float(sc["tablepos_lo"] + o.position * (sc["tablepos_hi"] - sc["tablepos_lo"]))
        wpp["kParamRandomPhase"] = float(o.random_phase * 100)
        if o.phase:
            wpp["kParamInitialPhase"] = float((o.phase % 1.0) * 360)
        modulator_on = (patch.sub.enabled if o.warp.endswith("_sub")
                        else patch.oscs[1 if i != 1 else 0].enabled)
        if o.warp != "none" and o.warp_amount > 0 and modulator_on:
            wpp["kParamWarpMenu"] = WARP_NAMES[o.warp]
            wpp["kParamWarp"] = float(np.clip(o.warp_amount, 0, 1))
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

    # --- filter + routing -----------------------------------------------------
    f = patch.filter
    fpp = _pp(st["VoiceFilter0"])
    if f.enabled:
        fpp["kParamEnable"] = 1.0
        fpp["kParamType"] = FILTER_NAMES.get(f.type, "L24")
        fpp["kParamFreq"] = cutoff_to_norm(f.cutoff, sc)
        fpp["kParamReso"] = float(f.reso * 100)
        if f.drive:
            fpp["kParamDrive"] = float(f.drive * 100)
        if f.keytrack:
            fpp["kParamKeyTrack"] = 1.0
        sources = [i for i, o in enumerate(patch.oscs[:3]) if o.enabled]
        if patch.noise.enabled:
            sources.append(3)
        if patch.sub.enabled:
            sources.append(4)
        for s in sources:
            _pp(st[f"RoutingSlot{s}"])["kParamRoutingDest"] = "kRoutingDestFilter"
    else:
        fpp["kParamEnable"] = 0.0

    # --- envelopes ------------------------------------------------------------
    for i, e in enumerate(patch.envs[:3]):
        epp = _pp(st[f"Env{i}"])
        epp.update({"kParamAttack": float(e.attack), "kParamHold": float(e.hold),
                    "kParamDecay": float(e.decay), "kParamSustain": float(e.sustain),
                    "kParamRelease": float(e.release)})

    # --- LFOs -----------------------------------------------------------------
    used = {m.source for m in patch.mods if m.amount}
    for k, l in enumerate(patch.lfos[:4]):
        if f"lfo{k+1}" not in used:
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
        pts = _lfo_points(l.shape, l.points)
        ys = [1 - y if sc["lfo_y_inverted"] else y for _, y in pts]
        mod["curveData"] = {"curveVals": [0.5] * len(pts), "numPoints": len(pts) - 1,
                            "xVals": [x for x, _ in pts], "yVals": ys}
        mod["curveDisplayName"] = "Custom"

    # --- mod matrix -----------------------------------------------------------
    slot = 0

    def add(src, mtype, mid, pname, amount, bipolar=False):
        nonlocal slot
        if slot >= 64 or src not in SOURCE_IDS:
            return
        pp = {"kParamAmount": float(np.clip(amount, -100, 100))}
        if bipolar:
            pp["kParamBipolar"] = 1.0
        st[f"ModSlot{slot}"] = {"destModuleID": mid, "destModuleParamID": PARAM_IDS[(mtype, pname)],
                                "destModuleParamName": pname, "destModuleTypeString": mtype,
                                "plainParams": pp, "source": [SOURCE_IDS[src], 0]}
        slot += 1

    span = cutoff_octaves_span(sc)
    active = [i for i, o in enumerate(patch.oscs[:3]) if o.enabled]
    for m in patch.mods:
        if not m.amount:
            continue
        a = m.amount
        if m.dest == "filter.cutoff":
            add(m.source, "VoiceFilter", 0, "kParamFreq", a * 10.0 / span * 100, m.bipolar)
        elif m.dest == "filter.reso":
            add(m.source, "VoiceFilter", 0, "kParamReso", a * 100, m.bipolar)
        elif m.dest.startswith("osc") and m.dest.endswith(".pos"):
            i = int(m.dest[3]) - 1
            add(m.source, "WTOsc", i, "kParamTablePos", a * 100, m.bipolar)
        elif m.dest.startswith("osc") and m.dest.endswith(".level"):
            i = int(m.dest[3]) - 1
            add(m.source, "Oscillator", i, "kParamVolume", a * 100, m.bipolar)
        elif m.dest == "amp":
            for i in active + ([3] if patch.noise.enabled else []) + ([4] if patch.sub.enabled else []):
                add(m.source, "Oscillator", i, "kParamVolume", a * 100, m.bipolar)
        elif m.dest == "pitch":
            semis = a * 48.0
            for i in active + ([4] if patch.sub.enabled else []):
                add(m.source, "Oscillator", i, "kParamCoarsePit",
                    semis / sc["pitch_semis_per_100"] * 100, m.bipolar)
        elif m.dest == "noise.level":
            add(m.source, "Oscillator", 3, "kParamVolume", a * 100, m.bipolar)

    # --- FX (main rack) -------------------------------------------------------
    fx = []
    d = patch.distortion
    if d.enabled:
        dp = {"kParamDrive": float(d.drive * 100), "kParamWet": float(d.mix * 100)}
        name = DIST_NAMES.get(d.mode)
        if name:
            dp["kParamMode"] = name
        fx.append({"FXDistortion": {"plainParams": dp}, "flex": [{}, {}], "kUIParamMixOrGain": 0.0, "type": 0})
    if patch.ott.enabled:
        cp = {"kParamMultiband": 1.0, "kParamWet": float(patch.ott.depth * 100),
              "kParamThresh": float(np.clip(1.0 - patch.ott.amount * 0.7, 0.05, 1.0))}
        fx.append({"FXComp": {"plainParams": cp}, "kUIParamMixOrGain": 0.0, "type": 5})
    if patch.eq.enabled:
        e = patch.eq
        ep = {"kParamFreq1": float(e.low_freq), "kParamGain1": float(e.low_gain), "kParamType1": 1.0,
              "kParamFreq2": float(e.high_freq), "kParamGain2": float(e.high_gain), "kParamType2": 1.0}
        fx.append({"FXEQ": {"plainParams": ep}, "type": 7})
    if patch.chorus.enabled:
        c = patch.chorus
        chp = {"kParamRate": float(c.rate_hz), "kParamDepth": float(c.depth * 25), "kParamWet": float(c.mix * 100)}
        fx.append({"FXChorus": {"lfophasor": 0.0, "plainParams": chp}, "type": 3})
    st["FXRack0"]["FX"] = fx

    # macro labels make the important knobs easy to find
    labels = ["CUTOFF", "RESO", "DRIVE", "WT POS"]
    for i, lab in enumerate(labels):
        st[f"Macro{i}"]["name"] = lab
    return st


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


def state_to_patch(st: dict) -> Patch:
    """Best-effort reverse mapping (used for tests and for importing presets as
    starting points)."""
    sc = scales()
    from ..synth.patch import Env, Filter, Osc
    p = Patch()
    for i in range(3):
        mod = st.get(f"Oscillator{i}", {})
        pp = mod.get("plainParams") if isinstance(mod.get("plainParams"), dict) else {}
        en = pp.get("kParamEnable", 1.0 if i == 0 else 0.0)
        wt = mod.get(f"WTOsc{i}", {})
        wpp = wt.get("plainParams") if isinstance(wt.get("plainParams"), dict) else {}
        pos = (wpp.get("kParamTablePos", sc["tablepos_lo"]) - sc["tablepos_lo"]) / (sc["tablepos_hi"] - sc["tablepos_lo"])
        table = "serum:" + wt["relativePathToWT"] if "relativePathToWT" in wt else "basic:saw"
        p.oscs[i] = Osc(enabled=bool(en), table=table, position=float(np.clip(pos, 0, 1)),
                        octave=int(pp.get("kParamOctave", 0)), semi=int(pp.get("kParamCoarsePit", 0)),
                        fine=float(pp.get("kParamFine", 0)), level=float(pp.get("kParamVolume", 0.75)),
                        unison=int(pp.get("kParamUnison", 1)), detune=float(pp.get("kParamDetune", 0.1)),
                        blend=float(pp.get("kParamDetuneWid", 75)) / 100)
    fpp = st.get("VoiceFilter0", {}).get("plainParams")
    if isinstance(fpp, dict) and fpp.get("kParamEnable"):
        inv = {v: k for k, v in FILTER_NAMES.items()}
        lo, hi = sc["cutoff_lo_hz"], sc["cutoff_hi_hz"]
        p.filter = Filter(enabled=True, type=inv.get(fpp.get("kParamType", "L24"), "lp24"),
                          cutoff=lo * (hi / lo) ** float(fpp.get("kParamFreq", 1.0)),
                          reso=float(fpp.get("kParamReso", 0)) / 100)
    for i in range(3):
        epp = st.get(f"Env{i}", {}).get("plainParams")
        if isinstance(epp, dict):
            p.envs[i] = Env(attack=epp.get("kParamAttack", 0.001), hold=epp.get("kParamHold", 0.0),
                            decay=epp.get("kParamDecay", 1.0), sustain=epp.get("kParamSustain", 1.0),
                            release=epp.get("kParamRelease", 0.01))
    return p
