"""Synth patch description shared by the renderer, the optimizer and the
Serum preset writer.

Units are chosen to be musically meaningful (Hz, seconds, semitones) so the
optimizer can reason about them; conversion to Serum's internal parameter
scales happens in ``fourier_analyzer.serum.mapping``.
"""
from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass, field
from typing import Optional

# Serum 2 has three primary oscillators (A, B, C) plus sub and noise.
MAX_OSCS = 3

N_ENVS = 4
N_LFOS = 10
N_MACROS = 8

LFO_SHAPES = ("sine", "triangle", "saw_down", "saw_up", "square", "custom")
LFO_TYPES = ("shape", "random", "lorenz", "rossler", "path")
# Filter types are Serum 2's own names (L24, MgL24, LadderAcid, Comb2, ...);
# the short legacy names lp12/lp24/hp12/hp24/bp12 are accepted too.
FILTER_TYPES = ("lp12", "lp24", "hp12", "hp24", "bp12")
# Distortion modes: Serum 2 names without the leading "k" (Overdrive,
# SoftSat, Diode1, ...); legacy soft/hard/tube/sinefold/linfold/diode accepted.
DIST_MODES = ("soft", "hard", "tube", "sinefold", "linfold", "diode")
# Warp modes: Serum 2 names without the leading "k" (Sync, PD_OSC, FM_SUB,
# BendPos, DistTube, FilterLPF, ...); legacy fm_b/fm_sub/am_b/rm_b/rm_sub ok.
WARP_MODES = ("none", "fm_b", "fm_sub", "am_b", "rm_b", "rm_sub")
MOD_SOURCES = (tuple(f"env{i}" for i in range(1, N_ENVS + 1)) + tuple(f"lfo{i}" for i in range(1, N_LFOS + 1))
               + ("velocity", "note", "modwheel") + tuple(f"macro{i}" for i in range(1, N_MACROS + 1)))
# amount 1.0 means: cutoff +10 octaves, pos +1 (whole table), level +1,
# pitch +48 semitones, drive +1, fine +100 cents, pan +1, warp +1, mix +1.
MOD_DESTS = ("osc1.pos", "osc2.pos", "osc3.pos", "osc1.level", "osc2.level",
             "osc3.level", "osc1.warp", "osc2.warp", "osc3.warp", "osc1.warp2", "osc2.warp2", "osc3.warp2",
             "osc1.fine", "osc2.fine", "osc3.fine", "osc1.pan", "osc2.pan", "osc3.pan",
             "filter.cutoff", "filter.reso", "filter.drive", "filter.var",
             "filter2.cutoff", "filter2.reso", "filter2.drive", "filter2.var",
             "pitch", "amp", "dist.drive", "noise.level", "sub.level",
             "fx.<index>.mix", "fx.<index>.drive")

# Effect modules (Patch.fx_chain / fx_buses entries are dicts with "type")
FX_TYPES = ("distortion", "flanger", "phaser", "chorus", "delay", "comp", "reverb", "eq", "filter",
            "hyperdim", "bode", "convolve", "utility", "split", "split3", "splitms")
FX_DEFAULTS = {
    "distortion": {"mode": "Overdrive", "drive": 0.3, "mix": 1.0, "filter": "off", "filter_freq": 0.5, "filter_bw": 0.5},
    "flanger": {"rate_hz": 0.3, "depth": 0.5, "feedback": 0.5, "mix": 0.5},
    "phaser": {"rate_hz": 0.3, "depth": 0.5, "freq": 800.0, "feedback": 0.4, "stages": 4, "mix": 0.5},
    "chorus": {"rate_hz": 0.8, "depth": 0.3, "delay_ms": 7.0, "feedback": 0.0, "mix": 0.3},
    "delay": {"time": 0.25, "time2": 0.25, "feedback": 0.35, "freq": 2000.0, "bw": 3.0, "mix": 0.25},
    "comp": {"multiband": True, "depth": 0.5, "amount": 0.6, "threshold_db": -18.0, "ratio": 4.0,
             "attack_ms": 5.0, "release_ms": 80.0, "makeup_db": 0.0},
    "reverb": {"kind": "Hall", "size": 0.5, "decay": 1.5, "damping": 0.5, "predelay": 0.0, "mix": 0.2},
    "eq": {"freq1": 120.0, "gain1": 0.0, "q1": 0.7, "type1": "lowshelf",
           "freq2": 6000.0, "gain2": 0.0, "q2": 0.7, "type2": "highshelf"},
    "filter": {"ftype": "L24", "cutoff": 2000.0, "reso": 0.2, "drive": 0.0, "var": 0.5, "mix": 1.0},
    "hyperdim": {"rate": 0.4, "detune": 0.3, "voices": 4, "dim_size": 0.5, "mix": 0.4},
    "bode": {"shift_hz": 50.0, "feedback": 0.0, "mix": 0.5},
    "convolve": {"size": 0.5, "decay": 1.0, "tone": 0.0, "predelay": 0.0, "mix": 0.2},
    "utility": {"gain_db": 0.0, "hpf": 0.0, "lpf": 0.0, "invert": False},
    "split": {"freq": 300.0, "bands": [[], []]},
    "split3": {"freq": 200.0, "freq2": 2000.0, "bands": [[], [], []]},
    "splitms": {"bands": [[], []]},
}


def fx(fx_type: str, /, **kw) -> dict:
    """New effect-module dict with defaults."""
    d = {"type": fx_type, "enabled": True}
    d.update(copy.deepcopy(FX_DEFAULTS.get(fx_type, {})))
    d.update(kw)
    return d


@dataclass
class Osc:
    enabled: bool = False
    table: str = "basic:saw"      # basic:<name> | serum:<relative path> | custom:<n>
    position: float = 0.0         # 0..1 across the table
    octave: int = 0
    semi: int = 0
    fine: float = 0.0             # cents
    level: float = 0.75           # 0..1 (linear amplitude)
    pan: float = 0.0
    unison: int = 1
    detune: float = 0.1           # 0..1, Serum-style detune amount
    blend: float = 0.75           # 0..1 level of the outer unison voices
    random_phase: float = 1.0     # 0..1 (0 = every note starts at `phase`)
    phase: float = 0.0            # 0..1 start phase of the cycle at note-on
    # Serum warp mode applied to this oscillator: none | fm_b | fm_sub | am_b |
    # rm_b | rm_sub (FM = phase modulation by osc B / sub, AM/RM = amplitude
    # / ring modulation). Osc B can sit at level 0 and act only as modulator.
    warp: str = "none"
    warp_amount: float = 0.0      # 0..1
    warp2: str = "none"           # second warp slot (applied after the first)
    warp2_amount: float = 0.0
    unison_stack: str = "none"    # none | Octave1..3 | OctaveFifth1..3 | Center12
    unison_range: float = 2.0     # semitones spread at detune 1.0 (Serum "range")


@dataclass
class SubOsc:
    enabled: bool = False
    shape: str = "sine"           # sine | triangle | saw | square
    octave: int = -1
    level: float = 0.5


@dataclass
class NoiseOsc:
    enabled: bool = False
    color: float = 0.5            # 0 = dark, 0.5 = white, 1 = bright
    level: float = 0.2


@dataclass
class Filter:
    enabled: bool = False
    type: str = "lp24"
    cutoff: float = 2000.0        # Hz
    reso: float = 0.1             # 0..1
    drive: float = 0.0            # 0..1
    keytrack: float = 0.0         # 0..1
    var: float = 0.5              # 0..1 type-specific (morph, comb damping, vowel ...)
    mix: float = 1.0              # 0..1 wet


@dataclass
class Env:
    attack: float = 0.001
    hold: float = 0.0
    decay: float = 0.3
    sustain: float = 1.0
    release: float = 0.05
    attack_curve: float = 0.5     # 0..1 (0.5 = linear); Serum curve 0..100 / 100
    decay_curve: float = 0.666
    release_curve: float = 0.666


@dataclass
class LFO:
    shape: str = "sine"
    rate_hz: float = 2.0
    sync: Optional[str] = None    # e.g. "1/8", "1/4T", "1/16D"; overrides rate_hz
    phase: float = 0.0            # 0..1
    mode: str = "trig"            # trig | free | env (one shot)
    points: Optional[list] = None  # custom shape: [[x, y], ...] or [[x, y, curve], ...]
    type: str = "shape"           # shape | random (S&H) | lorenz | rossler | path
    rise: float = 0.0             # seconds to fade in after note-on
    delay: float = 0.0            # seconds before the LFO starts
    smooth: float = 0.0           # 0..1 slew


@dataclass
class Mod:
    source: str = "lfo1"
    dest: str = "filter.cutoff"
    amount: float = 0.0           # -1..1, see MOD_DESTS comment
    bipolar: bool = False
    aux: Optional[str] = None     # optional second source that scales this one


@dataclass
class Distortion:
    enabled: bool = False
    mode: str = "soft"
    drive: float = 0.3            # 0..1 (0..36 dB of gain)
    mix: float = 1.0


@dataclass
class OTT:
    enabled: bool = False
    depth: float = 0.5            # 0..1 wet
    amount: float = 0.6           # 0..1 compression strength (up + down)


@dataclass
class EQ:
    enabled: bool = False
    low_freq: float = 120.0
    low_gain: float = 0.0         # dB (low shelf)
    high_freq: float = 6000.0
    high_gain: float = 0.0        # dB (high shelf)


@dataclass
class Chorus:
    enabled: bool = False
    rate_hz: float = 0.8
    depth: float = 0.3
    mix: float = 0.3


@dataclass
class Patch:
    name: str = "Reconstructed"
    oscs: list = field(default_factory=lambda: [Osc(enabled=True), Osc(), Osc()])
    sub: SubOsc = field(default_factory=SubOsc)
    noise: NoiseOsc = field(default_factory=NoiseOsc)
    filter: Filter = field(default_factory=Filter)
    filter2: Filter = field(default_factory=Filter)
    # per source (osc1, osc2, osc3, noise, sub): dest filter|direct|none,
    # balance 0 = filter 1 only .. 1 = filter 2 only, and FX bus sends 0..1
    routing: dict = field(default_factory=dict)
    envs: list = field(default_factory=lambda: [Env(), Env(sustain=0.0), Env(sustain=0.0), Env(sustain=0.0)])
    lfos: list = field(default_factory=lambda: [LFO() for _ in range(N_LFOS)])
    macros: list = field(default_factory=lambda: [0.0] * N_MACROS)
    mods: list = field(default_factory=list)
    fx_chain: list = field(default_factory=list)     # main FX rack, in order
    fx_buses: list = field(default_factory=lambda: [[], []])
    bus_levels: list = field(default_factory=lambda: [1.0, 1.0])
    distortion: Distortion = field(default_factory=Distortion)
    ott: OTT = field(default_factory=OTT)
    eq: EQ = field(default_factory=EQ)
    chorus: Chorus = field(default_factory=Chorus)
    mono: bool = True
    glide: float = 0.0            # seconds
    master: float = 0.7
    bpm: float = 140.0
    # Custom wavetables referenced as custom:<n>; each is a list of frames
    # (stored separately as numpy arrays, not serialised here).

    def copy(self) -> "Patch":
        return copy.deepcopy(self)

    def to_dict(self) -> dict:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=1)

    @classmethod
    def from_dict(cls, d: dict) -> "Patch":
        d = copy.deepcopy(d)
        p = cls()
        p.name = d.get("name", p.name)
        p.oscs = [Osc(**o) for o in d.get("oscs", [])] or p.oscs
        while len(p.oscs) < MAX_OSCS:
            p.oscs.append(Osc())
        p.sub = SubOsc(**d.get("sub", {}))
        p.noise = NoiseOsc(**d.get("noise", {}))
        p.filter = Filter(**d.get("filter", {}))
        p.filter2 = Filter(**d.get("filter2", {}))
        p.routing = d.get("routing", {}) or {}
        p.envs = [Env(**e) for e in d.get("envs", [])] or p.envs
        while len(p.envs) < N_ENVS:
            p.envs.append(Env(sustain=0.0))
        p.lfos = [LFO(**l) for l in d.get("lfos", [])] or p.lfos
        while len(p.lfos) < N_LFOS:
            p.lfos.append(LFO())
        p.macros = (list(d.get("macros", [])) + [0.0] * N_MACROS)[:N_MACROS]
        p.mods = [Mod(**m) for m in d.get("mods", [])]
        p.fx_chain = d.get("fx_chain", []) or []
        p.fx_buses = (d.get("fx_buses") or [[], []]) + [[]] * 0
        p.bus_levels = d.get("bus_levels", [1.0, 1.0])
        p.distortion = Distortion(**d.get("distortion", {}))
        p.ott = OTT(**d.get("ott", {}))
        p.eq = EQ(**d.get("eq", {}))
        p.chorus = Chorus(**d.get("chorus", {}))
        for k in ("mono", "glide", "master", "bpm"):
            if k in d:
                setattr(p, k, d[k])
        return p

    def active_oscs(self):
        return [(i, o) for i, o in enumerate(self.oscs) if o.enabled]

    def mods_for(self, dest: str):
        return [m for m in self.mods if m.dest == dest and m.amount != 0.0]

    def route(self, source: str) -> dict:
        """Routing of one source; default: through filter 1 if enabled."""
        r = {"dest": "filter", "balance": 0.0, "send1": 0.0, "send2": 0.0}
        r.update(self.routing.get(source, {}))
        return r

    def describe(self) -> list[str]:
        """Human-readable summary lines for the UI."""
        out = []
        for i, o in self.active_oscs():
            uni = f", unison {o.unison} x detune {o.detune:.2f}" if o.unison > 1 else ""
            warp = f", warp {o.warp} {o.warp_amount:.2f}" if o.warp != "none" and o.warp_amount else ""
            if o.warp2 != "none" and o.warp2_amount:
                warp += f" + {o.warp2} {o.warp2_amount:.2f}"
            if o.unison > 1 and o.unison_stack != "none":
                warp += f", stack {o.unison_stack}"
            out.append(f"Osc {'ABC'[i]}: {o.table} @ pos {o.position:.2f}, "
                       f"oct {o.octave:+d} semi {o.semi:+d} fine {o.fine:+.0f}c, "
                       f"level {o.level:.2f}{uni}{warp}")
        if self.sub.enabled:
            out.append(f"Sub: {self.sub.shape} oct {self.sub.octave:+d} level {self.sub.level:.2f}")
        if self.noise.enabled:
            out.append(f"Noise: color {self.noise.color:.2f} level {self.noise.level:.2f}")
        for label, f in (("Filter 1", self.filter), ("Filter 2", self.filter2)):
            if f.enabled:
                out.append(f"{label}: {f.type} cutoff {f.cutoff:.0f} Hz reso {f.reso:.2f} drive {f.drive:.2f}"
                           + (f" var {f.var:.2f}" if f.var != 0.5 else "") + (f" mix {f.mix:.2f}" if f.mix < 1 else ""))
        for src, r in sorted(self.routing.items()):
            out.append(f"Routing {src}: {r}")
        names = ["Env1 (amp)", "Env2", "Env3", "Env4"]
        for n, e in zip(names, self.envs):
            out.append(f"{n}: A {e.attack*1000:.1f}ms H {e.hold*1000:.0f}ms D {e.decay*1000:.0f}ms "
                       f"S {e.sustain:.2f} R {e.release*1000:.0f}ms")
        used = {m.source for m in self.mods}
        for i, l in enumerate(self.lfos):
            if f"lfo{i+1}" in used:
                rate = l.sync if l.sync else f"{l.rate_hz:.2f} Hz"
                kind = l.shape if l.type == "shape" else l.type
                extra = "".join(f", {k} {getattr(l, k):.2f}" for k in ("rise", "delay", "smooth") if getattr(l, k))
                out.append(f"LFO{i+1}: {kind} {rate} ({l.mode}){extra}")
        for m in self.mods:
            if m.amount:
                out.append(f"Mod: {m.source} -> {m.dest} amount {m.amount:+.3f}"
                           + (" (bipolar)" if m.bipolar else "") + (f" x {m.aux}" if m.aux else ""))
        for i, v in enumerate(self.macros):
            if v:
                out.append(f"Macro {i+1}: {v:.2f}")
        if self.distortion.enabled:
            d = self.distortion
            out.append(f"FX Distortion: {d.mode} drive {d.drive:.2f} mix {d.mix:.2f}")
        if self.ott.enabled:
            out.append(f"FX OTT/multiband: depth {self.ott.depth:.2f} amount {self.ott.amount:.2f}")
        if self.eq.enabled:
            out.append(f"FX EQ: low {self.eq.low_gain:+.1f} dB @ {self.eq.low_freq:.0f} Hz, "
                       f"high {self.eq.high_gain:+.1f} dB @ {self.eq.high_freq:.0f} Hz")
        if self.chorus.enabled:
            c = self.chorus
            out.append(f"FX Chorus: rate {c.rate_hz:.2f} Hz depth {c.depth:.2f} mix {c.mix:.2f}")
        def fx_lines(chain, prefix):
            for k, e in enumerate(chain):
                if not e.get("enabled", True):
                    continue
                params = ", ".join(f"{a} {b:.2f}" if isinstance(b, float) else f"{a} {b}"
                                   for a, b in e.items() if a not in ("type", "enabled", "bands"))
                out.append(f"{prefix}{k + 1} {e['type']}: {params}")
                for bi, band in enumerate(e.get("bands", []) or []):
                    fx_lines(band, f"{prefix}{k + 1}.band{bi + 1}.")
        fx_lines(self.fx_chain, "FX ")
        for b, chain in enumerate(self.fx_buses):
            if chain:
                fx_lines(chain, f"Bus{b + 1} FX ")
        out.append(f"Voicing: {'mono' if self.mono else 'poly'}, glide {self.glide*1000:.0f} ms")
        return out


@dataclass
class Note:
    start: float                  # seconds
    duration: float               # seconds
    pitch: float                  # MIDI note number (float allows detuning)
    velocity: float = 1.0
    # optional pitch-bend curve relative to `pitch`: [[t_rel_seconds, semis], ...]
    bend: Optional[list] = None

    def to_dict(self):
        return asdict(self)


def sync_to_hz(sync: str, bpm: float) -> float:
    """Convert a tempo division such as '1/8', '1/4T', '1/16D', '2/1' to Hz."""
    s = sync.strip()
    mult = 1.0
    if s.endswith("T"):
        mult, s = 2.0 / 3.0, s[:-1]
    elif s.endswith("D"):
        mult, s = 1.5, s[:-1]
    num, den = s.split("/")
    beats = 4.0 * float(num) / float(den) * mult   # length in quarter notes
    return bpm / 60.0 / beats


SYNC_DIVISIONS = ["4/1", "2/1", "1/1", "1/2", "1/2D", "1/2T", "1/4", "1/4D", "1/4T",
                  "1/8", "1/8D", "1/8T", "1/16", "1/16D", "1/16T", "1/32", "1/32T", "1/64"]
