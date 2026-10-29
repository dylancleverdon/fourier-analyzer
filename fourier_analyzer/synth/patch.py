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

LFO_SHAPES = ("sine", "triangle", "saw_down", "saw_up", "square", "custom")
FILTER_TYPES = ("lp12", "lp24", "hp12", "hp24", "bp12")
DIST_MODES = ("soft", "hard", "tube", "sinefold", "linfold", "diode")
WARP_MODES = ("none", "fm_b", "fm_sub", "am_b", "rm_b", "rm_sub")
MOD_SOURCES = ("env1", "env2", "env3", "lfo1", "lfo2", "lfo3", "lfo4")
# amount 1.0 means: cutoff +10 octaves, pos +1 (whole table), level +1,
# pitch +48 semitones, drive +1.
MOD_DESTS = ("osc1.pos", "osc2.pos", "osc3.pos", "osc1.level", "osc2.level",
             "osc3.level", "filter.cutoff", "filter.reso", "pitch", "amp",
             "dist.drive", "noise.level")


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


@dataclass
class Env:
    attack: float = 0.001
    hold: float = 0.0
    decay: float = 0.3
    sustain: float = 1.0
    release: float = 0.05


@dataclass
class LFO:
    shape: str = "sine"
    rate_hz: float = 2.0
    sync: Optional[str] = None    # e.g. "1/8", "1/4T", "1/16D"; overrides rate_hz
    phase: float = 0.0            # 0..1
    mode: str = "trig"            # trig | free | env (one shot)
    points: Optional[list] = None  # custom shape: [[x, y], ...] in 0..1


@dataclass
class Mod:
    source: str = "lfo1"
    dest: str = "filter.cutoff"
    amount: float = 0.0           # -1..1, see MOD_DESTS comment
    bipolar: bool = False


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
    envs: list = field(default_factory=lambda: [Env(), Env(sustain=0.0), Env(sustain=0.0)])
    lfos: list = field(default_factory=lambda: [LFO(), LFO(), LFO(), LFO()])
    mods: list = field(default_factory=list)
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
        p.envs = [Env(**e) for e in d.get("envs", [])] or p.envs
        p.lfos = [LFO(**l) for l in d.get("lfos", [])] or p.lfos
        p.mods = [Mod(**m) for m in d.get("mods", [])]
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

    def describe(self) -> list[str]:
        """Human-readable summary lines for the UI."""
        out = []
        for i, o in self.active_oscs():
            uni = f", unison {o.unison} x detune {o.detune:.2f}" if o.unison > 1 else ""
            warp = f", warp {o.warp} {o.warp_amount:.2f}" if o.warp != "none" and o.warp_amount else ""
            out.append(f"Osc {'ABC'[i]}: {o.table} @ pos {o.position:.2f}, "
                       f"oct {o.octave:+d} semi {o.semi:+d} fine {o.fine:+.0f}c, "
                       f"level {o.level:.2f}{uni}{warp}")
        if self.sub.enabled:
            out.append(f"Sub: {self.sub.shape} oct {self.sub.octave:+d} level {self.sub.level:.2f}")
        if self.noise.enabled:
            out.append(f"Noise: color {self.noise.color:.2f} level {self.noise.level:.2f}")
        f = self.filter
        if f.enabled:
            out.append(f"Filter: {f.type} cutoff {f.cutoff:.0f} Hz reso {f.reso:.2f} drive {f.drive:.2f}")
        names = ["Env1 (amp)", "Env2", "Env3"]
        for n, e in zip(names, self.envs):
            out.append(f"{n}: A {e.attack*1000:.1f}ms H {e.hold*1000:.0f}ms D {e.decay*1000:.0f}ms "
                       f"S {e.sustain:.2f} R {e.release*1000:.0f}ms")
        used = {m.source for m in self.mods}
        for i, l in enumerate(self.lfos):
            if f"lfo{i+1}" in used:
                rate = l.sync if l.sync else f"{l.rate_hz:.2f} Hz"
                out.append(f"LFO{i+1}: {l.shape} {rate} ({l.mode})")
        for m in self.mods:
            if m.amount:
                out.append(f"Mod: {m.source} -> {m.dest} amount {m.amount:+.3f}"
                           + (" (bipolar)" if m.bipolar else ""))
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
