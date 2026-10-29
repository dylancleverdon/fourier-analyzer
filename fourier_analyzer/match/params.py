"""Addressing patch parameters by path and normalising them for search."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..synth.patch import Patch


def _step(obj, part: str):
    if part.isdigit():
        return obj[int(part)]
    return obj[part] if isinstance(obj, dict) else getattr(obj, part)


def get_attr(p: Patch, path: str):
    obj = p
    for part in path.split("."):
        obj = _step(obj, part)
    return obj


def set_attr(p: Patch, path: str, value) -> None:
    parts = path.split(".")
    obj = p
    for part in parts[:-1]:
        obj = _step(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    elif isinstance(obj, dict):
        obj[last] = value
    else:
        setattr(obj, last, value)


@dataclass
class Param:
    path: str
    lo: float
    hi: float
    log: bool = False

    def to_unit(self, v: float) -> float:
        v = min(max(v, self.lo), self.hi)
        if self.log:
            return (math.log(v) - math.log(self.lo)) / (math.log(self.hi) - math.log(self.lo))
        return (v - self.lo) / (self.hi - self.lo)

    def from_unit(self, u: float) -> float:
        u = min(max(u, 0.0), 1.0)
        if self.log:
            return math.exp(math.log(self.lo) + u * (math.log(self.hi) - math.log(self.lo)))
        return self.lo + u * (self.hi - self.lo)


MOD_RANGES = {
    "filter.cutoff": (-0.8, 0.8), "pitch": (-1.0, 1.0), "amp": (-1.0, 0.0),
    "osc1.pos": (-1, 1), "osc2.pos": (-1, 1), "osc3.pos": (-1, 1),
    "osc1.level": (-1, 1), "osc2.level": (-1, 1), "osc3.level": (-1, 1),
    "dist.drive": (-1, 1), "noise.level": (-1, 1), "filter.reso": (-1, 1),
    "filter2.cutoff": (-0.8, 0.8), "osc1.warp": (-1, 1), "osc2.warp": (-1, 1), "osc3.warp": (-1, 1),
}

# continuous knobs of each effect module in Patch.fx_chain: key -> (lo, hi, log)
FX_RANGES = {
    "distortion": {"drive": (0.0, 1.0, False), "mix": (0.0, 1.0, False)},
    "flanger": {"rate_hz": (0.02, 8.0, True), "depth": (0.0, 1.0, False), "feedback": (0.0, 0.95, False),
                "mix": (0.0, 1.0, False)},
    "phaser": {"rate_hz": (0.02, 8.0, True), "depth": (0.0, 1.0, False), "freq": (100.0, 8000.0, True),
               "feedback": (0.0, 0.95, False), "mix": (0.0, 1.0, False)},
    "chorus": {"rate_hz": (0.05, 5.0, True), "depth": (0.0, 1.0, False), "delay_ms": (1.0, 25.0, True),
               "mix": (0.0, 1.0, False)},
    "delay": {"feedback": (0.0, 0.9, False), "freq": (200.0, 12000.0, True), "mix": (0.0, 0.8, False)},
    "comp": {"depth": (0.0, 1.0, False), "amount": (0.0, 1.0, False)},
    "reverb": {"size": (0.0, 1.0, False), "decay": (0.2, 8.0, True), "damping": (0.0, 1.0, False),
               "mix": (0.0, 0.8, False)},
    "eq": {"gain1": (-12.0, 12.0, False), "gain2": (-12.0, 12.0, False), "freq1": (30.0, 1000.0, True),
           "freq2": (1000.0, 14000.0, True)},
    "filter": {"cutoff": (30.0, 16000.0, True), "reso": (0.0, 0.9, False), "mix": (0.0, 1.0, False)},
    "hyperdim": {"detune": (0.0, 1.0, False), "mix": (0.0, 1.0, False)},
    "bode": {"shift_hz": (-500.0, 500.0, False), "mix": (0.0, 1.0, False)},
    "convolve": {"size": (0.0, 1.0, False), "decay": (0.1, 6.0, True), "mix": (0.0, 0.8, False)},
    "utility": {"gain_db": (-12.0, 12.0, False)},
    "split": {"freq": (60.0, 4000.0, True)},
    "split3": {"freq": (60.0, 1000.0, True), "freq2": (1000.0, 10000.0, True)},
}


def continuous_params(p: Patch, focus: set[str] | None = None) -> list[Param]:
    """Search space for the current patch structure. ``focus`` limits it to
    groups: osc, filter, env, mod, noise, sub, fx, glide."""
    want = (lambda g: True) if focus is None else (lambda g: g in focus)
    ps: list[Param] = []
    for i, o in p.active_oscs():
        if want("osc") and o.type == "sample":
            ps.append(Param(f"oscs.{i}.smp.start", 0.0, 0.5))
            if o.smp.loop != "off":
                ps.append(Param(f"oscs.{i}.smp.loop_start", 0.0, 0.9))
        elif want("osc") and o.type == "granular":
            ps += [Param(f"oscs.{i}.smp.start", 0.0, 0.9), Param(f"oscs.{i}.gran.density", 2.0, 200.0, log=True),
                   Param(f"oscs.{i}.gran.length", 0.01, 0.5, log=True), Param(f"oscs.{i}.gran.rand_offset", 0.0, 1.0)]
        elif want("osc") and o.type == "spectral":
            ps += [Param(f"oscs.{i}.smp.start", 0.0, 0.9), Param(f"oscs.{i}.spec.freq_hi", 1000.0, 20000.0, log=True)]
        if want("osc"):
            if o.type == "wavetable":
                ps.append(Param(f"oscs.{i}.position", 0.0, 1.0))
            if i > 0:
                ps.append(Param(f"oscs.{i}.level", 0.02, 1.0))
                ps.append(Param(f"oscs.{i}.fine", -50.0, 50.0))
            if o.warp != "none":
                ps.append(Param(f"oscs.{i}.warp_amount", 0.0, 1.0))
            if o.warp2 != "none":
                ps.append(Param(f"oscs.{i}.warp2_amount", 0.0, 1.0))
            if o.unison > 1:
                ps.append(Param(f"oscs.{i}.detune", 0.01, 1.0))
                ps.append(Param(f"oscs.{i}.blend", 0.0, 1.0))
    if p.filter.enabled and want("filter"):
        ps += [Param("filter.cutoff", 30.0, 16000.0, log=True), Param("filter.reso", 0.0, 0.9)]
        if p.filter.drive > 0:
            ps.append(Param("filter.drive", 0.0, 1.0))
        if p.filter.type not in ("lp12", "lp24", "hp12", "hp24", "bp12"):
            ps.append(Param("filter.var", 0.0, 1.0))
        if p.filter.mix < 1.0:
            ps.append(Param("filter.mix", 0.0, 1.0))
    if p.filter2.enabled and want("filter"):
        ps += [Param("filter2.cutoff", 30.0, 16000.0, log=True), Param("filter2.reso", 0.0, 0.9),
               Param("filter2.mix", 0.0, 1.0)]
    if want("env"):
        ps += [Param("envs.0.attack", 0.0005, 0.6, log=True), Param("envs.0.decay", 0.01, 6.0, log=True),
               Param("envs.0.sustain", 0.0, 1.0), Param("envs.0.release", 0.005, 2.0, log=True)]
        e0 = p.envs[0]
        if (e0.attack_curve, e0.decay_curve, e0.release_curve) != (0.5, 0.666, 0.666):
            # curve shapes are searched once the "envelope curves" move turned them on
            ps += [Param("envs.0.attack_curve", 0.0, 1.0), Param("envs.0.decay_curve", 0.0, 1.0),
                   Param("envs.0.release_curve", 0.0, 1.0)]
        if e0.hold > 0:
            ps.append(Param("envs.0.hold", 0.001, 1.0, log=True))
    used = {m.source for m in p.mods if m.amount}
    if want("mod"):
        for j, m in enumerate(p.mods):
            lo, hi = MOD_RANGES.get(m.dest, (-1, 1))
            ps.append(Param(f"mods.{j}.amount", lo, hi))
        for k in (1, 2, 3):
            if f"env{k+1}" in used:
                ps += [Param(f"envs.{k}.attack", 0.0005, 0.5, log=True),
                       Param(f"envs.{k}.decay", 0.005, 4.0, log=True),
                       Param(f"envs.{k}.sustain", 0.0, 1.0)]
        for k in range(len(p.lfos)):
            if f"lfo{k+1}" in used:
                ps.append(Param(f"lfos.{k}.phase", 0.0, 1.0))
                if not p.lfos[k].sync:
                    ps.append(Param(f"lfos.{k}.rate_hz", 0.05, 30.0, log=True))
    if p.noise.enabled and want("noise"):
        ps += [Param("noise.level", 0.0, 1.5), Param("noise.color", 0.0, 1.0)]
    if p.sub.enabled and want("sub"):
        ps.append(Param("sub.level", 0.0, 1.5))
    if want("fx"):
        if p.distortion.enabled:
            ps += [Param("distortion.drive", 0.0, 1.0), Param("distortion.mix", 0.0, 1.0)]
        if p.ott.enabled:
            ps += [Param("ott.depth", 0.0, 1.0), Param("ott.amount", 0.0, 1.0)]
        if p.eq.enabled:
            ps += [Param("eq.low_gain", -12.0, 12.0), Param("eq.high_gain", -12.0, 12.0),
                   Param("eq.high_freq", 1500.0, 12000.0, log=True)]
        ps += _fx_params(p.fx_chain, "fx_chain")
        for b, chain in enumerate(p.fx_buses):
            if chain:
                ps += _fx_params(chain, f"fx_buses.{b}")
    if p.mono and p.glide > 0 and want("glide"):
        ps.append(Param("glide", 0.001, 0.4, log=True))
    return ps


def _fx_params(chain: list, prefix: str) -> list[Param]:
    out = []
    for j, m in enumerate(chain or []):
        if not m.get("enabled", True):
            continue
        for key, (lo, hi, log) in FX_RANGES.get(m.get("type"), {}).items():
            if key in m:
                out.append(Param(f"{prefix}.{j}.{key}", lo, hi, log))
        for b, band in enumerate(m.get("bands") or []):
            out += _fx_params(band, f"{prefix}.{j}.bands.{b}")
    return out


def vector(p: Patch, ps: list[Param]) -> list[float]:
    return [pp.to_unit(float(get_attr(p, pp.path))) for pp in ps]


def apply(p: Patch, ps: list[Param], x) -> Patch:
    q = p.copy()
    for pp, u in zip(ps, x):
        set_attr(q, pp.path, pp.from_unit(float(u)))
    return q
