"""Addressing patch parameters by path and normalising them for search."""
from __future__ import annotations

import math
from dataclasses import dataclass

from ..synth.patch import Patch


def get_attr(p: Patch, path: str):
    obj = p
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def set_attr(p: Patch, path: str, value) -> None:
    parts = path.split(".")
    obj = p
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
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
}


def continuous_params(p: Patch, focus: set[str] | None = None) -> list[Param]:
    """Search space for the current patch structure. ``focus`` limits it to
    groups: osc, filter, env, mod, noise, sub, fx, glide."""
    want = (lambda g: True) if focus is None else (lambda g: g in focus)
    ps: list[Param] = []
    for i, o in p.active_oscs():
        if want("osc"):
            ps.append(Param(f"oscs.{i}.position", 0.0, 1.0))
            if i > 0:
                ps.append(Param(f"oscs.{i}.level", 0.02, 1.0))
                ps.append(Param(f"oscs.{i}.fine", -50.0, 50.0))
            if o.warp != "none":
                ps.append(Param(f"oscs.{i}.warp_amount", 0.0, 1.0))
            if o.unison > 1:
                ps.append(Param(f"oscs.{i}.detune", 0.01, 1.0))
                ps.append(Param(f"oscs.{i}.blend", 0.0, 1.0))
    if p.filter.enabled and want("filter"):
        ps += [Param("filter.cutoff", 30.0, 16000.0, log=True), Param("filter.reso", 0.0, 0.9)]
        if p.filter.drive > 0:
            ps.append(Param("filter.drive", 0.0, 1.0))
    if want("env"):
        ps += [Param("envs.0.attack", 0.0005, 0.6, log=True), Param("envs.0.decay", 0.01, 6.0, log=True),
               Param("envs.0.sustain", 0.0, 1.0), Param("envs.0.release", 0.005, 2.0, log=True)]
    used = {m.source for m in p.mods if m.amount}
    if want("mod"):
        for j, m in enumerate(p.mods):
            lo, hi = MOD_RANGES.get(m.dest, (-1, 1))
            ps.append(Param(f"mods.{j}.amount", lo, hi))
        for k in (1, 2):
            if f"env{k+1}" in used:
                ps += [Param(f"envs.{k}.attack", 0.0005, 0.5, log=True),
                       Param(f"envs.{k}.decay", 0.005, 4.0, log=True),
                       Param(f"envs.{k}.sustain", 0.0, 1.0)]
        for k in range(4):
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
    if p.mono and p.glide > 0 and want("glide"):
        ps.append(Param("glide", 0.001, 0.4, log=True))
    return ps


def vector(p: Patch, ps: list[Param]) -> list[float]:
    return [pp.to_unit(float(get_attr(p, pp.path))) for pp in ps]


def apply(p: Patch, ps: list[Param], x) -> Patch:
    q = p.copy()
    for pp, u in zip(ps, x):
        set_attr(q, pp.path, pp.from_unit(float(u)))
    return q
