"""Analysis-by-synthesis patch reconstruction.

Strategy (simplest patch that reaches the target accuracy wins):

1. Initial guess from Fourier analysis (pitch, envelope, cutoff, LFO, noise).
2. Table search: basic shapes first, then the user's stock Serum tables,
   each briefly tuned; keep the best few.
3. Structure search: try adding one component at a time (noise, unison, sub,
   2nd/3rd oscillator, LFO routings, filter envelope, distortion modes, OTT,
   EQ, glide). Keep a change only if it improves the score.
4. Pruning: remove each component in turn; if the score barely moves it
   was not needed (keeps the patch honest and simple).
5. Continuous fine-tune of every parameter with CMA-ES.
6. Only if the target is still not met and custom tables are allowed:
   build a wavetable from the measured harmonic spectra and re-tune.
"""
from __future__ import annotations

import logging
import math
import time
from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from ..analysis.features import Analysis
from ..synth.engine import Renderer
from ..synth.patch import (LFO, Distortion, EQ, Env, Filter, Mod, NoiseOsc, OTT, Osc, Patch,
                           SubOsc)
from ..synth.wavetables import (BASIC_TABLES, N_HARM_FEAT, StockIndex, TableBank,
                                custom_table_from_harmonics, harmonic_features,
                                lowpass_fit_error)
from . import params as P
from .score import Target

log = logging.getLogger(__name__)

OPT_SR = 22050
MAX_OPT_SECONDS = 6.0


@dataclass
class Options:
    target_score: float = 90.0
    allow_custom: bool = True          # allow generated wavetables
    stock_only: bool = False           # forbid basic generated shapes too (use Serum files)
    time_budget: float = 240.0         # seconds
    max_oscs: int = 3


@dataclass
class Result:
    patch: Patch
    score: float
    details: dict
    notes: list
    bank: TableBank
    history: list = field(default_factory=list)
    used_custom: bool = False
    stages: list = field(default_factory=list)


class Optimizer:
    def __init__(self, y: np.ndarray, sr: int, analysis: Analysis, options: Options | None = None,
                 bank: TableBank | None = None, stock: StockIndex | None = None,
                 progress: Callable[[str, float, float, str], None] | None = None):
        self.opts = options or Options()
        self.analysis = analysis
        self.bank = bank or TableBank()
        self.stock = stock
        self.progress_cb = progress
        if sr != OPT_SR:
            from ..audio import resample
            y = resample(y, sr, OPT_SR)
        self.full_y = y
        n = int(min(len(y), MAX_OPT_SECONDS * OPT_SR))
        self.y = y[:n]
        self.duration = n / OPT_SR
        self.target = Target(self.y, OPT_SR)
        self.renderer = Renderer(self.bank, OPT_SR)
        self.notes = [nt for nt in analysis.notes if nt.start < self.duration]
        self.t0 = time.time()
        self.best_score = -1.0
        self.evals = 0
        self.history: list = []
        self.stages: list = []

    # ------------------------------------------------------------------
    def time_left(self) -> float:
        return self.opts.time_budget - (time.time() - self.t0)

    def report(self, stage: str, msg: str = ""):
        frac = min(1.0, (time.time() - self.t0) / max(self.opts.time_budget, 1))
        if self.progress_cb:
            self.progress_cb(stage, frac, self.best_score, msg)
        log.info("[%s] %.1f %s", stage, self.best_score, msg)

    def evaluate(self, patch: Patch) -> float:
        self.evals += 1
        try:
            y = self.renderer.render(patch, self.notes, self.duration)
        except Exception as e:      # a broken hypothesis should never kill the run
            log.debug("render failed: %s", e)
            return 0.0
        if not np.all(np.isfinite(y)):
            return 0.0
        s = self.target.score(y)
        if s > self.best_score:
            self.best_score = s
        return s

    # ------------------------------------------------------------------
    def tune(self, patch: Patch, evals: int, focus: set[str] | None = None,
             sigma: float = 0.18) -> tuple[Patch, float]:
        """CMA-ES over the continuous parameters (normalised to 0..1)."""
        import cma
        ps = P.continuous_params(patch, focus)
        base = self.evaluate(patch)
        if not ps or evals <= 0:
            return patch, base
        x0 = P.vector(patch, ps)
        if len(ps) == 1:
            return self._tune_1d(patch, ps, x0[0], base, evals)
        popsize = max(6, min(14, 4 + int(3 * math.log(len(ps) + 1))))
        es = cma.CMAEvolutionStrategy(x0, sigma, {
            "bounds": [0.0, 1.0], "popsize": popsize, "maxfevals": evals,
            "verbose": -9, "seed": 1234, "tolfun": 1e-3})
        best_p, best_s = patch, base
        while not es.stop() and self.time_left() > 0:
            xs = es.ask()
            fits = []
            for x in xs:
                cand = P.apply(patch, ps, x)
                s = self.evaluate(cand)
                fits.append(-s)
                if s > best_s:
                    best_p, best_s = cand, s
            es.tell(xs, fits)
        return best_p, best_s

    def _tune_1d(self, patch, ps, u0, base, evals):
        """Coarse grid then local refinement (CMA-ES needs >= 2 dimensions)."""
        best_p, best_s, best_u = patch, base, u0
        coarse = max(5, min(9, evals // 2))
        for u in np.linspace(0, 1, coarse):
            cand = P.apply(patch, ps, [u])
            s = self.evaluate(cand)
            if s > best_s:
                best_p, best_s, best_u = cand, s, u
        step = 0.5 / coarse
        for _ in range(max(2, evals - coarse) // 2):
            for u in (best_u - step, best_u + step):
                if 0 <= u <= 1:
                    cand = P.apply(patch, ps, [u])
                    s = self.evaluate(cand)
                    if s > best_s:
                        best_p, best_s, best_u = cand, s, u
            step *= 0.5
        return best_p, best_s

    # ------------------------------------------------------------------
    def initial_patch(self) -> Patch:
        a = self.analysis
        p = Patch(name="Reconstructed", bpm=a.bpm)
        p.oscs[0] = Osc(enabled=True, table="basic:saw", level=0.8)
        e = a.env
        p.envs[0] = Env(attack=float(np.clip(e["attack"], 0.0005, 0.5)), decay=float(np.clip(e["decay"] or 0.3, 0.02, 4)),
                        sustain=float(e["sustain"]), release=float(np.clip(e["release"], 0.005, 2)))
        # cutoff guess: frequency where the average spectrum falls 24 dB below its peak
        if a.spectrum_db is not None:
            above = np.nonzero(a.spectrum_db > -24)[0]
            fc = float(a.spectrum_hz[above[-1]]) if len(above) else 2000.0
        else:
            fc = 2000.0
        p.filter = Filter(enabled=True, type="lp24", cutoff=float(np.clip(fc, 60, 15000)), reso=0.1)
        p.mono = True
        bends_between = any(nt.bend for nt in a.notes)
        p.glide = 0.0 if not bends_between else 0.0
        if a.percussive:
            # drum-like: short amp env and a pitch envelope via env2
            p.oscs[0].table = "basic:sine"
            p.envs[0] = Env(attack=0.0008, decay=float(np.clip(e["decay"] or 0.2, 0.03, 2)), sustain=0.0,
                            release=0.05)
            p.envs[1] = Env(attack=0.0005, decay=0.05, sustain=0.0, release=0.05)
            p.mods.append(Mod("env2", "pitch", 0.4))
            p.filter.enabled = False
            if a.noise_ratio > 0.4:
                p.noise = NoiseOsc(enabled=True, color=0.6, level=0.6)
        else:
            if a.unison_hint > 0.35:
                p.oscs[0].unison = 3
                p.oscs[0].detune = float(np.clip(0.1 + a.unison_hint * 0.3, 0.05, 0.6))
            if a.lfo and a.lfo["strength"] > 0.5:
                lf = a.lfo
                pts = lf["shape_points"] + [[1.0, lf["shape_points"][0][1]]]
                p.lfos[0] = LFO(shape="custom", sync=lf["sync"], rate_hz=lf["rate_hz"], mode="trig", points=pts)
                if lf["feature"] == "volume":
                    p.mods.append(Mod("lfo1", "amp", -0.5))
                else:
                    p.mods.append(Mod("lfo1", "filter.cutoff", float(np.clip(lf["depth"] / 10, 0.1, 0.5))))
                    p.filter.cutoff = float(np.clip(p.filter.cutoff / 2 ** (lf["depth"] / 2), 40, 12000))
            if a.noise_ratio > 0.12:
                p.noise = NoiseOsc(enabled=True, color=float(np.clip(0.5 + a.noise_tilt / 24, 0, 1)),
                                   level=float(np.clip(a.noise_ratio, 0.05, 0.8)))
        return p

    # ------------------------------------------------------------------
    def table_candidates(self) -> list[tuple[str, float]]:
        """(table_id, position) ordered basic-first, then stock by match."""
        prof = self.analysis.harm_profile_db[:N_HARM_FEAT]
        out = []
        if not self.opts.stock_only:
            basics = {f"basic:{n}": self.bank.frames(f"basic:{n}") for n in BASIC_TABLES}
            for tid, fr in basics.items():
                err = lowpass_fit_error(harmonic_features(fr), prof)
                j = int(np.argmin(err))
                out.append((tid, j / max(1, len(fr) - 1), float(err[j])))
        if self.stock is not None and len(self.stock.ids):
            for tid, pos, err in self.stock.search(prof, k=10):
                out.append((tid, pos, err))
        out.sort(key=lambda r: r[2])
        basic = [r for r in out if r[0].startswith("basic:")][:4]
        stock = [r for r in out if r[0].startswith("serum:")][:6]
        return [(t, pos) for t, pos, _ in basic + stock]

    # ------------------------------------------------------------------
    def structure_moves(self, p: Patch) -> list[tuple[str, Callable[[Patch], Patch], set]]:
        a = self.analysis
        moves = []

        def mv(name, fn, focus):
            moves.append((name, fn, focus))

        if not p.noise.enabled:
            def f(q):
                q.noise = NoiseOsc(enabled=True, color=0.5, level=float(np.clip(a.noise_ratio, 0.05, 0.6)))
                return q
            mv("add noise", f, {"noise"})
        o0 = p.oscs[0]
        if o0.unison == 1:
            for n in (3, 5, 7):
                def f(q, n=n):
                    q.oscs[0].unison = n
                    q.oscs[0].detune = float(np.clip(0.1 + a.unison_hint * 0.4, 0.05, 0.8))
                    return q
                mv(f"unison x{n}", f, {"osc"})
        if not p.sub.enabled and not a.percussive:
            for shape in ("sine", "triangle"):
                def f(q, shape=shape):
                    q.sub = SubOsc(enabled=True, shape=shape, octave=-1, level=0.4)
                    return q
                mv(f"add sub ({shape})", f, {"sub"})
        free = [i for i in range(1, self.opts.max_oscs) if not p.oscs[i].enabled]
        if free:
            i = free[0]
            for label, tab, octv in (("same table -1 oct", o0.table, -1), ("same table +1 oct", o0.table, 1),
                                     ("saw +0", "basic:saw", 0), ("square -1 oct", "basic:square", -1)):
                if self.opts.stock_only and tab.startswith("basic:"):
                    continue
                def f(q, i=i, tab=tab, octv=octv):
                    q.oscs[i] = Osc(enabled=True, table=tab, octave=q.oscs[0].octave + octv,
                                    position=q.oscs[0].position, level=0.4, fine=7.0 if octv == 0 else 0.0)
                    return q
                mv(f"osc {'ABC'[i]}: {label}", f, {"osc"})
        used = {m.source for m in p.mods}
        lfo_slot = next((k for k in range(4) if f"lfo{k+1}" not in used), None)
        if a.lfo and lfo_slot is not None:
            lf = a.lfo
            shapes = ["custom", "sine", "triangle", "saw_down", "square"]
            dests = ["filter.cutoff", "osc1.pos", "amp"]
            if lf["feature"] == "volume":
                dests = ["amp", "filter.cutoff", "osc1.pos"]
            for dest in dests:
                for shape in shapes[:3]:
                    def f(q, dest=dest, shape=shape, k=lfo_slot):
                        q.lfos[k] = LFO(shape=shape, sync=lf["sync"], rate_hz=lf["rate_hz"], mode="trig",
                                        points=lf["shape_points"] + [[1.0, lf["shape_points"][0][1]]]
                                        if shape == "custom" else None)
                        amt = {"filter.cutoff": 0.3, "osc1.pos": 0.5, "amp": -0.5}[dest]
                        if dest == "filter.cutoff" and not q.filter.enabled:
                            q.filter.enabled = True
                        q.mods.append(Mod(f"lfo{k+1}", dest, amt))
                        return q
                    mv(f"LFO{lfo_slot+1} ({shape}, {lf['sync'] or round(lf['rate_hz'], 2)}) -> {dest}", f, {"mod"})
        if "env2" not in used:
            def f(q):
                q.envs[1] = Env(attack=0.001, decay=0.15, sustain=0.0, release=0.05)
                q.filter.enabled = True
                q.mods.append(Mod("env2", "filter.cutoff", 0.3))
                return q
            mv("Env2 -> filter cutoff (pluck)", f, {"mod", "filter"})
            if a.percussive:
                def f(q):
                    q.envs[1] = Env(attack=0.0005, decay=0.04, sustain=0.0, release=0.05)
                    q.mods.append(Mod("env2", "pitch", 0.4))
                    return q
                mv("Env2 -> pitch (drum drop)", f, {"mod"})
        if not p.distortion.enabled:
            for mode in ("soft", "hard", "tube", "sinefold", "diode"):
                def f(q, mode=mode):
                    q.distortion = Distortion(enabled=True, mode=mode, drive=0.35, mix=1.0)
                    return q
                mv(f"distortion ({mode})", f, {"fx"})
        if not p.ott.enabled:
            def f(q):
                q.ott = OTT(enabled=True, depth=0.6, amount=0.7)
                return q
            mv("OTT multiband compression", f, {"fx"})
        if not p.eq.enabled:
            def f(q):
                q.eq = EQ(enabled=True)
                return q
            mv("EQ shelves", f, {"fx"})
        if p.filter.enabled:
            for ft in ("lp12", "bp12", "hp12"):
                if ft != p.filter.type:
                    def f(q, ft=ft):
                        q.filter.type = ft
                        return q
                    mv(f"filter type {ft}", f, {"filter"})
        if p.mono and p.glide == 0 and len(self.notes) > 1:
            def f(q):
                q.glide = 0.03
                return q
            mv("glide/portamento", f, {"glide"})
        return moves

    def prune_moves(self, p: Patch):
        moves = []

        def mv(name, fn):
            moves.append((name, fn))
        if p.noise.enabled:
            mv("noise", lambda q: (setattr(q.noise, "enabled", False), q)[1])
        if p.sub.enabled:
            mv("sub", lambda q: (setattr(q.sub, "enabled", False), q)[1])
        for i in (1, 2):
            if p.oscs[i].enabled:
                mv(f"osc {'ABC'[i]}", lambda q, i=i: (setattr(q.oscs[i], "enabled", False), q)[1])
        if p.oscs[0].unison > 1:
            mv("unison", lambda q: (setattr(q.oscs[0], "unison", 1), q)[1])
        if p.distortion.enabled:
            mv("distortion", lambda q: (setattr(q.distortion, "enabled", False), q)[1])
        if p.ott.enabled:
            mv("OTT", lambda q: (setattr(q.ott, "enabled", False), q)[1])
        if p.eq.enabled:
            mv("EQ", lambda q: (setattr(q.eq, "enabled", False), q)[1])
        for j, m in enumerate(p.mods):
            mv(f"mod {m.source}->{m.dest}", lambda q, j=j: (q.mods.pop(j), q)[1])
        if p.filter.enabled and not any(m.dest.startswith("filter") for m in p.mods):
            mv("filter", lambda q: (setattr(q.filter, "enabled", False), q)[1])
        return moves

    # ------------------------------------------------------------------
    def run(self) -> Result:
        opts = self.opts
        p = self.initial_patch()
        score = self.evaluate(p)
        self.report("init", f"initial guess from Fourier analysis: {score:.1f}")
        self.stages.append({"stage": "analysis guess", "score": round(score, 2)})

        # --- 2. table search -------------------------------------------------
        cands = self.table_candidates()
        results = []
        per = max(30, int(min(80, self.time_left() * 0.25 / max(1, len(cands)) / 0.03)))
        for tid, pos in cands:
            if self.time_left() < opts.time_budget * 0.6:
                break
            q = p.copy()
            q.oscs[0].table = tid
            q.oscs[0].position = float(pos)
            q, s = self.tune(q, per, focus={"osc", "filter", "env"})
            results.append((s, tid, q))
            self.report("tables", f"{tid}: {s:.1f}")
        results.sort(key=lambda r: -r[0])
        if results:
            score, _, p = results[0]
        self.stages.append({"stage": "wavetable search", "score": round(score, 2),
                            "table": p.oscs[0].table,
                            "candidates": [{"table": t, "score": round(s, 2)} for s, t, _ in results]})

        # --- 3. greedy structure search -------------------------------------
        added = []
        while self.time_left() > opts.time_budget * 0.3 and score < 99.0:
            if added and self.accuracy(p, score) >= opts.target_score:
                break
            moves = self.structure_moves(p)
            # cheap screen: a short tune of only the new component's params
            screened = []
            for name, fn, focus in moves:
                if self.time_left() < opts.time_budget * 0.3:
                    break
                q = fn(p.copy())
                q, s = self.tune(q, 12, focus=focus, sigma=0.2)
                screened.append((s, name, q, focus))
            screened.sort(key=lambda r: -r[0])
            best = (score, None, None)
            for s0, name, q, focus in screened[:4]:
                if self.time_left() < opts.time_budget * 0.3:
                    break
                q, s = self.tune(q, 48, focus=focus | {"filter", "osc", "env"}, sigma=0.1)
                if s > best[0] + 0.25:
                    best = (s, name, q)
            if best[1] is None:
                break
            score, name, p = best
            added.append({"added": name, "score": round(score, 2)})
            self.report("structure", f"kept '{name}' -> {score:.1f}")
        self.stages.append({"stage": "structure search", "score": round(score, 2), "steps": added})

        # --- 4. pruning (ablation) ------------------------------------------
        removed = []
        changed = True
        while changed and self.time_left() > opts.time_budget * 0.15:
            changed = False
            for name, fn in self.prune_moves(p):
                q = fn(p.copy())
                s = self.evaluate(q)
                if s >= score - 0.3:
                    p, score = q, s
                    removed.append({"removed": name, "score": round(s, 2)})
                    self.report("prune", f"removed unnecessary {name} -> {s:.1f}")
                    changed = True
                    break
        self.stages.append({"stage": "pruning", "score": round(score, 2), "steps": removed})

        # --- 5. fine-tune ----------------------------------------------------
        budget = int(max(40, self.time_left() * 0.6 / 0.03))
        p, score = self.tune(p, budget, sigma=0.08)
        self.report("fine-tune", f"{score:.1f}")
        self.stages.append({"stage": "fine-tune", "score": round(score, 2)})

        # --- 6. custom wavetable ---------------------------------------------
        used_custom = False
        if self.accuracy(p, score) < opts.target_score and opts.allow_custom and self.time_left() > 5:
            q = self.custom_table_patch(p)
            if q is not None:
                q, s = self.tune(q, int(max(30, self.time_left() * 0.9 / 0.03)), sigma=0.1)
                self.report("custom", f"custom wavetable: {s:.1f}")
                self.stages.append({"stage": "custom wavetable", "score": round(s, 2)})
                if s > score + 0.5:
                    p, score, used_custom = q, s, True

        total, details = self.final_score(p)
        return Result(patch=p, score=total, details=details, notes=self.analysis.notes, bank=self.bank,
                      history=self.history, used_custom=used_custom, stages=self.stages)

    # ------------------------------------------------------------------
    def custom_table_patch(self, p: Patch) -> Patch | None:
        """Bake the measured harmonic spectra into a wavetable.

        With a detected LFO the frames are one LFO cycle (phase-folded) and
        the LFO sweeps the table position, which is how wub basses are built.
        Without one, frames follow the longest note over time and Env2
        sweeps the position."""
        a = self.analysis
        mag, ph, times = a.harm_mag, a.harm_phase, a.times
        loud = a.rms_db > a.rms_db.max() - 30
        if not loud.any():
            return None
        q = p.copy()
        frames_mag, frames_ph = [], []
        if a.lfo and a.lfo["strength"] > 0.4:
            period = a.lfo["period_s"]
            nb = 32
            acc = [[] for _ in range(nb)]
            for nt in a.notes:
                i0, i1 = np.searchsorted(times, [nt.start, nt.start + nt.duration])
                for i in range(i0, i1):
                    if loud[i]:
                        b = int(((times[i] - nt.start) / period % 1.0) * nb)
                        acc[min(b, nb - 1)].append(i)
            for b in range(nb):
                idx = acc[b] or acc[(b - 1) % nb] or [int(np.argmax(a.rms_db))]
                m = mag[idx]
                m = m / (m.max(axis=1, keepdims=True) + 1e-12)
                frames_mag.append(np.median(m, axis=0))
                frames_ph.append(np.angle(np.exp(1j * ph[idx]).mean(axis=0)))
            tid = self.bank.add_custom(custom_table_from_harmonics(np.array(frames_mag), np.array(frames_ph)))
            k = next((k for k in range(4) if not any(m.source == f"lfo{k+1}" for m in q.mods)), 3)
            q.lfos[k] = LFO(shape="saw_up", sync=a.lfo["sync"], rate_hz=a.lfo["rate_hz"], mode="trig")
            q.oscs[0].position = 0.0
            q.mods = [m for m in q.mods if m.dest not in ("osc1.pos", "filter.cutoff")]
            q.mods.append(Mod(f"lfo{k+1}", "osc1.pos", 1.0))
        else:
            nt = max(a.notes, key=lambda n: n.duration)
            i0, i1 = np.searchsorted(times, [nt.start, nt.start + nt.duration])
            idx = [i for i in range(i0, i1) if loud[i]] or [int(np.argmax(a.rms_db))]
            nfr = int(min(64, len(idx)))
            # env2 decays exponentially (pos = 1 - exp(-4.6 t / d)); sample frames to match
            d = max(nt.duration, 0.05)
            pos = np.linspace(0, 0.99, nfr)
            t_sel = -d / 4.6 * np.log(1 - pos)
            sel = [idx[min(len(idx) - 1, int(np.searchsorted(times[idx] - nt.start, t)))] for t in t_sel]
            m = mag[sel]
            frames_mag = m / (m.max(axis=1, keepdims=True) + 1e-12)
            frames_ph = ph[sel]
            tid = self.bank.add_custom(custom_table_from_harmonics(frames_mag, frames_ph))
            q.envs[1] = Env(attack=0.0005, decay=d, sustain=0.0, release=0.05)
            q.mods = [m for m in q.mods if m.dest not in ("osc1.pos", "filter.cutoff")]
            q.oscs[0].position = 1.0
            q.mods.append(Mod("env2", "osc1.pos", -1.0))
        q.oscs[0].table = tid
        q.oscs[0].unison = 1
        q.filter.enabled = False      # the filter is baked into the table
        return q

    def accuracy(self, p: Patch, raw: float) -> float:
        """Phase-adjusted accuracy for a raw score (see final_score)."""
        y = self.renderer.render(p, self.notes, self.duration)
        alt = Renderer(self.bank, OPT_SR, seed=11).render(p, self.notes, self.duration)
        self_sim = float(np.clip(Target(alt, OPT_SR).score(y), 1.0, 100.0))
        return min(100.0, raw / self_sim * 100.0)

    def final_score(self, p: Patch):
        """Raw score plus a phase-adjusted accuracy.

        Unison, noise and random oscillator phase make every render of the
        *same* patch slightly different (Serum behaves the same way), so even
        a perfect patch cannot reach 100 raw. We measure that floor by
        comparing the patch with itself rendered with different random
        phases and report raw / self_similarity as the accuracy."""
        y = self.renderer.render(p, self.notes, self.duration)
        raw, details = self.target.score(y, detail=True)
        selfs = []
        for seed in (11, 23):
            alt = Renderer(self.bank, OPT_SR, seed=seed).render(p, self.notes, self.duration)
            selfs.append(Target(alt, OPT_SR).score(y))
        self_sim = float(np.clip(np.mean(selfs), 1.0, 100.0))
        accuracy = float(min(100.0, raw / self_sim * 100.0))
        details = dict(details)
        details["raw_score"] = round(raw, 2)
        details["self_similarity"] = round(self_sim, 2)
        return accuracy, details


def reconstruct(y, sr, analysis, options=None, bank=None, stock=None, progress=None) -> Result:
    return Optimizer(y, sr, analysis, options, bank, stock, progress).run()
