"""Analysis-by-synthesis patch reconstruction (custom wavetables only).

1. Custom wavetables are built from the measured harmonic spectra of the
   original (de-ducked if sidechain compression was detected):
     - "single cycle": the median spectrum as one frame,
     - "evolving": frames following the longest note through time (Env2
       sweeps the table position),
     - "LFO cycle": frames folded over one detected LFO period (the LFO
       sweeps the position; how wub basses are built).
   Each is tuned briefly and the best is kept.
2. Structure search: try adding one component at a time (noise, unison,
   sub, a second custom oscillator, LFO routings, filter, envelopes,
   distortion, OTT, EQ, glide); keep a change only if it improves the fit.
3. Pruning: remove each component in turn; drop it if the fit holds.
4. Fine-tune every parameter with CMA-ES.
5. Waveform stage: phase, fine tuning and attack are tuned to maximise the
   literal waveform correlation with the original.

Renders are multiplied by the measured sidechain gain before comparison, so
ducking that happens in the DAW is not mistaken for part of the patch.
Search stages use a perceptual spectro-temporal objective (smooth, good for
search); the reported result uses the objective metrics in metrics.py.
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
from ..synth.wavetables import StockIndex, TableBank, custom_table_from_harmonics
from . import params as P
from .metrics import Reference
from .score import Target

log = logging.getLogger(__name__)

OPT_SR = 22050
STAGE_TEXT = {
    "init": "Initial guess from Fourier analysis",
    "tables": "Building custom wavetables from the measured harmonics and picking the best",
    "structure": "Structure search: trying to add noise / unison / LFOs / FX one at a time",
    "prune": "Pruning: removing components that don't improve the match",
    "fine-tune": "Fine-tuning every parameter together (CMA-ES)",
    "waveform": "Aligning phase and tuning to maximise waveform similarity",
    "final": "Final scoring",
}
MAX_OPT_SECONDS = 6.0


@dataclass
class Options:
    target_score: float = 90.0         # waveform similarity (%) that ends the search early
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
                 progress: Callable[..., None] | None = None,
                 context_gain: np.ndarray | None = None):
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
        self.ref = Reference(self.y, OPT_SR)
        if context_gain is not None:
            g = np.asarray(context_gain, dtype=np.float64)[:n]
            self.gain = np.pad(g, (0, n - len(g)), constant_values=1.0)
        else:
            self.gain = None
        self.renderer = Renderer(self.bank, OPT_SR)
        self.notes = [nt for nt in analysis.notes if nt.start < self.duration]
        self.t0 = time.time()
        self.best_score = -1.0
        self.evals = 0
        self.stage = "init"
        self.activity = ""
        self._stage_t0 = self.t0
        self._stage_evals = 0
        self._last_beat = 0.0
        self.history: list = []
        self.stages: list = []
        self.custom_ids: dict = {}
        self.reserve = 0.0          # seconds kept back for later stages

    # ------------------------------------------------------------------
    def time_left(self) -> float:
        return self.opts.time_budget - (time.time() - self.t0)

    def set_stage(self, stage: str, activity: str = ""):
        if stage != self.stage:
            self.stage, self._stage_t0, self._stage_evals = stage, time.time(), self.evals
        self.activity = activity
        self.report(stage)

    def status(self) -> dict:
        now = time.time()
        st_el = max(now - self._stage_t0, 1e-6)
        n = self.evals - self._stage_evals
        return {"stage": self.stage, "stage_text": STAGE_TEXT.get(self.stage, self.stage),
                "activity": self.activity, "evals": self.evals, "stage_evals": n,
                "rate": round(n / st_el, 1), "stage_elapsed": round(st_el, 1),
                "elapsed": round(now - self.t0, 1), "budget": self.opts.time_budget,
                "best": round(self.best_score, 2)}

    def report(self, stage: str, msg: str = ""):
        frac = min(1.0, (time.time() - self.t0) / max(self.opts.time_budget, 1))
        self._last_beat = time.time()
        if self.progress_cb:
            try:
                self.progress_cb(stage, frac, self.best_score, msg, self.status())
            except TypeError:       # older 4-argument callbacks
                self.progress_cb(stage, frac, self.best_score, msg)
        if msg:
            log.info("[%s] %.1f %s", stage, self.best_score, msg)

    def render(self, patch: Patch) -> np.ndarray:
        """Patch output as heard in the track (sidechain applied)."""
        y = self.renderer.render(patch, self.notes, self.duration)
        return y * self.gain if self.gain is not None else y

    def evaluate(self, patch: Patch, objective: str = "search") -> float:
        self.evals += 1
        try:
            y = self.render(patch)
        except Exception as e:      # a broken hypothesis should never kill the run
            log.debug("render failed: %s", e)
            return 0.0
        if not np.all(np.isfinite(y)):
            return 0.0
        if objective == "wave":
            s = 0.7 * self.ref.waveform(y) + 0.3 * self.ref.spectral(y)
        else:
            # half perceptual (smooth landscape), half the reported spectral
            # similarity, so the search improves the number the user sees
            s = 0.5 * self.target.score(y) + 0.5 * self.ref.spectral(y)
            if s > self.best_score:
                self.best_score = s
        if time.time() - self._last_beat > 1.0:      # live heartbeat for the UI
            self.report(self.stage)
        return s

    # ------------------------------------------------------------------
    def tune(self, patch: Patch, evals: int, focus: set[str] | None = None,
             sigma: float = 0.18, objective: str = "search", params=None) -> tuple[Patch, float]:
        """CMA-ES over the continuous parameters (normalised to 0..1)."""
        import cma
        ps = params if params is not None else P.continuous_params(patch, focus)
        base = self.evaluate(patch, objective)
        if not ps or evals <= 0:
            return patch, base
        x0 = P.vector(patch, ps)
        if len(ps) == 1:
            return self._tune_1d(patch, ps, x0[0], base, evals, objective)
        popsize = max(6, min(14, 4 + int(3 * math.log(len(ps) + 1))))
        es = cma.CMAEvolutionStrategy(x0, sigma, {
            "bounds": [0.0, 1.0], "popsize": popsize, "maxfevals": evals,
            "verbose": -9, "seed": 1234, "tolfun": 1e-3})
        best_p, best_s = patch, base
        while not es.stop() and self.time_left() > self.reserve:
            xs = es.ask()
            fits = []
            for x in xs:
                cand = P.apply(patch, ps, x)
                s = self.evaluate(cand, objective)
                fits.append(-s)
                if s > best_s:
                    best_p, best_s = cand, s
            es.tell(xs, fits)
        return best_p, best_s

    def _tune_1d(self, patch, ps, u0, base, evals, objective="search"):
        """Coarse grid then local refinement (CMA-ES needs >= 2 dimensions)."""
        best_p, best_s, best_u = patch, base, u0
        coarse = max(5, min(9, evals // 2))
        for u in np.linspace(0, 1, coarse):
            cand = P.apply(patch, ps, [u])
            s = self.evaluate(cand, objective)
            if s > best_s:
                best_p, best_s, best_u = cand, s, u
        step = 0.5 / coarse
        for _ in range(max(2, evals - coarse) // 2):
            for u in (best_u - step, best_u + step):
                if 0 <= u <= 1:
                    cand = P.apply(patch, ps, [u])
                    s = self.evaluate(cand, objective)
                    if s > best_s:
                        best_p, best_s, best_u = cand, s, u
            step *= 0.5
        return best_p, best_s

    # ------------------------------------------------------------------
    def initial_patch(self) -> Patch:
        a = self.analysis
        p = Patch(name="Reconstructed", bpm=a.bpm)
        p.oscs[0] = Osc(enabled=True, table="basic:saw", level=0.8, random_phase=0.0)
        e = a.env
        p.envs[0] = Env(attack=float(np.clip(e["attack"], 0.0005, 0.5)), decay=float(np.clip(e["decay"] or 0.3, 0.02, 4)),
                        sustain=float(e["sustain"]), release=float(np.clip(e["release"], 0.005, 2)))
        # cutoff guess: frequency where the average spectrum falls 24 dB below its peak
        if a.spectrum_db is not None:
            above = np.nonzero(a.spectrum_db > -24)[0]
            fc = float(a.spectrum_hz[above[-1]]) if len(above) else 2000.0
        else:
            fc = 2000.0
        # the custom wavetable already contains the filtered spectrum, so the
        # filter starts disabled (structure search can still add one)
        p.filter = Filter(enabled=False, type="lp24", cutoff=float(np.clip(fc, 60, 15000)), reso=0.1)
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
            if a.lfo and a.lfo["strength"] > 0.5 and a.lfo["feature"] == "volume":
                lf = a.lfo
                pts = lf["shape_points"] + [[1.0, lf["shape_points"][0][1]]]
                p.lfos[0] = LFO(shape="custom", sync=lf["sync"], rate_hz=lf["rate_hz"], mode="trig", points=pts)
                p.mods.append(Mod("lfo1", "amp", -0.5))
            if a.noise_ratio > 0.12:
                p.noise = NoiseOsc(enabled=True, color=float(np.clip(0.5 + a.noise_tilt / 24, 0, 1)),
                                   level=float(np.clip(a.noise_ratio, 0.05, 0.8)))
        return p

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
            opts_b = [("same table -1 oct", o0.table, -1, 0.0), ("same table +1 oct", o0.table, 1, 0.0),
                      ("same table detuned", o0.table, 0, 7.0)]
            single = self.custom_ids.get("single cycle")
            if single and single != o0.table:
                opts_b.append(("single-cycle table", single, 0, 0.0))
            for label, tab, octv, fine in opts_b:
                def f(q, i=i, tab=tab, octv=octv, fine=fine):
                    q.oscs[i] = Osc(enabled=True, table=tab, octave=q.oscs[0].octave + octv,
                                    position=q.oscs[0].position, level=0.4, fine=fine, random_phase=0.0)
                    return q
                mv(f"osc {'ABC'[i]}: {label}", f, {"osc"})
        # Serum warp modes on osc A: FM / AM / RM from osc B (B as a pure
        # modulator at level 0) or from the sub oscillator
        if o0.warp == "none":
            free_b = not p.oscs[1].enabled
            for mode, ratio in (("fm_b", 0), ("fm_b", 12), ("fm_b", -12), ("fm_b", 7), ("fm_sub", None),
                                ("rm_b", 12), ("am_b", 0)):
                if mode.endswith("_b") and not free_b:
                    continue

                def f(q, mode=mode, ratio=ratio):
                    q.oscs[0].warp, q.oscs[0].warp_amount = mode, 0.25
                    if mode.endswith("_b"):
                        q.oscs[1] = Osc(enabled=True, table=self.sine_id(), level=0.0, random_phase=0.0,
                                        octave=q.oscs[0].octave + ratio // 12, semi=ratio % 12)
                    else:
                        q.sub = SubOsc(enabled=True, shape="sine", octave=-1, level=0.0)
                    return q
                label = {"fm": "FM", "am": "AM", "rm": "RM"}[mode[:2]]
                src = "sub" if mode.endswith("_sub") else f"osc B sine {'+' if (ratio or 0) >= 0 else ''}{ratio} st"
                mv(f"{label} warp on osc A from {src}", f, {"osc"})
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
        if not p.filter.enabled:
            for ft in ("lp24", "hp12"):
                def f(q, ft=ft):
                    q.filter.enabled, q.filter.type = True, ft
                    q.filter.cutoff = 8000.0 if ft == "lp24" else 40.0
                    return q
                mv(f"add filter ({ft})", f, {"filter"})
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
        if p.oscs[0].warp != "none":
            def drop_warp(q):
                q.oscs[0].warp, q.oscs[0].warp_amount = "none", 0.0
                if q.oscs[1].enabled and q.oscs[1].level == 0:
                    q.oscs[1].enabled = False
                if q.sub.enabled and q.sub.level == 0:
                    q.sub.enabled = False
                return q
            mv(f"{p.oscs[0].warp} warp", drop_warp)
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

        # --- 2. custom wavetables --------------------------------------------
        cands = self.custom_table_patches(p)
        self.set_stage("tables", f"{len(cands)} custom wavetable designs")
        results = []
        per = max(30, int(min(90, self.time_left() * 0.25 / max(1, len(cands)) / 0.03)))
        for label, q in cands:
            self.activity = f"tuning the '{label}' wavetable"
            q, s = self.tune(q, per, focus={"osc", "env", "mod"})
            results.append((s, label, q))
            self.report("tables", f"custom wavetable '{label}': {s:.1f}")
        results.sort(key=lambda r: -r[0])
        if results:
            score, label, p = results[0]
        self.stages.append({"stage": "custom wavetable", "score": round(score, 2), "table": label,
                            "candidates": [{"table": t, "score": round(s, 2)} for s, t, _ in results]})

        # --- 3. greedy structure search -------------------------------------
        added = []
        while self.time_left() > opts.time_budget * 0.3 and score < 99.0:
            if added and self.ref.waveform(self.render(p)) >= opts.target_score:
                break
            moves = self.structure_moves(p)
            self.set_stage("structure", f"screening {len(moves)} possible additions")
            # cheap screen: a short tune of only the new component's params
            screened = []
            for name, fn, focus in moves:
                if self.time_left() < opts.time_budget * 0.3:
                    break
                self.activity = f"screening: {name}"
                q = fn(p.copy())
                q, s = self.tune(q, 12, focus=focus, sigma=0.2)
                screened.append((s, name, q, focus))
            screened.sort(key=lambda r: -r[0])
            best = (score, None, None)
            for s0, name, q, focus in screened[:4]:
                if self.time_left() < opts.time_budget * 0.3:
                    break
                self.activity = f"testing best candidate: {name}"
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
        self.set_stage("prune", "removing components that do not help")
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
        self.set_stage("fine-tune", f"all {len(P.continuous_params(p))} parameters together")
        self.reserve = 0.25 * opts.time_budget      # keep time for waveform alignment
        p, score = self.tune(p, budget, sigma=0.08)
        self.report("fine-tune", f"{score:.1f}")
        self.stages.append({"stage": "fine-tune", "score": round(score, 2)})

        # --- 6. waveform alignment --------------------------------------------
        self.reserve = 0.0
        if self.time_left() > 2:
            self.set_stage("waveform", "start phase, fine tune and attack against the literal waveform")
            ps = []
            for i, o in p.active_oscs():
                ps += [P.Param(f"oscs.{i}.phase", 0.0, 1.0), P.Param(f"oscs.{i}.fine", -25.0, 25.0)]
            ps += [P.Param("envs.0.attack", 0.0005, 0.2, log=True), P.Param("envs.0.decay", 0.01, 6.0, log=True),
                   P.Param("envs.0.sustain", 0.0, 1.0), P.Param("envs.0.release", 0.005, 2.0, log=True)]
            before = self.ref.waveform(self.render(p))
            self.refine_note_timing(p)
            p, _ = self.tune(p, int(max(40, self.time_left() * 0.8 / 0.03)), sigma=0.25, objective="wave", params=ps)
            self.refine_note_timing(p, span=0.003)
            after = self.ref.waveform(self.render(p))
            self.report("waveform", f"waveform similarity {before:.1f} -> {after:.1f}")
            self.stages.append({"stage": "waveform alignment", "score": round(after, 2)})

        self.set_stage("final", "computing similarity metrics")
        metrics = self.ref.all(self.render(p))
        total = metrics["waveform_similarity"]
        return Result(patch=p, score=total, details=metrics, notes=self.analysis.notes, bank=self.bank,
                      history=self.history, used_custom=True, stages=self.stages)

    def refine_note_timing(self, p: Patch, span: float = 0.006, steps: int = 13) -> None:
        """Shift each note start (end fixed) within +-span to maximise the
        waveform correlation. The notes are shared with the analysis, so the
        exported MIDI gets the refined timing too."""
        for nt in self.notes:
            if self.time_left() < 1:
                return
            s0, e0 = nt.start, nt.start + nt.duration
            best_s, best_v = s0, self.ref.waveform(self.render(p))
            for d in np.linspace(-span, span, steps):
                nt.start = max(0.0, s0 + d)
                nt.duration = e0 - nt.start
                v = self.ref.waveform(self.render(p))
                if v > best_v:
                    best_s, best_v = nt.start, v
            nt.start, nt.duration = round(best_s, 5), round(e0 - best_s, 5)
            self._last_beat = 0     # keep the UI heartbeat alive

    def sine_id(self) -> str:
        """A one-frame custom sine table (modulator for FM/AM/RM)."""
        if "sine" not in self.custom_ids:
            k = np.zeros(2)
            k[0] = 1.0
            self.custom_ids["sine"] = self.bank.add_custom(custom_table_from_harmonics(k[None, :1]))
        return self.custom_ids["sine"]

    # ------------------------------------------------------------------
    def custom_table_patches(self, p: Patch) -> list[tuple[str, Patch]]:
        """Candidate patches, each with a wavetable built from the measured
        harmonic magnitudes and phases (relative to the fundamental)."""
        a = self.analysis
        mag, ph, times = a.harm_mag, a.harm_phase, a.times
        loud = (a.rms_db > a.rms_db.max() - 30) & (mag.max(axis=1) > 0)
        if not loud.any():
            loud = mag.max(axis=1) > 0
        out = []
        self.custom_ids = {}

        # frames keep their level relative to each other (a closing filter is
        # also a quieter frame); only the overall table is normalised
        ref = float(np.percentile(mag[loud].max(axis=1), 95)) + 1e-12 if loud.any() else 1.0

        def frames_of(idx):
            return np.clip(mag[idx] / ref, 0, 4)

        def circ_mean(idx):
            return np.angle(np.exp(1j * ph[idx]).mean(axis=0))

        idx_all = np.nonzero(loud)[0]
        if len(idx_all) == 0:
            return [("single cycle", p.copy())]
        # 1. single cycle: median spectrum
        tid = self.bank.add_custom(custom_table_from_harmonics(
            np.median(frames_of(idx_all), axis=0)[None, :], circ_mean(idx_all)[None, :]))
        self.custom_ids["single cycle"] = tid
        q = p.copy()
        q.oscs[0].table, q.oscs[0].position, q.oscs[0].random_phase = tid, 0.0, 0.0
        out.append(("single cycle", q))
        # 2. evolving through the longest note (Env2 sweeps the position)
        nt = max(a.notes, key=lambda n: n.duration)
        i0, i1 = np.searchsorted(times, [nt.start, nt.start + nt.duration])
        idx = [i for i in range(i0, i1) if loud[i]]
        if len(idx) >= 4:
            nfr = int(min(64, len(idx)))
            d = max(nt.duration, 0.05)
            pos = np.linspace(0, 0.99, nfr)
            t_sel = -d / 4.6 * np.log(1 - pos)   # Env2 decay is exponential
            sel = [idx[min(len(idx) - 1, int(np.searchsorted(times[idx] - nt.start, t)))] for t in t_sel]
            tid = self.bank.add_custom(custom_table_from_harmonics(frames_of(sel), ph[sel]))
            self.custom_ids["evolving"] = tid
            q = p.copy()
            q.oscs[0].table, q.oscs[0].position, q.oscs[0].random_phase = tid, 1.0, 0.0
            q.envs[1] = Env(attack=0.0005, decay=d, sustain=0.0, release=0.05)
            q.mods = [m for m in q.mods if m.dest != "osc1.pos"] + [Mod("env2", "osc1.pos", -1.0)]
            out.append(("evolving", q))
        # 3. one LFO cycle (the LFO sweeps the position)
        if a.lfo and a.lfo["strength"] > 0.4:
            period = a.lfo["period_s"]
            nb = 32
            acc = [[] for _ in range(nb)]
            for n in a.notes:
                j0, j1 = np.searchsorted(times, [n.start, n.start + n.duration])
                for i in range(j0, j1):
                    if loud[i]:
                        acc[min(int(((times[i] - n.start) / period % 1.0) * nb), nb - 1)].append(i)
            fm, fp = [], []
            for b in range(nb):
                ib = acc[b] or acc[(b - 1) % nb] or list(idx_all[:1])
                fm.append(np.median(frames_of(ib), axis=0))
                fp.append(circ_mean(ib))
            tid = self.bank.add_custom(custom_table_from_harmonics(np.array(fm), np.array(fp)))
            self.custom_ids["LFO cycle"] = tid
            q = p.copy()
            k = next((k for k in range(4) if not any(m.source == f"lfo{k+1}" for m in q.mods)), 3)
            q.lfos[k] = LFO(shape="saw_up", sync=a.lfo["sync"], rate_hz=a.lfo["rate_hz"], mode="trig")
            q.oscs[0].table, q.oscs[0].position, q.oscs[0].random_phase = tid, 0.0, 0.0
            q.mods = [m for m in q.mods if m.dest != "osc1.pos"] + [Mod(f"lfo{k+1}", "osc1.pos", 1.0)]
            out.append(("LFO cycle", q))
        return out


def reconstruct(y, sr, analysis, options=None, bank=None, progress=None, context_gain=None) -> Result:
    return Optimizer(y, sr, analysis, options, bank, None, progress, context_gain).run()
