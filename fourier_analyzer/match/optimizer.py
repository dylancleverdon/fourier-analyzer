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
from ..synth.patch import (fx as fx_module, LFO, Distortion, EQ, Env, Filter, GranularCfg, Mod, NoiseOsc, OTT,
                           Osc, Patch, SampleCfg, SpectralCfg, SubOsc)
from ..synth.samples import SampleBank
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
    time_budget: float = 240.0         # seconds (long budgets add exploration rounds)
    max_oscs: int = 3
    stop: Callable[[], bool] | None = None   # returns True when the user stops / cancels
    keep_going: bool = False           # do not stop early at the target score (long runs)


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
    samples: SampleBank | None = None


class Optimizer:
    def __init__(self, y: np.ndarray, sr: int, analysis: Analysis, options: Options | None = None,
                 bank: TableBank | None = None, stock: StockIndex | None = None,
                 progress: Callable[..., None] | None = None,
                 context_gain: np.ndarray | None = None, samples: SampleBank | None = None,
                 source_audio: tuple[np.ndarray, int] | None = None):
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
        self.samples = samples if samples is not None else SampleBank()
        # full-rate audio for the sample / granular / spectral oscillators
        self.source_audio = source_audio or (self.full_y, OPT_SR)
        self._source_sid: str | None = None
        self._source_root = 60.0
        self.renderer = Renderer(self.bank, OPT_SR, samples=self.samples)
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
        self.stopped = False
        self.stop_time = 0.0
        self._in_final = False
        budget = self.opts.time_budget
        self.pass_budget = min(budget, self.FIRST_PASS)
        self.wave_reserve = min(0.25 * budget, max(0.25 * self.pass_budget, 600.0))
        # first pass ends where the old schedule started waveform alignment
        self.pass_end = self.t0 + (0.75 * self.pass_budget if budget <= self.FIRST_PASS
                                   else self.pass_budget)

    # ------------------------------------------------------------------
    FIRST_PASS = 900.0                 # the standard schedule never takes longer than this

    STOP_GRACE = 30.0                  # after a stop: finish with a short waveform alignment

    def time_left(self) -> float:
        if not self.stopped and self.opts.stop is not None and self.opts.stop():
            self.stopped = True
            self.stop_time = time.time()
            self.reserve = 0.0
        if self.stopped:
            return max(0.0, self.STOP_GRACE - (time.time() - self.stop_time)) if self._in_final else 0.0
        return self.opts.time_budget - (time.time() - self.t0)

    def pass_left(self) -> float:
        """Time left in the first (standard) pass."""
        return min(self.time_left(), self.pass_end - time.time())

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
        self._pitch_and_velocity(p)
        return p

    def _pitch_and_velocity(self, p: Patch) -> None:
        """Movement the analysis attributes to the patch itself: LFO / envelope
        -> Main Tuning, portamento, and Serum's VELO -> Amp for note dynamics.
        These are kept through pruning (they carry what the MIDI no longer has)."""
        a = self.analysis
        vels = [n.velocity for n in self.notes]
        if len(vels) > 1 and float(np.std(vels)) > 0.04:
            # analysis velocity = sqrt(level ratio); a squared velocity curve makes
            # Serum's voice gain follow the measured note levels
            p.voice_amp = 0.0
            p.velo_curve = [[0.0, 0.0, 0.6667], [1.0, 1.0, 0.5]]
            p.mods.append(Mod("velocity", "voiceamp", 0.5))
        if getattr(a, "pitch_lfo", None):
            lf = a.pitch_lfo
            k = 7
            pts = lf["shape_points"] + [[1.0, lf["shape_points"][0][1]]]
            p.lfos[k] = LFO(shape="custom", sync=lf["sync"], rate_hz=lf["rate_hz"], mode="trig", points=pts)
            p.mods.append(Mod(f"lfo{k + 1}", "pitch", float(lf["semis"]) / 48.0, bipolar=True))
        if getattr(a, "pitch_env", None) and not a.percussive:
            pe = a.pitch_env
            p.envs[2] = Env(attack=0.0005, decay=float(np.clip(pe["decay_s"], 0.005, 8)), sustain=0.0, release=0.05)
            p.mods.append(Mod("env3", "pitch", float(pe["semis"]) / 48.0))
        if getattr(a, "glide", 0.0):
            p.glide = float(a.glide)
            p.legato = True
            p.glide_always = False
        self.protected = {(m.source, m.dest) for m in p.mods if m.dest in ("pitch", "voiceamp")}

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
        # --- Serum 2 extras: phase warps, warp slot 2, unison stacks --------
        if o0.warp == "none":
            for code in ("Sync", "BendPos", "BendNeg", "PWM", "ASYMPos", "Mirror", "Quantize", "SelfPD",
                         "EvenOdd", "Flip", "DistSoft"):
                def f(q, code=code):
                    q.oscs[0].warp, q.oscs[0].warp_amount = code, 0.3
                    return q
                mv(f"warp {code} on osc A", f, {"osc"})
        elif o0.warp2 == "none":
            for code in ("Sync", "BendPos", "SelfPD", "DistSoft", "Mirror"):
                if code == o0.warp:
                    continue

                def f(q, code=code):
                    q.oscs[0].warp2, q.oscs[0].warp2_amount = code, 0.3
                    return q
                mv(f"second warp {code} on osc A", f, {"osc"})
        if o0.unison > 1 and o0.unison_mode == "linear":
            for mode in ("super", "exp", "inv", "random"):
                def f(q, mode=mode):
                    q.oscs[0].unison_mode = mode
                    return q
                mv(f"unison detune mode {mode}", f, {"osc"})
        for i in (1, 2):                       # phase warps on osc B / C
            oi = p.oscs[i]
            if oi.enabled and oi.level > 0 and oi.warp == "none":
                for code in ("Sync", "BendPos", "PWM", "SelfPD", "DistSoft"):
                    def f(q, i=i, code=code):
                        q.oscs[i].warp, q.oscs[i].warp_amount = code, 0.3
                        return q
                    mv(f"warp {code} on osc {'ABC'[i]}", f, {"osc"})
        e0 = p.envs[0]
        if (e0.attack_curve, e0.decay_curve, e0.release_curve) == (0.5, 0.666, 0.666):
            def f(q):
                q.envs[0].attack_curve, q.envs[0].decay_curve, q.envs[0].release_curve = 0.45, 0.62, 0.62
                return q
            mv("amp envelope curve shapes", f, {"env"})
        if e0.hold == 0 and e0.sustain < 0.9:
            def f(q):
                q.envs[0].hold = 0.02
                return q
            mv("amp envelope hold", f, {"env"})
        if o0.unison > 1 and o0.unison_stack == "none":
            for stack in ("Octave1", "OctaveFifth1", "Center12"):
                def f(q, stack=stack):
                    q.oscs[0].unison_stack = stack
                    return q
                mv(f"unison stack {stack}", f, {"osc"})
        # --- Serum filter models ---------------------------------------------
        if p.filter.enabled:
            for ft in ("Ladder24", "Comb+", "Phase+", "Flange+", "FormantI", "Diffuser", "Scream24", "LNH12"):
                if ft != p.filter.type:
                    def f(q, ft=ft):
                        q.filter.type, q.filter.var = ft, 0.5
                        return q
                    mv(f"filter type {ft}", f, {"filter"})
            if not p.filter2.enabled:
                for ft, cut in (("lp24", 6000.0), ("hp12", 80.0), ("Comb+", 800.0)):
                    def f(q, ft=ft, cut=cut):
                        q.filter2 = Filter(enabled=True, type=ft, cutoff=cut, reso=0.1)
                        for src in ("osc1", "osc2", "osc3", "noise", "sub"):
                            q.routing.setdefault(src, {})["balance"] = 0.5
                        return q
                    mv(f"filter 2 in parallel ({ft})", f, {"filter"})
        # --- FX rack modules ---------------------------------------------------
        have = {m.get("type") for m in p.fx_chain}
        cand = [("chorus", {}), ("flanger", {}), ("phaser", {}), ("hyperdim", {}), ("bode", {"shift_hz": 30.0}),
                ("filter", {"ftype": "L24", "cutoff": 6000.0}), ("comp", {"multiband": False}),
                ("distortion", {"mode": "Tube"}), ("distortion", {"mode": "Diode1"}),
                ("distortion", {"mode": "Asym"})]
        beat = 60.0 / max(p.bpm, 1.0)
        # time-based effects are always screened: their tails usually end up
        # inside the segmented notes, so the analysis can't flag them reliably
        cand += [("reverb", {"kind": "Hall"}), ("reverb", {"kind": "Plate", "decay": 0.8}),
                 ("convolve", {}), ("delay", {"time": beat / 4, "time2": beat / 4}),
                 ("delay", {"time": beat * 3 / 8, "time2": beat * 3 / 8})]
        if len(p.fx_chain) < 8:
            for kind, kw in cand:
                if kind in have and kind != "distortion":
                    continue

                def f(q, kind=kind, kw=kw):
                    q.fx_chain.append(fx_module(kind, **kw))
                    return q
                extra = kw.get("mode") or kw.get("kind") or ""
                mv(f"FX {kind}{' ' + extra if extra else ''}", f, {"fx"})
        if len(p.fx_chain) == 1 and p.fx_chain[0].get("type") == "distortion":
            def f(q):
                d = q.fx_chain.pop(0)
                q.fx_chain.append(fx_module("split", freq=250.0, bands=[[], [d]]))
                return q
            mv("multiband split: distort highs only", f, {"fx"})
        if p.mono and p.glide == 0 and len(self.notes) > 1:
            def f(q):
                q.glide = 0.03
                return q
            mv("glide/portamento", f, {"glide"})
        self._sample_moves(p, mv)
        return moves

    # ------------------------------------------------------------------
    def source_sample(self) -> tuple[str | None, float]:
        """One note of the original, embedded as a sample (with the sidechain
        ducking divided out): the source for sample / granular / spectral
        oscillators. Returns (sample id, root note)."""
        if self._source_sid is not None or not self.notes:
            return self._source_sid, self._source_root
        y, sr = self.source_audio
        notes = sorted(self.notes, key=lambda n: n.start)
        # the longest note, ending before the next one starts (plus a short tail)
        best = max(range(len(notes)), key=lambda k: notes[k].duration)
        n = notes[best]
        nxt = notes[best + 1].start if best + 1 < len(notes) else len(y) / sr
        a = max(0.0, n.start - 0.003)
        b = min(nxt, n.start + n.duration + 0.25, len(y) / sr)
        seg = np.asarray(y[int(a * sr): int(b * sr)], dtype=np.float64)
        if self.gain is not None and len(seg):
            t = a + np.arange(len(seg)) / sr
            g = np.interp(t, np.arange(len(self.gain)) / OPT_SR, self.gain)
            seg = seg / np.maximum(g, 0.1)
        if len(seg) < int(0.03 * sr):
            self._source_sid = ""
            return "", 60.0
        fade = min(len(seg) // 4, int(0.01 * sr))
        seg[-fade:] *= np.linspace(1.0, 0.0, fade)
        seg = seg / max(1e-9, float(np.max(np.abs(seg))))
        self._source_root = float(n.pitch)
        self._source_sid = self.samples.add(seg.astype(np.float32), sr, root=self._source_root,
                                            name="Fourier Analyzer source")
        return self._source_sid, self._source_root

    def _sample_moves(self, p: Patch, mv) -> None:
        """Sample, granular and spectral oscillators playing a note of the
        original: in place of osc A, or as an extra layer."""
        if any(o.enabled and o.type != "wavetable" for o in p.oscs):
            return
        sid, root = self.source_sample()
        if not sid:
            return
        free = next((i for i in (1, 2) if not p.oscs[i].enabled), None)

        def make(kind: str, level: float) -> Osc:
            o = Osc(enabled=True, type=kind, level=level, table="basic:sine")
            o.smp = SampleCfg(source=sid, root=root, loop="forward" if kind == "sample" else "off",
                              loop_start=0.4, loop_end=0.95, crossfade=0.3)
            if kind == "granular":
                o.gran = GranularCfg(density=40.0, length=0.08, rand_offset=0.05)
            if kind == "spectral":
                o.spec = SpectralCfg()
            return o
        for kind in ("sample", "granular", "spectral"):
            def f(q, kind=kind):
                lvl = q.oscs[0].level
                q.oscs[0] = make(kind, lvl)
                return q
            mv(f"osc A as {kind} oscillator (a note of the original)", f, {"osc"})
            if free is not None:
                def g(q, kind=kind):
                    q.oscs[free] = make(kind, 0.35)
                    return q
                mv(f"{kind} layer on osc {'ABC'[free]} (a note of the original)", g, {"osc"})

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
        if p.oscs[0].warp2 != "none":
            def drop_w2(q):
                q.oscs[0].warp2, q.oscs[0].warp2_amount = "none", 0.0
                return q
            mv(f"{p.oscs[0].warp2} warp 2", drop_w2)
        if p.filter2.enabled:
            def drop_f2(q):
                q.filter2.enabled = False
                for r in q.routing.values():
                    r["balance"] = 0.0
                return q
            mv("filter 2", drop_f2)
        for j, m in enumerate(p.fx_chain):
            def drop_fx(q, j=j):
                q.fx_chain.pop(j)
                q.mods = [x for x in q.mods if not x.dest.startswith("fx.")]
                return q
            mv(f"FX {m.get('type')}", drop_fx)
        for j, m in enumerate(p.mods):
            if (m.source, m.dest) in getattr(self, "protected", set()):
                continue
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
        per = max(30, int(min(90, self.pass_left() * 0.33 / max(1, len(cands)) / 0.03)))
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
        while self.pass_left() > (self.pass_end - self.t0) * 0.07 and score < 99.0:
            if added and not opts.keep_going and self.ref.waveform(self.render(p)) >= opts.target_score:
                break
            moves = self.structure_moves(p)
            self.set_stage("structure", f"screening {len(moves)} possible additions")
            # quick screen: one render per candidate so every move gets a look
            # however many there are, then a short tune of the top dozen
            quick = []
            for name, fn, focus in moves:
                if self.pass_left() < (self.pass_end - self.t0) * 0.06:
                    break
                self.activity = f"screening: {name}"
                q = fn(p.copy())
                quick.append((self.evaluate(q), name, q, focus))
            quick.sort(key=lambda r: -r[0])
            screened = []
            for _, name, q, focus in quick[:12]:
                if self.pass_left() < (self.pass_end - self.t0) * 0.06:
                    break
                self.activity = f"short tune: {name}"
                q, s = self.tune(q, 12, focus=focus, sigma=0.2)
                screened.append((s, name, q, focus))
            screened.sort(key=lambda r: -r[0])
            best = (score, None, None)
            for s0, name, q, focus in screened[:4]:
                if self.pass_left() < (self.pass_end - self.t0) * 0.06:
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
        while changed and self.pass_left() > (self.pass_end - self.t0) * 0.03:
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
        budget = int(max(40, self.pass_left() * 0.9 / 0.03))
        self.set_stage("fine-tune", f"all {len(P.continuous_params(p))} parameters together")
        self.reserve = max(0.0, self.time_left() - self.pass_left())   # the rest belongs to later stages
        p, score = self.tune(p, budget, sigma=0.08)
        self.report("fine-tune", f"{score:.1f}")
        self.stages.append({"stage": "fine-tune", "score": round(score, 2)})

        # --- 5b. exploration rounds (long searches) ----------------------------
        p, score = self._explore(p, score)

        # --- 6. waveform alignment --------------------------------------------
        self.reserve = 0.0
        self._in_final = True
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
        if self.stopped:
            self.stages.append({"stage": "stopped", "score": round(score, 2),
                                "note": "stopped by the user; best patch so far"})
        metrics = self.ref.all(self.render(p))
        total = metrics["waveform_similarity"]
        return Result(patch=p, score=total, details=metrics, notes=self.analysis.notes, bank=self.bank,
                      history=self.history, used_custom=True, stages=self.stages, samples=self.samples)

    def _explore(self, p: Patch, score: float) -> tuple[Patch, float]:
        """Extra time beyond the standard schedule: repeated rounds of structure
        search (every candidate gets a short tune), pruning and CMA-ES restarts
        from the best patch. Only improvements are kept."""
        opts = self.opts
        self.reserve = self.wave_reserve
        rounds = []
        r = 0
        while self.time_left() > self.wave_reserve + 20 and score < 99.5:
            if not opts.keep_going and self.ref.waveform(self.render(p)) >= opts.target_score:
                break
            r += 1
            start_score = score
            self.set_stage("explore", f"round {r}: more structure ideas and restarts")
            moves = self.structure_moves(p)
            screened = []
            for name, fn, focus in moves:
                if self.time_left() <= self.wave_reserve + 20:
                    break
                self.activity = f"round {r}: trying {name}"
                q, s = self.tune(fn(p.copy()), 24, focus=focus, sigma=0.2)
                screened.append((s, name, q, focus))
            screened.sort(key=lambda t: -t[0])
            for s0, name, q, focus in screened[:6]:
                if self.time_left() <= self.wave_reserve + 20:
                    break
                self.activity = f"round {r}: tuning {name}"
                q, s = self.tune(q, 120, focus=focus | {"filter", "osc", "env", "mod"}, sigma=0.1)
                if s > score + 0.15:
                    p, score = q, s
                    self.report("explore", f"round {r}: kept '{name}' -> {score:.1f}")
            for name, fn in self.prune_moves(p):
                q = fn(p.copy())
                s = self.evaluate(q)
                if s >= score - 0.1:
                    p, score = q, s
            # restart the full optimisation from the best patch (wider, then narrower)
            for sigma in (0.15, 0.05):
                if self.time_left() <= self.wave_reserve + 20:
                    break
                self.activity = f"round {r}: all parameters, step {sigma}"
                evals = int(min(self.time_left() - self.wave_reserve, 600) * 0.5 / 0.03)
                q, s = self.tune(p, max(40, evals), sigma=sigma)
                if s > score:
                    p, score = q, s
            rounds.append({"round": r, "score": round(score, 2), "gain": round(score - start_score, 2)})
            self.report("explore", f"round {r} done: {score:.1f}")
        if rounds:
            self.stages.append({"stage": "exploration", "score": round(score, 2), "rounds": rounds})
        self.reserve = 0.0
        return p, score

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
