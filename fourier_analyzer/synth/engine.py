"""Offline Serum 2-style renderer used inside the optimizer loop and for
previewing imported presets.

Signal path (Serum 2 manual pp. 28-218):

* voices: mono (legato, portamento with curve/always/scaled) or poly with a
  voice limit; notes pass the keyboard (transpose, key/scale quantize) and the
  arpeggiator first
* per voice: 3 oscillators in Wavetable / Sample / Granular / Spectral /
  Multisample mode (pitch modes, unison modes/stacks/width/spreads, phase
  memory, two warp slots with every warp mode, key/velocity ranges), sub and
  noise; mixer routing of every source to filter 1/2 (balance), main, direct
  or none, FX-bus sends and the ENV 1 switch; two filters (every filter type,
  keytrack, pan, drive/clean, mix, level) that route to each other, main,
  direct or none
* modulation: 4 envelopes (curves, start/end levels, legato), 10 LFOs
  (shapes, path, S&H, chaos, modes incl. envelope loopback, direction, rise,
  delay, smooth, point modulation), velocity / note curves, macros, random
  and alternating note sources, the voice control panel, and a 64-slot mod
  matrix (curves, aux sources, slew, output, bypass) whose destinations can
  be any parameter ("@<path>", moved in Serum's knob space)
* global: main FX rack, two FX buses with their own destinations, direct
  output, master tuning / volume, concert pitch or a tuning table

It renders mono for the optimizer (fast) or stereo for previews. It is an
approximation of Serum's DSP; real Serum renders are used to check it.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

from . import knobs
from .patch import Env, Patch, Note, sync_to_hz
from .wavetables import FRAME, TableBank

CTRL = 64              # control-rate block size (samples)
FM_INDEX = 1.5         # cycles of phase deviation at 100 % FM warp (calibratable)
MAX_CUTOFF_FRAC = 0.45  # of sample rate

try:                    # optional speed-up
    import numba as _nb
except Exception:       # pragma: no cover - numba is optional
    _nb = None


# --------------------------------------------------------------------------
# voice timeline


class Timeline:
    """Per-sample note bookkeeping for one voice (mono = all notes)."""

    def __init__(self, notes: list[Note], n: int, sr: int, glide: float = 0.0, legato: bool = True,
                 glide_curve: float = 0.5, glide_always: bool = True, glide_scaled: bool = False,
                 mono: bool = True):
        self.n, self.sr = n, sr
        self.notes = sorted(notes, key=lambda x: x.start)
        idx = np.full(n, -1, dtype=np.int32)
        pitch = np.zeros(n, dtype=np.float64)
        t_on = np.zeros(n, dtype=np.float64)
        gate = np.zeros(n, dtype=bool)
        vel = np.zeros(n, dtype=np.float64)
        self.bounds = []        # (start_i, gate_off_i, next_start_i)
        self.retrig = []        # envelopes/LFOs restart at this note
        self.note_pitch = []
        self.note_vel = []
        prev_pitch = None
        prev_off = -1
        for k, nt in enumerate(self.notes):
            s = int(round(nt.start * sr))
            e = int(round((nt.start + nt.duration) * sr))
            nxt = int(round(self.notes[k + 1].start * sr)) if k + 1 < len(self.notes) else n
            s, e, nxt = max(0, min(s, n)), max(0, min(e, n)), max(0, min(nxt, n))
            if nxt <= s:
                continue
            overlap = prev_off >= s          # previous key still held -> legato
            retrig = not (mono and legato and overlap)
            seg = slice(s, nxt)
            idx[seg] = k
            tt = np.arange(nxt - s) / sr
            t_on[seg] = tt
            gate[s:min(e, nxt)] = True
            vel[seg] = nt.velocity
            p = np.full(nxt - s, float(nt.pitch))
            if nt.bend:
                bt = np.array([b[0] for b in nt.bend], dtype=float)
                bv = np.array([b[1] for b in nt.bend], dtype=float)
                p += np.interp(tt, bt, bv)
            if glide > 0 and prev_pitch is not None and (glide_always or overlap):
                T = glide
                if glide_scaled:
                    T = glide * max(abs(prev_pitch - nt.pitch), 0.5) / 12.0
                if abs(glide_curve - 0.5) < 0.02:
                    g = np.exp(-tt / (T / 3.0))
                else:
                    u = np.clip(tt / max(T, 1e-4), 0, 1)
                    k_ = 2 ** ((0.5 - glide_curve) * 4)
                    g = 1 - u ** k_
                p += (prev_pitch - nt.pitch) * g
            pitch[seg] = p
            prev_pitch = p[-1] if len(p) else nt.pitch
            prev_off = max(e, prev_off) if overlap else e
            self.bounds.append((s, e, nxt))
            self.retrig.append(retrig or not self.retrig)
            self.note_pitch.append(float(nt.pitch))
            self.note_vel.append(float(nt.velocity))
        if self.bounds:
            first = self.bounds[0][0]
            pitch[:first] = pitch[first] if first < n else 60.0
        self.idx, self.pitch, self.t_on, self.gate, self.vel = idx, pitch, t_on, gate, vel
        self.active = idx >= 0
        # time since the last retriggering note (envelopes / retrig LFOs)
        self.t_trig = np.zeros(n)
        self.groups = []         # (start, gate_off, next_start) of retrigger groups
        for (s, e, nxt), r in zip(self.bounds, self.retrig):
            if r or not self.groups:
                self.groups.append([s, e, nxt])
            else:
                self.groups[-1][1] = max(self.groups[-1][1], e)
                self.groups[-1][2] = nxt
        for s, e, nxt in self.groups:
            self.t_trig[s:nxt] = np.arange(nxt - s) / sr


# --------------------------------------------------------------------------
# envelopes and LFOs


def render_env(env: Env, tl: Timeline, per_group: list | None = None, time_scale: float = 1.0,
               legato_ignore: bool = False) -> np.ndarray:
    """ADSR with Serum's curves. ``per_group`` overrides params per retrigger
    group (modulated envelope times / levels at note-on)."""
    out = np.zeros(tl.n)
    sr = tl.sr
    groups = [(s, e, nxt) for s, e, nxt in tl.bounds] if (env.legato_inverted or legato_ignore) else tl.groups
    level_prev = float(env.start)
    for gi, (st, off, nxt) in enumerate(groups):
        prm = per_group[gi] if per_group and gi < len(per_group) else {}
        ts = time_scale * prm.get("time_scale", 1.0)
        a = max(prm.get("attack", env.attack) * ts, 1e-4)
        h = max(prm.get("hold", env.hold) * ts, 0.0)
        d = max(prm.get("decay", env.decay) * ts, 1e-4)
        s = float(np.clip(prm.get("sustain", env.sustain), 0, 1))
        r = max(prm.get("release", env.release) * ts, 1e-4)
        end = float(np.clip(env.end, 0, 1))
        pa = 2 ** ((0.5 - prm.get("attack_curve", env.attack_curve)) * 4)
        kd = 2 ** ((prm.get("decay_curve", env.decay_curve) - 0.666) * 4)
        kr = 2 ** ((prm.get("release_curve", env.release_curve) - 0.666) * 4)
        n_on = off - st
        t = np.arange(nxt - st) / sr
        lv = np.empty_like(t)
        l0 = max(level_prev, float(env.start))
        att = t < a
        lv[att] = l0 + (1 - l0) * (t[att] / a) ** pa
        hd = (~att) & (t < a + h)
        lv[hd] = 1.0
        dec = t >= a + h
        x = (t[dec] - a - h) / d
        lv[dec] = s + (1 - s) * np.exp(-4.6 * kd * x)
        if n_on < len(t):
            rel_start = max(n_on, 0)
            lr = lv[rel_start - 1] if rel_start > 0 else level_prev
            tr = t[rel_start:] - t[rel_start]
            lv[rel_start:] = end + (lr - end) * np.exp(-4.6 * kr * tr / r)
        out[st:nxt] = lv
        level_prev = lv[-1] if len(lv) else 0.0
    return out


def _curve_interp(ph, points):
    """Piecewise curve through [x, y(, curve)] points; curve 0.5 = straight."""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    cs = np.array([p[2] if len(p) > 2 else 0.5 for p in points], dtype=float)
    if len(xs) < 2:
        return np.full(np.shape(ph), ys[0] if len(ys) else 0.0)
    idx = np.clip(np.searchsorted(xs, ph, side="right") - 1, 0, len(xs) - 2)
    x0, x1 = xs[idx], xs[idx + 1]
    t = np.where(x1 > x0, (ph - x0) / np.maximum(x1 - x0, 1e-9), 0.0)
    k = 2 ** ((cs[idx] - 0.5) * 6)
    t = np.clip(t, 0, 1) ** k
    return ys[idx] + (ys[idx + 1] - ys[idx]) * t


def lfo_shape(shape: str, ph: np.ndarray, points=None) -> np.ndarray:
    ph = np.mod(ph, 1.0)
    if shape == "sine":
        return 0.5 - 0.5 * np.cos(2 * np.pi * ph)          # starts at 0, unipolar
    if shape == "triangle":
        return 1.0 - np.abs(2 * ph - 1.0)
    if shape == "saw_up":
        return ph
    if shape == "saw_down":
        return 1.0 - ph
    if shape == "square":
        return (ph < 0.5).astype(float)
    if shape == "custom" and points and len(points) >= 2:
        return _curve_interp(ph, points)
    return 0.5 - 0.5 * np.cos(2 * np.pi * ph)


def _path_y(points, ph, closed=True):
    """Path LFO: position along a drawn 2-D path; outputs its y coordinate."""
    pts = np.array([[p[0], p[1]] for p in points], dtype=float)
    if len(pts) < 2:
        return np.full(np.shape(ph), pts[0, 1] if len(pts) else 0.5)
    if closed:
        pts = np.vstack([pts, pts[:1]])
    seg = np.sqrt(np.sum(np.diff(pts, axis=0) ** 2, axis=1))
    cum = np.concatenate([[0], np.cumsum(seg)]) / max(seg.sum(), 1e-9)
    return np.interp(np.mod(ph, 1.0), cum, pts[:, 1])


def _chaos(kind: str, steps: int, dt: float, seed: int) -> np.ndarray:
    """Lorenz / Rossler attractor x-coordinate, normalised to 0..1."""
    x, y, z = 0.1 + 0.01 * seed, 0.0, 0.0
    out = np.empty(max(steps, 2))
    for i in range(len(out)):
        for _ in range(4):
            if kind == "lorenz":
                dx, dy, dz = 10 * (y - x), x * (28 - z) - y, x * y - 8 / 3 * z
                h = dt * 0.01
            else:
                dx, dy, dz = -y - z, x + 0.2 * y, 0.2 + z * (x - 5.7)
                h = dt * 0.05
            x, y, z = x + h * dx, y + h * dy, z + h * dz
        out[i] = x
    lo, hi = out.min(), out.max()
    return (out - lo) / (hi - lo + 1e-12)


def _apply_point_mods(points, mods, values):
    pts = [list(p) + [0.5] * (3 - len(p)) for p in points]
    for pm in mods:
        i = int(pm.get("point", 0))
        if not 0 <= i < len(pts):
            continue
        v = values.get(int(pm.get("bus", 0)), 0.0)
        axis = pm.get("axis", "y")
        if axis == "x":
            lo = pts[i - 1][0] if i > 0 else 0.0
            hi = pts[i + 1][0] if i + 1 < len(pts) else 1.0
            pts[i][0] = float(np.clip(pts[i][0] + v * (hi - lo), lo, hi))
        elif axis == "y":
            pts[i][1] = float(np.clip(pts[i][1] + v, 0, 1))
        else:
            pts[i][2] = float(np.clip(pts[i][2] + v, 0, 1))
    return pts


def render_lfo(lfo, tl: Timeline, bpm: float, index: int = 0, rate=None, phase_mod=None,
               bus_values: dict | None = None, time_scale: float = 1.0) -> np.ndarray:
    base = sync_to_hz(lfo.sync, bpm) if lfo.sync else lfo.rate_hz
    rate = base if rate is None else rate
    rate = np.asarray(rate, dtype=float) / max(time_scale, 1e-3)
    n, sr = tl.n, tl.sr
    t_abs = np.arange(n) / sr
    free = lfo.mode == "free" or lfo.mono or lfo.anchored
    if free:
        ph = (np.cumsum(np.broadcast_to(rate, (n,))) / sr) if rate.ndim else t_abs * float(rate)
        t_rel = t_abs if not tl.bounds else np.maximum(t_abs - tl.bounds[0][0] / sr, 0)
    else:
        t_rel = tl.t_trig
        if rate.ndim:
            ph = np.zeros(n)
            for s, _, nxt in tl.groups:
                ph[s:nxt] = np.concatenate([[0.0], np.cumsum(rate[s:nxt - 1])]) / sr
        else:
            ph = t_rel * float(rate)
    ph = ph + lfo.phase + (0.0 if phase_mod is None else phase_mod)
    if lfo.mode == "env":
        if lfo.loopback is not None and 0 <= lfo.loopback < 1:
            x0 = float(lfo.loopback)
            over = ph >= 1.0
            ph = np.where(over, x0 + np.mod(ph - 1.0, max(1 - x0, 1e-3)), ph)
        else:
            ph = np.minimum(ph, 0.9999)
    if lfo.direction == "reverse":
        ph = -ph
    elif lfo.direction == "pingpong":
        k = np.mod(ph, 2.0)
        ph = np.where(k < 1, k, 2 - k)
    kind = getattr(lfo, "type", "shape")
    pts = lfo.points
    if kind == "random":
        cyc = np.floor(ph).astype(np.int64)
        if not free:
            cyc = cyc + 1000 * np.maximum(tl.idx, 0)
        v = ((cyc * 2654435761 + 97 * (index + 1)) % 1000003) / 1000003.0
    elif kind in ("lorenz", "rossler"):
        steps = int(n / 64) + 2
        traj = _chaos(kind, steps, float(np.mean(rate)) / (sr / 64), index)
        v = np.interp(t_abs * sr / 64, np.arange(steps), traj)
    elif kind == "path" and lfo.path:
        v = _path_y(lfo.path, ph, closed=not lfo.path_open)
    elif lfo.point_mods and bus_values and pts:
        v = np.empty(n)
        blk = 256
        for b0 in range(0, n, blk):
            vals = {k: (float(np.mean(a[b0:b0 + blk])) if np.ndim(a) else float(a)) for k, a in bus_values.items()}
            v[b0:b0 + blk] = lfo_shape("custom", ph[b0:b0 + blk], _apply_point_mods(pts, lfo.point_mods, vals))
    else:
        v = lfo_shape(lfo.shape, ph, pts)
    smooth = getattr(lfo, "smooth", 0.0)
    if smooth > 0:
        a = np.exp(-1.0 / (smooth * 0.25 * sr))
        v = signal.lfilter([1 - a], [1, -a], v, zi=[v[0] * a])[0]
    rise, delay = getattr(lfo, "rise", 0.0), getattr(lfo, "delay", 0.0)
    if rise > 0 or delay > 0:
        hold = v[0] if len(v) else 0.0                 # before/while rising the LFO sits at its start value
        g = np.clip((t_rel - delay) / rise, 0, 1) if rise > 0 else (t_rel >= delay).astype(float)
        v = hold + (v - hold) * g
    return v


# --------------------------------------------------------------------------
# legacy fixed FX (kept for patches made by earlier versions)


def distort(x, mode, drive, mix):
    from .dsp import shape
    legacy = {"soft": "SoftClip", "hard": "HardClip", "tube": "Tube", "sinefold": "SinFold",
              "linfold": "LinFold", "diode": "Diode1"}
    y = shape(x, legacy.get(mode, mode), drive)
    return (1 - mix) * x + mix * y


def ott(x, depth, amount, sr):
    """OTT slot = the multiband compressor module it is written to Serum as
    (same defaults, so the preset sounds like the render)."""
    from .dsp import comp_fx
    from .patch import fx
    return comp_fx(x, fx("comp", multiband=True, depth=depth, amount=amount), sr)


def shelf(x, freq, gain_db, sr, high=False):
    from .dsp import tv_filter
    if abs(gain_db) < 0.05:
        return x
    return tv_filter(x, "highshelf" if high else "lowshelf", freq, 0.707, sr, gain_db=gain_db)


def apply_filter(x, ftype, cutoff_arr, reso, drive, sr):
    from .dsp import apply_filter_type
    return apply_filter_type(x, ftype, cutoff_arr, reso, drive, 0.0, sr)


# --------------------------------------------------------------------------
# oscillators


def _mip_level(max_hz: float, sr: int) -> int:
    max_h = (sr / 2) / max(max_hz, 1.0)
    if max_h >= 1024:
        return 0
    return int(min(10, np.ceil(np.log2(1024 / max(max_h, 1)))))


def _wt_lookup(table: np.ndarray, phase: np.ndarray, pos: np.ndarray) -> np.ndarray:
    nf = table.shape[0]
    flat = table.ravel()
    x = np.mod(phase, 1.0) * FRAME
    i0 = x.astype(np.int64)
    fx = x - i0
    i1 = (i0 + 1) & (FRAME - 1)
    i0 &= FRAME - 1
    if nf == 1:
        return flat[i0] * (1 - fx) + flat[i1] * fx
    fp = np.clip(pos, 0.0, 1.0) * (nf - 1)
    j0 = np.minimum(fp.astype(np.int64), nf - 2)
    fj = fp - j0
    b0 = j0 * FRAME
    b1 = b0 + FRAME
    a = flat[b0 + i0] * (1 - fx) + flat[b0 + i1] * fx
    b = flat[b1 + i0] * (1 - fx) + flat[b1 + i1] * fx
    return a * (1 - fj) + b * fj


UNISON_STACKS = {"Octave1": [0, 12], "Octave2": [0, 12, 24], "Octave3": [0, 12, 24, 36],
                 "OctaveFifth1": [0, 7, 12], "OctaveFifth2": [0, 7, 12, 19, 24],
                 "OctaveFifth3": [0, 7, 12, 19, 24, 31, 36], "Center12": "c12", "Center24": "c24"}


def unison_offsets(n: int, detune: float, rng_semis: float = 2.0, stack: str = "none",
                   mode: str = "linear", seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Semitone offsets and relative gains for n unison voices. ``rng_semis``
    is Serum's unison range (spread at full detune); MODE shapes the spacing;
    stacks add octaves/fifths (Center-12/24 drop the centre voices)."""
    if n <= 1:
        return np.zeros(1), np.ones(1)
    spread = 0.5 * rng_semis * float(np.clip(detune, 0, 1)) ** 1.5    # semitones at the outermost voice
    x = np.linspace(-1, 1, n)
    m = (mode or "linear").lower().replace("kdetune", "")
    if m == "super":
        x = np.sign(x) * np.abs(x) ** 0.6
    elif m == "exp":
        x = np.sign(x) * (2 ** (3 * np.abs(x)) - 1) / 7
    elif m == "inv":
        x = np.where(x < 0, x * 1.5, x * 0.5)
    elif m == "random":
        r = np.random.default_rng(1234 + seed)
        x = np.sort(r.uniform(-1, 1, n))
    off = x * spread
    st = UNISON_STACKS.get(str(stack)[1:] if str(stack).startswith("k") else stack)
    if isinstance(st, list):
        off = off + np.array([st[i % len(st)] for i in range(n)], dtype=float)
    elif st in ("c12", "c24"):
        centre = np.abs(np.linspace(-1, 1, n)) < 0.34
        off = off - np.where(centre, 12.0 if st == "c12" else 24.0, 0.0)
    return off, np.ones(n)


PHASE_WARPS = ("Sync", "BendPos", "BendNeg", "BendPosNeg", "PWM", "ASYMPos", "ASYMNeg", "ASYMPosNeg",
               "Quantize", "Mirror", "Remap_1", "Remap_2", "Remap_3", "Remap_4")
LEGACY_WARPS = {"fm_b": "FM_OSC", "fm_sub": "FM_SUB", "am_b": "AM_OSC", "rm_b": "RM_OSC", "rm_sub": "RM_SUB"}


def warp_code(w: str) -> str:
    w = LEGACY_WARPS.get(w, w or "none")
    return w[1:] if w.startswith("k") and w[1:2].isupper() else w


def warp_source(code: str) -> str | None:
    """Which signal a cross-modulation warp listens to."""
    for suffix, src in (("_OSC2", "osc2"), ("_OSC", "osc"), ("_SUB", "sub"), ("_NOISE", "noise"),
                        ("_FILT1", "filt1"), ("_FILT2", "filt2")):
        if code.endswith(suffix):
            return src
    return None


def _warp_phase(frac, code, a, var=0.5, remap=None):
    """Phase-domain warps (frac in 0..1, amount a scalar or array)."""
    if code == "Sync":
        k = 1 + a * 7
        y = np.mod(frac * k, 1.0)
        if var < 0.98:                     # WARP Var: hard sync (1) .. soft sync (0)
            soft = 1 - var
            w = np.clip((1 - np.mod(frac * k, 1.0)) / max(soft * 0.5, 1e-3), 0, 1)
            y = y * w + frac * (1 - w)
        return y, None
    if code in ("BendPos",):
        return frac ** (1 + 4 * a), None
    if code in ("BendNeg",):
        return 1 - (1 - frac) ** (1 + 4 * a), None
    if code == "BendPosNeg":
        b = (np.asarray(a) - 0.5) * 2         # 50 % = no change
        e = 1 + 4 * np.abs(b)
        lo = frac < 0.5
        up = np.where(lo, 0.5 * (2 * frac) ** e, 1 - 0.5 * np.clip(2 - 2 * frac, 0, None) ** e)
        dn = np.where(lo, 0.5 - 0.5 * np.clip(1 - 2 * frac, 0, None) ** e, 0.5 + 0.5 * np.clip(2 * frac - 1, 0, None) ** e)
        return np.where(b >= 0, up, dn), None
    if code == "PWM":
        w = 1 - 0.98 * a
        return np.minimum(frac / w, 0.999999), frac > w
    if code.startswith("ASYM"):
        d = 0.5 - 0.49 * a if code != "ASYMNeg" else 0.5 + 0.49 * a
        if code == "ASYMPosNeg":
            d = 0.5 - 0.98 * (np.asarray(a) - 0.5)
        return np.where(frac < d, frac * 0.5 / d, 0.5 + (frac - d) * 0.5 / (1 - d)), None
    if code == "Quantize":
        nq = np.maximum(2.0, 256.0 * (1 - a) ** 2)
        return np.floor(frac * nq) / nq, None
    if code == "Mirror":
        m = np.where(frac < 0.5, 2 * frac, 2 - 2 * frac)
        return (1 - a) * frac + a * m * 0.5, None
    if code.startswith("Remap"):
        pts = remap or [[0.0, 0.0, 0.5], [0.5, 0.75, 0.5], [1.0, 1.0, 0.5]]
        if code == "Remap_1":
            r = _curve_interp(frac, pts)
        elif code == "Remap_2":                  # mirrored: graph on each half
            h = np.where(frac < 0.5, 2 * frac, 2 * frac - 1)
            r = np.where(frac < 0.5, 0.5 * _curve_interp(h, pts), 0.5 + 0.5 * _curve_interp(h, pts))
        elif code == "Remap_3":                  # sinusoidal
            r = np.mod(frac + np.sin(2 * np.pi * frac) / (2 * np.pi), 1.0)
        else:                                    # Remap 4: graph four times
            q = np.mod(frac * 4, 1.0)
            r = (np.floor(frac * 4) + _curve_interp(q, pts)) / 4
        return (1 - a) * frac + a * r, None
    return frac, None


OUTPUT_WARPS = ("AM_", "RM_", "Dist", "FilterLPF", "FilterHPF", "Mask_")


def _output_warps(out, warps, mod_sigs, sr):
    """Amplitude-domain warps: AM, RM, distortion, filter."""
    from .dsp import onepole_lp, shape
    for code, a, _var in warps:
        src = mod_sigs.get(warp_source(code))
        if src is not None and np.ndim(src) == 2:
            src = src.mean(axis=0)
        if code.startswith("AM_") and src is not None:
            out = out * ((1 - a) + a * (0.5 + 0.5 * src))
        elif code.startswith("RM_") and src is not None:
            out = out * ((1 - a) + a * src)
        elif code.startswith("Mask_") and src is not None:
            out = out * ((1 - a) + a * (np.abs(src) > 0.1))
        elif code.startswith("Dist"):
            out = shape(out, code, a)
        elif code == "FilterLPF":
            out = onepole_lp(out, 20 * 1000 ** (1 - np.asarray(a, dtype=float)), sr)
        elif code == "FilterHPF":
            out = out - onepole_lp(out, 20 * 1000 ** np.asarray(a, dtype=float), sr)
    return out


def _pan_lr(pan):
    pan = np.clip(pan, -1, 1)
    ang = (pan + 1) * np.pi / 4
    return np.cos(ang) * np.sqrt(2), np.sin(ang) * np.sqrt(2)


def _curve_amt(v, c, pts=None):
    """Mod matrix source curve: drawn points, or the CRV bend (-1..1)."""
    if pts:
        return _curve_interp(np.clip(v, 0, 1), pts)
    if c:
        return np.clip(v, 0, 1) ** (2 ** (-2 * c))
    return v


def _slew(v, rise, fall, sr):
    if (rise <= 0 and fall <= 0) or np.ndim(v) == 0:
        return v
    ar = np.exp(-1 / max(rise * sr, 1.0))
    af = np.exp(-1 / max(fall * sr, 1.0))
    if _nb is not None:
        return _slew_k(np.ascontiguousarray(v, dtype=np.float64), ar, af)
    out = np.empty_like(v)
    s = v[0]
    for i in range(len(v)):
        a = ar if v[i] > s else af
        s = a * s + (1 - a) * v[i]
        out[i] = s
    return out


if _nb is not None:
    @_nb.njit(cache=True, fastmath=True)
    def _slew_k(v, ar, af):  # pragma: no cover - compiled
        out = np.empty_like(v)
        s = v[0]
        for i in range(v.shape[0]):
            a = ar if v[i] > s else af
            s = a * s + (1 - a) * v[i]
            out[i] = s
        return out


SCALE_TABLE = {}


def quantize_pitch(p: float, key: int, allowed) -> float:
    if not allowed or not any(allowed):
        return p
    pc = int(round(p)) % 12
    best = None
    for d in (0, -1, 1, -2, 2, -3, 3, -4, 4, -5, 5, -6, 6):
        if allowed[(pc + d - key) % 12]:
            best = d
            break
    return p + (best or 0)


ARP_DIVS = {"1/1": 4.0, "1/2": 2.0, "1/4": 1.0, "1/8": 0.5, "1/16": 0.25, "1/32": 0.125, "1/64": 0.0625}


def arpeggiate(notes: list[Note], arp, bpm: float, seed: int = 0) -> list[Note]:
    """Serum's ARP as a note transformer: held notes become a rhythmic pattern."""
    if not arp.enabled or not notes:
        return notes
    rng = np.random.default_rng(seed)
    div = arp.rate
    mult = 1.0
    if div.endswith("T"):
        mult, div = 2 / 3, div[:-1]
    elif div.endswith("D"):
        mult, div = 1.5, div[:-1]
    step = ARP_DIVS.get(div, 0.25) * mult * 60.0 / bpm
    end = max(n.start + n.duration for n in notes)
    out = []
    t = min(n.start for n in notes)
    k = 0
    last = None
    seq_cache = {}
    while t < end - 1e-6:
        held = sorted([n for n in notes if n.start <= t + 1e-6 < n.start + n.duration], key=lambda n: n.pitch)
        if not held:
            nxt = [n.start for n in notes if n.start > t]
            if not nxt:
                break
            t = min(nxt)
            k = 0
            continue
        played = sorted(held, key=lambda n: n.start)
        key = tuple(n.pitch for n in held)
        if key not in seq_cache:
            ups = [n.pitch for n in held]
            base = {"Up": ups, "Down": ups[::-1], "UpDown": ups + ups[-2:0:-1], "DownUp": ups[::-1] + ups[1:-1],
                    "UpAndDown": ups + ups[::-1], "DownAndUp": ups[::-1] + ups, "Played": [n.pitch for n in played],
                    "Converge": [ups[i // 2] if i % 2 == 0 else ups[-1 - i // 2] for i in range(len(ups))],
                    "Diverge": [ups[(len(ups) - 1) // 2 + ((i + 1) // 2) * (1 if i % 2 else -1)]
                                if 0 <= (len(ups) - 1) // 2 + ((i + 1) // 2) * (1 if i % 2 else -1) < len(ups)
                                else ups[0] for i in range(len(ups))]}.get(arp.shape)
            if arp.shape == "ConvAndDiv":
                base = ups[::-1] + ups
            if base is None:
                base = ups
            seq = []
            for r in range(max(1, arp.range)):
                tr = r * arp.shift if arp.transpose_shape != "Down" else -r * arp.shift
                for _ in range(max(1, arp.repeats)):
                    seq += [p + tr for p in base]
            seq_cache[key] = seq
        seq = seq_cache[key]
        if arp.shape == "Chord":
            pitches = [n.pitch for n in held]
        elif arp.shape.startswith("Rand"):
            choice = float(rng.choice(seq))
            if arp.shape == "RandNoDup" and last is not None and len(seq) > 1:
                while choice == last:
                    choice = float(rng.choice(seq))
            pitches = [choice]
        else:
            pitches = [seq[(k + arp.offset) % len(seq)]]
        if rng.random() <= arp.chance:
            vel = max(n.velocity for n in held)
            for p in pitches:
                out.append(Note(t, step * max(arp.gate, 0.05), p, vel))
            last = pitches[-1]
        k += 1
        t += step
    return out


class Renderer:
    def __init__(self, bank: TableBank | None = None, sr: int = 22050, seed: int = 7, samples=None,
                 stereo: bool = False):
        self.bank = bank or TableBank()
        self.sr = sr
        self.seed = seed
        self._samples = samples
        self.stereo = stereo

    @property
    def samples(self):
        if self._samples is None:
            from .samples import SampleBank
            self._samples = SampleBank()
        return self._samples

    # ------------------------------------------------------------------
    def render(self, patch: Patch, notes: list[Note], duration: float,
               solo: str | None = None) -> np.ndarray:
        """Mono render (the mean of both channels when stereo). ``solo``
        limits output to one part ("oscs", "noise", "sub") for the ablation
        checks."""
        y = self._render(patch, notes, duration, solo, 2 if self.stereo else 1)
        return (y.mean(axis=0) if y.ndim == 2 else y).astype(np.float32)

    def render_stereo(self, patch: Patch, notes: list[Note], duration: float, solo=None) -> np.ndarray:
        return self._render(patch, notes, duration, solo, 2).astype(np.float32)

    # ------------------------------------------------------------------
    def prepare_notes(self, patch: Patch, notes: list[Note]) -> list[Note]:
        """Keyboard transpose, key/scale quantize, arpeggiator, same-note limit."""
        out = []
        for nt in notes:
            p = nt.pitch + patch.transpose
            if patch.scale_notes:
                p = quantize_pitch(p, patch.scale_key, patch.scale_notes)
            out.append(Note(nt.start, nt.duration, p, nt.velocity, nt.bend))
        out = arpeggiate(out, patch.arp, patch.bpm, self.seed)
        if patch.same_note_limit and not patch.mono:
            out.sort(key=lambda n: n.start)
            for i, a in enumerate(out):
                for b in out[i + 1:]:
                    if b.pitch == a.pitch and b.start < a.start + a.duration:
                        a.duration = max(0.0, b.start - a.start)
        return out

    def _voices(self, patch: Patch, notes: list[Note]) -> list[list[Note]]:
        if patch.mono:
            return [notes]
        notes = sorted(notes, key=lambda n: n.start)
        poly = max(1, int(patch.poly or 16))
        active: list[Note] = []
        for nt in notes:
            active = [a for a in active if a.start + a.duration > nt.start]
            if len(active) >= poly:                      # steal the oldest voice
                old = min(active, key=lambda a: a.start)
                old.duration = max(0.0, nt.start - old.start)
                active.remove(old)
            active.append(nt)
        return [[nt] for nt in notes]

    def _render(self, patch: Patch, notes: list[Note], duration: float, solo, C: int) -> np.ndarray:
        from .dsp import process_chain
        sr = self.sr
        n = int(round(duration * sr))
        if n <= 0 or not notes:
            return np.zeros((C, max(n, 0)))
        notes = self.prepare_notes(patch, [Note(**nt.to_dict()) if hasattr(nt, "to_dict") else nt for nt in notes])
        voices = self._voices(patch, notes)
        main, direct = np.zeros((C, n)), np.zeros((C, n))
        buses = [np.zeros((C, n)), np.zeros((C, n))]
        fx_vals, fx_t = {}, {}
        rng = np.random.default_rng(self.seed)
        for vi, vnotes in enumerate(voices):
            res = self._render_voice(patch, vnotes, n, solo, C, vi, rng)
            if res is None:
                continue
            m, d, b1, b2, fv, start = res
            main += m
            direct += d
            buses[0] += b1
            buses[1] += b2
            for k, v in fv.items():            # FX are global: the latest note's modulation wins
                if np.ndim(v) == 0:
                    v = np.full(n, float(v))
                if k not in fx_vals:
                    fx_vals[k] = v.copy()
                else:
                    fx_vals[k][start:] = v[start:]
        legacy_mods = {k: v for k, v in fx_vals.items() if not k.startswith(("fx_chain.", "fx_buses."))}
        values = {k: v for k, v in fx_vals.items() if k.startswith(("fx_chain.", "fx_buses."))}
        # FX buses (bus 1 may feed bus 2 and vice versa)
        bus_out = [None, None]
        order = [1, 0] if patch.bus_dest[1:2] == ["bus1"] else [0, 1]
        for b in order:
            if not np.any(buses[b]):
                continue
            chain = patch.fx_buses[b] if b < len(patch.fx_buses) else []
            y = process_chain(buses[b], chain, sr, legacy_mods, prefix=f"bus{b + 1}", values=values,
                              path=f"fx_buses.{b}", samples=self.samples)
            y = y * (patch.bus_levels[b] if b < len(patch.bus_levels) else 1.0)
            dest = patch.bus_dest[b] if b < len(patch.bus_dest) else "main"
            if dest == f"bus{2 - b}" and b == order[0]:
                buses[1 - b] += y
            elif dest == "direct":
                direct += y
            else:
                bus_out[b] = y
        for y in bus_out:
            if y is not None:
                main += y
        main = self._fx_legacy(patch, main)
        main = process_chain(main, patch.fx_chain, sr, legacy_mods, values=values, path="fx_chain",
                             samples=self.samples)
        out = main + direct * patch.direct_level
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return out * patch.master

    # ------------------------------------------------------------------
    def _sources(self, patch: Patch, tl: Timeline, vindex: int, rng, prm=None):
        """Modulation sources (0..1 signals) for one voice. ``prm`` holds
        modulated envelope / LFO / macro / bus parameters (second pass)."""
        prm = prm or {}
        need = set()
        for m in patch.mods:
            if m.amount and not m.bypass:
                need.add(m.source)
                if m.aux:
                    need.add(m.aux)
        srcs = {}
        vc = patch.voice
        step = vc.steps[vindex % len(vc.steps)] if vc.steps else {}
        env_scale = vc.env_scale * (1 + vc.rand_env * (2 * rng.random() - 1)) * (1 + step.get("env", 0.0))
        for i, e in enumerate(patch.envs):
            if i == 0 or f"env{i+1}" in need:
                srcs[f"env{i+1}"] = render_env(e, tl, prm.get(f"env{i}"), env_scale)
        for i, l in enumerate(patch.lfos):
            if f"lfo{i+1}" in need:
                p = prm.get(f"lfo{i}", {})
                srcs[f"lfo{i+1}"] = render_lfo(l, tl, patch.bpm, i, rate=p.get("rate"), phase_mod=p.get("phase"),
                                               bus_values=prm.get("buses"), time_scale=vc.lfo_scale)
        if "velocity" in need or "release_velocity" in need:
            v = tl.vel
            if patch.velo_curve:
                v = _curve_interp(np.clip(v, 0, 1), patch.velo_curve)
            srcs["velocity"] = v
            srcs["release_velocity"] = v
        if "note" in need:
            nv = np.clip(tl.pitch / 127.0, 0, 1)
            srcs["note"] = _curve_interp(nv, patch.note_curve) if patch.note_curve else nv
        if "modwheel" in need:
            srcs["modwheel"] = np.full(tl.n, float(patch.modwheel))
        for i, v in enumerate(patch.macros):
            if f"macro{i+1}" in need:
                srcs[f"macro{i+1}"] = prm.get(f"macro{i}", np.full(tl.n, float(v)))
        if "fixed" in need:
            srcs["fixed"] = np.ones(tl.n)
        if need & {"rand1", "rand2", "rand_discrete"}:
            r1, r2 = np.zeros(tl.n), np.zeros(tl.n)
            for k, (s, _, nxt) in enumerate(tl.bounds):
                r1[s:nxt], r2[s:nxt] = rng.random(), rng.random()
            srcs["rand1"], srcs["rand2"], srcs["rand_discrete"] = r1, r2, r1
        if need & {"alt1", "alt2"}:
            a1 = np.zeros(tl.n)
            for k, (s, _, nxt) in enumerate(tl.bounds):
                a1[s:nxt] = float((vindex + k) % 2)
            srcs["alt1"], srcs["alt2"] = a1, 1 - a1
        if "voice_index" in need:
            nsteps = max(1, len(vc.steps) or 8)
            srcs["voice_index"] = np.full(tl.n, (vindex % nsteps) / max(nsteps - 1, 1))
        for k in ("voice_mod1", "voice_mod2"):
            if k in need:
                srcs[k] = np.full(tl.n, float(step.get(k.replace("voice_", ""), 0.0)))
        for k in ("pitchbend", "mpe_x", "mpe_y"):
            if k in need:
                srcs[k] = np.full(tl.n, 0.5)          # centred controller
        for k in ("aftertouch", "polyat", "mpe_z", "active_voices"):
            if k in need:
                srcs[k] = np.full(tl.n, 1.0 / max(patch.poly, 1) if k == "active_voices" else 0.0)
        return srcs, need

    def _matrix(self, patch: Patch, srcs: dict, tl: Timeline) -> dict:
        dest = {}
        for m in patch.mods:
            if not m.amount or m.bypass or m.source not in srcs:
                continue
            v = _curve_amt(srcs[m.source], m.curve, m.curve_points)
            if m.bipolar:            # centred on the knob: spans -amount/2 .. +amount/2
                v = v - 0.5
            if m.aux and m.aux in srcs:
                a = _curve_amt(srcs[m.aux], m.aux_curve, m.aux_curve_points)
                v = v * ((1 - a) if m.aux_invert else a)
            v = _slew(np.asarray(v, dtype=float), m.smooth_rise, m.smooth_fall, tl.sr)
            v = v * m.output
            d = "filter.cutoff" if m.dest == "filter1.cutoff" else m.dest
            dest[d] = dest.get(d, 0) + m.amount * v
        return dest

    def _second_pass_params(self, patch: Patch, dm: dict, tl: Timeline) -> dict | None:
        """Envelope / LFO / macro / LFO-bus parameters moved by the matrix."""
        keys = [k for k in dm if k.startswith(("@envs.", "@lfos.", "@macros.", "@lfo_buses."))]
        if not keys:
            return None
        prm = {}
        for k in keys:
            path = k[1:]
            parts = path.split(".")
            kind, idx, field_ = parts[0], int(parts[1]), (parts[2] if len(parts) > 2 else None)
            if kind == "envs" and field_:
                base = getattr(patch.envs[idx], field_)
                vals = knobs.modulated(path, base, dm[k])
                groups = []
                for s, _, _ in tl.groups:
                    groups.append({field_: float(vals[s]) if np.ndim(vals) else float(vals)})
                prev = prm.setdefault(f"env{idx}", [{} for _ in tl.groups])
                for g0, g1 in zip(prev, groups):
                    g0.update(g1)
            elif kind == "lfos" and field_:
                if field_ == "rate_hz":
                    l = patch.lfos[idx]
                    base = sync_to_hz(l.sync, patch.bpm) if l.sync else l.rate_hz
                    prm.setdefault(f"lfo{idx}", {})["rate"] = knobs.modulated(path, base, dm[k])
                elif field_ == "phase":
                    prm.setdefault(f"lfo{idx}", {})["phase"] = dm[k]
            elif kind == "macros":
                prm[f"macro{idx}"] = np.clip(patch.macros[idx] + dm[k], 0, 1)
            elif kind == "lfo_buses":
                base = patch.lfo_buses[idx] if idx < len(patch.lfo_buses) else 0.0
                prm.setdefault("buses", {})[idx] = base + dm[k]
        return prm

    # ------------------------------------------------------------------
    def _render_voice(self, patch: Patch, notes, n, solo, C, vindex, rng):
        sr = self.sr
        tl = Timeline(notes, n, sr, patch.glide if patch.mono else 0.0, legato=patch.legato,
                      glide_curve=patch.glide_curve, glide_always=patch.glide_always,
                      glide_scaled=patch.glide_scaled, mono=patch.mono)
        if not tl.bounds:
            return None
        srcs, need = self._sources(patch, tl, vindex, rng)
        if need & {"osc1", "osc2", "osc3", "noise", "sub", "filter1", "filter2"}:
            self._audio_sources(patch, tl, srcs, need, rng)
        dm = self._matrix(patch, srcs, tl)
        prm = self._second_pass_params(patch, dm, tl)
        if prm:
            srcs2, _ = self._sources(patch, tl, vindex, np.random.default_rng(self.seed + vindex), prm)
            for k in srcs:
                srcs2.setdefault(k, srcs[k])
            srcs = srcs2
            dm = self._matrix(patch, srcs, tl)

        def V(path, base):
            off = dm.get("@" + path)
            return base if off is None else knobs.modulated(path, base, off)

        vc = patch.voice
        step = vc.steps[vindex % len(vc.steps)] if vc.steps else {}
        detune_c = vc.rand_detune * (2 * rng.random() - 1) + step.get("detune", 0.0)
        voice_pan = vc.rand_pan * (2 * rng.random() - 1) + step.get("pan", 0.0)
        voice_cut = 2 ** (4 * vc.rand_cutoff * (2 * rng.random() - 1) + step.get("cutoff", 0.0))
        pitch = tl.pitch + 48.0 * dm.get("pitch", 0.0) + V("master_tune", patch.master_tune) + detune_c / 100
        if "@transpose" in dm:
            pitch = pitch + (V("transpose", float(patch.transpose)) - patch.transpose)
        sources = {}
        noise_sig = None
        if patch.noise.enabled:
            noise_sig = self._noise(patch, tl, pitch, V, C, rng)
        sub_sig = self._sub(patch, tl, pitch, V, C) if patch.sub.enabled else None
        plain = {}
        if solo in (None, "oscs"):
            needs = set()
            for o in patch.oscs:
                if o.enabled:
                    for w, amt in ((o.warp, o.warp_amount), (o.warp2, o.warp2_amount)):
                        src = warp_source(warp_code(w)) if w != "none" else None
                        if src:
                            needs.add(src)
            if needs & {"osc", "osc2", "filt1", "filt2"}:
                for i, o in enumerate(patch.oscs):
                    if o.enabled:
                        plain[i] = self._osc_any(patch, i, o, pitch, tl, dm, V, rng, 1, plain_only=True)[0]
            filt_mod = {}
            if needs & {"filt1", "filt2"}:
                mix = sum(plain.values()) if plain else np.zeros(n)
                for key, f in (("filt1", patch.filter), ("filt2", patch.filter2)):
                    if key in needs and f.enabled:
                        filt_mod[key] = self._filter(f, "filter" if key == "filt1" else "filter2", mix[None, :],
                                                     dm, tl, V)[0]
            for i, o in patch.active_oscs():
                others = [j for j in (0, 1, 2) if j != i]
                mod_sigs = {"osc": plain.get(others[0]), "osc2": plain.get(others[1]),
                            "sub": sub_sig.mean(axis=0) if sub_sig is not None else None,
                            "noise": noise_sig.mean(axis=0) if noise_sig is not None else rng.standard_normal(n) * 0.35}
                mod_sigs.update(filt_mod)
                lvl = np.clip(V(f"oscs.{i}.level", o.level) + dm.get(f"osc{i+1}.level", 0.0), 0, 1.5)
                if np.all(np.asarray(lvl) == 0) and not o.key_warp:
                    continue
                sig = self._osc_any(patch, i, o, pitch, tl, dm, V, rng, C, mod_sigs=mod_sigs)
                pan = np.clip(V(f"oscs.{i}.pan", o.pan) + dm.get(f"osc{i+1}.pan", 0.0) + voice_pan, -1, 1)
                sources[f"osc{i+1}"] = self._panned(sig * lvl, pan, C)
            if sub_sig is not None:
                lvl = np.clip(V("sub.level", patch.sub.level) + dm.get("sub.level", 0.0), 0, 1.5)
                pan = np.clip(V("sub.pan", patch.sub.pan) + voice_pan, -1, 1)
                sources["sub"] = self._panned(sub_sig * lvl, pan, C)
        if noise_sig is not None and solo in (None, "noise"):
            nl = np.clip(V("noise.level", patch.noise.level) + dm.get("noise.level", 0.0), 0, 1.5)
            pan = np.clip(V("noise.pan", patch.noise.pan) + voice_pan, -1, 1)
            sources["noise"] = self._panned(noise_sig * nl, pan, C)
        if solo == "sub":
            sources = {k: v for k, v in sources.items() if k == "sub"}

        # ---- mixer: sources -> filters / main / direct / buses ----------------
        Z = lambda: np.zeros((C, n))
        f_in = {1: Z(), 2: Z()}
        main_pre, main_post, direct_pre, direct_post = Z(), Z(), Z(), Z()
        b_pre, b_post = [Z(), Z()], [Z(), Z()]
        f1_on, f2_on = patch.filter.enabled, patch.filter2.enabled

        def send(sig, r, rname, after_env):
            dest = r["dest"]
            if dest in ("main", "master"):
                (main_post if after_env else main_pre).__iadd__(sig)
            elif dest == "direct":
                (direct_post if after_env else direct_pre).__iadd__(sig)
            for b in (1, 2):
                lv = V(f"routing.{rname}.send{b}", float(r.get(f"send{b}", 0.0) or 0.0))
                if np.any(np.asarray(lv) != 0):
                    (b_post[b - 1] if after_env else b_pre[b - 1]).__iadd__(sig * lv)

        for name, sig in sources.items():
            r = patch.route(name)
            env_on = r.get("env1", True)
            if r["dest"] == "filter" and (f1_on or f2_on):
                bal = V(f"routing.{name}.balance", float(np.clip(r.get("balance", 0.0), 0, 1)))
                if not (f1_on and f2_on):
                    bal = 0.0 if f1_on else 1.0
                f_in[1] += (1 - bal) * sig
                f_in[2] += bal * sig
                rr = dict(r, dest="none")
            elif r["dest"] == "filter":
                rr = dict(r, dest="main")      # filters off: Serum passes the signal on
            else:
                rr = r
            send(sig, rr, name, not env_on)
        # filters, in an order that lets one feed the other
        r1, r2 = patch.route("filter1"), patch.route("filter2")
        order = [2, 1] if (r2["dest"] == "filter" and r1["dest"] != "filter") else [1, 2]
        for fi in order:
            f = patch.filter if fi == 1 else patch.filter2
            rf = r1 if fi == 1 else r2
            if not f.enabled:
                if np.any(f_in[fi]):
                    main_pre += f_in[fi]
                continue
            if not np.any(f_in[fi]):
                continue
            y = self._filter(f, "filter" if fi == 1 else "filter2", f_in[fi], dm, tl, V, voice_cut)
            y = y * V(f"{'filter' if fi == 1 else 'filter2'}.level", f.level)
            if rf["dest"] == "filter" and fi == order[0]:
                f_in[3 - fi] += y
                rr = dict(rf, dest="none")
            else:
                rr = rf if rf["dest"] != "filter" else dict(rf, dest="main")
            send(y, rr, f"filter{fi}", not rf.get("env1", True))
        vamp = np.clip(2.0 * (patch.voice_amp + dm.get("voiceamp", 0.0)), 0, 2)
        amp = srcs["env1"] * np.clip(1.0 + dm.get("amp", 0.0), 0, 4) * vamp
        main = main_pre * amp + main_post
        direct = direct_pre * amp + direct_post
        b1 = b_pre[0] * amp + b_post[0]
        b2 = b_pre[1] * amp + b_post[1]
        fx_vals = {}
        for k, v in dm.items():
            if k.startswith(("fx.", "bus1.", "bus2.")):
                fx_vals[k] = v
            elif k.startswith(("@fx_chain.", "@fx_buses.")):
                path = k[1:]
                base = _get_path(patch, path)
                if isinstance(base, (int, float)) and not isinstance(base, bool):
                    fx_vals[path] = knobs.modulated(path, float(base), v)
        if "dist.drive" in dm:
            fx_vals["legacy.drive"] = dm["dist.drive"]
        return main, direct, b1, b2, fx_vals, tl.bounds[0][0]

    def _audio_sources(self, patch, tl, srcs, need, rng):
        """Oscillators / filters used as modulation sources (unmodulated renders)."""
        V = lambda path, base: base
        dm = {}
        plain = {}
        for i, o in enumerate(patch.oscs):
            if o.enabled and (f"osc{i+1}" in need or need & {"filter1", "filter2"}):
                plain[i] = self._osc_any(patch, i, o, tl.pitch, tl, dm, V, rng, 1, plain_only=True)[0]
                srcs[f"osc{i+1}"] = 0.5 + 0.5 * np.clip(plain[i], -1, 1)
        if "noise" in need and patch.noise.enabled:
            srcs["noise"] = 0.5 + 0.5 * np.clip(self._noise(patch, tl, tl.pitch, V, 1, rng)[0], -1, 1)
        if "sub" in need and patch.sub.enabled:
            srcs["sub"] = 0.5 + 0.5 * self._sub(patch, tl, tl.pitch, V, 1)[0]
        mix = sum(plain.values()) if plain else np.zeros(tl.n)
        for key, f in (("filter1", patch.filter), ("filter2", patch.filter2)):
            if key in need and f.enabled:
                y = self._filter(f, "filter" if key == "filter1" else "filter2", mix[None, :], dm, tl, V)[0]
                srcs[key] = 0.5 + 0.5 * np.clip(y, -1, 1)

    @staticmethod
    def _panned(sig, pan, C):
        sig = np.atleast_2d(sig)
        if C == 1:
            return sig.mean(axis=0, keepdims=True) if sig.shape[0] > 1 else sig
        if sig.shape[0] == 1:
            sig = np.vstack([sig, sig])
        if np.ndim(pan) == 0 and float(pan) == 0.0:
            return sig
        gl, gr = _pan_lr(pan)
        return np.vstack([sig[0] * gl, sig[1] * gr])

    def _noise(self, patch, tl, pitch, V, C, rng):
        nz = patch.noise
        n = tl.n
        if nz.kind == "sample" and nz.sample:
            smp = self.samples.get(nz.sample)
            if smp is not None:
                data = np.atleast_2d(smp.data).astype(float)
                if data.shape[0] == 1:
                    data = np.vstack([data, data])
                N = data.shape[1]
                out = np.zeros((2, n))
                semis = V("noise.pitch", nz.pitch) + nz.fine / 100
                for k, (s, _, nxt) in enumerate(tl.bounds):
                    kt = (tl.pitch[s:nxt] - 60) if nz.keytrack else 0.0
                    rate = 2 ** ((semis + kt) / 12) * smp.sr / self.sr
                    start = (nz.start + nz.random_start * rng.random()) * N
                    pos = start + np.cumsum(np.broadcast_to(rate, (nxt - s,))) - np.broadcast_to(rate, (nxt - s,))
                    if nz.one_shot:
                        ok = pos < N - 1
                    else:
                        pos = np.mod(pos, N - 1)
                        ok = np.ones_like(pos, dtype=bool)
                    i0 = np.clip(pos.astype(np.int64), 0, N - 2)
                    fr = pos - i0
                    seg = data[:, i0] * (1 - fr) + data[:, i0 + 1] * fr
                    out[:, s:nxt] = np.where(ok, seg, 0.0)
                return out if C == 2 else out.mean(axis=0, keepdims=True)
        chans = []
        for c in range(2 if C == 2 else 1):
            w = rng.standard_normal(n)
            kind = nz.kind if nz.kind in ("White", "Pink", "Brown", "Geiger") else "White"
            if kind == "Pink":
                w = signal.lfilter([0.049922035, -0.095993537, 0.050612699, -0.004408786],
                                   [1, -2.494956002, 2.017265875, -0.522189400], w) * 3
            elif kind == "Brown":
                w = signal.lfilter([1.0], [1, -0.995], w) * 0.08
            elif kind == "Geiger":
                w = (rng.random(n) < 0.002).astype(float) * np.sign(rng.standard_normal(n)) * 3
            chans.append(w)
        if len(chans) == 2:
            st = float(np.clip(nz.stereo, 0, 1))
            chans[1] = (1 - st) * chans[0] + st * chans[1]
        out = []
        c = nz.color
        for w in chans:
            if c < 0.5:     # darker: one-pole low-pass
                fc = 200 * 2 ** (c * 2 * 6.6)
                a = np.exp(-2 * np.pi * fc / self.sr)
                w = signal.lfilter([1 - a], [1, -a], w) * (1 + (0.5 - c) * 4)
            elif c > 0.5:   # brighter: first-difference tilt
                k = (c - 0.5) * 2
                w = (1 - k) * w + k * np.diff(w, prepend=0.0)
            out.append(0.35 * w)
        return np.vstack(out)

    def _filter(self, f, key, x, dm, tl, V, voice_cut=1.0):
        from .dsp import apply_filter_type
        cut_oct = 10.0 * dm.get(f"{key}.cutoff", 0.0) + f.keytrack * (tl.pitch - 60) / 12
        cutoff = f.cutoff * voice_cut * 2 ** np.asarray(cut_oct, dtype=float)
        cutoff = V(f"{key}.cutoff", cutoff)
        reso = float(np.clip(np.mean(V(f"{key}.reso", f.reso)) + np.mean(dm.get(f"{key}.reso", 0.0)), 0, 0.98))
        drive = float(np.clip(np.mean(V(f"{key}.drive", f.drive)) + np.mean(dm.get(f"{key}.drive", 0.0)), 0, 1))
        var = np.clip(V(f"{key}.var", f.var) + dm.get(f"{key}.var", 0.0), 0, 1)
        mix = np.clip(V(f"{key}.mix", f.mix), 0, 1)
        pan = float(np.mean(V(f"{key}.pan", f.pan)))
        out = []
        for c in range(x.shape[0]):
            ck = cutoff
            if x.shape[0] > 1 and pan:
                ck = cutoff * 2 ** (pan * (-1 if c == 0 else 1))
            ck = np.broadcast_to(np.asarray(ck, dtype=float), x[c].shape)
            out.append(apply_filter_type(x[c], f.type, ck, reso, drive, var, self.sr, mix=mix, pz=f.pz,
                                         clean=f.clean))
        return np.vstack(out)

    def _sub(self, patch, tl, pitch, V, C):
        s = patch.sub
        tid = {"sine": "basic:sine", "triangle": "basic:triangle", "saw": "basic:saw",
               "square": "basic:square", "pulse": "basic:pwm", "roundrect": "basic:square"}.get(s.shape, "basic:sine")
        p = pitch if s.pitch_track else np.full_like(pitch, 0.0)
        f = patch.tuning_a4 * 2 ** ((p + 12 * V("sub.octave", s.octave) + V("sub.coarse", s.coarse) - 69) / 12)
        tab = self.bank.mip(tid, _mip_level(float(np.max(f)), self.sr))
        pos = np.full(1, 0.7 if s.shape == "pulse" else 0.0)
        ph = np.cumsum(np.broadcast_to(f, pitch.shape) / self.sr)
        if not s.contiguous:
            offset = np.zeros(len(ph))
            for k, (s0, _, nxt) in enumerate(tl.bounds):
                base_ph = ph[s0 - 1] if s0 > 0 else 0.0
                offset[s0:nxt] = -base_ph + s.phase
            ph = ph + offset
        y = _wt_lookup(tab, ph, pos)
        if s.shape == "roundrect":
            y = np.tanh(2.5 * y) / np.tanh(2.5)
        return y[None, :]

    # ------------------------------------------------------------------
    def _osc_freq(self, patch, i, o, pitch, V, dm, tl):
        """Frequency (Hz per sample) of oscillator i: pitch modes, coarse,
        Hz offset, key-range folding, tuning."""
        if not o.pitch_track:
            note = np.full_like(pitch, 0.0 if o.type == "wavetable" else 60.0)
            note = note + (pitch - tl.pitch)           # keep pitch mods / master tuning
        else:
            note = pitch.copy()
        if o.key_fold and o.key_range != (0, 127):
            lo, hi = o.key_range
            while np.any(note < lo):
                note = np.where(note < lo, note + 12, note)
            while np.any(note > hi):
                note = np.where(note > hi, note - 12, note)
        fine = V(f"oscs.{i}.fine", o.fine) + 100.0 * np.asarray(dm.get(f"osc{i+1}.fine", 0.0))
        semis = (12 * V(f"oscs.{i}.octave", float(o.octave)) + V(f"oscs.{i}.semi", float(o.semi))
                 + V(f"oscs.{i}.coarse", o.coarse) + fine / 100.0)
        if patch.tuning_table and len(patch.tuning_table) >= 128:
            tab = np.asarray(patch.tuning_table, dtype=float)
            base = np.interp(note, np.arange(128), np.log2(np.maximum(tab, 1e-3)))
            f = 2 ** base * 2 ** (semis / 12)
        else:
            f = patch.tuning_a4 * 2 ** ((note + semis - 69) / 12)
        if o.pitch_mode in ("harmonics", "ratio"):
            ratio = V(f"oscs.{i}.ratio", o.ratio)
            if o.pitch_mode == "harmonics":
                ratio = np.maximum(1, np.round(ratio))
            ref = f
            if o.ratio_source >= 0 and o.ratio_source != i and o.ratio_source < len(patch.oscs):
                ref = self._osc_freq(patch, o.ratio_source, patch.oscs[o.ratio_source], pitch, V, dm, tl)
            f = ref * np.maximum(ratio, 1e-3)
        hz = V(f"oscs.{i}.hz_offset", o.hz_offset)
        if np.any(np.asarray(hz) != 0):
            f = np.maximum(f + hz, 0.0)
        return np.broadcast_to(f, pitch.shape).astype(float)

    def _key_mask(self, o, tl):
        if tuple(o.key_range) == (0, 127) and tuple(o.vel_range) == (0, 127):
            return None
        mask = np.zeros(tl.n)
        lo, hi = o.key_range
        vlo, vhi = o.vel_range
        for (s, _, nxt), p, v in zip(tl.bounds, tl.note_pitch, tl.note_vel):
            ok = (o.key_fold or lo <= p <= hi) and vlo <= v * 127 <= vhi
            mask[s:nxt] = 1.0 if ok else 0.0
        return mask

    def _unison(self, i, o, V):
        n_u = max(1, int(o.unison))
        offs, gains = unison_offsets(n_u, float(np.mean(V(f"oscs.{i}.detune", o.detune))), o.unison_range,
                                     o.unison_stack, o.unison_mode, seed=i)
        if n_u > 1:
            blend = float(np.mean(V(f"oscs.{i}.blend", o.blend)))
            gains = np.where(np.abs(np.linspace(-1, 1, n_u)) < 0.34, 1.0, blend)
            width = float(np.mean(V(f"oscs.{i}.unison_width", o.unison_width)))
            pans = np.linspace(-1, 1, n_u) * width
        else:
            pans = np.zeros(1)
        return offs, gains, pans

    def _osc_any(self, patch, i, o, pitch, tl, dm, V, rng, C, mod_sigs=None, plain_only=False):
        f = self._osc_freq(patch, i, o, pitch, V, dm, tl)
        mask = self._key_mask(o, tl)
        if o.type == "wavetable" or not o.type:
            y = self._osc_wt(patch, i, o, f, tl, dm, V, rng, C, mod_sigs or {}, plain_only)
        else:
            y = self._osc_sample(patch, i, o, f, tl, dm, V, rng, C, mod_sigs or {}, plain_only)
        if mask is not None:
            y = y * mask
        return y

    def _warps(self, i, o, V, dm, plain_only):
        if plain_only:
            return []
        w = [(warp_code(o.warp), np.clip(V(f"oscs.{i}.warp_amount", o.warp_amount) + dm.get(f"osc{i+1}.warp", 0.0), 0, 1),
              float(np.mean(V(f"oscs.{i}.warp_var", o.warp_var)))),
             (warp_code(o.warp2), np.clip(V(f"oscs.{i}.warp2_amount", o.warp2_amount) + dm.get(f"osc{i+1}.warp2", 0.0), 0, 1),
              float(np.mean(V(f"oscs.{i}.warp2_var", o.warp2_var))))]
        return [(c, a, v) for c, a, v in w if c != "none" and np.any(np.asarray(a) > 0)]

    def _osc_sample(self, patch, i, o, f, tl, dm, V, rng, C, mod_sigs, plain_only):
        from . import sampler
        warps = self._warps(i, o, V, dm, plain_only)
        for code, a, _v in warps:              # exponential / linear FM bend the playback rate
            src = mod_sigs.get(warp_source(code))
            if src is not None and code.startswith(("FM", "FMX", "FMP", "PD")):
                f = f * 2 ** (np.asarray(a) * 2.0 * src)
        offs, gains, pans = self._unison(i, o, V)
        uni = list(zip(offs, gains, pans))
        cfg = o.smp
        smp = self.samples.get(cfg.source) if o.type != "multisample" else None
        if o.type == "sample":
            y = sampler.render_sample(smp, cfg, f, tl.bounds, tl.gate, self.sr, rng, uni, patch.tuning_a4)
        elif o.type == "granular":
            y = sampler.render_granular(smp, cfg, o.gran, f, tl.bounds, tl.gate, self.sr, rng, patch.bpm, uni,
                                        patch.tuning_a4)
        elif o.type == "spectral":
            y = sampler.render_spectral(smp, cfg, o.spec, f, tl.bounds, tl.gate, self.sr, rng, warps, mod_sigs,
                                        uni, patch.tuning_a4)
        elif o.type == "multisample":
            y = sampler.render_multisample(self.samples, cfg, o.multi, f, tl.bounds, tl.vel, self.sr, rng,
                                           tl.note_pitch, patch.tuning_a4, uni)
        else:
            y = np.zeros((2, tl.n))
        out_w = [w for w in warps if w[0].startswith(OUTPUT_WARPS)]
        if out_w:
            y = np.vstack([_output_warps(c, out_w, mod_sigs, self.sr) for c in y])
        return y if C == 2 else y.mean(axis=0, keepdims=True)

    def _osc_wt(self, patch, i, o, f_base, tl, dm, V, rng, C, mod_sigs, plain_only):
        sr = self.sr
        offs, gains, pans = self._unison(i, o, V)
        f_max = float(np.max(f_base)) * 2 ** (offs.max() / 12)
        tab = self.bank.mip(o.table, _mip_level(f_max, sr))
        posmod = dm.get(f"osc{i+1}.pos", 0.0)
        pos0 = V(f"oscs.{i}.position", o.position)
        pos = np.clip(pos0 + posmod, 0, 1)
        pos = pos if np.ndim(pos) else np.array([float(pos)])
        if o.smooth_interp and pos.size > 1:
            pos = signal.lfilter([0.02], [1, -0.98], pos, zi=[pos[0] * 0.98])[0]
        warps = self._warps(i, o, V, dm, plain_only)
        spread_pos = float(np.mean(V(f"oscs.{i}.unison_wtpos", o.unison_wtpos)))
        spread_w = (float(np.mean(V(f"oscs.{i}.unison_warp", o.unison_warp))),
                    float(np.mean(V(f"oscs.{i}.unison_warp2", o.unison_warp2))))
        n_u = len(offs)
        starts = [b[0] for b in tl.bounds]
        out = np.zeros((2 if C == 2 else 1, tl.n))
        rnd = float(np.mean(V(f"oscs.{i}.random_phase", o.random_phase)))
        uv = np.linspace(-1, 1, n_u) if n_u > 1 else np.zeros(1)
        for k_u, (off, g, pan) in enumerate(zip(offs, gains, pans)):
            f = f_base * 2 ** (off / 12)
            vw = []
            for wi, (code, a, var) in enumerate(warps):
                sp = spread_w[min(wi, 1)]
                vw.append((code, np.clip(a + 0.5 * sp * uv[k_u], 0, 1) if sp else a, var))
            for code, a, var in vw:          # exponential / linear FM change the frequency itself
                src = mod_sigs.get(warp_source(code))
                if src is None:
                    continue
                if code.startswith("FMX"):
                    f = f * 2 ** (a * 2.0 * src)
                elif code.startswith("FMP"):
                    f = np.maximum(f * (1 + a * 4.0 * src), 0.0)
            ph = np.cumsum(f / sr)
            if o.phase_memory != "contiguous":
                offset = np.zeros(len(ph))
                for k, s0 in enumerate(starts):
                    s1 = starts[k + 1] if k + 1 < len(starts) else len(ph)
                    if not tl.retrig[k] and k > 0 and o.phase_memory == "all":
                        offset[s0:s1] = offset[s0 - 1] if s0 > 0 else 0.0
                        continue
                    base_ph = ph[s0 - 1] if s0 > 0 else 0.0
                    r = 0.0 if o.phase_memory == "per_voice" else rnd * rng.random()
                    offset[s0:s1] = -base_ph + o.phase + r
                ph = ph + offset
            mask = None
            for code, a, var in vw:
                src = mod_sigs.get(warp_source(code))
                if code.startswith("FM_") and src is not None:
                    ph = ph + a * FM_INDEX * src           # Serum FM = phase modulation (thru-zero)
                elif code.startswith("PD_") and src is not None:
                    ph = ph + a * 0.5 * (0.5 + 0.5 * src)  # unipolar phase distortion
            frac = np.mod(ph, 1.0)
            for code, a, var in vw:
                if code in PHASE_WARPS:
                    frac, m = _warp_phase(frac, code, a, var, o.remap_curve)
                    if m is not None:
                        mask = m if mask is None else (mask | m)
            p_u = np.clip(pos + 0.5 * spread_pos * uv[k_u], 0, 1) if spread_pos else pos
            v = _wt_lookup(tab, frac, p_u)
            for code, a, var in vw:
                if code == "SelfPD":
                    v = _wt_lookup(tab, np.mod(frac + a * 0.25 * v, 1.0), p_u)
                elif code == "EvenOdd":
                    v2 = _wt_lookup(tab, np.mod(frac + 0.5, 1.0), p_u)
                    even, odd = 0.5 * (v + v2), 0.5 * (v - v2)
                    v = 2 * (1 - a) * odd + 2 * a * even
                elif code == "Flip":
                    v = np.where(frac > 1 - a, -v, v)
            if mask is not None:
                v = np.where(mask, 0.0, v)
            if out.shape[0] == 2:
                gl, gr = _pan_lr(pan)
                out[0] += g * gl * v
                out[1] += g * gr * v
            else:
                out[0] += g * v
        out = out / max(float(np.sqrt(np.sum(np.square(gains)))), 1e-9)
        out_w = [w for w in warps if w[0].startswith(OUTPUT_WARPS)]
        if out_w:
            out = np.vstack([_output_warps(c, out_w, mod_sigs, sr) for c in out])
        return out

    def _fx_legacy(self, patch: Patch, x):
        sr = self.sr
        d = patch.distortion
        if d.enabled and d.mix > 0:
            x = np.vstack([distort(c, d.mode, d.drive, d.mix) for c in x])
        if patch.ott.enabled and patch.ott.depth > 0:
            x = ott(x, patch.ott.depth, patch.ott.amount, sr)
        if patch.eq.enabled:
            x = np.vstack([shelf(shelf(c, patch.eq.low_freq, patch.eq.low_gain, sr, high=False),
                                 patch.eq.high_freq, patch.eq.high_gain, sr, high=True) for c in x])
        if patch.chorus.enabled and patch.chorus.mix > 0:
            from .dsp import chorus_fx
            c = patch.chorus
            x = chorus_fx(x, {"rate_hz": c.rate_hz, "depth": c.depth, "mix": c.mix}, sr)
        return x


def _get_path(obj, path: str):
    for part in path.split("."):
        if obj is None:
            return None
        if part.isdigit():
            obj = obj[int(part)] if int(part) < len(obj) else None
        elif isinstance(obj, dict):
            obj = obj.get(part)
        else:
            obj = getattr(obj, part, None)
    return obj
