"""Offline Serum-style renderer used inside the optimizer loop.

It is an *approximation* of Serum 2's signal path (wavetable oscillators with
unison, sub, noise, one voice filter, 3 envelopes, 4 LFOs, a mod matrix and a
small FX chain). Its job is to be fast and differentiable-by-search so the
optimizer can test hundreds of hypotheses; the final patch is checked against
real Serum on the Mac by ``fourier_analyzer.serum.host`` when available.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

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

    def __init__(self, notes: list[Note], n: int, sr: int, glide: float = 0.0):
        self.n, self.sr = n, sr
        self.notes = sorted(notes, key=lambda x: x.start)
        idx = np.full(n, -1, dtype=np.int32)
        pitch = np.zeros(n, dtype=np.float64)
        t_on = np.zeros(n, dtype=np.float64)
        gate = np.zeros(n, dtype=bool)
        vel = np.zeros(n, dtype=np.float64)
        self.bounds = []        # (start_i, gate_off_i, next_start_i)
        prev_pitch = None
        for k, nt in enumerate(self.notes):
            s = int(round(nt.start * sr))
            e = int(round((nt.start + nt.duration) * sr))
            nxt = int(round(self.notes[k + 1].start * sr)) if k + 1 < len(self.notes) else n
            s, e, nxt = max(0, min(s, n)), max(0, min(e, n)), max(0, min(nxt, n))
            if nxt <= s:
                continue
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
            if glide > 0 and prev_pitch is not None:
                g = np.exp(-tt / (glide / 3.0))
                p += (prev_pitch - nt.pitch) * g
            pitch[seg] = p
            prev_pitch = p[-1] if len(p) else nt.pitch
            self.bounds.append((s, e, nxt))
        # before the first note keep the first pitch so filters/oscs are stable
        if self.bounds:
            first = self.bounds[0][0]
            pitch[:first] = pitch[first] if first < n else 60.0
        self.idx, self.pitch, self.t_on, self.gate, self.vel = idx, pitch, t_on, gate, vel
        self.active = idx >= 0


# --------------------------------------------------------------------------
# envelopes and LFOs


def _env_curve(x, power):
    return 1.0 - (1.0 - np.clip(x, 0, 1)) ** power


def render_env(env: Env, tl: Timeline) -> np.ndarray:
    out = np.zeros(tl.n)
    sr = tl.sr
    a, h, d = max(env.attack, 1e-4), max(env.hold, 0.0), max(env.decay, 1e-4)
    s, r = float(np.clip(env.sustain, 0, 1)), max(env.release, 1e-4)
    # curves: 0.5 attack = linear; 0.666 decay/release = Serum's default
    # exponential (~ -40 dB at the end of the stage); others bend the shape
    pa = 2 ** ((0.5 - getattr(env, "attack_curve", 0.5)) * 4)
    kd = 2 ** ((getattr(env, "decay_curve", 0.666) - 0.666) * 4)
    kr = 2 ** ((getattr(env, "release_curve", 0.666) - 0.666) * 4)
    level_prev = 0.0
    for (st, off, nxt) in tl.bounds:
        n_on = off - st
        t = np.arange(nxt - st) / sr
        lv = np.empty_like(t)
        # attack from the current level (Serum retriggers smoothly in mono)
        att = t < a
        lv[att] = level_prev + (1 - level_prev) * (t[att] / a) ** pa
        hd = (~att) & (t < a + h)
        lv[hd] = 1.0
        dec = t >= a + h
        x = (t[dec] - a - h) / d
        lv[dec] = s + (1 - s) * np.exp(-4.6 * kd * x)
        if n_on < len(t):
            rel_start = max(n_on, 0)
            l0 = lv[rel_start - 1] if rel_start > 0 else level_prev
            tr = t[rel_start:] - t[rel_start]
            lv[rel_start:] = l0 * np.exp(-4.6 * kr * tr / r)
        out[st:nxt] = lv
        level_prev = lv[-1] if len(lv) else 0.0
    return out


def _curve_interp(ph, points):
    """Piecewise curve through [x, y(, curve)] points; curve 0.5 = straight."""
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    cs = np.array([p[2] if len(p) > 2 else 0.5 for p in points], dtype=float)
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


def render_lfo(lfo, tl: Timeline, bpm: float, index: int = 0) -> np.ndarray:
    rate = sync_to_hz(lfo.sync, bpm) if lfo.sync else lfo.rate_hz
    t_abs = np.arange(tl.n) / tl.sr
    t_rel = t_abs if lfo.mode == "free" else tl.t_on
    ph = t_rel * rate + lfo.phase
    if lfo.mode == "env":
        ph = np.minimum(ph, 0.9999)
    kind = getattr(lfo, "type", "shape")
    if kind == "random":
        cyc = np.floor(ph).astype(np.int64)
        if lfo.mode != "free":
            # a new random sequence per note
            cyc = cyc + 1000 * np.maximum(tl.idx, 0)
        v = ((cyc * 2654435761 + 97 * (index + 1)) % 1000003) / 1000003.0
    elif kind in ("lorenz", "rossler"):
        steps = int(tl.n / 64) + 2
        traj = _chaos(kind, steps, rate / (tl.sr / 64), index)
        v = np.interp(t_abs * tl.sr / 64, np.arange(steps), traj)
    else:
        v = lfo_shape(lfo.shape, ph, lfo.points)
    smooth = getattr(lfo, "smooth", 0.0)
    if smooth > 0:
        a = np.exp(-1.0 / (smooth * 0.25 * tl.sr))
        v = signal.lfilter([1 - a], [1, -a], v, zi=[v[0] * a])[0]
    rise, delay = getattr(lfo, "rise", 0.0), getattr(lfo, "delay", 0.0)
    if rise > 0 or delay > 0:
        g = (t_rel - delay) / rise if rise > 0 else (t_rel >= delay).astype(float)
        v = v * np.clip(g, 0, 1)
    return v


# --------------------------------------------------------------------------
# filters


def _biquad(kind: str, fc: float, q: float, sr: int):
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
    else:  # bp, constant 0 dB peak
        b = [alpha, 0.0, -alpha]
    a = [1 + alpha, -2 * cw, 1 - alpha]
    return np.array(b) / a[0], np.array(a) / a[0]


def _biquad_coeffs_vec(kind, fc, q, sr):
    """Vectorised RBJ coefficients for arrays of cutoff values."""
    w0 = 2 * np.pi * fc / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    a0 = 1 + alpha
    if kind == "lp":
        b0 = (1 - cw) / 2; b1 = 1 - cw; b2 = b0
    elif kind == "hp":
        b0 = (1 + cw) / 2; b1 = -(1 + cw); b2 = b0
    else:
        b0 = alpha; b1 = np.zeros_like(alpha); b2 = -alpha
    return (np.stack([b0, b1, b2], 1) / a0[:, None],
            np.stack([np.ones_like(a0), -2 * cw / a0, (1 - alpha) / a0], 1))


if _nb is not None:
    @_nb.njit(cache=True, fastmath=True)
    def _tv_biquad(x, B, A, block):  # pragma: no cover - exercised on the Mac
        y = np.empty_like(x)
        z1 = 0.0
        z2 = 0.0
        for i in range(x.shape[0]):
            k = i // block
            b0, b1, b2 = B[k, 0], B[k, 1], B[k, 2]
            a1, a2 = A[k, 1], A[k, 2]
            xi = x[i]
            yi = b0 * xi + z1
            z1 = b1 * xi - a1 * yi + z2
            z2 = b2 * xi - a2 * yi
            y[i] = yi
        return y
else:
    _tv_biquad = None


def time_varying_filter(x: np.ndarray, kind: str, fc: np.ndarray, q: float, sr: int) -> np.ndarray:
    """Biquad with a cutoff that changes every CTRL samples."""
    nb = int(np.ceil(len(x) / CTRL))
    fc_blocks = np.clip(fc[::CTRL][:nb], 15.0, MAX_CUTOFF_FRAC * sr)
    if np.ptp(fc_blocks) < 1e-3 * fc_blocks.mean():
        b, a = _biquad(kind, float(fc_blocks[0]), q, sr)
        return signal.lfilter(b, a, x)
    B, A = _biquad_coeffs_vec(kind, fc_blocks, np.full(nb, q), sr)
    if _tv_biquad is not None:
        return _tv_biquad(np.ascontiguousarray(x, dtype=np.float64), B, A, CTRL)
    y = np.empty_like(x)
    zi = np.zeros(2)
    for k in range(nb):
        s = slice(k * CTRL, min((k + 1) * CTRL, len(x)))
        y[s], zi = signal.lfilter(B[k], A[k], x[s], zi=zi)
    return y


def apply_filter(x, ftype, cutoff_arr, reso, drive, sr):
    if drive > 0:
        g = 1 + 8 * drive
        x = np.tanh(x * g) / np.tanh(g)
    q = 0.707 + 14.0 * reso ** 2
    if ftype in ("lp24", "hp24"):
        kind = ftype[:2]
        q1 = 0.54 + 0.5 * (q - 0.707)
        y = time_varying_filter(x, kind, cutoff_arr, q1, sr)
        return time_varying_filter(y, kind, cutoff_arr, 1.31 if reso < 0.05 else q1 * 1.4, sr)
    kind = {"lp12": "lp", "hp12": "hp", "bp12": "bp"}.get(ftype, "lp")
    return time_varying_filter(x, kind, cutoff_arr, q, sr)


# --------------------------------------------------------------------------
# FX


def distort(x, mode, drive, mix):
    g = 10 ** (drive * 36 / 20)
    z = x * g
    if mode == "hard":
        y = np.clip(z, -1, 1)
    elif mode == "tube":
        y = np.tanh(z + 0.25) - np.tanh(0.25)
    elif mode == "sinefold":
        y = np.sin(z * np.pi / 2)
    elif mode == "linfold":
        y = 4 * np.abs(np.mod((z + 1) / 4, 1.0) - 0.5) - 1
    elif mode == "diode":
        y = np.where(z > 0, np.tanh(z), 0.3 * np.tanh(z))
        y = y - np.mean(y)
    else:
        y = np.tanh(z)
    return (1 - mix) * x + mix * y


def _onepole_env(x2, sr, att=0.003, rel=0.08):
    """RMS-ish envelope with separate attack/release (block approximate)."""
    a = np.exp(-1 / (att * sr))
    r = np.exp(-1 / (rel * sr))
    fast = signal.lfilter([1 - a], [1, -a], x2)
    slow = signal.lfilter([1 - r], [1, -r], x2)
    return np.sqrt(np.maximum(np.maximum(fast * 0.5, slow), 1e-12))


def ott(x, depth, amount, sr):
    peak = np.max(np.abs(x)) + 1e-9
    x = x / peak * 0.5
    lo_sos = signal.butter(4, 88, "low", fs=sr, output="sos")
    hi_sos = signal.butter(4, min(2500, sr * 0.45), "high", fs=sr, output="sos")
    low = signal.sosfilt(lo_sos, x)
    high = signal.sosfilt(hi_sos, x)
    mid = x - low - high
    out = np.zeros_like(x)
    for band in (low, mid, high):
        env_db = 20 * np.log10(_onepole_env(band ** 2, sr))
        down = -np.maximum(0, env_db + 20) * (1 - 1 / 8.0)
        up = np.maximum(0, -40 - env_db) * (1 - 1 / 3.0)
        g_db = np.clip(amount * (down + np.minimum(up, 24)), -40, 24)
        out += band * 10 ** (g_db / 20)
    return ((1 - depth) * x + depth * out) * peak * 2


def shelf(x, freq, gain_db, sr, high=False):
    if abs(gain_db) < 0.05:
        return x
    A = 10 ** (gain_db / 40)
    w0 = 2 * np.pi * min(freq, sr * 0.45) / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / 2 * np.sqrt(2)
    sa = 2 * np.sqrt(A) * alpha
    if not high:
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    else:
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    return signal.lfilter(np.array(b) / a[0], np.array(a) / a[0], x)


def chorus(x, rate, depth, mix, sr):
    n = np.arange(len(x))
    out = np.zeros_like(x)
    for k, ph in enumerate((0.0, 0.5)):
        d = (0.007 + depth * 0.005 * (0.5 + 0.5 * np.sin(2 * np.pi * (rate * n / sr + ph)))) * sr
        idx = n - d
        i0 = np.floor(idx).astype(int)
        fr = idx - i0
        i0c = np.clip(i0, 0, len(x) - 1)
        i1c = np.clip(i0 + 1, 0, len(x) - 1)
        v = x[i0c] * (1 - fr) + x[i1c] * fr
        v[i0 < 0] = 0
        out += 0.5 * v
    return (1 - mix) * x + mix * out


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
                 "OctaveFifth3": [0, 7, 12, 19, 24, 31, 36], "Center12": [0, 12]}


def unison_offsets(n: int, detune: float, rng_semis: float = 2.0, stack: str = "none") -> tuple[np.ndarray, np.ndarray]:
    """Semitone offsets and relative gains for n unison voices. ``rng_semis``
    is Serum's unison range (spread at full detune); stacks add octaves/fifths."""
    if n <= 1:
        return np.zeros(1), np.ones(1)
    spread = 0.5 * rng_semis * detune ** 1.5          # semitones at the outermost voice
    off = np.linspace(-1, 1, n) * spread
    st = UNISON_STACKS.get(str(stack).replace("k", "", 1) if str(stack).startswith("k") else stack)
    if st:
        off = off + np.array([st[i % len(st)] for i in range(n)], dtype=float)
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
                        ("_FILT1", "osc"), ("_FILT2", "osc2")):
        if code.endswith(suffix):
            return src
    return None


def _warp_phase(frac, code, a):
    """Phase-domain warps (frac in 0..1, amount a scalar or array)."""
    if code == "Sync":
        return np.mod(frac * (1 + a * 7), 1.0), None
    if code in ("BendPos", "Remap_1"):
        return frac ** (1 + 4 * a), None
    if code in ("BendNeg", "Remap_2"):
        return 1 - (1 - frac) ** (1 + 4 * a), None
    if code == "BendPosNeg":
        lo = frac < 0.5
        f2 = np.where(lo, 0.5 * (2 * frac) ** (1 + 4 * a), 1 - 0.5 * (2 - 2 * frac) ** (1 + 4 * a))
        return f2, None
    if code == "PWM":
        w = 1 - 0.98 * a
        return np.minimum(frac / w, 0.999999), frac > w
    if code.startswith("ASYM"):
        d = 0.5 - 0.49 * a if code != "ASYMNeg" else 0.5 + 0.49 * a
        if code == "ASYMPosNeg":
            d = 0.5 - 0.49 * a * np.sign(np.sin(2 * np.pi * frac) + 1e-9)
        return np.where(frac < d, frac * 0.5 / d, 0.5 + (frac - d) * 0.5 / (1 - d)), None
    if code in ("Quantize", "Remap_4"):
        nq = np.maximum(2.0, 256.0 * (1 - a) ** 2)
        return np.floor(frac * nq) / nq, None
    if code == "Mirror":
        m = np.where(frac < 0.5, 2 * frac, 2 - 2 * frac)
        return (1 - a) * frac + a * m * 0.5, None
    if code == "Remap_3":
        return np.mod(frac + a * np.sin(2 * np.pi * frac) / (2 * np.pi), 1.0), None
    return frac, None


class Renderer:
    def __init__(self, bank: TableBank | None = None, sr: int = 22050, seed: int = 7):
        self.bank = bank or TableBank()
        self.sr = sr
        self.seed = seed

    # ------------------------------------------------------------------
    def render(self, patch: Patch, notes: list[Note], duration: float,
               solo: str | None = None) -> np.ndarray:
        """Render notes through the patch. ``solo`` limits output to one part
        ("oscs", "noise", "sub") which the ablation checks use."""
        from .dsp import process_chain
        sr = self.sr
        n = int(round(duration * sr))
        if n <= 0 or not notes:
            return np.zeros(max(n, 0), dtype=np.float32)
        voices = [notes] if patch.mono else [[nt] for nt in notes]
        out = np.zeros(n)
        buses = [np.zeros(n), np.zeros(n)]
        fx_mods = {}
        for vnotes in voices:
            main, b1, b2, fm = self._render_voice(patch, vnotes, n, solo)
            out += main
            buses[0] += b1
            buses[1] += b2
            for k, v in fm.items():
                fx_mods[k] = v if k not in fx_mods else np.maximum(fx_mods[k], v)
        out = self._fx(patch, out)
        out = process_chain(out, patch.fx_chain, sr, fx_mods)
        for b in range(2):
            if np.any(buses[b]):
                lvl = patch.bus_levels[b] if b < len(patch.bus_levels) else 1.0
                out += lvl * process_chain(buses[b], patch.fx_buses[b] if b < len(patch.fx_buses) else [],
                                           sr, fx_mods, prefix=f"bus{b + 1}")
        if not np.all(np.isfinite(out)):
            out = np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
        return (out * patch.master).astype(np.float32)

    # ------------------------------------------------------------------
    def _mods(self, patch: Patch, tl: Timeline):
        srcs = {}
        need = {m.source for m in patch.mods if m.amount} | {m.aux for m in patch.mods if m.amount and m.aux}
        for i, e in enumerate(patch.envs):
            if i == 0 or f"env{i+1}" in need:
                srcs[f"env{i+1}"] = render_env(e, tl)
        for i, l in enumerate(patch.lfos):
            if f"lfo{i+1}" in need:
                srcs[f"lfo{i+1}"] = render_lfo(l, tl, patch.bpm, i)
        if "velocity" in need:
            srcs["velocity"] = tl.vel
        if "note" in need:
            srcs["note"] = np.clip((tl.pitch - 0) / 127.0, 0, 1)
        if "modwheel" in need:
            srcs["modwheel"] = np.zeros(tl.n)
        for i, v in enumerate(patch.macros):
            if f"macro{i+1}" in need:
                srcs[f"macro{i+1}"] = np.full(tl.n, float(v))
        dest = {}
        for m in patch.mods:
            if not m.amount or m.source not in srcs:
                continue
            v = srcs[m.source]
            if m.bipolar:
                v = 2 * v - 1
            if m.aux and m.aux in srcs:
                v = v * srcs[m.aux]
            d = "filter.cutoff" if m.dest == "filter1.cutoff" else m.dest
            dest[d] = dest.get(d, 0) + m.amount * v
        return srcs, dest

    def _noise(self, patch, n, rng):
        nz = rng.standard_normal(n)
        c = patch.noise.color
        if c < 0.5:     # darker: one-pole low-pass
            fc = 200 * 2 ** (c * 2 * 6.6)
            a = np.exp(-2 * np.pi * fc / self.sr)
            nz = signal.lfilter([1 - a], [1, -a], nz) * (1 + (0.5 - c) * 4)
        elif c > 0.5:   # brighter: first-difference tilt
            k = (c - 0.5) * 2
            nz = (1 - k) * nz + k * np.diff(nz, prepend=0.0)
        return 0.35 * nz

    def _filter(self, f, key, x, dm, tl):
        from .dsp import apply_filter_type
        cut_oct = 10.0 * dm.get(f"{key}.cutoff", 0.0) + f.keytrack * (tl.pitch - 60) / 12
        cutoff = f.cutoff * 2 ** np.asarray(cut_oct, dtype=float)
        reso = float(np.clip(f.reso + np.mean(dm.get(f"{key}.reso", 0.0)), 0, 0.98))
        drive = float(np.clip(f.drive + np.mean(dm.get(f"{key}.drive", 0.0)), 0, 1))
        var = float(np.clip(f.var + np.mean(dm.get(f"{key}.var", 0.0)), 0, 1))
        if f.type in ("lp12", "lp24", "hp12", "hp24", "bp12") and var == 0.5 and f.mix >= 1:
            return apply_filter(x, f.type, np.broadcast_to(cutoff, x.shape).astype(float), reso, drive, self.sr)
        return apply_filter_type(x, f.type, np.broadcast_to(cutoff, x.shape), reso, drive, var, self.sr, mix=f.mix)

    def _render_voice(self, patch: Patch, notes, n, solo):
        sr = self.sr
        z = np.zeros(n)
        tl = Timeline(notes, n, sr, patch.glide if patch.mono else 0.0)
        if not tl.bounds:
            return z, z, z, {}
        srcs, dm = self._mods(patch, tl)
        pitch = tl.pitch + 48.0 * dm.get("pitch", 0.0)
        rng = np.random.default_rng(self.seed)
        sources = {}            # name -> signal before filters
        noise_sig = None
        if patch.noise.enabled:
            noise_sig = self._noise(patch, n, rng)
        if solo in (None, "oscs"):
            sub_sig = self._sub(patch, pitch) if patch.sub.enabled else None
            # plain (unwarped) osc signals serve as modulators for cross-warps
            plain = {}
            needs = set()
            for o in patch.oscs:
                if o.enabled:
                    for w, amt in ((o.warp, o.warp_amount), (o.warp2, o.warp2_amount)):
                        src = warp_source(warp_code(w)) if amt else None
                        if src in ("osc", "osc2"):
                            needs.add(src)
            for i, o in enumerate(patch.oscs):
                if o.enabled and needs:
                    plain[i] = self._osc(o, pitch, dm.get(f"osc{i+1}.pos", 0.0), rng, tl, plain_only=True)
            for i, o in patch.active_oscs():
                others = [j for j in (0, 1, 2) if j != i]
                mod_sigs = {"osc": plain.get(others[0]), "osc2": plain.get(others[1]), "sub": sub_sig,
                            "noise": noise_sig if noise_sig is not None else rng.standard_normal(n) * 0.35}
                lvl = np.clip(o.level + dm.get(f"osc{i+1}.level", 0.0), 0, 1.5)
                if np.all(lvl == 0):
                    continue
                sig = self._osc(o, pitch, dm.get(f"osc{i+1}.pos", 0.0), rng, tl, mod_sigs=mod_sigs,
                                warp_mod=(dm.get(f"osc{i+1}.warp", 0.0), dm.get(f"osc{i+1}.warp2", 0.0)),
                                fine_mod=dm.get(f"osc{i+1}.fine", 0.0))
                sources[f"osc{i+1}"] = lvl * sig
            if sub_sig is not None and patch.sub.level > 0:
                sources["sub"] = np.clip(patch.sub.level + dm.get("sub.level", 0.0), 0, 1.5) * sub_sig
        if noise_sig is not None and solo in (None, "noise"):
            nl = np.clip(patch.noise.level + dm.get("noise.level", 0.0), 0, 1.5)
            sources["noise"] = nl * noise_sig
        # routing: each source -> filter 1 / filter 2 (balance) / direct, + bus sends
        f1_in, f2_in, direct = np.zeros(n), np.zeros(n), np.zeros(n)
        b1, b2 = np.zeros(n), np.zeros(n)
        f1_on, f2_on = patch.filter.enabled, patch.filter2.enabled
        for name, sig in sources.items():
            r = patch.route(name)
            if r["dest"] == "none":
                continue
            if r["dest"] == "filter" and (f1_on or f2_on):
                bal = float(np.clip(r["balance"], 0, 1)) if (f1_on and f2_on) else (0.0 if f1_on else 1.0)
                f1_in += (1 - bal) * sig
                f2_in += bal * sig
            else:
                direct += sig
            if r.get("send1"):
                b1 += r["send1"] * sig
            if r.get("send2"):
                b2 += r["send2"] * sig
        x = direct
        if f1_on:
            x = x + self._filter(patch.filter, "filter", f1_in, dm, tl)
        elif np.any(f1_in):
            x = x + f1_in
        if f2_on:
            x = x + self._filter(patch.filter2, "filter2", f2_in, dm, tl)
        elif np.any(f2_in):
            x = x + f2_in
        amp = srcs["env1"] * tl.vel * np.clip(1.0 + dm.get("amp", 0.0), 0, 4)
        fx_mods = {k: v for k, v in dm.items() if k.startswith(("fx.", "bus1.", "bus2."))}
        if "dist.drive" in dm:
            fx_mods["legacy.drive"] = dm["dist.drive"]
        return x * amp, b1 * amp, b2 * amp, fx_mods

    def _sub(self, patch, pitch):
        s = patch.sub
        tid = {"sine": "basic:sine", "triangle": "basic:triangle", "saw": "basic:saw",
               "square": "basic:square", "pulse": "basic:pwm", "roundrect": "basic:square"}.get(s.shape, "basic:sine")
        f = 440.0 * 2 ** ((pitch + 12 * s.octave - 69) / 12)
        tab = self.bank.mip(tid, _mip_level(float(f.max()), self.sr))
        pos = np.full(1, 0.7 if s.shape == "pulse" else 0.0)
        y = _wt_lookup(tab, np.cumsum(f / self.sr), pos)
        if s.shape == "roundrect":
            y = np.tanh(2.5 * y) / np.tanh(2.5)
        return y

    def _osc(self, o, pitch, posmod, rng, tl=None, mod_sigs=None, warp_mod=(0.0, 0.0), fine_mod=0.0,
             plain_only=False):
        sr = self.sr
        base = pitch + 12 * o.octave + o.semi + (o.fine + 100.0 * np.asarray(fine_mod)) / 100.0
        offs, gains = unison_offsets(max(1, int(o.unison)), o.detune, getattr(o, "unison_range", 2.0),
                                     getattr(o, "unison_stack", "none"))
        if len(gains) > 1:
            gains = np.where(np.abs(np.linspace(-1, 1, len(gains))) < 0.34, 1.0, o.blend)
        f_max = 440.0 * 2 ** ((float(np.max(base)) + offs.max() - 69) / 12)
        tab = self.bank.mip(o.table, _mip_level(f_max, sr))
        pos = np.clip(o.position + posmod, 0, 1) if np.ndim(posmod) else np.array([np.clip(o.position + posmod, 0, 1)])
        posarg = pos if pos.size > 1 else np.full(1, pos[0])
        warps = [] if plain_only else [
            (warp_code(o.warp), np.clip(o.warp_amount + warp_mod[0], 0, 1)),
            (warp_code(o.warp2), np.clip(o.warp2_amount + warp_mod[1], 0, 1))]
        warps = [(c, a) for c, a in warps if c != "none" and np.any(np.asarray(a) > 0)]
        mod_sigs = mod_sigs or {}
        out = 0.0
        starts = [b[0] for b in tl.bounds] if tl is not None else [0]
        for off, g in zip(offs, gains):
            f = 440.0 * 2 ** ((base + off - 69) / 12)
            for code, a in warps:          # exponential FM changes the frequency itself
                if code.startswith("FMX"):
                    src = mod_sigs.get(warp_source(code))
                    if src is not None:
                        f = f * 2 ** (a * 2.0 * src)
            ph = np.cumsum(f / sr)
            # phase restarts at every note-on (Serum's default retrigger), at
            # `phase` plus an optional random offset per note
            offset = np.zeros(len(ph))
            for k, s0 in enumerate(starts):
                s1 = starts[k + 1] if k + 1 < len(starts) else len(ph)
                base_ph = ph[s0 - 1] if s0 > 0 else 0.0
                offset[s0:s1] = -base_ph + o.phase + o.random_phase * rng.random()
            ph = ph + offset
            mask = None
            for code, a in warps:
                src = mod_sigs.get(warp_source(code))
                if code.startswith(("FM_", "FMP_")) and src is not None:
                    ph = ph + a * FM_INDEX * src           # Serum FM = phase modulation
                elif code.startswith("PD_") and src is not None:
                    ph = ph + a * 0.5 * (0.5 + 0.5 * src)  # unipolar phase distortion
            frac = np.mod(ph, 1.0)
            for code, a in warps:
                if code in PHASE_WARPS:
                    frac, m = _warp_phase(frac, code, a)
                    if m is not None:
                        mask = m if mask is None else (mask | m)
            v = _wt_lookup(tab, frac, posarg)
            for code, a in warps:
                if code == "SelfPD":
                    v = _wt_lookup(tab, np.mod(frac + a * 0.25 * v, 1.0), posarg)
                elif code == "EvenOdd":
                    v2 = _wt_lookup(tab, np.mod(frac + 0.5, 1.0), posarg)
                    even, odd = 0.5 * (v + v2), 0.5 * (v - v2)
                    v = 2 * (1 - a) * even + 2 * a * odd
                elif code == "Flip":
                    v = np.where(frac > 1 - a, -v, v)
            if mask is not None:
                v = np.where(mask, 0.0, v)
            out = out + g * v
        out = out / max(float(np.sqrt(np.sum(np.square(gains)))), 1e-9)
        for code, a in warps:              # output-domain warps
            src = mod_sigs.get(warp_source(code))
            if code.startswith("AM_") and src is not None:
                out = out * ((1 - a) + a * (0.5 + 0.5 * src))
            elif code.startswith("RM_") and src is not None:
                out = out * ((1 - a) + a * src)
            elif code.startswith("Dist"):
                from .dsp import shape
                out = shape(out, code, float(np.mean(a)))
            elif code == "FilterLPF":
                from .dsp import onepole_lp
                out = onepole_lp(out, 20 * 1000 ** (1 - np.asarray(a, dtype=float)), sr)
            elif code == "FilterHPF":
                from .dsp import onepole_lp
                out = out - onepole_lp(out, 20 * 1000 ** np.asarray(a, dtype=float), sr)
        return out

    def _fx(self, patch: Patch, x):
        sr = self.sr
        d = patch.distortion
        if d.enabled and d.mix > 0:
            x = distort(x, d.mode, d.drive, d.mix)
        if patch.ott.enabled and patch.ott.depth > 0:
            x = ott(x, patch.ott.depth, patch.ott.amount, sr)
        if patch.eq.enabled:
            x = shelf(x, patch.eq.low_freq, patch.eq.low_gain, sr, high=False)
            x = shelf(x, patch.eq.high_freq, patch.eq.high_gain, sr, high=True)
        if patch.chorus.enabled and patch.chorus.mix > 0:
            c = patch.chorus
            x = chorus(x, c.rate_hz, c.depth, c.mix, sr)
        return x
