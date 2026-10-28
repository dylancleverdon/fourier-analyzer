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
    level_prev = 0.0
    for (st, off, nxt) in tl.bounds:
        n_on = off - st
        t = np.arange(nxt - st) / sr
        lv = np.empty_like(t)
        # attack from the current level (Serum retriggers smoothly in mono)
        att = t < a
        lv[att] = level_prev + (1 - level_prev) * (t[att] / a)
        hd = (~att) & (t < a + h)
        lv[hd] = 1.0
        dec = t >= a + h
        x = (t[dec] - a - h) / d
        lv[dec] = s + (1 - s) * np.exp(-4.6 * x)      # ~ -40 dB at x = 1
        if n_on < len(t):
            rel_start = max(n_on, 0)
            l0 = lv[rel_start - 1] if rel_start > 0 else level_prev
            tr = t[rel_start:] - t[rel_start]
            lv[rel_start:] = l0 * np.exp(-4.6 * tr / r)
        out[st:nxt] = lv
        level_prev = lv[-1] if len(lv) else 0.0
    return out


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
    if shape == "custom" and points:
        xs = np.array([p[0] for p in points], dtype=float)
        ys = np.array([p[1] for p in points], dtype=float)
        return np.interp(ph, xs, ys)
    return 0.5 - 0.5 * np.cos(2 * np.pi * ph)


def render_lfo(lfo, tl: Timeline, bpm: float) -> np.ndarray:
    rate = sync_to_hz(lfo.sync, bpm) if lfo.sync else lfo.rate_hz
    if lfo.mode == "free":
        ph = np.arange(tl.n) / tl.sr * rate + lfo.phase
    else:
        ph = tl.t_on * rate + lfo.phase
        if lfo.mode == "env":
            ph = np.minimum(ph, 0.9999)
    return lfo_shape(lfo.shape, ph, lfo.points)


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


def unison_offsets(n: int, detune: float) -> tuple[np.ndarray, np.ndarray]:
    """Semitone offsets and relative gains for n unison voices."""
    if n <= 1:
        return np.zeros(1), np.ones(1)
    spread = 1.0 * detune ** 1.5          # semitones at the outermost voice
    off = np.linspace(-1, 1, n) * spread
    return off, np.ones(n)


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
        sr = self.sr
        n = int(round(duration * sr))
        if n <= 0 or not notes:
            return np.zeros(max(n, 0), dtype=np.float32)
        voices = [notes] if patch.mono else [[nt] for nt in notes]
        out = np.zeros(n)
        for vnotes in voices:
            out += self._render_voice(patch, vnotes, n, solo)
        out = self._fx(patch, out)
        return (out * patch.master).astype(np.float32)

    # ------------------------------------------------------------------
    def _mods(self, patch: Patch, tl: Timeline):
        srcs = {}
        need = {m.source for m in patch.mods if m.amount}
        for i, e in enumerate(patch.envs):
            if i == 0 or f"env{i+1}" in need:
                srcs[f"env{i+1}"] = render_env(e, tl)
        for i, l in enumerate(patch.lfos):
            if f"lfo{i+1}" in need:
                srcs[f"lfo{i+1}"] = render_lfo(l, tl, patch.bpm)
        dest = {}
        for m in patch.mods:
            if not m.amount or m.source not in srcs:
                continue
            v = srcs[m.source]
            if m.bipolar:
                v = 2 * v - 1
            dest[m.dest] = dest.get(m.dest, 0) + m.amount * v
        return srcs, dest

    def _render_voice(self, patch: Patch, notes, n, solo):
        sr = self.sr
        tl = Timeline(notes, n, sr, patch.glide if patch.mono else 0.0)
        if not tl.bounds:
            return np.zeros(n)
        srcs, dm = self._mods(patch, tl)
        pitch = tl.pitch + 48.0 * dm.get("pitch", 0.0)
        rng = np.random.default_rng(self.seed)
        x = np.zeros(n)
        if solo in (None, "oscs"):
            for i, o in patch.active_oscs():
                lvl = np.clip(o.level + dm.get(f"osc{i+1}.level", 0.0), 0, 1.5)
                if np.all(lvl == 0):
                    continue
                x += lvl * self._osc(o, pitch, dm.get(f"osc{i+1}.pos", 0.0), rng)
            if patch.sub.enabled:
                s = patch.sub
                tid = {"sine": "basic:sine", "triangle": "basic:triangle", "saw": "basic:saw",
                       "square": "basic:square"}.get(s.shape, "basic:sine")
                f = 440.0 * 2 ** ((pitch + 12 * s.octave - 69) / 12)
                tab = self.bank.mip(tid, _mip_level(float(f.max()), sr))
                ph = np.cumsum(f / sr)
                x += s.level * _wt_lookup(tab, ph, np.zeros(1))
        if patch.noise.enabled and solo in (None, "noise"):
            nz = rng.standard_normal(n)
            c = patch.noise.color
            if c < 0.5:     # darker: one-pole low-pass
                fc = 200 * 2 ** (c * 2 * 6.6)
                a = np.exp(-2 * np.pi * fc / sr)
                nz = signal.lfilter([1 - a], [1, -a], nz) * (1 + (0.5 - c) * 4)
            elif c > 0.5:   # brighter: first-difference tilt
                k = (c - 0.5) * 2
                nz = (1 - k) * nz + k * np.diff(nz, prepend=0.0)
            nl = np.clip(patch.noise.level + dm.get("noise.level", 0.0), 0, 1.5)
            x += nl * 0.35 * nz
        f = patch.filter
        if f.enabled:
            cut_oct = 10.0 * dm.get("filter.cutoff", 0.0) + f.keytrack * (tl.pitch - 60) / 12
            cutoff = f.cutoff * 2 ** cut_oct if np.ndim(cut_oct) else np.full(n, f.cutoff * 2 ** cut_oct)
            reso = float(np.clip(f.reso + np.mean(dm.get("filter.reso", 0.0)), 0, 0.98))
            x = apply_filter(x, f.type, np.asarray(cutoff, dtype=float), reso, f.drive, sr)
        amp = srcs["env1"] * tl.vel * np.clip(1.0 + dm.get("amp", 0.0), 0, 4)
        return x * amp

    def _osc(self, o, pitch, posmod, rng):
        sr = self.sr
        base = pitch + 12 * o.octave + o.semi + o.fine / 100.0
        offs, gains = unison_offsets(max(1, int(o.unison)), o.detune)
        if len(gains) > 1:
            gains = np.where(np.abs(np.linspace(-1, 1, len(gains))) < 0.34, 1.0, o.blend)
        f_max = 440.0 * 2 ** ((float(np.max(base)) + offs.max() - 69) / 12)
        tab = self.bank.mip(o.table, _mip_level(f_max, sr))
        pos = np.clip(o.position + posmod, 0, 1) if np.ndim(posmod) else np.array([np.clip(o.position + posmod, 0, 1)])
        out = 0.0
        for off, g in zip(offs, gains):
            f = 440.0 * 2 ** ((base + off - 69) / 12)
            ph = np.cumsum(f / sr) + o.random_phase * rng.random()
            out = out + g * _wt_lookup(tab, ph, pos if pos.size > 1 else np.full(1, pos[0]))
        return out / np.sqrt(np.sum(np.square(gains)))

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
