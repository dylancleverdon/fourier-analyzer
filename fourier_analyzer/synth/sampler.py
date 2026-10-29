"""Sample-based oscillator modes: Sample, Granular, Spectral and Multisample
(Serum 2 manual pp. 58-122), plus the spectral-only warp modes.

All renderers take the voice's frequency curve (Hz per output sample) and
its note bounds, and return a (2, n) stereo signal; the engine folds it to
mono when it renders mono.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np

from .samples import Sample, SampleBank, parse_sfz, pick_region


def _note_hz(note: float, a4: float = 440.0) -> float:
    return a4 * 2 ** ((note - 69) / 12)


def _interp(data: np.ndarray, pos: np.ndarray) -> np.ndarray:
    """Linear interpolation of (C, N) data at fractional positions; outside the
    sample is silence."""
    n = data.shape[1]
    valid = (pos >= 0) & (pos <= n - 1)
    p = np.clip(pos, 0, n - 1.000001)
    i0 = p.astype(np.int64)
    fr = p - i0
    out = data[:, i0] * (1 - fr) + data[:, np.minimum(i0 + 1, n - 1)] * fr
    return np.where(valid, out, 0.0)


def _fold_loop(pos, ls, le, mode):
    """Map a straight playhead onto the loop region (positions in samples)."""
    L = max(le - ls, 1.0)
    over = pos > le
    if not np.any(over):
        return pos
    x = pos[over] - le
    if mode == "forward" or mode == "tailed":
        y = ls + np.mod(x, L)
    elif mode == "reverse":
        y = le - np.mod(x, L)
    else:                                  # pingpong
        k = np.mod(x, 2 * L)
        y = np.where(k < L, le - k, ls + (k - L))
    out = pos.copy()
    out[over] = y
    return out


def _stereo(x: np.ndarray) -> np.ndarray:
    x = np.atleast_2d(x)
    return np.vstack([x, x]) if x.shape[0] == 1 else x[:2]


def _pan_gains(pan) -> tuple:
    pan = np.clip(pan, -1, 1)
    a = (pan + 1) * np.pi / 4
    return np.cos(a) * np.sqrt(2), np.sin(a) * np.sqrt(2)


# ---------------------------------------------------------------------------
# Sample mode

def render_sample(smp: Sample, cfg, freq: np.ndarray, bounds, gate: np.ndarray, sr: int, rng,
                  unison=None, a4: float = 440.0) -> np.ndarray:
    """Classic sampler: playback rate follows the note relative to the
    sample's root; start/end markers, loop modes, crossfade, scan speed."""
    n = len(freq)
    out = np.zeros((2, n))
    if smp is None or not bounds:
        return out
    data = _stereo(smp.data).astype(np.float64) * cfg.gain
    N = data.shape[1]
    root_hz = _note_hz(smp.root if cfg.root is None else cfg.root, a4) * 2 ** (cfg.detune / 1200)
    sr_ratio = smp.sr / sr
    s_idx = cfg.start * (N - 1)
    e_idx = max(cfg.end * (N - 1), s_idx + 1)
    ls, le = cfg.loop_start * (N - 1), cfg.loop_end * (N - 1)
    if cfg.loop == "tailed":
        ls = 0.5 * (s_idx + e_idx)
        le = e_idx
    if cfg.link_loop:
        le = min(e_idx, ls + max(le - ls, 1))
    if le < ls:
        ls, le = le, ls
    ls, le = max(ls, s_idx), min(le, e_idx)
    scan = cfg.scan * (-1 if cfg.reverse else 1)
    voices = unison or [(0.0, 1.0, 0.0)]           # (semitone offset, gain, pan)
    for k, (s0, off_i, nxt) in enumerate(bounds):
        seg = slice(s0, nxt)
        f = freq[seg]
        for vi, (semi, g, pan) in enumerate(voices):
            rate = f / root_hz * 2 ** (semi / 12) * sr_ratio * scan
            start = s_idx + (cfg.span * vi / max(len(voices) - 1, 1) + cfg.random_start * rng.random()) * (e_idx - s_idx)
            if scan < 0:
                start = e_idx - (start - s_idx)
            pos = start + np.concatenate([[0.0], np.cumsum(rate[:-1])])
            if cfg.loop in ("forward", "reverse", "pingpong", "tailed") and le - ls > 2:
                rel_gate = off_i - s0
                if cfg.exit_on_release and rel_gate < len(pos):
                    looped = _fold_loop(pos[:rel_gate], ls, le, cfg.loop)
                    tail = pos[rel_gate:] - pos[rel_gate] + (looped[-1] if len(looped) else pos[rel_gate])
                    pos = np.concatenate([looped, tail])
                else:
                    pos = _fold_loop(pos, ls, le, cfg.loop)
            y = _interp(data, pos)
            y = np.where((pos > e_idx) | (pos < s_idx - 1), 0.0, y)
            if cfg.crossfade > 0 and cfg.loop != "off":
                xf = cfg.crossfade * (le - ls)
                near = (pos > le - xf) & (pos <= le)
                if np.any(near):
                    w = (pos[near] - (le - xf)) / max(xf, 1)
                    alt = _interp(data, pos[near] - (le - ls))
                    y[:, near] = y[:, near] * (1 - w) + alt * w
            gl, gr = _pan_gains(pan)
            out[0, seg] += g * gl * y[0]
            out[1, seg] += g * gr * y[1]
    return out / np.sqrt(sum(v[1] ** 2 for v in voices))


# ---------------------------------------------------------------------------
# Granular mode

def _window(shape: str, m: int, skew: float, amount: float) -> np.ndarray:
    if m <= 1:
        return np.ones(max(m, 1))
    x = np.linspace(0, 1, m)
    if skew:                         # move the peak left (-1) or right (+1)
        c = np.clip(0.5 + 0.45 * skew, 0.05, 0.95)
        x = np.where(x < c, 0.5 * x / c, 0.5 + 0.5 * (x - c) / (1 - c))
    s = (shape or "Hann").replace("kWindow", "")
    if s == "Welch":
        w = 1 - (2 * x - 1) ** 2
    elif s == "Gaussian":
        w = np.exp(-0.5 * ((x - 0.5) / 0.15) ** 2)
    elif s == "BlackmanHarris":
        a = 2 * np.pi * x
        w = 0.35875 - 0.48829 * np.cos(a) + 0.14128 * np.cos(2 * a) - 0.01168 * np.cos(3 * a)
    elif s == "Sinc":
        w = np.sinc(6 * (x - 0.5)) * (0.5 - 0.5 * np.cos(2 * np.pi * x))
    elif s == "Tukey":
        w = np.clip(np.minimum(x, 1 - x) / 0.25, 0, 1)
        w = 0.5 - 0.5 * np.cos(np.pi * w)
    elif s == "Triangle":
        w = 1 - np.abs(2 * x - 1)
    elif s == "Trapezoid":
        w = np.clip(np.minimum(x, 1 - x) / 0.2, 0, 1)
    elif s == "ExpDec":
        w = np.exp(-5 * x) * np.clip(x / 0.01, 0, 1)
    elif s == "ExpDecRev":
        w = np.exp(-5 * (1 - x)) * np.clip((1 - x) / 0.01, 0, 1)
    else:                            # Hann
        w = 0.5 - 0.5 * np.cos(2 * np.pi * x)
    return (1 - amount) + amount * w


def render_granular(smp: Sample, cfg, gcfg, freq, bounds, gate, sr, rng, bpm: float = 140.0,
                    unison=None, a4: float = 440.0) -> np.ndarray:
    n = len(freq)
    out = np.zeros((2, n))
    if smp is None or not bounds:
        return out
    data = _stereo(smp.data).astype(np.float64) * cfg.gain
    N = data.shape[1]
    root_hz = _note_hz(smp.root if cfg.root is None else cfg.root, a4) * 2 ** (cfg.detune / 1200)
    sr_ratio = smp.sr / sr
    s_idx, e_idx = cfg.start * (N - 1), max(cfg.end * (N - 1), cfg.start * (N - 1) + 1)
    beat = 60.0 / max(bpm, 1.0)
    if gcfg.density_mode == "bpm":
        period = max(gcfg.density, 1e-3) * 4 * beat
    else:
        period = 1.0 / max(gcfg.density, 0.05)
    if gcfg.length_mode == "bpm":
        glen = gcfg.length * 4 * beat
    elif gcfg.length_mode == "percent":
        glen = gcfg.length * period
    else:
        glen = gcfg.length
    if gcfg.density_mode == "grains":            # N grains overlapping at all times
        period = glen / max(gcfg.density, 0.5)
    glen = float(np.clip(glen, 0.002, 4.0))
    voices = unison or [(0.0, 1.0, 0.0)]
    nv = len(voices)
    overlap = max(1.0, glen / period * nv)
    norm = 1.0 / np.sqrt(overlap)
    for (s0, off_i, nxt) in bounds:
        dur = (nxt - s0) / sr
        if dur <= 0:
            continue
        # playhead through the sample (scan = speed relative to real time)
        t_sp = np.arange(0, dur, period)
        if gcfg.jump_start:
            t_sp = np.concatenate([-np.arange(1, int(min(overlap, 8))) * period * 0, t_sp])
        for vi, (semi, g, pan) in enumerate(voices):
            if gcfg.spawn == "even":
                shift = period * vi / nv
            elif gcfg.spawn == "exp":
                shift = period * (2 ** (vi / max(nv - 1, 1)) - 1)
            elif gcfg.spawn == "random":
                shift = period * rng.random()
            else:
                shift = 0.0
            for tg in t_sp + shift:
                if tg >= dur:
                    break
                i0 = s0 + int(tg * sr)
                fi = float(freq[min(i0, n - 1)])
                ratio = fi / root_hz * 2 ** ((semi + gcfg.rand_pitch * (2 * rng.random() - 1)) / 12) * sr_ratio
                if cfg.manual_position:
                    head = cfg.position
                else:
                    sp = cfg.scan * (ratio / sr_ratio if cfg.scan_keytrack else 1.0)
                    head = (tg * smp.sr * sp) / max(e_idx - s_idx, 1)
                    head = np.mod(head, 1.0) if cfg.loop != "off" else min(head, 1.0)
                src = s_idx + (head + gcfg.rand_offset * (rng.random() - 0.5)) * (e_idx - s_idx)
                L = glen * (1 + gcfg.rand_length * (2 * rng.random() - 1))
                m = int(max(4, L * sr))
                rev = (rng.random() < 0.5 * gcfg.rand_dir) ^ bool(gcfg.reverse_grains)
                step = ratio * (-1 if rev else 1)
                pos = src + step * np.arange(m)
                y = _interp(data, pos)
                w = _window(gcfg.window, m, np.clip(gcfg.window_skew + gcfg.rand_skew * (2 * rng.random() - 1), -1, 1),
                            np.clip(gcfg.window_amount - gcfg.rand_window * rng.random(), 0, 1))
                amp = g * norm * (1 - gcfg.rand_gain * rng.random())
                gp = pan + gcfg.rand_pan * (2 * rng.random() - 1)
                gl, gr = _pan_gains(gp)
                e = min(i0 + m, n, s0 + (nxt - s0) + int(0.5 * sr))
                k = e - i0
                if k <= 0:
                    continue
                out[0, i0:e] += amp * gl * w[:k] * y[0, :k]
                out[1, i0:e] += amp * gr * w[:k] * y[1, :k]
    return out


# ---------------------------------------------------------------------------
# Spectral mode

SPEC_N = 2048
SPEC_HOP = 512


@lru_cache(maxsize=16)
def _stft_cached(key: int, data_bytes: bytes, shape: tuple):
    x = np.frombuffer(data_bytes, dtype=np.float32).reshape(shape).mean(axis=0)
    win = np.hanning(SPEC_N)
    pad = np.concatenate([np.zeros(SPEC_N // 2), x, np.zeros(SPEC_N)])
    nf = max(1, (len(pad) - SPEC_N) // SPEC_HOP)
    idx = np.arange(SPEC_N)[None, :] + SPEC_HOP * np.arange(nf)[:, None]
    F = np.fft.rfft(pad[idx] * win, axis=1)
    mag = np.abs(F)
    ph = np.angle(F)
    # instantaneous frequency (bins) from the phase advance between frames
    k = np.arange(F.shape[1])
    expected = 2 * np.pi * SPEC_HOP * k / SPEC_N
    dphi = np.diff(ph, axis=0, prepend=ph[:1]) - expected
    dphi = np.mod(dphi + np.pi, 2 * np.pi) - np.pi
    inst = k + dphi * SPEC_N / (2 * np.pi * SPEC_HOP)
    return mag.astype(np.float32), inst.astype(np.float32)


def spectral_frames(smp: Sample):
    d = np.ascontiguousarray(_stereo(smp.data), dtype=np.float32)
    return _stft_cached(id(smp.data), d.tobytes(), d.shape)


def _curve_eval(points, x):
    if not points:
        return np.ones_like(x)
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    return np.interp(x, xs, ys)


def spectral_warp(mag, freqs, code: str, a: float, var: float = 0.5, modmag=None, state=None):
    """One spectral warp on a magnitude frame (``freqs`` = bin centre Hz)."""
    c = code
    nb = len(mag)
    b = np.arange(nb, dtype=float)
    if a <= 0 or c in ("none", ""):
        return mag
    if c == "SpectralComb":
        teeth = 2 + 30 * var
        return mag * (1 - a + a * (0.5 + 0.5 * np.cos(2 * np.pi * b / teeth)))
    if c == "Spread":
        src = b * (1 + a) ** -1
        return np.interp(src * (1 + 0.5 * a * b / nb), b, mag, right=0)
    if c == "Gate":
        thr = a * mag.max()
        return np.where(mag >= thr, mag, 0.0)
    if c in ("Addharmonics", "Addsubharmonics"):
        k = 2.0 if c == "Addharmonics" else 0.5
        return mag + a * np.interp(b / k, b, mag, right=0)
    if c == "Detune":
        alt = np.where((b.astype(int) % 2) == 0, 1, -1)
        return np.interp(b * 2 ** (-alt * a * 0.5 / 12), b, mag, right=0)
    if c == "Smear":
        if state is not None:
            prev = state.get("smear")
            if prev is not None:
                mag = (1 - a) * mag + a * prev
            state["smear"] = mag.copy()
        return mag
    if c in ("PeakOctaveDown", "PeakOctaveUp", "PeakHarmDown", "PeakHarmUp", "PeakFollow"):
        thr = 0.25 * mag.max()
        peaks = (mag > thr) & (mag >= np.roll(mag, 1)) & (mag >= np.roll(mag, -1))
        if c == "PeakFollow":
            keep = np.where(peaks, mag, 0.0)
            return (1 - a) * mag + a * keep
        k = {"PeakOctaveDown": 0.5, "PeakOctaveUp": 2.0}.get(c)
        moved = np.zeros_like(mag)
        for p in np.nonzero(peaks)[0]:
            q = p * k if k else (p + (-1 if c == "PeakHarmDown" else 1) * max(1, p // max(1, np.argmax(peaks) or 1)))
            q = int(round(q))
            if 0 <= q < nb:
                moved[q] += mag[p]
        rest = np.where(peaks, 0.0, mag)
        return (1 - a) * mag + a * (rest + moved)
    if c == "SpectralPitchShift":
        r = 2 ** ((a - 0.5) * 24 / 12)
        return np.interp(b / r, b, mag, right=0)
    if c == "SpectralShift":
        shift_bins = (a - 0.5) * 2 * nb * 0.1
        return np.interp(b - shift_bins, b, mag, left=0, right=0)
    if c in ("ShepardFilter", "ShepardNarrow"):
        w = 0.08 if c == "ShepardNarrow" else 0.2
        oc = np.log2(np.maximum(freqs, 1.0) / 20.0)
        d = np.mod(oc - a * 1.0, 1.0) - 0.5
        return mag * np.exp(-0.5 * (d / w) ** 2)
    if c in ("Mask_OSC", "Mask_NOISE", "Mask_OSC2") and modmag is not None:
        m = modmag / (modmag.max() + 1e-9)
        return mag * (1 - a + a * m)
    if c in ("Vocode_OSC", "Vocode_NOISE", "Vocode_OSC2") and modmag is not None:
        env = np.convolve(modmag, np.ones(9) / 9, mode="same")
        own = np.convolve(mag, np.ones(9) / 9, mode="same") + 1e-9
        return mag * (1 - a) + a * mag / own * env
    return mag


SPECTRAL_WARPS = ("SpectralComb", "Spread", "Gate", "Addharmonics", "Addsubharmonics", "Detune", "Smear",
                  "PeakOctaveDown", "PeakOctaveUp", "PeakHarmDown", "PeakHarmUp", "PeakFollow",
                  "SpectralPhaseTwist", "SpectralPitchShift", "SpectralShift", "ShepardFilter", "ShepardNarrow",
                  "Mask_OSC", "Mask_OSC2", "Mask_NOISE", "Vocode_OSC", "Vocode_OSC2", "Vocode_NOISE", "DLM")


def render_spectral(smp: Sample, cfg, scfg, freq, bounds, gate, sr, rng, warps=(), mod_sigs=None,
                    unison=None, a4: float = 440.0) -> np.ndarray:
    """Phase-vocoder resynthesis: scan through the sample's spectrum, shifted to
    the played pitch, band limited, spectrally filtered and warped."""
    n = len(freq)
    out = np.zeros((2, n))
    if smp is None or not bounds:
        return out
    mag_in, inst_in = spectral_frames(smp)
    nfr, nb = mag_in.shape
    root_hz = _note_hz(smp.root if cfg.root is None else cfg.root, a4) * 2 ** (cfg.detune / 1200)
    # output frames: same frame duration in seconds as the analysis frames
    hop_out = max(16, int(round(SPEC_HOP * sr / smp.sr)))
    nfft_out = hop_out * 4
    nb_out = nfft_out // 2 + 1
    win = np.hanning(nfft_out)
    bin_hz_in = smp.sr / SPEC_N
    f_out = np.arange(nb_out) * sr / nfft_out
    lo, hi = scfg.freq_lo, scfg.freq_hi
    if scfg.smooth_edges:
        band = 1 / np.sqrt(1 + (lo / np.maximum(f_out, 1e-3)) ** 8) / np.sqrt(1 + (f_out / max(hi, 1)) ** 8)
    else:
        band = ((f_out >= lo) & (f_out <= hi)).astype(float)
    xo = np.clip(np.log2(np.maximum(f_out, 20) / 20) / 10 - scfg.filter_shift / 100.0, 0, 1)
    mask = _curve_eval(scfg.filter_curve, xo) if scfg.filter_curve else np.clip(1 - xo, 0, 1)
    fmask = (1 - scfg.filter_mix) + scfg.filter_mix * mask
    s_fr, e_fr = cfg.start * (nfr - 1), max(cfg.end * (nfr - 1), cfg.start * (nfr - 1) + 1)
    spec_warps = [(c, a, v) for c, a, v in warps if c in SPECTRAL_WARPS and np.mean(a) > 0]
    voices = unison or [(0.0, 1.0, 0.0)]
    mod_spec = {}
    for c, _, _ in spec_warps:
        if c.endswith(("_OSC", "_OSC2", "_NOISE")) and mod_sigs:
            key = {"_OSC": "osc", "_OSC2": "osc2", "_NOISE": "noise"}[c[c.index("_"):]]
            sig = mod_sigs.get(key)
            if sig is not None:
                sig = np.asarray(sig, dtype=float)
                sig = sig.mean(axis=0) if sig.ndim == 2 else sig
                pad = np.concatenate([np.zeros(nfft_out // 2), sig, np.zeros(nfft_out)])
                mod_spec[key] = pad
    for (s0, off_i, nxt) in bounds:
        length = nxt - s0
        nfo = int(np.ceil(length / hop_out)) + 4
        for vi, (semi, g, pan) in enumerate(voices):
            phase = rng.random(nb_out) * 2 * np.pi if not scfg.phase_lock else np.zeros(nb_out)
            y = np.zeros(nfo * hop_out + nfft_out)
            state = {}
            pos = s_fr + cfg.span * vi / max(len(voices) - 1, 1) * (e_fr - s_fr)
            for j in range(nfo):
                ti = min(s0 + j * hop_out, n - 1)
                ratio = float(freq[ti]) / root_hz * 2 ** (semi / 12)
                fi = pos if not cfg.manual_position else s_fr + cfg.position * (e_fr - s_fr)
                if cfg.loop != "off" and fi > e_fr:
                    fi = s_fr + np.mod(fi - s_fr, max(e_fr - s_fr, 1))
                if fi > e_fr or fi < 0:
                    break
                k0 = int(fi)
                fr = fi - k0
                k1 = min(k0 + 1, nfr - 1)
                m = mag_in[min(k0, nfr - 1)] * (1 - fr) + mag_in[k1] * fr
                inst = inst_in[min(k0, nfr - 1)]
                # move every input bin to ratio x its (instantaneous) frequency
                tgt_hz = inst * bin_hz_in * ratio
                tb = np.clip(np.round(tgt_hz / (sr / nfft_out)).astype(int), 0, nb_out - 1)
                mo = np.zeros(nb_out)
                np.add.at(mo, tb, m * (tgt_hz < sr / 2))
                fo = np.zeros(nb_out)
                np.maximum.at(fo, tb, tgt_hz)
                if not scfg.post_warp:
                    mo *= band
                for c, a, v in spec_warps:
                    mm = None
                    for key, pad in mod_spec.items():
                        if c.endswith(("_OSC", "_OSC2", "_NOISE")):
                            seg = pad[ti: ti + nfft_out]
                            if len(seg) == nfft_out:
                                mm = np.abs(np.fft.rfft(seg * win))
                    mo = spectral_warp(mo, f_out, c, float(np.mean(a)), v, mm, state)
                    if c == "SpectralPhaseTwist":
                        phase = phase + float(np.mean(a)) * np.arange(nb_out) * 0.05
                if scfg.post_warp:
                    mo *= band
                mo *= fmask
                phase = phase + 2 * np.pi * np.where(fo > 0, fo, f_out) * hop_out / sr
                if scfg.transients and j > 0:
                    pass                          # transients keep the input phase jump (approximate)
                frame = np.fft.irfft(mo * np.exp(1j * phase), n=nfft_out) * win
                y[j * hop_out: j * hop_out + nfft_out] += frame
                pos += cfg.scan * (ratio if cfg.scan_keytrack else 1.0) * hop_out * smp.sr / sr / SPEC_HOP
            y = y[nfft_out // 2: nfft_out // 2 + length] * (2.0 / 3.0) / (SPEC_N / 2) * 4
            gl, gr = _pan_gains(pan)
            out[0, s0:nxt] += g * gl * y[: nxt - s0]
            out[1, s0:nxt] += g * gr * y[: nxt - s0]
    return out / np.sqrt(sum(v[1] ** 2 for v in voices)) * cfg.gain


# ---------------------------------------------------------------------------
# Multisample mode

def _sfz_env(t, rel_t, eg: dict, vel_amp: float) -> np.ndarray:
    d, a, h = eg.get("delay", 0.0), max(eg.get("attack", 0.0), 1e-4), eg.get("hold", 0.0)
    dec, s, r = max(eg.get("decay", 0.0), 1e-4), eg.get("sustain", 100.0) / 100.0, max(eg.get("release", 0.01), 1e-4)
    tt = t - d
    env = np.where(tt < 0, 0.0, np.where(tt < a, tt / a, np.where(tt < a + h, 1.0,
                   s + (1 - s) * np.exp(-4.6 * (tt - a - h) / dec))))
    if rel_t is not None:
        lvl = np.interp(rel_t, t, env) if len(t) else 0.0
        env = np.where(t >= rel_t, lvl * np.exp(-4.6 * (t - rel_t) / r), env)
    return env * vel_amp


def render_multisample(bank: SampleBank, cfg, mcfg, freq, bounds, vel, sr, rng, notes_pitch, a4=440.0,
                       unison=None) -> np.ndarray:
    n = len(freq)
    out = np.zeros((2, n))
    regions = parse_sfz(mcfg.sfz)
    if not regions or not bounds:
        return out
    voices = unison or [(0.0, 1.0, 0.0)]
    for k, (s0, off_i, nxt) in enumerate(bounds):
        note = float(notes_pitch[k]) if k < len(notes_pitch) else 60.0
        v = float(vel[s0]) if s0 < len(vel) else 1.0
        reg = pick_region(regions, note + mcfg.timbre, v)
        if reg is None:
            continue
        rel = reg.sample if not mcfg.folder else mcfg.folder.rstrip("/") + "/" + reg.sample
        smp = bank.get("serum:" + rel)
        if smp is None:
            continue
        data = _stereo(smp.data).astype(np.float64)
        N = data.shape[1]
        seg = slice(s0, nxt)
        t = np.arange(nxt - s0) / sr
        f = freq[seg]
        base_hz = _note_hz(reg.keycenter - reg.transpose - reg.tune / 100.0, a4)
        eg = dict(reg.ampeg)
        if mcfg.env_override and mcfg.env:
            eg = {"delay": mcfg.env.get("delay", 0.0), "attack": mcfg.env.get("attack", 0.001),
                  "hold": mcfg.env.get("hold", 0.0), "decay": mcfg.env.get("decay", 1.0),
                  "sustain": 100 * mcfg.env.get("sustain", 1.0), "release": mcfg.env.get("release", 0.1)}
        track = reg.amp_veltrack / 100.0 * mcfg.vel_track
        vel_amp = (1 - track) + track * v
        env = _sfz_env(t, (off_i - s0) / sr if off_i < nxt else None, eg, vel_amp * 10 ** (reg.volume / 20))
        for vi, (semi, g, pan) in enumerate(voices):
            rate = f / base_hz * 2 ** (semi / 12) * smp.sr / sr
            start = reg.offset + mcfg.random * rng.random() * 0.05 * N
            pos = start + np.concatenate([[0.0], np.cumsum(rate[:-1])])
            if reg.loop_mode in ("loop_continuous", "loop_sustain") and reg.loop_end > reg.loop_start + 2:
                if reg.loop_mode == "loop_sustain" and off_i < nxt:
                    gi = off_i - s0
                    pos = np.concatenate([_fold_loop(pos[:gi], reg.loop_start, reg.loop_end, "forward"),
                                          pos[gi:]])
                else:
                    pos = _fold_loop(pos, reg.loop_start, reg.loop_end, "forward")
            y = _interp(data, pos) * env
            gl, gr = _pan_gains(pan)
            out[0, seg] += g * gl * y[0]
            out[1, seg] += g * gr * y[1]
    return out / np.sqrt(sum(v[1] ** 2 for v in voices))
