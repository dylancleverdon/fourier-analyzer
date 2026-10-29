"""DSP building blocks approximating Serum 2's oscillator warps, filter types
and effects. Everything is mono (the analyzer compares mono signals).

These are behavioural models: they reproduce what each Serum module does to
the spectrum and envelope (a comb filter makes comb teeth at the cutoff
frequency, a ladder filter saturates and self-oscillates, a delay repeats,
a Bode shifter moves partials by a fixed Hz amount, ...), not bit-exact
copies of Xfer's DSP. Real Serum renders are used to check the difference.
"""
from __future__ import annotations

import numpy as np
from scipy import signal

try:
    import numba as _nb
except Exception:       # pragma: no cover
    _nb = None


def _jit(fn):
    return _nb.njit(cache=True, fastmath=True)(fn) if _nb is not None else fn


# --------------------------------------------------------------------------
# waveshapers (Serum distortion modes / Dist* warps)

def shape(x: np.ndarray, mode: str, drive: float) -> np.ndarray:
    """Waveshaper with ``drive`` 0..1 (0..36 dB of input gain)."""
    m = (mode or "SoftClip").replace("Dist", "")
    if m == "Downsample":
        k = int(1 + drive * 40)
        return np.repeat(x[::k], k)[: len(x)] if k > 1 else x
    g = 10 ** (drive * 36 / 20)
    z = x * g
    if m in ("hard", "HardClip"):
        return np.clip(z, -1, 1)
    if m in ("tube", "Tube"):
        return np.tanh(z + 0.25) - np.tanh(0.25)
    if m in ("sinefold", "SinFold"):
        return np.sin(z * np.pi / 2)
    if m in ("linfold", "LinFold"):
        return 4 * np.abs(np.mod((z + 1) / 4, 1.0) - 0.5) - 1
    if m in ("diode", "Diode1"):
        y = np.where(z > 0, np.tanh(z), 0.3 * np.tanh(z))
        return y - y.mean()
    if m == "Diode2":
        y = np.where(z > 0, 1 - np.exp(-z), -0.5 * (1 - np.exp(z)))
        return y - y.mean()
    if m == "Overdrive":
        return np.tanh(1.5 * z) * 0.9 + 0.1 * np.tanh(4 * z)
    if m == "SoftSat":
        return z / (1 + np.abs(z))
    if m == "TapeSat":
        return np.tanh(z) * 0.85 + 0.15 * np.tanh(z ** 3)
    if m in ("Asym", "XShaperAsym"):
        y = np.tanh(z + 0.4) - np.tanh(0.4)
        return y - y.mean()
    if m == "SineShaper":
        return np.sin(z)
    if m == "StompBox":
        return np.clip(1.4 * np.tanh(z), -1, 1)
    if m == "ZeroSquare":
        return np.sign(z) * np.minimum(z * z, 1.0)
    if m == "XShaper":
        zc = np.clip(z, -1.5, 1.5)
        return zc - zc ** 3 / 3.375 * 1.0
    if m == "Rectify":
        y = np.abs(z)
        return np.tanh(y - y.mean())
    return np.tanh(z)     # SoftClip / soft / default


DIST_MODE_NAMES = ("Overdrive", "SoftSat", "Diode1", "Downsample", "SoftClip", "TapeSat", "Diode2", "Asym",
                   "SineShaper", "HardClip", "ZeroSquare", "StompBox", "XShaper", "SinFold", "Rectify",
                   "LinFold", "XShaperAsym", "Tube")


# --------------------------------------------------------------------------
# filters

def filter_family(ftype: str) -> str:
    """Map Serum's ~80 filter type names onto the families implemented here."""
    t = (ftype or "L24")
    low = t.lower()
    legacy = {"lp12": "lp12", "lp24": "lp24", "hp12": "hp12", "hp24": "hp24", "bp12": "bp12"}
    if low in legacy:
        return legacy[low]
    if t.startswith(("Ladder", "DirtyMg", "ZDF")):
        return "ladder"
    if t.startswith("Comb") or t == "Combs" or t.startswith("DistComb"):
        return "comb_neg" if "N" in t[4:] else "comb"
    if t.startswith("Flange"):
        return "flange"
    if t.startswith("Phase") or t == "Allpasses":
        return "phaser"
    if t in ("Diffuser", "Reverb1"):
        return "diffuser"
    if t.startswith("Formant"):
        return "formant"
    if t in ("RM", "RMT"):
        return "ringmod"
    if t == "SNH1":
        return "downsample"
    if t == "DJMixer":
        return "dj"
    if t == "ADD_BASS":
        return "bassboost"
    if t.startswith("Scream"):
        return "scream"
    if t in ("BandReject", "N12", "NN12"):
        return "notch12"
    if t == "N24":
        return "notch24"
    if t in ("P12", "PP12", "PN12"):
        return "peak"
    if t.startswith(("HEQ",)):
        return "hp12"
    if t.startswith(("LH", "LBH", "LNH", "LPH", "LB", "LN")):
        return "morph"
    if t.startswith(("Exp", "Wsp", "BPN", "BN", "BP", "B")):
        return "bp24" if "24" in t else "bp12"
    if t.startswith(("H", "HP")):
        return {"6": "hp6", "8": "hp18", "4": "hp24"}.get(t[-1], "hp12")
    if t.startswith(("Mg", "L", "PZ")):
        return {"6": "lp6", "8": "lp18", "4": "lp24"}.get(t[-1], "lp12")
    return "lp24"


def _rbj(kind, fc, q, sr, gain_db=0.0):
    w0 = 2 * np.pi * np.clip(fc, 10, sr * 0.45) / sr
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2 * q)
    A = 10 ** (gain_db / 40)
    if kind == "lp":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "hp":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "bp":
        b = [alpha, 0.0, -alpha]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "notch":
        b = [1, -2 * cw, 1]; a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "peak":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]; a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "lowshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "highshelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    else:
        raise ValueError(kind)
    return np.array(b) / a[0], np.array(a) / a[0]


@_jit
def _tv_biquad_k(x, B, A, block):
    y = np.empty_like(x)
    z1 = 0.0
    z2 = 0.0
    for i in range(x.shape[0]):
        k = i // block
        xi = x[i]
        yi = B[k, 0] * xi + z1
        z1 = B[k, 1] * xi - A[k, 1] * yi + z2
        z2 = B[k, 2] * xi - A[k, 2] * yi
        y[i] = yi
    return y


def tv_filter(x, kind, fc, q, sr, block=64, gain_db=0.0):
    """Biquad whose cutoff (array) is updated every ``block`` samples."""
    x = np.ascontiguousarray(x, dtype=np.float64)
    fc = np.broadcast_to(np.asarray(fc, dtype=float), x.shape)
    nb = int(np.ceil(len(x) / block))
    fcb = np.clip(fc[::block][:nb], 10.0, sr * 0.45)
    if np.ptp(fcb) < 1e-3 * (fcb.mean() + 1e-9):
        b, a = _rbj(kind, float(fcb[0]), q, sr, gain_db)
        return signal.lfilter(b, a, x)
    B = np.zeros((nb, 3)); A = np.zeros((nb, 3))
    for k in range(nb):
        B[k], A[k] = _rbj(kind, float(fcb[k]), q, sr, gain_db)
    if _nb is not None:
        return _tv_biquad_k(x, B, A, block)
    y = np.empty_like(x); zi = np.zeros(2)
    for k in range(nb):
        s = slice(k * block, min((k + 1) * block, len(x)))
        y[s], zi = signal.lfilter(B[k], A[k], x[s], zi=zi)
    return y


@_jit
def _onepole_tv_k(x, coef):
    y = np.empty_like(x)
    s = 0.0
    for i in range(x.shape[0]):
        s = s + coef[i] * (x[i] - s)
        y[i] = s
    return y


def onepole_lp(x, fc, sr):
    fc = np.broadcast_to(np.asarray(fc, dtype=float), np.shape(x))
    coef = 1 - np.exp(-2 * np.pi * np.clip(fc, 5, sr * 0.45) / sr)
    if _nb is not None:
        return _onepole_tv_k(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(coef))
    y = np.empty(len(x)); s = 0.0
    for i in range(len(x)):
        s += coef[i] * (x[i] - s); y[i] = s
    return y


@_jit
def _comb_k(x, delay, fb, ff, damp):
    n = x.shape[0]
    size = 8192
    buf = np.zeros(size)
    y = np.empty(n)
    w = 0
    lp = 0.0
    for i in range(n):
        d = delay[i]
        if d < 1.0:
            d = 1.0
        if d > size - 2:
            d = size - 2.0
        rp = w - d
        while rp < 0:
            rp += size
        i0 = int(rp)
        fr = rp - i0
        i1 = (i0 + 1) % size
        dl = buf[i0] * (1 - fr) + buf[i1] * fr
        lp = lp + (1 - damp) * (dl - lp)
        out = x[i] + ff * lp
        buf[w] = x[i] + fb * lp
        y[i] = out
        w = (w + 1) % size
    return y


def comb(x, fc, reso, sr, negative=False, damp=0.0, mix=1.0):
    """Feedback/feed-forward comb tuned so its teeth sit at multiples of fc."""
    fc = np.broadcast_to(np.asarray(fc, dtype=float), np.shape(x))
    delay = sr / np.clip(fc, 20, sr * 0.45)
    sign = -1.0 if negative else 1.0
    fb = sign * float(np.clip(reso, 0, 0.98)) * 0.97
    y = _comb_k(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(delay), fb, sign * 0.5,
                float(np.clip(damp, 0, 0.95))) if _nb is not None else x
    return (1 - mix) * x + mix * y


@_jit
def _allpass_chain_k(x, coef, stages, fb):
    n = x.shape[0]
    y = np.empty(n)
    st = np.zeros(stages)
    last = 0.0
    for i in range(n):
        a = coef[i]
        v = x[i] + fb * last
        for s in range(stages):
            o = -a * v + st[s]
            st[s] = v + a * o
            v = o
        last = v
        y[i] = v
    return y


def phaser_core(x, fc, sr, stages=4, fb=0.0, mix=0.5):
    fc = np.broadcast_to(np.asarray(fc, dtype=float), np.shape(x))
    t = np.tan(np.pi * np.clip(fc, 20, sr * 0.45) / sr)
    coef = (t - 1) / (t + 1)
    if _nb is not None:
        ap = _allpass_chain_k(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(coef),
                              int(max(1, stages)), float(np.clip(fb, -0.95, 0.95)))
    else:
        ap = x
    return (1 - mix) * x + mix * 0.5 * (x + ap)


FORMANTS = np.array([[800, 1150, 2900], [400, 1600, 2700], [350, 1700, 2700],
                     [450, 800, 2830], [325, 700, 2530]], dtype=float)   # a e i o u


def formant(x, fc, var, reso, sr):
    pos = np.clip(var, 0, 1) * (len(FORMANTS) - 1)
    j = int(min(pos, len(FORMANTS) - 2))
    f = FORMANTS[j] * (1 - (pos - j)) + FORMANTS[j + 1] * (pos - j)
    shift = float(np.mean(fc)) / 1000.0
    q = 4 + reso * 12
    y = np.zeros_like(x, dtype=float)
    for k, fk in enumerate(f):
        b, a = _rbj("bp", fk * max(0.25, shift), q, sr)
        y += signal.lfilter(b, a, x) * (1.0, 0.6, 0.35)[k]
    return y


def apply_filter_type(x, ftype, cutoff, reso, drive, var, sr, mix=1.0):
    """Voice/FX filter: any Serum filter type (see filter_family)."""
    fam = filter_family(ftype)
    x = np.asarray(x, dtype=float)
    if drive > 0 and fam not in ("ladder", "scream"):
        g = 1 + 8 * drive
        x = np.tanh(x * g) / np.tanh(g)
    q = 0.707 + 14.0 * reso ** 2
    cut = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    if fam in ("lp12", "hp12", "bp12"):
        y = tv_filter(x, fam[:2], cut, q, sr)
    elif fam in ("lp24", "hp24", "bp24"):
        q1 = 0.54 + 0.5 * (q - 0.707)
        y = tv_filter(tv_filter(x, fam[:2], cut, q1, sr), fam[:2], cut, 1.31 if reso < 0.05 else q1 * 1.4, sr)
    elif fam in ("lp6", "hp6"):
        lp = onepole_lp(x, cut, sr)
        y = lp if fam == "lp6" else x - lp
    elif fam in ("lp18", "hp18"):
        y = tv_filter(x, fam[:2], cut, q, sr)
        lp = onepole_lp(y, cut, sr)
        y = lp if fam == "lp18" else y - lp
    elif fam == "ladder":
        g = 1 + 6 * drive
        xin = np.tanh(x * g) / np.tanh(g) if drive > 0 else x
        q1 = 0.6 + 18 * reso ** 2
        y = tv_filter(tv_filter(xin, "lp", cut, q1, sr), "lp", cut, 0.6 + 4 * reso, sr)
        y = np.tanh(y * (1 + reso)) / (1 + 0.5 * reso)
    elif fam in ("notch12", "notch24"):
        y = tv_filter(x, "notch", cut, 0.5 + 6 * reso, sr)
        if fam == "notch24":
            y = tv_filter(y, "notch", cut, 0.5 + 6 * reso, sr)
    elif fam == "peak":
        y = tv_filter(x, "peak", cut, 0.5 + 8 * reso, sr, gain_db=6 + 18 * var)
    elif fam == "morph":
        lp = tv_filter(x, "lp", cut, q, sr)
        hp = tv_filter(x, "hp", cut, q, sr)
        y = (1 - var) * lp + var * hp
    elif fam in ("comb", "comb_neg"):
        y = comb(x, cut, reso, sr, negative=fam == "comb_neg", damp=1 - var)
    elif fam == "flange":
        y = comb(x, cut * 4, reso, sr, negative="N" in str(ftype), damp=0.0, mix=0.5)
    elif fam == "phaser":
        stages = 12 if "48" in str(ftype) else 6 if "24" in str(ftype) else 3
        y = phaser_core(x, cut, sr, stages=stages, fb=0.9 * reso * (1 if "P" in str(ftype)[-1:] else -1), mix=1.0)
    elif fam == "diffuser":
        y = phaser_core(x, cut, sr, stages=8, fb=0.5 * reso, mix=0.5 + 0.5 * var)
    elif fam == "formant":
        y = formant(x, cut, var, reso, sr)
    elif fam == "ringmod":
        ph = np.cumsum(cut / sr)
        y = x * np.sin(2 * np.pi * ph)
    elif fam == "downsample":
        hold = np.maximum(1, (sr / np.clip(cut, 50, sr)).astype(int))
        k = int(np.median(hold))
        y = np.repeat(x[::k], k)[: len(x)]
    elif fam == "dj":
        y = tv_filter(x, "lp" if var < 0.5 else "hp", cut, q, sr)
    elif fam == "bassboost":
        y = tv_filter(x, "lowshelf", cut, 0.7, sr, gain_db=6 + 12 * var)
    elif fam == "scream":
        y = np.tanh(3 * (1 + 4 * drive) * tv_filter(x, "bp", cut, 1 + 10 * reso, sr)) * 0.5
    else:
        y = tv_filter(x, "lp", cut, q, sr)
    return (1 - mix) * x + mix * y if mix < 1 else y


# --------------------------------------------------------------------------
# effects

@_jit
def _delay_k(x, d1, d2, fb, bp_b, bp_a):
    n = x.shape[0]
    size = 1
    while size < max(d1, d2) + 4:
        size *= 2
    b1 = np.zeros(size)
    b2 = np.zeros(size)
    y = np.empty(n)
    w = 0
    z1 = 0.0
    z2 = 0.0
    for i in range(n):
        r1 = b1[(w - d1) % size]
        r2 = b2[(w - d2) % size]
        tap = 0.5 * (r1 + r2)
        f = bp_b[0] * tap + z1                       # bandpass in the feedback path
        z1 = bp_b[1] * tap - bp_a[1] * f + z2
        z2 = bp_b[2] * tap - bp_a[2] * f
        b1[w] = x[i] + fb * f
        b2[w] = x[i] + fb * f
        y[i] = tap
        w = (w + 1) % size
    return y


def delay_fx(x, p, sr):
    d1 = int(np.clip(p.get("time", 0.25), 0.001, 2.0) * sr)
    d2 = int(np.clip(p.get("time2", p.get("time", 0.25)), 0.001, 2.0) * sr)
    q = max(0.3, 4.0 / max(p.get("bw", 3.0), 0.3))
    b, a = _rbj("bp", p.get("freq", 2000.0), q, sr)
    fb = float(np.clip(p.get("feedback", 0.35), 0, 0.97))
    if _nb is not None:
        wet = _delay_k(np.ascontiguousarray(x, dtype=np.float64), d1, d2, fb, b, a)
    else:
        wet = np.concatenate([np.zeros(d1), x[:-d1]]) if d1 < len(x) else np.zeros_like(x)
    mix = p.get("mix", 0.25)
    return (1 - mix) * x + mix * wet * 1.5


@_jit
def _fdn_k(x, delays, g, damp):
    n = x.shape[0]
    m = delays.shape[0]
    size = 1
    while size < delays.max() + 4:
        size *= 2
    buf = np.zeros((m, size))
    lp = np.zeros(m)
    y = np.empty(n)
    w = 0
    outs = np.zeros(m)
    for i in range(n):
        s = 0.0
        for j in range(m):
            v = buf[j, (w - delays[j]) % size]
            lp[j] = lp[j] + (1 - damp) * (v - lp[j])
            outs[j] = lp[j] * g[j]
            s += outs[j]
        y[i] = s / m
        # Householder feedback matrix
        tot = 0.0
        for j in range(m):
            tot += outs[j]
        tot *= 2.0 / m
        for j in range(m):
            buf[j, w] = x[i] + outs[j] - tot
        w = (w + 1) % size
    return y


REVERB_KINDS = {"Hall": (1.0, 0.45), "Vintage": (0.7, 0.65), "Abyss": (1.6, 0.25), "Space": (1.3, 0.35),
                "Plate": (0.6, 0.3), "Room": (0.4, 0.5)}


def reverb_fx(x, p, sr):
    kind = str(p.get("kind", "Hall"))
    if kind.startswith("k") and kind[1:2].isupper():
        kind = kind[1:]
    scale, base_damp = REVERB_KINDS.get(kind, (1.0, 0.45))
    size = 0.3 + 1.2 * float(np.clip(p.get("size", 0.5), 0, 1))
    base = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], dtype=float) * size * scale * sr / 44100
    delays = base.astype(np.int64)
    rt60 = max(0.1, float(p.get("decay", 1.5)))
    g = 10 ** (-3 * (delays / sr) / rt60)
    damp = float(np.clip(base_damp * 0.5 + 0.5 * p.get("damping", 0.5), 0, 0.95))
    pre = int(max(0.0, p.get("predelay", 0.0)) * sr)
    xin = np.concatenate([np.zeros(pre), x])[: len(x)] if pre else np.asarray(x, dtype=float)
    wet = _fdn_k(np.ascontiguousarray(xin, dtype=np.float64), delays, g, damp) if _nb is not None else xin
    mix = p.get("mix", 0.2)
    return (1 - mix) * x + mix * wet * 2.0


def convolve_fx(x, p, sr):
    """Serum's convolver with a synthetic room IR (decay, size, tone)."""
    dec = float(np.clip(p.get("decay", 1.0), 0.05, 6.0))
    n = int(min(4.0, dec * 1.5) * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    ir = rng.standard_normal(n) * np.exp(-6.9 * t / dec)
    tone = float(p.get("tone", 0.0))
    if tone < 0:
        ir = onepole_lp(ir, 20000 * 2 ** (tone * 6), sr)
    size = float(np.clip(p.get("size", 0.5), 0.05, 1))
    ir[: int(0.002 * sr)] *= size
    pre = int(max(0.0, p.get("predelay", 0.0)) * sr)
    ir = np.concatenate([np.zeros(pre), ir])
    ir /= np.sqrt(np.sum(ir ** 2)) + 1e-9
    wet = signal.fftconvolve(x, ir)[: len(x)]
    mix = p.get("mix", 0.2)
    return (1 - mix) * x + mix * wet


@_jit
def _moddelay_k(x, d, fb):
    n = x.shape[0]
    size = 8192
    buf = np.zeros(size)
    y = np.empty(n)
    w = 0
    last = 0.0
    for i in range(n):
        rp = w - d[i]
        while rp < 0:
            rp += size
        i0 = int(rp)
        fr = rp - i0
        v = buf[i0 % size] * (1 - fr) + buf[(i0 + 1) % size] * fr
        buf[w] = x[i] + fb * v
        y[i] = v
        last = v
        w = (w + 1) % size
    return y


def _mod_delay(x, base_ms, depth_ms, rate, phase, fb, sr):
    t = np.arange(len(x)) / sr
    d = (base_ms + depth_ms * (0.5 + 0.5 * np.sin(2 * np.pi * (rate * t + phase)))) * sr / 1000
    if _nb is not None:
        return _moddelay_k(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(d), float(fb))
    idx = np.arange(len(x)) - d
    return np.interp(idx, np.arange(len(x)), x, left=0)


def chorus_fx(x, p, sr):
    base = p.get("delay_ms", 7.0)
    depth = 5.0 * p.get("depth", 0.3)
    wet = 0.5 * (_mod_delay(x, base, depth, p.get("rate_hz", 0.8), 0.0, p.get("feedback", 0.0), sr)
                 + _mod_delay(x, base * 1.3, depth, p.get("rate_hz", 0.8), 0.5, p.get("feedback", 0.0), sr))
    mix = p.get("mix", 0.3)
    return (1 - mix) * x + mix * wet


def flanger_fx(x, p, sr):
    wet = _mod_delay(x, 0.5, 4.0 * p.get("depth", 0.5), p.get("rate_hz", 0.3), 0.0,
                     0.95 * p.get("feedback", 0.5), sr)
    mix = p.get("mix", 0.5)
    return (1 - mix) * x + mix * 0.5 * (x + wet)


def phaser_fx(x, p, sr):
    t = np.arange(len(x)) / sr
    f0 = p.get("freq", 800.0)
    fc = f0 * 2 ** (2.5 * p.get("depth", 0.5) * np.sin(2 * np.pi * p.get("rate_hz", 0.3) * t))
    return phaser_core(x, fc, sr, stages=int(p.get("stages", 4)), fb=0.9 * p.get("feedback", 0.4),
                       mix=p.get("mix", 0.5))


def hyperdim_fx(x, p, sr):
    v = int(np.clip(p.get("voices", 4), 1, 7))
    wet = np.zeros_like(x, dtype=float)
    for k in range(v):
        wet += _mod_delay(x, 3 + 9 * p.get("dim_size", 0.5) * (k + 1) / v, 2.5 * p.get("detune", 0.3),
                          0.2 + 2 * p.get("rate", 0.4) * (1 + 0.3 * k), k / v, 0.0, sr)
    wet /= v
    mix = p.get("mix", 0.4)
    return (1 - mix) * x + mix * wet


def bode_fx(x, p, sr):
    a = signal.hilbert(x)
    t = np.arange(len(x)) / sr
    wet = np.real(a * np.exp(2j * np.pi * p.get("shift_hz", 50.0) * t))
    mix = p.get("mix", 0.5)
    return (1 - mix) * x + mix * wet


def utility_fx(x, p, sr):
    y = np.asarray(x, dtype=float)
    if p.get("hpf", 0) > 0:
        y = tv_filter(y, "hp", p["hpf"], 0.707, sr)
    if p.get("lpf", 0) > 0:
        y = tv_filter(y, "lp", p["lpf"], 0.707, sr)
    y = y * 10 ** (p.get("gain_db", 0.0) / 20)
    return -y if p.get("invert") else y


def _env_follow(x2, sr, att, rel):
    a = np.exp(-1 / (max(att, 1e-4) * sr))
    r = np.exp(-1 / (max(rel, 1e-3) * sr))
    fast = signal.lfilter([1 - a], [1, -a], x2)
    slow = signal.lfilter([1 - r], [1, -r], x2)
    return np.sqrt(np.maximum(np.maximum(fast * 0.5, slow), 1e-12))


def comp_fx(x, p, sr):
    peak = np.max(np.abs(x)) + 1e-9
    xn = x / peak * 0.5
    if p.get("multiband", True):
        lo = signal.sosfilt(signal.butter(4, 88, "low", fs=sr, output="sos"), xn)
        hi = signal.sosfilt(signal.butter(4, min(2500, sr * 0.45), "high", fs=sr, output="sos"), xn)
        bands = (lo, xn - lo - hi, hi)
        out = np.zeros_like(xn)
        for band in bands:
            e = 20 * np.log10(_env_follow(band ** 2, sr, 0.003, 0.08))
            down = -np.maximum(0, e + 20) * (1 - 1 / 8.0)
            up = np.maximum(0, -40 - e) * (1 - 1 / 3.0)
            g = np.clip(p.get("amount", 0.6) * (down + np.minimum(up, 24)), -40, 24)
            out += band * 10 ** (g / 20)
        depth = p.get("depth", 0.5)
        return ((1 - depth) * xn + depth * out) * peak * 2
    e = 20 * np.log10(_env_follow(xn ** 2, sr, p.get("attack_ms", 5) / 1000, p.get("release_ms", 80) / 1000))
    over = np.maximum(0, e - p.get("threshold_db", -18.0))
    g = -over * (1 - 1 / max(1.0, p.get("ratio", 4.0))) + p.get("makeup_db", 0.0)
    return xn * 10 ** (g / 20) * peak * 2


def eq_fx(x, p, sr):
    y = np.asarray(x, dtype=float)
    for k in (1, 2):
        gain = p.get(f"gain{k}", 0.0)
        kind = p.get(f"type{k}", "peak")
        if abs(gain) < 0.05 and kind in ("lowshelf", "highshelf", "peak"):
            continue
        if kind in ("lp", "hp"):
            b, a = _rbj(kind, p.get(f"freq{k}", 1000.0), max(0.3, p.get(f"q{k}", 0.7)), sr)
        else:
            b, a = _rbj(kind, p.get(f"freq{k}", 1000.0), max(0.3, p.get(f"q{k}", 0.7)), sr, gain)
        y = signal.lfilter(b, a, y)
    return y


def distortion_fx(x, p, sr, drive_mod=None):
    pre = p.get("filter", "off")
    xin = np.asarray(x, dtype=float)
    if pre == "pre":
        xin = tv_filter(xin, "bp", 20 * 1000 ** p.get("filter_freq", 0.5), 0.5 + 2 * p.get("filter_bw", 0.5), sr)
    drive = p.get("drive", 0.3)
    if drive_mod is not None and np.ndim(drive_mod):
        d = np.clip(drive + drive_mod, 0, 1)
        y = np.tanh(xin * 10 ** (d * 36 / 20)) if p.get("mode") in (None, "SoftClip") else \
            shape(xin, p.get("mode"), float(np.mean(d)))
    else:
        y = shape(xin, p.get("mode", "Overdrive"), drive)
    if pre == "post":
        y = tv_filter(y, "bp", 20 * 1000 ** p.get("filter_freq", 0.5), 0.5 + 2 * p.get("filter_bw", 0.5), sr)
    mix = p.get("mix", 1.0)
    return (1 - mix) * x + mix * y


def filter_fx(x, p, sr):
    return apply_filter_type(x, p.get("ftype", "L24"), p.get("cutoff", 2000.0), p.get("reso", 0.2),
                             p.get("drive", 0.0), p.get("var", 0.5), sr, mix=p.get("mix", 1.0))


FX_FUNCS = {"delay": delay_fx, "reverb": reverb_fx, "convolve": convolve_fx, "chorus": chorus_fx,
            "flanger": flanger_fx, "phaser": phaser_fx, "hyperdim": hyperdim_fx, "bode": bode_fx,
            "utility": utility_fx, "comp": comp_fx, "eq": eq_fx, "distortion": distortion_fx,
            "filter": filter_fx}


def process_chain(x: np.ndarray, chain: list, sr: int, mods: dict | None = None, prefix: str = "fx") -> np.ndarray:
    """Run an FX rack (list of module dicts, splitters nest sub-chains).
    ``mods`` maps "fx.<i>.mix" / "fx.<i>.drive" to per-sample modulation."""
    y = np.asarray(x, dtype=float)
    mods = mods or {}
    for i, m in enumerate(chain or []):
        if not m.get("enabled", True):
            continue
        kind = m.get("type")
        key = f"{prefix}.{i}"
        if kind in ("split", "split3", "splitms"):
            y = _split(y, m, sr, mods, key)
            continue
        fn = FX_FUNCS.get(kind)
        if fn is None:
            continue
        before = y
        if kind == "distortion" and f"{key}.drive" in mods:
            y = distortion_fx(y, m, sr, drive_mod=mods[f"{key}.drive"])
        else:
            y = fn(y, m, sr)
        if f"{key}.mix" in mods:          # modulated wet/dry on top of the module's own mix
            w = np.clip(mods[f"{key}.mix"], -1, 1)
            y = before + (y - before) * np.clip(1 + w, 0, 2)
        if not np.all(np.isfinite(y)):
            y = before
    return y


def _split(x, m, sr, mods, key):
    bands = m.get("bands") or []
    kind = m["type"]
    if kind == "splitms":                 # mono signal: everything is "mid"
        return process_chain(x, bands[0] if bands else [], sr, mods, key + ".b0")
    if kind == "split":
        f = m.get("freq", 300.0)
        lo = signal.sosfiltfilt(signal.butter(4, f, "low", fs=sr, output="sos"), x)
        parts = [lo, x - lo]
    else:
        f1, f2 = m.get("freq", 200.0), m.get("freq2", 2000.0)
        lo = signal.sosfiltfilt(signal.butter(4, f1, "low", fs=sr, output="sos"), x)
        hi = signal.sosfiltfilt(signal.butter(4, min(f2, sr * 0.45), "high", fs=sr, output="sos"), x)
        parts = [lo, x - lo - hi, hi]
    out = np.zeros_like(x, dtype=float)
    for b, part in enumerate(parts):
        chain = bands[b] if b < len(bands) else []
        out += process_chain(part, chain, sr, mods, f"{key}.b{b}")
    return out
