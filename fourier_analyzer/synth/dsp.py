"""DSP building blocks approximating Serum 2's oscillator warps, filter types
and effects. Filters are mono (the engine runs one per channel); effects take
(channels, n) arrays and are stereo where Serum's are (ping-pong delay, chorus
and flanger stereo phase, reverb width, M/S splitter, utility width ...).

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




def _mean(v) -> float:
    return float(np.mean(v)) if np.ndim(v) else float(v)


def _fat(x, var):
    """Serum's FAT knob: saturation in the resonance path (approximated on the
    filter input/output)."""
    if _mean(var) <= 0.01:
        return x
    g = 1 + 4 * np.asarray(var)
    return np.tanh(x * g) / np.tanh(g)


# --------------------------------------------------------------------------
# filter families (every Serum 2 VoiceFilter / FX Filter type name)

def filter_family(ftype: str) -> str:
    """Map Serum's filter type names onto the models implemented here."""
    t = (ftype or "L24")
    low = t.lower()
    legacy = {"lp12": "lp12", "lp24": "lp24", "hp12": "hp12", "hp24": "hp24", "bp12": "bp12"}
    if low in legacy:
        return legacy[low]
    table = {
        "LadderMg": "ladder", "LadderAcid": "acid", "LadderEMS": "ems", "DirtyMg": "dirty", "ZDF_A": "ladder",
        "PZ_SVF": "pz", "Comb2": "comb2", "Exp": "expmm", "ExpBPF": "bp12", "Wsp": "wsp", "DJMixer": "dj",
        "Diffuser": "diffuser", "Reverb1": "reverb1", "Combs": "combs", "Allpasses": "allpasses",
        "ADD_BASS": "bassboost", "BandReject": "bandreject", "RM": "ringmod", "RMT": "ringmod2",
        "SNH1": "snh", "SNH2": "snh_minus", "Scream": "scream_lp", "Scream3LP": "scream_lp",
        "Scream3BP": "scream_bp", "FormantONE": "formant1", "FormantTWO": "formant2", "FormantTWB": "formant3",
        "DistComb1LP": "distcomb1_lp", "DistComb1BP": "distcomb1_bp", "DistComb2LP": "distcomb2_lp",
        "DistComb2BP": "distcomb2_bp", "FrenchLP": "french", "GermanLP": "ladder",
        "B12": "bp12", "B24": "bp24", "P12": "peak12", "N12": "notch12", "N24": "notch24",
        "HEQ6": "heq6", "HEQ12": "heq12", "LEQ6": "leq6", "LEQ12": "leq12", "BEQ6": "beq6", "BEQ12": "beq12",
    }
    if t in table:
        return table[t]
    if t.startswith("MgL"):
        return "mg" + t[3:]                                   # mg6/12/18/24
    if t[:1] in "LH" and t[1:].isdigit():
        return ("lp" if t[0] == "L" else "hp") + t[1:]        # lp6 ... hp24 (SVF)
    if t in ("LH12", "LB12", "LP12", "LN12", "HB12", "HP12", "HN12", "BP12", "BN12", "PP12", "PN12", "NN12"):
        return "dual_" + t[:2].lower()
    if t[:3] in ("LBH", "LPH", "LNH", "BPN"):
        return "morph_" + t[:3].lower() + ("24" if t.endswith("24") else "12")
    for pre, fam in (("Comb", "comb"), ("Flange", "flange"), ("Phase", "phase")):
        if t.startswith(pre):
            rest = t[len(pre):]
            neg = rest.endswith("N")
            tone = "hl" if "HL" in rest else "h" if "H" in rest else "l" if "L" in rest else ""
            stages = 0
            for d in ("48", "36", "24", "12"):
                if rest.startswith(d) or ("Phase" + d) in t:
                    stages = int(d)
                    break
            return f"{fam}{'_' + tone if tone else ''}{'_neg' if neg else ''}" + (f"@{stages}" if stages else "")
    if low.startswith("h"):
        return "hp12"
    return "lp24"


def _svf(x, kind, cut, reso, sr, order):
    """State-variable style low/high/band/notch/peak filter of 6..24 dB/oct."""
    q = 0.707 + 14.0 * reso ** 2
    if kind in ("lp", "hp") and order == 6:
        lp = onepole_lp(x, cut, sr)
        return lp if kind == "lp" else x - lp
    if order <= 12:
        return tv_filter(x, kind, cut, q, sr)
    if order == 18:
        y = tv_filter(x, kind, cut, q, sr)
        lp = onepole_lp(y, cut, sr)
        return lp if kind == "lp" else (y - lp if kind == "hp" else lp)
    q1 = 0.54 + 0.5 * (q - 0.707)
    return tv_filter(tv_filter(x, kind, cut, q1, sr), kind, cut, 1.31 if reso < 0.05 else q1 * 1.4, sr)


def _ladder(x, cut, reso, drive, sr, flavour="mg", var=0.0):
    g = 1 + 6 * drive
    xin = np.tanh(x * g) / np.tanh(g) if drive > 0 else x
    if flavour == "dirty":                       # PAIN: more distortion everywhere
        p = 1 + 8 * var
        xin = np.tanh(xin * p) / np.tanh(p)
    q1 = 0.6 + 18 * reso ** 2
    c = cut
    if var and flavour in ("mg", "acid", "ems"):  # SMOOTH: slower cutoff changes
        c = onepole_lp(np.broadcast_to(np.asarray(cut, float), x.shape).copy(), 2 + 200 * (1 - var), sr)
    if flavour == "acid":                        # diode ladder: 18 dB, squelchier resonance
        y = tv_filter(xin, "lp", c, q1 * 1.3, sr)
        y = onepole_lp(y, c, sr)
    elif flavour == "ems":
        y = tv_filter(tv_filter(xin, "lp", c, 0.5 + 10 * reso ** 2, sr), "lp", c, 0.6, sr)
    else:
        y = tv_filter(tv_filter(xin, "lp", c, q1, sr), "lp", c, 0.6 + 4 * reso, sr)
    return np.tanh(y * (1 + reso)) / (1 + 0.5 * reso)


def _pz_filter(x, pz, cut, sr):
    """PZ SVF (drawable): eight sections per shape; uses the first shape,
    with section frequencies scaled by the cutoff knob."""
    if not pz:
        return x
    shape0 = pz[0]
    fp = shape0.get("fp", [])
    qp = shape0.get("qp", [0.5] * len(fp))
    qz = shape0.get("qz", [0.5] * len(fp))
    types = shape0.get("types", [2] * len(fp))
    scale = _mean(cut) / 1000.0
    y = np.asarray(x, dtype=float)
    for f, q1, q2, t in zip(fp, qp, qz, types):
        hz = float(np.clip(20 * 1000 ** f * scale, 20, sr * 0.45))
        q = 0.3 + 6 * q1
        if t == 0:
            y = tv_filter(y, "hp", hz, q, sr)
        elif t == 1:
            y = tv_filter(y, "lp", hz, q, sr)
        elif t == 3:
            y = tv_filter(y, "bp", hz, q, sr)
        elif t == 4:
            y = tv_filter(y, "notch", hz, q, sr)
        elif t in (5, 6, 7):
            y = tv_filter(y, "peak", hz, q, sr, gain_db=(q1 - q2) * 24)
    return y * 2 * shape0.get("gain", 0.5)


def apply_filter_type(x, ftype, cutoff, reso, drive, var, sr, mix=1.0, pz=None, clean=False):
    """Voice/FX filter: any Serum filter type (see filter_family). ``cutoff``
    may be an array (modulation); ``var`` is the type's variable knob 0..1."""
    fam = filter_family(ftype)
    x = np.asarray(x, dtype=float)
    dry = x
    drive = _mean(drive)
    var_a = var
    var = _mean(var)
    if clean:
        x = x * 10 ** (-24 / 20)
    if drive > 0 and not fam.startswith(("ladder", "acid", "ems", "dirty", "scream", "mg")):
        g = 1 + 8 * drive
        x = np.tanh(x * g) / np.tanh(g)
    cut = np.broadcast_to(np.asarray(cutoff, dtype=float), x.shape)
    q = 0.707 + 14.0 * reso ** 2
    if fam in ("lp12", "hp12", "bp12", "lp24", "hp24", "bp24", "lp6", "hp6", "lp18", "hp18"):
        y = _svf(x, fam[:2], cut, reso, sr, int(fam[2:]))
        y = _fat(y, var) if fam[:2] in ("lp", "hp") and var > 0.01 and ftype not in ("lp12", "lp24", "hp12", "hp24", "bp12") else y
    elif fam.startswith("mg"):
        order = int(fam[2:]) if fam[2:].isdigit() else 24
        y = _ladder(x, cut, reso, drive, sr) if order >= 24 else _svf(x, "lp", cut, reso, sr, order)
        y = _fat(y, var)
    elif fam in ("ladder", "acid", "ems", "dirty"):
        y = _ladder(x, cut, reso, drive, sr, {"ladder": "mg"}.get(fam, fam), var)
    elif fam in ("notch12", "notch24", "bandreject"):
        qn = 0.5 + 6 * reso if fam != "bandreject" else 0.3 + 4 * (1 - var)
        y = tv_filter(x, "notch", cut, qn, sr)
        if fam == "notch24":
            y = tv_filter(y, "notch", cut, qn, sr)
    elif fam == "peak12":
        y = tv_filter(x, "peak", cut, 0.5 + 8 * reso, sr, gain_db=6 + 18 * var)
    elif fam.startswith("dual_"):
        a, b = fam[5], fam[6]
        kinds = {"l": "lp", "h": "hp", "b": "bp", "n": "notch", "p": "peak"}
        cut2 = cut * 2 ** ((var - 0.5) * 8)       # VAR = second filter's cutoff
        y1 = tv_filter(x, kinds[a], cut, q, sr, gain_db=12 if a == "p" else 0)
        y = tv_filter(y1, kinds[b], cut2, q, sr, gain_db=12 if b == "p" else 0)
    elif fam.startswith("morph_"):
        names = {"lbh": ("lp", "bp", "hp"), "lph": ("lp", "peak", "hp"), "lnh": ("lp", "notch", "hp"),
                 "bpn": ("bp", "peak", "notch")}[fam[6:9]]
        order = 24 if fam.endswith("24") else 12
        outs = [_svf(x, k, cut, reso, sr, order) if k in ("lp", "hp", "bp") else
                tv_filter(x, k, cut, 0.5 + 6 * reso, sr, gain_db=12 if k == "peak" else 0) for k in names]
        m = np.clip(var_a, 0, 1) * 2
        y = np.where(m < 1, outs[0] * (1 - m) + outs[1] * m, outs[1] * (2 - m) + outs[2] * (m - 1))
    elif fam.startswith(("comb", "flange", "phase")) and not fam.startswith("combs"):
        base, _, stages = fam.partition("@")
        neg = base.endswith("_neg")
        tone = base.split("_")[1] if "_" in base and base.split("_")[1] in ("l", "h", "hl") else ""
        if base.startswith("comb"):
            y = comb(x, cut, reso, sr, negative=neg, damp=0.0)
        elif base.startswith("flange"):
            y = comb(x, cut * 4, reso, sr, negative=neg, damp=0.0, mix=0.5)
        else:
            st = int(stages) // 4 if stages else 3
            y = phaser_core(x, cut, sr, stages=max(st, 1), fb=0.9 * reso * (-1 if neg else 1), mix=1.0)
        if tone:                                  # LP/HP inside the feedback loop: VAR = its cutoff
            fc2 = 20 * 1000 ** var
            if "l" in tone:
                y = onepole_lp(y, fc2 if tone == "l" else max(fc2, _mean(cut)), sr)
            if "h" in tone:
                y = y - onepole_lp(y, fc2 if tone == "h" else min(fc2, _mean(cut)) * 0.5, sr)
    elif fam == "comb2":
        y = comb(comb(x, cut, reso, sr), cut * 2 ** (var * 4), reso * 0.8, sr)
    elif fam == "combs":
        y = 0.5 * (comb(x, cut, reso, sr, damp=var) + comb(x, cut * 1.37, reso, sr, damp=var))
    elif fam in ("allpasses", "reverb1"):
        y = phaser_core(x, cut, sr, stages=6, fb=0.6 * reso, mix=0.5 + 0.5 * var)
        if fam == "reverb1":
            y = 0.5 * (y + comb(y, cut * 0.25, 0.7 * reso + 0.2, sr, damp=var))
    elif fam == "diffuser":
        y = phaser_core(x, cut, sr, stages=int(1 + 15 * var), fb=0.5 * reso, mix=1.0)
    elif fam in ("formant1", "formant2", "formant3"):
        y = formant(x, cut, var, reso, sr)
        if fam == "formant3":
            y = 0.5 * (y + formant(x, cut * 1.5, 1 - var, reso, sr))
    elif fam in ("ringmod", "ringmod2"):
        ph = np.cumsum(cut / sr)
        y = x * np.sin(2 * np.pi * ph)
        if fam == "ringmod2":
            y = 0.5 * (y + x * np.sin(2 * np.pi * ph * (1 + 3 * var)))
    elif fam in ("snh", "snh_minus"):
        k = int(np.clip(sr / max(_mean(cut), 50), 1, 4096))
        held = np.repeat(x[::k], k)[: len(x)]
        y = held if fam == "snh" else held - x
    elif fam == "dj":
        y = tv_filter(x, "lp" if var < 0.5 else "hp", cut, q, sr)
    elif fam == "bassboost":
        lp = tv_filter(x, "lp", cut, 0.707 + reso, sr)
        y = np.tanh(1.5 * lp) + var * tv_filter(x, "hp", cut, 0.707, sr)      # THRU adds phase-rotated dry
    elif fam in ("scream_lp", "scream_bp"):
        fb_cut = cut * 2 ** ((var - 0.5) * 4)
        core = tv_filter(x, "lp" if fam == "scream_lp" else "bp", cut, 1 + 10 * reso, sr)
        y = np.tanh(3 * (1 + 4 * drive) * tv_filter(core, "bp", fb_cut, 1 + 6 * reso, sr)) * 0.5 + 0.5 * core
    elif fam.startswith("distcomb"):
        neg = fam.startswith("distcomb2")
        y = comb(x, 20 * 1000 ** var, 0.85, sr, negative=neg)
        y = tv_filter(np.tanh(y * (1 + 3 * drive)), "lp" if fam.endswith("lp") else "bp", cut, q, sr)
    elif fam == "wsp":
        lp = tv_filter(x, "lp", cut, q, sr)
        hp = tv_filter(x, "hp", cut, q, sr)
        nt = tv_filter(x, "notch", cut, 0.5 + 6 * reso, sr)
        m = np.clip(var_a, 0, 1) * 2
        y = np.where(m < 1, lp * (1 - m) + nt * m, nt * (2 - m) + hp * (m - 1))
        y = np.tanh(1.3 * y)
    elif fam == "expmm":
        lp = _svf(x, "lp", cut, reso, sr, 24)
        nt = tv_filter(x, "notch", cut, 0.5 + 6 * reso, sr)
        hp = _svf(x, "hp", cut, reso, sr, 24)
        m = np.clip(var_a, 0, 1) * 2
        y = np.where(m < 1, lp * (1 - m) + nt * m, nt * (2 - m) + hp * (m - 1))
    elif fam == "french":
        y = _ladder(x, cut, max(reso, var), drive, sr)
    elif fam in ("leq6", "leq12", "heq6", "heq12", "beq6", "beq12"):
        gain = (var - 0.5) * 48                   # DB +/-
        kind = {"l": "lowshelf", "h": "highshelf", "b": "peak"}[fam[0]]
        y = tv_filter(x, kind, cut, 0.707 + (reso if fam.endswith("12") else 0), sr, gain_db=gain)
    elif fam == "pz":
        y = _pz_filter(x, pz, cut, sr)
    else:
        y = tv_filter(x, "lp", cut, q, sr)
    if clean:
        y = y * 10 ** (24 / 20)
    if np.ndim(mix) or mix < 1:
        y = (1 - np.asarray(mix)) * dry + np.asarray(mix) * y
    return y


FILTER_TYPES_ALL = (
    "MgL6", "MgL12", "MgL18", "MgL24", "L6", "L12", "L18", "L24", "H6", "H12", "H18", "H24", "B12", "B24",
    "P12", "N12", "N24", "LH12", "LB12", "LP12", "LN12", "HB12", "HP12", "HN12", "BP12", "BN12", "PP12", "PN12",
    "NN12", "LBH12", "LBH24", "LPH12", "LPH24", "LNH12", "LNH24", "BPN12", "BPN24", "CombL6P", "CombL6N",
    "CombH6P", "CombH6N", "CombHL6P", "CombHL6N", "CombP", "CombN", "FlangeL6P", "FlangeL6N", "FlangeH6P",
    "FlangeHL6P", "FlangeHL6N", "FlangeP", "FlangeN", "Phase12P", "Phase12N", "Phase24P", "Phase24N",
    "Phase36P", "Phase36N", "Phase48P", "Phase48N", "Phase48H6P", "Phase48H6N", "Phase48HL6P", "Phase48HL6N",
    "FlangePhase12HL6P", "LEQ6", "LEQ12", "BEQ6", "BEQ12", "HEQ6", "HEQ12", "RM", "RMT", "SNH1", "SNH2",
    "Combs", "Allpasses", "Reverb1", "FrenchLP", "ZDF_A", "ADD_BASS", "FormantONE", "FormantTWO", "FormantTWB",
    "BandReject", "DistComb1LP", "DistComb1BP", "DistComb2LP", "DistComb2BP", "Scream", "Scream3LP",
    "Scream3BP", "Wsp", "DJMixer", "Diffuser", "LadderMg", "LadderAcid", "LadderEMS", "DirtyMg", "PZ_SVF",
    "Comb2", "Exp", "ExpBPF")


# --------------------------------------------------------------------------
# effects. Every effect takes x with shape (C, n) and a param dict whose
# values may be arrays (modulated parameters).

def _ch(x):
    x = np.asarray(x, dtype=float)
    return x[None, :] if x.ndim == 1 else x


def _mix(dry, wet, mix):
    m = np.asarray(mix, dtype=float)
    return dry * (1 - m) + wet * m


def _per_channel(fn, x, *a, **kw):
    return np.vstack([fn(c, *a, **kw) for c in x])


def _ms(x):
    if x.shape[0] < 2:
        return x[0], np.zeros_like(x[0])
    return 0.5 * (x[0] + x[1]), 0.5 * (x[0] - x[1])


def _from_ms(m, s, C):
    if C < 2:
        return m[None, :]
    return np.vstack([m + s, m - s])


@_jit
def _stereo_delay_k(xl, xr, dl, dr, fb, mode, bp_b, bp_a):
    n = xl.shape[0]
    size = 1
    while size < max(dl, dr) + 4:
        size *= 2
    bl = np.zeros(size)
    br = np.zeros(size)
    yl = np.empty(n)
    yr = np.empty(n)
    w = 0
    z1l = z2l = z1r = z2r = 0.0
    for i in range(n):
        rl = bl[(w - dl) % size]
        rr = br[(w - dr) % size]
        fl = bp_b[0] * rl + z1l                       # bandpass in the feedback path
        z1l = bp_b[1] * rl - bp_a[1] * fl + z2l
        z2l = bp_b[2] * rl - bp_a[2] * fl
        fr = bp_b[0] * rr + z1r
        z1r = bp_b[1] * rr - bp_a[1] * fr + z2r
        z2r = bp_b[2] * rr - bp_a[2] * fr
        if mode == 1:                                 # ping-pong: each side feeds the other
            bl[w] = xl[i] + fb * fr
            br[w] = xr[i] * 0.0 + fb * fl
        elif mode == 2:                               # tap -> delay: left tap once into the right delay
            bl[w] = xl[i]
            br[w] = fl + fb * fr
        else:
            bl[w] = xl[i] + fb * fl
            br[w] = xr[i] + fb * fr
        yl[i] = rl
        yr[i] = rr
        w = (w + 1) % size
    return yl, yr


def delay_fx(x, p, sr):
    x = _ch(x)
    C = x.shape[0]
    t1 = _mean(p.get("time", 0.25)) * _mean(p.get("offset_l", 1.0))
    t2 = (t1 if p.get("link") else _mean(p.get("time2", p.get("time", 0.25)))) * _mean(p.get("offset_r", 1.0))
    d1 = int(np.clip(t1, 0.001, 4.0) * sr)
    d2 = int(np.clip(t2, 0.001, 4.0) * sr)
    q = max(0.3, 4.0 / max(_mean(p.get("bw", 3.0)), 0.3))
    b, a = _rbj("bp", _mean(p.get("freq", 2000.0)), q, sr)
    fb = float(np.clip(_mean(p.get("feedback", 0.35)), 0, 0.98))
    mode = {"normal": 0, "pingpong": 1, "tap": 2}.get(p.get("mode", "normal"), 0)
    xl, xr = (x[0], x[1]) if C > 1 else (x[0], x[0])
    if mode == 1 and C == 1:
        xr = xl
    if _nb is not None:
        yl, yr = _stereo_delay_k(np.ascontiguousarray(xl), np.ascontiguousarray(xr), d1, d2, fb, mode, b, a)
    else:
        yl = np.concatenate([np.zeros(d1), xl[:-d1]])[: len(xl)]
        yr = np.concatenate([np.zeros(d2), xr[:-d2]])[: len(xr)]
    wet = np.vstack([yl, yr]) * 1.5 if C > 1 else 0.75 * (yl + yr)[None, :]
    return _mix(x, wet, p.get("mix", 0.25))


REVERB_KINDS = {"Hall": (1.0, 0.45), "Vintage": (0.7, 0.65), "Abyss": (1.6, 0.25), "Basin": (1.6, 0.25),
                "Space": (1.3, 0.35), "Nitrous": (1.3, 0.35), "Plate": (0.6, 0.3), "Room": (0.4, 0.5)}


def reverb_fx(x, p, sr):
    x = _ch(x)
    kind = str(p.get("kind", "Hall"))
    if kind.startswith("k") and kind[1:2].isupper():
        kind = kind[1:]
    scale, base_damp = REVERB_KINDS.get(kind, (1.0, 0.45))
    size = 0.3 + 1.2 * float(np.clip(_mean(p.get("size", 0.5)), 0, 1))
    rt60 = max(0.1, _mean(p.get("decay", 1.5)))
    damp = float(np.clip(base_damp * 0.5 + 0.5 * _mean(p.get("damping", 0.5)), 0, 0.95))
    pre = int(max(0.0, _mean(p.get("predelay", 0.0))) * sr)
    src = x.mean(axis=0)
    lo, hi = _mean(p.get("lo_cut", 0.0)), _mean(p.get("hi_cut", 0.0))
    if lo > 0:
        src = src - onepole_lp(src, 20 * 2 ** (lo * 9), sr)
    if hi > 0:
        src = onepole_lp(src, 20000 * 2 ** (-hi * 8), sr)
    xin = np.concatenate([np.zeros(pre), src])[: len(src)] if pre else src
    outs = []
    for k, prime in enumerate((1.0, 1.071)):        # two decorrelated tanks for L / R
        base = np.array([1116, 1188, 1277, 1356, 1422, 1491, 1557, 1617], dtype=float) * prime
        delays = (base * size * scale * sr / 44100).astype(np.int64)
        g = 10 ** (-3 * (delays / sr) / rt60)
        wet = _fdn_k(np.ascontiguousarray(xin, dtype=np.float64), delays, g, damp) if _nb is not None else xin
        outs.append(wet * 2.0)
    if x.shape[0] > 1:
        m, s = 0.5 * (outs[0] + outs[1]), 0.5 * (outs[0] - outs[1])
        s = s * np.clip(_mean(p.get("width", 1.0)), 0, 1)
        wet = np.vstack([m + s, m - s])
    else:
        wet = (0.5 * (outs[0] + outs[1]))[None, :]
    return _mix(x, wet, p.get("mix", 0.2))


def _synthetic_ir(p, sr):
    dec = float(np.clip(_mean(p.get("decay", 1.0)), 0.05, 8.0))
    n = int(min(4.0, dec * 1.5) * sr)
    t = np.arange(n) / sr
    rng = np.random.default_rng(3)
    return rng.standard_normal((2, n)) * np.exp(-6.9 * t / dec)


def convolve_fx(x, p, sr, samples=None):
    """Serum's convolver: the preset's impulse response when it can be found
    (Serum's IR folder or embedded), otherwise a synthetic room."""
    x = _ch(x)
    ir = None
    if p.get("ir") and samples is not None:
        smp = samples.get(p["ir"])
        if smp is not None:
            ir = np.atleast_2d(smp.data).astype(float)
            if smp.sr != sr:
                from scipy.signal import resample_poly
                g = np.gcd(int(smp.sr), int(sr))
                ir = resample_poly(ir, sr // g, smp.sr // g, axis=1)
    if ir is None:
        ir = _synthetic_ir(p, sr)
    if ir.shape[0] == 1:
        ir = np.vstack([ir, ir])
    n = ir.shape[1]
    size = float(np.clip(_mean(p.get("size", 0.5)), 0.05, 4.0))
    if abs(size - 0.5) > 0.02:                      # stretch / squeeze the impulse
        m = max(16, int(n * size * 2))
        idx = np.linspace(0, n - 1, m)
        ir = np.vstack([np.interp(idx, np.arange(n), c) for c in ir])
        n = m
    t = np.arange(n) / sr
    dec = _mean(p.get("decay", 1.0))
    if p.get("ir"):
        ir = ir * np.exp(-np.maximum(t - dec, 0) * 6.9 / max(0.05, dec * 0.25))
    att = _mean(p.get("attack", 0.0))
    if att > 0:
        ir = ir * np.clip(t / att, 0, 1)
    damp = _mean(p.get("damp", 0.0))
    if damp > 0:
        ir = np.vstack([onepole_lp(c, 20000 * 2 ** (-damp * 7 * t / max(t[-1], 1e-3)), sr) for c in ir])
    tone = _mean(p.get("tone", 0.0))
    if tone < 0:
        ir = np.vstack([onepole_lp(c, 20000 * 2 ** (tone * 6), sr) for c in ir])
    elif tone > 0:
        ir = np.vstack([c - onepole_lp(c, 20 * 2 ** (tone * 9), sr) for c in ir])
    if p.get("min_phase"):
        spec = np.fft.rfft(ir, axis=1)
        logm = np.log(np.abs(spec) + 1e-9)
        cep = np.fft.irfft(logm, axis=1)
        w = np.zeros(cep.shape[1]); w[0] = 1; w[1:cep.shape[1] // 2] = 2
        ir = np.fft.irfft(np.exp(np.fft.rfft(cep * w, axis=1)), axis=1)[:, :n]
    pre = int(max(0.0, _mean(p.get("predelay", 0.0))) * sr)
    if pre:
        ir = np.hstack([np.zeros((2, pre)), ir])
    ir /= np.sqrt(np.sum(ir ** 2) / 2) + 1e-9
    ir *= 10 ** (_mean(p.get("ir_gain_db", 0.0)) / 20)
    if x.shape[0] > 1:
        wet = np.vstack([signal.fftconvolve(x[c], ir[c])[: x.shape[1]] for c in range(2)])
    else:
        wet = signal.fftconvolve(x[0], ir.mean(axis=0))[: x.shape[1]][None, :]
    return _mix(x, wet, p.get("mix", 0.2))


def _lfo(rate, n, sr, phase=0.0):
    rate = np.asarray(rate, dtype=float)
    if rate.ndim:
        ph = np.cumsum(rate) / sr + phase
    else:
        ph = np.arange(n) * float(rate) / sr + phase
    return 0.5 + 0.5 * np.sin(2 * np.pi * ph)


def _mod_delay(x, base_ms, depth_ms, rate, phase, fb, sr):
    lf = _lfo(rate, len(x), sr, phase)
    d = (np.asarray(base_ms) + np.asarray(depth_ms) * lf) * sr / 1000
    d = np.broadcast_to(d, x.shape).astype(float)
    if _nb is not None:
        return _moddelay_k(np.ascontiguousarray(x, dtype=np.float64), np.ascontiguousarray(d), float(fb))
    idx = np.arange(len(x)) - d
    return np.interp(idx, np.arange(len(x)), x, left=0)


def chorus_fx(x, p, sr):
    x = _ch(x)
    base = _mean(p.get("delay_ms", 7.0))
    base2 = _mean(p.get("delay2_ms", 0.0)) or base * 1.3
    depth = 5.0 * np.asarray(p.get("depth", 0.3), dtype=float)
    rate = p.get("rate_hz", 0.8)
    fb = _mean(p.get("feedback", 0.0))
    src = x.mean(axis=0)
    taps = [_mod_delay(src, base, depth, rate, ph, fb, sr) for ph in (0.0, 0.25)] + \
           [_mod_delay(src, base2, depth, rate, ph, fb, sr) for ph in (0.5, 0.75)]
    left, right = 0.5 * (taps[0] + taps[2]), 0.5 * (taps[1] + taps[3])
    fhz = _mean(p.get("filt_hz", 20000.0))
    if fhz < 19000 or p.get("filt_mode") == "hp":
        if p.get("filt_mode") == "hp":
            left, right = left - onepole_lp(left, fhz, sr), right - onepole_lp(right, fhz, sr)
        else:
            left, right = onepole_lp(left, fhz, sr), onepole_lp(right, fhz, sr)
    wet = np.vstack([left, right]) if x.shape[0] > 1 else (0.5 * (left + right))[None, :]
    return _mix(x, wet, p.get("mix", 0.3))


def flanger_fx(x, p, sr):
    x = _ch(x)
    depth = 4.0 * np.asarray(p.get("depth", 0.5), dtype=float)
    fb = 0.95 * _mean(p.get("feedback", 0.5))
    ph = _mean(p.get("phase", 0.0))
    wets = [_mod_delay(c, 0.5, depth, p.get("rate_hz", 0.3), ph * k, fb, sr) for k, c in enumerate(x)]
    wet = np.vstack(wets)
    return _mix(x, 0.5 * (x + wet), p.get("mix", 0.5))


def phaser_fx(x, p, sr):
    x = _ch(x)
    n = x.shape[1]
    out = []
    stages = int(np.clip(_mean(p.get("stages", 4)), 1, 12))
    for k, c in enumerate(x):
        lf = _lfo(p.get("rate_hz", 0.3), n, sr, _mean(p.get("phase", 0.0)) * k)
        f0 = np.asarray(p.get("freq", 800.0), dtype=float)
        fc = f0 * 2 ** (5 * np.asarray(p.get("depth", 0.5)) * (lf - 0.5)) * (1 + 0.5 * _mean(p.get("depth2", 0.5)))
        out.append(phaser_core(c, fc, sr, stages=stages, fb=0.9 * _mean(p.get("feedback", 0.4)), mix=1.0))
    return _mix(x, np.vstack(out), p.get("mix", 0.5))


def hyperdim_fx(x, p, sr):
    x = _ch(x)
    v = int(np.clip(_mean(p.get("voices", 4)), 0, 7))
    src = x.mean(axis=0)
    y = x
    if v > 0:
        wl, wr = np.zeros_like(src), np.zeros_like(src)
        for k in range(v):
            tap = _mod_delay(src, 3 + 9 * (k + 1) / v, 2.5 * np.asarray(p.get("detune", 0.3)),
                             0.2 + 2 * _mean(p.get("rate", 0.4)) * (1 + 0.3 * k), k / v, 0.0, sr)
            if k % 2:
                wr += tap
            else:
                wl += tap
        wl /= max(1, (v + 1) // 2)
        wr /= max(1, v // 2) if v > 1 else 1
        if v == 1:
            wr = wl
        wet = np.vstack([wl, wr]) if x.shape[0] > 1 else (0.5 * (wl + wr))[None, :]
        y = _mix(x, wet, p.get("mix", 0.4))
    dm = _mean(p.get("dim_mix", 0.0))
    if dm > 0:                                       # DIMENSION: 4 out-of-phase delay lines
        s = _mean(p.get("dim_size", 0.5))
        m = y.mean(axis=0)
        lines = [_mod_delay(m, 5 + 15 * s * (k + 1) / 4, 0.3, 0.1 * (k + 1), k * 0.25, 0.0, sr) for k in range(4)]
        side = 0.25 * (lines[0] - lines[1] + lines[2] - lines[3])
        dim = np.vstack([m + side, m - side]) if y.shape[0] > 1 else m[None, :]
        y = _mix(y, dim, dm)
    return y


def bode_fx(x, p, sr):
    """Frequency shifter: up and down shifted copies, DIR/WIDTH stereo
    spread, optional feedback delay."""
    x = _ch(x)
    n = x.shape[1]
    src = x.mean(axis=0) if p.get("mono_input") or x.shape[0] == 1 else None
    shift = np.asarray(p.get("shift_hz", 50.0), dtype=float)
    ph = np.cumsum(np.broadcast_to(shift, (n,))) / sr
    rot = np.exp(2j * np.pi * ph)
    chans = [src] * x.shape[0] if src is not None else list(x)
    d = _mean(p.get("direction", 0.0))              # -1 down only .. 0 both (opposite) .. 1 up only
    out = []
    fb = _mean(p.get("feedback", 0.0))
    dt = int(max(0.0, _mean(p.get("delay_time", 0.0))) * sr)
    for k, c in enumerate(chans):
        a = signal.hilbert(c)
        up, down = np.real(a * rot), np.real(a * np.conj(rot))
        if x.shape[0] > 1 and abs(d) < 0.99:
            w = np.clip(_mean(p.get("width", 0.5)), 0, 1)
            y = up if (k == 0) else ((1 - w) * up + w * down)
        else:
            y = up if d >= 0 else down
        if dt > 0 and fb > 0:
            buf = y.copy()
            for i in range(dt, n):
                buf[i] += fb * buf[i - dt]
            y = buf
        out.append(y)
    return _mix(x, np.vstack(out), p.get("mix", 0.5))


def utility_fx(x, p, sr):
    y = _ch(x).copy()
    if _mean(p.get("hpf", 0)) > 0:
        y = _per_channel(lambda c: tv_filter(c, "hp", p["hpf"], 0.707, sr), y)
    if _mean(p.get("lpf", 0)) > 0:
        y = _per_channel(lambda c: tv_filter(c, "lp", p["lpf"], 0.707, sr), y)
    if p.get("invert"):
        y[0] = -y[0]
    if p.get("invert_r") and y.shape[0] > 1:
        y[1] = -y[1]
    if y.shape[0] > 1:
        m, s = _ms(y)
        if p.get("mono_bass"):
            s = s - onepole_lp(s, _mean(p.get("mono_freq", 120.0)), sr)
        s = s * np.asarray(p.get("width", 1.0))
        y = _from_ms(m, s, 2)
        bal = np.asarray(p.get("balance", 0.0), dtype=float)
        y = np.vstack([y[0] * np.clip(1 - bal, 0, 1), y[1] * np.clip(1 + bal, 0, 1)])
    y = y * 10 ** (np.asarray(p.get("gain_db", 0.0)) / 20)
    return _mix(_ch(x), y, p.get("mix", 1.0))


def _env_follow(x2, sr, att, rel):
    a = np.exp(-1 / (max(att, 1e-4) * sr))
    r = np.exp(-1 / (max(rel, 1e-3) * sr))
    fast = signal.lfilter([1 - a], [1, -a], x2)
    slow = signal.lfilter([1 - r], [1, -r], x2)
    return np.sqrt(np.maximum(np.maximum(fast * 0.5, slow), 1e-12))


def comp_fx(x, p, sr):
    """Single-band compressor / limiter or Serum's multiband up/down
    compressor (OTT style) with per-band thresholds, ratios and gains."""
    x = _ch(x)
    link = x.mean(axis=0)
    peak = np.max(np.abs(x)) + 1e-9
    xn = x / peak * 0.5
    ln = link / peak * 0.5
    att = _mean(p.get("attack_ms", 5.0)) / 1000
    rel = _mean(p.get("release_ms", 80.0)) / 1000
    if p.get("multiband", True):
        f_lo = float(np.clip(_mean(p.get("xover_lo", 88.0 if p.get("band_cfg") is None else 200.0)), 20, sr * 0.2))
        f_hi = float(np.clip(_mean(p.get("xover_hi", 2500.0)), f_lo * 1.5, sr * 0.45))
        lo_sos = signal.butter(4, f_lo, "low", fs=sr, output="sos")
        hi_sos = signal.butter(4, f_hi, "high", fs=sr, output="sos")
        bands_cfg = p.get("band_cfg") or [{}, {}, {}]
        amount = _mean(p.get("amount", 0.6))
        out = np.zeros_like(xn)
        for bi in range(3):
            if bi == 0:
                parts = signal.sosfilt(lo_sos, xn, axis=1); key = signal.sosfilt(lo_sos, ln)
            elif bi == 2:
                parts = signal.sosfilt(hi_sos, xn, axis=1); key = signal.sosfilt(hi_sos, ln)
            else:
                parts = xn - signal.sosfilt(lo_sos, xn, axis=1) - signal.sosfilt(hi_sos, xn, axis=1)
                key = ln - signal.sosfilt(lo_sos, ln) - signal.sosfilt(hi_sos, ln)
            bc = bands_cfg[bi] if bi < len(bands_cfg) else {}
            e = 20 * np.log10(_env_follow(key ** 2, sr, att if p.get("band_cfg") else 0.003, rel if p.get("band_cfg") else 0.08))
            thr_dn = bc.get("thresh_down_db", -20.0)
            thr_up = bc.get("thresh_up_db", -40.0)
            r_dn = max(1.0, bc.get("ratio", 8.0))
            r_up = max(1.0, bc.get("ratio_below", 3.0))
            down = -np.maximum(0, e - thr_dn) * (1 - 1 / r_dn)
            up = np.maximum(0, thr_up - e) * (1 - 1 / r_up)
            g = np.clip(amount * (down + np.minimum(up, 24)), -40, 24) + bc.get("gain_db", 0.0)
            out += parts * 10 ** (g / 20)
        y = out * 10 ** (_mean(p.get("makeup_db", 0.0)) / 20)
        return _mix(xn, y, p.get("depth", 0.5)) * peak * 2
    ratio = _mean(p.get("ratio", 4.0))
    thr = _mean(p.get("threshold_db", -18.0))
    if ratio >= 100:                                # LIMIT: true-peak style limiter
        e = 20 * np.log10(np.maximum(_env_follow(ln ** 2, sr, 0.0002, max(rel, 0.01)) * 1.41, 1e-9))
        g = -np.maximum(0, e - thr)
    else:
        e = 20 * np.log10(_env_follow(ln ** 2, sr, att, rel))
        g = -np.maximum(0, e - thr) * (1 - 1 / max(1.0, ratio))
    g = g + _mean(p.get("makeup_db", 0.0))
    y = xn * 10 ** (g / 20)
    return _mix(xn, y, p.get("depth", 1.0) if not p.get("multiband", True) else 1.0) * peak * 2


def eq_fx(x, p, sr):
    y = _ch(x).copy()
    for k in (1, 2):
        gain = p.get(f"gain{k}", 0.0)
        kind = p.get(f"type{k}", "peak")
        if _mean(np.abs(gain)) < 0.05 and np.ndim(gain) == 0 and kind in ("lowshelf", "highshelf", "peak"):
            continue
        q = max(0.3, _mean(p.get(f"q{k}", 0.7)))
        freq = p.get(f"freq{k}", 1000.0)
        g = 0.0 if kind in ("lp", "hp") else _mean(gain)
        y = _per_channel(lambda c: tv_filter(c, kind, freq, q, sr, gain_db=g), y)
    return y


def distortion_fx(x, p, sr):
    """13 distortion modes (+ X-Shaper with drawn curves), optional pre/post
    filter morphing LP -> BP -> HP."""
    x = _ch(x)
    xin = x
    filt = p.get("filter", "off")

    def _flt(c):
        ff = _mean(p.get("filter_freq", 0.5))
        hz = float(np.clip(ff if ff > 1.5 else 20 * 1000 ** ff, 20, sr * 0.45))
        bw = _mean(p.get("filter_bw", 0.5))
        q = 0.5 + 4 * (1 - bw)
        t = _mean(p.get("filter_type", 0.0))
        lp, bp, hp = tv_filter(c, "lp", hz, q, sr), tv_filter(c, "bp", hz, q, sr), tv_filter(c, "hp", hz, q, sr)
        return np.where(t < 0.5, lp * (1 - 2 * t) + bp * 2 * t, bp * (2 - 2 * t) + hp * (2 * t - 1))
    if filt == "pre":
        xin = _per_channel(_flt, xin)
    mode = p.get("mode", "Overdrive")
    drive = np.asarray(p.get("drive", 0.3), dtype=float)
    if mode in ("XShaper", "XShaperAsym") and (p.get("curve_a") or p.get("curve_b")):
        y = _xshaper(xin, p.get("curve_a"), p.get("curve_b"), drive, asym=mode == "XShaperAsym")
    elif mode == "Downsample":
        y = shape(xin.reshape(-1), mode, _mean(drive)).reshape(xin.shape)
    else:
        y = np.vstack([shape(c, mode, drive) for c in xin])
    if filt == "post":
        y = _per_channel(_flt, y)
    return _mix(x, y, p.get("mix", 1.0))


def _xshaper(x, ca, cb, morph, asym=False):
    def curve(pts, v):
        if not pts:
            return np.tanh(2 * v)
        xs = np.array([q[0] for q in pts], dtype=float)
        ys = np.array([q[1] for q in pts], dtype=float)
        if asym:
            return 2 * np.interp((v + 1) / 2, xs, ys) - 1
        return np.sign(v) * np.interp(np.abs(v), xs, ys)
    v = np.clip(x, -1, 1)
    m = np.clip(morph, 0, 1)
    return curve(ca, v) * (1 - m) + curve(cb, v) * m


def filter_fx(x, p, sr):
    x = _ch(x)
    cut = np.asarray(p.get("cutoff", 2000.0), dtype=float)
    pan = _mean(p.get("pan", 0.0))
    out = []
    for k, c in enumerate(x):
        ck = cut * (2 ** (pan * (1 if k == 0 else -1)) if x.shape[0] > 1 and pan else 1.0)
        out.append(apply_filter_type(c, p.get("ftype", "L24"), ck, _mean(p.get("reso", 0.2)), p.get("drive", 0.0),
                                     p.get("var", 0.5), sr, mix=p.get("mix", 1.0), pz=p.get("pz"),
                                     clean=bool(p.get("clean"))))
    return np.vstack(out)


FX_FUNCS = {"delay": delay_fx, "reverb": reverb_fx, "convolve": convolve_fx, "chorus": chorus_fx,
            "flanger": flanger_fx, "phaser": phaser_fx, "hyperdim": hyperdim_fx, "bode": bode_fx,
            "utility": utility_fx, "comp": comp_fx, "eq": eq_fx, "distortion": distortion_fx,
            "filter": filter_fx}


def process_chain(x, chain: list, sr: int, mods: dict | None = None, prefix: str = "fx",
                  values: dict | None = None, path: str = "fx_chain", samples=None) -> np.ndarray:
    """Run an FX rack (list of module dicts, splitters nest sub-chains) on a
    (C, n) or (n,) signal.

    ``mods`` maps the legacy "fx.<i>.mix" / "fx.<i>.drive" keys to per-sample
    modulation; ``values`` maps "<path>.<j>.<param>" (e.g.
    "fx_chain.2.feedback") to modulated parameter values (arrays)."""
    one_d = np.ndim(x) == 1
    y = _ch(x)
    mods = mods or {}
    values = values or {}
    for i, m in enumerate(chain or []):
        if not m.get("enabled", True):
            continue
        kind = m.get("type")
        key = f"{prefix}.{i}"
        mpath = f"{path}.{i}"
        p = dict(m)
        for k in list(p):
            v = values.get(f"{mpath}.{k}")
            if v is not None:
                p[k] = v
        if kind in ("split", "split3", "splitms"):
            y = _split(y, p, sr, mods, key, values, mpath, samples)
            continue
        fn = FX_FUNCS.get(kind)
        if fn is None:
            continue
        before = y
        if f"{key}.drive" in mods and kind == "distortion":
            p["drive"] = np.clip(np.asarray(p.get("drive", 0.3)) + mods[f"{key}.drive"], 0, 1)
        y = fn(y, p, sr, samples) if kind == "convolve" else fn(y, p, sr)
        if f"{key}.mix" in mods:          # modulated wet/dry on top of the module's own mix
            w = np.clip(mods[f"{key}.mix"], -1, 1)
            y = before + (y - before) * np.clip(1 + w, 0, 2)
        lvl = p.get("level", 1.0)
        if np.ndim(lvl) or lvl != 1.0:
            y = y * np.asarray(lvl)
        if not np.all(np.isfinite(y)):
            y = before
    return y[0] if one_d else y


def _split(x, m, sr, mods, key, values, mpath, samples):
    bands = m.get("bands") or []
    kind = m["type"]
    out = np.zeros_like(x, dtype=float)
    if kind == "splitms":
        mid, side = _ms(x)
        mo = process_chain(mid[None, :], bands[0] if bands else [], sr, mods, key + ".b0", values,
                           mpath + ".bands.0", samples)[0]
        so = process_chain(side[None, :], bands[1] if len(bands) > 1 else [], sr, mods, key + ".b1", values,
                           mpath + ".bands.1", samples)[0] if x.shape[0] > 1 else side
        return _from_ms(mo, so, x.shape[0])
    if kind == "split":
        f = float(np.clip(_mean(m.get("freq", 300.0)), 20, sr * 0.45))
        lo = signal.sosfiltfilt(signal.butter(4, f, "low", fs=sr, output="sos"), x, axis=1)
        parts = [lo, x - lo]
    else:
        f1 = float(np.clip(_mean(m.get("freq", 200.0)), 20, sr * 0.45))
        f2 = float(np.clip(_mean(m.get("freq2", 2000.0)), f1 * 1.1, sr * 0.45))
        lo = signal.sosfiltfilt(signal.butter(4, f1, "low", fs=sr, output="sos"), x, axis=1)
        hi = signal.sosfiltfilt(signal.butter(4, f2, "high", fs=sr, output="sos"), x, axis=1)
        parts = [lo, x - lo - hi, hi]
    for b, part in enumerate(parts):
        chain = bands[b] if b < len(bands) else []
        out += process_chain(part, chain, sr, mods, f"{key}.b{b}", values, f"{mpath}.bands.{b}", samples)
    return out
