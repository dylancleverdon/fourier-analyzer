"""Learning from real Serum 2: what our engine gets wrong, and corrections.

After a reconstruction is verified in real Serum 2, the same preset exists as
two renders: our engine's prediction and what Serum actually played. Their
difference is the *engine gap* (separate from the fitting gap, which is how
far the patch is from the original). For every run this module

1. documents exactly how the two renders differ (level, frequency bands,
   tilt, brightness, envelope per note, pitch, timing, harmonics, noise,
   modulation) and turns that into plain-language findings;
2. finds the culprit: each patch component is switched off in both engines
   and the component whose removal closes the gap most is the one we emulate
   wrongly;
3. records how far the Serum tuning stage had to move each parameter;
4. appends one record to ``calibration/log.jsonl`` (numbers only, no audio)
   and learns per-parameter export corrections once the same offset shows up
   run after run. Corrections are applied when writing presets.
"""
from __future__ import annotations

import json
import logging
import math
import time
from pathlib import Path

import numpy as np

from . import config
from .match import params as P
from .match.metrics import Reference, best_lag
from .synth.patch import Patch

log = logging.getLogger(__name__)

MIN_RUNS = 5                 # observations of a parameter before a correction changes
CONSISTENT = 0.7             # share of observations that must agree in sign
RATIO_LIMITS = (0.25, 4.0)
MAX_COMPONENTS = 16


# ----------------------------------------------------------------------------
# storage

def cal_dir() -> Path:
    d = config.data_dir() / "calibration"
    d.mkdir(parents=True, exist_ok=True)
    return d


def log_path() -> Path:
    return cal_dir() / "log.jsonl"


def corrections_path() -> Path:
    return cal_dir() / "corrections.json"


def load_corrections() -> dict:
    try:
        c = json.loads(corrections_path().read_text())
    except (OSError, ValueError):
        c = {}
    c.setdefault("params", {})
    c.setdefault("pending", {})
    c.setdefault("history", [])
    return c


def save_corrections(c: dict) -> None:
    corrections_path().write_text(json.dumps(c, indent=1))


def reset_corrections() -> None:
    save_corrections({"params": {}, "pending": {}, "history": []})


def checks_path() -> Path:
    return cal_dir() / "structure_checks.jsonl"


def part_kind(name: str) -> str:
    """'LFO1 (sine, 1/8) -> filter.cutoff' -> 'LFO -> filter.cutoff': one row
    per kind of part on the Engine accuracy page."""
    import re
    k = re.sub(r"\([^)]*\)", "", name)
    k = re.sub(r"\b(on|from) osc [ABC]\b", "", k)
    k = re.sub(r"\bx\d+\b", "", k)
    k = re.sub(r"[+-]?\d+(\.\d+)?(\s?(st|ct|ms|Hz)\b|%)?", "", k)
    return re.sub(r"\s+", " ", k).strip(" :") or name


def record_checks(rows: list[dict], meta: dict | None = None) -> None:
    """Structure decisions checked in real Serum 2 (our engine's verdict vs
    Serum's). The disagreements are evidence of what our engine gets wrong."""
    if not rows:
        return
    try:
        with checks_path().open("a") as fh:
            for r in rows:
                fh.write(json.dumps({"t": time.time(), **(meta or {}), **r, "kind_of_part": part_kind(r["part"])},
                                    default=_jsonable) + "\n")
    except OSError as e:
        log.warning("could not write the structure-check log: %s", e)


def read_checks() -> list[dict]:
    try:
        lines = checks_path().read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


def read_log(limit: int | None = None) -> list[dict]:
    try:
        lines = log_path().read_text().splitlines()
    except OSError:
        return []
    out = []
    for line in lines[-limit:] if limit else lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            pass
    return out


# ----------------------------------------------------------------------------
# parameter keys and corrections

def _src_kind(src: str) -> str:
    return "".join(ch for ch in src if not ch.isdigit()) or src


def param_key(patch: Patch, path: str) -> str:
    """A key shared by every patch where the same emulation error would show:
    'filter.lp24.cutoff', 'fx.reverb.decay', 'osc.wavetable.warp_amount.BendPos'."""
    parts = path.split(".")
    try:
        if parts[0] == "oscs":
            o = patch.oscs[int(parts[1])]
            rest = ".".join(parts[2:])
            if rest == "warp_amount":
                return f"osc.{o.type}.warp_amount.{o.warp}"
            if rest == "warp2_amount":
                return f"osc.{o.type}.warp_amount.{o.warp2}"
            return f"osc.{o.type}.{rest}"
        if parts[0] in ("filter", "filter2"):
            f = getattr(patch, parts[0])
            return f"filter.{f.type}.{'.'.join(parts[1:])}"
        if parts[0] == "envs":
            return ("amp_env." if parts[1] == "0" else "env.") + ".".join(parts[2:])
        if parts[0] == "mods":
            m = patch.mods[int(parts[1])]
            return f"mod.{_src_kind(m.source)}->{m.dest}.{'.'.join(parts[2:])}"
        if parts[0] in ("fx_chain", "fx_buses"):
            mod = P.get_attr(patch, ".".join(parts[:-1]))
            if isinstance(mod, dict) and mod.get("type"):
                return f"fx.{mod['type']}.{parts[-1]}"
    except (IndexError, KeyError, ValueError, AttributeError, TypeError):
        pass
    return ".".join(p for p in parts if not p.isdigit())


def apply_corrections(patch: Patch, corr: dict | None = None) -> Patch:
    """The patch as it should be written for Serum: learned corrections applied
    to every parameter they cover. Returns the patch itself when none apply."""
    corr = load_corrections() if corr is None else corr
    table = corr.get("params", {})
    if not table:
        return patch
    q = None
    for prm in P.continuous_params(patch):
        c = table.get(param_key(patch, prm.path))
        if not c:
            continue
        q = q or patch.copy()
        v = float(P.get_attr(q, prm.path))
        if c["kind"] == "ratio":
            v *= float(c["value"])
        else:
            v += float(c["value"])
        P.set_attr(q, prm.path, float(min(max(v, prm.lo), prm.hi)))
    return q or patch


def tuning_changes(before: Patch, after: Patch) -> list[dict]:
    """How far the Serum tuning stage moved each parameter (our units)."""
    out = []
    for prm in P.continuous_params(before):
        try:
            a, b = float(P.get_attr(before, prm.path)), float(P.get_attr(after, prm.path))
        except (IndexError, KeyError, AttributeError):
            continue
        out.append({"path": prm.path, "key": param_key(before, prm.path), "kind": "ratio" if prm.log else "offset",
                    "before": a, "after": b, "span": prm.hi - prm.lo})
    return out


def learn(changes: list[dict], corr: dict | None = None) -> tuple[dict, list[str]]:
    """Add one run's tuning moves; update a correction once MIN_RUNS runs agree.
    Returns (corrections, notes on what changed)."""
    corr = load_corrections() if corr is None else corr
    per_key: dict[str, list] = {}
    for ch in changes:
        if ch["kind"] == "ratio":
            if ch["before"] <= 0 or ch["after"] <= 0:
                continue
            obs = math.log(ch["after"] / ch["before"])
        else:
            obs = (ch["after"] - ch["before"]) / max(ch["span"], 1e-9)     # share of the range
        per_key.setdefault(ch["key"], []).append((ch["kind"], obs, ch["span"]))
    notes = []
    for key, rows in per_key.items():
        kind, span = rows[0][0], rows[0][2]
        pend = corr["pending"].setdefault(key, [])
        pend.append(float(np.mean([r[1] for r in rows])))
        del pend[:-20]
        if len(pend) < MIN_RUNS:
            continue
        arr = np.array(pend)
        med = float(np.median(arr))
        agree = float(np.mean(np.sign(arr) == np.sign(med))) if med else 0.0
        big = abs(med) > (math.log(1.03) if kind == "ratio" else 0.02)
        if not (big and agree >= CONSISTENT):
            continue
        cur = corr["params"].get(key, {"kind": kind, "value": 1.0 if kind == "ratio" else 0.0, "runs": 0})
        if kind == "ratio":
            cur["value"] = float(np.clip(cur["value"] * math.exp(med), *RATIO_LIMITS))
            text = f"{key}: Serum needs x{math.exp(med):.3f} (correction now x{cur['value']:.3f})"
        else:
            cur["value"] = float(np.clip(cur["value"] + med * span, -0.5 * span, 0.5 * span))
            text = f"{key}: Serum needs {med * span:+.4f} (correction now {cur['value']:+.4f})"
        cur["runs"] = int(cur.get("runs", 0)) + len(pend)
        cur["updated"] = time.time()
        corr["params"][key] = cur
        corr["pending"][key] = []
        corr["history"].append({"t": time.time(), "key": key, "value": cur["value"], "kind": kind,
                                "from_runs": len(pend)})
        del corr["history"][:-200]
        notes.append(text)
    save_corrections(corr)
    return corr, notes


# ----------------------------------------------------------------------------
# what differs between two renders

BANDS = [31.5 * 2 ** (k / 3) for k in range(0, 30)]          # 1/3 octaves, 31.5 Hz .. 16 kHz


def _db(x):
    return 10 * np.log10(np.maximum(x, 1e-20))


def _avg_spectrum(y, sr, nfft=4096):
    hop = nfft // 2
    if len(y) < nfft:
        y = np.pad(y, (0, nfft - len(y)))
    win = np.hanning(nfft)
    frames = [np.abs(np.fft.rfft(y[i:i + nfft] * win)) ** 2 for i in range(0, len(y) - nfft + 1, hop)]
    return np.mean(frames, axis=0), np.fft.rfftfreq(nfft, 1 / sr)


def _band_levels(spec, fr):
    out = []
    for f in BANDS:
        lo, hi = f / 2 ** (1 / 6), f * 2 ** (1 / 6)
        m = (fr >= lo) & (fr < hi)
        out.append(float(_db(spec[m].sum())) if m.any() else -200.0)
    return np.array(out)


def _fmt_hz(f):
    return f"{f / 1000:.1f} kHz" if f >= 1000 else f"{f:.0f} Hz"


def _rms_env(y, sr, hop_s=0.005):
    h = max(1, int(hop_s * sr))
    n = len(y) // h
    if n == 0:
        return np.zeros(1), h
    return np.sqrt(np.mean(y[: n * h].reshape(n, h) ** 2, axis=1)), h


def _note_env_stats(env, h, sr, note, total_n):
    a = int(note.start * sr / h)
    b = int((note.start + note.duration) * sr / h)
    end = min(len(env), b + int(0.6 * sr / h))
    seg = env[a:max(a + 1, b)]
    if len(seg) < 3:
        return None
    peak = float(seg.max()) + 1e-12
    ref = float(np.percentile(seg, 95)) + 1e-12   # robust "full level" (ignores single spikes)
    i10 = int(np.argmax(seg >= 0.1 * ref))
    i90 = int(np.argmax(seg >= 0.9 * ref))
    mid = float(np.median(seg[len(seg) // 3: max(len(seg) // 3 + 1, 2 * len(seg) // 3)]))
    after = np.nonzero(seg[i90:] < 0.5 * peak)[0]
    decay = (after[0] * h / sr) if len(after) else None
    tail = env[b:end]
    rel = np.nonzero(tail < peak * 0.01)[0]
    release = (rel[0] * h / sr) if len(rel) else (len(tail) * h / sr if len(tail) else None)
    on = np.nonzero(env[max(0, a - int(0.03 * sr / h)):a + int(0.05 * sr / h)] > 0.1 * peak)[0]
    onset = ((on[0] * h / sr) - 0.03) if len(on) else None
    return {"attack_s": max(0, i90 - i10) * h / sr, "peak_db": 20 * math.log10(peak), "sustain_rel_db": 20 * math.log10(mid / peak + 1e-9),
            "decay_s": decay, "release_s": release, "onset_s": onset}


def _f0(seg, sr, fmin=30.0, fmax=2000.0):
    if len(seg) < int(sr / fmin) * 3:
        return None
    seg = seg - seg.mean()
    n = 1 << int(np.ceil(np.log2(2 * len(seg))))
    F = np.fft.rfft(seg * np.hanning(len(seg)), n)
    ac = np.fft.irfft(np.abs(F) ** 2)[: len(seg)]
    lo, hi = int(sr / fmax), min(len(ac) - 2, int(sr / fmin))
    if hi <= lo or ac[0] <= 0:
        return None
    k = lo + int(np.argmax(ac[lo:hi]))
    if ac[k] < 0.3 * ac[0]:
        return None
    a, b, c = ac[k - 1], ac[k], ac[k + 1]
    d = 0.5 * (a - c) / (a - 2 * b + c) if (a - 2 * b + c) else 0.0
    return sr / (k + d)


def _harmonics(seg, sr, f0, n=12):
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
    fr = np.fft.rfftfreq(len(seg), 1 / sr)
    out = []
    for k in range(1, n + 1):
        f = k * f0
        if f > sr / 2 - 100:
            break
        m = (fr > f - f0 * 0.25) & (fr < f + f0 * 0.25)
        out.append(float(20 * np.log10(spec[m].max() + 1e-12)) if m.any() else -200.0)
    return np.array(out)


def _flatness(y, sr):
    spec, fr = _avg_spectrum(y, sr, 2048)
    m = (fr > 100) & (fr < 12000)
    s = spec[m] + 1e-20
    return float(np.exp(np.mean(np.log(s))) / np.mean(s))


def _mod_rate(env, h, sr, note):
    """Strongest amplitude-modulation rate (Hz) and depth inside a note."""
    a = int((note.start + 0.08) * sr / h)
    b = int((note.start + note.duration) * sr / h)
    seg = env[a:b]
    if len(seg) < 40:
        return None
    x = np.log(seg + 1e-9)
    x = x - x.mean()
    sp = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), h / sr)
    m = (fr > 0.5) & (fr < 25)
    if not m.any():
        return None
    k = np.argmax(np.where(m, sp, 0))
    depth = float(np.std(x) * 8.686)
    return {"rate_hz": float(fr[k]), "depth_db": depth}


def describe_difference(ours: np.ndarray, serum: np.ndarray, sr: int, notes) -> dict:
    """Everything measurable that differs between our render and Serum's
    render of the same preset, plus findings in plain language."""
    n = min(len(ours), len(serum))
    a = np.asarray(ours[:n], dtype=np.float64)
    b = np.asarray(serum[:n], dtype=np.float64)
    out: dict = {"seconds": round(n / sr, 2)}
    find: list[tuple[float, str]] = []           # (importance, text)
    ra, rb = float(np.sqrt(np.mean(a ** 2))), float(np.sqrt(np.mean(b ** 2)))
    if ra < 1e-7 or rb < 1e-7:
        out["silent"] = {"ours": ra < 1e-7, "serum": rb < 1e-7}
        find.append((100, "Serum rendered silence" if rb < 1e-7 else "our engine rendered silence"))
        out["findings"] = [t for _, t in find]
        return out
    level = 20 * math.log10(rb / ra)
    out["level_db"] = round(level, 2)
    if abs(level) > 1.0:
        find.append((abs(level), f"Serum is {abs(level):.1f} dB {'louder' if level > 0 else 'quieter'} overall"))
    lag, r = best_lag(a, b, int(0.03 * sr))
    lag = -lag                                  # best_lag returns the shift that aligns b; > 0 here = Serum late
    out["lag_ms"] = round(1000 * lag / sr, 2)
    out["correlation"] = round(float(r), 4)
    if abs(lag) / sr > 0.0015:
        find.append((abs(lag) / sr * 1000, f"Serum's output is shifted {1000 * lag / sr:+.1f} ms"))
    ref = Reference(b, sr)
    m = ref.all(a)
    out["similarity"] = {k: m[k] for k in ("waveform_similarity", "spectral_similarity", "lsd_db",
                                           "envelope_correlation", "si_sdr_db")}
    # frequency bands (shape after the overall level difference is removed)
    sa, fr = _avg_spectrum(a, sr)
    sb, _ = _avg_spectrum(b, sr)
    la, lb = _band_levels(sa, fr), _band_levels(sb, fr)
    top = max(la.max(), lb.max())
    audible = (np.maximum(la, lb) > top - 60) & (np.array(BANDS) < sr / 2)
    diff = np.where(audible, (lb - la) - level, 0.0)
    out["bands"] = [{"hz": round(f, 1), "diff_db": round(float(d), 2)} for f, d, ok in zip(BANDS, diff, audible) if ok]
    run, start = [], None
    for k in range(len(BANDS) + 1):            # contiguous regions over 3 dB
        sgn = np.sign(diff[k]) if k < len(BANDS) and abs(diff[k]) > 3 else 0
        if start is not None and (sgn != np.sign(diff[start]) or sgn == 0):
            seg = diff[start:k]
            run.append((start, k - 1, float(seg.mean())))
            start = None
        if sgn != 0 and start is None:
            start = k
    out["band_regions"] = []
    for i0, i1, d in run:
        lo, hi = BANDS[i0] / 2 ** (1 / 6), BANDS[i1] * 2 ** (1 / 6)
        out["band_regions"].append({"lo_hz": round(lo), "hi_hz": round(hi), "diff_db": round(d, 1)})
        find.append((abs(d) * (i1 - i0 + 1) ** 0.5,
                     f"Serum has {abs(d):.1f} dB {'more' if d > 0 else 'less'} energy at {_fmt_hz(lo)}-{_fmt_hz(hi)}"))
    if audible.sum() >= 4:
        x = np.log2(np.array(BANDS)[audible])
        tilt = float(np.polyfit(x, diff[audible], 1)[0])
        out["tilt_db_per_octave"] = round(tilt, 2)
        if abs(tilt) > 0.75:
            find.append((abs(tilt) * 4, f"Serum is {'brighter' if tilt > 0 else 'darker'} ({tilt:+.1f} dB per octave)"))
    cent = lambda s: float((fr * s).sum() / (s.sum() + 1e-20))
    out["centroid_hz"] = {"ours": round(cent(sa)), "serum": round(cent(sb))}
    fa, fb = _flatness(a, sr), _flatness(b, sr)
    out["noisiness"] = {"ours": round(fa, 4), "serum": round(fb, 4)}
    if max(fa, fb) > 0.02 and abs(math.log((fb + 1e-6) / (fa + 1e-6))) > 0.5:
        find.append((3, f"Serum is {'noisier' if fb > fa else 'cleaner'} (spectral flatness {fb:.3f} vs {fa:.3f})"))
    # per note: envelope, pitch, harmonics, modulation
    ea, h = _rms_env(a, sr)
    eb, _ = _rms_env(b, sr)
    per = []
    for nt in sorted(notes, key=lambda x: -x.duration)[:6]:
        if nt.start >= n / sr - 0.05:
            continue
        sa_, sb_ = _note_env_stats(ea, h, sr, nt, n), _note_env_stats(eb, h, sr, nt, n)
        if not sa_ or not sb_:
            continue
        row = {"start": round(nt.start, 3), "pitch": round(float(nt.pitch), 2), "ours": sa_, "serum": sb_}
        i0 = int((nt.start + min(0.1, nt.duration * 0.3)) * sr)
        i1 = int((nt.start + nt.duration * 0.9) * sr)
        if i1 - i0 > int(0.04 * sr):
            f_a, f_b = _f0(a[i0:i1], sr), _f0(b[i0:i1], sr)
            if f_a and f_b:
                row["pitch_cents"] = round(1200 * math.log2(f_b / f_a), 1)
                ha, hb = _harmonics(a[i0:i1], sr, f_a), _harmonics(b[i0:i1], sr, f_a)
                k = min(len(ha), len(hb))
                if k >= 2:
                    row["harmonics_db"] = [round(float((hb[j] - hb[0]) - (ha[j] - ha[0])), 1) for j in range(1, k)]
        ma, mb = _mod_rate(ea, h, sr, nt), _mod_rate(eb, h, sr, nt)
        if ma and mb:
            row["modulation"] = {"ours": ma, "serum": mb}
        per.append(row)
    out["notes"] = per

    def avg(get):
        vals = [v for v in (get(r) for r in per) if v is not None]
        return float(np.mean(vals)) if vals else None
    att = avg(lambda r: r["serum"]["attack_s"] - r["ours"]["attack_s"])
    dec = avg(lambda r: (r["serum"]["decay_s"] - r["ours"]["decay_s"])
              if r["serum"]["decay_s"] is not None and r["ours"]["decay_s"] is not None else None)
    rel = avg(lambda r: (r["serum"]["release_s"] - r["ours"]["release_s"])
              if r["serum"]["release_s"] is not None and r["ours"]["release_s"] is not None else None)
    sus = avg(lambda r: r["serum"]["sustain_rel_db"] - r["ours"]["sustain_rel_db"])
    ons = avg(lambda r: (r["serum"]["onset_s"] - r["ours"]["onset_s"])
              if r["serum"]["onset_s"] is not None and r["ours"]["onset_s"] is not None else None)
    pc = avg(lambda r: r.get("pitch_cents"))
    out["envelope"] = {"attack_s": att, "decay_s": dec, "release_s": rel, "sustain_db": sus, "onset_s": ons}
    out["pitch_cents"] = pc
    our_att = avg(lambda r: r["ours"]["attack_s"])
    if att is not None and abs(att) > max(0.004, 0.3 * (our_att or 0)):
        find.append((abs(att) * 400, f"Serum's attack is {abs(att) * 1000:.0f} ms {'slower' if att > 0 else 'faster'}"))
    if dec is not None and abs(dec) > 0.02:
        find.append((abs(dec) * 60, f"Serum's decay is {abs(dec) * 1000:.0f} ms {'longer' if dec > 0 else 'shorter'}"))
    if sus is not None and abs(sus) > 1.5:
        find.append((abs(sus), f"Serum sustains {abs(sus):.1f} dB {'higher' if sus > 0 else 'lower'} relative to the peak"))
    if rel is not None and abs(rel) > 0.02:
        find.append((abs(rel) * 40, f"Serum's release tail is {abs(rel) * 1000:.0f} ms {'longer' if rel > 0 else 'shorter'}"))
    if ons is not None and abs(ons) > 0.002:
        find.append((abs(ons) * 1000, f"Serum's notes start {abs(ons) * 1000:.1f} ms {'later' if ons > 0 else 'earlier'}"))
    if pc is not None and abs(pc) > 3:
        find.append((abs(pc) / 2, f"Serum plays {abs(pc):.0f} cents {'sharp' if pc > 0 else 'flat'}"))
    hs = [r["harmonics_db"] for r in per if r.get("harmonics_db")]
    if hs:
        k = min(len(x) for x in hs)
        hd = np.mean([x[:k] for x in hs], axis=0)
        out["harmonics_db"] = [round(float(v), 1) for v in hd]
        big = [(j + 2, v) for j, v in enumerate(hd) if abs(v) > 3]
        if big:
            find.append((float(np.mean([abs(v) for _, v in big])) * 1.5,
                         "harmonic balance differs: " + ", ".join(f"H{j} {v:+.0f} dB" for j, v in big[:6])))
    mods = [r["modulation"] for r in per if r.get("modulation")]
    if mods:
        dr = float(np.mean([m_["serum"]["rate_hz"] - m_["ours"]["rate_hz"] for m_ in mods]))
        dd = float(np.mean([m_["serum"]["depth_db"] - m_["ours"]["depth_db"] for m_ in mods]))
        out["modulation"] = {"rate_hz": round(dr, 3), "depth_db": round(dd, 2)}
        if abs(dd) > 2:
            find.append((abs(dd), f"Serum's amplitude movement is {abs(dd):.1f} dB {'deeper' if dd > 0 else 'shallower'}"))
        if abs(dr) > 0.3 and max(m_["serum"]["depth_db"] for m_ in mods) > 2:
            find.append((abs(dr) * 3, f"Serum's modulation rate differs by {dr:+.2f} Hz"))
    out["findings"] = [t for _, t in sorted(find, key=lambda x: -x[0])]
    return out


# ----------------------------------------------------------------------------
# culprit search

def components(patch: Patch) -> list[tuple[str, callable]]:
    """Parts of the patch that can be switched off one at a time."""
    out = []

    def add(label, fn):
        out.append((label, fn))
    for j, m in enumerate(patch.fx_chain):
        def f(q, j=j):
            q.fx_chain.pop(j)
            return q
        add(f"FX {m.get('type', '?')}", f)
    for b, bus in enumerate(getattr(patch, "fx_buses", []) or []):
        for j, m in enumerate(bus or []):
            def f(q, b=b, j=j):
                q.fx_buses[b].pop(j)
                return q
            add(f"FX bus {b + 1} {m.get('type', '?')}", f)
    for name in ("filter", "filter2"):
        flt = getattr(patch, name)
        if flt.enabled:
            def f(q, name=name):
                getattr(q, name).enabled = False
                return q
            add(f"{'filter 1' if name == 'filter' else 'filter 2'} ({flt.type})", f)
    active = patch.active_oscs()
    for i, o in active:
        osc = "ABC"[i]
        for slot in ("warp", "warp2"):
            code = getattr(o, slot)
            if code != "none":
                def f(q, i=i, slot=slot):
                    setattr(q.oscs[i], slot, "none")
                    return q
                add(f"warp {code} on osc {osc}", f)
        if o.unison > 1:
            def f(q, i=i):
                q.oscs[i].unison = 1
                return q
            add(f"unison x{o.unison} ({o.unison_mode}) on osc {osc}", f)
        if len(active) > 1:
            def f(q, i=i):
                q.oscs[i].enabled = False
                return q
            add(f"osc {osc} ({o.type})", f)
    if patch.noise.enabled:
        def f(q):
            q.noise.enabled = False
            return q
        add("noise oscillator", f)
    if patch.sub.enabled:
        def f(q):
            q.sub.enabled = False
            return q
        add("sub oscillator", f)
    for j, m in enumerate(patch.mods):
        if m.amount and not getattr(m, "bypass", False) and not m.source.startswith("macro"):
            def f(q, j=j):
                q.mods[j].amount = 0.0
                return q
            add(f"mod {m.source} -> {m.dest}", f)
    if patch.glide > 0:
        def f(q):
            q.glide = 0.0
            return q
        add("glide", f)
    return out[:MAX_COMPONENTS]


def _gap(ours, serum, sr) -> float:
    """How different two renders of the same preset are (0 = identical)."""
    if not np.any(np.abs(serum) > 1e-7) or not np.any(np.abs(ours) > 1e-7):
        return 100.0
    return 100.0 - Reference(np.asarray(serum, dtype=np.float64), sr).spectral(ours)


def find_culprits(patch: Patch, bank, samples, notes, duration: float, sr: int, session, render_ours,
                  build, time_budget: float = 180.0, stop=None) -> dict:
    """Switch each component off in both engines; the one whose removal closes
    the gap most is what we emulate wrongly."""
    t0 = time.time()
    base_o = render_ours(patch)
    base_s = session.render(build(patch), notes, duration)
    g0 = _gap(base_o, base_s, sr)
    rows = []
    for label, fn in components(patch):
        if time.time() - t0 > time_budget or (stop and stop()):
            break
        q = fn(patch.copy())
        try:
            g = _gap(render_ours(q), session.render(build(q), notes, duration), sr)
        except Exception as e:                 # a failed render should not end the analysis
            log.warning("culprit render failed for %s: %s", label, e)
            continue
        rows.append({"component": label, "gap_without": round(g, 2), "explains": round(g0 - g, 2)})
    rows.sort(key=lambda r: -r["explains"])
    return {"gap": round(g0, 2), "components": rows, "seconds": round(time.time() - t0, 1)}


# ----------------------------------------------------------------------------
# one run

def record_run(*, patch: Patch, ours: np.ndarray, serum: np.ndarray, sr: int, notes, our_score: float,
               serum_score: float, culprits: dict | None, tuning: dict | None, meta: dict) -> dict:
    """Describe the engine gap of one run, learn from the tuning moves and
    append the record to the log. Returns the record."""
    diff = describe_difference(ours, serum, sr, notes)
    findings = list(diff.get("findings", []))
    if culprits and culprits.get("components"):
        top = [c for c in culprits["components"] if c["explains"] > 1.0][:3]
        for c in top:
            findings.insert(0, f"{c['component']} explains {c['explains']:.1f} of the {culprits['gap']:.1f} % "
                               f"difference between our engine and Serum (switching it off brings them to "
                               f"{c['gap_without']:.1f} %)")
    learned = []
    if tuning and tuning.get("changes") and tuning.get("learnable", True):
        _, learned = learn(tuning["changes"])
        moved = sorted((c for c in tuning["changes"] if c["before"] and
                        (abs(math.log(max(c["after"], 1e-9) / max(c["before"], 1e-9))) > math.log(1.1)
                         if c["kind"] == "ratio" else abs(c["after"] - c["before"]) > 0.05 * c["span"])),
                       key=lambda c: -abs(c["after"] - c["before"]) / max(c["span"], 1e-9))
        for c in moved[:4]:
            findings.append(f"the Serum tuning moved {c['key']} from {c['before']:.4g} to {c['after']:.4g}")
    findings += [f"learned correction: {t}" for t in learned]
    rec = {
        "t": time.time(), "version": __import__("fourier_analyzer").__version__, **meta,
        "our_score": round(our_score, 2), "serum_score": round(serum_score, 2),
        "score_drop": round(our_score - serum_score, 2),
        "engine_gap": diff, "culprits": culprits,
        "tuning": {k: v for k, v in (tuning or {}).items() if k != "details"},
        "learned": learned, "findings": findings,
        "features": features(patch),
    }
    try:
        with log_path().open("a") as fh:
            fh.write(json.dumps(rec, default=_jsonable) + "\n")
    except OSError as e:
        log.warning("could not write the calibration log: %s", e)
    return rec


def _jsonable(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def features(patch: Patch) -> list[str]:
    """The Serum features a patch uses (to group the log by feature)."""
    out = []
    for i, o in patch.active_oscs():
        out.append(f"osc:{o.type}")
        for code in (o.warp, o.warp2):
            if code != "none":
                out.append(f"warp:{code}")
        if o.unison > 1:
            out.append(f"unison:{o.unison_mode}")
    for name in ("filter", "filter2"):
        f = getattr(patch, name)
        if f.enabled:
            out.append(f"filter:{f.type}")
    out += [f"fx:{m.get('type')}" for m in patch.fx_chain]
    out += [f"mod:{_src_kind(m.source)}->{m.dest}" for m in patch.mods if m.amount]
    if patch.noise.enabled:
        out.append("noise")
    if patch.sub.enabled:
        out.append("sub")
    return sorted(set(out))


def report_text(rec: dict) -> str:
    """Human-readable engine-gap report for the results folder."""
    lines = [f"Our engine predicted {rec['our_score']} % similarity to the original; real Serum 2 scored "
             f"{rec['serum_score']} % (difference {rec['score_drop']:+.2f}).", ""]
    eg = rec["engine_gap"]
    if "similarity" in eg:
        s = eg["similarity"]
        lines.append(f"Our render vs Serum's render of the same preset: waveform {s['waveform_similarity']} %, "
                     f"spectral {s['spectral_similarity']} %, log-spectral distance {s['lsd_db']} dB.")
    lines += ["", "What differs:"] + [f"- {f}" for f in rec["findings"]] or ["- nothing measurable"]
    if eg.get("band_regions"):
        lines += ["", "Frequency bands (Serum minus ours, after matching overall level):"]
        lines += [f"  {b['hz']:>8} Hz  {b['diff_db']:+.1f} dB" for b in eg.get("bands", [])]
    cul = rec.get("culprits") or {}
    if cul.get("components"):
        lines += ["", f"Culprit search (gap {cul['gap']} %; each component switched off in both engines):"]
        lines += [f"  {c['component']}: gap {c['gap_without']} % without it ({c['explains']:+.1f})"
                  for c in cul["components"]]
    lines += ["", "Features used: " + ", ".join(rec.get("features", []))]
    return "\n".join(lines) + "\n"


def emulation_key(label: str) -> str | None:
    """'FX hyperdim' / 'FX bus 1 reverb' / 'warp BendPos on osc A' /
    'together: warp Sync on osc A + LFO2 on warp amount' -> 'FX hyperdim',
    'FX reverb', 'warp BendPos' (None for parts that are not an emulation)."""
    import re
    m = re.search(r"\bFX (?:bus \d )?([a-z0-9]+)", label)
    if m and m.group(1) not in ("order",):
        return f"FX {m.group(1)}"
    m = re.search(r"\b(FM|AM|RM) warp", label)
    if m:
        return f"warp {m.group(1)}"
    m = re.search(r"\bwarp (?!on\b)(\w+)", label)
    if m:
        return f"warp {m.group(1)}"
    return None


def fix_first(recs: list[dict] | None = None) -> list[dict]:
    """FX modules and warps ranked by how much they make our engine differ
    from real Serum: the culprit search's gap explained, plus how often and
    how far Serum disagreed when they were added or removed."""
    recs = read_log() if recs is None else recs
    rows: dict[str, dict] = {}

    def row(k):
        return rows.setdefault(k, {"emulation": k, "explained": 0.0, "culprit_runs": 0, "checks": 0,
                                   "disagreed": 0, "gap_sum": 0.0})
    for r in recs:
        for c in (r.get("culprits") or {}).get("components", []):
            k = emulation_key(c["component"])
            if k and c.get("explains", 0) > 0:
                x = row(k)
                x["explained"] += c["explains"]
                x["culprit_runs"] += 1
    for c in read_checks():
        k = emulation_key(c.get("part", ""))
        if not k:
            continue
        x = row(k)
        x["checks"] += 1
        if not c.get("agreed", True):
            x["disagreed"] += 1
            x["gap_sum"] += abs(c.get("engine_delta", 0.0) - c.get("serum_delta", 0.0))
    out = []
    for x in rows.values():
        x["priority"] = round(x["explained"] + x["gap_sum"], 2)
        x["explained"] = round(x["explained"], 2)
        x["gap_sum"] = round(x["gap_sum"], 2)
        if x["priority"] > 0:
            out.append(x)
    return sorted(out, key=lambda x: -x["priority"])[:20]


def summary() -> dict:
    """Totals over every logged run for the Engine accuracy page."""
    recs = read_log()
    comp: dict[str, list] = {}
    feat: dict[str, list] = {}
    bands: dict[float, list] = {}
    for r in recs:
        for c in (r.get("culprits") or {}).get("components", []):
            comp.setdefault(c["component"].split(" on osc")[0].split(" (")[0], []).append(c["explains"])
        for f in r.get("features", []):
            feat.setdefault(f, []).append(r.get("score_drop", 0.0))
        for b in r.get("engine_gap", {}).get("bands", []):
            bands.setdefault(b["hz"], []).append(b["diff_db"])
    corr = load_corrections()
    by_part: dict[str, list] = {}
    for c in read_checks():
        by_part.setdefault(f"{'add' if c.get('kind') == 'add' else 'remove'}: {c.get('kind_of_part', c.get('part'))}",
                           []).append(c)
    structure = sorted(({"part": k, "checks": len(v), "disagreed": sum(not c["agreed"] for c in v),
                         "avg_engine_delta": round(float(np.mean([c["engine_delta"] for c in v])), 2),
                         "avg_serum_delta": round(float(np.mean([c["serum_delta"] for c in v])), 2)}
                        for k, v in by_part.items()), key=lambda x: (-x["disagreed"], -x["checks"]))
    return {
        "fix_first": fix_first(recs),
        "structure_checks": structure[:30],
        "runs": len(recs),
        "avg_drop": round(float(np.mean([r.get("score_drop", 0) for r in recs])), 2) if recs else None,
        "avg_engine_similarity": round(float(np.mean([r["engine_gap"]["similarity"]["spectral_similarity"]
                                                      for r in recs if "similarity" in r.get("engine_gap", {})])), 2)
        if recs else None,
        "culprits": sorted(({"component": k, "runs": len(v), "avg_explains": round(float(np.mean(v)), 2)}
                            for k, v in comp.items()), key=lambda x: -x["avg_explains"])[:20],
        "features": sorted(({"feature": k, "runs": len(v), "avg_drop": round(float(np.mean(v)), 2)}
                            for k, v in feat.items()), key=lambda x: -x["avg_drop"])[:30],
        "bands": [{"hz": k, "avg_diff_db": round(float(np.mean(v)), 2), "runs": len(v)} for k, v in sorted(bands.items())],
        "corrections": [{"key": k, **v} for k, v in sorted(corr["params"].items())],
        "pending": {k: len(v) for k, v in corr["pending"].items() if v},
        "recent": [{"t": r["t"], "name": r.get("name"), "our_score": r["our_score"], "serum_score": r["serum_score"],
                    "findings": r.get("findings", [])[:5]} for r in recs[-10:]][::-1],
    }
