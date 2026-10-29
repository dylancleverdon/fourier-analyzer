"""Fourier / signal analysis of an isolated sound.

Produces everything the patch builder needs as a starting point:
notes + timing, harmonic magnitudes and phases per frame, amplitude envelope
stats, noise content, unison/detune hints, periodic modulation (LFO) estimates
and tempo-grid alignment.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from ..synth.patch import Note, SYNC_DIVISIONS, sync_to_hz

HOP = 256
N_FFT = 4096
N_HARM = 256


@dataclass
class Analysis:
    sr: int
    duration: float
    bpm: float
    notes: list
    times: np.ndarray                  # frame times (s)
    f0: np.ndarray                     # Hz per frame (nan = unvoiced)
    voiced_ratio: float
    rms_db: np.ndarray
    centroid: np.ndarray               # Hz per frame
    harm_mag: np.ndarray               # (frames, N_HARM) linear magnitudes
    harm_phase: np.ndarray             # (frames, N_HARM) phase relative to fundamental
    harm_profile_db: np.ndarray        # (N_HARM,) median dB profile (normalised)
    noise_ratio: float                 # 0..1 energy between harmonics / total
    noise_tilt: float                  # dB/oct of the noise floor
    unison_hint: float                 # 0..1 (partial width / beating)
    env: dict                          # attack, decay, sustain, release (s / 0..1)
    percussive: bool
    lfo: dict | None                   # {feature, period_s, sync, shape_points, depth, mode}
    grid: dict = field(default_factory=dict)   # timing report
    # pitch movement that belongs in the patch rather than in MIDI bends
    pitch_lfo: dict | None = None      # vibrato / synced pitch wobble: detect_lfo() + depth in semitones
    pitch_env: dict | None = None      # recurring per-note pitch sweep: {semis, decay_s}
    glide: float = 0.0                 # portamento time between legato notes (s)
    notes_raw: list | None = None      # notes with their full measured bends
    spectrum_db: np.ndarray | None = None      # long term average spectrum (for UI)
    spectrum_hz: np.ndarray | None = None

    def summary(self) -> dict:
        return {
            "bpm": round(self.bpm, 2),
            "duration": round(self.duration, 3),
            "notes": [n.to_dict() for n in self.notes],
            "voiced_ratio": round(self.voiced_ratio, 3),
            "noise_ratio": round(self.noise_ratio, 3),
            "unison_hint": round(self.unison_hint, 3),
            "envelope": {k: round(v, 4) for k, v in self.env.items()},
            "percussive": self.percussive,
            "lfo": self.lfo,
            "pitch_lfo": self.pitch_lfo,
            "pitch_env": self.pitch_env,
            "glide": round(self.glide, 4),
            "timing": self.grid,
            "top_harmonics_db": [round(float(x), 1) for x in self.harm_profile_db[:16]],
        }


def tempo_candidates(y: np.ndarray, sr: int, lo: float = 70.0, hi: float = 190.0):
    """Tempo estimation with sub-BPM precision.

    1. Onset-strength envelope at ~6 ms resolution.
    2. For every tempo in 0.05 BPM steps, average the (overlap-normalised)
       autocorrelation at 1..8 beat lags. Interpolating at fractional lags
       gives precision far below the frame rate.
    3. Multiply by a broad prior centred on 140 BPM (bass music) to resolve
       half/double-time ambiguity.
    4. Refine the winner by fitting a beat grid (tempo + phase) to the onsets,
       and snap to a whole BPM when that fits equally well.

    Returns a list of {"bpm", "score"} sorted best first, including the
    half/double/2:3 alternatives so the UI can offer them."""
    import librosa
    hop = 128
    fps = sr / hop
    env = librosa.onset.onset_strength(y=y, sr=sr, hop_length=hop, aggregate=np.mean, n_mels=96)
    env = np.maximum(env - np.median(env), 0)
    n = len(env)
    if n < 16 or not np.any(env):
        return [{"bpm": 140.0, "score": 0.0}]
    f = np.fft.rfft(env - env.mean(), 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac = ac / (n - np.arange(n))           # unbiased
    ac = ac / (ac[0] + 1e-12)
    bpms = np.arange(lo, hi + 1e-9, 0.05)
    period = fps * 60.0 / bpms
    comb = np.zeros_like(bpms)
    cnt = np.zeros_like(bpms)
    for k in range(1, 9):
        lag = period * k
        ok = lag < n * 0.7
        comb[ok] += np.interp(lag[ok], np.arange(n), ac)
        cnt[ok] += 1
    comb = np.where(cnt > 0, comb / np.maximum(cnt, 1), 0)
    prior = np.exp(-0.5 * (np.log2(bpms / 140.0) / 0.55) ** 2)
    score = np.maximum(comb, 0) * prior

    def grid_fit(bpm):
        per = fps * 60.0 / bpm
        best = 0.0
        for ph in np.linspace(0, per, 96, endpoint=False):
            pos = np.arange(ph, n - 1, per / 2)   # beats and off-beats
            w = np.where(np.arange(len(pos)) % 2 == 0, 1.0, 0.5)
            v = float((np.interp(pos, np.arange(n), env) * w).sum() / max(1, w.sum()))
            best = max(best, v)
        return best

    # local maxima of the score as candidates
    peaks = [i for i in range(1, len(score) - 1) if score[i] >= score[i - 1] and score[i] >= score[i + 1]]
    peaks = sorted(peaks, key=lambda i: -score[i])[:6]
    out = []
    for i in peaks:
        c = bpms[i]
        fine = np.arange(c - 0.6, c + 0.6, 0.02)
        fits = np.array([grid_fit(x) for x in fine])
        bpm = float(fine[np.argmax(fits)])
        r = round(bpm)
        if abs(bpm - r) < 0.4 and grid_fit(r) >= fits.max() * 0.95:
            bpm = float(r)
        elif abs(bpm * 2 - round(bpm * 2)) < 0.2 and grid_fit(round(bpm * 2) / 2) >= fits.max() * 0.95:
            bpm = round(bpm * 2) / 2
        out.append({"bpm": round(bpm, 2), "score": round(float(score[i]), 4)})
    # deduplicate (within 0.5 BPM) and add musical relatives of the winner
    uniq = []
    for c in out:
        if all(abs(c["bpm"] - u["bpm"]) > 0.5 for u in uniq):
            uniq.append(c)
    # alternatives = musical relatives of the winner only (half/double time,
    # triplet feel); unrelated autocorrelation peaks just confuse
    best = uniq[0]
    out = [best]
    for mult in (0.5, 2.0, 2 / 3, 1.5):
        alt = round(best["bpm"] * mult * 2) / 2
        if 50 <= alt <= 220:
            out.append({"bpm": alt, "score": 0.0})
    return out


def estimate_bpm(y: np.ndarray, sr: int) -> float:
    return float(tempo_candidates(y, sr)[0]["bpm"])


def _frames_stft(y, sr):
    pad = N_FFT // 2
    yp = np.pad(y, (pad, pad))
    nfr = 1 + (len(yp) - N_FFT) // HOP
    idx = np.arange(N_FFT)[None, :] + HOP * np.arange(nfr)[:, None]
    win = np.hanning(N_FFT)
    # zero-phase framing: rotate each frame so its centre is sample 0, so the
    # phase of every bin is referenced to the frame centre (with a symmetric
    # window this makes harmonic phases independent of bin mis-centring)
    frames = np.fft.ifftshift(yp[idx] * win[None, :], axes=1)
    return np.fft.rfft(frames, axis=1)   # (frames, bins)


def _short_rms_db(y: np.ndarray, n_frames: int, win: int = 1024) -> np.ndarray:
    """Loudness per STFT frame from a short window (sharp attacks), dB."""
    yp = np.pad(y.astype(np.float64), (win // 2, win // 2 + HOP * n_frames))
    c = np.concatenate([[0.0], np.cumsum(yp ** 2)])
    starts = np.arange(n_frames) * HOP
    e = (c[starts + win] - c[starts]) / win
    return 10 * np.log10(e + 1e-12)


def track_f0(Sx: np.ndarray, sr: int, fmin: float = 27.5, fmax: float = 1200.0):
    """Fast harmonic-sum f0 tracker on an existing STFT.

    For every candidate f0 (1/8 semitone grid) sum compressed magnitudes at
    the first 10 harmonics with decaying weights (true f0 beats its octave
    errors because f0/2 misses odd harmonics and 2*f0 misses even ones).
    The estimate is refined from the measured harmonic peak frequencies.
    Returns (f0 Hz with nan for unvoiced, voicing confidence 0..1)."""
    Am = np.abs(Sx)
    # remove window leakage / noise floor so sidelobes cannot add up for
    # sub-octave candidates, then compress
    floor = np.median(Am, axis=1, keepdims=True) * 4
    A = np.maximum(Am - floor, 0) ** 0.5
    nfr, nb = A.shape
    binhz = sr / N_FFT
    cands = fmin * 2 ** (np.arange(0, 12 * np.log2(fmax / fmin) * 8 + 1) / 96.0)
    K = 10
    w = 0.82 ** np.arange(K)
    pos = (cands[:, None] * np.arange(1, K + 1)[None, :]) / binhz        # (C, K)
    valid = pos < nb - 2
    p0 = np.clip(pos.astype(int), 0, nb - 2)
    fr = pos - p0
    scores = np.zeros((nfr, len(cands)))
    for k in range(K):
        a = A[:, p0[:, k]] * (1 - fr[:, k]) + A[:, p0[:, k] + 1] * fr[:, k]
        # take the local max over +-1 bin to tolerate detune / inharmonicity
        a = np.maximum(a, np.maximum(A[:, np.clip(p0[:, k] - 1, 0, nb - 1)], A[:, np.clip(p0[:, k] + 2, 0, nb - 1)]) * 0.8)
        scores += w[k] * a * valid[None, :, k]
    # favour candidates whose fundamental sits on a real spectral peak
    p1 = p0[:, 0]
    loc = np.maximum(A[:, np.clip(p1, 0, nb - 1)], A[:, np.clip(p1 + 1, 0, nb - 1)])
    nbr = np.maximum(A[:, np.clip(p1 - 3, 0, nb - 1)], A[:, np.clip(p1 + 4, 0, nb - 1)])
    peaky = (loc > nbr * 1.2).astype(float)
    scores = scores * (0.75 + 0.25 * peaky)
    best = np.argmax(scores, axis=1)
    f0 = cands[best].astype(float)
    tot = A.sum(1) + 1e-12
    conf = scores[np.arange(nfr), best] / (tot / np.sqrt(nb) * w.sum() * 6 + 1e-12)
    conf = np.clip(conf / (np.percentile(conf, 95) + 1e-12), 0, 1)
    # refine with measured peak positions of harmonics 1..4
    for i in range(nfr):
        est, wt = 0.0, 0.0
        for k in range(1, 5):
            b = int(round(k * f0[i] / binhz))
            if b < 2 or b >= nb - 2:
                continue
            seg = Am[i, b - 2:b + 3]
            j = int(np.argmax(seg)) + b - 2
            y0, y1, y2 = Am[i, j - 1], Am[i, j], Am[i, j + 1]
            den = y0 - 2 * y1 + y2
            d = 0.5 * (y0 - y2) / den if den != 0 else 0.0
            est += (j + d) * binhz / k * y1
            wt += y1
        if wt > 0:
            f0[i] = est / wt
    from scipy.ndimage import median_filter
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = 12 * np.log2(np.maximum(f0, 1e-3) / 440.0) + 69
    midi = median_filter(midi, size=5, mode="nearest")
    f0 = 440.0 * 2 ** ((midi - 69) / 12)
    f0[conf < 0.25] = np.nan
    return f0, conf


def harmonics(Sx: np.ndarray, f0: np.ndarray, sr: int, n_harm: int = N_HARM):
    """Sample magnitude and phase at k*f0 (parabolic peak refinement)."""
    nfr = min(len(f0), Sx.shape[0])
    mag = np.zeros((nfr, n_harm))
    ph = np.zeros((nfr, n_harm))
    binhz = sr / N_FFT
    A = np.abs(Sx)
    for i in range(nfr):
        f = f0[i]
        if not np.isfinite(f) or f <= 0:
            continue
        ks = np.arange(1, n_harm + 1)
        fk = ks * f
        ok = fk < sr / 2 * 0.95
        b = np.round(fk[ok] / binhz).astype(int)
        # search +-2 bins for the actual peak (inharmonicity / detune)
        cand = b[:, None] + np.arange(-2, 3)[None, :]
        cand = np.clip(cand, 1, A.shape[1] - 2)
        best = cand[np.arange(len(b)), np.argmax(A[i][cand], axis=1)]
        mag[i, : ok.sum()] = A[i][best] / (N_FFT / 4)
        ph[i, : ok.sum()] = np.angle(Sx[i][best])
    # phases relative to fundamental: phi_k - k * phi_1
    rel = ph - np.arange(1, n_harm + 1)[None, :] * ph[:, :1]
    rel = np.angle(np.exp(1j * rel))
    return mag, rel


def _refine_onset(y, sr, t):
    """Move an onset to where the energy starts rising into the attack near t.
    The frame-level detector can fire up to one loudness window (~46 ms)
    early, so the search reaches forward that far."""
    w = 64
    i0 = max(0, int((t - 0.02) * sr))
    i1 = min(len(y) - w, int((t + 0.06) * sr))
    if i1 <= i0 + w:
        return t
    seg = y[i0:i1 + w] ** 2
    e = np.convolve(seg, np.ones(w) / w, mode="valid")
    k_max = int(np.argmax(e))
    thr = 0.03 * e[k_max]                       # -15 dB below the attack peak
    quiet = np.nonzero(e[:k_max + 1] < thr)[0]
    if len(quiet):                              # attack out of (near) silence
        return (i0 + int(quiet[-1]) + 1) / sr
    # retrigger inside a sounding note: steepest rise among loud frames
    d = np.diff(10 * np.log10(e + 1e-10))
    d[e[1:] < 0.1 * e[k_max]] = -np.inf
    return (i0 + int(np.argmax(d))) / sr


def _segment_notes(y, sr, f0, conf, times, rms_db, percussive_hint):
    """Split into notes on silence gaps, sharp amplitude retriggers and pitch
    jumps. Smooth timbre/volume wobbles (LFOs) and slow slides do not split."""
    n = len(times)
    floor = rms_db.max() - 35
    active = rms_db > floor
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = np.where(np.isfinite(f0), 12 * np.log2(np.maximum(f0, 1e-3) / 440.0) + 69, np.nan)
    bounds = [0] if active[0] else []
    pitch_bounds = set()
    refine_ok = set()
    for i in range(1, n):
        if not active[i]:
            continue
        if not active[i - 1]:
            bounds.append(i)
            continue
        # retrigger: >9 dB rise inside two frames right after a local dip,
        # at least 70 ms after the previous attack
        far = not bounds or (times[i] - times[bounds[-1]]) > 0.07
        if far and i >= 2 and rms_db[i] - rms_db[i - 2] > 9 and rms_db[i - 1] <= rms_db[i - 2] + 4.5:
            bounds.append(i)
            continue
        # pitch jump: >0.8 semitone change within two frames that persists
        # (ignored right after an attack, where drums/plucks sweep pitch)
        recent = bounds and (times[i] - times[bounds[-1]]) < 0.08
        if not percussive_hint and not recent and i >= 6 and i + 5 < n:
            # compare a steady window before with the window from i on; frames
            # right at the transition may be unvoiced (clicks, retriggers)
            before = midi[i - 6:i - 1]
            after = midi[i:i + 5]
            if np.isfinite(before).sum() >= 3 and np.isfinite(after).sum() >= 3:
                mb = np.nanmedian(before)
                jump = np.nanmedian(after) - mb
                steady = np.nanmax(before) - np.nanmin(before) < 0.6   # slides are not steady
                prev_ok = not np.isfinite(midi[i - 1]) or abs(midi[i - 1] - mb) < 0.5
                cur_new = not np.isfinite(midi[i]) or abs(midi[i] - mb) > 0.5
                if abs(jump) > 0.8 and steady and prev_ok and cur_new:
                    # a legato note starts where the pitch leaves the old note
                    # (a glide), not where it arrives
                    j = i
                    target = mb + jump
                    margin = max(0.25, 0.05 * abs(jump))
                    while j - 1 > (bounds[-1] if bounds else 0) and active[j - 1] and np.isfinite(midi[j - 1]) \
                            and rms_db[j - 1] > rms_db[i] - 10 \
                            and min(mb, target) + margin < midi[j - 1] < max(mb, target) and i - j < 12:
                        j -= 1
                    bounds.append(j)
                    pitch_bounds.add(j)
                    if j == i:
                        refine_ok.add(j)
    # a pitch change right before a real attack is the old note's fading tail
    bounds = sorted(set(bounds))
    bounds = [b for k, b in enumerate(bounds)
              if not (b in pitch_bounds and k + 1 < len(bounds) and bounds[k + 1] not in pitch_bounds
                      and times[bounds[k + 1]] - times[b] < 0.08)]
    # merge boundaries closer than 40 ms
    merged = []
    for b in bounds:
        if not merged or times[b] - times[merged[-1]] > 0.04:
            merged.append(b)
    notes = []
    dur_total = len(y) / sr
    for k, b in enumerate(merged):
        e_lim = merged[k + 1] if k + 1 < len(merged) else n
        seg_active = np.nonzero(active[b:e_lim])[0]
        if len(seg_active) == 0:
            continue
        last = b + int(seg_active[-1])
        t0 = (_refine_onset(y, sr, float(times[b])) if (b not in pitch_bounds or b in refine_ok) else float(times[b])) if b > 0 else 0.0
        t_next = times[e_lim] if e_lim < n else dur_total
        t_end = min(float(times[last]) + HOP / sr, t_next)
        dur = max(t_end - t0, 0.02)
        mseg = midi[b:last + 1]
        good = np.isfinite(mseg)
        bend = None
        if good.sum() >= 2:
            track = mseg[good]
            # note pitch = where the note settles after its attack (first third)
            third = max(3, len(track) // 3)
            pitch = float(np.median(track[min(2, len(track) - 1):third]))
            rel = track - pitch
            from scipy.ndimage import median_filter
            sm = median_filter(rel, size=3, mode="nearest")
            if (np.abs(sm) > 0.12).sum() >= 4 and not percussive_hint:
                rel = sm
                tt = times[b:last + 1][good] - t0
                step = max(1, int(0.02 / (HOP / sr)))
                bend = [[round(float(a), 4), round(float(v), 3)] for a, v in zip(tt[::step], rel[::step])]
        else:
            pitch = 36.0 if percussive_hint else 48.0
        lv = float(np.clip(10 ** ((rms_db[b:last + 1].max() - rms_db.max()) / 40), 0.2, 1.0))
        notes.append(Note(start=round(float(t0), 5), duration=round(dur, 4), pitch=round(pitch, 3),
                          velocity=round(lv, 3), bend=bend))
    if not notes:
        notes = [Note(0.0, dur_total, 48.0)]
    return notes


def refine_pitches(y: np.ndarray, sr: int, notes) -> None:
    """Sub-cent pitch per note: one long zero-padded FFT over the steady part
    of each note (frequency resolution ~ 1/duration), quadratic peak
    interpolation on the first harmonics, magnitude-weighted average of
    f_k / k. Notes with pitch bends keep their tracked pitch."""
    for n in notes:
        if n.bend:
            continue
        a = int((n.start + min(0.03, n.duration * 0.2)) * sr)
        b = int((n.start + n.duration) * sr)
        seg = np.asarray(y[a:b], dtype=np.float64)
        if len(seg) < 2048:
            continue
        nfft = 1 << int(np.ceil(np.log2(len(seg) * 8)))
        S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), nfft))
        binhz = sr / nfft
        f0 = 440.0 * 2 ** ((n.pitch - 69) / 12)
        est, wt = 0.0, 0.0
        for k in range(1, 9):
            fk = k * f0
            if fk > sr / 2 * 0.9:
                break
            lo = int((fk * 2 ** (-0.4 / 12)) / binhz)
            hi = int((fk * 2 ** (0.4 / 12)) / binhz) + 1
            if hi - lo < 3:
                continue
            j = lo + int(np.argmax(S[lo:hi]))
            if j <= 0 or j >= len(S) - 1:
                continue
            l0, l1, l2 = np.log(S[j - 1:j + 2] + 1e-20)
            den = l0 - 2 * l1 + l2
            d = 0.5 * (l0 - l2) / den if den != 0 else 0.0
            est += (j + d) * binhz / k * S[j]
            wt += S[j]
        if wt > 0:
            f = est / wt
            new = 69 + 12 * np.log2(f / 440.0)
            if abs(new - n.pitch) < 0.5:
                n.pitch = round(float(new), 4)


def _env_stats(y, sr, notes, rms_db, times):
    """Attack/decay/sustain/release from the loudest note."""
    if not notes:
        return {"attack": 0.005, "decay": 0.3, "sustain": 0.8, "release": 0.05}
    lin = 10 ** (rms_db / 20)
    best = max(notes, key=lambda n: n.duration)
    i0, i1 = np.searchsorted(times, [best.start, best.start + best.duration])
    seg = lin[i0:max(i1, i0 + 2)]
    if len(seg) < 2:
        return {"attack": 0.005, "decay": 0.3, "sustain": 0.8, "release": 0.05}
    # attack peak = loudest point in the first 150 ms (later peaks are LFOs)
    early = seg[: max(2, int(0.15 * sr / HOP))]
    pk = int(np.argmax(early))
    peak = seg[pk]
    above10 = np.nonzero(seg[:pk + 1] >= 0.1 * peak)[0]
    t10 = above10[0] if len(above10) else 0
    attack = max((pk - t10) * HOP / sr, 0.0005)
    tail = seg[pk:]
    sustain = float(np.median(tail[len(tail) // 2:]) / peak) if len(tail) > 4 else 1.0
    target = sustain + (1 - sustain) * 0.01
    below = np.nonzero(tail <= max(target, 1e-4))[0]
    decay = (below[0] * HOP / sr) if len(below) else max(len(tail) * HOP / sr, 0.05)
    # release: decay after the note end until -40 dB
    j0 = i1
    nxt = [n.start for n in notes if n.start > best.start + best.duration - 1e-3]
    j1 = np.searchsorted(times, min(nxt) if nxt else times[-1])
    rel = 0.03
    if j1 > j0 + 1:
        after = lin[j0:j1]
        drop = np.nonzero(after <= after[0] * 0.01)[0]
        rel = (drop[0] * HOP / sr) if len(drop) else (len(after) * HOP / sr)
    return {"attack": float(attack), "decay": float(min(decay, 8.0)),
            "sustain": float(np.clip(sustain, 0, 1)), "release": float(np.clip(rel, 0.005, 4.0))}


def _noise_stats(Sx, f0, sr):
    """Energy between harmonic peaks relative to total, and its spectral tilt."""
    A2 = np.abs(Sx) ** 2
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    nfr = min(len(f0), A2.shape[0])
    inter, total, tilt_acc = 0.0, 0.0, []
    for i in range(0, nfr, 2):
        f = f0[i]
        row = A2[i]
        tot = row.sum()
        if tot <= 0:
            continue
        if np.isfinite(f) and f > 0:
            dist = np.abs(freqs / f - np.round(freqs / f))
            mask = (dist > 0.3) & (freqs > f * 0.7)
        else:
            mask = freqs > 30
        e = row[mask].sum() * (1.0 / max(mask.mean(), 1e-3)) * 0.4 if np.isfinite(f) else row[mask].sum()
        inter += min(e, tot)
        total += tot
        if mask.sum() > 20:
            fb = freqs[mask]
            vb = 10 * np.log10(row[mask] + 1e-12)
            sel = fb > 200
            if sel.sum() > 10:
                tilt_acc.append(np.polyfit(np.log2(fb[sel]), vb[sel], 1)[0])
    ratio = inter / total if total > 0 else 0.0
    tilt = float(np.median(tilt_acc)) if tilt_acc else -6.0
    return float(np.clip(ratio, 0, 1)), tilt


def _unison_hint(Sx, f0, sr):
    """Width of harmonic peaks relative to a single sinusoid (beating/detune)."""
    A = np.abs(Sx)
    binhz = sr / N_FFT
    widths = []
    for i in range(0, min(len(f0), A.shape[0]), 4):
        f = f0[i]
        if not np.isfinite(f):
            continue
        for k in range(3, 12):
            b = int(round(k * f / binhz))
            if b + 6 >= A.shape[1]:
                break
            w = A[i, b - 6:b + 7]
            if w.max() <= 0:
                continue
            widths.append((w > w.max() * 0.25).sum())
    if not widths:
        return 0.0
    # a clean hann peak at N_FFT=4096 is ~3-4 bins above 25 %
    return float(np.clip((np.median(widths) - 4) / 6, 0, 1))


def detect_lfo(times, feature: np.ndarray, notes, bpm, name: str):
    """Find periodic modulation of a per-frame feature inside sustained notes."""
    best = None
    dt = times[1] - times[0] if len(times) > 1 else HOP / 22050
    for n in notes:
        if n.duration < 0.25:
            continue
        i0, i1 = np.searchsorted(times, [n.start + 0.02, n.start + n.duration])
        x = feature[i0:i1]
        if len(x) < 16 or np.std(x) < 1e-6:
            continue
        x = x - np.polyval(np.polyfit(np.arange(len(x)), x, 1), np.arange(len(x)))
        ac = np.correlate(x, x, mode="full")[len(x) - 1:]
        ac /= ac[0] + 1e-12
        lo = max(2, int(0.03 / dt))
        hi = min(len(ac) - 1, int(len(x) * 0.6))
        neg = np.nonzero(ac[1:hi] < 0)[0]
        if len(neg):                       # skip the zero-lag lobe
            lo = max(lo, int(neg[0]) + 1)
        if hi <= lo:
            continue
        k = lo + int(np.argmax(ac[lo:hi]))
        strength = float(ac[k])
        if best is None or strength > best["strength"]:
            best = {"strength": strength, "period": k * dt, "note": n, "x": x, "i0": i0}
    if best is None or best["strength"] < 0.35:
        return None
    period = best["period"]
    # snap to a tempo-synced division if close (within 6 %)
    sync, err = None, 1e9
    for d in SYNC_DIVISIONS:
        p = 1.0 / sync_to_hz(d, bpm)
        e = abs(p - period) / period
        if e < err:
            sync, err = d, e
    if err > 0.06:
        sync = None
    else:
        period = 1.0 / sync_to_hz(sync, bpm)
    # fold the feature into one cycle (aligned to note start => trig mode)
    x = best["x"]
    # phase relative to the note start, so the shape lines up with a retriggered LFO
    t_rel = times[best["i0"]: best["i0"] + len(x)] - best["note"].start
    ph = (t_rel / period) % 1.0
    bins = np.linspace(0, 1, 17)
    cyc = np.array([x[(ph >= a) & (ph < b)].mean() if np.any((ph >= a) & (ph < b)) else np.nan
                    for a, b in zip(bins[:-1], bins[1:])])
    cyc = np.nan_to_num(cyc, nan=np.nanmean(cyc) if np.any(np.isfinite(cyc)) else 0.0)
    span = cyc.max() - cyc.min()
    pts = (cyc - cyc.min()) / span if span > 0 else np.zeros_like(cyc)
    return {
        "feature": name,
        "period_s": round(float(period), 4),
        "rate_hz": round(1.0 / float(period), 4),
        "sync": sync,
        "strength": round(best["strength"], 3),
        "depth": round(float(span), 4),
        "shape_points": [[round(float(a), 4), round(float(b), 4)]
                         for a, b in zip(np.linspace(0, 1, len(pts) + 1)[:-1], pts)],
        "mode": "trig",
    }


def _note_rel_tracks(notes, times, f0):
    """Per note: (frame times relative to the note start, pitch relative to
    the note pitch in semitones) from the f0 track."""
    with np.errstate(invalid="ignore", divide="ignore"):
        midi = np.where(np.isfinite(f0), 12 * np.log2(np.maximum(f0, 1e-3) / 440.0) + 69, np.nan)
    out = []
    for n in notes:
        i0, i1 = np.searchsorted(times, [n.start, n.start + n.duration])
        seg = midi[i0:i1]
        good = np.isfinite(seg)
        idx = np.arange(i0, i1)[good]
        tt = times[idx] - n.start
        rel = seg[good] - n.pitch
        ok = np.abs(rel) < 24
        out.append((tt[ok], rel[ok], idx[ok]))
    return out


def _pitch_movement(notes, times, f0, bpm):
    """Pitch movement a Serum patch would make itself: an LFO on pitch
    (vibrato, synced wobbles), a pitch envelope that repeats on every note,
    and portamento between legato notes."""
    tracks = _note_rel_tracks(notes, times, f0)
    # --- periodic pitch modulation (LFO -> Main Tuning) -----------------------
    feat = np.zeros(len(times))
    for n, (tt, rel, idx) in zip(notes, tracks):
        if len(tt):
            # unvoiced frames inside the note take the neighbouring value
            seg = np.interp(np.arange(idx[0], idx[-1] + 1), idx, rel)
            feat[idx[0]:idx[-1] + 1] = seg
    pitch_lfo = detect_lfo(times, feat, notes, bpm, "pitch")
    if pitch_lfo is not None:
        # depth in semitones: peak-to-peak of the folded cycle
        if pitch_lfo["depth"] < 0.08 or pitch_lfo["strength"] < 0.45:
            pitch_lfo = None
        else:
            pitch_lfo["semis"] = round(float(pitch_lfo["depth"]), 3)
    # --- recurring pitch sweep at the note start (Env -> Main Tuning) ---------
    starts = []
    for tt, rel, _ in tracks:
        late = rel[(tt > 0.25) & (tt < 0.6)]
        base = float(np.median(late)) if len(late) else 0.0
        head = (tt < 0.15)
        if head.sum() >= 3 and len(rel) >= 6:
            hv = rel[head] - base
            if np.all(np.sign(hv) == np.sign(hv[0])) and abs(hv[0]) > 0.2:
                # exponential sweep: extrapolate log|offset| back to t = 0
                sl, ic = np.polyfit(tt[head], np.log(np.abs(hv) + 1e-6), 1)
                a0 = float(np.sign(hv[0]) * np.exp(ic)) if sl < 0 else float(hv[0])
            else:
                a0 = float(hv[0])
            starts.append((a0 + base, base, tt, rel))
    pitch_env = None
    big = [s for s in starts if abs(s[0] - s[1]) > 1.0]
    if len(starts) >= 2 and len(big) >= 0.6 * len(starts):
        sgn = np.sign(np.median([s[0] for s in big]))
        if all(np.sign(s[0]) == sgn for s in big):
            semis = float(np.median([s[0] for s in big]))
            # decay: time until the sweep has fallen to 1/e of its start
            taus = []
            for a0, _, tt, rel in big:
                below = np.nonzero(np.abs(rel) < abs(a0) / np.e)[0]
                if len(below):
                    taus.append(float(tt[below[0]]))
            if taus:
                tau = float(np.median(taus))
                # the long analysis window averages the first ~90 ms: fast sweeps
                # read smaller than they are; the optimizer tunes depth and time
                smear = float(np.clip(0.093 / max(tau, 0.01), 1.0, 3.0))
                pitch_env = {"semis": round(semis * smear, 3), "decay_s": round(tau * 4.6 / smear, 4)}
    # --- portamento between legato notes ---------------------------------------
    glide_times = []
    for k in range(1, len(notes)):
        a, b = notes[k - 1], notes[k]
        if b.start - (a.start + a.duration) > 0.06:
            continue
        tt, rel, _ = tracks[k]
        interval = a.pitch - b.pitch
        if abs(interval) < 1 or len(rel) < 4:
            continue
        r0 = np.median(rel[tt < 0.012]) if np.any(tt < 0.012) else rel[0]
        if abs(r0) > 0.15 * abs(interval) and np.sign(r0) == np.sign(interval):
            close = np.nonzero(np.abs(rel) < 0.3)[0]
            if len(close):
                glide_times.append(float(tt[close[0]]) * 1.5)
    glide = float(np.median(glide_times)) if len(glide_times) >= max(1, 0.5 * max(1, len(notes) - 1)) else 0.0
    return pitch_lfo, pitch_env, round(glide, 4)


def pitch_model(t, pitch_lfo, pitch_env, glide, prev_interval=0.0, bpm=140.0, parts=False):
    """Pitch offset (semitones) the patch itself produces at note-relative
    time ``t``: what gets removed from the MIDI bends. With ``parts`` returns
    (periodic part, transient part)."""
    t = np.asarray(t, dtype=float)
    trans = np.zeros_like(t)
    per = np.zeros_like(t)
    if pitch_env:
        trans += pitch_env["semis"] * np.exp(-4.6 * t / max(pitch_env["decay_s"], 1e-3))
    if glide and prev_interval:
        trans += prev_interval * np.exp(-t / (glide / 3.0))
    if pitch_lfo:
        pts = pitch_lfo["shape_points"]
        xs = [p[0] for p in pts] + [1.0]
        ys = [p[1] for p in pts] + [pts[0][1]]
        ph = np.mod(t / pitch_lfo["period_s"], 1.0)
        per += (np.interp(ph, xs, ys) - 0.5) * pitch_lfo["semis"]
    return (per, trans) if parts else per + trans


def strip_modelled_pitch(notes, pitch_lfo, pitch_env, glide, threshold: float = 0.15, window_s: float | None = None) -> None:
    """Remove the pitch movement the patch now produces from each note's MIDI
    bend so pitch is not applied twice.

    The analysis window smears fast sweeps, so a modelled transient (pitch
    envelope, glide) is not subtracted point by point: the start of the note
    (until the sweep has settled, smeared by the window) belongs to it, and
    only the rest is compared with the periodic (LFO) part. A bend is
    dropped when 90 % of what is left is within ``threshold`` semitones."""
    win = window_s if window_s is not None else N_FFT / 22050.0
    for k, n in enumerate(notes):
        if not n.bend:
            continue
        legato = k > 0 and n.start - (notes[k - 1].start + notes[k - 1].duration) <= 0.06
        tt = np.array([b[0] for b in n.bend], dtype=float)
        v = np.array([b[1] for b in n.bend], dtype=float)
        per, _ = pitch_model(tt, pitch_lfo, None, 0.0, parts=True)
        res = v - per
        settle = 0.02
        if pitch_env:
            settle = max(settle, 2 * pitch_env["decay_s"] / 4.6 * 2.3 + win / 2)
        if glide and legato:
            settle = max(settle, 2 * glide + win / 2)
        if settle > 0.02:
            settle = max(settle, 0.25)
        body = res[tt >= settle]
        if settle > 0.02 and len(body) >= 3 and abs(float(np.median(body))) > 0.05:
            # the note pitch was read while the sweep was settling: re-centre it
            off = float(np.median(body))
            n.pitch = round(float(n.pitch + off), 3)
            res = res - off
            body = body - off
        if len(body) == 0 or np.percentile(np.abs(body), 90) < threshold:
            n.bend = None
        else:
            n.bend = [[round(float(a), 4), round(float(b), 3)] for a, b in zip(tt, np.where(tt >= settle, res, 0.0))]


def analyze(y: np.ndarray, sr: int, bpm: float | None = None) -> Analysis:
    import librosa
    y = np.asarray(y, dtype=np.float32)
    duration = len(y) / sr
    if bpm is None:
        bpm = estimate_bpm(y, sr)
    Sx = _frames_stft(y, sr)
    times = np.arange(Sx.shape[0]) * HOP / sr
    rms_db = _short_rms_db(y, Sx.shape[0])
    freqs = np.fft.rfftfreq(N_FFT, 1 / sr)
    A = np.abs(Sx)
    centroid = (A * freqs[None, :]).sum(1) / (A.sum(1) + 1e-12)
    f0, vprob = track_f0(Sx, sr)
    f0[rms_db < rms_db.max() - 40] = np.nan
    loud = rms_db > rms_db.max() - 30
    voiced_ratio = float(np.mean(np.isfinite(f0[loud]))) if loud.any() else 0.0
    # short, mostly unvoiced events behave like drums
    percussive_hint = voiced_ratio < 0.35
    notes = _segment_notes(y, sr, f0, vprob, times, rms_db, percussive_hint)
    if not percussive_hint:
        refine_pitches(y, sr, notes)
    # fill f0 gaps inside notes with the note pitch for harmonic analysis
    f0_filled = f0.copy()
    for nt in notes:
        i0, i1 = np.searchsorted(times, [nt.start, nt.start + nt.duration])
        seg = f0_filled[i0:i1]
        seg[~np.isfinite(seg)] = librosa.midi_to_hz(nt.pitch)
    mag, rel = harmonics(Sx, f0_filled, sr)
    good = loud & np.isfinite(f0_filled)
    if good.any():
        prof = 20 * np.log10(np.median(mag[good], axis=0) + 1e-9)
    else:
        prof = np.full(N_HARM, -80.0)
    prof = prof - prof.max()
    noise_ratio, noise_tilt = _noise_stats(Sx, f0, sr)
    env = _env_stats(y, sr, notes, rms_db, times)
    med_dur = float(np.median([nt.duration for nt in notes]))
    percussive = percussive_hint or (env["sustain"] < 0.15 and env["decay"] < 0.4 and med_dur < 0.45)
    lfo = None
    cands = []
    for name, feat in (("brightness", np.log2(centroid + 1)), ("volume", rms_db / 20.0),
                       ("harmonics", 20 * np.log10(mag[:, 4:24].sum(1) + 1e-9) / 20)):
        r = detect_lfo(times, feat, notes, bpm, name)
        if r:
            cands.append(r)
    if cands:
        lfo = max(cands, key=lambda r: r["strength"] * (1 + r["depth"]))
    pitch_lfo, pitch_env, glide, notes_raw = None, None, 0.0, None
    if not percussive_hint:
        pitch_lfo, pitch_env, glide = _pitch_movement(notes, times, f0, bpm)
        if pitch_lfo or pitch_env or glide:
            import copy as _copy
            notes_raw = _copy.deepcopy(notes)
            strip_modelled_pitch(notes, pitch_lfo, pitch_env, glide, window_s=N_FFT / sr)
    beat = 60.0 / bpm
    grid = []
    for nt in notes:
        sixteenth = beat / 4
        q = round(nt.start / sixteenth) * sixteenth
        grid.append({"start": nt.start, "grid_16th": round(q, 4), "offset_ms": round((nt.start - q) * 1000, 1),
                     "length_16ths": round(nt.duration / sixteenth, 2)})
    ltas = 10 * np.log10(np.mean(np.abs(Sx[loud] if loud.any() else Sx) ** 2, axis=0) + 1e-12)
    return Analysis(sr=sr, duration=duration, bpm=bpm, notes=notes, times=times, f0=f0,
                    voiced_ratio=voiced_ratio, rms_db=rms_db, centroid=centroid,
                    harm_mag=mag, harm_phase=rel, harm_profile_db=prof,
                    noise_ratio=noise_ratio, noise_tilt=noise_tilt,
                    unison_hint=_unison_hint(Sx, f0, sr), env=env, percussive=percussive,
                    lfo=lfo, grid={"notes": grid, "bpm": bpm},
                    pitch_lfo=pitch_lfo, pitch_env=pitch_env, glide=glide, notes_raw=notes_raw,
                    spectrum_db=ltas - ltas.max(), spectrum_hz=freqs)
