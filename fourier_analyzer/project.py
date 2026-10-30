"""Projects (one per dropped audio file) and background jobs."""
from __future__ import annotations

import json
import logging
import shutil
import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import audio, config

log = logging.getLogger(__name__)


# ----------------------------------------------------------------------------
# jobs


class Cancelled(BaseException):
    """Raised inside a job when the user cancels it (BaseException so fallbacks
    that catch ``Exception`` do not swallow it)."""


@dataclass
class Job:
    id: str
    kind: str
    project: str | None = None
    status: str = "queued"          # queued | running | done | error | cancelled
    progress: float = 0.0
    stage: str = ""
    best: float | None = None
    log: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)   # live status (what is happening right now)
    finished: float | None = None
    cancel_requested: bool = False
    cancels: int = 0                 # a reconstruction's 2nd Stop also ends Serum tuning

    def cancel(self):
        self.cancel_requested = True
        self.cancels += 1
        self.msg("stop requested" if self.kind == "reconstruct" else "cancel requested")

    def check(self):
        """Abort the job here if the user cancelled it."""
        if self.cancel_requested:
            raise Cancelled()

    def set(self, stage: str, text: str, **extra):
        """Update the live status line without adding to the log."""
        self.stage = stage
        self.detail = {"stage_text": text, "stage_t0": time.time(), **extra}

    def msg(self, text: str):
        self.log.append({"t": round(time.time() - self.created, 1), "msg": text})
        self.log = self.log[-300:]
        log.info("[job %s] %s", self.id, text)

    def to_dict(self):
        end = self.finished or time.time()
        d = dict(self.detail)
        if "stage_t0" in d:
            d["stage_elapsed"] = round(end - d.pop("stage_t0"), 1)
        return {"id": self.id, "kind": self.kind, "project": self.project, "status": self.status,
                "progress": round(self.progress, 3), "stage": self.stage, "best": self.best,
                "elapsed": round(end - self.created, 1), "detail": d,
                "log": self.log[-300:], "result": self.result, "error": self.error,
                "cancel_requested": self.cancel_requested, "cancels": self.cancels}


def parallel_jobs() -> int:
    """How many jobs may run at the same time (setting ``parallel_jobs``)."""
    import os
    try:
        n = int(config.settings().get("parallel_jobs", 0))
    except (TypeError, ValueError):
        n = 0
    if n <= 0:
        n = max(2, min(4, (os.cpu_count() or 4) // 3))
    return n


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        # several jobs at once so separate windows can each process a track
        self.pool = ThreadPoolExecutor(max_workers=parallel_jobs(), thread_name_prefix="fa-job")
        self.lock = threading.Lock()

    def submit(self, kind: str, fn, project: str | None = None) -> Job:
        job = Job(id=uuid.uuid4().hex[:10], kind=kind, project=project)
        with self.lock:
            self.jobs[job.id] = job

        def run():
            if job.cancel_requested:
                job.status, job.finished = "cancelled", time.time()
                return
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done"
                job.progress = 1.0
            except Cancelled:
                job.status = "cancelled"
                job.msg("cancelled")
            except Exception as e:
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                job.msg(traceback.format_exc())
            finally:
                job.finished = time.time()
        self.pool.submit(run)
        return job

    def get(self, jid: str) -> Job | None:
        return self.jobs.get(jid)


JOBS = JobManager()


# ----------------------------------------------------------------------------
# projects


def project_dir(pid: str) -> Path:
    d = config.projects_dir() / pid
    if not d.exists():
        raise FileNotFoundError(pid)
    return d


def load_meta(pid: str) -> dict:
    return json.loads((project_dir(pid) / "meta.json").read_text())


def save_meta(pid: str, meta: dict) -> None:
    (config.projects_dir() / pid / "meta.json").write_text(json.dumps(meta, indent=1))


def list_projects() -> list[dict]:
    out = []
    for d in sorted(config.projects_dir().glob("*/meta.json"), key=lambda p: -p.stat().st_mtime):
        try:
            m = json.loads(d.read_text())
            out.append({"id": m["id"], "name": m["name"], "created": m.get("created"),
                        "status": m.get("status")})
        except Exception:
            continue
    return out


def delete_project(pid: str) -> None:
    d = project_dir(pid)
    if any(j.project == pid and j.status in ("queued", "running") for j in JOBS.jobs.values()):
        raise RuntimeError("a job is still running for this project")
    if d.resolve().parent != config.projects_dir().resolve():
        raise RuntimeError("refusing to delete outside the projects folder")
    shutil.rmtree(d)


def create_project(upload_path: Path, original_name: str) -> dict:
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = config.projects_dir() / pid
    (d / "stems").mkdir(parents=True)
    y, sr = audio.load(upload_path, sr=44100, mono=False)
    audio.save(d / "original.wav", y, sr)      # untouched copy; mix.wav may be clipped
    audio.save(d / "mix.wav", y, sr)
    meta = {"id": pid, "name": Path(original_name).stem, "created": time.time(), "status": "new",
            "duration": round(y.shape[-1] / sr, 3), "stems": {"mix": "mix.wav"}, "instruments": [],
            "results": [], "bpm": None}
    save_meta(pid, meta)
    return meta


def clip_mix(pid: str, start: float | None, end: float | None) -> dict:
    """Cut mix.wav down to [start, end] of the original upload (None = full)."""
    d = project_dir(pid)
    meta = load_meta(pid)
    src = d / "original.wav" if (d / "original.wav").exists() else d / "mix.wav"
    y, sr = audio.load(src, sr=44100, mono=False)
    if start is not None or end is not None:
        a = int(max(0.0, start or 0.0) * sr)
        b = int(min(y.shape[-1] / sr, end or y.shape[-1] / sr) * sr)
        if b - a < sr // 4:
            raise ValueError("clip must be at least 0.25 s long")
        y = y[..., a:b]
        meta["clip"] = [round(a / sr, 4), round(b / sr, 4)]
    else:
        meta.pop("clip", None)
    audio.save(d / "mix.wav", y, sr)
    meta["duration"] = round(y.shape[-1] / sr, 3)
    meta["stems"] = {"mix": "mix.wav"}
    meta["instruments"] = []
    save_meta(pid, meta)
    return meta


def combine_stems(pid: str, names: list[str], new_name: str | None = None) -> str:
    """Sum several stems/layers into a new stem."""
    d = project_dir(pid)
    meta = load_meta(pid)
    if len(names) < 2:
        raise ValueError("pick at least two stems to combine")
    parts = [audio.load(d / meta["stems"][n], sr=44100, mono=False)[0] for n in names]
    n = max(p.shape[-1] for p in parts)
    ch = max(p.shape[0] if p.ndim == 2 else 1 for p in parts)
    out = np.zeros((ch, n), dtype=np.float32)
    for p in parts:
        p = np.atleast_2d(p)
        out[:, : p.shape[-1]] += p if p.shape[0] == ch else np.repeat(p[:1], ch, axis=0)
    base = new_name or "+".join(names)
    key = "".join(c if c.isalnum() or c in "_+-" else "_" for c in base)[:60]
    while key in meta["stems"]:
        key += "_2"
    path = audio.save(d / "stems" / "combined" / f"{key}.wav", out, 44100)
    meta["stems"][key] = str(path.relative_to(d))
    from .analysis.instruments import describe_all
    meta["instruments"] += describe_all({key: path}, d / "mix.wav")
    save_meta(pid, meta)
    return key


def start_separation(pid: str, drums: bool = True, clip: tuple | None = None) -> Job:
    def work(job: Job):
        from .analysis.features import tempo_candidates
        from .analysis.instruments import describe_all
        from .separation.stems import separate_all
        d = project_dir(pid)
        if clip is not None:
            job.set("clip", "Clipping the audio to your selection")
            clip_mix(pid, *clip)
            job.msg(f"clipped to {clip[0] or 0:.2f}-{clip[1] or 0:.2f} s of the original")
        meta = load_meta(pid)
        meta["status"] = "separating"
        save_meta(pid, meta)
        job.set("tempo", "Detecting tempo")
        y, sr = audio.load(d / "mix.wav", sr=22050)
        cands = tempo_candidates(y, sr)
        meta["bpm"] = cands[0]["bpm"]
        meta["bpm_candidates"] = [c["bpm"] for c in cands[:6]]
        job.msg(f"tempo estimate {meta['bpm']} BPM (alternatives: "
                f"{', '.join(str(c) for c in meta['bpm_candidates'][1:4])})")
        job.set("separating", "Separating stems with AI models (Demucs, then DrumSep on the drums). "
                              "This is the slow part: about 1-3 min per minute of audio")
        job.progress = 0.05

        def prog(text):
            job.check()
            job.msg(text)
            job.progress = min(0.85, job.progress + 0.1)
        for old in (d / "stems").glob("*"):          # re-separation replaces old stems
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
            else:
                old.unlink(missing_ok=True)
        try:
            stems = separate_all(d / "mix.wav", d / "stems", progress=prog, drums=drums)
            job.check()
        except Cancelled:
            meta["status"] = "new"            # old stems were cleared above
            meta["stems"], meta["instruments"] = {"mix": "mix.wav"}, []
            save_meta(pid, meta)
            raise
        meta["separation"] = getattr(separate_all, "last_method", "ml")
        meta["stems"] = {"mix": "mix.wav", **{k: str(Path(v).relative_to(d)) for k, v in stems.items()}}
        job.set("instruments", "Labelling instruments in each stem")
        job.progress = 0.9
        meta["instruments"] = describe_all({k: d / v for k, v in meta["stems"].items() if k != "mix"},
                                           d / "mix.wav")
        meta["status"] = "ready"
        save_meta(pid, meta)
        found = [i["label"] for i in meta["instruments"] if i.get("present")]
        job.msg("detected: " + ", ".join(found))
        return {"stems": list(meta["stems"]), "instruments": meta["instruments"]}
    return JOBS.submit("separate", work, project=pid)


def start_layer_split(pid: str, stem: str, n_layers: int, start=None, end=None) -> Job:
    def work(job: Job):
        from .analysis.instruments import describe_all
        from .separation.stems import split_layers
        d = project_dir(pid)
        meta = load_meta(pid)
        src = d / meta["stems"][stem]
        job.stage = "NMF layer split"
        job.msg(f"splitting {stem} into {n_layers} layers")
        out = split_layers(src, d / "stems" / "layers", n_layers, start, end)
        job.check()
        for k, v in out.items():
            meta["stems"][k] = str(Path(v).relative_to(d))
        meta["instruments"] = [i for i in meta["instruments"] if i["stem"] not in out]
        meta["instruments"] += describe_all(out, d / "mix.wav")
        save_meta(pid, meta)
        return {"layers": list(out)}
    return JOBS.submit("split", work, project=pid)


def redetect_tempo(pid: str) -> dict:
    from .analysis.features import tempo_candidates
    d = project_dir(pid)
    meta = load_meta(pid)
    y, sr = audio.load(d / "mix.wav", sr=22050)
    cands = tempo_candidates(y, sr)
    meta["bpm"] = cands[0]["bpm"]
    meta["bpm_candidates"] = [c["bpm"] for c in cands[:6]]
    save_meta(pid, meta)
    return {"bpm": meta["bpm"], "candidates": meta["bpm_candidates"]}


def set_bpm(pid: str, bpm: float) -> None:
    meta = load_meta(pid)
    meta["bpm"] = float(bpm)
    save_meta(pid, meta)


def load_segment(pid: str, stem: str, start: float | None, end: float | None, sr: int = 44100):
    d = project_dir(pid)
    meta = load_meta(pid)
    y, sr = audio.load(d / meta["stems"][stem], sr=sr, mono=True)
    a = int(max(0.0, start or 0.0) * sr)
    b = int((end if end else len(y) / sr) * sr)
    return y[a:b], sr


def detect_sidechain(pid: str, y: np.ndarray, sr: int, notes, start, end, bpm) -> dict:
    """Kick times from the kick stem (or the beat grid) + ducking model."""
    from .analysis import sidechain as SC
    meta = load_meta(pid)
    if "drum_kick" in meta["stems"]:
        yk, _ = load_segment(pid, "drum_kick", start, end, sr=sr)
        kicks, src = SC.kick_times_from_audio(yk, sr), "kick stem"
    else:
        ym, _ = load_segment(pid, "mix", start, end, sr=sr)
        kicks, src = SC.kick_times_from_grid(ym, sr, bpm or 140.0), "beat grid"
    return SC.estimate(y, sr, notes, kicks, source=src)


def quick_analysis(pid: str, stem: str, start, end, bpm=None) -> dict:
    """Fourier analysis for the UI (no optimisation)."""
    from .analysis.features import analyze
    meta = load_meta(pid)
    y, sr = load_segment(pid, stem, start, end, sr=22050)
    if len(y) < 2048:
        raise ValueError("selection too short")
    bpm = bpm or meta.get("bpm")
    a = analyze(y, sr, bpm=bpm)
    sc = detect_sidechain(pid, y, sr, a.notes, start, end, bpm)
    if sc["detected"]:
        from .analysis.sidechain import gain_curve, logic_settings
        a = analyze(y / np.maximum(gain_curve(sc, len(y), sr), 0.1), sr, bpm=bpm)
        sc["logic"] = logic_settings(sc)
    s = a.summary()
    s["sidechain"] = sc
    hz = a.spectrum_hz
    keep = (hz >= 20) & (hz <= 11000)
    idx = np.unique(np.geomspace(1, keep.sum() - 1, 400).astype(int))
    s["spectrum"] = {"hz": hz[keep][idx].round(1).tolist(), "db": a.spectrum_db[keep][idx].round(1).tolist()}
    s["harmonics_db"] = a.harm_profile_db[:64].round(1).tolist()
    s["f0"] = {"t": a.times[::2].round(3).tolist(),
               "hz": [None if not np.isfinite(v) else round(float(v), 2) for v in a.f0[::2]]}
    return s


def tune_serum(job: Job, res, opt, bank, y44: np.ndarray, sc: dict, bpm: float, budget_s: float,
               corrected: bool = False, span: tuple = (0.9, 0.91), learnable: bool = True) -> None:
    """Fine-tune the patch inside real Serum 2 (best effort). ``corrected``:
    the patch values are already what Serum gets (it came from a preset);
    ``span``: the job progress range this stage covers; ``learnable``: its
    parameter moves measure our engine's error (not a fit from scratch)."""
    from .analysis import sidechain as SC
    from .match.serum_tune import tune_in_serum
    from .serum.host import SerumHostError
    stopped_search = job.cancel_requested
    stop = (lambda: job.cancels >= 2) if stopped_search else (lambda: job.cancel_requested)
    if stop():
        return
    n = int(round(opt.duration * 44100))
    y = y44[:n]
    gain = SC.gain_curve(sc, len(y), 44100) if sc["detected"] and not ducks_itself(res.patch) else None
    notes = [nt for nt in (res.notes or opt.notes) if nt.start < opt.duration]
    job.set("serum tune", "Fine-tuning the patch inside the real Serum 2 plugin", budget=budget_s, elapsed=0,
            tune_after_stop=stopped_search)
    job.msg(f"fine-tuning in real Serum 2 for up to {budget_s / 60:.0f} min"
            + (" (press Stop again to end it)" if stopped_search else ""))

    def prog(text, best, elapsed, extra):
        job.best = round(best, 2)
        job.progress = min(span[1], span[0] + (span[1] - span[0]) * elapsed / max(budget_s, 1))
        job.detail.update(extra)
        job.detail["activity"] = f"{extra['evals']} renders in Serum, best {best:.1f}%"
        if text:
            job.msg(text)
    try:
        patch, info = tune_in_serum(res.patch, bank, notes, y, 44100, budget_s, gain=gain, samples=res.samples,
                                    stop=stop, progress=prog, use_corrections=not corrected)
    except SerumHostError as e:
        job.msg(f"Serum tuning skipped: {e}")
        res.serum_tune = {"ok": False, "error": str(e)}
        return
    info.pop("details", None)
    info["learnable"] = learnable
    res.serum_tune = info
    job.msg(f"Serum tuning: {info['before']}% -> {info['after']}% in real Serum "
            f"({info['evals']} renders, {info['seconds']:.0f} s)")
    if patch is not res.patch:
        res.patch = patch
        res.details = opt.ref.all(opt.render(patch))
        res.score = res.details["waveform_similarity"]
        res.stages.append({"stage": "Serum 2 tuning", "score": info["after"],
                           "note": f"real Serum score {info['before']} -> {info['after']}"})


def make_judge(job: Job, opt, bank, y44, sc: dict, opts: dict, budget_s: float):
    """Real Serum 2 as the judge of structure decisions (None when it is not
    installed: the search then runs on our engine alone)."""
    from .analysis import sidechain as SC
    from .match.serum_judge import SerumJudge
    from .serum import host as serum_host
    try:
        serum_host._require_serum()
    except serum_host.SerumHostError as e:
        job.msg(f"structure search in our engine only: {e}")
        return None
    n = int(round(opt.duration * 44100))
    y = y44[:n]
    gain = SC.gain_curve(sc, len(y), 44100) if sc["detected"] else None
    notes = [nt for nt in opt.notes if nt.start < opt.duration]
    finalists = int(opts.get("serum_finalists") or 3)
    job.msg(f"structure decisions confirmed in real Serum 2: top {finalists} candidates per round, "
            f"up to {budget_s / 60:.0f} min of Serum time")
    return SerumJudge(bank, opt.samples, notes, y, 44100, budget_s, gain=gain,
                      stop=lambda: job.cancel_requested, finalists=finalists)


def ducks_itself(patch) -> bool:
    """The patch contains an active LFO -> Voice Amp route (the sidechain
    ducking written into the preset), so its renders are already ducked."""
    return any(m.dest == "voiceamp" and m.source.startswith("lfo") and m.amount and not m.bypass
               for m in patch.mods)


def learn_from_serum(job: Job, res, opt, bank, notes, r44, serum_raw, y44, gain44, serum: dict, out: Path,
                     meta: dict, budget_s: float, corrected: bool = False) -> dict | None:
    """Compare our render with Serum's render of the same preset, find which
    component causes the difference, log it and learn corrections."""
    from . import calibration
    from .match.metrics import Reference
    from .serum.host import SerumHostError, SerumSession
    from .serum.mapping import build_preset_bytes
    from .synth.engine import Renderer
    job.set("learning", "Learning from Serum: comparing our engine with real Serum 2")
    job.msg("comparing our engine's render with Serum's render of the same preset")
    ours = r44 * gain44[: len(r44)] if gain44 is not None else r44
    our_score = Reference(np.asarray(y44, dtype=np.float64), 44100).all(ours)["waveform_similarity"]
    culprits = None
    if budget_s > 0 and job.cancels < 2:
        dur = opt.duration
        nts = [nt for nt in notes if nt.start < dur]
        corr = {"params": {}} if corrected else calibration.load_corrections()
        rnd = Renderer(bank, 44100, samples=res.samples)
        job.set("learning", "Learning from Serum: switching each component off in both engines",
                budget=budget_s, elapsed=0)
        try:
            with SerumSession(44100) as sess:
                culprits = calibration.find_culprits(
                    res.patch, bank, res.samples, nts, dur, 44100, sess,
                    render_ours=lambda q: rnd.render(q, nts, dur),
                    build=lambda q: build_preset_bytes(calibration.apply_corrections(q, corr), bank,
                                                       name="FA calibration", samples=res.samples),
                    time_budget=budget_s, stop=lambda: job.cancels >= 2)
        except SerumHostError as e:
            job.msg(f"culprit search skipped: {e}")
    try:
        rec = calibration.record_run(patch=res.patch, ours=r44, serum=serum_raw, sr=44100, notes=notes,
                                     our_score=our_score, serum_score=float(serum["score"]), culprits=culprits,
                                     tuning=getattr(res, "serum_tune", None), meta=meta)
    except Exception as e:                     # learning must never cost the user the result
        log.warning("calibration record failed: %s", e, exc_info=True)
        job.msg(f"learning step failed: {e}")
        return None
    (out / "engine_gap.txt").write_text(calibration.report_text(rec))
    job.msg(f"our engine {rec['our_score']}% vs real Serum {rec['serum_score']}%")
    for f in rec["findings"][:6]:
        job.msg("  " + f)
    return {k: rec[k] for k in ("our_score", "serum_score", "score_drop", "findings", "learned", "culprits")} | {
        "engine_similarity": rec["engine_gap"].get("similarity")}


def safe_name(name: str) -> str:
    """A file/folder name valid on Windows and macOS."""
    import re
    out = re.sub(r'[<>:"/\\|?*\x00-\x1f]', "_", name).strip().rstrip(". ")
    return out or "untitled"


def load_result_patch(pid: str, rid: str, bank, bpm: float):
    """The patch of an earlier result, read back from its .SerumPreset (so it
    is exactly what Serum played). Returns (patch, sample bank)."""
    from .serum.mapping import state_to_patch
    from .serum.preset import read_preset_bytes
    from .synth.samples import SampleBank
    out = project_dir(pid) / "results" / rid
    src = next(out.glob("*.SerumPreset"), None)
    if src is None:
        raise ValueError(f"result {rid} has no preset")
    _, st = read_preset_bytes(src.read_bytes())
    samples = SampleBank()
    return state_to_patch(st, bank, bpm=bpm, samples=samples), samples


def load_result_report(pid: str, rid: str) -> dict:
    p = project_dir(pid) / "results" / rid / "report.json"
    if not p.exists():
        raise ValueError(f"no result {rid}")
    import json
    return json.loads(p.read_text())


def start_reconstruct(pid: str, stem: str, start, end, opts: dict, mode: str = "full",
                      from_rid: str | None = None) -> Job:
    """mode "full": search in our engine (then optionally tune in Serum);
    "serum_only": skip our engine's search, tune only inside real Serum 2;
    "retune": load result ``from_rid``'s preset and tune it further in Serum."""
    def work(job: Job):
        from .analysis.features import analyze
        from .export import write_midi, write_report
        from .match.ablation import run_checks
        from .match.metrics import METRIC_INFO
        from .match.optimizer import OPT_SR, Options, Optimizer, Result
        from .synth.engine import Renderer as _R
        from .serum import host as serum_host
        from .serum.mapping import build_preset_bytes
        from .synth.engine import Renderer
        from .analysis import sidechain as SC
        from .synth.wavetables import TableBank

        meta = load_meta(pid)
        bpm = opts.get("bpm") or meta.get("bpm") or 140.0
        y44, _ = load_segment(pid, stem, start, end, sr=44100)
        if len(y44) < 4096:
            raise ValueError("selection too short")
        y22 = audio.resample(y44, 44100, OPT_SR)
        job.set("analysis", "Fourier analysis of the selection (pitch, notes, harmonics, LFO)")
        job.msg("Fourier analysis of the selection")
        a = analyze(y22, OPT_SR, bpm=bpm)
        # --- sidechain: kick times from the kick stem (or the beat grid) ---
        job.set("sidechain", "Looking for sidechain ducking from the kick")
        sc = detect_sidechain(pid, y22, OPT_SR, a.notes, start, end, bpm)
        src = sc["source"]
        gain22 = None
        if sc["detected"]:
            gain22 = SC.gain_curve(sc, len(y22), OPT_SR)
            job.msg(f"sidechain detected ({src}): -{sc['depth_db']} dB, attack {sc['attack_ms']} ms, "
                    f"hold {sc['hold_ms']} ms, release {sc['release_ms']} ms over {sc['kicks_used']} kicks; "
                    f"removing it before fitting the patch")
            a = analyze(y22 / np.maximum(gain22, 0.1), OPT_SR, bpm=bpm)   # analyse the un-ducked synth
        else:
            job.msg(f"no sidechain ducking found ({sc.get('reason', '')})")
        job.msg(f"{len(a.notes)} notes, voiced {a.voiced_ratio:.0%}, noise {a.noise_ratio:.0%}, "
                f"LFO: {a.lfo['sync'] or a.lfo['rate_hz'] if a.lfo else 'none'}")
        bank = TableBank()

        def prog(stage, frac, best, text, status=None):
            job.stage, job.progress = stage, 0.05 + 0.85 * frac
            job.best = round(best, 2) if best is not None else None
            if status:
                job.detail = dict(status)
            if text:
                job.msg(text)
        budget = float(opts.get("time_budget") or 0)
        if budget <= 0:                       # "until I stop it"
            budget = 30 * 24 * 3600.0
        t_job0 = time.time()
        struct_s = float(opts.get("serum_structure_minutes") or 5) * 60
        o = Options(target_score=float(opts.get("target", 90)),
                    time_budget=min(budget, struct_s) if mode == "serum_only" else budget,
                    keep_going=bool(opts.get("keep_going", False)),
                    prefer_synthesis=float(opts.get("prefer_synthesis", 0.5)),
                    checkpoint_minutes=float(opts.get("checkpoint_minutes") or 3.0),
                    holdout=bool(opts.get("holdout", True)),
                    stop=lambda: job.cancel_requested)
        if budget > 900:
            job.msg(f"long run: {'until stopped' if budget > 7 * 24 * 3600 else f'{budget / 60:.0f} min'}; "
                    "press Stop to finish with the best patch so far")
        opt = Optimizer(y22, OPT_SR, a, o, bank=bank, progress=prog, context_gain=gain22,
                        source_audio=(y44, 44100))
        corrected = False
        judge = None
        if mode in ("full", "serum_only") and (opts.get("serum_structure", True) or opts.get("serum_checkpoints", True)
                                                or mode == "serum_only"):
            # long runs get more real-Serum time (checkpoints run all along)
            serum_s = struct_s if budget <= 900 else max(struct_s, min(0.2 * budget, 4 * 3600.0))
            judge = make_judge(job, opt, bank, y44, sc, opts, serum_s)
            if judge is not None and not opts.get("serum_structure", True):
                judge.finalists = 0
            if judge is not None and not opts.get("serum_checkpoints", True):
                opt.CHECKPOINT_LAG = float("inf")
            opt.judge = judge
        if mode == "full":
            try:
                res = opt.run()
            finally:
                if judge is not None:
                    judge.close()
            if opts.get("serum_tune", False) and float(opts.get("serum_tune_minutes") or 0) > 0:
                tune_serum(job, res, opt, bank, y44, sc, bpm, float(opts["serum_tune_minutes"]) * 60)
        else:
            serum_host._require_serum()          # these modes need the real plugin
            if mode == "serum_only":
                job.set("start", "Building the starting patch (analysis guess + custom wavetable)")
                job.msg("Serum only: parts are screened in our engine but only added or removed when real "
                        "Serum 2 agrees; the parameters are then tuned in Serum")
                try:
                    res = opt.run_serum_only()
                finally:
                    if judge is not None:
                        judge.close()
                left = budget - (time.time() - t_job0)
                tune_serum(job, res, opt, bank, y44, sc, bpm, max(30.0, left), span=(0.5, 0.9), learnable=False)
            else:
                job.set("start", f"Loading result {from_rid}")
                patch, samples = load_result_patch(pid, from_rid, bank, bpm)
                opt.samples = samples
                opt.renderer = _R(bank, OPT_SR, samples=samples)
                metrics = opt.ref.all(opt.render(patch))
                res = Result(patch=patch, score=metrics["waveform_similarity"], details=metrics, notes=a.notes,
                             bank=bank, stages=[{"stage": f"start: result {from_rid}",
                                                 "score": round(metrics["waveform_similarity"], 2)}],
                             samples=samples)
                corrected = True
                minutes = float(opts.get("serum_tune_minutes") or 5)
                job.msg(f"tuning result {from_rid} in real Serum 2 for {minutes:g} min")
                tune_serum(job, res, opt, bank, y44, sc, bpm, minutes * 60, corrected=True, span=(0.05, 0.9))
            st = getattr(res, "serum_tune", None) or {}
            if not st.get("ok"):
                raise ValueError(f"tuning in Serum 2 failed: {st.get('error', 'no result')}")
        serum_structure = None
        if judge is not None:
            serum_structure = {**judge.summary(), "checks": judge.checks}
            if judge.error:
                job.msg(f"structure checks in real Serum 2 unavailable ({judge.error}); our engine decided alone")
            elif judge.checks:
                job.msg(f"real Serum 2 checked {len(judge.checks)} structure decisions in {judge.used_s:.0f} s "
                        f"({judge.renders} renders) and agreed with our engine on "
                        f"{sum(c['agreed'] for c in judge.checks)}")
                from . import calibration
                calibration.record_checks(judge.checks, {"project": pid, "stem": stem})
        ho = getattr(res, "holdout", None)
        if ho:
            job.msg(f"held-out notes ({ho['start']}-{ho['end']} s, not used for fitting): waveform "
                    f"{ho['waveform_similarity']}%, spectral {ho['spectral_similarity']}% "
                    f"(fitted part: {ho['fit_waveform_similarity']}% / {ho['fit_spectral_similarity']}%)")
        pc = getattr(res, "pitch_check", None)
        if pc:
            job.msg(f"transposed-pitch check of the whole patch: {'ok' if pc['ok'] else 'PROBLEM'}: {pc['why']}")
        d = res.details
        job.msg(f"waveform similarity {d['waveform_similarity']}%, spectral similarity "
                f"{d['spectral_similarity']}%, SI-SDR {d['si_sdr_db']} dB; running checks")
        job.set("verification", "Verification: removing each component, noise attribution, timing and pitch checks")
        job.progress = 0.92
        checks = run_checks(res.patch, opt.y, OPT_SR, opt.notes, a.f0[: int(len(opt.y) / 256) + 1],
                            opt.renderer, opt.target, gain=opt.gain)

        if sc["detected"] and not corrected:
            note = SC.add_to_patch(res.patch, sc, bpm, enabled=bool(opts.get("sidechain_in_preset", False)))
            if note:
                job.msg(note)
        if not corrected:
            from .match.macros import assign_macros
            labels = assign_macros(res.patch)
            if labels:
                job.msg("macros: " + ", ".join(labels))
            if getattr(res, "synth_patch", None) is not None:
                if sc["detected"]:
                    SC.add_to_patch(res.synth_patch, sc, bpm, enabled=bool(opts.get("sidechain_in_preset", False)))
                assign_macros(res.synth_patch)
        rid = time.strftime("%H%M%S-") + uuid.uuid4().hex[:4]
        out = project_dir(pid) / "results" / rid
        out.mkdir(parents=True)
        name = safe_name(f"FA {meta['name'][:24]} {stem} {rid[:6]}")
        res.patch.name = name
        from . import calibration
        # learned corrections: the preset is written so real Serum sounds like our render
        preset = build_preset_bytes(res.patch if corrected else calibration.apply_corrections(res.patch), bank,
                                    name=name, samples=res.samples,
                                    description=f"Reconstructed by Fourier Analyzer, accuracy {res.score:.1f}")
        (out / f"{name}.SerumPreset").write_bytes(preset)
        synth = None
        if getattr(res, "synth_patch", None) is not None:
            sname = name + " synth"
            res.synth_patch.name = sname
            (out / f"{sname}.SerumPreset").write_bytes(build_preset_bytes(
                res.synth_patch if corrected else calibration.apply_corrections(res.synth_patch), bank, name=sname,
                samples=res.samples, description=f"Synthesis only (no samples), accuracy "
                                                 f"{res.synth_details['waveform_similarity']:.1f}"))
            synth = {"name": sname, "file": f"{sname}.SerumPreset", "details": res.synth_details,
                     "accuracy": res.synth_details["waveform_similarity"],
                     "description": res.synth_patch.describe()}
            job.msg(f"two presets: best overall {res.score:.1f}% (uses a sample-based oscillator), "
                    f"synthesis only {synth['accuracy']:.1f}%")
        write_midi(out / "notes.mid", a.notes, bpm)
        r44 = Renderer(bank, 44100, samples=res.samples).render(res.patch, a.notes, len(y44) / 44100)
        audio.save(out / "render.wav", audio.normalize(r44), 44100)
        gain44 = SC.gain_curve(sc, len(r44), 44100) if sc["detected"] else None
        if gain44 is not None:
            audio.save(out / "render_with_sidechain.wav", audio.normalize(r44 * gain44), 44100)
        audio.save(out / "target.wav", audio.normalize(y44), 44100)
        for tid, frames in bank.custom.items():
            from .synth.wavetables import write_serum_wav
            write_serum_wav(out / f"{name} {tid.replace(':', ' ')}.wav", frames)

        serum = {"ok": False, "error": "skipped"}
        learning = bool(opts.get("learn", True))
        if opts.get("verify_serum", True) or learning:
            job.set("serum", "Rendering the preset in the real Serum 2 plugin")
            job.progress = 0.96
            job.msg("rendering in real Serum 2 for verification")
            serum = serum_host.verify(preset, a.notes, y44, 44100, gain=gain44)
            if serum.get("ok"):
                audio.save(out / "serum_render.wav", audio.normalize(serum.pop("audio")), 44100)
                job.msg(f"real Serum score: {serum['score']}")
            else:
                job.msg(f"Serum verification unavailable: {serum.get('error')}")
        raw_serum = serum.pop("raw_audio", None)
        engine_gap = None
        if learning and serum.get("ok") and raw_serum is not None:
            engine_gap = learn_from_serum(job, res, opt, bank, a.notes, r44, raw_serum, y44, gain44, serum,
                                          out, {"name": name, "stem": stem, "project": pid},
                                          float(opts.get("learn_minutes") or 3) * 60, corrected=corrected)
        summary = {
            "id": rid, "mode": mode, "from_result": from_rid, "stem": stem, "start": start, "end": end, "bpm": bpm, "name": name,
            "accuracy": round(res.score, 2), "metric": "waveform_similarity", "details": res.details,
            "metric_info": METRIC_INFO, "target": o.target_score,
            "met_target": res.score >= o.target_score, "used_custom_wavetable": True,
            "sidechain": {k: v for k, v in sc.items() if k != "kicks"},
            "sidechain_logic": SC.logic_settings(sc),
            "patch": res.patch.to_dict(), "description": res.patch.describe(), "stages": res.stages,
            "checks": checks, "serum": serum, "serum_tune": getattr(res, "serum_tune", None),
            "serum_structure": serum_structure,
            "synth_only": synth, "holdout": getattr(res, "holdout", None),
            "pitch_check": getattr(res, "pitch_check", None), "checkpoints": getattr(res, "checkpoints", []),
            "prefer_synthesis": float(opts.get("prefer_synthesis", 0.5)),
            "engine_gap": engine_gap, "analysis": a.summary(), "evals": opt.evals,
            "files": sorted(p.name for p in out.iterdir()),
        }
        exp = config.exports_dir() / safe_name(meta["name"][:60]) / safe_name(f"{stem} {rid}")
        try:
            shutil.copytree(out, exp, dirs_exist_ok=True)
            summary["export_dir"] = str(exp)
        except OSError as e:
            job.msg(f"could not copy to exports folder: {e}")
        write_report(out / "report.json", summary)
        if summary.get("export_dir"):
            shutil.copy2(out / "report.json", exp / "report.json")
        meta = load_meta(pid)
        meta["results"].append({"id": rid, "stem": stem, "accuracy": summary["accuracy"], "name": name})
        save_meta(pid, meta)
        return summary
    return JOBS.submit("reconstruct", work, project=pid)


def save_stems(pid: str, names: list[str] | None = None) -> Path:
    """Copy stems (all, or ``names``) as WAV files into the export folder:
    <exports>/<project>/stems/<project> - <instrument>.wav. Returns the folder."""
    meta = load_meta(pid)
    d = project_dir(pid)
    labels = {i["stem"]: i.get("label") for i in meta.get("instruments", [])}
    out = config.exports_dir() / safe_name(meta["name"][:60]) / "stems"
    out.mkdir(parents=True, exist_ok=True)
    for name, rel in meta["stems"].items():
        if names and name not in names:
            continue
        src = d / rel
        if not src.exists():
            continue
        label = "full mix" if name == "mix" else (labels.get(name) or name)
        tag = name if label == name else f"{label} ({name})"
        shutil.copy2(src, out / safe_name(f"{meta['name'][:40]} - {tag}.wav"))
    return out


def install_preset(pid: str, rid: str) -> Path:
    out = project_dir(pid) / "results" / rid
    src = next(out.glob("*.SerumPreset"))
    dst_dir = config.serum_user_presets_dir()
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    for wav in out.glob("* custom *.wav"):
        tables = config.serum_tables_dir()
        if tables:
            tdir = tables / "User" / "Fourier Analyzer"
            tdir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(wav, tdir / wav.name)
    return dst
