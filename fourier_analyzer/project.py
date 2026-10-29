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


@dataclass
class Job:
    id: str
    kind: str
    project: str | None = None
    status: str = "queued"          # queued | running | done | error
    progress: float = 0.0
    stage: str = ""
    best: float | None = None
    log: list = field(default_factory=list)
    result: dict | None = None
    error: str | None = None
    created: float = field(default_factory=time.time)
    detail: dict = field(default_factory=dict)   # live status (what is happening right now)
    finished: float | None = None

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
                "log": self.log[-300:], "result": self.result, "error": self.error}


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
            job.status = "running"
            try:
                job.result = fn(job)
                job.status = "done"
                job.progress = 1.0
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
            job.msg(text)
            job.progress = min(0.85, job.progress + 0.1)
        for old in (d / "stems").glob("*"):          # re-separation replaces old stems
            if old.is_dir():
                shutil.rmtree(old, ignore_errors=True)
            else:
                old.unlink(missing_ok=True)
        stems = separate_all(d / "mix.wav", d / "stems", progress=prog, drums=drums)
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


def start_reconstruct(pid: str, stem: str, start, end, opts: dict) -> Job:
    def work(job: Job):
        from .analysis.features import analyze
        from .export import write_midi, write_report
        from .match.ablation import run_checks
        from .match.metrics import METRIC_INFO
        from .match.optimizer import OPT_SR, Options, Optimizer
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
        o = Options(target_score=float(opts.get("target", 90)),
                    time_budget=float(opts.get("time_budget", 240)))
        opt = Optimizer(y22, OPT_SR, a, o, bank=bank, progress=prog, context_gain=gain22)
        res = opt.run()
        d = res.details
        job.msg(f"waveform similarity {d['waveform_similarity']}%, spectral similarity "
                f"{d['spectral_similarity']}%, SI-SDR {d['si_sdr_db']} dB; running checks")
        job.set("verification", "Verification: removing each component, noise attribution, timing and pitch checks")
        job.progress = 0.92
        checks = run_checks(res.patch, opt.y, OPT_SR, opt.notes, a.f0[: int(len(opt.y) / 256) + 1],
                            opt.renderer, opt.target, gain=opt.gain)

        rid = time.strftime("%H%M%S-") + uuid.uuid4().hex[:4]
        out = project_dir(pid) / "results" / rid
        out.mkdir(parents=True)
        name = f"FA {meta['name'][:24]} {stem} {rid[:6]}"
        res.patch.name = name
        preset = build_preset_bytes(res.patch, bank, name=name,
                                    description=f"Reconstructed by Fourier Analyzer, accuracy {res.score:.1f}")
        (out / f"{name}.SerumPreset").write_bytes(preset)
        write_midi(out / "notes.mid", a.notes, bpm)
        r44 = Renderer(bank, 44100).render(res.patch, a.notes, len(y44) / 44100)
        audio.save(out / "render.wav", audio.normalize(r44), 44100)
        gain44 = SC.gain_curve(sc, len(r44), 44100) if sc["detected"] else None
        if gain44 is not None:
            audio.save(out / "render_with_sidechain.wav", audio.normalize(r44 * gain44), 44100)
        audio.save(out / "target.wav", audio.normalize(y44), 44100)
        for tid, frames in bank.custom.items():
            from .synth.wavetables import write_serum_wav
            write_serum_wav(out / f"{name} {tid.replace(':', ' ')}.wav", frames)

        serum = {"ok": False, "error": "skipped"}
        if opts.get("verify_serum", True):
            job.set("serum", "Rendering the preset in the real Serum 2 plugin")
            job.progress = 0.96
            job.msg("rendering in real Serum 2 for verification")
            serum = serum_host.verify(preset, a.notes, y44, 44100, gain=gain44)
            if serum.get("ok"):
                audio.save(out / "serum_render.wav", audio.normalize(serum.pop("audio")), 44100)
                job.msg(f"real Serum score: {serum['score']}")
            else:
                job.msg(f"Serum verification unavailable: {serum.get('error')}")
        summary = {
            "id": rid, "stem": stem, "start": start, "end": end, "bpm": bpm, "name": name,
            "accuracy": round(res.score, 2), "metric": "waveform_similarity", "details": res.details,
            "metric_info": METRIC_INFO, "target": o.target_score,
            "met_target": res.score >= o.target_score, "used_custom_wavetable": True,
            "sidechain": {k: v for k, v in sc.items() if k != "kicks"},
            "sidechain_logic": SC.logic_settings(sc),
            "patch": res.patch.to_dict(), "description": res.patch.describe(), "stages": res.stages,
            "checks": checks, "serum": serum, "analysis": a.summary(), "evals": opt.evals,
            "files": sorted(p.name for p in out.iterdir()),
        }
        exp = config.exports_dir() / meta["name"][:60] / f"{stem} {rid}"
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
