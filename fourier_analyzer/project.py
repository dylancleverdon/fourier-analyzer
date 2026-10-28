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

    def msg(self, text: str):
        self.log.append({"t": round(time.time() - self.created, 1), "msg": text})
        self.log = self.log[-300:]
        log.info("[job %s] %s", self.id, text)

    def to_dict(self):
        return {"id": self.id, "kind": self.kind, "project": self.project, "status": self.status,
                "progress": round(self.progress, 3), "stage": self.stage, "best": self.best,
                "log": self.log[-80:], "result": self.result, "error": self.error}


class JobManager:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fa-job")
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


def create_project(upload_path: Path, original_name: str) -> dict:
    pid = time.strftime("%Y%m%d-%H%M%S-") + uuid.uuid4().hex[:4]
    d = config.projects_dir() / pid
    (d / "stems").mkdir(parents=True)
    y, sr = audio.load(upload_path, sr=44100, mono=False)
    audio.save(d / "mix.wav", y, sr)
    meta = {"id": pid, "name": Path(original_name).stem, "created": time.time(), "status": "new",
            "duration": round(y.shape[-1] / sr, 3), "stems": {"mix": "mix.wav"}, "instruments": [],
            "results": [], "bpm": None}
    save_meta(pid, meta)
    return meta


def start_separation(pid: str, drums: bool = True) -> Job:
    def work(job: Job):
        from .analysis.features import estimate_bpm
        from .analysis.instruments import describe_all
        from .separation.stems import separate_all
        d = project_dir(pid)
        meta = load_meta(pid)
        meta["status"] = "separating"
        save_meta(pid, meta)
        job.stage = "tempo"
        y, sr = audio.load(d / "mix.wav", sr=22050)
        meta["bpm"] = round(estimate_bpm(y, sr), 2)
        job.msg(f"tempo estimate {meta['bpm']} BPM")
        job.stage = "separating"
        job.progress = 0.05

        def prog(text):
            job.msg(text)
            job.progress = min(0.85, job.progress + 0.1)
        stems = separate_all(d / "mix.wav", d / "stems", progress=prog, drums=drums)
        meta["stems"] = {"mix": "mix.wav", **{k: str(Path(v).relative_to(d)) for k, v in stems.items()}}
        job.stage = "detecting instruments"
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


def load_segment(pid: str, stem: str, start: float | None, end: float | None, sr: int = 44100):
    d = project_dir(pid)
    meta = load_meta(pid)
    y, sr = audio.load(d / meta["stems"][stem], sr=sr, mono=True)
    a = int(max(0.0, start or 0.0) * sr)
    b = int((end if end else len(y) / sr) * sr)
    return y[a:b], sr


def quick_analysis(pid: str, stem: str, start, end, bpm=None) -> dict:
    """Fourier analysis for the UI (no optimisation)."""
    from .analysis.features import analyze
    meta = load_meta(pid)
    y, sr = load_segment(pid, stem, start, end, sr=22050)
    if len(y) < 2048:
        raise ValueError("selection too short")
    a = analyze(y, sr, bpm=bpm or meta.get("bpm"))
    s = a.summary()
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
        from .match.optimizer import OPT_SR, Options, Optimizer
        from .serum import host as serum_host
        from .serum.mapping import build_preset_bytes
        from .synth.engine import Renderer
        from .synth.wavetables import StockIndex, TableBank

        meta = load_meta(pid)
        bpm = opts.get("bpm") or meta.get("bpm") or 140.0
        y44, _ = load_segment(pid, stem, start, end, sr=44100)
        if len(y44) < 4096:
            raise ValueError("selection too short")
        y22 = audio.resample(y44, 44100, OPT_SR)
        job.stage = "analysis"
        job.msg("Fourier analysis of the selection")
        a = analyze(y22, OPT_SR, bpm=bpm)
        job.msg(f"{len(a.notes)} notes, voiced {a.voiced_ratio:.0%}, noise {a.noise_ratio:.0%}, "
                f"LFO: {a.lfo['sync'] or a.lfo['rate_hz'] if a.lfo else 'none'}")
        bank = TableBank()
        stock = None
        if config.serum_tables_dir():
            job.stage = "indexing Serum tables"
            job.msg(f"indexing stock wavetables in {config.serum_tables_dir()}")
            stock = StockIndex().build()
            job.msg(f"{len(stock.ids)} stock tables indexed")
        else:
            job.msg("Serum tables folder not found: using built-in basic shapes only")

        def prog(stage, frac, best, text):
            job.stage, job.progress = stage, 0.05 + 0.85 * frac
            job.best = round(best, 2) if best is not None else None
            if text:
                job.msg(text)
        o = Options(target_score=float(opts.get("target", 90)),
                    allow_custom=bool(opts.get("allow_custom", True)),
                    stock_only=bool(opts.get("stock_only", False)),
                    time_budget=float(opts.get("time_budget", 240)))
        opt = Optimizer(y22, OPT_SR, a, o, bank=bank, stock=stock, progress=prog)
        res = opt.run()
        job.stage = "verification"
        job.msg(f"accuracy {res.score:.1f} (raw {res.details['raw_score']}); running checks")
        checks = run_checks(res.patch, opt.y, OPT_SR, opt.notes, a.f0[: int(len(opt.y) / 256) + 1],
                            opt.renderer, opt.target)

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
        audio.save(out / "target.wav", audio.normalize(y44), 44100)
        for tid, frames in bank.custom.items():
            from .synth.wavetables import write_serum_wav
            write_serum_wav(out / f"{name} {tid.replace(':', ' ')}.wav", frames)

        serum = {"ok": False, "error": "skipped"}
        if opts.get("verify_serum", True):
            job.msg("rendering in real Serum 2 for verification")
            serum = serum_host.verify(preset, a.notes, y44, 44100)
            if serum.get("ok"):
                audio.save(out / "serum_render.wav", audio.normalize(serum.pop("audio")), 44100)
                job.msg(f"real Serum score: {serum['score']}")
            else:
                job.msg(f"Serum verification unavailable: {serum.get('error')}")
        summary = {
            "id": rid, "stem": stem, "start": start, "end": end, "bpm": bpm, "name": name,
            "accuracy": round(res.score, 2), "details": res.details, "target": o.target_score,
            "met_target": res.score >= o.target_score, "used_custom_wavetable": res.used_custom,
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
