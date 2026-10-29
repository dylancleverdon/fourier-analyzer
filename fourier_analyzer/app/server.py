"""Local HTTP API + static UI (served only on 127.0.0.1)."""
from __future__ import annotations

import io
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from .. import audio, config, project, updater

log = logging.getLogger(__name__)
STATIC = Path(__file__).with_name("static")

app = FastAPI(title="Fourier Analyzer")
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")

_update_state: dict = {"info": None, "applying": False, "log": []}


@app.get("/", response_class=HTMLResponse)
def index():
    html = (STATIC / "index.html").read_text()
    return html.replace("{{VERSION}}", config.version())


# ---------------------------------------------------------------- system


@app.get("/api/system")
def system():
    from ..separation.stems import ml_available
    try:
        import pedalboard  # noqa: F401
        pb = True
    except Exception:
        pb = False
    return {
        "version": config.version(),
        "ml_separation": ml_available(),
        "pedalboard": pb,
        "serum_plugin": str(config.serum_plugin_path() or ""),
        "serum_tables": str(config.serum_tables_dir() or ""),
        "serum_presets": str(config.serum_presets_dir() or ""),
        "exports": str(config.exports_dir()),
        "preset_install_dir": str(config.serum_user_presets_dir()),
        "diagnostics": str(config.diagnostics_path()) if config.diagnostics_path().exists() else "",
        "settings": config.settings(),
    }


class PickIn(BaseModel):
    prompt: str = "Choose a folder"
    start: str | None = None


@app.post("/api/pick-folder")
def pick_folder(body: PickIn):
    """Native folder chooser (macOS Finder dialog)."""
    import subprocess
    import sys
    if sys.platform == "darwin":
        start = body.start if body.start and Path(body.start).expanduser().exists() else str(Path.home())
        script = [
            "tell application \"System Events\"", "activate",
            f"set f to choose folder with prompt \"{body.prompt}\" default location (POSIX file \"{start}\")",
            "end tell", "POSIX path of f"]
        args = ["osascript"]
        for line in script:
            args += ["-e", line]
        r = subprocess.run(args, capture_output=True, text=True)
        if r.returncode != 0:
            return {"cancelled": True}
        return {"path": r.stdout.strip().rstrip("/")}
    try:
        import webview
        res = webview.windows[0].create_file_dialog(getattr(webview, "FOLDER_DIALOG", 20))
        return {"path": res[0]} if res else {"cancelled": True}
    except Exception as e:
        raise HTTPException(501, f"no folder dialog available: {e}")


@app.post("/api/window/new")
def new_window(request: Request):
    """Open another analyzer window on the same server (each window works on
    its own project; their jobs run in parallel)."""
    url = str(request.base_url)
    try:
        import webview
        if not webview.windows:
            raise RuntimeError("no native window")
        n = len(webview.windows) + 1
        webview.create_window(f"Fourier Analyzer {config.version()} ({n})", url, width=1360, height=900,
                              min_size=(900, 600))
        return {"opened": True, "windows": n}
    except Exception:
        return {"opened": False, "url": url}     # browser mode: the page opens a tab


@app.get("/api/diagnostics")
def diagnostics_file():
    p = config.diagnostics_path()
    if not p.exists():
        raise HTTPException(404, "no diagnostics yet (run Calibrate with Serum 2 first)")
    return FileResponse(p, filename="serum_diagnostics.txt", media_type="text/plain")


@app.post("/api/diagnostics/reveal")
def reveal_diagnostics():
    import subprocess
    import sys
    p = config.diagnostics_path()
    if not p.exists():
        raise HTTPException(404, "no diagnostics yet")
    if sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(p)])
    return {"path": str(p)}


class SettingsIn(BaseModel):
    values: dict


@app.post("/api/settings")
def set_settings(body: SettingsIn):
    return config.save_settings(**body.values)


# ---------------------------------------------------------------- updates


@app.get("/api/update/check")
def update_check():
    info = updater.check()
    _update_state["info"] = info
    return info


@app.post("/api/update/apply")
def update_apply():
    if _update_state["applying"]:
        return {"status": "already running"}
    _update_state["applying"] = True
    _update_state["log"] = []

    def run():
        try:
            res = updater.apply(log_cb=_update_state["log"].append)
            _update_state["result"] = res
            if res.get("updated"):
                _update_state["log"].append("restarting...")
                threading.Timer(1.0, updater.relaunch).start()
        except Exception as e:
            _update_state["log"].append(f"update failed: {e}")
        finally:
            _update_state["applying"] = False
    threading.Thread(target=run, daemon=True).start()
    return {"status": "started"}


@app.get("/api/update/status")
def update_status():
    return {"applying": _update_state["applying"], "log": _update_state["log"],
            "result": _update_state.get("result")}


# ---------------------------------------------------------------- projects


@app.get("/api/projects")
def projects():
    return project.list_projects()


@app.post("/api/projects")
async def upload(file: UploadFile = File(...), drums: bool = True, separate: bool = False):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        meta = project.create_project(tmp_path, file.filename or "audio")
    finally:
        tmp_path.unlink(missing_ok=True)
    # by default the user first clips the region they want, then separates
    job = project.start_separation(meta["id"], drums=drums) if separate else None
    return {"project": meta, "job": job.id if job else None}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    try:
        return project.load_meta(pid)
    except FileNotFoundError:
        raise HTTPException(404, "no such project")


class SeparateIn(BaseModel):
    start: float | None = None
    end: float | None = None
    drums: bool = True
    clip: bool = False          # true: cut the mix to start/end of the original first


@app.delete("/api/projects/{pid}")
def delete_project(pid: str):
    try:
        project.delete_project(pid)
    except FileNotFoundError:
        raise HTTPException(404, "no such project")
    except RuntimeError as e:
        raise HTTPException(409, str(e))
    return {"deleted": pid}


@app.post("/api/projects/{pid}/separate")
def reseparate(pid: str, body: SeparateIn | None = None):
    body = body or SeparateIn()
    clip = (body.start, body.end) if body.clip else None
    return {"job": project.start_separation(pid, drums=body.drums, clip=clip).id}


class ClipIn(BaseModel):
    start: float | None = None
    end: float | None = None


@app.post("/api/projects/{pid}/clip")
def clip(pid: str, body: ClipIn):
    """Without start/end: restore the full original so a new region can be picked."""
    try:
        meta = project.clip_mix(pid, body.start, body.end)
    except ValueError as e:
        raise HTTPException(400, str(e))
    meta["status"] = "new"
    project.save_meta(pid, meta)
    return meta


class CombineIn(BaseModel):
    stems: list[str]
    name: str | None = None


@app.post("/api/projects/{pid}/combine")
def combine(pid: str, body: CombineIn):
    try:
        return {"stem": project.combine_stems(pid, body.stems, body.name)}
    except (ValueError, KeyError) as e:
        raise HTTPException(400, str(e))


def _stem_path(pid: str, stem: str) -> Path:
    meta = project.load_meta(pid)
    if stem not in meta["stems"]:
        raise HTTPException(404, "no such stem")
    return project.project_dir(pid) / meta["stems"][stem]


@app.get("/api/projects/{pid}/audio/{stem}")
def stem_audio(pid: str, stem: str):
    return FileResponse(_stem_path(pid, stem), media_type="audio/wav", filename=f"{stem}.wav")


@app.get("/api/projects/{pid}/peaks/{stem}")
def stem_peaks(pid: str, stem: str, n: int = 1600, start: float | None = None, end: float | None = None):
    """Waveform overview. With start/end only that window is returned (for
    zooming), with signed min/max per column so single cycles are visible."""
    y, sr = _load_cached(_stem_path(pid, stem))
    dur = len(y) / sr
    n = max(50, min(n, 8000))
    a = int(max(0.0, start or 0.0) * sr)
    b = int(min(dur, end if end else dur) * sr)
    seg = y[a:max(b, a + 1)]
    if len(seg) <= n:          # fewer samples than columns: return samples
        return {"duration": dur, "start": a / sr, "end": b / sr, "peaks": np.abs(seg).round(4).tolist(),
                "mins": seg.round(4).tolist(), "maxs": seg.round(4).tolist()}
    edges = np.linspace(0, len(seg), n + 1).astype(int)
    mx = np.maximum.reduceat(seg, edges[:-1])
    mn = np.minimum.reduceat(seg, edges[:-1])
    return {"duration": dur, "start": a / sr, "end": b / sr,
            "peaks": np.maximum(np.abs(mx), np.abs(mn)).round(4).tolist(),
            "mins": mn.round(4).tolist(), "maxs": mx.round(4).tolist()}


_audio_cache: dict = {}


def _load_cached(path: Path):
    """Zooming fetches many windows; keep the last few stems decoded."""
    key = (str(path), path.stat().st_mtime)
    if key not in _audio_cache:
        if len(_audio_cache) > 6:
            _audio_cache.pop(next(iter(_audio_cache)))
        _audio_cache[key] = audio.load(path, sr=22050, mono=True)
    return _audio_cache[key]


@app.get("/api/projects/{pid}/spectrogram/{stem}")
def spectrogram(pid: str, stem: str, start: float = 0.0, end: float = 0.0, w: int = 1200, h: int = 300):
    import librosa
    from PIL import Image
    y, sr = audio.load(_stem_path(pid, stem), sr=22050, mono=True)
    if end > start:
        y = y[int(start * sr): int(end * sr)]
    if len(y) < 2048:
        raise HTTPException(400, "selection too short")
    S = np.abs(librosa.stft(y, n_fft=2048, hop_length=max(64, len(y) // w)))
    M = librosa.feature.melspectrogram(S=S ** 2, sr=sr, n_mels=h, fmin=20, fmax=sr / 2)
    db = librosa.power_to_db(M, ref=np.max, top_db=80)
    v = ((db + 80) / 80).clip(0, 1)[::-1]
    # perceptual palette: black -> purple -> orange -> yellow
    stops = np.array([[0, 0, 0], [40, 10, 80], [150, 30, 120], [240, 110, 40], [255, 240, 150]], float)
    xs = np.linspace(0, 1, len(stops))
    rgb = np.stack([np.interp(v, xs, stops[:, c]) for c in range(3)], -1).astype(np.uint8)
    img = Image.fromarray(rgb).resize((w, h))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return Response(buf.getvalue(), media_type="image/png")


@app.post("/api/projects/{pid}/tempo")
def tempo(pid: str):
    return project.redetect_tempo(pid)


class BpmIn(BaseModel):
    bpm: float


@app.post("/api/projects/{pid}/bpm")
def set_bpm(pid: str, body: BpmIn):
    project.set_bpm(pid, body.bpm)
    return {"bpm": body.bpm}


class SplitIn(BaseModel):
    stem: str
    layers: int = 2
    start: float | None = None
    end: float | None = None


@app.post("/api/projects/{pid}/split")
def split(pid: str, body: SplitIn):
    return {"job": project.start_layer_split(pid, body.stem, max(2, min(body.layers, 4)),
                                             body.start, body.end).id}


class RegionIn(BaseModel):
    stem: str
    start: float | None = None
    end: float | None = None
    bpm: float | None = None
    target: float = 90.0
    time_budget: float = 240.0
    verify_serum: bool = True


@app.post("/api/projects/{pid}/analyze")
def analyze(pid: str, body: RegionIn):
    try:
        return project.quick_analysis(pid, body.stem, body.start, body.end, body.bpm)
    except ValueError as e:
        raise HTTPException(400, str(e))


@app.post("/api/projects/{pid}/reconstruct")
def reconstruct(pid: str, body: RegionIn):
    job = project.start_reconstruct(pid, body.stem, body.start, body.end, body.model_dump())
    return {"job": job.id}


@app.get("/api/jobs")
def jobs(project: str | None = None):
    return [{"id": j.id, "kind": j.kind, "status": j.status} for j in project_mod_jobs(project)]


def project_mod_jobs(pid):
    return [j for j in project.JOBS.jobs.values()
            if j.status in ("queued", "running") and (pid is None or j.project == pid)]


@app.get("/api/jobs/{jid}")
def job(jid: str):
    j = project.JOBS.get(jid)
    if not j:
        raise HTTPException(404, "no such job")
    return j.to_dict()


@app.get("/api/projects/{pid}/results/{rid}")
def result(pid: str, rid: str):
    p = project.project_dir(pid) / "results" / rid / "report.json"
    if not p.exists():
        raise HTTPException(404, "no such result")
    return JSONResponse(content=__import__("json").loads(p.read_text()))


@app.get("/api/projects/{pid}/results/{rid}/file/{name}")
def result_file(pid: str, rid: str, name: str):
    base = (project.project_dir(pid) / "results" / rid).resolve()
    p = (base / name).resolve()
    if base not in p.parents or not p.exists():
        raise HTTPException(404, "no such file")
    return FileResponse(p, filename=p.name)


@app.post("/api/projects/{pid}/results/{rid}/install")
def install(pid: str, rid: str):
    dst = project.install_preset(pid, rid)
    return {"installed": str(dst)}


@app.post("/api/projects/{pid}/results/{rid}/reveal")
def reveal(pid: str, rid: str):
    import subprocess
    import sys
    p = project.project_dir(pid) / "results" / rid
    if sys.platform == "darwin":
        subprocess.Popen(["open", str(p)])
    return {"path": str(p)}


@app.post("/api/calibrate")
def calibrate():
    from ..serum.host import calibrate

    def work(job):
        return calibrate(progress=job.msg)
    return {"job": project.JOBS.submit("calibrate", work).id}
