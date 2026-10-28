"""Local HTTP API + static UI (served only on 127.0.0.1)."""
from __future__ import annotations

import io
import logging
import tempfile
import threading
from pathlib import Path

import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
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
        "settings": config.settings(),
    }


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
async def upload(file: UploadFile = File(...), drums: bool = True):
    suffix = Path(file.filename or "audio.wav").suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await file.read())
        tmp_path = Path(tmp.name)
    try:
        meta = project.create_project(tmp_path, file.filename or "audio")
    finally:
        tmp_path.unlink(missing_ok=True)
    job = project.start_separation(meta["id"], drums=drums)
    return {"project": meta, "job": job.id}


@app.get("/api/projects/{pid}")
def get_project(pid: str):
    try:
        return project.load_meta(pid)
    except FileNotFoundError:
        raise HTTPException(404, "no such project")


@app.post("/api/projects/{pid}/separate")
def reseparate(pid: str):
    return {"job": project.start_separation(pid).id}


def _stem_path(pid: str, stem: str) -> Path:
    meta = project.load_meta(pid)
    if stem not in meta["stems"]:
        raise HTTPException(404, "no such stem")
    return project.project_dir(pid) / meta["stems"][stem]


@app.get("/api/projects/{pid}/audio/{stem}")
def stem_audio(pid: str, stem: str):
    return FileResponse(_stem_path(pid, stem), media_type="audio/wav", filename=f"{stem}.wav")


@app.get("/api/projects/{pid}/peaks/{stem}")
def stem_peaks(pid: str, stem: str, n: int = 1600):
    y, sr = audio.load(_stem_path(pid, stem), sr=22050, mono=True)
    n = max(100, min(n, 6000))
    hop = max(1, len(y) // n)
    m = len(y) // hop
    fr = np.abs(y[: m * hop]).reshape(m, hop)
    return {"duration": len(y) / sr, "peaks": fr.max(1).round(4).tolist()}


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
    allow_custom: bool = True
    stock_only: bool = False
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
    from ..serum.host import calibrate_cutoff

    def work(job):
        return calibrate_cutoff(progress=job.msg)
    return {"job": project.JOBS.submit("calibrate", work).id}
