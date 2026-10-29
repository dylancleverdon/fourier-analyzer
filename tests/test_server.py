import io
import time

import numpy as np
import soundfile as sf
from fastapi.testclient import TestClient

from fourier_analyzer.app.server import app
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import Filter, Note, Osc, Patch


def _song_wav():
    r = Renderer(sr=44100)
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, cutoff=500)
    y = r.render(p, [Note(0, 0.5, 36), Note(0.6, 0.5, 39), Note(1.2, 0.8, 34)], 2.2)
    rng = np.random.default_rng(0)
    for t in (0.0, 0.6, 1.2, 1.8):    # add some clicks as "drums"
        i = int(t * 44100)
        y[i:i + 2000] += rng.standard_normal(2000) * np.exp(-np.arange(2000) / 300) * 0.5
    buf = io.BytesIO()
    sf.write(buf, y, 44100, format="WAV")
    return buf.getvalue()


def _wait(c, jid, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{jid}").json()
        if j["status"] in ("done", "error"):
            return j
        time.sleep(0.3)
    raise AssertionError("job timed out")


def test_full_flow():
    c = TestClient(app)
    assert c.get("/").status_code == 200
    assert "version" in c.get("/api/system").json()
    r = c.post("/api/projects", files={"file": ("loop.wav", _song_wav(), "audio/wav")})
    assert r.status_code == 200, r.text
    pid = r.json()["project"]["id"]
    assert r.json()["job"] is None                       # nothing separated until clipped
    # clip to 0.1-2.1 s of the original, then separate
    jid = c.post(f"/api/projects/{pid}/separate", json={"clip": True, "start": 0.1, "end": 2.1}).json()["job"]
    j = _wait(c, jid)
    assert j["status"] == "done", j
    meta = c.get(f"/api/projects/{pid}").json()
    assert abs(meta["duration"] - 2.0) < 0.01 and meta["clip"] == [0.1, 2.1]
    assert "bass" in meta["stems"]
    assert c.get(f"/api/projects/{pid}/peaks/bass").status_code == 200
    # combine two stems into a new one
    cb = c.post(f"/api/projects/{pid}/combine", json={"stems": ["bass", "other"], "name": "bass+other"})
    assert cb.status_code == 200, cb.text
    assert cb.json()["stem"] in c.get(f"/api/projects/{pid}").json()["stems"]
    a = c.post(f"/api/projects/{pid}/analyze", json={"stem": "mix", "start": 0.0, "end": 1.9})
    assert a.status_code == 200, a.text
    assert len(a.json()["notes"]) >= 1
    rr = c.post(f"/api/projects/{pid}/reconstruct",
                json={"stem": "bass", "start": 0.0, "end": 1.9, "time_budget": 15, "verify_serum": False})
    j = _wait(c, rr.json()["job"], timeout=300)
    assert j["status"] == "done", j.get("error")
    res = j["result"]
    assert any(f.endswith(".SerumPreset") for f in res["files"])
    assert {"waveform_similarity", "spectral_similarity", "si_sdr_db"} <= set(res["details"])
    assert "sidechain" in res and res["patch"]["oscs"][0]["table"].startswith("custom:")
    f = c.get(f"/api/projects/{pid}/results/{res['id']}/file/notes.mid")
    assert f.status_code == 200
