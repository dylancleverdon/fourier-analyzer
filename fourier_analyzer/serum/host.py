"""Host the real Serum 2 plugin (macOS) to verify and calibrate patches.

Uses Spotify's ``pedalboard`` to load Serum 2 (AU or VST3), injects our
preset into the plugin state, renders the detected MIDI notes and scores the
result against the original. Serum keeps its preset inside the plugin state
blob (the same ``XferJson`` container as .SerumPreset files), so we locate
that container inside ``raw_state`` and swap it for ours. Everything is
best-effort and reports clearly when it cannot work.
"""
from __future__ import annotations

import base64
import logging
import plistlib
import struct
from pathlib import Path

import numpy as np

from .. import config
from ..export import BEND_RANGE
from .preset import MAGIC

log = logging.getLogger(__name__)


class SerumHostError(RuntimeError):
    pass


def _preset_len(blob: bytes, start: int) -> int:
    """Length of an XferJson container starting at ``start`` in blob."""
    import zstandard
    pos = start + len(MAGIC)
    (jlen,) = struct.unpack_from("<Q", blob, pos)
    pos += 8 + jlen
    pos += 8   # uncompressed size + version
    d = zstandard.ZstdDecompressor().decompressobj()
    d.decompress(blob[pos:])
    unused = len(d.unused_data) if hasattr(d, "unused_data") else 0
    return (len(blob) - unused) - start


ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def _zstd_frames(blob: bytes):
    """(offset, compressed_len, decompressed bytes) of every zstd frame."""
    import zstandard
    pos = 0
    while True:
        k = blob.find(ZSTD_MAGIC, pos)
        if k < 0:
            return
        try:
            d = zstandard.ZstdDecompressor().decompressobj()
            out = d.decompress(blob[k:])
            used = len(blob) - k - len(d.unused_data)
            yield k, used, out
        except Exception:
            pass
        pos = k + 4


def _fix_lengths(new: bytes, k: int, old_vals: dict, new_vals: dict) -> bytes:
    """Rewrite little/big-endian u32/u64 length fields in the 16 bytes before
    offset k whose value equals one of the old lengths."""
    b = bytearray(new)
    for fmt, size in (("<I", 4), ("<Q", 8), (">I", 4)):
        for off in range(max(0, k - 16), k - size + 1):
            (v,) = struct.unpack_from(fmt, b, off)
            for key, ov in old_vals.items():
                if v == ov and ov != new_vals[key] and ov > 16:
                    struct.pack_into(fmt, b, off, new_vals[key])
    return bytes(b)


def _swap_payload(blob: bytes, preset: bytes) -> bytes | None:
    """Serum may store only the zstd-compressed CBOR state (no XferJson
    header). Replace that frame with ours and fix nearby length fields."""
    import cbor2
    import zstandard
    from .preset import encode_state, read_preset_bytes
    _, state = read_preset_bytes(preset)
    payload = encode_state(state)
    comp = zstandard.ZstdCompressor(level=3).compress(payload)
    for k, used, raw in _zstd_frames(blob):
        try:
            obj = cbor2.loads(raw)
        except Exception:
            continue
        if not (isinstance(obj, dict) and any(str(x).startswith(("Oscillator", "Env", "FXRack")) for x in obj)):
            continue
        new = blob[:k] + comp + blob[k + used:]
        new = _fix_lengths(new, k, {"c": used, "u": len(raw), "t": len(blob)},
                           {"c": len(comp), "u": len(payload), "t": len(new)})
        # a total length at the very start of the blob
        for fmt, size in (("<I", 4), ("<Q", 8), (">I", 4)):
            if len(blob) >= size:
                (v,) = struct.unpack_from(fmt, blob, 0)
                if v in (len(blob), len(blob) - size):
                    nb = bytearray(new)
                    struct.pack_into(fmt, nb, 0, len(new) - (len(blob) - v))
                    new = bytes(nb)
                    break
        return new
    return None


def describe_blob(blob: bytes) -> str:
    """Short structural summary of an unknown plugin state (for diagnostics)."""
    parts = [f"{len(blob)} bytes", "head " + blob[:24].hex()]
    if blob.find(MAGIC) >= 0:
        parts.append(f"XferJson@{blob.find(MAGIC)}")
    for k, used, raw in _zstd_frames(blob):
        parts.append(f"zstd@{k} {used}->{len(raw)}B head {raw[:8].hex()}")
    if blob[:6] == b"bplist" or blob[:5] == b"<?xml":
        parts.append("plist")
    return "; ".join(parts)


def _swap(blob: bytes, preset: bytes) -> bytes | None:
    k = blob.find(MAGIC)
    if k < 0:
        return _swap_payload(blob, preset)
    old_len = _preset_len(blob, k)
    new = blob[:k] + preset + blob[k + old_len:]
    # fix a length prefix just before the container, if there is one
    for fmt, size in (("<Q", 8), ("<I", 4), (">I", 4)):
        if k >= size:
            (v,) = struct.unpack_from(fmt, blob, k - size)
            if v == old_len:
                new = new[:k - size] + struct.pack(fmt, len(preset)) + new[k:]
                break
            if v == len(blob) - k:
                new = new[:k - size] + struct.pack(fmt, len(new) - k) + new[k:]
                break
    return new


def _swap_in_state(state: bytes, preset: bytes) -> bytes | None:
    """Handle raw blobs, AU plists and base64-in-plist variants."""
    direct = _swap(state, preset)
    if direct is not None:
        return direct
    try:
        pl = plistlib.loads(state)
    except Exception:
        return None

    def walk(obj):
        if isinstance(obj, (bytes, bytearray)):
            r = _swap(bytes(obj), preset)
            return (r, True) if r is not None else (obj, False)
        if isinstance(obj, str) and len(obj) > 64:
            try:
                raw = base64.b64decode(obj, validate=True)
            except Exception:
                return obj, False
            r = _swap(raw, preset)
            return (base64.b64encode(r).decode(), True) if r is not None else (obj, False)
        if isinstance(obj, dict):
            hit = False
            for kk in list(obj):
                obj[kk], h = walk(obj[kk])
                hit |= h
            return obj, hit
        if isinstance(obj, list):
            hit = False
            for i, v in enumerate(obj):
                obj[i], h = walk(v)
                hit |= h
            return obj, hit
        return obj, False

    pl, hit = walk(pl)
    if not hit:
        return None
    return plistlib.dumps(pl, fmt=plistlib.FMT_BINARY)


def _open_plugin(paths: list[Path]):
    """Load the Serum 2 *instrument* (not "Serum 2 FX") from the first plugin
    file that works. Every failure is collected so the real reason is shown."""
    import re

    import pedalboard
    errors = []
    for path in paths:
        tried = set()
        queue = ["Serum 2"]
        while queue:
            name = queue.pop(0)
            if name in tried:
                continue
            tried.add(name)
            try:
                return pedalboard.load_plugin(str(path), plugin_name=name), f"{path} [{name}]"
            except Exception as e:
                msg = str(e)
                errors.append(f"{path.name} as '{name}': {type(e).__name__}: {msg[:300]}")
                # pedalboard lists the plugins inside the file in the message
                for found in re.findall(r'"([^"]+)"', msg):
                    if "fx" not in found.lower() and found not in tried:
                        queue.append(found)
    raise SerumHostError("Could not load the Serum 2 instrument. Tried: " + " | ".join(errors)
                         if errors else "Serum 2 plugin not found (looked for Serum2.vst3 / Serum2.component)")


class SerumHost:
    """Runs inside the helper process (main thread). Use render_jobs() from
    the app, never this class directly on a worker thread."""

    def __init__(self, plugin_path: str | Path | None = None, sr: int = 44100):
        paths = [Path(plugin_path)] if plugin_path else config.serum_plugin_paths()
        if not paths:
            raise SerumHostError("Serum 2 plugin not found (looked for Serum2.vst3 / Serum2.component)")
        try:
            import pedalboard  # noqa: F401
        except Exception as e:
            raise SerumHostError(f"pedalboard not installed: {e}")
        self.sr = sr
        self.plugin, self.description = _open_plugin(paths)
        self.base_state = bytes(self.plugin.raw_state)

    def load_preset(self, preset: bytes) -> None:
        new = _swap_in_state(self.base_state, preset)
        if new is None:
            dump = config.app_dir() / "serum_state_dump.bin"
            try:
                dump.write_bytes(self.base_state)
            except OSError:
                pass
            raise SerumHostError("could not find Serum's preset data in the plugin state ("
                                 + describe_blob(self.base_state) + f"). Saved to {dump}")
        self.plugin.raw_state = new
        # confirm Serum actually took the preset: its state should now hold
        # our filter settings / table data rather than the init patch
        after = bytes(self.plugin.raw_state)
        if after == self.base_state:
            raise SerumHostError("Serum ignored the injected state (it still reports the init patch)")

    def render(self, notes, duration: float) -> np.ndarray:
        import mido
        msgs = []
        for n in notes:
            p = int(round(n.pitch))
            v = int(np.clip(round(n.velocity * 127), 1, 127))
            msgs.append(mido.Message("note_on", note=p, velocity=v, time=float(n.start)))
            msgs.append(mido.Message("note_off", note=p, velocity=0, time=float(n.start + n.duration)))
            if n.bend:
                for t, semis in n.bend:
                    val = int(np.clip(semis / BEND_RANGE * 8191, -8192, 8191))
                    msgs.append(mido.Message("pitchwheel", pitch=val, time=float(n.start + t)))
        msgs.sort(key=lambda m: m.time)
        out = self.plugin(msgs, duration=duration + 0.5, sample_rate=self.sr, num_channels=2,
                          buffer_size=512, reset=True)
        y = np.asarray(out, dtype=np.float32)
        if y.ndim == 2:
            y = y.mean(axis=0)
        return y[: int(duration * self.sr)]


# ----------------------------------------------------------------------------
# helper process: plugins are loaded on that process's main thread, and a
# plugin crash cannot take the app down


def host_main(job_file: str) -> int:
    """Entry point of ``python -m fourier_analyzer serum-host job.json``."""
    import json

    import soundfile as sf
    from ..synth.patch import Note
    job = json.loads(Path(job_file).read_text())
    result = {"ok": False}
    try:
        host = SerumHost(sr=job.get("sr", 44100))
        result["plugin"] = host.description
        for r in job["renders"]:
            host.load_preset(Path(r["preset"]).read_bytes())
            notes = [Note(**n) for n in r["notes"]]
            y = host.render(notes, float(r["duration"]))
            sf.write(r["out"], y, host.sr, subtype="FLOAT")
        result["ok"] = True
    except Exception as e:
        import traceback
        result["error"] = f"{type(e).__name__}: {e}"
        result["trace"] = traceback.format_exc()[-2000:]
    Path(job["result"]).write_text(json.dumps(result))
    return 0 if result["ok"] else 1


def render_jobs(renders: list[tuple[bytes, list, float]], sr: int = 44100, timeout: float | None = None):
    """Render several (preset_bytes, notes, duration) in real Serum 2 via the
    helper process. Returns (list of arrays, plugin description)."""
    import json
    import subprocess
    import sys
    import tempfile

    import soundfile as sf
    if not config.serum_plugin_paths():
        raise SerumHostError("Serum 2 plugin not found (looked for Serum2.vst3 / Serum2.component)")
    with tempfile.TemporaryDirectory(prefix="fa-serum-") as td:
        td = Path(td)
        job = {"sr": sr, "result": str(td / "result.json"), "renders": []}
        for i, (preset, notes, dur) in enumerate(renders):
            (td / f"p{i}.SerumPreset").write_bytes(preset)
            job["renders"].append({"preset": str(td / f"p{i}.SerumPreset"),
                                   "notes": [n.to_dict() if hasattr(n, "to_dict") else n for n in notes],
                                   "duration": dur, "out": str(td / f"r{i}.wav")})
        (td / "job.json").write_text(json.dumps(job))
        cmd = [sys.executable, "-m", "fourier_analyzer", "serum-host", str(td / "job.json")]
        try:
            proc = subprocess.run(cmd, cwd=str(config.code_dir()), capture_output=True, text=True,
                                  timeout=timeout or (90 + 30 * len(renders)))
        except subprocess.TimeoutExpired:
            raise SerumHostError("Serum 2 did not finish rendering in time (helper process timed out)")
        res_path = td / "result.json"
        if not res_path.exists():
            tail = (proc.stderr or proc.stdout or "")[-1500:]
            raise SerumHostError(f"Serum helper process crashed (exit {proc.returncode}). {tail}")
        res = json.loads(res_path.read_text())
        if not res.get("ok"):
            raise SerumHostError(res.get("error", "unknown error"))
        out = [sf.read(r["out"], dtype="float32")[0] for r in job["renders"]]
        return out, res.get("plugin", "")


def verify(preset: bytes, notes, y_target: np.ndarray, sr: int, gain=None) -> dict:
    """Render the preset in real Serum and score it. Never raises."""
    try:
        (y,), plugin = render_jobs([(preset, notes, len(y_target) / sr)], sr=sr)
        if not np.any(np.abs(y) > 1e-5):
            return {"ok": False, "error": "Serum rendered silence (preset not accepted?)"}
        from ..match.metrics import Reference
        if gain is not None:
            y = y[: len(gain)] * gain[: len(y)]
        d = Reference(np.asarray(y_target, dtype=np.float64), sr).all(y)
        return {"ok": True, "score": d["waveform_similarity"], "details": d, "audio": y, "plugin": plugin}
    except Exception as e:
        log.warning("Serum verification unavailable: %s", e)
        return {"ok": False, "error": str(e)}


def calibrate_cutoff(progress=None) -> dict:
    """Measure Serum's cutoff curve: render white noise through L12 at several
    kParamFreq values, find the -3 dB point and fit the exponential map."""
    from ..synth.patch import Filter, Note, NoiseOsc, Osc, Patch
    from .mapping import patch_to_state
    from .preset import write_preset_bytes
    xs_all = list(np.linspace(0.3, 0.85, 6))
    renders = []
    for x in xs_all:
        p = Patch()
        p.oscs[0] = Osc(enabled=False)
        p.noise = NoiseOsc(enabled=True, color=0.5, level=1.0)
        p.filter = Filter(enabled=True, type="lp12", cutoff=1000)
        st = patch_to_state(p)
        st["VoiceFilter0"]["plainParams"]["kParamFreq"] = float(x)
        renders.append((write_preset_bytes(st, "cal"), [Note(0, 2.0, 60)], 2.0))
    if progress:
        progress("rendering 6 test patches in Serum 2 (helper process)")
    ys, plugin = render_jobs(renders)
    if progress:
        progress(f"loaded {plugin}")
    xs, fs = [], []
    for x, y in zip(xs_all, ys):
        if len(y) <= 8820 or not np.any(np.abs(y) > 1e-6):
            if progress:
                progress(f"cutoff {x:.2f}: silent render")
            continue
        seg = y[4410:]
        spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2
        fr = np.fft.rfftfreq(len(seg), 1 / 44100)
        sm = np.convolve(spec, np.ones(200) / 200, mode="same")
        ref = np.median(sm[(fr > 40) & (fr < 80)])
        below = np.nonzero((fr > 50) & (sm < ref * 0.5))[0]
        if len(below):
            xs.append(float(x))
            fs.append(float(fr[below[0]]))
        if progress:
            progress(f"cutoff {x:.2f} -> {fs[-1] if fs else '?'} Hz")
    if len(xs) < 3:
        raise SerumHostError("calibration failed: Serum rendered silence or no measurable cutoff "
                             "(the preset may not have been accepted)")
    k, c = np.polyfit(xs, np.log(fs), 1)
    lo, hi = float(np.exp(c)), float(np.exp(c + k))
    s = config.settings().get("serum_scales", {})
    s.update({"cutoff_lo_hz": lo, "cutoff_hi_hz": hi})
    config.save_settings(serum_scales=s)
    return {"cutoff_lo_hz": lo, "cutoff_hi_hz": hi, "points": list(zip(xs, fs)), "plugin": plugin}
