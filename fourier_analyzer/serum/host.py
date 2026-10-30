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
import re
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


# ----------------------------------------------------------------------------
# JUCE state wrapper: b"VC2!" + uint32 length + UTF-8 XML + NUL. Binary blobs
# inside the XML (e.g. <IComponent> of a VST3) use JUCE's own base64:
# "<numBytes>." followed by 6-bit groups read LSB-first from the bytes, with
# the alphabet below (MemoryBlock::toBase64Encoding).
JUCE_B64 = ".ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+"
_JUCE_IDX = {c: i for i, c in enumerate(JUCE_B64)}
_JUCE_TEXT = re.compile(r">(\d+\.[.A-Za-z0-9+]+)<")


def juce_b64_encode(data: bytes) -> str:
    n = len(data)
    nchars = ((n << 3) + 5) // 6
    bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8), bitorder="little")
    bits = np.concatenate([bits, np.zeros(nchars * 6 - len(bits), dtype=np.uint8)])
    vals = bits.reshape(-1, 6).astype(np.int64) @ (1 << np.arange(6))
    return f"{n}." + "".join(JUCE_B64[v] for v in vals)


def juce_b64_decode(text: str) -> bytes:
    num, _, body = text.strip().partition(".")
    n = int(num)
    vals = np.array([_JUCE_IDX[c] for c in body], dtype=np.uint8)
    bits = ((vals[:, None] >> np.arange(6)) & 1).astype(np.uint8).ravel()
    out = np.packbits(bits, bitorder="little").tobytes()
    return (out + b"\0" * n)[:n]


def _vc2_xml(blob: bytes) -> str | None:
    if blob[:4] != b"VC2!" or len(blob) < 8:
        return None
    (n,) = struct.unpack_from("<I", blob, 4)
    return blob[8:8 + n].decode("utf-8", errors="strict")


def _vc2_wrap(xml: str) -> bytes:
    b = xml.encode("utf-8")
    return b"VC2!" + struct.pack("<I", len(b)) + b + b"\0"


_CTRL_RE = re.compile(r"<IEditController>[^<]*</IEditController>|<IEditController\s*/>")


def _swap_vc2(blob: bytes, preset: bytes, drop_controller: bool = False) -> bytes | None:
    """Swap the preset inside the first JUCE-base64 blob of a VC2! XML state
    that contains Serum's data."""
    xml = _vc2_xml(blob)
    if xml is None:
        return None
    done = False

    def repl(m):
        nonlocal done
        if done:
            return m.group(0)
        try:
            inner = juce_b64_decode(m.group(1))
        except Exception:
            return m.group(0)
        new_inner = _swap_in_state(inner, preset)
        if new_inner is None:
            return m.group(0)
        done = True
        return ">" + juce_b64_encode(new_inner) + "<"
    new_xml = _JUCE_TEXT.sub(repl, xml)
    if done and drop_controller:
        # without a controller blob the host only applies the component state,
        # so an old controller state cannot reset the preset we just loaded
        new_xml = _CTRL_RE.sub("", new_xml)
    return _vc2_wrap(new_xml) if done else None


def describe_blob(blob: bytes, depth: int = 0) -> str:
    """Short structural summary of an unknown plugin state (for diagnostics)."""
    parts = [f"{len(blob)} bytes", "head " + blob[:24].hex()]
    if blob.find(MAGIC) >= 0:
        parts.append(f"XferJson@{blob.find(MAGIC)}")
    for k, used, raw in _zstd_frames(blob):
        info = f"zstd@{k} {used}->{len(raw)}B head {raw[:8].hex()}"
        try:
            import cbor2
            obj = cbor2.loads(raw)
            if isinstance(obj, dict):
                keys = list(obj)
                info += f" cbor dict {len(keys)} keys e.g. {keys[:6]}"
                for key in ("presetName", "productVersion", "version"):
                    if key in obj:
                        info += f" {key}={obj[key]!r}"
        except Exception:
            pass
        parts.append(info)
    if blob[:6] == b"bplist" or blob[:5] == b"<?xml":
        parts.append("plist")
    xml = _vc2_xml(blob) if depth < 2 else None
    if xml:
        tags = re.findall(r"<([A-Za-z_][\w.-]*)", xml)
        parts.append("VC2 xml tags " + ",".join(dict.fromkeys(tags)))
        for m in _JUCE_TEXT.finditer(xml):
            try:
                parts.append("[inner " + describe_blob(juce_b64_decode(m.group(1)), depth + 1) + "]")
            except Exception as e:
                parts.append(f"[inner undecodable: {e}]")
    return "; ".join(parts)


def _deep_merge(old, ours):
    """Our values written over Serum's current state, keeping every key and
    structure Serum's own (newer) format has that our patch doesn't mention."""
    import copy
    if isinstance(old, dict) and isinstance(ours, dict):
        out = copy.copy(old)
        for k, v in ours.items():
            out[k] = _deep_merge(old[k], v) if k in old else v
        return out
    return ours


def _rebuild_container(old: bytes, preset: bytes) -> bytes:
    """New XferJson container in the *plugin's own* format: keep the existing
    header (Serum 2.1+ uses a component header and format version 11, not
    the .SerumPreset file header) and merge our state into its CBOR."""
    import hashlib
    import json

    import zstandard
    from .preset import encode_state, read_preset_bytes
    pos = len(MAGIC)
    (jlen,) = struct.unpack_from("<Q", old, pos)
    header = json.loads(old[pos + 8:pos + 8 + jlen].decode("utf-8"))
    pos += 8 + jlen
    usize, ver = struct.unpack_from("<II", old, pos)
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(old[pos + 8:])
    import cbor2
    base_state = cbor2.loads(raw)
    _, ours = read_preset_bytes(preset)
    merged = _deep_merge(base_state, {k: v for k, v in ours.items()
                                      if k not in ("presetAuthor", "presetDescription", "tags", "hash",
                                                   "product", "productVersion", "version", "vendor",
                                                   "url", "fileType")})
    payload = encode_state(merged)
    comp = zstandard.ZstdCompressor(level=3).compress(payload)
    if "hash" in header:
        header["hash"] = hashlib.md5(comp).hexdigest()
    hjson = json.dumps(header, separators=(",", ":")).encode("utf-8")
    return MAGIC + struct.pack("<Q", len(hjson)) + hjson + struct.pack("<II", len(payload), ver) + comp


# which container strategy to use: "merge" (Serum's header + merged state)
# or "replace" (our .SerumPreset container as-is)
SWAP_MODE = {"mode": "merge"}


def _swap(blob: bytes, preset: bytes) -> bytes | None:
    k = blob.find(MAGIC)
    if k < 0:
        return _swap_payload(blob, preset)
    old_len = _preset_len(blob, k)
    replacement = preset
    if SWAP_MODE["mode"] == "merge":
        try:
            replacement = _rebuild_container(blob[k:k + old_len], preset)
        except Exception as e:
            log.warning("merge into plugin state failed (%s); replacing container", e)
    new = blob[:k] + replacement + blob[k + old_len:]
    preset = replacement
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
    """Handle raw blobs, JUCE VC2! XML states, AU plists and base64-in-plist."""
    if state[:4] == b"VC2!":
        return _swap_vc2(state, preset)
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

    def _pump(self, seconds: float = 0.05) -> None:
        """Let the plugin run (some plugins load presets asynchronously)."""
        try:
            self.plugin([], duration=seconds, sample_rate=self.sr, num_channels=2, buffer_size=512, reset=False)
        except Exception:
            pass

    def load_preset(self, preset: bytes) -> None:
        import time
        if preset == getattr(self, "_loaded", None):
            return
        # wait for the state to differ from what it was before this load (not
        # only from the init patch), so back-to-back loads in one session wait
        # for the new preset
        before = bytes(self.plugin.raw_state)
        attempts = []
        for mode, drop in (("merge", True), ("merge", False), ("replace", True), ("replace", False)):
            SWAP_MODE["mode"] = mode
            if self.base_state[:4] == b"VC2!":
                new = _swap_vc2(self.base_state, preset, drop_controller=drop)
            elif drop:
                new = _swap_in_state(self.base_state, preset)
            else:
                continue
            if new is None:
                break
            try:
                self.plugin.raw_state = new
            except Exception as e:
                attempts.append((f"{mode}, drop_controller={drop}", new, None,
                                 f"setting state raised {type(e).__name__}: {e}"))
                continue
            # confirm Serum took it: its state must stop being the init patch.
            # Serum 2 may load in the background, so give it up to ~3 s.
            after = b""
            for _ in range(30):
                self._pump()
                after = bytes(self.plugin.raw_state)
                if after != self.base_state and after != before:
                    self._loaded = preset
                    return
                time.sleep(0.1)
            attempts.append((f"{mode}, drop_controller={drop}", new, after, "state unchanged after 3 s"))
        diag = self._write_diagnostics(preset, attempts)
        if not attempts:
            raise SerumHostError("could not find Serum's preset data in the plugin state ("
                                 + describe_blob(self.base_state) + f"). Details: {diag}")
        raise SerumHostError("Serum ignored the injected preset (it still reports the init patch). "
                             f"Please send {diag}")

    def _write_diagnostics(self, preset: bytes, attempts) -> Path:
        path = config.diagnostics_path()
        lines = [f"plugin: {getattr(self, 'description', '?')}",
                 f"base state: {describe_blob(self.base_state)}"]
        for drop, new, after, why in attempts:
            lines.append(f"--- attempt {drop}: {why}")
            lines.append(f"injected: {describe_blob(new)}")
            if after is not None:
                lines.append(f"after:    {describe_blob(after)}")
        try:
            names = list(self.plugin.parameters.keys())
            lines.append(f"--- {len(names)} automatable parameters:")
            lines += names[:3000]
        except Exception as e:
            lines.append(f"parameters unavailable: {e}")
        try:
            path.write_text("\n".join(lines))
            (config.app_dir() / "serum_state_dump.bin").write_bytes(self.base_state)
        except OSError:
            pass
        return path

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


class FakeSerumHost:
    """Stands in for Serum when ``FA_FAKE_SERUM`` is set (tests and development
    without the plugin): presets are imported and rendered with our engine.
    ``FA_FAKE_SERUM=cutoff:0.7`` scales filter cutoffs to mimic an engine
    that differs from ours."""

    description = "fake Serum (Fourier Analyzer engine)"

    def __init__(self, sr: int = 44100, spec: str = "1"):
        self.sr = sr
        self.cutoff_scale = 1.0
        for part in spec.split(","):
            k, _, v = part.partition(":")
            if k == "cutoff" and v:
                self.cutoff_scale = float(v)
        self.patch = None

    def load_preset(self, preset: bytes) -> None:
        from ..synth.samples import SampleBank
        from ..synth.wavetables import TableBank
        from .mapping import state_to_patch
        from .preset import read_preset_bytes
        _, st = read_preset_bytes(preset)
        self.bank = TableBank()
        self.samples = SampleBank()
        self.patch = state_to_patch(st, self.bank, samples=self.samples)
        for f in (self.patch.filter, self.patch.filter2):
            f.cutoff = float(np.clip(f.cutoff * self.cutoff_scale, 20.0, 20000.0))

    def render(self, notes, duration: float) -> np.ndarray:
        from ..synth.engine import Renderer
        return Renderer(self.bank, self.sr, samples=self.samples).render(self.patch, notes, duration).astype(np.float32)


def _make_host(sr: int):
    import os
    fake = os.environ.get("FA_FAKE_SERUM")
    return FakeSerumHost(sr, fake) if fake else SerumHost(sr=sr)


# ----------------------------------------------------------------------------
# helper process: plugins are loaded on that process's main thread, and a
# plugin crash cannot take the app down

SESSION_TAG = "FA-SERUM "


def serve_main() -> int:
    """``serum-host serve``: keep Serum loaded and render one request per
    stdin line ({"preset", "notes", "duration", "out"}); every answer is a
    stdout line starting with SESSION_TAG (the plugin may print too)."""
    import json
    import sys

    import soundfile as sf
    from ..synth.patch import Note

    def say(obj):
        sys.stdout.write("\n" + SESSION_TAG + json.dumps(obj) + "\n")
        sys.stdout.flush()
    try:
        host = _make_host(44100)
    except Exception as e:
        say({"ok": False, "error": f"{type(e).__name__}: {e}"})
        return 1
    say({"ok": True, "ready": True, "plugin": host.description})
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            r = json.loads(line)
            if r.get("quit"):
                break
            host.sr = int(r.get("sr", host.sr))
            host.load_preset(Path(r["preset"]).read_bytes())
            y = host.render([Note(**n) for n in r["notes"]], float(r["duration"]))
            sf.write(r["out"], y, host.sr, subtype="FLOAT")
            say({"ok": True})
        except Exception as e:
            say({"ok": False, "error": f"{type(e).__name__}: {e}"})
    return 0


def _parse_value(text: str):
    """'1.25 kHz' -> 1250.0, '12.0 ms' -> 0.012, '440 Hz' -> 440.0, '35 %' -> 35.0."""
    m = re.search(r"(-?\d+(?:\.\d+)?)\s*([kKmM]?)(Hz|hz|s|%|dB|st|ct)?", text or "")
    if not m:
        return None
    v = float(m.group(1))
    pre, unit = m.group(2), (m.group(3) or "")
    if unit.lower() == "hz" and pre in "kK" and pre:
        v *= 1000.0
    if unit == "s" and pre == "m":
        v /= 1000.0
    return v


def host_main(job_file: str) -> int:
    """Entry point of ``python -m fourier_analyzer serum-host job.json``."""
    import json

    import soundfile as sf
    from ..synth.patch import Note
    if job_file == "serve":
        return serve_main()
    job = json.loads(Path(job_file).read_text())
    result = {"ok": False}
    try:
        host = _make_host(job.get("sr", 44100))
        result["plugin"] = host.description
        if job.get("mode") == "params":
            tables = {}
            params = host.plugin.parameters
            for name in job["params"]:
                if name not in params:
                    continue
                prm = params[name]
                rows = []
                for x in np.linspace(0.0, 1.0, int(job.get("steps", 41))):
                    prm.raw_value = float(x)
                    text = getattr(prm, "string_value", None) or str(prm)
                    rows.append([float(x), text, _parse_value(text)])
                tables[name] = rows
            result["tables"] = tables
            result["ok"] = True
            Path(job["result"]).write_text(json.dumps(result))
            return 0
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
    _require_serum()
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
                                  encoding="utf-8", errors="replace", env=_child_env(),
                                  creationflags=config.no_window_flags(),
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


def _child_env() -> dict:
    """Environment for the helper process: UTF-8 pipes on every platform."""
    import os
    return {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUTF8": "1"}


def _require_serum() -> None:
    import os
    if not config.serum_plugin_paths() and not os.environ.get("FA_FAKE_SERUM"):
        raise SerumHostError("Serum 2 plugin not found (looked for Serum2.vst3 / Serum2.component)")


class SerumSession:
    """A long-lived helper process with Serum 2 loaded, for many renders in a
    row (tuning a patch inside real Serum). Use as a context manager."""

    def __init__(self, sr: int = 44100, start_timeout: float = 180.0):
        import queue
        import subprocess
        import sys
        import tempfile
        import threading
        _require_serum()
        self.sr = sr
        self.dir = Path(tempfile.mkdtemp(prefix="fa-serum-session-"))
        self.err = open(self.dir / "stderr.log", "w")
        self.proc = subprocess.Popen(
            [sys.executable, "-m", "fourier_analyzer", "serum-host", "serve"], cwd=str(config.code_dir()),
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self.err, text=True, bufsize=1,
            encoding="utf-8", errors="replace", env=_child_env(), creationflags=config.no_window_flags())
        self.replies: queue.Queue = queue.Queue()

        def read():
            import json
            for line in self.proc.stdout:
                if line.startswith(SESSION_TAG):
                    try:
                        self.replies.put(json.loads(line[len(SESSION_TAG):]))
                    except ValueError:
                        pass
            self.replies.put(None)                 # process ended
        threading.Thread(target=read, daemon=True).start()
        first = self._reply(start_timeout)
        if not first.get("ok"):
            self.close()
            raise SerumHostError(first.get("error", "Serum helper failed to start"))
        self.plugin = first.get("plugin", "")
        self.n = 0

    def _reply(self, timeout: float) -> dict:
        import queue
        try:
            r = self.replies.get(timeout=timeout)
        except queue.Empty:
            self.close()
            raise SerumHostError("Serum 2 stopped responding (helper process timed out)")
        if r is None:
            tail = ""
            try:
                self.err.flush()
                tail = (self.dir / "stderr.log").read_text()[-1500:]
            except OSError:
                pass
            raise SerumHostError(f"Serum helper process crashed (exit {self.proc.poll()}). {tail}")
        return r

    def render(self, preset: bytes, notes, duration: float, timeout: float = 90.0) -> np.ndarray:
        import json

        import soundfile as sf
        self.n += 1
        pp, out = self.dir / f"p{self.n % 2}.SerumPreset", self.dir / "r.wav"
        pp.write_bytes(preset)
        req = {"preset": str(pp), "duration": float(duration), "out": str(out), "sr": self.sr,
               "notes": [n.to_dict() if hasattr(n, "to_dict") else n for n in notes]}
        self.proc.stdin.write(json.dumps(req) + "\n")
        self.proc.stdin.flush()
        r = self._reply(timeout)
        if not r.get("ok"):
            raise SerumHostError(r.get("error", "render failed"))
        return sf.read(out, dtype="float32")[0]

    def close(self) -> None:
        import shutil
        try:
            if self.proc.poll() is None:
                self.proc.stdin.write('{"quit": true}\n')
                self.proc.stdin.flush()
                self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        self.err.close()
        shutil.rmtree(self.dir, ignore_errors=True)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def verify(preset: bytes, notes, y_target: np.ndarray, sr: int, gain=None) -> dict:
    """Render the preset in real Serum and score it. Never raises."""
    try:
        (y,), plugin = render_jobs([(preset, notes, len(y_target) / sr)], sr=sr)
        if not np.any(np.abs(y) > 1e-5):
            return {"ok": False, "error": "Serum rendered silence (preset not accepted?)"}
        from ..match.metrics import Reference
        raw = y
        if gain is not None:
            y = y[: len(gain)] * gain[: len(y)]
        d = Reference(np.asarray(y_target, dtype=np.float64), sr).all(y)
        return {"ok": True, "score": d["waveform_similarity"], "details": d, "audio": y, "raw_audio": raw,
                "plugin": plugin}
    except Exception as e:
        log.warning("Serum verification unavailable: %s", e)
        return {"ok": False, "error": str(e)}


CALIBRATE_PARAMS = ("filter_1_freq_hz", "env_1_attack", "env_1_decay", "env_1_release", "env_1_sustain",
                    "lfo_1_rate", "a_uni_detune", "a_wt_pos", "a_level")


def query_params(names, steps: int = 41) -> tuple[dict, str]:
    """Sweep plugin parameters 0..1 in the helper process and return
    {name: [[raw, display text, parsed SI value], ...]}."""
    import json
    import subprocess
    import sys
    import tempfile
    if not config.serum_plugin_paths():
        raise SerumHostError("Serum 2 plugin not found (looked for Serum2.vst3 / Serum2.component)")
    with tempfile.TemporaryDirectory(prefix="fa-serum-") as td:
        td = Path(td)
        job = {"mode": "params", "params": list(names), "steps": steps, "result": str(td / "result.json")}
        (td / "job.json").write_text(json.dumps(job))
        proc = subprocess.run([sys.executable, "-m", "fourier_analyzer", "serum-host", str(td / "job.json")],
                              cwd=str(config.code_dir()), capture_output=True, text=True,
                                  encoding="utf-8", errors="replace", env=_child_env(),
                                  creationflags=config.no_window_flags(), timeout=120)
        if not (td / "result.json").exists():
            raise SerumHostError(f"Serum helper process crashed (exit {proc.returncode}). "
                                 + (proc.stderr or proc.stdout or "")[-1500:])
        res = json.loads((td / "result.json").read_text())
        if not res.get("ok"):
            raise SerumHostError(res.get("error", "unknown error"))
        return res["tables"], res.get("plugin", "")


def calibrate(progress=None) -> dict:
    """Read Serum's own value displays while sweeping its parameters: this
    gives the exact normalised->Hz cutoff curve (and env/LFO/detune tables)
    without rendering audio or loading presets."""
    if progress:
        progress("reading Serum 2 parameter curves (helper process)")
    tables, plugin = query_params(CALIBRATE_PARAMS)
    if progress:
        progress(f"loaded {plugin}; got {len(tables)} parameter curves")
    cut = [(x, v) for x, _, v in tables.get("filter_1_freq_hz", []) if v and v > 0]
    if len(cut) < 5:
        sample = tables.get("filter_1_freq_hz", [])[:5]
        raise SerumHostError(f"could not read Serum's cutoff values (sample: {sample})")
    s = config.settings().get("serum_scales", {})
    s["cutoff_table"] = [[round(x, 5), round(v, 3)] for x, v in cut]
    s["cutoff_lo_hz"], s["cutoff_hi_hz"] = float(cut[0][1]), float(cut[-1][1])
    config.save_settings(serum_scales=s, serum_param_tables=tables)
    if progress:
        for x, v in cut[::8]:
            progress(f"cutoff {x:.2f} -> {v:.1f} Hz")
    return {"plugin": plugin, "cutoff_hz_at_0": cut[0][1], "cutoff_hz_at_0.5": dict(cut).get(0.5),
            "cutoff_hz_at_1": cut[-1][1], "params_read": sorted(tables)}


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
