"""Read and write Serum 2 ``.SerumPreset`` files.

File layout (reverse engineered from factory presets, Serum 2.0.x):

    b"XferJson\\0"                      9-byte magic
    uint64 LE  json_len
    json_len bytes of UTF-8 JSON        header (name, author, tags, hash, ...)
    uint32 LE  uncompressed_size        size of the CBOR payload
    uint32 LE  2                        payload format version
    zstd frame                          compressed CBOR document

``header["hash"]`` is the MD5 hex digest of the zstd frame.

Serum writes most numbers as float32 CBOR values. cbor2 always emits
float64, which is valid CBOR but doubles the size of embedded wavetables, so
the encoder below writes floats that survive a float32 round trip as float32.
"""
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from typing import Any

import cbor2
import numpy as np
import zstandard

MAGIC = b"XferJson\x00"
PAYLOAD_VERSION = 2

HEADER_DEFAULTS = {
    "fileType": "SerumPreset",
    "product": "Serum2",
    "productVersion": "2.0.11",
    "url": "https://xferrecords.com/",
    "vendor": "Xfer Records",
    "version": 4.0,
}


class PresetFormatError(ValueError):
    pass


def read_preset_bytes(data: bytes) -> tuple[dict, dict]:
    """Return ``(header, state)`` for raw preset bytes."""
    if not data.startswith(MAGIC):
        raise PresetFormatError("not a Serum 2 preset (missing XferJson magic)")
    pos = len(MAGIC)
    (json_len,) = struct.unpack_from("<Q", data, pos)
    pos += 8
    header = json.loads(data[pos : pos + json_len].decode("utf-8"))
    pos += json_len
    usize, _ver = struct.unpack_from("<II", data, pos)
    pos += 8
    payload = zstandard.ZstdDecompressor().decompress(data[pos:], max_output_size=usize)
    state = cbor2.loads(payload)
    return header, state


def read_preset(path: str | Path) -> tuple[dict, dict]:
    return read_preset_bytes(Path(path).read_bytes())


def _is_f32(x: float) -> bool:
    return bool(np.isfinite(x)) and float(np.float32(x)) == x


def _encode(obj: Any, out: bytearray) -> None:
    """Minimal CBOR encoder matching Serum's float32-first style."""
    if isinstance(obj, bool):
        out.append(0xF5 if obj else 0xF4)
    elif obj is None:
        out.append(0xF6)
    elif isinstance(obj, (int, np.integer)) :
        _head(out, 0, int(obj)) if obj >= 0 else _head(out, 1, -1 - int(obj))
    elif isinstance(obj, (float, np.floating)):
        f = float(obj)
        if _is_f32(f):
            out.append(0xFA)
            out += struct.pack(">f", f)
        else:
            out.append(0xFB)
            out += struct.pack(">d", f)
    elif isinstance(obj, str):
        b = obj.encode("utf-8")
        _head(out, 3, len(b))
        out += b
    elif isinstance(obj, (bytes, bytearray)):
        _head(out, 2, len(obj))
        out += obj
    elif isinstance(obj, np.ndarray):
        _encode_float_array(obj, out)
    elif isinstance(obj, (list, tuple)):
        _head(out, 4, len(obj))
        for v in obj:
            _encode(v, out)
    elif isinstance(obj, dict):
        _head(out, 5, len(obj))
        for k, v in obj.items():
            _encode(k, out)
            _encode(v, out)
    else:
        raise TypeError(f"cannot encode {type(obj).__name__}")


def _encode_float_array(arr: np.ndarray, out: bytearray) -> None:
    a = np.asarray(arr, dtype=">f4").ravel()
    _head(out, 4, a.size)
    # each element: 0xFA + 4 bytes big-endian float32
    buf = np.empty((a.size, 5), dtype=np.uint8)
    buf[:, 0] = 0xFA
    buf[:, 1:] = a.view(np.uint8).reshape(-1, 4)
    out += buf.tobytes()


def _head(out: bytearray, major: int, n: int) -> None:
    m = major << 5
    if n < 24:
        out.append(m | n)
    elif n < 0x100:
        out += bytes((m | 24, n))
    elif n < 0x10000:
        out.append(m | 25)
        out += struct.pack(">H", n)
    elif n < 0x100000000:
        out.append(m | 26)
        out += struct.pack(">I", n)
    else:
        out.append(m | 27)
        out += struct.pack(">Q", n)


def encode_state(state: dict) -> bytes:
    out = bytearray()
    _encode(state, out)
    return bytes(out)


def write_preset_bytes(state: dict, name: str, author: str = "Fourier Analyzer",
                       description: str = "", tags: list[str] | None = None) -> bytes:
    payload = encode_state(state)
    comp = zstandard.ZstdCompressor(level=3).compress(payload)
    header = dict(HEADER_DEFAULTS)
    header.update({
        "hash": hashlib.md5(comp).hexdigest(),
        "presetAuthor": author,
        "presetDescription": description,
        "presetName": name,
        "tags": tags or ["Wavetable", "Bass"],
    })
    hjson = json.dumps(dict(sorted(header.items())), separators=(",", ":")).encode("utf-8")
    return (MAGIC + struct.pack("<Q", len(hjson)) + hjson
            + struct.pack("<II", len(payload), PAYLOAD_VERSION) + comp)


def write_preset(path: str | Path, state: dict, name: str, **kw) -> Path:
    path = Path(path)
    path.write_bytes(write_preset_bytes(state, name, **kw))
    return path
