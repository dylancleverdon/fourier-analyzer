import sys
import types
from pathlib import Path

import pytest

from fourier_analyzer.serum import host


def _fake_pedalboard(monkeypatch, behaviour):
    mod = types.ModuleType("pedalboard")
    mod.load_plugin = behaviour
    monkeypatch.setitem(sys.modules, "pedalboard", mod)


def test_picks_instrument_not_fx(monkeypatch):
    calls = []

    def load(path, plugin_name=None):
        calls.append(plugin_name)
        if plugin_name == "Serum 2":
            raise ValueError('contains 2 plugins: "Serum 2 Instrument" "Serum 2 FX"')
        if plugin_name == "Serum 2 Instrument":
            return "PLUGIN"
        raise AssertionError(f"should not try {plugin_name}")
    _fake_pedalboard(monkeypatch, load)
    plug, desc = host._open_plugin([Path("/x/Serum2.component")])
    assert plug == "PLUGIN" and "Serum 2 FX" not in calls


def test_reports_real_errors(monkeypatch):
    def load(path, plugin_name=None):
        raise RuntimeError("must be loaded on the main thread")
    _fake_pedalboard(monkeypatch, load)
    with pytest.raises(host.SerumHostError) as e:
        host._open_plugin([Path("/x/Serum2.vst3"), Path("/x/Serum2.component")])
    msg = str(e.value)
    assert "main thread" in msg and "Serum2.vst3" in msg and "Serum2.component" in msg


def test_swaps_bare_zstd_state_and_fixes_lengths():
    import struct

    import cbor2
    import zstandard

    from fourier_analyzer.serum.mapping import load_skeleton, patch_to_state
    from fourier_analyzer.serum.preset import encode_state, write_preset_bytes
    from fourier_analyzer.synth.patch import Filter, Patch
    init = encode_state(load_skeleton())
    comp = zstandard.ZstdCompressor(level=19).compress(init)
    blob = struct.pack("<II", len(init), 2) + comp          # Serum-like: size, version, zstd(CBOR)
    p = Patch()
    p.filter = Filter(enabled=True, cutoff=1234)
    new = host._swap_in_state(blob, write_preset_bytes(patch_to_state(p), "t"))
    assert new is not None
    k = new.find(host.ZSTD_MAGIC)
    raw = zstandard.ZstdDecompressor().decompressobj().decompress(new[k:])
    assert cbor2.loads(raw)["VoiceFilter0"]["plainParams"]["kParamEnable"] == 1.0
    assert struct.unpack_from("<I", new, 0)[0] == len(raw)


def test_juce_base64_roundtrip():
    import os
    for n in (0, 1, 2, 3, 5, 64, 1001):
        data = os.urandom(n)
        enc = host.juce_b64_encode(data)
        assert enc.startswith(f"{n}.")
        assert host.juce_b64_decode(enc) == data


def test_swaps_inside_juce_vc2_xml_state():
    import struct

    import cbor2
    import zstandard

    from fourier_analyzer.serum.mapping import load_skeleton, patch_to_state
    from fourier_analyzer.serum.preset import encode_state, write_preset_bytes
    from fourier_analyzer.synth.patch import Filter, Patch
    init = encode_state(load_skeleton())
    comp = zstandard.ZstdCompressor(level=19).compress(init)
    component = struct.pack("<II", len(init), 2) + comp
    xml = ('<?xml version="1.0" encoding="UTF-8"?>\n\n<VST3PluginState><IComponent>'
           + host.juce_b64_encode(component) + "</IComponent><IEditController>"
           + host.juce_b64_encode(b"ctrl") + "</IEditController></VST3PluginState>")
    blob = b"VC2!" + struct.pack("<I", len(xml)) + xml.encode() + b"\0"
    assert "VST3PluginState" in host.describe_blob(blob)
    p = Patch()
    p.filter = Filter(enabled=True, cutoff=1234)
    new = host._swap_in_state(blob, write_preset_bytes(patch_to_state(p), "t"))
    assert new is not None and new[:4] == b"VC2!"
    (n,) = struct.unpack_from("<I", new, 4)
    assert len(new) == n + 9
    new_xml = new[8:8 + n].decode()
    comp_text = new_xml.split("<IComponent>")[1].split("</IComponent>")[0]
    inner = host.juce_b64_decode(comp_text)
    k = inner.find(host.ZSTD_MAGIC)
    st = cbor2.loads(zstandard.ZstdDecompressor().decompressobj().decompress(inner[k:]))
    assert st["VoiceFilter0"]["plainParams"]["kParamEnable"] == 1.0
    assert host.juce_b64_encode(b"ctrl") in new_xml          # controller state untouched
