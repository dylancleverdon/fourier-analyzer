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


def test_controller_dropped_and_ignored_state_reports_diagnostics(monkeypatch, tmp_path):
    import struct

    import zstandard

    from fourier_analyzer import config
    from fourier_analyzer.serum.mapping import load_skeleton
    from fourier_analyzer.serum.preset import encode_state, write_preset_bytes
    init = encode_state(load_skeleton())
    comp = struct.pack("<II", len(init), 2) + zstandard.ZstdCompressor().compress(init)
    xml = ("<VST3PluginState><IComponent>" + host.juce_b64_encode(comp) + "</IComponent><IEditController>"
           + host.juce_b64_encode(b"ctrl") + "</IEditController></VST3PluginState>")
    base = b"VC2!" + struct.pack("<I", len(xml)) + xml.encode() + b"\0"
    preset = write_preset_bytes(load_skeleton(), "t")
    assert b"IEditController" not in host._swap_vc2(base, preset, drop_controller=True)

    class Stubborn:                      # a plugin that never changes its state
        @property
        def raw_state(self):
            return base

        @raw_state.setter
        def raw_state(self, value):
            pass
        parameters = {"A Level": 0, "Fil Cutoff": 0}

        def __call__(self, *a, **k):
            return None
    h = host.SerumHost.__new__(host.SerumHost)
    h.plugin, h.base_state, h.sr, h.description = Stubborn(), base, 44100, "stub"
    monkeypatch.setattr(config, "app_dir", lambda: tmp_path)
    monkeypatch.setattr("time.sleep", lambda s: None)
    import pytest
    with pytest.raises(host.SerumHostError) as e:
        h.load_preset(preset)
    assert "serum_diagnostics.txt" in str(e.value)
    text = (tmp_path / "serum_diagnostics.txt").read_text()
    assert "Fil Cutoff" in text and "drop_controller=True" in text


def test_parse_values_and_cutoff_table(monkeypatch):
    assert host._parse_value("1.25 kHz") == 1250.0
    assert host._parse_value("440 Hz") == 440.0
    assert abs(host._parse_value("12.0 ms") - 0.012) < 1e-9
    from fourier_analyzer.serum import mapping
    sc = dict(mapping.SCALES, cutoff_table=[[0.0, 20.0], [0.25, 100.0], [0.5, 1000.0], [0.75, 5000.0], [1.0, 20000.0]])
    assert abs(mapping.cutoff_to_norm(1000.0, sc) - 0.5) < 1e-6


def test_merge_keeps_serum_header_and_new_keys():
    import json
    import struct

    import cbor2
    import zstandard

    from fourier_analyzer.serum.mapping import load_skeleton, patch_to_state
    from fourier_analyzer.serum.preset import MAGIC, encode_state, write_preset_bytes
    from fourier_analyzer.synth.patch import Filter, Patch
    base_state = dict(load_skeleton(), NewV11Module={"plainParams": "default"})
    payload = encode_state(base_state)
    comp = zstandard.ZstdCompressor().compress(payload)
    hdr = json.dumps({"component": "processor", "productVersion": "2.1.5", "version": 11.0}).encode()
    old = MAGIC + struct.pack("<Q", len(hdr)) + hdr + struct.pack("<II", len(payload), 2) + comp
    p = Patch()
    p.filter = Filter(enabled=True, cutoff=1234)
    new = host._rebuild_container(old, write_preset_bytes(patch_to_state(p), "t"))
    (jl,) = struct.unpack_from("<Q", new, len(MAGIC))
    h = json.loads(new[len(MAGIC) + 8:len(MAGIC) + 8 + jl])
    assert h["component"] == "processor" and h["version"] == 11.0
    st = cbor2.loads(zstandard.ZstdDecompressor().decompressobj().decompress(new[len(MAGIC) + 8 + jl + 8:]))
    assert "NewV11Module" in st and st["VoiceFilter0"]["plainParams"]["kParamEnable"] == 1.0
