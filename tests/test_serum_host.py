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
