import numpy as np

from fourier_analyzer.serum.mapping import build_preset_bytes, load_skeleton, patch_to_state, state_to_patch
from fourier_analyzer.serum.preset import encode_state, read_preset_bytes, write_preset_bytes
from fourier_analyzer.synth.patch import LFO, Distortion, Filter, Mod, NoiseOsc, Osc, Patch
from fourier_analyzer.synth.wavetables import TableBank


def test_roundtrip_skeleton():
    st = load_skeleton()
    h, st2 = read_preset_bytes(write_preset_bytes(st, "Init"))
    assert st2 == st
    assert h["presetName"] == "Init" and h["product"] == "Serum2"


def test_float32_encoding_is_compact():
    a = encode_state({"x": 0.5})
    assert b"\xfa" in a          # float32 marker
    b = encode_state({"x": 0.1})  # not representable as float32 -> float64
    assert b"\xfb" in b


def test_hash_is_md5_of_compressed_payload():
    import hashlib
    import json
    import struct
    data = write_preset_bytes({"a": 1}, "h")
    n = struct.unpack_from("<Q", data, 9)[0]
    hdr = json.loads(data[17:17 + n])
    comp = data[17 + n + 8:]
    assert hdr["hash"] == hashlib.md5(comp).hexdigest()


def test_patch_to_preset_fields():
    bank = TableBank()
    tid = bank.add_custom(np.zeros((3, 2048), np.float32))
    p = Patch(bpm=150)
    p.oscs[0] = Osc(enabled=True, table=tid, unison=5, detune=0.2, position=0.5)
    p.oscs[1] = Osc(enabled=True, table="serum:S2 Tables/Analog/Saw.wav", octave=-1, level=0.4)
    p.filter = Filter(enabled=True, type="lp12", cutoff=1000, reso=0.4)
    p.noise = NoiseOsc(enabled=True, level=0.3)
    p.lfos[0] = LFO(shape="sine", sync="1/8")
    p.mods = [Mod("lfo1", "filter.cutoff", 0.3), Mod("env2", "pitch", 0.25)]
    p.distortion = Distortion(enabled=True, mode="hard", drive=0.5)
    h, st = read_preset_bytes(build_preset_bytes(p, bank, name="T"))
    w0 = st["Oscillator0"]["WTOsc0"]
    assert len(w0["embeddedWTData"]) == 3 * 2048
    assert st["Oscillator0"]["plainParams"]["kParamUnison"] == 5.0
    assert st["Oscillator1"]["WTOsc1"]["relativePathToWT"] == "S2 Tables/Analog/Saw.wav"
    assert st["VoiceFilter0"]["plainParams"]["kParamType"] == "L12"
    assert st["ModSlot0"]["destModuleTypeString"] == "VoiceFilter"
    assert st["ModSlot0"]["source"] == [6, 0]
    assert abs(st["LFO0"]["plainParams"]["kParamRate"] - 150 / 60 * 2) < 1e-6
    assert st["FXRack0"]["FX"][0]["FXDistortion"]["plainParams"]["kParamMode"] == "kHardClip"
    back = state_to_patch(st)
    assert abs(back.filter.cutoff - 1000) < 1


def test_state_contains_all_skeleton_modules():
    st = patch_to_state(Patch())
    assert set(load_skeleton()) <= set(st)
