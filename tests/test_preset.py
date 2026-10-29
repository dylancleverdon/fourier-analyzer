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


def test_full_feature_roundtrip():
    from fourier_analyzer.synth.engine import Renderer
    from fourier_analyzer.synth.patch import Note, fx
    bank = TableBank()
    p = Patch(bpm=140)
    p.oscs[0] = Osc(enabled=True, table="basic:saw", unison=5, unison_stack="Octave1", warp="Sync",
                    warp_amount=0.4, warp2="DistSoft", warp2_amount=0.3)
    p.oscs[1] = Osc(enabled=True, table="basic:sine", level=0.0)
    p.filter = Filter(enabled=True, type="Ladder24", cutoff=900, reso=0.3, var=0.4)
    p.filter2 = Filter(enabled=True, type="Comb+", cutoff=500, mix=0.6)
    p.routing = {"osc1": {"balance": 0.5, "send1": 0.4}}
    p.lfos[5] = LFO(shape="sine", sync="1/4", type="random")
    p.mods = [Mod("lfo6", "filter2.cutoff", 0.3), Mod("env3", "osc1.warp", 0.5), Mod("macro2", "fx.1.mix", 0.4)]
    p.fx_chain = [fx("split3", bands=[[fx("distortion", mode="Tube")], [], [fx("delay")]]),
                  fx("reverb", kind="Plate", mix=0.3), fx("bode", shift_hz=40.0)]
    p.fx_buses = [[fx("hyperdim")], []]
    h, st = read_preset_bytes(build_preset_bytes(p, bank, name="full"))
    q = state_to_patch(st, bank)
    o = q.oscs[0]
    assert (o.warp, o.warp2, o.unison_stack) == ("Sync", "DistSoft", "Octave1")
    assert q.filter.type == "Ladder24" and q.filter2.enabled and q.filter2.type == "Comb+"
    assert [m["type"] for m in q.fx_chain] == ["split3", "reverb", "bode"]
    assert q.fx_chain[0]["bands"][0][0]["type"] == "distortion"
    assert q.fx_chain[0]["bands"][2][0]["type"] == "delay"
    assert q.fx_buses[0] and q.fx_buses[0][0]["type"] == "hyperdim"
    assert {(m.source, m.dest) for m in q.mods} == {("lfo6", "filter2.cutoff"), ("env3", "osc1.warp"),
                                                     ("macro2", "fx.1.mix")}
    y = Renderer(bank, 22050).render(q, [Note(0, 0.4, 36), Note(0.5, 0.4, 43)], 1.2)
    assert np.all(np.isfinite(y)) and np.abs(y).max() > 1e-3
