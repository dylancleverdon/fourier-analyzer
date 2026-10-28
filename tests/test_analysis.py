from fourier_analyzer.analysis.features import analyze
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import LFO, Env, Filter, Mod, Note, Osc, Patch


def test_notes_pitch_timing_and_lfo():
    p = Patch(bpm=140)
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, type="lp24", cutoff=300, reso=0.3)
    p.lfos[0] = LFO(shape="sine", sync="1/8")
    p.mods = [Mod("lfo1", "filter.cutoff", 0.35)]
    p.envs[0] = Env(0.003, 0, 0.2, 0.8, 0.05)
    notes = [Note(0.0, 0.84, 38), Note(0.857, 0.4, 41), Note(1.286, 1.2, 36)]
    r = Renderer()
    a = analyze(r.render(p, notes, 2.8), r.sr, bpm=140)
    assert len(a.notes) == 3
    for got, want in zip(a.notes, notes):
        assert abs(got.pitch - want.pitch) < 0.3
        assert abs(got.start - want.start) < 0.02
    assert a.lfo is not None and a.lfo["sync"] == "1/8"


def test_percussive_detection():
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:sine")
    p.envs[0] = Env(0.001, 0, 0.12, 0.0, 0.05)
    p.envs[1] = Env(0.0005, 0, 0.04, 0.0, 0.05)
    p.mods = [Mod("env2", "pitch", 0.5)]
    r = Renderer()
    notes = [Note(t, 0.2, 33) for t in (0.0, 0.43, 0.86)]
    a = analyze(r.render(p, notes, 1.3), r.sr, bpm=140)
    assert a.percussive
