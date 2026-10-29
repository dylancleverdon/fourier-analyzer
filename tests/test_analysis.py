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


def _pitch_case(mods=None, lfo=None, env3=None, glide=0.0, notes=None):
    from fourier_analyzer.synth.engine import Renderer
    from fourier_analyzer.synth.patch import Env, LFO, Mod, Note, Osc, Patch
    sr = 22050
    p = Patch(bpm=140)
    p.oscs[0] = Osc(enabled=True, table="basic:saw", random_phase=0)
    if lfo:
        p.lfos[0] = lfo
    if env3:
        p.envs[2] = env3
    p.mods = mods or []
    p.glide, p.glide_always = glide, False
    notes = notes or [Note(0, 0.9, 45), Note(1.0, 0.9, 48), Note(2.0, 0.9, 43)]
    return analyze(Renderer(sr=sr).render(p, notes, 3.0), sr, bpm=140)


def test_vibrato_becomes_lfo_not_midi_bends():
    from fourier_analyzer.synth.patch import LFO, Mod
    a = _pitch_case([Mod("lfo1", "pitch", 1.0 / 48, bipolar=True)], lfo=LFO(shape="sine", sync="1/8", mode="trig"))
    assert a.pitch_lfo and a.pitch_lfo["sync"] == "1/8"
    assert 0.4 < a.pitch_lfo["semis"] < 1.3
    assert not any(n.bend for n in a.notes)
    assert [round(n.start, 2) for n in a.notes] == [0.0, 1.0, 2.0]


def test_pitch_drop_and_glide_go_into_the_patch():
    from fourier_analyzer.synth.patch import Env, Mod, Note
    a = _pitch_case([Mod("env3", "pitch", 7 / 48)], env3=Env(attack=0.0005, decay=0.15, sustain=0, release=0.05))
    assert a.pitch_env and a.pitch_env["semis"] > 3
    assert not any(n.bend for n in a.notes)
    g = _pitch_case(glide=0.12, notes=[Note(0, 1.05, 45), Note(1.0, 1.05, 52), Note(2.0, 0.9, 40)])
    assert 0.05 < g.glide < 0.4
    assert not any(n.bend for n in g.notes)
    assert [round(n.pitch) for n in g.notes] == [45, 52, 40]


def test_one_off_bend_stays_in_midi_and_plain_notes_stay_plain():
    from fourier_analyzer.synth.patch import Note
    a = _pitch_case(notes=[Note(0, 0.9, 45), Note(1.0, 0.9, 48, bend=[[0, 0], [0.3, 0], [0.6, 2.0], [0.89, 2.0]]),
                           Note(2.0, 0.9, 43)])
    assert a.notes[1].bend and not a.pitch_lfo and not a.pitch_env
    plain = _pitch_case()
    assert plain.pitch_lfo is None and plain.pitch_env is None and plain.glide == 0
    assert not any(n.bend for n in plain.notes)


def test_legato_midi_bends_do_not_reset_the_next_note(tmp_path):
    import mido
    from fourier_analyzer.export import write_midi
    from fourier_analyzer.synth.patch import Note
    notes = [Note(0.0, 0.6, 45, bend=[[0, 0], [0.5, 1.0]]), Note(0.5, 0.5, 48, bend=[[0, 2.0], [0.4, 2.0]])]
    mid = mido.MidiFile(str(write_midi(tmp_path / "x.mid", notes, 120)))
    t, last_bend = 0, None
    for msg in mid.tracks[0]:
        t += msg.time
        if msg.type == "pitchwheel":
            last_bend = (t, msg.pitch)
    # the final bend in the file is note 2's +2 st, not a reset from note 1's end
    assert last_bend[1] == 0 or last_bend[0] >= int(0.95 * 960 * 2)
    events = [(m.type, getattr(m, "pitch", None)) for m in mid.tracks[0] if m.type == "pitchwheel"]
    assert any(p and p > 1000 for _, p in events)
