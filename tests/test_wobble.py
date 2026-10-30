"""LFO wobble stays out of embedded samples (sample / granular / spectral)."""
import numpy as np

from fourier_analyzer.analysis.features import analyze
from fourier_analyzer.match import wobble
from fourier_analyzer.match.optimizer import OPT_SR, Optimizer, Options
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import LFO, Filter, Mod, Note, Osc, Patch, SampleCfg

BPM = 140


def _wobble_patch():
    p = Patch(bpm=BPM)
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, type="lp24", cutoff=500, reso=0.3)
    p.lfos[0] = LFO(shape="sine", sync="1/8", mode="trig")
    p.mods = [Mod("lfo1", "filter.cutoff", 0.6), Mod("lfo1", "amp", -0.4)]
    return p


def _opt():
    p = _wobble_patch()
    notes = [Note(0, 1.6, 36), Note(1.7, 1.6, 43), Note(3.4, 1.6, 31)]
    y44 = Renderer(sr=44100).render(p, notes, 5.1)
    y22 = Renderer(sr=OPT_SR).render(p, notes, 5.1)
    a = analyze(y22, OPT_SR, bpm=BPM)
    assert a.lfo and a.lfo["sync"] == "1/8"
    return Optimizer(y22, OPT_SR, a, Options(time_budget=5), source_audio=(y44, 44100)), y44


def test_flattening_takes_the_wobble_out_of_a_note():
    y = Renderer(sr=44100).render(_wobble_patch(), [Note(0, 1.6, 36)], 1.6)
    f0 = 440 * 2 ** ((36 - 69) / 12)
    before = wobble.modulation(y, 44100)["depth_db"]
    after = wobble.modulation(wobble.flatten_movement(y, 44100, f0), 44100)["depth_db"]
    assert before > 15 and after < 0.3 * before, (before, after)


def test_a_baked_wobble_is_caught_at_transposed_pitches():
    opt, y44 = _opt()
    opt.source_sample()
    n = max(opt.notes, key=lambda n: n.duration)
    raw = y44[int(n.start * 44100): int((n.start + n.duration) * 44100)]
    q = opt.initial_patch()
    sid = opt.samples.add(raw.astype(np.float32), 44100, root=n.pitch)
    q.oscs[0] = Osc(enabled=True, type="sample", level=0.8)
    q.oscs[0].smp = SampleCfg(source=sid, root=n.pitch, loop="forward", loop_start=0.1, loop_end=0.95)
    chk = opt.wobble_check(q)
    assert not chk["ok"] and "follows the pitch" in chk["why"]
    rates = [r["rate_hz"] for r in chk["pitches"]]
    assert rates[0] < rates[1] < rates[2]          # lower note, slower baked wobble
    # the real thing (wobble in an LFO routing) passes
    assert opt.wobble_check(_wobble_patch())["ok"]


def test_sample_based_candidates_keep_the_wobble_in_the_lfo():
    opt, _ = _opt()
    moves = []
    opt._sample_moves(opt.initial_patch(), lambda name, fn, focus: moves.append((name, fn)))
    names = [m[0] for m in moves]
    assert names and all("synced LFO wobble" in n for n in names)
    assert names[0].startswith("osc A as granular")              # granular / spectral preferred
    assert all(opt.move_penalty.get(n, 0) > 0 for n in names if "sample" in n.split("(")[0])
    smp = opt.samples.get(opt._source_sid)
    assert "flattened" in smp.name
    period = 1.0 / opt.analysis.lfo["rate_hz"]
    for name, fn in moves:
        q = fn(opt.initial_patch())
        o = next(o for o in q.oscs if o.enabled and o.type != "wavetable")
        assert any(m.source.startswith("lfo") and m.dest in ("filter.cutoff", "amp") for m in q.mods)
        if o.type == "sample":
            loop_s = (o.smp.loop_end - o.smp.loop_start) * len(smp.data[0]) / smp.sr
            assert loop_s < 0.5 * period, loop_s
        else:
            assert o.smp.scan_tempo_lock and o.smp.base_tempo == BPM
        assert opt.wobble_check(q)["ok"], name
    assert not opt.wobble_rejects


def test_tempo_lock_ties_the_scan_to_the_song_tempo():
    from fourier_analyzer.synth.sampler import tempo_factor
    cfg = SampleCfg(scan_tempo_lock=True, base_tempo=140.0)
    assert tempo_factor(cfg, 70.0) == 0.5 and tempo_factor(SampleCfg(), 70.0) == 1.0
    # and it survives the preset
    from fourier_analyzer.serum.mapping import build_preset_bytes, state_to_patch
    from fourier_analyzer.serum.preset import read_preset_bytes
    from fourier_analyzer.synth.samples import SampleBank
    from fourier_analyzer.synth.wavetables import TableBank
    bank, smp = TableBank(), SampleBank()
    sid = smp.add(np.sin(np.arange(22050) * 0.05).astype(np.float32), 44100, root=60)
    p = Patch(bpm=BPM)
    p.oscs[0] = Osc(enabled=True, type="granular")
    p.oscs[0].smp = SampleCfg(source=sid, root=60, scan_tempo_lock=True, base_tempo=BPM, loop="forward")
    _, st = read_preset_bytes(build_preset_bytes(p, bank, samples=smp))
    r = state_to_patch(st, TableBank(), samples=SampleBank())
    assert r.oscs[0].smp.scan_tempo_lock and abs(r.oscs[0].smp.base_tempo - BPM) < 1e-6
