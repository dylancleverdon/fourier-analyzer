from fourier_analyzer.analysis.features import analyze
from fourier_analyzer.match.optimizer import Options, Optimizer
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import Env, Filter, Note, Osc, Patch


def test_reconstructs_simple_saw_bass():
    p = Patch(bpm=140)
    p.oscs[0] = Osc(enabled=True, table="basic:saw", random_phase=0.0)
    p.filter = Filter(enabled=True, cutoff=600, reso=0.2)
    p.envs[0] = Env(0.003, 0, 0.3, 0.7, 0.05)
    notes = [Note(0, 0.4, 36), Note(0.43, 0.4, 39), Note(0.86, 0.8, 34)]
    r = Renderer()
    y = r.render(p, notes, 1.8)
    a = analyze(y, r.sr, bpm=140)
    res = Optimizer(y, r.sr, a, Options(time_budget=30, target_score=90)).run()
    assert res.details["spectral_similarity"] > 95, res.details
    assert res.score > 85, res.details          # literal waveform correlation
    assert res.patch.oscs[0].table.startswith("custom:")


def test_serum_tuning_matches_the_real_plugin(monkeypatch):
    """Tuning inside 'Serum' (our engine with cutoffs scaled by 0.7 standing in
    for a plugin that differs from our approximation) runs through the helper
    process and moves the cutoff towards what that plugin needs."""
    from fourier_analyzer.match.serum_tune import tune_in_serum
    from fourier_analyzer.synth.engine import Renderer
    from fourier_analyzer.synth.patch import Filter, Note, Osc, Patch
    from fourier_analyzer.synth.wavetables import TableBank
    monkeypatch.setenv("FA_FAKE_SERUM", "cutoff:0.7")
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, cutoff=600, reso=0.2)
    notes = [Note(0, 0.4, 36), Note(0.5, 0.4, 40)]
    y = Renderer(sr=44100).render(p, notes, 1.0)
    q, info = tune_in_serum(p, TableBank(), notes, y, 44100, budget_s=20)
    assert info["ok"] and info["evals"] > 50
    assert info["after"] > info["before"]
    assert q.filter.cutoff > 700


def test_sample_oscillator_from_the_original_survives_the_preset():
    """The search's sample / granular / spectral moves embed a note of the
    original; the exported preset carries that audio and renders the same."""
    import numpy as np
    from fourier_analyzer.analysis.features import analyze
    from fourier_analyzer.match.optimizer import OPT_SR, Optimizer, Options
    from fourier_analyzer.serum.mapping import build_preset_bytes, state_to_patch
    from fourier_analyzer.serum.preset import read_preset_bytes
    from fourier_analyzer.synth.engine import Renderer
    from fourier_analyzer.synth.patch import Note, Osc, Patch
    from fourier_analyzer.synth.samples import SampleBank
    from fourier_analyzer.synth.wavetables import TableBank
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    notes = [Note(0, 0.5, 45), Note(0.6, 0.5, 50)]
    y = Renderer(sr=OPT_SR).render(p, notes, 1.2)
    opt = Optimizer(y, OPT_SR, analyze(y, OPT_SR, bpm=120), Options(time_budget=5))
    moves = dict((name, fn) for name, fn, _ in opt.structure_moves(opt.initial_patch()))
    for kind in ("sample", "granular", "spectral"):
        name = f"osc A as {kind} oscillator (a note of the original)"
        q = moves[name](opt.initial_patch().copy())
        assert q.oscs[0].type == kind and q.oscs[0].smp.source.startswith("embedded:")
        _, st = read_preset_bytes(build_preset_bytes(q, opt.bank, samples=opt.samples))
        back = SampleBank()
        r = state_to_patch(st, TableBank(), samples=back, bpm=q.bpm)   # Serum takes the host tempo
        assert r.oscs[0].type == kind
        a = Renderer(opt.bank, OPT_SR, samples=opt.samples).render(q, notes, 1.2)
        b = Renderer(TableBank(), OPT_SR, samples=back).render(r, notes, 1.2)
        assert np.corrcoef(a, b)[0, 1] > 0.98, kind
