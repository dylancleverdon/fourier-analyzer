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
