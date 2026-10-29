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
