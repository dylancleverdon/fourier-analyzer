import numpy as np

from fourier_analyzer.match.score import Target
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import LFO, Env, Filter, Mod, Note, Osc, Patch, sync_to_hz

NOTES = [Note(0, 0.5, 36), Note(0.5, 0.5, 39), Note(1.0, 1.0, 34)]


def mk(cut=800.0, table="basic:saw"):
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table=table)
    p.filter = Filter(enabled=True, type="lp24", cutoff=cut, reso=0.2)
    p.envs[0] = Env(0.005, 0, 0.3, 0.8, 0.1)
    return p


def test_render_basic():
    y = Renderer().render(mk(), NOTES, 2.5)
    assert y.shape == (int(2.5 * 22050),)
    assert np.isfinite(y).all() and np.abs(y).max() > 0.01


def test_lfo_and_mods_render():
    p = mk()
    p.lfos[0] = LFO(shape="custom", sync="1/8", points=[[0, 0], [0.5, 1], [1, 0]])
    p.mods = [Mod("lfo1", "filter.cutoff", 0.3), Mod("env2", "pitch", 0.1)]
    y = Renderer().render(p, NOTES, 2.5)
    assert np.isfinite(y).all()


def test_score_identity_and_ordering():
    r = Renderer()
    t = Target(r.render(mk(), NOTES, 2.5), r.sr)
    assert t.score(r.render(mk(), NOTES, 2.5)) > 99.9
    near = t.score(r.render(mk(cut=900), NOTES, 2.5))
    far = t.score(r.render(mk(cut=3000), NOTES, 2.5))
    sine = t.score(r.render(mk(table="basic:sine"), NOTES, 2.5))
    assert near > far > 0
    assert near > sine


def test_sync_to_hz():
    assert abs(sync_to_hz("1/4", 120) - 2.0) < 1e-9
    assert abs(sync_to_hz("1/8T", 120) - 6.0) < 1e-9
