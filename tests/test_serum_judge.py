"""Structure decisions confirmed in (a stand-in for) real Serum 2."""
import numpy as np

from fourier_analyzer.analysis.features import analyze
from fourier_analyzer.match.optimizer import OPT_SR, Optimizer, Options
from fourier_analyzer.match.serum_judge import SerumJudge
from fourier_analyzer.serum.mapping import state_to_patch
from fourier_analyzer.serum.preset import read_preset_bytes
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import Filter, Note, Osc, Patch
from fourier_analyzer.synth.samples import SampleBank
from fourier_analyzer.synth.wavetables import TableBank


class FakeSession:
    """Renders presets with our engine; ``mute_samples`` mimics a plugin that
    plays sample / granular / spectral oscillators silently, so it disagrees
    with our engine about every one of them."""

    def __init__(self, sr, mute_samples=False):
        self.sr, self.mute_samples, self.renders = sr, mute_samples, 0

    def render(self, preset, notes, duration):
        self.renders += 1
        _, st = read_preset_bytes(preset)
        bank, smp = TableBank(), SampleBank()
        p = state_to_patch(st, bank, samples=smp)
        if self.mute_samples:
            for o in p.oscs:
                if o.type != "wavetable":
                    o.level = 0.0
        return Renderer(bank, self.sr, samples=smp).render(p, notes, duration)

    def close(self):
        pass


def _setup(budget=40):
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, type="lp24", cutoff=700, reso=0.3)
    notes = [Note(0, 0.45, 40), Note(0.5, 0.45, 43), Note(1.0, 0.5, 38)]
    y = Renderer(sr=OPT_SR).render(p, notes, 1.6)
    opt = Optimizer(y, OPT_SR, analyze(y, OPT_SR, bpm=120), Options(time_budget=budget))
    y44 = Renderer(sr=44100).render(p, notes, 1.6)
    return opt, y44


def _judge(opt, y44, **kw):
    return SerumJudge(opt.bank, opt.samples, [n for n in opt.notes if n.start < opt.duration],
                      y44[: int(opt.duration * 44100)], 44100, budget_s=120, tune_renders=0, **kw)


def test_serum_rejects_what_it_does_not_hear():
    opt, y44 = _setup()
    judge = _judge(opt, y44, session_factory=lambda sr: FakeSession(sr, mute_samples=True))
    opt.judge = judge
    opt.samples_unlocked = True                     # samples in play from the start ...
    opt.opts.prefer_synthesis = 0.0                 # ... at no cost, so they make the finalists
    p0 = opt.initial_patch()
    p0.filter.enabled = False
    p, _ = opt.structure_search(p0, opt.evaluate(p0))
    smp = [c for c in judge.checks if c["kind"] == "add" and "a note of the original" in c["part"]]
    assert smp, judge.checks                        # our engine loves them (top finalists) ...
    assert all(not c["accepted"] and not c["agreed"] for c in smp)   # ... this "Serum" does not
    assert all(o.type == "wavetable" for o in p.oscs if o.enabled)
    st = opt.stages[-1]
    assert st["stage"] == "structure search" and st["serum_checks"]


def test_serum_agrees_with_an_identical_engine():
    opt, y44 = _setup()
    sess = FakeSession(44100)
    opt.judge = _judge(opt, y44, session_factory=lambda sr: sess)
    p0 = opt.initial_patch()
    p0.filter.enabled = False
    p, _ = opt.structure_search(p0, opt.evaluate(p0))
    kept = [c for c in opt.judge.checks if c["accepted"]]
    assert kept and all(c["agreed"] for c in kept)
    assert any(s.get("agreed") for s in opt.stages[-1]["steps"])
    assert sess.renders >= 2
    # pruning: removals are confirmed too
    p2, _ = opt.prune(p, opt.evaluate(p))
    assert opt.stages[-1]["stage"] == "pruning"


def test_without_serum_the_engine_decides_alone():
    from fourier_analyzer.serum.host import SerumHostError

    def unavailable(sr):
        raise SerumHostError("Serum 2 plugin not found")
    opt, y44 = _setup()
    opt.judge = _judge(opt, y44, session_factory=unavailable)
    p0 = opt.initial_patch()
    p0.filter.enabled = False
    p, s = opt.structure_search(p0, opt.evaluate(p0))
    assert opt.judge.error and not opt.judge.checks
    assert opt.stages[-1]["steps"], "engine-only search still adds parts"
    assert s > opt.evaluate(p0)


def test_serum_only_mode_never_adds_without_serum():
    opt, y44 = _setup(budget=60)
    sess = FakeSession(44100)
    opt.judge = _judge(opt, y44, session_factory=lambda sr: sess)
    res = opt.run_serum_only()
    names = [s["stage"] for s in res.stages]
    assert "fine-tune" not in names and "structure search" in names
    for st in res.stages:
        for step in st.get("steps", []):
            assert step.get("agreed"), step


def test_judge_uses_the_persistent_serum_session(monkeypatch):
    """Through the real helper process (FA_FAKE_SERUM stands in for the plugin):
    one session serves every render, and repeated patches are not re-rendered."""
    monkeypatch.setenv("FA_FAKE_SERUM", "1")
    opt, y44 = _setup()
    judge = _judge(opt, y44)
    opt.judge = judge
    p = opt.initial_patch()
    try:
        a = judge.score(p)
        b = judge.score(p)
        q = p.copy()
        q.oscs[0].level *= 0.5
        c = judge.score(q)
    finally:
        judge.close()
    assert a is not None and a == b and c is not None
    assert judge.renders == 2 and judge.error is None
