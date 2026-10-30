"""Synthesis before samples, human-style moves, checkpoints, hold-out, macros."""
import numpy as np

from fourier_analyzer.analysis.features import analyze
from fourier_analyzer.match.optimizer import OPT_SR, Optimizer, Options
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import LFO, Filter, Mod, Note, Osc, Patch, SampleCfg, fx

NOTES = [Note(0, 0.4, 40), Note(0.5, 0.4, 43), Note(1.0, 0.4, 38), Note(1.5, 0.4, 45), Note(2.1, 0.5, 40)]


def _target(p=None, notes=NOTES, dur=2.8):
    if p is None:
        p = Patch(bpm=128)
        p.oscs[0] = Osc(enabled=True, table="basic:saw")
        p.filter = Filter(enabled=True, type="lp24", cutoff=900, reso=0.2)
    return Renderer(sr=OPT_SR).render(p, notes, dur)


def _opt(budget=30, **kw):
    y = _target()
    return Optimizer(y, OPT_SR, analyze(y, OPT_SR, bpm=128), Options(time_budget=budget, **kw))


def test_only_sample_like_parts_cost_anything():
    opt = _opt(prefer_synthesis=1.0)
    p = opt.initial_patch()
    free = p.copy()
    free.oscs[0].warp, free.oscs[0].warp_amount = "BendPos", 0.3
    free.fx_chain = [fx("distortion"), fx("reverb"), fx("chorus")]
    free.mods.append(Mod("lfo1", "filter.cutoff", 0.3))
    assert opt.cost(p) == 0 and opt.cost(free) == 0             # design parts are free, any number of them
    smp = p.copy()
    sid = opt.samples.add(np.zeros(4410, dtype=np.float32), 44100, root=60)
    smp.oscs[0] = Osc(enabled=True, type="granular", level=0.8)
    smp.oscs[0].smp = SampleCfg(source=sid, root=60)
    multi = p.copy()
    multi.oscs[0].table = opt.bank.add_custom(np.random.default_rng(0).standard_normal((4, 2048)))
    assert opt.cost(smp) > opt.cost(multi) > 0
    assert _opt(prefer_synthesis=0.0).cost(smp) == 0               # "prefer accuracy" end of the slider


def test_samples_wait_until_synthesis_stops_improving():
    opt = _opt()
    p = opt.initial_patch()
    names = [n for n, _, _ in opt.structure_moves(p, samples=False)]
    assert names and not any(opt.SAMPLE_TAG in n for n in names)
    res = opt.run()
    assert not opt.uses_samples(res.patch)                      # a saw needs no sample
    assert res.synth_patch is None                              # synthesis-only is the result itself


def test_compound_and_fx_layout_moves():
    opt = _opt()
    p = opt.initial_patch()
    p.fx_chain = [fx("distortion", mode="Tube"), fx("reverb")]
    names = [n for n, _, _ in opt.structure_moves(p, samples=False)]
    assert any(n.startswith("FX order: reverb before distortion") for n in names)
    assert "FX reverb on parallel bus 1" in names
    assert any(n.startswith("together: ") for n in names)
    assert any("LFO" in n and "warp amount" in n for n in names)
    mv = dict((n, f) for n, f, _ in opt.structure_moves(p, samples=False))
    q = mv["FX reverb on parallel bus 1"](p.copy())
    assert q.fx_buses[0][0]["type"] == "reverb" and [m["type"] for m in q.fx_chain] == ["distortion"]
    assert np.any(opt.render(q))


def test_pairs_are_pruned_when_only_together_they_do_nothing():
    """+12 dB then -12 dB high shelf: removing either alone changes the sound,
    removing both does not, so pruning removes the pair."""
    opt = _opt()
    p = opt.initial_patch()
    p.filter = Filter(enabled=True, type="lp24", cutoff=900, reso=0.2)
    p.fx_chain = [fx("eq", freq2=500.0, gain2=12.0, type2="highshelf"),
                  fx("eq", freq2=500.0, gain2=-12.0, type2="highshelf")]
    q, s = opt.prune(p, opt.evaluate(p))
    assert q.fx_chain == [], [m["type"] for m in q.fx_chain]
    assert any(" + " in st["removed"] for st in opt.stages[-1]["steps"])


def test_kept_sample_is_shortened_then_turned_into_a_wavetable():
    opt = _opt()
    p = opt.initial_patch()
    y44 = Renderer(sr=44100).render(p, [Note(0, 1.0, 40)], 1.0)
    sid = opt.samples.add(y44.astype(np.float32), 44100, root=40)
    p.oscs[0] = Osc(enabled=True, type="sample", level=0.8)
    p.oscs[0].smp = SampleCfg(source=sid, root=40, loop="forward", loop_start=0.3, loop_end=0.95)
    q, s, steps = opt._refine_sample(p, opt.evaluate(p))
    assert steps
    assert q.oscs[0].type == "wavetable" or q.oscs[0].smp.loop_end - q.oscs[0].smp.loop_start < 0.65


def test_both_presets_when_the_best_uses_a_sample():
    opt = _opt()
    synth = opt.initial_patch()
    opt.best_synth = (synth, opt.evaluate(synth))
    p = synth.copy()
    sid = opt.samples.add(np.ones(4410, dtype=np.float32) * 0.1, 44100, root=60)
    p.oscs[1] = Osc(enabled=True, type="sample", level=0.3)
    p.oscs[1].smp = SampleCfg(source=sid, root=60, loop="forward")
    res = opt.finish(p, opt.evaluate(p))
    assert res.synth_patch is not None and not opt.uses_samples(res.synth_patch)
    assert "waveform_similarity" in res.synth_details


def test_holdout_and_grouped_fine_tune():
    opt = _opt(budget=20)
    assert opt.holdout_range and opt.duration < 2.8                # the last note is kept out of the fit
    p = opt.initial_patch()
    p.fx_chain = [fx(k) for k in ("distortion", "chorus", "reverb", "delay", "phaser", "eq", "flanger", "hyperdim", "comp")]
    from fourier_analyzer.match import params as P
    assert len(P.continuous_params(p)) > 30
    q, s = opt.fine_tune(p, opt.evaluate(p), 120)
    assert s >= opt.evaluate(p) - 1e-6
    ho = opt.holdout_score(q)
    assert ho and 0 <= ho["waveform_similarity"] <= 100


class _Sess:
    """Stand-in Serum whose filters are 2x brighter: it lags our engine."""

    def __init__(self, sr):
        self.sr = sr

    def render(self, preset, notes, duration):
        from fourier_analyzer.serum.mapping import state_to_patch
        from fourier_analyzer.serum.preset import read_preset_bytes
        from fourier_analyzer.synth.samples import SampleBank
        from fourier_analyzer.synth.wavetables import TableBank
        _, st = read_preset_bytes(preset)
        b, s = TableBank(), SampleBank()
        q = state_to_patch(st, b, samples=s)
        q.filter.cutoff = min(18000.0, q.filter.cutoff * 2.0)
        return Renderer(b, self.sr, samples=s).render(q, notes, duration)

    def close(self):
        pass


def test_checkpoints_tune_up_in_serum_when_it_lags():
    from fourier_analyzer.match.serum_judge import SerumJudge
    opt = _opt()
    y44 = _target(dur=opt.duration).astype(float)
    y44 = Renderer(sr=44100).render(_patch_target(), [n for n in opt.notes], opt.duration)
    opt.judge = SerumJudge(opt.bank, opt.samples, opt.notes, y44, 44100, budget_s=120, session_factory=_Sess)
    p = opt.initial_patch()
    p.filter = Filter(enabled=True, type="lp24", cutoff=900, reso=0.2)
    q, s = opt.checkpoint(p, opt.evaluate(p), "test")
    row = opt.checkpoints[-1]
    assert row["at"] == "test" and "serum" in row and row["lag"] > opt.CHECKPOINT_LAG
    # our engine learned that this "Serum" is brighter, so it now predicts Serum's score ...
    assert opt.ecorr.v["cutoff"] > 1.3 or opt.ecorr.v["tilt"] > 0.4, opt.ecorr.as_dict()
    assert "engine_corrected" in row and abs(row["engine_corrected"] - row["serum"]) < row["lag"]
    # ... and a tune-up inside Serum runs only if it still lags after that
    if row["engine_corrected"] - row["serum"] > opt.CHECKPOINT_LAG:
        assert "tuned" in row


def test_a_wrong_engine_correction_is_dropped():
    from fourier_analyzer.match.engine_correction import EngineCorrection
    from fourier_analyzer.match.serum_judge import SerumJudge

    class Same(_Sess):                       # a "Serum" identical to our engine
        def render(self, preset, notes, duration):
            from fourier_analyzer.serum.mapping import state_to_patch
            from fourier_analyzer.serum.preset import read_preset_bytes
            from fourier_analyzer.synth.samples import SampleBank
            from fourier_analyzer.synth.wavetables import TableBank
            _, st = read_preset_bytes(preset)
            b, s = TableBank(), SampleBank()
            return Renderer(b, self.sr, samples=s).render(state_to_patch(st, b, samples=s), notes, duration)
    opt = _opt()
    opt.ecorr = EngineCorrection({"cutoff": 0.5, "level_db": 4.0})
    y44 = Renderer(sr=44100).render(_patch_target(), opt.notes, opt.duration)
    opt.judge = SerumJudge(opt.bank, opt.samples, opt.notes, y44, 44100, budget_s=120, session_factory=Same)
    p = opt.initial_patch()
    p.filter = Filter(enabled=True, type="lp24", cutoff=900, reso=0.2)
    opt.checkpoint(p, opt.evaluate(p), "wrong start")
    log = opt.ecorr_log[-1]
    assert "dropped" in log or abs(opt.ecorr.v["cutoff"] - 1.0) < 0.3, log


def _patch_target():
    p = Patch(bpm=128)
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, type="lp24", cutoff=900, reso=0.2)
    return p


def test_macros_name_the_key_knobs_without_changing_the_sound():
    from fourier_analyzer.match.macros import assign_macros
    p = _patch_target()
    p.oscs[0].warp, p.oscs[0].warp_amount = "BendPos", 0.3
    p.lfos[0] = LFO(shape="sine", sync="1/8", mode="trig")
    p.mods = [Mod("lfo1", "filter.cutoff", 0.4)]
    p.fx_chain = [fx("distortion", mode="Tube", drive=0.4)]
    a = Renderer(sr=OPT_SR).render(p, NOTES, 2.8)
    labels = assign_macros(p)
    assert labels == ["CUTOFF", "WARP", "LFO RATE", "LFO DEPTH", "DRIVE"]
    assert np.max(np.abs(Renderer(sr=OPT_SR).render(p, NOTES, 2.8) - a)) < 1e-6
    assert p.macro_names[:5] == labels


def test_fix_first_ranks_emulations_by_disagreement():
    from fourier_analyzer import calibration
    calibration.record_checks([
        {"kind": "add", "part": "FX hyperdim", "engine_delta": 3.0, "serum_delta": -2.0, "agreed": False},
        {"kind": "add", "part": "warp BendPos on osc A", "engine_delta": 1.0, "serum_delta": 0.5, "agreed": False},
        {"kind": "add", "part": "FX reverb Hall", "engine_delta": 1.0, "serum_delta": 1.1, "agreed": True},
    ])
    ff = calibration.summary()["fix_first"]
    assert ff[0]["emulation"] == "FX hyperdim"
    assert {"warp BendPos"} <= {x["emulation"] for x in ff}


def test_engine_corrections_that_held_up_start_the_next_run():
    from fourier_analyzer.match import engine_correction as EC
    EC.reset_saved()
    EC.save_run(EC.EngineCorrection({"cutoff": 1.4}), helped=False)
    assert EC.load_saved().is_identity()                      # not validated: not saved
    EC.save_run(EC.EngineCorrection({"cutoff": 1.4, "tilt": 0.6}), helped=True)
    EC.save_run(EC.EngineCorrection({"cutoff": 1.2, "tilt": 0.2}), helped=True)
    v = EC.load_saved().v
    assert 1.2 < v["cutoff"] < 1.4 and abs(v["tilt"] - 0.4) < 1e-6
    assert _opt().ecorr.v["cutoff"] == v["cutoff"]           # a new run starts from it
    assert _opt(engine_learning=False).ecorr.is_identity()
    EC.reset_saved()
