import numpy as np
from scipy import signal

from fourier_analyzer import calibration
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import Filter, Note, Osc, Patch, fx

SR = 44100


def _patch():
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, type="lp24", cutoff=3000, reso=0.1)
    p.fx_chain = [fx("chorus")]
    return p


def test_difference_report_names_level_bands_and_timing():
    notes = [Note(0, 0.5, 45), Note(0.6, 0.5, 50)]
    ours = Renderer(sr=SR).render(_patch(), notes, 1.3)
    b, a = signal.butter(2, 1500 / (SR / 2))
    serum = 1.4 * signal.lfilter(b, a, ours)                         # darker and louder
    serum = np.concatenate([np.zeros(int(0.003 * SR)), serum])[: len(ours)]   # 3 ms late
    d = calibration.describe_difference(ours, serum, SR, notes)
    text = " ".join(d["findings"])
    assert d["tilt_db_per_octave"] < -0.75 and "darker" in text
    assert any(r["diff_db"] < -3 and r["lo_hz"] > 1000 for r in d["band_regions"])
    assert abs(d["lag_ms"] - 3.0) < 0.5
    assert d["pitch_cents"] is not None and abs(d["pitch_cents"]) < 5


def test_corrections_are_learned_from_consistent_tuning_moves():
    calibration.reset_corrections()
    p = _patch()
    tuned = p.copy()
    tuned.filter.cutoff = p.filter.cutoff * 1.25
    for _ in range(calibration.MIN_RUNS):
        _, notes = calibration.learn(calibration.tuning_changes(p, tuned))
    corr = calibration.load_corrections()["params"]
    assert abs(corr["filter.lp24.cutoff"]["value"] - 1.25) < 0.01, corr
    assert notes and "filter.lp24.cutoff" in notes[0]
    assert abs(calibration.apply_corrections(p).filter.cutoff - 3750) < 40
    assert calibration.apply_corrections(p).filter.reso == p.filter.reso
    # inconsistent moves teach nothing
    calibration.reset_corrections()
    for k in range(calibration.MIN_RUNS):
        q = p.copy()
        q.filter.reso = p.filter.reso + (0.1 if k % 2 else -0.1)
        calibration.learn(calibration.tuning_changes(p, q))
    assert not calibration.load_corrections()["params"]
    calibration.reset_corrections()


def test_culprit_search_points_at_the_mis_emulated_part():
    """A 'Serum' whose chorus differs from ours: the chorus explains the gap."""
    p = _patch()
    notes = [Note(0, 0.5, 45)]
    r = Renderer(sr=SR)

    def serum_render(q):
        q = q.copy()
        for m in q.fx_chain:
            if m["type"] == "chorus":
                m["mix"] = 1.0
                m["depth"] = 1.0
        return r.render(q, notes, 0.7)

    class Session:
        def render(self, preset, n, dur):
            return serum_render(preset)
    res = calibration.find_culprits(p, None, None, notes, 0.7, SR, Session(),
                                    render_ours=lambda q: r.render(q, notes, 0.7), build=lambda q: q)
    assert res["components"][0]["component"] == "FX chorus", res
    assert res["components"][0]["explains"] > 0


def test_structure_disagreements_are_logged_by_kind_of_part():
    rows = [
        {"kind": "add", "part": "FX reverb", "engine_delta": 2.0, "serum_delta": -1.0, "agreed": False},
        {"kind": "add", "part": "FX reverb", "engine_delta": 1.5, "serum_delta": 0.8, "agreed": True},
        {"kind": "add", "part": "LFO1 (sine, 1/8) -> filter.cutoff", "engine_delta": 1.0, "serum_delta": 1.2,
         "agreed": True},
        {"kind": "remove", "part": "unison x3", "engine_delta": 0.0, "serum_delta": -2.0, "agreed": False},
    ]
    calibration.record_checks(rows, {"project": "p"})
    s = {r["part"]: r for r in calibration.summary()["structure_checks"]}
    assert s["add: FX reverb"]["checks"] >= 2 and s["add: FX reverb"]["disagreed"] >= 1
    assert "add: LFO -> filter.cutoff" in s
    assert s["remove: unison"]["disagreed"] >= 1
