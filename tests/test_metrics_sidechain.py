import numpy as np

from fourier_analyzer.analysis.features import analyze
from fourier_analyzer.analysis.sidechain import estimate, gain_curve
from fourier_analyzer.match.metrics import Reference
from fourier_analyzer.synth.engine import Renderer
from fourier_analyzer.synth.patch import Filter, Note, Osc, Patch

SR = 22050


def test_metrics_identity_shift_and_noise():
    rng = np.random.default_rng(0)
    t = np.arange(SR) / SR
    y = np.sin(2 * np.pi * 110 * t) * np.exp(-t)
    ref = Reference(y, SR)
    m = ref.all(y)
    assert m["waveform_similarity"] > 99.9 and m["spectral_similarity"] > 99.9
    shifted = np.concatenate([np.zeros(50), y[:-50]])
    assert ref.waveform(shifted) > 99            # lag search finds 50 samples
    assert ref.waveform(0.3 * y) > 99.9          # gain invariant
    assert ref.all(rng.standard_normal(SR))["waveform_similarity"] < 5
    # same harmonic magnitudes, scrambled harmonic phases: identical spectrum,
    # different waveform shape (a time shift cannot fix it)
    saw = sum(np.sin(2 * np.pi * 110 * k * t) / k for k in range(1, 30)) * np.exp(-t)
    scr = sum(np.sin(2 * np.pi * 110 * k * t + rng.uniform(0, 2 * np.pi)) / k for k in range(1, 30)) * np.exp(-t)
    ref2 = Reference(saw, SR)
    assert ref2.spectral(scr) > 98               # phase independent
    assert ref2.waveform(scr) < 90               # phase sensitive (by design)


def test_sidechain_detected_and_absent():
    r = Renderer(sr=SR)
    p = Patch()
    p.oscs[0] = Osc(enabled=True, table="basic:saw")
    p.filter = Filter(enabled=True, cutoff=900)
    notes = [Note(0, 1.9, 36), Note(2.0, 1.9, 41)]
    y = r.render(p, notes, 4.0)
    kicks = np.arange(0.2, 4, 60 / 140)
    true = {"detected": True, "depth_db": 8.0, "attack_ms": 3, "hold_ms": 30, "release_ms": 120,
            "kicks": list(kicks)}
    yd = y * gain_curve(true, len(y), SR)
    m = estimate(yd, SR, analyze(yd, SR, bpm=140).notes, kicks)
    assert m["detected"]
    assert 6 < m["depth_db"] < 11
    assert 60 < m["release_ms"] + m["hold_ms"] < 250
    assert not estimate(y, SR, analyze(y, SR, bpm=140).notes, kicks)["detected"]
