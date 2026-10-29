import numpy as np
import pytest

from fourier_analyzer.analysis.features import estimate_bpm, tempo_candidates

SR = 22050


def _loop(bpm, halftime, dur=6.0):
    rng = np.random.default_rng(0)
    y = np.zeros(int(dur * SR))
    b = 60 / bpm

    def hit(t, kind):
        i = int(t * SR)
        n = min(len(y) - i, int(0.25 * SR))
        if n <= 0:
            return
        tt = np.arange(n) / SR
        if kind == "k":
            s = np.sin(2 * np.pi * (50 * tt + 1.8 * (1 - np.exp(-tt / 0.03)))) * np.exp(-tt / 0.12)
        elif kind == "s":
            s = rng.standard_normal(n) * np.exp(-tt / 0.08) * 0.6
        else:
            s = np.diff(rng.standard_normal(n + 1)) * np.exp(-tt / 0.02) * 0.15
        y[i:i + n] += s

    for k in range(int(dur / b) + 1):
        t = k * b
        if halftime:
            if k % 4 == 0:
                hit(t, "k")
            if k % 4 == 2:
                hit(t, "s")
        else:
            hit(t, "k")
            if k % 2:
                hit(t, "s")
        hit(t, "h")
        hit(t + b / 2, "h")
    return y.astype(np.float32)


@pytest.mark.parametrize("bpm,half", [(140, True), (150, True), (174, False), (128, False), (100, True), (170, True)])
def test_tempo_exact(bpm, half):
    assert abs(estimate_bpm(_loop(bpm, half), SR) - bpm) < 0.5


def test_alternatives_include_half_time():
    c = [x["bpm"] for x in tempo_candidates(_loop(140, True), SR)]
    assert any(abs(x - 70) < 0.5 for x in c)
