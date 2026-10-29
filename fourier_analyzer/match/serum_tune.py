"""Fine-tune a finished patch inside the real Serum 2 plugin.

The main search runs on our own engine, which approximates Serum. This last
stage renders candidate presets in real Serum 2 (one helper process kept
open for the whole stage), scores them against the original with the same
metrics and nudges the continuous parameters with CMA-ES, so the exported
preset is matched to the real plugin rather than to our approximation.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from ..serum.host import SerumSession
from ..serum.mapping import build_preset_bytes
from ..synth.patch import Patch
from . import params as P
from .metrics import Reference

log = logging.getLogger(__name__)


def objective(d: dict) -> float:
    """Waveform similarity, steadied by spectral similarity (phase independent)."""
    return 0.7 * d["waveform_similarity"] + 0.3 * d["spectral_similarity"]


def tune_in_serum(patch: Patch, bank, notes, y_target: np.ndarray, sr: int, budget_s: float,
                  gain: np.ndarray | None = None, stop: Callable[[], bool] | None = None,
                  progress: Callable[[str, float, float, dict], None] | None = None,
                  session_factory=SerumSession, sigma: float = 0.06, samples=None) -> tuple[Patch, dict]:
    """Return (best patch, info). Never worse than the input: the starting
    patch is kept unless a candidate scores higher in real Serum. Raises
    SerumHostError only if Serum cannot render the starting patch."""
    import cma

    from .. import calibration
    corr = calibration.load_corrections()          # learned export corrections
    t0 = time.time()
    ref = Reference(np.asarray(y_target, dtype=np.float64), sr)
    duration = len(y_target) / sr
    stopped = (lambda: False) if stop is None else stop

    def left() -> float:
        return budget_s - (time.time() - t0)

    with session_factory(sr) as sess:
        def score(p: Patch) -> tuple[float, dict]:
            y = sess.render(build_preset_bytes(calibration.apply_corrections(p, corr), bank, name="FA tuning",
                                                samples=samples), notes, duration)
            y = np.asarray(y, dtype=np.float64)[: len(y_target)]
            if len(y) < len(y_target):
                y = np.pad(y, (0, len(y_target) - len(y)))
            if gain is not None:
                y = y * gain[: len(y)]
            if not np.any(np.abs(y) > 1e-6):
                return 0.0, {"waveform_similarity": 0.0, "spectral_similarity": 0.0}
            d = ref.all(y)
            return objective(d), d

        base, d0 = score(patch)
        best_p, best_s, best_d = patch, base, d0
        evals = 1
        info = {"ok": True, "plugin": getattr(sess, "plugin", ""), "before": round(d0["waveform_similarity"], 2)}

        def report(text=""):
            if progress:
                progress(text, best_d["waveform_similarity"], time.time() - t0,
                         {"evals": evals, "budget": budget_s, "elapsed": round(time.time() - t0, 1)})
        report(f"real Serum 2 starting score {d0['waveform_similarity']:.1f}% ({info['plugin']})")

        ps = P.continuous_params(patch)
        if ps and left() > 0 and not stopped():
            if len(ps) == 1:                   # CMA-ES needs >= 2 dimensions
                ps.append(P.Param("envs.0.sustain", 0.0, 1.0))
            popsize = max(6, min(10, 4 + int(3 * np.log(len(ps) + 1))))
            gen, restart, idle = 0, 0, 0
            try:
                # CMA-ES from the best patch so far; restart with another step
                # size whenever it converges, until the time is used up
                while left() > 0 and not stopped() and idle < 4:
                    start_p, start_s = best_p, best_s
                    sig = (sigma, sigma * 2, sigma / 2)[restart % 3]
                    es = cma.CMAEvolutionStrategy(P.vector(start_p, ps), sig, {
                        "bounds": [0.0, 1.0], "popsize": popsize, "verbose": -9,
                        "seed": 4321 + restart, "tolfun": 1e-4})
                    while not es.stop() and left() > 0 and not stopped():
                        xs, fits = es.ask(), []
                        for x in xs:
                            if left() <= 0 or stopped():
                                break
                            cand = P.apply(start_p, ps, x)
                            sc, d = score(cand)
                            evals += 1
                            fits.append(-sc)
                            if sc > best_s:
                                best_p, best_s, best_d = cand, sc, d
                        if len(fits) < len(xs):
                            break
                        es.tell(xs, fits)
                        gen += 1
                        if gen % 5 == 0:
                            report(f"Serum tuning round {gen}: best {best_d['waveform_similarity']:.1f}%")
                    restart += 1
                    idle = idle + 1 if best_s <= start_s + 1e-3 else 0
            except Exception as e:             # keep the best so far if Serum fails mid-way
                log.warning("Serum tuning stopped: %s", e)
                info["error"] = str(e)
    info.update({"after": round(best_d["waveform_similarity"], 2), "evals": evals,
                 "seconds": round(time.time() - t0, 1), "details": best_d,
                 "params": len(ps), "stopped": bool(stopped()),
                 "changes": calibration.tuning_changes(patch, best_p) if best_p is not patch else []})
    return best_p, info
