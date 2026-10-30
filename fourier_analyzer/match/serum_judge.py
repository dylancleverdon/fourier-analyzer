"""Real Serum 2 as the judge of structure decisions.

The structure search screens every possible addition in our engine (fast);
the few finalists are then rendered in the real plugin through one
persistent helper process, and a part is only added (or removed) when real
Serum agrees. The judge has its own time budget; once it is used up, or if
Serum is not available, the search carries on with our engine alone.
"""
from __future__ import annotations

import logging
import time
from typing import Callable

import numpy as np

from ..serum.host import SerumHostError, SerumSession
from ..serum.mapping import build_preset_bytes
from ..synth.patch import Patch
from . import params as P
from .metrics import Reference
from .serum_tune import objective

log = logging.getLogger(__name__)

AGREE_MARGIN = 0.25          # Serum objective points a change must gain (or may lose, pruning)


class SerumJudge:
    """Scores patches in real Serum 2 within its own time budget."""

    def __init__(self, bank, samples, notes, y_target: np.ndarray, sr: int, budget_s: float,
                 gain: np.ndarray | None = None, stop: Callable[[], bool] | None = None,
                 finalists: int = 3, tune_renders: int = 8, use_corrections: bool = True,
                 session_factory=SerumSession):
        from .. import calibration
        self.bank, self.samples, self.sr = bank, samples, sr
        self.notes = list(notes)
        self.y = np.asarray(y_target, dtype=np.float64)
        self.duration = len(self.y) / sr
        self.gain = gain
        self.ref = Reference(self.y, sr)
        self.budget_s = float(budget_s)
        self.stop = stop or (lambda: False)
        self.finalists = max(1, int(finalists))
        self.tune_renders = max(0, int(tune_renders))
        self.corr = calibration.load_corrections() if use_corrections else {"params": {}}
        self._apply = calibration.apply_corrections
        self.session_factory = session_factory
        self.session = None
        self.error: str | None = None
        self.used_s = 0.0
        self.renders = 0
        self.checks: list[dict] = []           # every verdict, for the log and the Engine accuracy page
        self._cache: dict[bytes, tuple[float, dict]] = {}
        # how a render is scored: set by the optimizer to its own search
        # measure, so Serum and our engine are judged by the same yardstick
        self.scorer: Callable[[np.ndarray], float] | None = None
        self.last_audio: np.ndarray | None = None      # Serum's most recent render (sidechain applied)
        self._audio_key: bytes | None = None

    # ------------------------------------------------------------------
    def available(self) -> bool:
        return self.error is None and self.used_s < self.budget_s and not self.stop()

    def time_left(self) -> float:
        return max(0.0, self.budget_s - self.used_s)

    def _open(self) -> bool:
        if self.session is not None:
            return True
        if self.error is not None:
            return False
        t0 = time.time()
        try:
            self.session = self.session_factory(self.sr)
        except SerumHostError as e:
            self.error = str(e)
            log.info("Serum judge unavailable: %s", e)
            return False
        finally:
            self.used_s += time.time() - t0
        return True

    def close(self) -> None:
        if self.session is not None:
            try:
                self.session.close()
            except Exception:
                pass
            self.session = None

    def score(self, patch: Patch, want_audio: bool = False) -> tuple[float, dict] | None:
        """(objective, metrics) of the patch rendered in real Serum, or None
        when Serum is not available or its time is used up. ``want_audio``:
        also keep Serum's render in ``last_audio`` (re-rendered if needed)."""
        import hashlib
        preset = build_preset_bytes(self._apply(patch, self.corr), self.bank, name="FA structure",
                                    samples=self.samples)
        key = hashlib.sha1(preset).digest()
        if key in self._cache and (not want_audio or key == self._audio_key):
            return self._cache[key]
        if not self.available() or not self._open():
            return None
        t0 = time.time()
        try:
            y = np.asarray(self.session.render(preset, self.notes, self.duration), dtype=np.float64)
        except SerumHostError as e:
            self.error = str(e)
            log.warning("Serum judge stopped: %s", e)
            self.close()
            return None
        finally:
            self.used_s += time.time() - t0
        self.renders += 1
        y = y[: len(self.y)]
        if len(y) < len(self.y):
            y = np.pad(y, (0, len(self.y) - len(y)))
        if self.gain is not None:
            y = y * self.gain[: len(y)]
        self.last_audio, self._audio_key = y, key
        if not np.any(np.abs(y) > 1e-6):
            out = (0.0, {"waveform_similarity": 0.0, "spectral_similarity": 0.0})
        else:
            d = self.ref.all(y)
            out = (float(self.scorer(y)) if self.scorer is not None else objective(d), d)
        self._cache[key] = out
        return out

    def tune_up(self, patch: Patch, focus: set | None, base: float, renders: int | None = None,
                all_params: bool = False) -> tuple[Patch, float]:
        """A short tune-up inside Serum (the move's own parameters, or all of
        them): a few random steps around the patch, keeping the best."""
        ps = P.continuous_params(patch, None if all_params else focus) if (focus or all_params) else []
        best_p, best_s = patch, base
        n_renders = self.tune_renders if renders is None else renders
        if not ps or n_renders <= 0:
            return best_p, best_s
        rng = np.random.default_rng(len(self.checks) + 11)
        x0 = np.array(P.vector(patch, ps))
        sigma = 0.12 if not all_params else 0.05
        for _ in range(n_renders):
            if not self.available():
                break
            x = np.clip(np.array(P.vector(best_p, ps)) + rng.normal(0, sigma, len(x0)), 0, 1)
            q = P.apply(best_p, ps, x)
            r = self.score(q)
            if r is None:
                break
            if r[0] > best_s:
                best_p, best_s = q, r[0]
            else:
                sigma *= 0.85
        return best_p, best_s

    def note(self, kind: str, part: str, engine_before: float, engine_after: float,
             serum_before: float, serum_after: float, accepted: bool) -> dict:
        """Record one verdict. Serum agrees when it moves the same way as our
        engine (both gain for an addition; both hold for a removal)."""
        e_d, s_d = engine_after - engine_before, serum_after - serum_before
        if kind == "add":
            agreed = (e_d > AGREE_MARGIN) == (s_d > AGREE_MARGIN)
        else:
            agreed = (e_d >= -0.3) == (s_d >= -AGREE_MARGIN)
        row = {"kind": kind, "part": part, "engine_before": round(engine_before, 2),
               "engine_after": round(engine_after, 2), "serum_before": round(serum_before, 2),
               "serum_after": round(serum_after, 2), "engine_delta": round(e_d, 2),
               "serum_delta": round(s_d, 2), "agreed": bool(agreed), "accepted": bool(accepted)}
        self.checks.append(row)
        return row

    def summary(self) -> dict:
        return {"renders": self.renders, "seconds": round(self.used_s, 1), "budget_s": self.budget_s,
                "error": self.error, "finalists": self.finalists,
                "agreed": sum(c["agreed"] for c in self.checks), "checks": len(self.checks)}
